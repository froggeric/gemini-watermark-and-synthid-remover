#include "gui/jobs.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <fmt/format.h>
#include <opencv2/imgcodecs.hpp>

#include "core/still_remove.hpp"
#include "core/watermark_engine.hpp"
#include "gui/run_dir.hpp"
#include "gui/security.hpp"

namespace wmr::gui {

namespace fs = std::filesystem;

namespace {

// Spec: at most 8 pending jobs (pending = outstanding = queued OR running; the
// worker occupies one slot while it runs, so the bound caps total backlogged
// work, not just the queue depth).
constexpr int kMaxPendingJobs = 8;
// Spec: total bytes stored under the gui root across all live runs stay
// <= 4 GB; the bound gates new uploads only (finished jobs are not evicted).
constexpr std::uintmax_t kAggregateCapBytes = std::uintmax_t(4) * 1024 * 1024 * 1024;

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    if (in.bad()) throw std::runtime_error("read error: " + p.string());
    return ss.str();
}

bool write_file(const fs::path& p, const std::string& bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    out.flush();   // surface a flush/close-time failure (e.g. disk full) now
    return out.good();
}

}  // namespace

// The internal per-job record. Guarded by the manager mutex except where
// commented: only the worker mutates a RUNNING job's files, and it copies
// records out from under the mutex before handing them to the processor.
struct JobManager::Job {
    std::string id;
    JobStatus status = JobStatus::Queued;
    std::time_t created = 0;
    std::uint64_t seq = 0;   // creation sequence: list() ordering
    JobOptions opts;
    std::vector<GuiFile> files;
    bool cancel_requested = false;
    int current_file_index = -1;
};

JobManager::JobManager(fs::path run_dir) : run_dir_(std::move(run_dir)) {
    // Seed the aggregate-cap counter with one walk (covers this root's
    // existing bytes, incl. sibling runs alive at startup).
    stored_bytes_ = stored_bytes_under(run_dir_.parent_path());
    // The default processor is this member lambda over the reused engine (it
    // captures `this`, which owns engine_); tests replace it wholesale before
    // their first create_job.
    processor_ = [this](GuiFile& f, const JobOptions& o, const fs::path& job_dir) {
        return process_file(f, o, job_dir);
    };
    worker_ = std::thread(&JobManager::worker_loop, this);
}

JobManager::~JobManager() {
    cancel_all_and_join(std::chrono::seconds(5));
}

// The default processor body (the heart of the worker): read the persisted
// orig bytes back from disk, decode, bounds-check the rect, run the shared
// detect->remove policy, write the artifact. Never throws for engine reasons
// (remove_still maps exceptions to Failed); a missing/unreadable orig throws
// and the worker's per-file catch turns that into a failed row.
GuiFile JobManager::process_file(GuiFile& f, const JobOptions& o, const fs::path& job_dir) {
    // f.orig_ext was decided by the upload sniff; orig bytes were persisted at
    // upload by create_job - read them back (never held resident before this).
    const std::string bytes = read_file(job_dir / "orig" / fmt::format("{}.{}", f.index, f.orig_ext));
    cv::Mat img = cv::imdecode(cv::Mat(1, (int)bytes.size(), CV_8UC1, (void*)bytes.data()),
                               cv::IMREAD_COLOR);
    if (img.empty()) { f.outcome = FileOutcome::Failed; f.error = "decode failed"; return f; }
    // 64-bit arithmetic: the validator caps each rect field at INT32_MAX, so
    // int x+w can overflow (wrapping negative) and slip past this guard.
    if (o.rect && (o.rect->x < 0 || o.rect->y < 0 ||
                   (long long)o.rect->x + o.rect->width > img.cols ||
                   (long long)o.rect->y + o.rect->height > img.rows)) {
        f.outcome = FileOutcome::Failed;
        f.error = fmt::format("rect {}x{} at ({},{}) exceeds image {}x{}",
                              o.rect->width, o.rect->height, o.rect->x, o.rect->y,
                              img.cols, img.rows);
        return f;
    }
    StillRemoveOptions sro;                    // CLI-verbatim defaults
    sro.force = o.force;
    sro.force_variant = o.legacy ? std::optional(WatermarkVariant::V1) : std::nullopt;
    sro.try_v1_fallback = !o.legacy;
    sro.geometry.rect = o.rect;
    sro.geometry.preset = o.geo_preset;
    sro.denoise_method = o.denoise;
    StillRemoveOutcome r = remove_still(*engine_, img, sro);
    if (r.outcome == StillOutcome::Failed) { f.outcome = FileOutcome::Failed; f.error = r.error; return f; }
    if (r.outcome == StillOutcome::NoWatermark) { f.outcome = FileOutcome::NoWatermark; return f; }
    fs::path out = job_dir / "out" / fmt::format("{}_clean.{}", f.index, f.orig_ext);
    if (!write_still_output(out, img, o.keep_provenance)) {
        f.outcome = FileOutcome::Failed; f.error = "write failed"; return f;
    }
    std::error_code sec;
    const auto out_bytes = fs::file_size(out, sec);
    if (!sec) stored_bytes_ += out_bytes;        // aggregate-cap accounting
    f.outcome = FileOutcome::Removed;
    f.bbox = r.bbox; f.score = r.score;
    // NB: geometry_source reflects the V2 geometry search even when the V1 fallback performed the removal.
    f.geometry_source = r.geometry_source; f.variant = r.variant; f.forced = r.forced;
    // Forced rows carry no bbox (no search ran), so record the mark size the
    // engine's own size rule erased; the UI reports it (and positions its
    // preview overlay) without hot-loading the original.
    if (r.forced) {
        const bool large = img.cols > 1024 && img.rows > 1024;
        f.mark_size = (r.variant == "V1") ? (large ? 96 : 48) : (large ? 96 : 36);
    }
    f.has_clean = true;
    return f;
}

std::optional<std::string> JobManager::create_job(
    const std::vector<PendingUpload>& files,
    const std::vector<std::pair<std::string, std::string>>& failed,
    const JobOptions& opts, int max_files, long long max_bytes,
    std::string& job_id_out)
{
    std::lock_guard<std::mutex> lk(mtx_);

    int outstanding = 0;
    for (const auto& j : jobs_)
        if (j->status == JobStatus::Queued || j->status == JobStatus::Running) ++outstanding;
    if (outstanding >= kMaxPendingJobs) return "too_many_pending";

    long long incoming = 0;
    for (const auto& u : files) incoming += static_cast<long long>(u.bytes.size());
    if (static_cast<int>(files.size()) > max_files || incoming > max_bytes)
        return "payload_too_large";

    // Aggregate gate: bytes stored under the gui root plus this upload must
    // stay under 4 GiB. Tracked as a monotonic counter seeded by ONE walk at
    // startup (the per-POST recursive walk stalled every poll behind the
    // mutex as the tree grew). The counter covers this instance exactly;
    // a sibling instance started later is covered by its own seed walk.
    if (stored_bytes_.load() + static_cast<std::uintmax_t>(incoming) > kAggregateCapBytes)
        return "payload_too_large";

    auto job = std::make_shared<Job>();
    job->id = new_job_id_locked();
    job->created = std::time(nullptr);
    job->seq = next_seq_++;
    job->opts = opts;

    const fs::path orig_dir = run_dir_ / job->id / "orig";
    std::error_code ec;
    fs::create_directories(orig_dir, ec);
    if (ec) return "storage_error";
    for (std::size_t i = 0; i < files.size(); ++i) {
        if (!write_file(orig_dir / fmt::format("{}.{}", i, files[i].ext), files[i].bytes)) {
            std::error_code rm;
            fs::remove_all(run_dir_ / job->id, rm);
            return "storage_error";
        }
        GuiFile f;
        f.index = static_cast<int>(i);
        f.name = files[i].name;
        f.orig_ext = files[i].ext;
        job->files.push_back(std::move(f));
    }
    // Sniff-rejected uploads become terminal failed rows (no orig on disk).
    for (const auto& [name, error] : failed) {
        GuiFile f;
        f.index = static_cast<int>(job->files.size());
        f.name = name;
        f.outcome = FileOutcome::Failed;
        f.error = error;
        f.has_orig = false;
        job->files.push_back(std::move(f));
    }

    jobs_.push_back(job);
    queue_.push_back(job);
    stored_bytes_ += static_cast<std::uintmax_t>(incoming);
    wake_cv_.notify_all();
    job_id_out = job->id;
    return std::nullopt;
}

std::optional<fs::path> JobManager::file_artifact(std::string_view id, int index,
                                                  bool cleaned) const {
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto& j : jobs_) {
        if (j->id != id) continue;
        for (const auto& f : j->files) {
            if (f.index != index) continue;
            if (cleaned) {
                if (!f.has_clean) return std::nullopt;
                return run_dir_ / j->id / "out" / fmt::format("{}_clean.{}", f.index, f.orig_ext);
            }
            if (!f.has_orig) return std::nullopt;
            return run_dir_ / j->id / "orig" / fmt::format("{}.{}", f.index, f.orig_ext);
        }
        return std::nullopt;
    }
    return std::nullopt;
}

std::optional<JobSnapshot> JobManager::get(std::string_view id) const {
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto& j : jobs_) {
        if (j->id != id) continue;
        int queue_position = 0;
        if (j->status == JobStatus::Queued) {
            int pos = 1;
            for (const auto& q : queue_) {
                if (q == j) { queue_position = pos; break; }
                ++pos;
            }
        }
        return JobSnapshot{j->id, j->status, j->created, queue_position,
                           j->current_file_index, j->files};
    }
    return std::nullopt;
}

std::vector<JobSummary> JobManager::list() const {
    std::lock_guard<std::mutex> lk(mtx_);
    std::vector<JobSummary> out;
    out.reserve(jobs_.size());
    for (auto it = jobs_.rbegin(); it != jobs_.rend(); ++it)
        out.push_back(JobSummary{(*it)->id, (*it)->status, (*it)->created,
                                 static_cast<int>((*it)->files.size())});
    return out;
}

bool JobManager::cancel(std::string_view id, std::optional<JobStatus>& current,
                        std::string& error_code) {
    std::lock_guard<std::mutex> lk(mtx_);
    for (const auto& j : jobs_) {
        if (j->id != id) continue;
        current = j->status;
        if (j->status == JobStatus::Done || j->status == JobStatus::Failed ||
            j->status == JobStatus::Cancelled) {
            error_code = "job_already_finished";
            return false;
        }
        j->cancel_requested = true;
        if (j->status == JobStatus::Queued) {
            // Leaves the queue (the worker skips non-Queued pops); no files
            // run. Originals were persisted at upload and stay downloadable.
            finalize_locked(*j, JobStatus::Cancelled);
        }
        // Running: flag only - the worker checks between files and finalizes.
        return true;
    }
    current = std::nullopt;
    error_code = "unknown_job";
    return false;
}

void JobManager::cancel_all_and_join(std::chrono::seconds bound) {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        request_stop_locked();
    }
    wake_cv_.notify_all();
    std::unique_lock<std::mutex> lk(mtx_);
    if (!done_cv_.wait_for(lk, bound, [this] { return worker_done_; })) {
        // Bound expired: skip static teardown entirely. The leftover run dir
        // dies at the next start's dead-run cleanup; only the clean-join path
        // lets the caller remove the run dir and exit gracefully.
        std::_Exit(0);
    }
    lk.unlock();
    if (worker_.joinable())
        worker_.join();
}

void JobManager::worker_loop() {
    // One engine for the whole process, built here on the worker thread and
    // REUSED across files and jobs (per-instance alpha decode is cheap but
    // pointless to repeat). The ctor can throw (cv::imdecode of the embedded
    // alpha PNGs, e.g. on OOM) and sits OUTSIDE the per-file/job catch pairs
    // below - an escape here would std::terminate the process through the
    // thread, so fail loudly instead.
    try {
        engine_ = std::make_unique<WatermarkEngine>();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "wmr gui: watermark engine failed to initialize: %s\n", e.what());
        std::_Exit(1);
    }
    std::unique_lock<std::mutex> lk(mtx_);
    for (;;) {
        wake_cv_.wait(lk, [this] { return shutting_down_ || !queue_.empty(); });
        if (shutting_down_) break;
        std::shared_ptr<Job> job = queue_.front();
        queue_.pop_front();
        if (job->status != JobStatus::Queued) continue;   // cancelled while queued
        job->status = JobStatus::Running;
        lk.unlock();
        run_job(*job);
        lk.lock();
        if (shutting_down_) break;
    }
    worker_done_ = true;
    done_cv_.notify_all();
}

// Runs WITHOUT the manager mutex held (the processor can take seconds); the
// file record is copied out under the mutex, processed, and committed back
// under the mutex, so snapshot readers never observe a half-written row.
void JobManager::run_job(Job& job) {
    const fs::path dir = run_dir_ / job.id;
    try {
        const int n = static_cast<int>(job.files.size());   // fixed after create_job
        for (int i = 0; i < n; ++i) {
            GuiFile work;
            {
                std::lock_guard<std::mutex> lk(mtx_);
                if (shutting_down_ || job.cancel_requested) {
                    job.current_file_index = -1;
                    finalize_locked(job, JobStatus::Cancelled);  // completed files keep outcomes
                    return;
                }
                job.current_file_index = job.files[static_cast<std::size_t>(i)].index;
                work = job.files[static_cast<std::size_t>(i)];
                if (work.outcome != FileOutcome::Pending)
                    continue;   // terminal at creation (sniff-rejected): never runs
            }
            GuiFile result = work;
            try {
                result = processor_(work, job.opts, dir);
            } catch (const std::exception& e) {
                result = work;
                result.outcome = FileOutcome::Failed;
                result.error = e.what();
            } catch (...) {
                result = work;
                result.outcome = FileOutcome::Failed;
                result.error = "unknown error";
            }
            {
                std::lock_guard<std::mutex> lk(mtx_);
                job.files[static_cast<std::size_t>(i)] = result;
            }
        }
        std::lock_guard<std::mutex> lk(mtx_);
        job.current_file_index = -1;
        finalize_locked(job, JobStatus::Done);   // even when every file failed
    } catch (...) {
        // An exception escaping the loop itself (spec: worker isolation): the
        // JOB fails, the worker survives and runs the next job.
        std::lock_guard<std::mutex> lk(mtx_);
        job.current_file_index = -1;
        finalize_locked(job, JobStatus::Failed);
    }
}

// Terminal writes (by the worker or by cancel) are mutually exclusive under
// the mutex: the first terminal state wins, so a cancel racing the last file
// may legitimately lose to `done`.
void JobManager::finalize_locked(Job& job, JobStatus status) {
    if (job.status == JobStatus::Done || job.status == JobStatus::Failed ||
        job.status == JobStatus::Cancelled)
        return;
    job.status = status;
    if (status == JobStatus::Cancelled)
        for (auto& f : job.files)
            if (f.outcome == FileOutcome::Pending) f.outcome = FileOutcome::NotRun;
}

void JobManager::request_stop_locked() {
    shutting_down_ = true;
    for (auto& j : jobs_) {
        j->cancel_requested = true;
        if (j->status == JobStatus::Queued)
            finalize_locked(*j, JobStatus::Cancelled);
    }
    queue_.clear();
}

// 12 chars [a-z0-9] from the CSPRNG, never derived from user input. The first
// 12 chars of generate_token()'s 32 hex chars carry 48 CSPRNG bits; the
// (loop) uniqueness check below covers the theoretical collision.
std::string JobManager::new_job_id_locked() const {
    for (;;) {
        const std::string id = generate_token().substr(0, 12);
        bool dup = false;
        for (const auto& j : jobs_)
            if (j->id == id) { dup = true; break; }
        if (!dup) return id;
    }
}

}  // namespace wmr::gui
