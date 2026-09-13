#pragma once
// The GUI job model (Task 4): one serialized worker thread runs each uploaded
// file through the shared still remove policy (core/still_remove); HTTP
// threads only read snapshots and flip cancel flags, all under one mutex.
// Spec: docs/superpowers/specs/2026-09-12-gui-webui-design.md ("Job model").
//
// Engine-level TU (the still_remove precedent): no httplib, no CLI11. Task 5's
// REST layer consumes exactly these signatures; the signatures are the contract.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace wmr {
class WatermarkEngine;
}

namespace wmr::gui {

enum class JobStatus { Queued, Running, Done, Failed, Cancelled };
enum class FileOutcome { Pending, Removed, NoWatermark, Failed, NotRun };

struct JobOptions {
    std::string denoise = "off";
    bool legacy = false;
    std::optional<std::string> geo_preset;
    std::optional<cv::Rect> rect;
    bool keep_provenance = false;
    bool force = false;
};

struct GuiFile {
    int index = 0;
    std::string name;                 // sanitized display filename
    FileOutcome outcome = FileOutcome::Pending;
    std::optional<cv::Rect> bbox;
    float score = 0.0f;
    std::string geometry_source, variant, error;
    std::string orig_ext;             // from the sniff ("png"|"jpg"|"webp")
    bool has_clean = false;           // out/<i>_clean.<ext> exists
    bool forced = false;              // removal ran because force was set
    bool has_orig = true;             // false for sniff-rejected entries (no orig/<i>.<ext>)
    int mark_size = 0;                // forced rows only: the px size the size rule erased (0 = n/a)
};

struct PendingUpload { std::string name; std::string ext; std::string bytes; };

// Snapshot copies for the API layer (built under the manager mutex).
struct JobSnapshot {
    std::string id; JobStatus status; std::time_t created = 0;
    int queue_position = 0, current_file_index = -1;   // queue_position: 1-based, 0 = not queued
    std::vector<GuiFile> files;
};
struct JobSummary { std::string id; JobStatus status; std::time_t created; int file_count; };

class JobManager {
public:
    using Processor = std::function<GuiFile(GuiFile&, const JobOptions&, const std::filesystem::path& job_dir)>;
    explicit JobManager(std::filesystem::path run_dir);
    ~JobManager();   // bounded stop (5 s); never an unbounded join
    // Create the job, persist each PendingUpload to <run_dir>/<job_id>/orig/<i>.<ext>
    // (after the caps checks, under the mutex - the POST handler only sniffs and
    // collects), append the sniff-rejected `failed` entries as terminal rows,
    // enqueue, and fill job_id_out on success. The optional return is the API
    // error CODE on rejection: "too_many_pending" (more than 8 outstanding =
    // queued+running jobs), "payload_too_large" (the per-job max_files/max_bytes
    // caps AND the 4 GiB aggregate gate via stored_bytes_under), "storage_error"
    // (orig persistence failed; the partial job dir is removed).
    std::optional<std::string> create_job(const std::vector<PendingUpload>& files,
                                          const std::vector<std::pair<std::string, std::string>>& failed,
                                          const JobOptions& opts, int max_files,
                                          long long max_bytes, std::string& job_id_out);
    // Absolute artifact path for the files endpoint (kind=cleaned requires
    // has_clean; kind=original requires has_orig). Nullopt when unknown
    // job/index or no artifact. Resolves ONLY through this in-memory map; no
    // request string ever builds a filesystem path.
    std::optional<std::filesystem::path> file_artifact(std::string_view id, int index,
                                                       bool cleaned) const;
    std::optional<JobSnapshot> get(std::string_view id) const;
    std::vector<JobSummary> list() const;            // newest first
    // Cancel per the spec table. Returns false + current status when terminal
    // ("job_already_finished"); false + nullopt status when unknown ("unknown_job").
    // True once the cancel took effect: queued -> cancelled immediately (files
    // become not-run); running -> cancel flag only, the worker notices between
    // files and finalizes (completed files keep outcomes + artifacts).
    bool cancel(std::string_view id, std::optional<JobStatus>& current, std::string& error_code);
    // Shutdown path: flags every job cancelled, wakes the worker, waits up to
    // `bound` for it to finish. On expiry calls std::_Exit(0) directly (skip
    // static teardown; the leftover run dir dies at the next start's dead-run
    // cleanup). Only the clean-join path returns.
    void cancel_all_and_join(std::chrono::seconds bound);   // shutdown path
    void set_processor_for_tests(Processor p) { processor_ = std::move(p); }  // before the first create_job
private:
    struct Job;   // internal record (defined in the .cpp)
    void worker_loop();
    void run_job(Job& job);
    // The default processor body: decode -> rect bounds -> remove_still ->
    // write_still_output (runs on the worker thread; engine_ is already built).
    GuiFile process_file(GuiFile& f, const JobOptions& o, const std::filesystem::path& job_dir);
    void finalize_locked(Job& job, JobStatus status);   // first terminal state wins
    void request_stop_locked();
    std::string new_job_id_locked() const;

    std::filesystem::path run_dir_;
    // Aggregate-cap accounting: monotonic bytes-stored counter (orig uploads
    // + cleaned outputs; nothing is evicted mid-run), seeded by ONE walk of
    // the gui root at construction. Atomic: the worker adds output sizes
    // without the mutex; create_job compares under it.
    std::atomic<std::uintmax_t> stored_bytes_{0};
    Processor processor_;
    std::unique_ptr<WatermarkEngine> engine_;   // built in the worker thread, REUSED across all files/jobs
    std::thread worker_;
    mutable std::mutex mtx_;
    std::condition_variable wake_cv_;   // a job was queued / shutdown began
    std::condition_variable done_cv_;   // worker_done_ signal (the bounded-stop bound)
    std::deque<std::shared_ptr<Job>> queue_;
    std::vector<std::shared_ptr<Job>> jobs_;   // insertion order; list() returns newest first
    bool shutting_down_ = false;
    bool worker_done_ = false;
    std::uint64_t next_seq_ = 0;   // creation sequence (time_t created has 1 s granularity)
};

}  // namespace wmr::gui
