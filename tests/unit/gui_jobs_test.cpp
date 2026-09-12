// GUI job manager unit tests (Task 4): per-run dirs + dead-pid cleanup, the
// cancel transition table, queue caps, per-file exception isolation, and one
// end-to-end case with the REAL processor (engine round trip on an
// add_watermark image). Deterministic cases inject a processor; the
// promise/future pairs pin the worker between files so the manager's
// between-files cancel check is exercised deterministically.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "core/watermark_engine.hpp"
#include "gui/jobs.hpp"
#include "gui/run_dir.hpp"

using namespace wmr::gui;
namespace fs = std::filesystem;

namespace {

// RAII temp dir (unique name: pid + counter + steady-clock ns).
struct TempDir {
    fs::path p;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        const auto ns = std::chrono::steady_clock::now().time_since_epoch().count();
        p = fs::temp_directory_path() / ("wmr_gui_jobs_" + std::to_string(getpid()) + "_" +
                                         std::to_string(counter.fetch_add(1)) + "_" +
                                         std::to_string(ns));
        fs::create_directories(p);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(p, ec);
    }
};

// Fires a blocking processor's release promise when the test scope exits, even
// on assertion failure - otherwise a worker stuck in the processor would ride
// JobManager's 5 s bounded stop into std::_Exit(0) and kill the whole binary.
// Declare AFTER the JobManager so it is destroyed (and releases) BEFORE the
// manager's destructor joins the worker.
struct ReleaseOnExit {
    std::promise<void>& p;
    ~ReleaseOnExit() {
        try {
            p.set_value();
        } catch (...) {
        }
    }
};

std::string png_bytes(int width, int height) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(90, 90, 90));
    std::vector<unsigned char> buf;
    cv::imencode(".png", img, buf);
    return std::string(buf.begin(), buf.end());
}

// The canonical detectable fixture (same recipe as still_remove_test): a V1
// mark added by the engine at the model position on gray content.
std::string marked_png_bytes() {
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(90, 90, 90));
    wmr::WatermarkEngine engine;
    engine.add_watermark(img);
    std::vector<unsigned char> buf;
    cv::imencode(".png", img, buf);
    return std::string(buf.begin(), buf.end());
}

// Poll until the job reaches a terminal state; fail the test on timeout.
JobStatus wait_terminal(const JobManager& m, const std::string& id, int timeout_ms = 10000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    for (;;) {
        auto snap = m.get(id);
        if (snap && (snap->status == JobStatus::Done || snap->status == JobStatus::Failed ||
                     snap->status == JobStatus::Cancelled))
            return snap->status;
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// An injected processor that instantly "removes" every file (no disk work).
GuiFile instant_removed(GuiFile& f, const JobOptions&, const fs::path&) {
    f.outcome = FileOutcome::Removed;
    f.has_clean = true;
    return f;
}

void write_text(const fs::path& p, const std::string& s) {
    std::ofstream out(p, std::ios::trunc);
    out << s;
}

}  // namespace

TEST_CASE("run dir: create + dead-pid cleanup keeps live siblings", "[gui][gui-jobs]") {
    TempDir root;
    const fs::path run = create_run_dir(root.p);
    REQUIRE(fs::exists(run));
    REQUIRE(fs::exists(run / "pid"));
    std::error_code ec;
    const auto perms = fs::status(run, ec).permissions();
    REQUIRE((perms & fs::perms::mask) == fs::perms::owner_all);  // 0700, private cache

    // A live sibling (this process) must survive cleanup...
    const fs::path live = root.p / "20260101000001-2";
    fs::create_directories(live);
    write_text(live / "pid", std::to_string(getpid()));
    // ...a dir without a readable integer pid sidecar is never touched...
    const fs::path stray = root.p / "not-a-run";
    fs::create_directories(stray);
    const fs::path garbage = root.p / "20260101000002-3";
    fs::create_directories(garbage);
    write_text(garbage / "pid", "not-a-number");

    cleanup_dead_runs(root.p);

    REQUIRE(fs::exists(run));  // our own run dir (live pid)
    REQUIRE(fs::exists(live));
    REQUIRE(fs::exists(stray));
    REQUIRE(fs::exists(garbage));

#ifndef _WIN32
    // A genuinely dead pid (reaped child) is cleaned up.
    const pid_t child = fork();
    REQUIRE(child >= 0);
    if (child == 0) _exit(0);  // child touches nothing but _exit
    REQUIRE(waitpid(child, nullptr, 0) == child);
    const fs::path dead = root.p / "20260101000003-4";
    fs::create_directories(dead);
    write_text(dead / "pid", std::to_string(child));
    cleanup_dead_runs(root.p);
    REQUIRE_FALSE(fs::exists(dead));
    REQUIRE(fs::exists(live));
#endif

    remove_run_dir(run);
    REQUIRE_FALSE(fs::exists(run));
}

TEST_CASE("create_job persists origs and appends sniff-rejected rows", "[gui][gui-jobs]") {
    TempDir root;
    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests(instant_removed);

    std::string id;
    auto err = mgr.create_job({{"photo.png", "png", png_bytes(30, 30)}},
                              {{"x.heic", "unsupported format: x.heic (use PNG, JPEG, or WebP)"}},
                              JobOptions{}, 100, 1LL << 30, id);
    REQUIRE_FALSE(err.has_value());
    REQUIRE(id.size() == 12);

    REQUIRE(wait_terminal(mgr, id) == JobStatus::Done);
    auto snap = mgr.get(id);
    REQUIRE(snap.has_value());
    REQUIRE(snap->files.size() == 2);
    REQUIRE(snap->files[0].name == "photo.png");
    REQUIRE(snap->files[0].outcome == FileOutcome::Removed);
    REQUIRE(snap->files[1].name == "x.heic");
    REQUIRE(snap->files[1].outcome == FileOutcome::Failed);
    REQUIRE(snap->files[1].has_orig == false);
    REQUIRE(snap->files[1].error.find("unsupported") != std::string::npos);

    // The orig the manager persisted (single owner) resolves via file_artifact.
    auto orig = mgr.file_artifact(id, 0, false);
    REQUIRE(orig.has_value());
    REQUIRE(fs::exists(*orig));
    REQUIRE(orig->filename() == "0.png");
    // The sniff-rejected row has no artifacts of either kind.
    REQUIRE_FALSE(mgr.file_artifact(id, 1, false).has_value());
    REQUIRE_FALSE(mgr.file_artifact(id, 1, true).has_value());
    // Unknown job / unknown index -> nullopt, never a path.
    REQUIRE_FALSE(mgr.file_artifact("nope", 0, false).has_value());
    REQUIRE_FALSE(mgr.file_artifact(id, 9, true).has_value());
}

TEST_CASE("cancel on a queued job is immediate; files never run", "[gui][gui-jobs]") {
    TempDir root;
    std::promise<void> first_entered;
    std::promise<void> release;
    std::shared_future<void> go = release.get_future().share();
    JobManager mgr(root.p / "run");
    std::atomic<int> calls{0};
    mgr.set_processor_for_tests([&first_entered, &calls, go](GuiFile& f, const JobOptions&,
                                                             const fs::path&) mutable {
        if (calls.fetch_add(1) == 0) {  // block only inside job 1's single file
            first_entered.set_value();
            go.wait();
        }
        f.outcome = FileOutcome::Removed;
        f.has_clean = true;
        return f;
    });
    ReleaseOnExit on_fail{release};

    std::string id1, id2;
    REQUIRE_FALSE(mgr.create_job({{"a.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                                 1LL << 30, id1));
    first_entered.get_future().wait();  // worker is inside job 1's file
    REQUIRE_FALSE(mgr.create_job({{"b.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                                 1LL << 30, id2));
    auto snap = mgr.get(id2);
    REQUIRE(snap->status == JobStatus::Queued);
    REQUIRE(snap->queue_position == 1);

    std::optional<JobStatus> cur;
    std::string code;
    REQUIRE(mgr.cancel(id2, cur, code));
    REQUIRE(*cur == JobStatus::Queued);
    // Immediate, no worker involvement needed.
    auto after = mgr.get(id2);
    REQUIRE(after->status == JobStatus::Cancelled);
    REQUIRE(after->files[0].outcome == FileOutcome::NotRun);

    release.set_value();
    REQUIRE(wait_terminal(mgr, id1) == JobStatus::Done);
    // The cancelled job never ran, even after the worker drained the queue.
    auto drained = mgr.get(id2);
    REQUIRE(drained->status == JobStatus::Cancelled);
    REQUIRE(drained->files[0].outcome == FileOutcome::NotRun);
}

TEST_CASE("cancel on a running job keeps completed files and not-runs the rest",
          "[gui][gui-jobs]") {
    TempDir root;
    std::promise<void> f0_done;
    std::promise<void> release;
    std::shared_future<void> go = release.get_future().share();
    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests([&f0_done, go](GuiFile& f, const JobOptions&,
                                               const fs::path&) mutable {
        f.outcome = FileOutcome::Removed;
        f.has_clean = true;
        if (f.index == 0) {
            f0_done.set_value();
            go.wait();  // simulate long work on file 0; the cancel lands here
        }
        return f;
    });
    ReleaseOnExit on_fail{release};

    std::string id;
    REQUIRE_FALSE(mgr.create_job({{"a.png", "png", png_bytes(30, 30)},
                                  {"b.png", "png", png_bytes(30, 30)}},
                                 {}, JobOptions{}, 100, 1LL << 30, id));
    f0_done.get_future().wait();

    auto snap = mgr.get(id);
    REQUIRE(snap->status == JobStatus::Running);
    REQUIRE(snap->current_file_index == 0);

    std::optional<JobStatus> cur;
    std::string code;
    REQUIRE(mgr.cancel(id, cur, code));  // running: flag only, worker finalizes
    REQUIRE(*cur == JobStatus::Running);

    release.set_value();
    REQUIRE(wait_terminal(mgr, id) == JobStatus::Cancelled);

    auto done = mgr.get(id);
    REQUIRE(done->files[0].outcome == FileOutcome::Removed);  // outcome kept...
    REQUIRE(mgr.file_artifact(id, 0, true).has_value());      // ...and downloadable
    REQUIRE(done->files[1].outcome == FileOutcome::NotRun);   // never reached
    REQUIRE_FALSE(mgr.file_artifact(id, 1, true).has_value());
}

TEST_CASE("cancel on a terminal job is rejected; unknown id is unknown_job",
          "[gui][gui-jobs]") {
    TempDir root;
    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests(instant_removed);
    std::string id;
    REQUIRE_FALSE(mgr.create_job({{"a.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                                 1LL << 30, id));
    REQUIRE(wait_terminal(mgr, id) == JobStatus::Done);

    std::optional<JobStatus> cur;
    std::string code;
    REQUIRE_FALSE(mgr.cancel(id, cur, code));
    REQUIRE(*cur == JobStatus::Done);
    REQUIRE(code == "job_already_finished");

    cur.reset();
    code.clear();
    REQUIRE_FALSE(mgr.cancel("deadbeefdead", cur, code));
    REQUIRE_FALSE(cur.has_value());
    REQUIRE(code == "unknown_job");
}

TEST_CASE("a ninth outstanding job is rejected as too_many_pending", "[gui][gui-jobs]") {
    TempDir root;
    std::promise<void> release;
    std::shared_future<void> go = release.get_future().share();
    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests([go](GuiFile& f, const JobOptions&, const fs::path&) {
        go.wait();  // every call blocks until released: 1 running + 7 queued
        f.outcome = FileOutcome::Removed;
        f.has_clean = true;
        return f;
    });
    ReleaseOnExit on_fail{release};

    std::string id;
    for (int i = 0; i < 8; ++i) {
        INFO("create job " << i);
        REQUIRE_FALSE(mgr.create_job({{"a.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                                     1LL << 30, id));
    }
    // 8 outstanding (1 running inside the processor + 7 queued): the 9th is rejected.
    std::string id9 = "sentinel";
    auto err = mgr.create_job({{"a.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                              1LL << 30, id9);
    REQUIRE(err.has_value());
    REQUIRE(*err == "too_many_pending");
    REQUIRE(id9 == "sentinel");  // out-param untouched on rejection

    release.set_value();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        bool all_done = true;
        for (const auto& j : mgr.list())
            if (j.status != JobStatus::Done) all_done = false;
        if (all_done) break;
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    REQUIRE(mgr.list().size() == 8);
}

TEST_CASE("a throwing file fails alone; the job still completes", "[gui][gui-jobs]") {
    TempDir root;
    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests([](GuiFile& f, const JobOptions&, const fs::path&) {
        if (f.index == 0) throw std::runtime_error("boom");
        f.outcome = FileOutcome::Removed;
        f.has_clean = true;
        return f;
    });
    std::string id;
    REQUIRE_FALSE(mgr.create_job({{"a.png", "png", png_bytes(30, 30)},
                                  {"b.png", "png", png_bytes(30, 30)}},
                                 {}, JobOptions{}, 100, 1LL << 30, id));
    REQUIRE(wait_terminal(mgr, id) == JobStatus::Done);
    auto snap = mgr.get(id);
    REQUIRE(snap->files[0].outcome == FileOutcome::Failed);
    REQUIRE(snap->files[0].error == "boom");
    REQUIRE(snap->files[0].has_clean == false);
    REQUIRE(snap->files[1].outcome == FileOutcome::Removed);
}

TEST_CASE("a job where every file fails is still done, not failed", "[gui][gui-jobs]") {
    TempDir root;
    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests([](GuiFile&, const JobOptions&, const fs::path&) -> GuiFile {
        throw std::runtime_error("always broken");
    });
    std::string id;
    REQUIRE_FALSE(mgr.create_job({{"a.png", "png", png_bytes(30, 30)},
                                  {"b.png", "png", png_bytes(30, 30)}},
                                 {}, JobOptions{}, 100, 1LL << 30, id));
    REQUIRE(wait_terminal(mgr, id) == JobStatus::Done);  // per-file failures never fail the job
    auto snap = mgr.get(id);
    REQUIRE(snap->files[0].outcome == FileOutcome::Failed);
    REQUIRE(snap->files[1].outcome == FileOutcome::Failed);
}

TEST_CASE("per-job file and byte caps reject with payload_too_large", "[gui][gui-jobs]") {
    TempDir root;
    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests(instant_removed);
    std::string id;
    auto err = mgr.create_job({{"a.png", "png", png_bytes(30, 30)},
                               {"b.png", "png", png_bytes(30, 30)}},
                              {}, JobOptions{}, /*max_files=*/1, 1LL << 30, id);
    REQUIRE(err == std::optional<std::string>("payload_too_large"));
    err = mgr.create_job({{"a.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                         /*max_bytes=*/10, id);
    REQUIRE(err == std::optional<std::string>("payload_too_large"));
    REQUIRE(id.empty());
}

TEST_CASE("the 4 GiB aggregate gate rejects a job over stored bytes", "[gui][gui-jobs]") {
    TempDir root;
    // stored_bytes_under sums regular files recursively.
    const fs::path nested = root.p / "sub" / "deep";
    fs::create_directories(nested);
    { std::ofstream out(root.p / "a.bin", std::ios::binary); out << std::string(1000, 'x'); }
    { std::ofstream out(nested / "b.bin", std::ios::binary); out << std::string(3000, 'x'); }
    REQUIRE(stored_bytes_under(root.p) == 4000);

    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests(instant_removed);
    // Under the cap: accepted and completes.
    std::string id;
    REQUIRE_FALSE(mgr.create_job({{"a.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                                 1LL << 30, id));
    REQUIRE(wait_terminal(mgr, id) == JobStatus::Done);

#ifndef _WIN32
    // A 5 GiB sparse file: logical size counts, no disk is consumed.
    const fs::path sparse = root.p / "huge.bin";
    const int fd = ::open(sparse.c_str(), O_WRONLY | O_CREAT, 0600);
    REQUIRE(fd >= 0);
    REQUIRE(::ftruncate(fd, 5LL << 30) == 0);
    ::close(fd);
    REQUIRE(stored_bytes_under(root.p) > std::uintmax_t(4LL << 30));

    std::string id2;
    auto err = mgr.create_job({{"a.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                              1LL << 30, id2);
    REQUIRE(err == std::optional<std::string>("payload_too_large"));
    REQUIRE(id2.empty());
#endif
}

TEST_CASE("list returns jobs newest first", "[gui][gui-jobs]") {
    TempDir root;
    JobManager mgr(root.p / "run");
    mgr.set_processor_for_tests(instant_removed);
    std::string ida, idb, idc;
    REQUIRE_FALSE(mgr.create_job({{"a.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                                 1LL << 30, ida));
    REQUIRE_FALSE(mgr.create_job({{"b.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                                 1LL << 30, idb));
    REQUIRE_FALSE(mgr.create_job({{"c.png", "png", png_bytes(30, 30)}}, {}, JobOptions{}, 100,
                                 1LL << 30, idc));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        bool all_done = true;
        for (const auto& j : mgr.list())
            if (j.status != JobStatus::Done) all_done = false;
        if (all_done) break;
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const auto jobs = mgr.list();
    REQUIRE(jobs.size() == 3);
    REQUIRE(jobs[0].id == idc);  // newest first
    REQUIRE(jobs[1].id == idb);
    REQUIRE(jobs[2].id == ida);
    REQUIRE(jobs[0].file_count == 1);
}

TEST_CASE("real processor: undecodable bytes and an out-of-bounds rect fail their file alone",
          "[gui][gui-jobs]") {
    TempDir root;
    JobManager mgr(root.p / "run");  // the real engine processor (no injection)

    JobOptions opts;
    opts.rect = cv::Rect(460, 460, 100, 100);   // 460+100 > 500: exceeds the image
    std::string id;
    REQUIRE_FALSE(mgr.create_job(
        {{"garbage.png", "png", "not a png at all"},
         {"small.png", "png", png_bytes(30, 30)}},   // valid decode, bad rect
        {}, opts, 100, 1LL << 30, id));
    REQUIRE(wait_terminal(mgr, id) == JobStatus::Done);   // per-file failures, job fine

    auto snap = mgr.get(id);
    REQUIRE(snap->files[0].outcome == FileOutcome::Failed);
    REQUIRE(snap->files[0].error == "decode failed");
    REQUIRE_FALSE(mgr.file_artifact(id, 0, true).has_value());
    REQUIRE(snap->files[1].outcome == FileOutcome::Failed);
    REQUIRE(snap->files[1].error == "rect 100x100 at (460,460) exceeds image 30x30");
    REQUIRE_FALSE(mgr.file_artifact(id, 1, true).has_value());
}

TEST_CASE("end to end: the real processor removes a mark and writes a decodable artifact",
          "[gui][gui-jobs]") {
    TempDir root;
    const fs::path run = create_run_dir(root.p);
    JobManager mgr(run);  // no injection: the default engine processor runs

    std::string id;
    REQUIRE_FALSE(mgr.create_job({{"marked.png", "png", marked_png_bytes()}}, {}, JobOptions{},
                                  100, 1LL << 30, id));
    REQUIRE(wait_terminal(mgr, id) == JobStatus::Done);

    auto snap = mgr.get(id);
    REQUIRE(snap->files[0].outcome == FileOutcome::Removed);
    REQUIRE(snap->files[0].has_clean);
    REQUIRE(snap->files[0].bbox.has_value());
    REQUIRE(snap->files[0].score > 0.0f);
    REQUIRE_FALSE(snap->files[0].geometry_source.empty());

    auto cleaned = mgr.file_artifact(id, 0, true);
    REQUIRE(cleaned.has_value());
    REQUIRE(fs::exists(*cleaned));
    cv::Mat back = cv::imread(cleaned->string());
    REQUIRE_FALSE(back.empty());
    REQUIRE(back.cols == 500);
    REQUIRE(back.rows == 500);
    auto orig = mgr.file_artifact(id, 0, false);
    REQUIRE(orig.has_value());
    REQUIRE(fs::exists(*orig));
}
