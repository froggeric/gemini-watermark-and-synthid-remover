// CLI progress UX for wmr.
//
// A small TTY-aware reporter for the long-running operations (SynthID regen,
// video, batch, model downloads). Per docs/research/cli-progress-ux-design.md:
//   - Progress writes to STDERR; spdlog info stays on stdout (so
//     `wmr ... > /dev/null` still shows progress, and `> out.png` keeps stdout
//     clean for scripted use).
//   - TTY: a refreshing single line per stage (\r-throttled, ANSI bold labels,
//     ASCII bar). Non-TTY (piped/CI): append-only milestone lines, on a
//     percent-bucket change, each prefixed with a timestamp.
//   - Color/fancy mode is gated on isatty + NO_COLOR unset + TERM != "dumb" +
//     CI unset; progress_is_tty() is the single gate. Never color-only: the
//     text is fully readable with the codes stripped.
//   - --no-progress (set_progress_enabled(false)) turns every reporter into a
//     no-op so only errors + the final summary remain.
//
// The pure helpers (format_*, RateEstimator) are unit-tested in
// tests/unit/progress_test.cpp. The reporters are pimpl + stateful and are
// verified manually on a TTY.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace wmr {

enum class ProgressStream { Stderr, Stdout };

// The combined TTY + color/fancy gate. True only when the stream is a TTY AND
// NO_COLOR is unset AND TERM != "dumb" AND CI is unset. This is the single
// switch the reporters use to pick refreshing-line vs append-only-milestone
// rendering, and to decide whether to emit ANSI bold.
bool progress_is_tty(ProgressStream s);

// Master on/off (set once from --no-progress in run_cli, before any reporter
// is constructed). Default enabled. When disabled, Stage/ProgressReporter/
// ByteProgress are no-ops.
void set_progress_enabled(bool enabled);
bool progress_enabled();

// --- Pure helpers (unit-tested) ---

// Human-readable byte count with binary units: "512 B", "4.0 KiB",
// "18.20 MiB", "3.41 GiB", "1.20 TiB".
std::string format_bytes(uint64_t bytes);

// Human-readable duration: 0/negative/NaN -> "--", <10s -> "<10s",
// <60s -> "45s", <3600s -> "2m47s", else "1h5m".
std::string format_eta(double seconds);

// ASCII progress bar: "[####---------]". frac is clamped to [0,1]; width is
// the count of inner cells (default 24). Filled cells are '#', empty '-'.
std::string format_bar(double frac, int width = 24);

// Exponentially-weighted moving average of time-per-unit (e.g. seconds/tile).
// ETA is hidden (eta_seconds returns 0) until BOTH >=3 samples AND >=5% of the
// total is done. A "confident" point estimate (vs a wide range) requires more
// data (>=6 samples AND >=20% done). Pure; unit-tested.
class RateEstimator {
public:
    // NOTE: not `explicit` so a pimpl aggregate with a RateEstimator member can
    // be value-initialized (`new Impl{}`); the conversion risk (implicit
    // RateEstimator from int) is negligible — the class is constructed deliberately.
    RateEstimator(int total = 0, double alpha = 0.3);

    // Record one more completed unit. unit_seconds is the time this unit took.
    void sample(double unit_seconds);

    // EWMA of units-per-second (1 / ewma_time_per_unit). 0 until the first
    // sample.
    double rate_per_sec() const;

    // ETA in seconds for `remaining` units, or 0 when not enough data (see the
    // hide-until rule above). When non-zero the caller may still format it as a
    // range unless eta_confident() is true.
    double eta_seconds(int remaining) const;

    // True once the ETA gate is satisfied (enough data to show any ETA).
    bool eta_available() const;
    // True once the estimate is tight enough to show as a single value rather
    // than a range.
    bool eta_confident() const;

    // The rolling EWMA of seconds-per-unit (0 until the first sample). Exposed so
    // a refresh can project intra-unit progress (fraction of a unit elapsed).
    double avg_unit_seconds() const;

    int sample_count() const;

private:
    int total_;
    double alpha_;
    double ewma_unit_ = 0.0;  // EWMA of seconds-per-unit
    int samples_ = 0;
};

// Discrete progress: tiles / frames / batch items. One refreshing line on a
// TTY (\r-throttled), append-only milestone lines off a TTY.
class ProgressReporter {
public:
    // throttle_ms: min interval between refreshes on a TTY (default 66 ms for
    // frame/tile loops). Pass a larger value for slower loops (e.g. 2000 for
    // video frames) to match the old throttle floor.
    ProgressReporter(std::string label, int total, std::string unit,
                     ProgressStream s = ProgressStream::Stderr,
                     int64_t throttle_ms = 66);
    ~ProgressReporter();
    ProgressReporter(const ProgressReporter&) = delete;
    ProgressReporter& operator=(const ProgressReporter&) = delete;

    // Report `done` of total units. `extra` is appended verbatim (e.g.
    // "coreml-gpu, 4.1s/tile" or "scene 1/3"). Throttled except on the final
    // update (done == total).
    void update(int done, const std::string& extra = "");

    // Print the final line (done == total) and end it. Optional on a TTY (the
    // destructor also ends the line), but lets the caller emit a clean final
    // state before constructing the next Stage.
    void finish(const std::string& summary = "");

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// Byte-based progress for downloads, driven by curl's xferinfo callback.
// `update(done)` returns false to cancel the transfer (the curl contract:
// xferinfo returns non-zero to abort); the default implementation never
// cancels, but the return is forwarded so a future signal hook can.
class ByteProgress {
public:
    ByteProgress(std::string label, uint64_t total_bytes,
                 ProgressStream s = ProgressStream::Stderr,
                 int64_t throttle_ms = 500);
    ~ByteProgress();
    ByteProgress(const ByteProgress&) = delete;
    ByteProgress& operator=(const ByteProgress&) = delete;

    // For resumable downloads where the total is only known once curl receives
    // the Content-Length / Content-Range. Calling this after construction with
    // a non-zero total enables the percent/bar/ETA fields.
    void set_total(uint64_t total_bytes);

    // Report `done` cumulative bytes. Returns true to continue, false to abort.
    bool update(uint64_t done);

    // Print the final 100% line and end it. Optional (destructor cleans up).
    void finish();

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

// One-shot stage frame: "[k/N] <name> (<note>)". Printed once, on its own
// line, when the stage is entered. When total == 0 the "[k/N] " prefix is
// omitted (a plain labeled line). On a TTY the prefix + name are bold.
class Stage {
public:
    Stage(int index, int total, std::string name, std::string note = "",
          ProgressStream s = ProgressStream::Stderr);
    ~Stage() = default;
};

} // namespace wmr
