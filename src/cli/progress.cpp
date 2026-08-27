// CLI progress UX implementation. See progress.hpp for the design.

#include "cli/progress.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#  include <io.h>
#  ifndef NOMINMAX
#    define NOMINMAX  // windows.h min/max macros break std::min/std::max below
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#  define WMR_FILENO_STDERR (_fileno(stderr))
#  define WMR_FILENO_STDOUT (_fileno(stdout))
#  define WMR_ISATTY(fd) (_isatty(fd) != 0)
#else
#  include <unistd.h>
#  include <sys/ioctl.h>
#  define WMR_FILENO_STDERR STDERR_FILENO
#  define WMR_FILENO_STDOUT STDOUT_FILENO
#  define WMR_ISATTY(fd) (isatty(fd) != 0)
#endif

#include <fmt/format.h>

namespace wmr {

namespace {

std::atomic<bool> g_enabled{true};

// NOTE: getenv is read on every progress_is_tty() call rather than cached, so a
// test that sets NO_COLOR/TERM/CI in one process sees the effect immediately.
// (These env vars do not change during a real run.)
bool env_present_nonempty(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0';
}

FILE* stream_file(ProgressStream s) {
    return s == ProgressStream::Stderr ? stderr : stdout;
}

int stream_fd(ProgressStream s) {
    return s == ProgressStream::Stderr ? WMR_FILENO_STDERR : WMR_FILENO_STDOUT;
}

// Visible column count of the terminal behind `f` (fallback 80 when unknown).
// Refreshing lines are capped to this so a \r-rewrite never wraps to a second
// terminal row (a wrapped row survives above the cursor, which reads as one
// new line per refresh).
int terminal_width(FILE* f) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO ci{};
    if (GetConsoleScreenBufferInfo(reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(f))), &ci)) {
        int w = ci.srWindow.Right - ci.srWindow.Left + 1;
        if (w > 0) return w;
    }
    return 80;
#else
    struct winsize ws{};
    if (ioctl(fileno(f), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) {
        return static_cast<int>(ws.ws_col);
    }
    return 80;
#endif
}

std::string timestamp_now() {
    auto now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", &tm);
    return buf;
}

constexpr const char* kBold = "\033[1m";
constexpr const char* kReset = "\033[0m";

// Render `text` bold when `color` is on; plain text otherwise.
std::string bold_if(bool color, const std::string& text) {
    return color ? (kBold + text + kReset) : text;
}

} // namespace

// ---------------------------------------------------------------------------
// Gate + master switch
// ---------------------------------------------------------------------------
void set_progress_enabled(bool enabled) { g_enabled.store(enabled); }
bool progress_enabled() { return g_enabled.load(); }

bool progress_is_tty(ProgressStream s) {
    if (!g_enabled.load()) return false;
    if (!WMR_ISATTY(stream_fd(s))) return false;
    if (env_present_nonempty("NO_COLOR")) return false;
    if (env_present_nonempty("CI")) return false;
    if (const char* t = std::getenv("TERM"); t != nullptr && std::string(t) == "dumb") return false;
    return true;
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------
std::string format_bytes(uint64_t bytes) {
    constexpr uint64_t KiB = 1024ull;
    constexpr uint64_t MiB = KiB * 1024ull;
    constexpr uint64_t GiB = MiB * 1024ull;
    constexpr uint64_t TiB = GiB * 1024ull;
    if (bytes < KiB) return fmt::format("{} B", bytes);
    if (bytes < MiB) return fmt::format("{:.1f} KiB", static_cast<double>(bytes) / KiB);
    if (bytes < GiB) return fmt::format("{:.2f} MiB", static_cast<double>(bytes) / MiB);
    if (bytes < TiB) return fmt::format("{:.2f} GiB", static_cast<double>(bytes) / GiB);
    return fmt::format("{:.2f} TiB", static_cast<double>(bytes) / TiB);
}

std::string format_eta(double seconds) {
    if (!(seconds > 0.0)) return "--";  // 0, negative, NaN
    if (seconds < 10.0) return "<10s";
    if (seconds < 60.0) return fmt::format("{:.0f}s", seconds);
    if (seconds < 3600.0) {
        int total = static_cast<int>(seconds);
        return fmt::format("{}m{}s", total / 60, total % 60);
    }
    int total = static_cast<int>(seconds);
    return fmt::format("{}h{}m", total / 3600, (total % 3600) / 60);
}

std::string format_bar(double frac, int width) {
    if (width < 2) width = 2;
    if (!(frac > 0.0)) frac = 0.0;      // also clamps NaN
    if (frac > 1.0) frac = 1.0;
    int filled = static_cast<int>(std::lround(frac * width));
    if (filled > width) filled = width;
    if (filled < 0) filled = 0;
    return "[" + std::string(filled, '#') + std::string(width - filled, '-') + "]";
}

WindowedRate::WindowedRate(double window_s)
    : window_s_(window_s > 0.5 ? window_s : 0.5) {}

void WindowedRate::sample(uint64_t done, double t) {
    if (!started_) {
        started_ = true;
        win_done_ = done;
        win_t0_ = t;
        first_done_ = done;
        first_t0_ = t;
        return;
    }
    if (t <= win_t0_) return;  // non-monotonic time: ignore the sample
    const double win = t - win_t0_;
    if (win >= window_s_) {
        // A window closed: average over it, lightly blended with the previous
        // window so the displayed rate steps rather than snaps. A done < anchor
        // (a server that restarts the transfer) re-anchors without a rate.
        if (done >= win_done_) {
            const double inst = static_cast<double>(done - win_done_) / win;
            rate_ = (rate_ > 0.0) ? (0.5 * inst + 0.5 * rate_) : inst;
        }
        ++closed_;
        win_done_ = done;
        win_t0_ = t;
        return;
    }
    if (rate_ <= 0.0 && closed_ == 0) {
        // First window still filling: average since the first sample, shown
        // only once >= 2s of data exists (a shorter span is one TCP burst).
        const double span = t - first_t0_;
        if (span >= 2.0) {
            rate_ = static_cast<double>(done - first_done_) / span;
        }
    }
    // Otherwise keep the last completed window's rate (display stability).
}

double WindowedRate::rate_per_sec() const { return rate_; }
bool WindowedRate::full_window() const { return closed_ >= 2; }

std::string format_byte_line(const std::string& label, uint64_t done, uint64_t total,
                             double rate_bps, bool bold, int max_cols, bool eta_trusted) {
    // Clamped percent: a mis-reported total (or a server under-reporting
    // Content-Length) must never render "678855410%" (seen live on a fresh-mac
    // model download; it also overflowed the milestone bucket off a TTY).
    int pct = 0;
    if (total > 0) {
        double p = static_cast<double>(done) * 100.0 / static_cast<double>(total);
        if (p > 0.0) {
            pct = static_cast<int>(std::lround(p));
            if (pct > 100) pct = 100;
        }
    }
    std::string payload;
    if (total > 0) {
        payload = "  " + format_bytes(done) + " / " + format_bytes(total);
        payload += fmt::format("  {}%", pct);
        if (rate_bps > 0.0) {
            payload += "  " + format_bytes(static_cast<uint64_t>(rate_bps)) + "/s";
            double eta = (done < total && eta_trusted)
                ? static_cast<double>(total - done) / rate_bps : 0.0;
            if (eta > 0.0) payload += "  ETA " + format_eta(eta);
        }
    } else {
        // Total unknown: show what we have.
        payload = "  " + format_bytes(done);
        if (rate_bps > 0.0) {
            payload += "  " + format_bytes(static_cast<uint64_t>(rate_bps)) + "/s";
        }
    }

    const double frac = total > 0 ? static_cast<double>(done) / static_cast<double>(total) : 0.0;
    int bar_w = total > 0 ? 24 : 0;  // rendered bar = "  " + (cells + brackets) = w + 4
    if (max_cols > 0) {
        const size_t cap = static_cast<size_t>(max_cols);
        // Fit in priority order: shrink the bar (24 -> 0), then truncate the label,
        // then drop the bar / cut the payload on a degenerate (< ~40-col) terminal.
        // The payload (sizes / rate / ETA) is the information; the bar and label
        // are decoration.
        while (bar_w > 0 &&
               label.size() + payload.size() + static_cast<size_t>(bar_w) + 4 > cap) {
            --bar_w;
        }
        const size_t bar_len = bar_w > 0 ? static_cast<size_t>(bar_w) + 4 : 0;
        std::string lab = label;
        if (lab.size() + payload.size() + bar_len > cap) {
            size_t keep = cap > payload.size() + bar_len ? cap - payload.size() - bar_len : 0;
            lab = lab.substr(0, std::min(lab.size(), keep));
        }
        std::string bar = bar_w > 0 ? "  " + format_bar(frac, bar_w) : std::string();
        if (lab.size() + payload.size() + bar.size() > cap) {
            bar.clear();
            if (lab.size() + payload.size() > cap) {
                lab.clear();
                if (payload.size() > cap) payload.resize(cap);
            }
        }
        // Bold AFTER truncation so the ANSI codes stay balanced; escape bytes
        // never count toward the visible width.
        return bold_if(bold, lab) + payload + bar;
    }
    // Uncapped (non-TTY log lines).
    std::string bar = bar_w > 0 ? "  " + format_bar(frac, bar_w) : std::string();
    return bold_if(bold, label) + payload + bar;
}

// ---------------------------------------------------------------------------
// RateEstimator
// ---------------------------------------------------------------------------
RateEstimator::RateEstimator(int total, double alpha) : total_(total), alpha_(alpha) {}

void RateEstimator::sample(double unit_seconds) {
    if (!(unit_seconds > 0.0)) return;  // skip 0/negative/NaN (no time elapsed)
    if (samples_ == 0) ewma_unit_ = unit_seconds;
    else ewma_unit_ = alpha_ * unit_seconds + (1.0 - alpha_) * ewma_unit_;
    ++samples_;
}

double RateEstimator::rate_per_sec() const {
    if (samples_ == 0 || !(ewma_unit_ > 0.0)) return 0.0;
    return 1.0 / ewma_unit_;
}

bool RateEstimator::eta_available() const {
    if (samples_ < 3) return false;
    if (total_ <= 0) return true;  // no total gate available
    return static_cast<double>(samples_) / static_cast<double>(total_) >= 0.05;
}

bool RateEstimator::eta_confident() const {
    if (samples_ < 6) return false;
    if (total_ <= 0) return true;
    return static_cast<double>(samples_) / static_cast<double>(total_) >= 0.20;
}

double RateEstimator::eta_seconds(int remaining) const {
    if (!eta_available()) return 0.0;
    if (remaining <= 0) return 0.0;
    if (!(ewma_unit_ > 0.0)) return 0.0;
    return remaining * ewma_unit_;
}

int RateEstimator::sample_count() const { return samples_; }

double RateEstimator::avg_unit_seconds() const { return ewma_unit_; }

// ---------------------------------------------------------------------------
// Shared line rendering helpers
// ---------------------------------------------------------------------------
namespace {


// Format a discrete rate: "<rate> fps" when unit=="frame", else "<rate> <unit>/s".
std::string format_rate(double rate_per_sec, const std::string& unit) {
    if (unit == "frame") return fmt::format("{:.1f} fps", rate_per_sec);
    return fmt::format("{:.1f} {}/s", rate_per_sec, unit);
}

} // namespace

// ---------------------------------------------------------------------------
// ProgressReporter
// ---------------------------------------------------------------------------

// Forward declaration: the intra-unit refresh thread (in the constructor) and
// update()/finish() both build lines via this helper, which is defined below.
static std::string build_discrete_line(bool color, const std::string& label,
                                       int done_display, int total,
                                       double progress_frac,
                                       const RateEstimator& est,
                                       double eta_val,
                                       const std::string& unit,
                                       const std::string& extra);

struct ProgressReporter::Impl {
    std::string label;
    int total = 0;
    std::string unit;
    FILE* file = stderr;
    bool tty = false;
    int64_t throttle_ms = 66;
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point last_sample_time;
    std::chrono::steady_clock::time_point last_print_time;
    std::chrono::steady_clock::time_point unit_start_time;  // start of the in-flight unit
    int prev_done = 0;
    int last_bucket = -1;
    int prev_line_len = 0;
    std::string last_extra;       // reused by the intra-unit refresh thread
    bool ever_printed = false;    // false until the first render (bypasses throttle)
    bool final_printed = false;   // update() already printed the done==total state
    bool finished = false;
    RateEstimator est;

    // TTY intra-unit refresh: a background thread that periodically redraws the
    // line with the ETA + bar projected by elapsed time within the current unit
    // (so a 26s tile does not sit static). update()/finish() take this mutex too.
    std::mutex mu;
    std::condition_variable cv;
    std::thread refresh_thread;
    std::atomic<bool> refresh_stop{false};
    int64_t refresh_ms = 800;
};

ProgressReporter::ProgressReporter(std::string label, int total, std::string unit,
                                   ProgressStream s, int64_t throttle_ms) {
    if (!progress_enabled()) return;  // pimpl_ stays null -> all methods no-op
    pimpl_.reset(new Impl{});
    pimpl_->label = std::move(label);
    pimpl_->total = std::max(0, total);
    pimpl_->unit = std::move(unit);
    pimpl_->file = stream_file(s);
    pimpl_->tty = progress_is_tty(s);
    pimpl_->throttle_ms = throttle_ms;
    auto now = std::chrono::steady_clock::now();
    pimpl_->start = now;
    pimpl_->last_sample_time = now;
    pimpl_->last_print_time = now;
    pimpl_->unit_start_time = now;
    pimpl_->est = RateEstimator(pimpl_->total);

    // Spawn the intra-unit refresh thread only on a TTY (the refreshing medium).
    // It no-ops until the first update has run and there is rate data to project.
    if (pimpl_->tty) {
        Impl* impl = pimpl_.get();
        pimpl_->refresh_thread = std::thread([impl]() {
            while (!impl->refresh_stop.load()) {
                std::unique_lock<std::mutex> lk(impl->mu);
                impl->cv.wait_for(lk, std::chrono::milliseconds(impl->refresh_ms),
                                  [impl]{ return impl->refresh_stop.load(); });
                if (impl->refresh_stop.load()) return;
                if (impl->finished || !impl->ever_printed) continue;
                if (impl->total <= 0 || impl->prev_done >= impl->total) continue;
                double ewma = impl->est.avg_unit_seconds();
                if (!(ewma > 0.0)) continue;  // no rate data yet -> nothing to project
                double elapsed_unit = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - impl->unit_start_time).count();
                double in_flight = elapsed_unit / ewma;
                if (in_flight < 0.0) in_flight = 0.0;
                if (in_flight > 0.999) in_flight = 0.999;  // never complete a unit early
                double projected = static_cast<double>(impl->prev_done) + in_flight;
                double progress_frac = projected / static_cast<double>(impl->total);
                double remaining_units = static_cast<double>(impl->total) - projected;
                double eta_val = impl->est.eta_available()
                    ? std::max(0.0, remaining_units * ewma) : 0.0;
                std::string line = build_discrete_line(
                    /*color=*/true, impl->label, impl->prev_done, impl->total,
                    progress_frac, impl->est, eta_val, impl->unit, impl->last_extra);
                std::fputc('\r', impl->file);
                std::fputs(line.c_str(), impl->file);
                int len = static_cast<int>(line.size());
                if (len < impl->prev_line_len) {
                    std::fputs(std::string(static_cast<size_t>(impl->prev_line_len - len), ' ').c_str(),
                               impl->file);
                }
                impl->prev_line_len = len;
                std::fflush(impl->file);
            }
        });
    }
}

ProgressReporter::~ProgressReporter() {
    if (!pimpl_) return;
    {
        std::lock_guard<std::mutex> lk(pimpl_->mu);
        pimpl_->refresh_stop.store(true);
    }
    pimpl_->cv.notify_all();
    if (pimpl_->refresh_thread.joinable()) pimpl_->refresh_thread.join();
    if (!pimpl_->finished && pimpl_->tty) {
        // End any in-flight \r line so subsequent output starts on a fresh line.
        std::fputc('\n', pimpl_->file);
        std::fflush(pimpl_->file);
    }
}

// Build a discrete progress line. `progress_frac` drives the percent + bar
// (so an intra-unit projection can move them without bumping the integer count,
// which is shown as `done_display`). `eta_val` is precomputed by the caller
// (0 => hidden; otherwise a single point estimate). Rate is shown only when
// meaningful (>= 0.1 unit/s) so a slow tile does not print "0.0 tile/s".
static std::string build_discrete_line(bool color, const std::string& label,
                                       int done_display, int total,
                                       double progress_frac,
                                       const RateEstimator& est,
                                       double eta_val,
                                       const std::string& unit,
                                       const std::string& extra) {
    std::string line = bold_if(color, label);
    line += fmt::format(" {}/{}", done_display, total);

    if (total > 0) {
        if (!(progress_frac > 0.0)) progress_frac = 0.0;      // also clamps NaN
        if (progress_frac > 1.0) progress_frac = 1.0;
        int pct = static_cast<int>(progress_frac * 100.0 + 0.5);
        // V3: "frame 1/240" would otherwise read 0%; clamp so the first completed
        // unit shows >=1%.
        if (done_display >= 1 && pct < 1) pct = 1;
        line += fmt::format("  {}%", pct);
    }

    // C1: rate needs >=2 samples (a single reading is pure noise: "frame 1/240
    // 165.6 fps"). Also drop when too slow to be meaningful at 1 decimal
    // (< 0.1 unit/s) so a slow tile does not print "0.0 tile/s".
    if (est.sample_count() >= 2) {
        double r = est.rate_per_sec();
        if (r >= 0.1) line += "  " + format_rate(r, unit);
    }

    // ETA. Hidden until the gate (>=3 samples AND >=5% done); from then a single
    // point estimate (the EWMA), which the intra-unit refresh thread ticks down.
    if (eta_val > 0.0) {
        line += "  ETA " + format_eta(eta_val);
    }

    // C3: extra (filename / scene index) renders BEFORE the bar so the most
    // useful identifying field is the last to truncate on a narrow terminal
    // (... ETA 23s  scene 1/3  [####...]).
    if (!extra.empty()) line += "  " + extra;

    if (total > 0) {
        line += "  " + format_bar(progress_frac);
    }

    return line;
}

void ProgressReporter::update(int done, const std::string& extra) {
    if (!pimpl_) return;
    if (pimpl_->finished) return;
    auto& p = *pimpl_;
    if (done < 0) done = 0;
    if (p.total > 0 && done > p.total) done = p.total;

    std::lock_guard<std::mutex> lk(p.mu);
    auto now = std::chrono::steady_clock::now();
    if (done > p.prev_done) {
        double dt = std::chrono::duration<double>(now - p.last_sample_time).count();
        int units = done - p.prev_done;
        if (dt > 0.0 && units > 0) {
            p.est.sample(dt / static_cast<double>(units));
        }
        p.prev_done = done;
        p.last_sample_time = now;
        p.unit_start_time = now;  // the next unit begins now
    }
    p.last_extra = extra;

    bool force = (p.total > 0 && done >= p.total) || (p.total == 0 && done > 0);
    // The very first render always goes through (immediate feedback that work
    // started); after that the TTY throttle calms high-frequency loops.
    if (!force && p.ever_printed) {
        double since_print_ms = std::chrono::duration<double>(now - p.last_print_time).count() * 1000.0;
        if (since_print_ms < static_cast<double>(p.throttle_ms)) return;
    }

    double progress_frac = (p.total > 0) ? static_cast<double>(done) / p.total : 1.0;
    int remaining = (p.total > 0) ? (p.total - done) : 0;
    double eta_val = p.est.eta_seconds(remaining);
    std::string line = build_discrete_line(p.tty, p.label, done, p.total,
                                           progress_frac, p.est, eta_val,
                                           p.unit, extra);

    if (p.tty) {
        std::fputc('\r', p.file);
        std::fputs(line.c_str(), p.file);
        int len = static_cast<int>(line.size());
        if (len < p.prev_line_len) {
            std::fputs(std::string(static_cast<size_t>(p.prev_line_len - len), ' ').c_str(), p.file);
        }
        p.prev_line_len = len;
        std::fflush(p.file);
    } else {
        int bucket = (p.total > 0) ? (done * 20 / p.total) : done;
        if (force || bucket != p.last_bucket) {
            std::fprintf(p.file, "[%s] %s\n", timestamp_now().c_str(), line.c_str());
            p.last_bucket = bucket;
        }
    }
    p.ever_printed = true;
    if (force) p.final_printed = true;  // update() just rendered the done==total state
    p.last_print_time = now;
}

void ProgressReporter::finish(const std::string& summary) {
    if (!pimpl_ || pimpl_->finished) return;
    auto& p = *pimpl_;
    // Stop the intra-unit refresh thread first so it cannot race the final write.
    {
        std::lock_guard<std::mutex> lk(p.mu);
        p.refresh_stop.store(true);
    }
    p.cv.notify_all();
    if (p.refresh_thread.joinable()) p.refresh_thread.join();

    if (p.final_printed) {
        // update() already rendered the final state; just terminate the TTY line.
        if (p.tty) {
            std::fputc('\n', p.file);
            std::fflush(p.file);
        }
        p.finished = true;
        return;
    }
    if (p.total > 0 && p.prev_done < p.total) {
        // Caller never reached the final unit; snap the display to total.
        p.prev_done = p.total;
        auto now = std::chrono::steady_clock::now();
        p.last_sample_time = now;
    }
    double progress_frac = 1.0;  // finish() always renders the full bar
    double eta_val = 0.0;
    std::string line = build_discrete_line(p.tty, p.label,
                                           p.total > 0 ? p.total : p.prev_done,
                                           p.total, progress_frac, p.est,
                                           eta_val, p.unit, summary);
    if (p.tty) {
        std::fputc('\r', p.file);
        std::fputs(line.c_str(), p.file);
        int len = static_cast<int>(line.size());
        if (len < p.prev_line_len) {
            std::fputs(std::string(static_cast<size_t>(p.prev_line_len - len), ' ').c_str(), p.file);
        }
        std::fputc('\n', p.file);
        std::fflush(p.file);
    } else {
        std::fprintf(p.file, "[%s] %s\n", timestamp_now().c_str(), line.c_str());
    }
    p.finished = true;
}

// ---------------------------------------------------------------------------
// ByteProgress
// ---------------------------------------------------------------------------
struct ByteProgress::Impl {
    std::string label;
    uint64_t total = 0;
    FILE* file = stderr;
    bool tty = false;
    int64_t throttle_ms = 500;
    std::chrono::steady_clock::time_point last_print_time;
    uint64_t last_bytes = 0;
    int last_bucket = -1;
    int prev_line_len = 0;
    WindowedRate rate{4.0};  // bytes/sec over the last window (see WindowedRate)
    bool final_printed = false;
    bool finished = false;
};

ByteProgress::ByteProgress(std::string label, uint64_t total_bytes,
                           ProgressStream s, int64_t throttle_ms) {
    if (!progress_enabled()) return;  // pimpl_ stays null -> all methods no-op
    pimpl_.reset(new Impl{});
    pimpl_->label = std::move(label);
    pimpl_->total = total_bytes;
    pimpl_->file = stream_file(s);
    pimpl_->tty = progress_is_tty(s);
    pimpl_->throttle_ms = throttle_ms;
    auto now = std::chrono::steady_clock::now();
    pimpl_->last_print_time = now;
}

ByteProgress::~ByteProgress() {
    if (!pimpl_) return;
    if (!pimpl_->finished && pimpl_->tty) {
        std::fputc('\n', pimpl_->file);
        std::fflush(pimpl_->file);
    }
}

void ByteProgress::set_total(uint64_t total_bytes) {
    if (!pimpl_) return;
    // Last non-zero total wins: an earlier bogus value (e.g. a redirect page's
    // Content-Length arriving before the final response) must not latch for the
    // rest of the transfer ("6.46 GiB / 1022 B" on a real 6.9 GB fetch).
    if (total_bytes > 0) pimpl_->total = total_bytes;
}

bool ByteProgress::update(uint64_t done) {
    if (!pimpl_) return true;
    auto& p = *pimpl_;

    auto now = std::chrono::steady_clock::now();
    const double t = std::chrono::duration<double>(now.time_since_epoch()).count();
    // Sample every tick (even byte-less ones) so a stall closes empty windows
    // and the displayed rate decays instead of freezing at the last burst.
    p.rate.sample(done, t);
    p.last_bytes = done;

    bool force = (p.total > 0 && done >= p.total);
    if (!force) {
        double since_print_ms = std::chrono::duration<double>(now - p.last_print_time).count() * 1000.0;
        if (since_print_ms < static_cast<double>(p.throttle_ms)) return true;
    }

    if (p.tty) {
        // Capped to the terminal width: an overlong line WRAPS, and the wrapped
        // row cannot be rewritten by \r (it survives above the cursor), which
        // reads as one new line per refresh.
        std::string line = format_byte_line(p.label, done, p.total, p.rate.rate_per_sec(),
                                            /*bold=*/true, terminal_width(p.file),
                                            p.rate.full_window());
        std::fputc('\r', p.file);
        std::fputs(line.c_str(), p.file);
        int len = static_cast<int>(line.size());
        if (len < p.prev_line_len) {
            std::fputs(std::string(static_cast<size_t>(p.prev_line_len - len), ' ').c_str(), p.file);
        }
        p.prev_line_len = len;
        std::fflush(p.file);
    } else {
        std::string line = format_byte_line(p.label, done, p.total, p.rate.rate_per_sec(),
                                            /*bold=*/false, /*max_cols=*/0,
                                            p.rate.full_window());
        // Milestone bucket from the CLAMPED percent (0..20): with a bogus total
        // the raw done*20/total quotient exploded past 2^31 and changed every
        // tick, printing a line per throttle window.
        int pct = 0;
        if (p.total > 0) {
            double v = static_cast<double>(done) * 100.0 / static_cast<double>(p.total);
            pct = v > 0.0 ? static_cast<int>(std::lround(v)) : 0;
            if (pct > 100) pct = 100;
        }
        int bucket = pct / 5;
        // force (done == total) prints only ONCE: the final xferinfo ticks of a
        // fast transfer all report the complete state and would otherwise print
        // several identical 100% lines.
        if ((force && !p.final_printed) || bucket != p.last_bucket) {
            std::fprintf(p.file, "[%s] %s\n", timestamp_now().c_str(), line.c_str());
            p.last_bucket = bucket;
        }
    }
    if (force) p.final_printed = true;
    p.last_print_time = now;
    return true;  // never cancel from inside progress (forwarded to curl xferinfo)
}

void ByteProgress::finish() {
    if (!pimpl_ || pimpl_->finished) return;
    auto& p = *pimpl_;
    if (p.final_printed) {
        if (p.tty) {
            std::fputc('\n', p.file);
            std::fflush(p.file);
        }
        p.finished = true;
        return;
    }
    std::string line = format_byte_line(p.label, p.last_bytes, p.total, p.rate.rate_per_sec(),
                                        /*bold=*/p.tty,
                                        p.tty ? terminal_width(p.file) : 0,
                                        p.rate.full_window());
    if (p.tty) {
        std::fputc('\r', p.file);
        std::fputs(line.c_str(), p.file);
        int len = static_cast<int>(line.size());
        if (len < p.prev_line_len) {
            std::fputs(std::string(static_cast<size_t>(p.prev_line_len - len), ' ').c_str(), p.file);
        }
        std::fputc('\n', p.file);
        std::fflush(p.file);
    } else {
        std::fprintf(p.file, "[%s] %s\n", timestamp_now().c_str(), line.c_str());
    }
    p.finished = true;
}

// ---------------------------------------------------------------------------
// Stage
// ---------------------------------------------------------------------------
Stage::Stage(int index, int total, std::string name, std::string note,
             ProgressStream s) {
    if (!progress_enabled()) return;
    FILE* file = stream_file(s);
    bool tty = progress_is_tty(s);
    std::string prefix;
    if (total > 0) {
        prefix = fmt::format("[{}/{}] ", index, total);
    }
    std::string body = prefix + name;
    if (tty) body = bold_if(true, body);
    if (!note.empty()) body += "  (" + note + ")";
    if (tty) {
        // Clean header line on a terminal (no timestamp).
        std::fputs(body.c_str(), file);
        std::fputc('\n', file);
    } else {
        // Append-only with a timestamp in CI/piped logs.
        std::fprintf(file, "[%s] %s\n", timestamp_now().c_str(), body.c_str());
    }
}

} // namespace wmr
