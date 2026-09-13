#include "gui/update.hpp"

// The background worker is a plain thread joined by stop_update_check() in
// run_gui's graceful tail; the _Exit paths (shutdown timeout, second Ctrl-C)
// die with the process instead. state()'s static is only ever destroyed on a
// path that already joined (run_gui's normal return).

#include <chrono>
#include <cstdlib>
#include <mutex>
#include <thread>

#include "core/paths.hpp"

#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"
#endif

namespace wmr::gui {
namespace {

struct UpdateState {
    std::mutex mu;
    UpdateInfo info;
    std::thread worker;   // moved out + joined by stop_update_check()

    // Defensive: an exception escaping run_gui after start (e.g. the JobManager
    // ctor throwing) reaches static destruction with a joinable worker, which
    // would std::terminate and turn a clean error exit into a crash. The
    // worker is bounded (~3 s by its own timeouts), so joining here is safe.
    ~UpdateState() {
        if (worker.joinable()) worker.join();
    }
};
UpdateState& state() { static UpdateState s; return s; }

#ifdef WMR_UPDATE_CHECK
constexpr long long kIntervalS = 86400;  // same 24 h budget as the CLI

long long now_epoch_s() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// Publish a result under the lock. Never throws.
void resolve(const std::string& latest) {
    auto& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    s.info.known = true;
    s.info.current = APP_VERSION;
    s.info.latest = latest;
    s.info.newer = !latest.empty() && wmr::compare_versions(APP_VERSION, latest) < 0;
}
#endif

}  // namespace

void start_update_check_async() {
#ifdef WMR_UPDATE_CHECK
    // The same env opt-outs as the CLI's should_show gate (the gui
    // subcommand has no --no-update-check flag; the env vars are the GUI's
    // opt-out). is_tty=true: the notice goes to the page, not stderr, so the
    // CLI's TTY requirement does not apply.
    const bool enabled = wmr::should_show(
        /*no_update_flag=*/false,
        /*env_no_update=*/std::getenv("WMR_NO_UPDATE_CHECK") != nullptr,
        /*env_ci=*/std::getenv("CI") != nullptr,
        /*env_do_not_track=*/std::getenv("DO_NOT_TRACK") != nullptr,
        /*is_tty=*/true);
    if (!enabled) return;

    auto& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    if (s.info.enabled) return;  // one check per process
    s.info.enabled = true;

    const std::filesystem::path cache_path = wmr::user_cache_dir() / "update-check.json";
    const wmr::CacheData cd = wmr::read_cache(cache_path);
    if (!cd.latest_version.empty() &&
        !wmr::should_fetch(now_epoch_s() - cd.last_check_epoch, kIntervalS)) {
        // Fresh cache (shared with the CLI): no network at all.
        s.info.known = true;
        s.info.current = APP_VERSION;
        s.info.latest = cd.latest_version;
        s.info.newer = wmr::compare_versions(APP_VERSION, cd.latest_version) < 0;
        return;
    }

    s.worker = std::thread([cache_path, cd] {
        const wmr::FetchResult r = wmr::fetch_latest_release(cd.etag);
        std::string latest;
        if (r.ok && (r.http_code == 304 || r.body.empty())) {
            latest = cd.latest_version;  // 304 Not Modified: the cache stands
        } else if (r.ok) {
            if (auto tag = wmr::parse_release_json(r.body))
                latest = wmr::parse_tag(*tag);
        }
        if (!latest.empty()) {
            wmr::CacheData out = cd;
            out.latest_version = latest;
            out.etag = r.etag.empty() ? cd.etag : r.etag;
            out.last_check_epoch = now_epoch_s();
            wmr::write_cache(cache_path, out);
            resolve(latest);
        } else if (!cd.latest_version.empty()) {
            resolve(cd.latest_version);  // fetch failed: the cached answer
        }
        // else: nothing known; stays known=false (no notice shown).
    });
#endif  // WMR_UPDATE_CHECK
}

UpdateInfo update_info() {
    auto& s = state();
    std::lock_guard<std::mutex> lk(s.mu);
    return s.info;
}

void stop_update_check() {
    std::thread t;
    {
        auto& s = state();
        std::lock_guard<std::mutex> lk(s.mu);
        t = std::move(s.worker);
    }
    if (t.joinable()) t.join();
}

}  // namespace wmr::gui
