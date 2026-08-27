#ifdef WMR_BUILD_REGEN
#include "core/model_downloader.hpp"

#include <openssl/sha.h>  // SHA256_* (acceptable per correction #3; deprecated in OSSL 3, no -Werror)
#include <spdlog/spdlog.h>
#include <curl/curl.h>
#include "cli/progress.hpp"
#include "core/paths.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

namespace wmr {

bool verify_sha256(const fs::path& file, const std::string& expected) {
    if (expected.empty()) return true;  // verify-only; the downloader gates this separately
    if (!fs::exists(file)) return false;
    SHA256_CTX c;
    SHA256_Init(&c);
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    std::vector<char> buf(1 << 20);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        if (f.gcount() > 0) {
            SHA256_Update(&c, buf.data(), static_cast<size_t>(f.gcount()));
        }
    }
    unsigned char h[SHA256_DIGEST_LENGTH];
    SHA256_Final(h, &c);
    char s[65];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        std::snprintf(s + i * 2, 3, "%02x", h[i]);
    }
    s[SHA256_DIGEST_LENGTH * 2] = '\0';
    // Constant-time-ish compare vs a wrong-length pin (a 64-hex-char string).
    return expected.size() == SHA256_DIGEST_LENGTH * 2 && expected == std::string(s);
}

namespace {
struct DlState {
    std::ofstream out;
    fs::path part;
    uint64_t have = 0;           // bytes already in .part when this attempt started
    bool server_partial = false;  // HTTP 206 in the header callback
    bool truncated = false;       // truncate-once guard for the 200-on-Range case
    int cur_code = 0;             // status of the response whose headers are arriving
    bool redirecting = false;     // a 3xx+Location was seen; we are NOT on the file body yet
    DownloadProgressFn progress;
    uint64_t written = 0;
};

// Case-insensitive ASCII prefix compare (`prefix` must already be lowercase).
bool ieq_prefix(const std::string& s, const char* prefix) {
    const size_t n = std::strlen(prefix);
    if (s.size() < n) return false;
    for (size_t i = 0; i < n; ++i) {
        if (std::tolower(static_cast<unsigned char>(s[i])) != prefix[i]) return false;
    }
    return true;
}

// Decide append-vs-truncate from the response status: if the server ignored our Range
// request and returned 200 (full body), the .part must be truncated once before we
// append the (now full) payload. file:// always 200s -> truncates -> writes full.
size_t curl_write_cb(char* ptr, size_t sz, size_t n, void* ud) {
    DlState* st = static_cast<DlState*>(ud);
    size_t bytes = sz * n;
    if (!st->truncated && st->have > 0 && !st->server_partial) {
        st->out.close();
        st->out.open(st->part, std::ios::binary | std::ios::trunc);
        st->have = 0;
        st->written = 0;
        st->truncated = true;
    }
    if (!st->out.is_open()) return 0;  // fail the transfer
    st->out.write(ptr, static_cast<std::streamsize>(bytes));
    st->written += bytes;
    return bytes;
}

// Track per-response state: 206 detection and, critically, whether the response
// being received is a redirect that curl will follow (3xx + Location). The status
// prefix check is "HTTP/" so BOTH "HTTP/1.1 206" and "HTTP/2 206" match (HuggingFace
// serves HTTP/2; the old "HTTP/1."-only check never saw its status lines).
size_t curl_header_cb(char* buf, size_t sz, size_t n, void* ud) {
    DlState* st = static_cast<DlState*>(ud);
    std::string line(buf, static_cast<size_t>(sz * n));
    if (line.size() >= 5 && line.compare(0, 5, "HTTP/") == 0) {
        st->redirecting = false;  // a new response begins; its Location (if any) re-sets this
        size_t sp = line.find(' ');
        if (sp != std::string::npos && sp + 1 < line.size()) {
            int code = std::atoi(line.c_str() + sp + 1);
            st->cur_code = code;
            if (code == 206) st->server_partial = true;
        }
        return sz * n;
    }
    if (st->cur_code >= 300 && st->cur_code < 400 && ieq_prefix(line, "location:")) {
        st->redirecting = true;
    }
    return sz * n;
}

int curl_xfer_cb(void* ud, curl_off_t dltotal, curl_off_t dlnow,
                 curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    DlState* st = static_cast<DlState*>(ud);
    if (!st->progress) return 0;
    // libcurl fires this callback during redirect responses too: a HuggingFace fetch
    // is 302 -> CDN, and the ticks during the 302's OWN ~1 KB body report dltotal =
    // the redirect page size (observed 996/1000/1022 B), not the file. Forwarding
    // them latched a bogus total downstream ("6.46 GiB / 1022 B  678855410%").
    // Suppress everything until the final (non-redirect) response.
    if (st->redirecting) return 0;
    uint64_t total = dltotal > 0 ? static_cast<uint64_t>(dltotal) : 0;
    if (st->server_partial && total > 0) {
        // 206 resume: dltotal is the REMAINING bytes; report the absolute size.
        total += st->have;
    }
    // add the resumed offset so the user sees absolute progress on a resumed fetch
    uint64_t abs_done = static_cast<uint64_t>(dlnow) + (st->server_partial ? st->have : 0);
    if (!st->progress(abs_done, total)) return 1;  // non-zero -> abort
    return 0;
}
}  // namespace

DownloadResult download_pinned_file(const std::string& url, const fs::path& dest,
                                    const std::string& expected, bool allow_download,
                                    bool allow_empty_hash, DownloadProgressFn progress) {
    DownloadResult r;
    if (expected.empty() && !allow_empty_hash) {
        r.error = "refusing unpinned download (SHA256 pin empty); set the pin or pass allow_empty_hash";
        return r;
    }
    fs::create_directories(dest.parent_path().empty() ? fs::current_path() : dest.parent_path());
    if (fs::exists(dest) && verify_sha256(dest, expected)) {
        r.ok = true;
        r.path = dest.string();
        return r;
    }
    if (!allow_download) {
        r.error = "download disabled and file absent/unverified";
        return r;
    }
    fs::path part = dest;
    part += ".part";

    for (int attempt = 0; attempt < 2; ++attempt) {
        DlState st;
        st.part = part;
        st.progress = progress;
        st.have = fs::exists(part) ? fs::file_size(part) : 0;
        // Open append-if-resuming, truncate-if-fresh. The write cb also guards the
        // 200-on-Range case (truncates once, deterministically).
        std::ios::openmode om = std::ios::binary | (st.have > 0 ? std::ios::app : std::ios::trunc);
        st.out.open(part, om);
        if (!st.out) {
            r.error = "cannot open .part for write";
            fs::remove(part);
            return r;
        }

        CURL* curl = curl_easy_init();
        if (!curl) {
            r.error = "curl_easy_init failed";
            st.out.close();
            fs::remove(part);
            return r;
        }
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &st);
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, curl_header_cb);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &st);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, curl_xfer_cb);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &st);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "wmr-regen/1.0");
        if (st.have > 0) {
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(st.have));
        }
        // Defensive timeouts so a stalled handshake or a mid-transfer stall fails
        // fast instead of hanging forever (a 5 GB model fetch on a flaky CDN was
        // observed blocked for 24+ min in curl's multi_wait with no timeout). No
        // absolute CURLOPT_TIMEOUT: that would kill legitimate slow transfers of a
        // 5 GB file. CONNECTTIMEOUT bounds DNS+TCP+TLS; LOW_SPEED_* abort once the
        // average rate drops under ~1 KB/s for 60 s (a true stall — even a slow CDN
        // ramp-up stays well above this). The xferinfo cb only aborts on a
        // user-cancel, so it cannot detect a stall by itself.
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1024L);
        curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 60L);

        spdlog::info("regen: downloading {} (resume from {} bytes, attempt {})",
                     url, st.have, attempt + 1);
        CURLcode rc = curl_easy_perform(curl);
        long http = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http);
        curl_easy_cleanup(curl);
        st.out.close();
        // 416 = our resume offset >= content length -> the .part is already complete.
        if (rc != CURLE_OK && rc != CURLE_HTTP_RETURNED_ERROR) {
            r.error = std::string("curl: ") + curl_easy_strerror(rc);
            spdlog::warn("regen: download attempt {} failed (rc={}, http={}): {}",
                         attempt + 1, static_cast<int>(rc), http, r.error);
            // KEEP .part for resume on the next attempt; do NOT remove.
            continue;
        }
        if (http == 416) {
            /* treat as complete, fall through to verify */
        } else if (http != 0 && http != 200 && http != 206) {
            r.error = "http " + std::to_string(http);
            fs::remove(part);
            continue;
        }
        if (!verify_sha256(part, expected)) {
            spdlog::warn("regen: downloaded file failed SHA256 (attempt {}); restarting from 0",
                         attempt + 1);
            fs::remove(part);  // bad content -> drop the partial, restart clean next attempt
            continue;
        }
        std::error_code ec;
        fs::rename(part, dest, ec);
        if (ec) {
            r.error = "rename failed: " + ec.message();
            fs::remove(part);
            return r;
        }
        r.ok = true;
        r.path = dest.string();
        return r;
    }
    fs::remove(part);
    return r;
}

DownloadProgressFn make_byte_progress(const std::string& filename) {
    auto bp = std::make_shared<ByteProgress>("  " + filename, 0ull);
    return [bp](uint64_t done, uint64_t total) -> bool {
        // The total arrives lazily via curl's xferinfo once the FINAL response's
        // headers land (Content-Length / Content-Range); the downloader filters
        // redirect-page ticks, and set_total lets the last non-zero total win.
        if (total > 0) bp->set_total(total);
        return bp->update(done);
    };
}

LeftoverCpuModels find_leftover_cpu_models() {
    // Same filenames resolve_model/resolve_vae fall back to in the user cache,
    // plus their .part side files from an interrupted fetch.
    static const char* kNames[] = {
        "sd_xl_base_1.0.safetensors",
        "sd_xl_base_1.0.safetensors.part",
        "sdxl_vae-fp16-fix.safetensors",
        "sdxl_vae-fp16-fix.safetensors.part",
    };
    LeftoverCpuModels out;
    const fs::path dir = user_cache_dir();
    for (const char* name : kNames) {
        std::error_code ec;
        fs::path p = dir / name;
        if (fs::exists(p, ec)) {
            const uintmax_t sz = fs::file_size(p, ec);
            if (!ec) out.bytes += static_cast<uint64_t>(sz);
            out.paths.push_back(std::move(p));
        }
    }
    return out;
}

int remove_leftover_cpu_models(const LeftoverCpuModels& leftovers) {
    int removed = 0;
    for (const auto& p : leftovers.paths) {
        std::error_code ec;
        if (fs::remove(p, ec) && !ec) ++removed;
    }
    return removed;
}

}  // namespace wmr
#endif
