#ifdef WMR_BUILD_REGEN
#include "core/model_downloader.hpp"

#include <openssl/sha.h>  // SHA256_* (acceptable per correction #3; deprecated in OSSL 3, no -Werror)
#include <spdlog/spdlog.h>
#include <curl/curl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
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
    DownloadProgressFn progress;
    uint64_t written = 0;
};

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

// Mark server_partial on the first 206 status line; capturing total via
// Content-Range is not required for correctness (we verify the full hash at the end).
size_t curl_header_cb(char* buf, size_t sz, size_t n, void* ud) {
    DlState* st = static_cast<DlState*>(ud);
    std::string line(buf, static_cast<size_t>(sz * n));
    if (line.substr(0, 7) == "HTTP/1." && line.size() >= 12) {
        int code = std::atoi(line.c_str() + 9);
        if (code == 206) st->server_partial = true;
    }
    return sz * n;
}

int curl_xfer_cb(void* ud, curl_off_t dltotal, curl_off_t dlnow,
                 curl_off_t /*ultotal*/, curl_off_t /*ulnow*/) {
    DlState* st = static_cast<DlState*>(ud);
    if (st->progress) {
        uint64_t total = dltotal > 0 ? static_cast<uint64_t>(dltotal) : 0;
        // add the resumed offset so the user sees absolute progress on a resumed fetch
        uint64_t abs_done = static_cast<uint64_t>(dlnow) + (st->server_partial ? st->have : 0);
        if (!st->progress(abs_done, total)) return 1;  // non-zero -> abort
    }
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

}  // namespace wmr
#endif
