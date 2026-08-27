#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#include <openssl/sha.h>
#include "core/model_downloader.hpp"
namespace fs = std::filesystem;
using namespace wmr;

// Byte-deterministic fixture: a FIXED-BYTE buffer written directly to a file (NOT via
// cv::imwrite). A PNG fixture's bytes depend on the libpng version, so CI with a
// different libpng would fail the pinned-hash check below. A raw byte sequence is
// stable regardless of which image libs are linked. Pinned via shasum -a 256.
constexpr const char* kFixtureSha256 =
    "12f51875c1545afac5b3bd4b3fd5131a9ed9f50ff248b84b01986d0934716f68";
constexpr size_t kFixtureLen = 4096;

static fs::path make_fixture(const fs::path& dir, const std::string& name) {
    fs::create_directories(dir);
    // Deterministic byte sequence: byte[i] = (i*31 + 17) & 0xff. Pure function of i,
    // so the file (and its SHA256) is identical on every host and every build.
    std::vector<unsigned char> buf(kFixtureLen);
    for (size_t i = 0; i < buf.size(); ++i)
        buf[i] = static_cast<unsigned char>((i * 31 + 17) & 0xff);
    fs::path p = dir / name;
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    return p;
}

TEST_CASE("regen downloader", "[regen][downloader]") {
    fs::path tmp = fs::temp_directory_path() / ("wmr_regen_dl_" + std::to_string((long long)&tmp));
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    fs::path src = make_fixture(tmp, "src.bin");

    std::string url = std::string("file://") + src.string();

    SECTION("correct hash downloads + verifies") {
        // Real correctness gate for the OpenSSL hash wiring: the fixture is a fixed
        // byte buffer, so its SHA256 is a hardcoded constant. If verify_sha256's
        // OpenSSL wiring were broken (or the hash hand-rolled and buggy), this would fail.
        auto r0 = download_pinned_file(url, tmp / "out0.bin", kFixtureSha256,
                                       /*allow_download=*/true, /*allow_empty_hash=*/false);
        REQUIRE(r0.ok);
        REQUIRE(fs::exists(r0.path));
        REQUIRE(verify_sha256(r0.path, kFixtureSha256));          // match -> true
        REQUIRE(verify_sha256(r0.path, ""));                      // empty = skip -> true
        REQUIRE_FALSE(verify_sha256(r0.path, "0"));               // mismatch -> false
    }

    SECTION("wrong hash is rejected and cleaned") {
        auto r = download_pinned_file(url, tmp / "out_bad.bin",
                                      "0000000000000000000000000000000000000000000000000000000000000000",
                                      /*allow_download=*/true);
        REQUIRE_FALSE(r.ok);
        REQUIRE_FALSE(fs::exists(tmp / "out_bad.bin"));
        REQUIRE_FALSE(fs::exists(tmp / "out_bad.bin.part"));
    }

    SECTION("empty hash WITHOUT allow_empty_hash is refused (model-pin safety)") {
        auto r = download_pinned_file(url, tmp / "out_nopin.bin", "",
                                      /*allow_download=*/true, /*allow_empty_hash=*/false);
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.error.find("unpinned") != std::string::npos);
    }

    SECTION("no download allowed + absent file -> ok=false") {
        auto r = download_pinned_file(url, tmp / "out_nodl.bin", "",
                                      /*allow_download=*/false, /*allow_empty_hash=*/true);
        REQUIRE_FALSE(r.ok);
    }

    SECTION("resume: a stale .part is handled, final file verifies") {
        // Pre-seed a .part with garbage bytes (simulating a partial/interrupted fetch).
        // file:// ignores Range -> the write callback detects 200-on-Range and truncates
        // once, so the final file is the full correct body regardless of the stale bytes.
        fs::path stale = tmp / "out_res.bin.part";
        {
            std::ofstream f(stale, std::ios::binary);
            f << "STALE_PARTIAL_BYTES";
        }
        auto r = download_pinned_file(url, tmp / "out_res.bin", kFixtureSha256,
                                      /*allow_download=*/true);
        REQUIRE(r.ok);
        REQUIRE(fs::exists(r.path));
        REQUIRE_FALSE(fs::exists(stale));  // renamed away on success
        // The final file must equal the source byte-for-byte (truncated, not appended-to).
        std::ifstream a(src, std::ios::binary), b(r.path, std::ios::binary);
        REQUIRE(a.good());
        REQUIRE(b.good());
        std::vector<char> va((std::istreambuf_iterator<char>(a)), {});
        std::vector<char> vb((std::istreambuf_iterator<char>(b)), {});
        REQUIRE(va == vb);
    }

    SECTION("progress callback receives milestones and can abort") {
        // Happy path: the progress fn IS invoked with (downloaded, total). For a
        // file:// fetch curl still drives the xferinfo callback at least once.
        int call_count = 0;
        uint64_t last_dl = 0, last_total = 0;
        auto r = download_pinned_file(url, tmp / "out_prog.bin", kFixtureSha256,
                                      /*allow_download=*/true, /*allow_empty_hash=*/false,
                                      [&](uint64_t dl, uint64_t total) {
                                          ++call_count;
                                          last_dl = dl;
                                          last_total = total;
                                          return true;
                                      });
        REQUIRE(r.ok);
        REQUIRE(call_count > 0);
        // Total is reported (fixture is small but non-zero); downloaded reaches it.
        REQUIRE(last_total > 0);
        REQUIRE(last_dl == last_total);
    }

    SECTION("progress callback returning false aborts the transfer") {
        auto r = download_pinned_file(url, tmp / "out_abort.bin", kFixtureSha256,
                                      /*allow_download=*/true, /*allow_empty_hash=*/false,
                                      [](uint64_t, uint64_t) { return false; });
        REQUIRE_FALSE(r.ok);
        // A user-initiated abort is not a server/format failure: surface a clear error.
        REQUIRE_FALSE(r.error.empty());
    }

    fs::remove_all(tmp);
}

// Regression (seen live on a fresh mac, v1.16.10): a real HuggingFace fetch is a
// 302 redirect to a CDN. libcurl fires the xferinfo callback during the 302's OWN
// ~1 KB body too, so the downloader receives dltotal = the redirect page size
// (observed 996/1000/1022 B, varying with the filename) BEFORE the final 200's
// real Content-Length. ByteProgress.set_total latched that first bogus value and
// ignored the real one: "6.46 GiB / 1022 B  678855410%". The downloader must
// only report progress once the FINAL response's body is flowing.
#if !defined(_WIN32)
TEST_CASE("regen downloader redirect total", "[regen][downloader]") {
    // Same deterministic body as the python server: byte[i] = (i*7+3) & 0xff.
    constexpr size_t kBodyLen = 262144;
    constexpr int kPort = 8461;

    if (std::system("python3 --version >/dev/null 2>&1") != 0) {
        SKIP("python3 not available; skipping redirect-server test");
    }

    fs::path tmp = fs::temp_directory_path() / "wmr_regen_redir";
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    // Minimal redirect server: /redir -> 302 with a ~1 KB body -> /file -> 200
    // with the 256 KiB body. The 302 body size mimics HuggingFace's redirect page.
    fs::path script = tmp / "wmr_redir_server.py";
    {
        std::ofstream f(script);
        f << R"PY(
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
BODY = bytes((i*7+3)&0xff for i in range(262144))
REDIR = b"x" * 1022
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def do_GET(self):
        if self.path.startswith("/redir"):
            self.send_response(302)
            self.send_header("Location", "/file")
            self.send_header("Content-Length", str(len(REDIR)))
            self.end_headers()
            self.wfile.write(REDIR)
        elif self.path.startswith("/file"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(BODY)))
            self.end_headers()
            self.wfile.write(BODY)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
    def log_message(self, *a):
        pass
srv = HTTPServer(("127.0.0.1", int(sys.argv[1])), H)
open(sys.argv[2], "w").write("ok")
srv.serve_forever()
)PY";
    }
    fs::path ready = tmp / "ready";
    std::system(("python3 " + script.string() + " " + std::to_string(kPort) +
                 " " + ready.string() + " >/dev/null 2>&1 &")
                    .c_str());
    bool up = false;
    for (int i = 0; i < 100 && !up; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        up = fs::exists(ready);
    }
    if (!up) {
        std::system("pkill -f wmr_redir_server.py >/dev/null 2>&1");
        SKIP("local redirect server did not start (port busy?)");
    }

    // SHA256 of the deterministic body, computed the same way the downloader does.
    std::vector<unsigned char> body(kBodyLen);
    for (size_t i = 0; i < body.size(); ++i)
        body[i] = static_cast<unsigned char>((i * 7 + 3) & 0xff);
    unsigned char h[SHA256_DIGEST_LENGTH];
    SHA256(reinterpret_cast<const unsigned char*>(body.data()), body.size(), h);
    char sha[65];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i)
        std::snprintf(sha + i * 2, 3, "%02x", h[i]);

    uint64_t min_total = UINT64_MAX, max_total = 0;
    int calls = 0;
    auto r = download_pinned_file("http://127.0.0.1:" + std::to_string(kPort) + "/redir",
                                  tmp / "out.bin", sha,
                                  /*allow_download=*/true, /*allow_empty_hash=*/false,
                                  [&](uint64_t, uint64_t total) {
                                      ++calls;
                                      if (total > 0) {
                                          min_total = std::min(min_total, total);
                                          max_total = std::max(max_total, total);
                                      }
                                      return true;
                                  });
    std::system("pkill -f wmr_redir_server.py >/dev/null 2>&1");

    REQUIRE(r.ok);
    REQUIRE(calls > 0);
    // Every non-zero total must be the FINAL response's body size, never the
    // 302 redirect page's ~1 KB Content-Length.
    REQUIRE(min_total == kBodyLen);
    REQUIRE(max_total == kBodyLen);

    fs::remove_all(tmp);
}
#endif  // !defined(_WIN32)
