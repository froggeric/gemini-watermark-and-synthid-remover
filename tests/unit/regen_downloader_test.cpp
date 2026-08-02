#include <catch2/catch_test_macros.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>  // cv::imwrite (correction #2; core.hpp is not enough)
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include "core/model_downloader.hpp"
namespace fs = std::filesystem;
using namespace wmr;

// Byte-deterministic fixture: same 64x64 Vec3b Mat + libpng default compression ->
// same SHA256 across runs/hosts (correction #1). Pinned once via openssl dgst.
constexpr const char* kFixtureSha256 =
    "1924111b77ee6b7e299bb3d043a311c99825e41d47a6dc178009145ddc76e337";

static fs::path make_fixture(const fs::path& dir, const std::string& name) {
    fs::create_directories(dir);
    cv::Mat m = cv::Mat_<cv::Vec3b>(64, 64, cv::Vec3b(10, 20, 30));
    for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
            m.at<cv::Vec3b>(y, x) = cv::Vec3b(x, y, (x + y) & 255);
    fs::path p = dir / name;
    cv::imwrite(p.string(), m);
    return p;
}

TEST_CASE("regen downloader", "[regen][downloader]") {
    fs::path tmp = fs::temp_directory_path() / ("wmr_regen_dl_" + std::to_string((long long)&tmp));
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    fs::path src = make_fixture(tmp, "src.png");

    std::string url = std::string("file://") + src.string();

    SECTION("correct hash downloads + verifies") {
        // Real correctness gate for the OpenSSL hash wiring (correction #1): the
        // fixture PNG is byte-stable, so its SHA256 is a hardcoded constant. If
        // verify_sha256's OpenSSL wiring were broken (or the hash hand-rolled and
        // buggy), this assertion would fail.
        auto r0 = download_pinned_file(url, tmp / "out0.png", kFixtureSha256,
                                       /*allow_download=*/true, /*allow_empty_hash=*/false);
        REQUIRE(r0.ok);
        REQUIRE(fs::exists(r0.path));
        REQUIRE(verify_sha256(r0.path, kFixtureSha256));          // match -> true
        REQUIRE(verify_sha256(r0.path, ""));                      // empty = skip -> true
        REQUIRE_FALSE(verify_sha256(r0.path, "0"));               // mismatch -> false
    }

    SECTION("wrong hash is rejected and cleaned") {
        auto r = download_pinned_file(url, tmp / "out_bad.png",
                                      "0000000000000000000000000000000000000000000000000000000000000000",
                                      /*allow_download=*/true);
        REQUIRE_FALSE(r.ok);
        REQUIRE_FALSE(fs::exists(tmp / "out_bad.png"));
        REQUIRE_FALSE(fs::exists(tmp / "out_bad.png.part"));
    }

    SECTION("empty hash WITHOUT allow_empty_hash is refused (model-pin safety)") {
        auto r = download_pinned_file(url, tmp / "out_nopin.png", "",
                                      /*allow_download=*/true, /*allow_empty_hash=*/false);
        REQUIRE_FALSE(r.ok);
        REQUIRE(r.error.find("unpinned") != std::string::npos);
    }

    SECTION("no download allowed + absent file -> ok=false") {
        auto r = download_pinned_file(url, tmp / "out_nodl.png", "",
                                      /*allow_download=*/false, /*allow_empty_hash=*/true);
        REQUIRE_FALSE(r.ok);
    }

    SECTION("resume: a stale .part is handled, final file verifies") {
        // Pre-seed a .part with garbage bytes (simulating a partial/interrupted fetch).
        // file:// ignores Range -> the write callback detects 200-on-Range and truncates
        // once, so the final file is the full correct body regardless of the stale bytes.
        fs::path stale = tmp / "out_res.png.part";
        {
            std::ofstream f(stale, std::ios::binary);
            f << "STALE_PARTIAL_BYTES";
        }
        auto r = download_pinned_file(url, tmp / "out_res.png", kFixtureSha256,
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
        auto r = download_pinned_file(url, tmp / "out_prog.png", kFixtureSha256,
                                      /*allow_download=*/true, /*allow_empty_hash=*/false,
                                      [&](uint64_t dl, uint64_t total) {
                                          ++call_count;
                                          last_dl = dl;
                                          last_total = total;
                                          return true;
                                      });
        REQUIRE(r.ok);
        REQUIRE(call_count > 0);
        // Total is reported (libpng file is small but non-zero); downloaded reaches it.
        REQUIRE(last_total > 0);
        REQUIRE(last_dl == last_total);
    }

    SECTION("progress callback returning false aborts the transfer") {
        auto r = download_pinned_file(url, tmp / "out_abort.png", kFixtureSha256,
                                      /*allow_download=*/true, /*allow_empty_hash=*/false,
                                      [](uint64_t, uint64_t) { return false; });
        REQUIRE_FALSE(r.ok);
        // A user-initiated abort is not a server/format failure: surface a clear error.
        REQUIRE_FALSE(r.error.empty());
    }

    fs::remove_all(tmp);
}
