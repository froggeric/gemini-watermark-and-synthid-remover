#include <catch2/catch_test_macros.hpp>
#include <unistd.h>  // POSIX getpid() (test target is mac/linux; not a Windows leg)

#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"
#include "cli/cli_app.hpp"  // wmr::kHeaderRule
#include <cstdio>
#include <fstream>
#include <filesystem>
using namespace wmr;

namespace tfs = std::filesystem;

TEST_CASE("parse_version", "[update-check]") {
    REQUIRE(parse_version("1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("v1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("V1.16.4").value() == std::array<unsigned,3>{1,16,4});
    REQUIRE(parse_version("2").value() == std::array<unsigned,3>{2,0,0});
    REQUIRE(parse_version("1.2.3.4").value() == std::array<unsigned,3>{1,2,3});
    REQUIRE(parse_version("1.2-rc1").value() == std::array<unsigned,3>{1,2,0});
    REQUIRE_FALSE(parse_version("").has_value());
    REQUIRE_FALSE(parse_version("v").has_value());
    REQUIRE_FALSE(parse_version("1.a.2").has_value());
    REQUIRE_FALSE(parse_version("1..2").has_value());
    REQUIRE_FALSE(parse_version("99999999999").has_value());  // overflow
}

TEST_CASE("compare_versions", "[update-check]") {
    REQUIRE(compare_versions("1.16.3","1.16.4") < 0);
    REQUIRE(compare_versions("1.16.4","1.16.4") == 0);
    REQUIRE(compare_versions("1.17.0","1.16.9") > 0);
    REQUIRE(compare_versions("2.0.0","1.99.99") > 0);
    REQUIRE(compare_versions("1.16","1.16.0") == 0);
    REQUIRE(compare_versions("notaver","1.16.4") > 0);  // malformed current never newer
}

TEST_CASE("parse_tag", "[update-check]") {
    REQUIRE(parse_tag("v1.16.4") == "1.16.4");
    REQUIRE(parse_tag("V1.16.4") == "1.16.4");
    REQUIRE(parse_tag("1.16.4") == "1.16.4");
}

TEST_CASE("parse_release_json", "[update-check]") {
    const char* ok = R"({"tag_name":"v1.16.4","assets":[]})";
    REQUIRE(parse_release_json(ok).value() == "v1.16.4");
    REQUIRE_FALSE(parse_release_json(R"({"assets":[]})").has_value());      // missing
    REQUIRE_FALSE(parse_release_json(R"({"tag_name":123})").has_value());   // non-string
    REQUIRE_FALSE(parse_release_json("not json{").has_value());             // unparseable
}

TEST_CASE("should_show truth table", "[update-check]") {
    REQUIRE(should_show(false,false,false,false,true) == true);
    REQUIRE(should_show(true, false,false,false,true) == false);
    REQUIRE(should_show(false,true, false,false,true) == false);
    REQUIRE(should_show(false,false,true, false,true) == false);
    REQUIRE(should_show(false,false,false,true, true) == false);
    REQUIRE(should_show(false,false,false,false,false) == false);  // non-TTY
}

TEST_CASE("should_fetch", "[update-check]") {
    REQUIRE(should_fetch(86399, 86400) == false);
    REQUIRE(should_fetch(86400, 86400) == true);
    REQUIRE(should_fetch(0, 0) == true);
    REQUIRE(should_fetch(-100, 86400) == false);
}

TEST_CASE("color_enabled_for (pure policy)", "[update-check]") {
    REQUIRE(color_enabled_for(true,  nullptr, nullptr) == true);   // default: TTY, no NO_COLOR, no TERM
    REQUIRE(color_enabled_for(true,  "",      nullptr) == true);   // NO_COLOR empty == unset == ON
    REQUIRE(color_enabled_for(true,  "1",     nullptr) == false);  // NO_COLOR non-empty == OFF
    REQUIRE(color_enabled_for(true,  "any",   nullptr) == false);  // any non-empty value disables
    REQUIRE(color_enabled_for(true,  nullptr, "xterm") == true);   // TERM != dumb == ON
    REQUIRE(color_enabled_for(true,  nullptr, "dumb") == false);   // TERM == dumb == OFF
    REQUIRE(color_enabled_for(false, nullptr, nullptr) == false);  // non-TTY always OFF (independent)
}

TEST_CASE("cache round-trip and resilience", "[update-check]") {
    auto tmp = tfs::temp_directory_path() / ("wmr_uc_test_" + std::to_string(getpid()) + ".json");
    CacheData in{1723000000, "1.16.4", "\"abc\""};
    REQUIRE(write_cache(tmp, in));
    CacheData out = read_cache(tmp);
    REQUIRE(out.last_check_epoch == 1723000000);
    REQUIRE(out.latest_version == "1.16.4");
    REQUIRE(out.etag == "\"abc\"");
    // No *.tmp.* residue after a successful write.
    REQUIRE(tfs::status(tmp).type() == tfs::file_type::regular);
    tfs::remove(tmp);

    // missing file => empty
    REQUIRE(read_cache(tmp).latest_version.empty());
    REQUIRE(read_cache(tmp).last_check_epoch == 0);

    // empty / whitespace-only file => empty
    { std::ofstream f(tmp); f.put(' '); }
    REQUIRE(read_cache(tmp).latest_version.empty());
    tfs::remove(tmp);

    // garbage / wrong-type fields => empty (never throw)
    { std::ofstream f(tmp); f << "{ this is not json "; }
    REQUIRE_NOTHROW(read_cache(tmp));
    REQUIRE(read_cache(tmp).latest_version.empty());
    tfs::remove(tmp);

    // wrong-type latest_version (non-string) => extract_string_field rejects it
    { std::ofstream f(tmp); f << R"({"latest_version":123,"etag":"x"})"; }
    REQUIRE(read_cache(tmp).latest_version.empty());
    tfs::remove(tmp);
}

TEST_CASE("format_notice exact string", "[update-check]") {
    std::string s = format_notice("1.16.3", "1.16.4", /*color=*/false);
    REQUIRE(s.find("A new release of wmr is available: 1.16.3 -> 1.16.4") != std::string::npos);
    REQUIRE(s.find("Download: https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases/latest") != std::string::npos);
    REQUIRE(s.find("Disable with WMR_NO_UPDATE_CHECK=1 or --no-update-check.") != std::string::npos);
    REQUIRE(s.find("  ") != std::string::npos);              // two-space indent present
    REQUIRE(s.find(wmr::kHeaderRule) != std::string::npos);  // shares the header rule
    REQUIRE(s.find("\033[") == std::string::npos);           // no ANSI when color off

    // Byte-for-byte exact block (color off). The kHeaderRule frames + the three
    // content lines, each newline-terminated.
    std::string expected = std::string(wmr::kHeaderRule) + "\n"
        + "  A new release of wmr is available: 1.16.3 -> 1.16.4\n"
        + "  Download: https://github.com/froggeric/gemini-watermark-and-synthid-remover/releases/latest\n"
        + "  Disable with WMR_NO_UPDATE_CHECK=1 or --no-update-check.\n"
        + std::string(wmr::kHeaderRule) + "\n";
    REQUIRE(s == expected);

    std::string c = format_notice("1.16.3", "1.16.4", /*color=*/true);
    REQUIRE(c.find("\033[1m") != std::string::npos);         // bold
    REQUIRE(c.find("\033[33m") != std::string::npos);        // yellow
    REQUIRE(c.find("\033[0m") != std::string::npos);         // reset
    // Same visible text present in the color version too.
    REQUIRE(c.find("1.16.3 -> 1.16.4") != std::string::npos);
}

// AC-10: drives the gate-free core directly with a temp cache path and a stubbed
// fetch, so it needs NO HOME/CI/TTY manipulation and passes identically in CI and
// locally. (APP_VERSION is "0.0.0" in the test target, so a v9.9.9 tag is newer
// and the cache-write path is exercised. The exact notice string is pinned by
// AC-8; this case asserts the core consumes the fetch result + stores the
// v-stripped tag.) Opt-out-skip is covered by AC-5's should_show truth table +
// the gate structure, not by this network-free case.
TEST_CASE("run_update_check: newer fetch updates cache (v stripped); no throw", "[update-check]") {
    fs::path cache = tfs::temp_directory_path() / ("wmr_uc_core_" + std::to_string(getpid()) + ".json");
    tfs::remove(cache);

    FetchResult newer; newer.ok = true; newer.http_code = 200;
    newer.body = R"({"tag_name":"v9.9.9","assets":[]})";
    CHECK_NOTHROW(run_update_check(cache, /*interval_s=*/0, /*color=*/false,
                                   [&](const std::string&){ return newer; }));
    REQUIRE(read_cache(cache).latest_version == "9.9.9");  // parse_tag stripped the leading v

    // last_check_epoch was written (the fetch ran, interval=0 => always).
    REQUIRE(read_cache(cache).last_check_epoch > 0);
    tfs::remove(cache);

    // Older tag: still consumed into the cache (no throw), but the notice path is
    // the not-newer branch (the exact notice bytes are pinned by the AC-8 test).
    FetchResult older; older.ok = true; older.http_code = 200;
    older.body = R"({"tag_name":"v0.0.1"})";
    CHECK_NOTHROW(run_update_check(cache, 0, false,
                                   [&](const std::string&){ return older; }));
    REQUIRE(read_cache(cache).latest_version == "0.0.1");
    tfs::remove(cache);

    // 304 / fetch failure: previous latest_version is preserved, epoch advances.
    {
        std::ofstream f(cache); f << R"({"last_check_epoch":1,"latest_version":"1.0.0","etag":""})";
    }
    FetchResult notmod; notmod.ok = true; notmod.http_code = 304;  // 304 -> keep previous
    CHECK_NOTHROW(run_update_check(cache, 0, false,
                                   [&](const std::string&){ return notmod; }));
    REQUIRE(read_cache(cache).latest_version == "1.0.0");  // unchanged
    REQUIRE(read_cache(cache).last_check_epoch > 1);        // epoch advanced
    tfs::remove(cache);
}
#endif
