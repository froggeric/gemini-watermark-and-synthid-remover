// Copyright 2026 wmr contributors
// Licensed under the Apache License, Version 2.0 (see LICENSE in repo root).
//
// Unit tests for the CoreML execution-cache manager (src/core/coreml_cache.cpp).
// Covers the PURE decision logic (model_key composition, staleness, size guard),
// the early-exit size walk, the sidecar + clear flow, and the manual clear.
// The macos_version() helper is asserted to be non-empty on mac (Darwin).
#if defined(WMR_BUILD_AI_COREML_SD) && defined(__APPLE__)

#include <catch2/catch_test_macros.hpp>

#include "core/coreml_cache.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace fs = std::filesystem;
using namespace wmr;

namespace {
std::string read_text(const fs::path& p) {
    std::ifstream f(p);
    if (!f) return std::string{};
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void write_bytes(const fs::path& p, size_t n) {
    std::ofstream f(p, std::ios::binary);
    std::string s(n, 'x');
    f.write(s.data(), static_cast<std::streamsize>(s.size()));
}
fs::path unique_tmp(const char* tag) {
    return fs::temp_directory_path() /
           (std::string("wmr_coreml_cache_") + tag + "_" +
            std::to_string(static_cast<unsigned long long>(__LINE__)) + "_" +
            std::to_string(reinterpret_cast<std::uintptr_t>(&tag)));
}
}  // namespace

// ---- Pure: model_key composition. ----

TEST_CASE("model_key composition is deterministic and pipe-delimited", "[coreml-cache]") {
    REQUIRE(compose_model_key("1.16.7", "u", "ve", "vd", "25.5") ==
            "1.16.7|u|ve|vd|25.5");
}

TEST_CASE("model_key changes when any input changes", "[coreml-cache]") {
    const std::string base = compose_model_key("1.16.7", "u", "ve", "vd", "25.5");
    REQUIRE(compose_model_key("1.16.8", "u",  "ve",  "vd",  "25.5") != base);  // wmr upgrade
    REQUIRE(compose_model_key("1.16.7", "U",  "ve",  "vd",  "25.5") != base);  // unet re-pin
    REQUIRE(compose_model_key("1.16.7", "u",  "VE",  "vd",  "25.5") != base);  // vae-enc re-pin
    REQUIRE(compose_model_key("1.16.7", "u",  "ve",  "VD",  "25.5") != base);  // vae-dec re-pin
    REQUIRE(compose_model_key("1.16.7", "u",  "ve",  "vd",  "26.0") != base);  // macOS upgrade
}

TEST_CASE("model_key composition is stable across calls", "[coreml-cache]") {
    const auto a = compose_model_key("1.16.7", "abc", "def", "ghi", "15.5");
    const auto b = compose_model_key("1.16.7", "abc", "def", "ghi", "15.5");
    REQUIRE(a == b);
}

// ---- Pure: staleness decision. ----

TEST_CASE("should_clear_for_staleness: absent sidecar -> clear", "[coreml-cache]") {
    REQUIRE(should_clear_for_staleness("", "any-key"));
}

TEST_CASE("should_clear_for_staleness: matching keys -> no clear", "[coreml-cache]") {
    REQUIRE_FALSE(should_clear_for_staleness("k", "k"));
    REQUIRE_FALSE(should_clear_for_staleness("abc|def|ghi", "abc|def|ghi"));
}

TEST_CASE("should_clear_for_staleness: differing keys -> clear", "[coreml-cache]") {
    REQUIRE(should_clear_for_staleness("k1", "k2"));
    REQUIRE(should_clear_for_staleness("abc|def|ghi", "abc|DEF|ghi"));
}

// ---- Pure: size-guard decision. ----

TEST_CASE("should_clear_for_size: below or equal threshold -> no clear", "[coreml-cache]") {
    REQUIRE_FALSE(should_clear_for_size(0, 1024));
    REQUIRE_FALSE(should_clear_for_size(1023, 1024));
    REQUIRE_FALSE(should_clear_for_size(1024, 1024));  // equal is NOT over
}

TEST_CASE("should_clear_for_size: strictly above threshold -> clear", "[coreml-cache]") {
    REQUIRE(should_clear_for_size(1025, 1024));
    REQUIRE(should_clear_for_size(1ULL << 40, 1ULL << 30));
}

// ---- directory_size_capped: early-exit walk. ----

TEST_CASE("directory_size_capped: missing dir -> 0", "[coreml-cache]") {
    fs::path tmp = unique_tmp("missing");
    fs::remove_all(tmp);
    REQUIRE(directory_size_capped(tmp, 1000) == 0);
}

TEST_CASE("directory_size_capped: cap above total returns full total", "[coreml-cache]") {
    fs::path tmp = unique_tmp("full");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    write_bytes(tmp / "a.bin", 100);
    write_bytes(tmp / "b.bin", 100);
    write_bytes(tmp / "c.bin", 100);
    REQUIRE(directory_size_capped(tmp, 1000) == 300);
    fs::remove_all(tmp);
}

TEST_CASE("directory_size_capped: cap below total early-exits at first overflow", "[coreml-cache]") {
    fs::path tmp = unique_tmp("early");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    write_bytes(tmp / "a.bin", 100);
    write_bytes(tmp / "b.bin", 100);
    write_bytes(tmp / "c.bin", 100);
    // cap=150: a (100) <= 150, b pushes to 200 > 150 -> early-exit at >= 200, < 400.
    const uintmax_t s = directory_size_capped(tmp, 150);
    REQUIRE(s > 150);
    REQUIRE(s < 400);
    fs::remove_all(tmp);
}

// ---- manage_coreml_execution_cache: end-to-end sidecar + clear flow. ----

TEST_CASE("manage: first run with no sidecar clears (no-op) + writes sidecar", "[coreml-cache]") {
    fs::path tmp = unique_tmp("first");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    fs::path sidecar = tmp / "com.apple.e5rt.e5bundlecache.wmr_meta";

    manage_coreml_execution_cache(tmp, "key-1");
    REQUIRE(fs::exists(sidecar));
    REQUIRE(read_text(sidecar) == "key-1");
    fs::remove_all(tmp);
}

TEST_CASE("manage: same key + small cache -> no clear, sidecar unchanged", "[coreml-cache]") {
    fs::path tmp = unique_tmp("same");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    fs::path e5 = tmp / "com.apple.e5rt.e5bundlecache";
    fs::path sidecar = tmp / "com.apple.e5rt.e5bundlecache.wmr_meta";

    // First run seeds the sidecar.
    manage_coreml_execution_cache(tmp, "key-1");
    REQUIRE(read_text(sidecar) == "key-1");
    // Simulate a small compiled-graph file CoreML would write.
    fs::create_directories(e5);
    write_bytes(e5 / "compile.bin", 50);
    // Second run, same key, small size: MUST NOT clear.
    manage_coreml_execution_cache(tmp, "key-1");
    REQUIRE(read_text(sidecar) == "key-1");
    REQUIRE(fs::exists(e5 / "compile.bin"));  // untouched
    fs::remove_all(tmp);
}

TEST_CASE("manage: stale key -> clears e5bundlecache + rewrites sidecar", "[coreml-cache]") {
    fs::path tmp = unique_tmp("stale");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    fs::path e5 = tmp / "com.apple.e5rt.e5bundlecache";
    fs::path sidecar = tmp / "com.apple.e5rt.e5bundlecache.wmr_meta";

    manage_coreml_execution_cache(tmp, "key-1");
    fs::create_directories(e5);
    write_bytes(e5 / "compile.bin", 50);
    REQUIRE(fs::exists(e5 / "compile.bin"));

    manage_coreml_execution_cache(tmp, "key-2");  // stale
    REQUIRE(read_text(sidecar) == "key-2");
    REQUIRE(!fs::exists(e5));  // cleared
    fs::remove_all(tmp);
}

TEST_CASE("manage: missing sidecar with existing e5bundlecache -> clears (treated as stale)",
          "[coreml-cache]") {
    fs::path tmp = unique_tmp("nosidecar");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    fs::path e5 = tmp / "com.apple.e5rt.e5bundlecache";
    fs::path sidecar = tmp / "com.apple.e5rt.e5bundlecache.wmr_meta";

    fs::create_directories(e5);
    write_bytes(e5 / "compile.bin", 50);
    REQUIRE(!fs::exists(sidecar));

    manage_coreml_execution_cache(tmp, "key-1");
    REQUIRE(read_text(sidecar) == "key-1");
    REQUIRE(!fs::exists(e5));  // cleared (absent sidecar => stale)
    fs::remove_all(tmp);
}

// ---- clear_coreml_execution_cache (the manual subcommand path). ----

TEST_CASE("clear_coreml_execution_cache: absent dir -> success (no-op)", "[coreml-cache]") {
    fs::path tmp = unique_tmp("clear_absent");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    REQUIRE(clear_coreml_execution_cache(tmp));
    REQUIRE(!fs::exists(tmp / "com.apple.e5rt.e5bundlecache"));
    fs::remove_all(tmp);
}

TEST_CASE("clear_coreml_execution_cache: present -> removed (recursive)", "[coreml-cache]") {
    fs::path tmp = unique_tmp("clear_present");
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    fs::path e5 = tmp / "com.apple.e5rt.e5bundlecache";
    fs::create_directories(e5 / "sub" / "dir");
    write_bytes(e5 / "compile.bin", 50);
    write_bytes(e5 / "sub" / "deep.bin", 50);
    REQUIRE(clear_coreml_execution_cache(tmp));
    REQUIRE(!fs::exists(e5));
    fs::remove_all(tmp);
}

// ---- macos_version sanity. ----

TEST_CASE("macos_version returns a non-empty major.minor string", "[coreml-cache]") {
    const std::string v = macos_version();
    REQUIRE(!v.empty());
    REQUIRE(v.find('.') != std::string::npos);
}

// ---- coreml_app_cache_dir: the directory CoreML writes its compile cache under. ----

TEST_CASE("coreml_app_cache_dir: ends with Library/Caches/wmr when HOME is set",
          "[coreml-cache]") {
    const char* home = std::getenv("HOME");
    if (!home || home[0] == '\0') {
        WARN("HOME unset; skipping coreml_app_cache_dir path-shape test");
        return;
    }
    fs::path got = coreml_app_cache_dir();
    REQUIRE(!got.empty());
    // Must end with Library/Caches/wmr (the macOS-conventional app-scoped Caches
    // location CoreML actually writes to; NOT ~/.cache/wmr/).
    REQUIRE(got.filename() == "wmr");
    REQUIRE(got.parent_path().filename() == "Caches");
    REQUIRE(got.parent_path().parent_path().filename() == "Library");
}

TEST_CASE("coreml_app_cache_dir: empty path when HOME unset", "[coreml-cache]") {
    // Save + clear HOME, restore on scope exit (RAII).
    const char* saved = std::getenv("HOME");
    struct EnvGuard {
        const char* s;
        ~EnvGuard() {
            if (s) ::setenv("HOME", s, 1);
            else   ::unsetenv("HOME");
        }
    } guard{saved};
    ::unsetenv("HOME");
    REQUIRE(coreml_app_cache_dir().empty());
}

TEST_CASE("manage: empty cache dir is a graceful no-op (HOME-unset guard)",
          "[coreml-cache]") {
    // Empty path MUST never touch the filesystem (especially not relative paths
    // resolved against cwd). Verify no sidecar is created in cwd.
    fs::path cwd_before = fs::current_path();
    manage_coreml_execution_cache(fs::path{}, "any-key");
    REQUIRE(!fs::exists(cwd_before / "com.apple.e5rt.e5bundlecache.wmr_meta"));
    REQUIRE(!fs::exists(cwd_before / "com.apple.e5rt.e5bundlecache"));
}

#endif  // WMR_BUILD_AI_COREML_SD && __APPLE__
