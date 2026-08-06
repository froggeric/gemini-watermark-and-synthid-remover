// Copyright 2025 wmr contributors
// Licensed under the Apache License, Version 2.0 (see LICENSE in repo root).
#ifdef WMR_BUILD_REGEN
#ifdef WMR_BUILD_AI_COREML_SD

#include <catch2/catch_test_macros.hpp>

#include "core/coreml_sd_model_fetch.hpp"
#include "core/model_downloader.hpp"

#include <openssl/sha.h>  // SHA256_* for the in-process fixture hasher below

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace wmr;

// SHA256 of a file computed IN-PROCESS via OpenSSL (the SAME backing the production
// verify_sha256 uses), NOT a `shasum` shellout. A shellout would be a portability
// hazard: `shasum` is a perl script absent on some Linux images (coreutils ships
// `sha256sum`), and it would make this mac-only test's result host-dependent the
// day it runs anywhere but the dev Mac. The test exe already links OpenSSL::Crypto
// (the regen block links it for model_downloader.cpp).
static std::string sha256_of(const fs::path& p) {
    SHA256_CTX c;
    SHA256_Init(&c);
    std::ifstream f(p, std::ios::binary);
    std::vector<char> buf(1 << 16);
    while (f) {
        f.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        if (f.gcount() > 0) SHA256_Update(&c, buf.data(), static_cast<size_t>(f.gcount()));
    }
    unsigned char h[SHA256_DIGEST_LENGTH];
    SHA256_Final(h, &c);
    char s[65];
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) std::snprintf(s + i * 2, 3, "%02x", h[i]);
    s[SHA256_DIGEST_LENGTH * 2] = '\0';
    return s;
}

// Build a tiny .tar.gz at `archive` whose TOP-LEVEL entry is the DIRECTORY `top_dir`
// holding `entry` with `content`. The top-level dir mirrors how the real archives
// ship: coreml-sdxl-unet.mlpackage.tar.gz extracts to a Stable_Diffusion_*.mlpackage/
// DIRECTORY, so CoreMLModelFile.mlpackage_dir names that extracted directory and the
// post-extract assertions (models/<mlpackage_dir>/<entry>) resolve. Tarring a bare
// file (an earlier draft) extracts <entry> straight into models_dir and the
// mlpackage_dir-scoped assertions never pass. Uses system tar; the production
// extract_tar_gz also shells out to tar, so this is consistent and extracts cleanly
// under both BSD tar (mac) and GNU tar (linux).
static std::string make_targz_fixture(const fs::path& staging, const fs::path& archive,
                                      const std::string& top_dir, const std::string& entry,
                                      const std::string& content) {
    fs::create_directories(staging / top_dir);
    { std::ofstream f(staging / top_dir / entry, std::ios::binary); f << content; }
    std::string cmd = "tar -czf \"" + archive.string() + "\" -C \"" +
                      staging.string() + "\" " + top_dir;
    REQUIRE(::system(cmd.c_str()) == 0);
    return sha256_of(archive);
}

static std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), {});
}

TEST_CASE("coreml model cache: re-pin re-downloads stale archive", "[coreml-sd][cache]") {
    fs::path tmp = fs::temp_directory_path() / ("wmr_cache_test_" + std::to_string((long long)&tmp));
    fs::remove_all(tmp);
    fs::create_directories(tmp);

    fs::path src = tmp / "src";
    fs::create_directories(src);
    fs::path correct_archive = src / "foo.tar.gz";
    std::string correct_sha =
        make_targz_fixture(tmp / "stage", correct_archive, "foo_extracted", "hello.txt", "correct-content");
    std::string base_url = "file://" + src.string();

    // A single archive artifact named foo.tar.gz that extracts to foo_extracted/.
    auto file_for = [&](const std::string& sha) {
        CoreMLModelFile f;
        f.filename = "foo.tar.gz";
        f.sha256 = sha.c_str();
        f.is_archive = true;
        f.mlpackage_dir = "foo_extracted";
        return f;
    };

    SECTION("fresh download + extract") {
        fs::path models = tmp / "m1";
        REQUIRE(ensure_coreml_model_files(models, {file_for(correct_sha)}, true, base_url));
        REQUIRE(fs::exists(models / "foo.tar.gz"));
        REQUIRE(fs::exists(models / "foo_extracted" / "hello.txt"));
        REQUIRE(read_file(models / "foo_extracted" / "hello.txt") == "correct-content");
    }

    SECTION("warm cache, pin unchanged, works with downloads disabled (no network)") {
        fs::path models = tmp / "m2";
        REQUIRE(ensure_coreml_model_files(models, {file_for(correct_sha)}, true, base_url));
        // Second call with downloads OFF: a verified cache must still return true.
        REQUIRE(ensure_coreml_model_files(models, {file_for(correct_sha)}, false, base_url));
        REQUIRE(read_file(models / "foo_extracted" / "hello.txt") == "correct-content");
    }

    SECTION("re-pin triggers re-download of a stale cached archive (THE BUG)") {
        fs::path models = tmp / "m3";
        // Step 1: seed the cache with a STALE archive (different bytes, different SHA)
        // under the canonical name, plus a stale extracted dir.
        fs::path stale_archive = src / "stale.tar.gz";
        std::string stale_sha =
            make_targz_fixture(tmp / "stage_stale", stale_archive, "foo_extracted", "hello.txt", "stale-old-content");
        REQUIRE(stale_sha != correct_sha);
        fs::create_directories(models);
        fs::copy_file(stale_archive, models / "foo.tar.gz", fs::copy_options::overwrite_existing);
        fs::create_directories(models / "foo_extracted");
        { std::ofstream(models / "foo_extracted" / "hello.txt") << "stale-old-content"; }
        // Step 2: ship the NEW pin. The cache holds the OLD bytes.
        REQUIRE(ensure_coreml_model_files(models, {file_for(correct_sha)}, true, base_url));
        // The archive now matches the NEW pin, and the extracted content is the NEW one.
        REQUIRE(verify_sha256(models / "foo.tar.gz", correct_sha));
        REQUIRE(read_file(models / "foo_extracted" / "hello.txt") == "correct-content");
    }

    SECTION("stale cache + download disabled -> clean fail-closed (no stale run)") {
        fs::path models = tmp / "m4";
        fs::path stale_archive = src / "stale2.tar.gz";
        make_targz_fixture(tmp / "stage_stale2", stale_archive, "foo_extracted", "hello.txt", "stale-old-content");
        fs::create_directories(models);
        fs::copy_file(stale_archive, models / "foo.tar.gz", fs::copy_options::overwrite_existing);
        // Stale bytes, no network: must NOT silently succeed on existence alone.
        REQUIRE_FALSE(ensure_coreml_model_files(models, {file_for(correct_sha)}, false, base_url));
    }

    fs::remove_all(tmp);
}

#endif // WMR_BUILD_AI_COREML_SD
#endif // WMR_BUILD_REGEN
