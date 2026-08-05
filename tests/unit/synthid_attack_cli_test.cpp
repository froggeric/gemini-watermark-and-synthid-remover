// Honesty lock for the --synthid-attack regen path.
//
// The regen path (SDXL img2img) is the only *validated* SynthID scrub in the
// literature, validated against Google's official "Verify with SynthID" verifier.
// It is still LOSSY, leaves a forensic footprint, and that verifier is manual +
// rate-limited with no API, so per-run success cannot be auto-confirmed in-process
// and the help must not over-claim an absolute "removal". This
// test locks the user-facing help text to:
//   (a) carry the honest caveats: "lossy", "download", "verifier", "--synthid-attack", "regen";
//   (b) NOT claim the regen path "removes SynthID" / "removes the SynthID" /
//       "SynthID removal" (the forbidden-claim style from the removed
//       synthid_wording_test.cpp, preserved here as the single honesty-lock now
//       that the spectral path is gone).
//   (c) advertise regen as the only method (the spectral path was removed in 1.16.0).
//
// It asserts on the inline synthid_attack_help_text() (defined in cli/cli_app.hpp)
// directly, so it is deterministic and runs in every CI environment, with no
// subprocess/popen and no dependency on cli_app.cpp (which would drag in FFmpeg).
// The function is pure prose and exists in BOTH the lean (WMR_BUILD_REGEN off) and
// full builds, so this test is always compiled and gates the wording regardless of
// whether the regen backend is linked.

#include <catch2/catch_test_macros.hpp>

#include <regex>
#include <string>

#include "cli/cli_app.hpp"

// Case-insensitive substring search. NOT static — the needle varies per call, so a
// static regex would keep the first needle and silently re-check it.
static bool contains_ci(const std::string& text, const std::string& needle) {
    const std::regex re(needle,
        std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_search(text, re);
}

TEST_CASE("synthid-attack help text carries the honest caveats", "[synthid][cli][wording]") {
    const std::string text = wmr::synthid_attack_help_text();

    // (a) The caveats MUST be present.
    REQUIRE(contains_ci(text, "--synthid-attack"));
    REQUIRE(contains_ci(text, "regen"));
    REQUIRE(contains_ci(text, "lossy"));
    REQUIRE(contains_ci(text, "download"));
    REQUIRE(contains_ci(text, "verifier"));

    // (b) The regen path MUST NOT be claimed as a verifiable "removal" of SynthID.
    static const std::regex forbidden(
        R"(removes?\s+(the\s+)?SynthID\s+(invisible\s+)?watermark|SynthID\s+removal|removes?\s+SynthID)",
        std::regex_constants::ECMAScript | std::regex_constants::icase);
    CHECK_FALSE(std::regex_search(text, forbidden));

    // (c) The removed spectral path must NOT be advertised. The spectral detector +
    //     suppressor was deleted in 1.16.0 (it did not work); the help text must
    //     not promise it as an option.
    CHECK_FALSE(contains_ci(text, "spectral"));
    CHECK_FALSE(contains_ci(text, "codebook"));
}
