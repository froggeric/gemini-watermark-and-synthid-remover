// Task 7: honesty lock for the --synthid-attack regen path.
//
// The regen path (SDXL img2img) is the only *validated* SynthID scrub in the
// literature, but it is LOSSY, leaves a forensic footprint, and there is no public
// SynthID verifier, so it cannot be claimed as a verifiable "removal" either. This
// test locks the user-facing help text to:
//   (a) carry the honest caveats: "lossy", "download", "verifier", "--synthid-attack";
//   (b) NOT claim the regen path "removes SynthID" / "removes the SynthID" /
//       "SynthID removal" (the same forbidden-claim style as synthid_wording_test.cpp).
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

// Case-insensitive substring search (the help text uses "LOSSY" for emphasis;
// the lock should not break on capitalization). NOT static — the needle varies per
// call, so a static regex would keep the first needle and silently re-check it.
static bool contains_ci(const std::string& text, const std::string& needle) {
    const std::regex re(needle,
        std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_search(text, re);
}

TEST_CASE("synthid-attack help text carries the honest caveats", "[synthid][cli][wording]") {
    const std::string text = wmr::synthid_attack_help_text();

    // (a) The caveats MUST be present (case-insensitive: the text uses "LOSSY").
    REQUIRE(contains_ci(text, "--synthid-attack"));
    REQUIRE(contains_ci(text, "lossy"));
    REQUIRE(contains_ci(text, "download"));
    REQUIRE(contains_ci(text, "verifier"));

    // (b) The regen path MUST NOT be claimed as a verifiable "removal" of SynthID.
    // Mirrors the forbidden regex in synthid_wording_test.cpp (case-insensitive).
    static const std::regex forbidden(
        R"(removes?\s+(the\s+)?SynthID\s+(invisible\s+)?watermark|SynthID\s+removal|removes?\s+SynthID)",
        std::regex_constants::ECMAScript | std::regex_constants::icase);
    CHECK_FALSE(std::regex_search(text, forbidden));
}
