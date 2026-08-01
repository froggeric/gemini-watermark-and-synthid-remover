// WS1a: lock the softened SynthID wording.
//
// The spectral path cannot verifiably REMOVE the invisible SynthID-Image watermark.
// SynthID-Image is a content-conditional neural watermark (arXiv:2510.09263) and
// Google publishes no public verifier, so a frequency-domain subtractor can at best
// SUPPRESS the carrier (a heuristic, not a verifiable removal). Every user-facing
// string that refers to this path must say "suppress"/"heuristic" rather than "remove".
//
// The VISIBLE Gemini/Veo diamond path genuinely removes (exact reverse-alpha-blend),
// so legitimate "remove" wording for the visible mark is out of scope and is NOT
// matched by the checks below.
//
// This is a source-level scan: deterministic, needs no fixture, and runs in every
// CI environment. It is the direct lock for acceptance criterion #2 of WS1a.

#include <catch2/catch_test_macros.hpp>

#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace {

// Read a UTF-8 text file from a path relative to CWD (catch_discover_tests runs the
// suite from ${CMAKE_SOURCE_DIR}, so repo-relative paths resolve). Returns an empty
// string when the file is missing; callers assert non-empty before scanning.
std::string read_file(const std::string& rel_path) {
    std::ifstream in(rel_path, std::ios::binary);
    if (!in) return "";
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// The forbidden wording: a claim that the SPECTRAL path "removes" SynthID.
// Matches:
//   "remove SynthID"  /  "removes SynthID"          (remove[whitespace]SynthID)
//   "removes the SynthID watermark"                 (removes?[whitespace]the[whitespace]SynthID[whitespace]watermark)
//   "remove the SynthID invisible watermark"        (optional "invisible " before "watermark")
// Case-insensitive. Legitimate visible-mark "remove" wording is not matched because
// the pattern requires "SynthID" right after "remove"/"removes the".
bool contains_forbidden_synthid_claim(const std::string& text) {
    static const std::regex re(
        R"(remove\s+SynthID|removes?\s+the\s+SynthID\s+(invisible\s+)?watermark)",
        std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_search(text, re);
}

bool contains_softened_wording(const std::string& text) {
    static const std::regex re(R"(suppress|heuristic)",
                               std::regex_constants::ECMAScript | std::regex_constants::icase);
    return std::regex_search(text, re);
}

}  // namespace

TEST_CASE("SynthID invisible-watermark claims use 'suppress'/'heuristic', not 'remove'",
          "[synthid][wording]") {
    const std::string cli = read_file("src/cli/cli_app.cpp");
    const std::string codebook = read_file("src/synthid/codebook_subtractor.cpp");
    const std::string noise = read_file("src/synthid/noise_residual_subtractor.cpp");
    const std::string detector = read_file("src/detection/synthid_detector.cpp");
    const std::string readme = read_file("README.md");

    // Files must be readable (CWD must be the repo root).
    REQUIRE_FALSE(cli.empty());
    REQUIRE_FALSE(codebook.empty());
    REQUIRE_FALSE(noise.empty());
    REQUIRE_FALSE(detector.empty());
    REQUIRE_FALSE(readme.empty());

    SECTION("no file claims the spectral path 'removes' SynthID") {
        CHECK_FALSE(contains_forbidden_synthid_claim(cli));
        CHECK_FALSE(contains_forbidden_synthid_claim(codebook));
        CHECK_FALSE(contains_forbidden_synthid_claim(noise));
        CHECK_FALSE(contains_forbidden_synthid_claim(detector));
        CHECK_FALSE(contains_forbidden_synthid_claim(readme));
    }

    SECTION("SynthID-facing files advertise the softened 'suppress'/'heuristic' wording") {
        CHECK(contains_softened_wording(cli));
        CHECK(contains_softened_wording(codebook));
        CHECK(contains_softened_wording(noise));
        CHECK(contains_softened_wording(readme));
    }
}
