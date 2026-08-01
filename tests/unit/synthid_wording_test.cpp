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
// matched by the checks below (the pattern requires "SynthID" right after
// "remove"/"removes", or the noun "SynthID removal").
//
// Intentional exception: the internal C++ symbols remove_synthid (method),
// RemovalStrength / RemovalConfig (types), and params.removal (field) are NOT
// renamed. They live in public headers and have many call sites, and they are not
// user-facing. The forbidden regex deliberately does not match them: it requires
// whitespace (not an underscore or ".") between "remove(s)" and "SynthID", and the
// noun form requires the literal phrase "SynthID removal".
//
// This is a source-level scan: deterministic, needs no fixture, and runs in every
// CI environment. It is the direct lock for acceptance criterion #2 of WS1a.
// The forbidden-claim scan auto-discovers every .cpp under src/synthid/ and src/cli/
// (plus the SynthID detector) so a new source file (e.g. a "SynthID removal" log added
// to batch_processor.cpp) cannot silently bypass it.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

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

// Sorted list of regular .cpp file paths under a directory (relative to CWD).
std::vector<std::string> list_cpp_files(const std::string& dir_rel) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir_rel, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".cpp") {
            out.push_back(entry.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

// The forbidden wording: a claim that the SPECTRAL path "removes" SynthID. Catches
// the rollback vectors that matter, both verb and noun forms:
//   "remove SynthID"  /  "removes SynthID"               (removes?[ws]SynthID)
//   "removes the SynthID watermark"                      (removes?[ws]the[ws]SynthID[ws]watermark)
//   "remove the SynthID invisible watermark"             (optional "invisible " before "watermark)
//   "SynthID removal"  (the noun form the log strings actually used)   (SynthID[ws]removal)
// Case-insensitive. Does NOT match remove_synthid / RemovalStrength / params.removal
// (underscore and "." are not whitespace, and those tokens contain no "SynthID removal"
// phrase), so the intentionally-kept internal symbols are unaffected.
bool contains_forbidden_synthid_claim(const std::string& text) {
    static const std::regex re(
        R"(removes?\s+(the\s+)?SynthID\s+(invisible\s+)?watermark|SynthID\s+removal|removes?\s+SynthID)",
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
    // Auto-discover every .cpp under the SynthID and CLI trees so a new source file
    // cannot silently bypass the forbidden-claim scan (e.g. a "SynthID removal" log
    // added to src/cli/batch_processor.cpp). Both directories must exist.
    REQUIRE(std::filesystem::is_directory("src/synthid"));
    REQUIRE(std::filesystem::is_directory("src/cli"));

    std::vector<std::string> sources = list_cpp_files("src/synthid");
    const auto cli_sources = list_cpp_files("src/cli");
    sources.insert(sources.end(), cli_sources.begin(), cli_sources.end());
    sources.push_back("src/detection/synthid_detector.cpp");

    const std::string readme = read_file("README.md");
    REQUIRE_FALSE(readme.empty());

    SECTION("no source file claims the spectral path 'removes' SynthID") {
        for (const auto& path : sources) {
            const std::string text = read_file(path);
            INFO("scanning " << path);
            REQUIRE_FALSE(text.empty());  // file must be readable from repo root
            CHECK_FALSE(contains_forbidden_synthid_claim(text));
        }
        INFO("scanning README.md");
        CHECK_FALSE(contains_forbidden_synthid_claim(readme));
    }

    SECTION("SynthID-facing files advertise the softened 'suppress'/'heuristic' wording") {
        // Files that carry user-facing SynthID log/CLI text. If you add a new
        // SynthID-facing log/CLI string in a different file, add its path here too.
        const std::vector<std::string> softened_sources = {
            "src/cli/cli_app.cpp",
            "src/synthid/codebook_subtractor.cpp",
            "src/synthid/noise_residual_subtractor.cpp",
            "README.md",
        };
        for (const auto& path : softened_sources) {
            const std::string text = read_file(path);
            INFO("softened-wording check on " << path);
            CHECK(contains_softened_wording(text));
        }
    }

    SECTION("forbidden-claim regex actually catches a real rollback") {
        // Positive cases: if these ever stop matching, the negative checks above
        // become vacuous. These are the exact softened strings (reverted to their
        // pre-reframe form) plus the verb forms the regex must also catch.
        CHECK(contains_forbidden_synthid_claim("SynthID removal complete"));
        CHECK(contains_forbidden_synthid_claim("SynthID removal requires --codebook"));
        CHECK(contains_forbidden_synthid_claim("Codebook-free SynthID removal: 800x600"));
        CHECK(contains_forbidden_synthid_claim("removes SynthID"));
        CHECK(contains_forbidden_synthid_claim("Remove SynthID watermark only"));
        CHECK(contains_forbidden_synthid_claim("removes the SynthID invisible watermark"));
        // Negative cases: the intentionally-kept internal symbols must NOT trip it.
        CHECK_FALSE(contains_forbidden_synthid_claim("subtractor.remove_synthid(image, config);"));
        CHECK_FALSE(contains_forbidden_synthid_claim("RemovalStrength::Aggressive"));
        CHECK_FALSE(contains_forbidden_synthid_claim("float removal_factor = params.removal;"));
    }
}
