// Wording lock for the anti-detection help text (the honesty lock). Asserts on
// the inline antidetect_help_text() WITHOUT linking cli_app.cpp (the
// synthid_attack_cli_test.cpp pattern). Keep in sync with cli_app.hpp.
#include <catch2/catch_test_macros.hpp>

#include "cli/cli_app.hpp"

#include <regex>
#include <string>

TEST_CASE("antidetect help text states the honest scope", "[antidetect][cli]") {
    const std::string t = wmr::antidetect_help_text();

    // The option is named and both stages are described.
    for (const char* needle : {"--antidetect", "physics", "adversarial", "Square Attack",
                               "LPIPS", "camera", "sensor noise"}) {
        INFO("missing needle: " << needle);
        REQUIRE(t.find(needle) != std::string::npos);
    }

    // The honesty-lock caveats (adjacent string literals join without newlines,
    // so these are plain substrings of one long line).
    for (const char* needle : {"LOCAL surrogate", "untested", "out of scope",
                               "No guarantee", "LOSSY", "MLLM reasoning detectors"}) {
        INFO("missing caveat: " << needle);
        REQUIRE(t.find(needle) != std::string::npos);
    }

    // No over-claims, anywhere.
    for (const char* bad : {"undetectable", "guaranteed", "beats any detector",
                            "fool any detector", "100%"}) {
        INFO("over-claim present: " << bad);
        REQUIRE(t.find(bad) == std::string::npos);
    }

    // Determinism + download disclosure.
    REQUIRE(t.find("Deterministic") != std::string::npos);
    REQUIRE(t.find("downloads") != std::string::npos);
}
