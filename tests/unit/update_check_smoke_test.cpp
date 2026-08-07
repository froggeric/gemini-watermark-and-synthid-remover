#include <catch2/catch_test_macros.hpp>

#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"
#include <cstdlib>
using namespace wmr;

// Opt-in network smoke test: proves the live GitHub Releases fetch returns a
// gzip-decoded (not raw gzip) body and a parseable tag. SKIPs unless
// WMR_NETWORK_TESTS=1, so it never runs in CI (no network) and never makes an
// unexpected request. Note: we do NOT assert the fetched tag is >= APP_VERSION;
// the local tree may be AHEAD of the latest published release (the 1.16.3
// release was not pushed), so a version-comparison assertion would fail flakily.
// The smoke proves the fetch + gzip-decode + parse path works, nothing more.
TEST_CASE("live GitHub releases/latest fetch decodes", "[update-check][smoke]") {
    if (!std::getenv("WMR_NETWORK_TESTS")) {
        WARN("set WMR_NETWORK_TESTS=1 to run");
        return;  // SKIP
    }

    FetchResult fr = fetch_latest_release("");
    REQUIRE(fr.ok);
    REQUIRE(fr.http_code == 200);
    // Body must be decoded JSON, not raw gzip bytes: "tag_name" is readable.
    REQUIRE(fr.body.find("tag_name") != std::string::npos);
    auto tag = parse_release_json(fr.body);
    REQUIRE(tag.has_value());
    REQUIRE_FALSE(tag->empty());
    // Sanity: the tag is a parseable release version.
    REQUIRE(parse_version(*tag).has_value());
}
#endif  // WMR_UPDATE_CHECK
