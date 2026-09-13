// Drift guard for the embedded GUI assets (Task 7): each committed byte array
// in assets/embedded_gui_assets.hpp must equal the corresponding assets/gui/
// source on disk. Tests run from the source root (catch_discover_tests
// WORKING_DIRECTORY), so the relative paths resolve; a source file that is
// absent SKIPs (a packaged-tree run), but a file that DISAGREES fails:
// edit assets/gui/* -> re-run scripts/embed_gui_assets.py -> commit both.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "embedded_gui_assets.hpp"  // generated (assets/ is on the include path)

namespace {

std::vector<unsigned char> read_asset(const std::string& rel) {
    std::ifstream f(rel, std::ios::binary);
    if (!f.is_open()) SKIP("asset source not present (run from the source root): " + rel);
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(f),
                                      std::istreambuf_iterator<char>());
}

void require_same(const std::string& rel, const unsigned char* data, std::size_t size) {
    const std::vector<unsigned char> src = read_asset(rel);
    REQUIRE(size > 0);  // a zero-length array means the embed never ran
    REQUIRE(src.size() == size);
    for (std::size_t i = 0; i < size; ++i) {
        if (src[i] != data[i]) {
            FAIL(rel << " differs from the embedded copy at byte " << i
                 << " (re-run scripts/embed_gui_assets.py and commit both)");
        }
    }
}

}  // namespace

TEST_CASE("embedded GUI html matches assets/gui/index.html", "[gui][gui-assets]") {
    require_same("assets/gui/index.html", wmr::gui::index_data, wmr::gui::index_data_size);
}

TEST_CASE("embedded GUI js matches assets/gui/app.js", "[gui][gui-assets]") {
    require_same("assets/gui/app.js", wmr::gui::app_data, wmr::gui::app_data_size);
}

TEST_CASE("embedded GUI css matches assets/gui/style.css", "[gui][gui-assets]") {
    require_same("assets/gui/style.css", wmr::gui::style_data, wmr::gui::style_data_size);
}

TEST_CASE("embedded GUI icons match assets/gui", "[gui][gui-assets]") {
    require_same("assets/gui/favicon-32.png", wmr::gui::favicon_32_data,
                 wmr::gui::favicon_32_data_size);
    require_same("assets/gui/favicon-96.png", wmr::gui::favicon_96_data,
                 wmr::gui::favicon_96_data_size);
    require_same("assets/gui/apple-touch-icon.png", wmr::gui::apple_touch_data,
                 wmr::gui::apple_touch_data_size);
}
