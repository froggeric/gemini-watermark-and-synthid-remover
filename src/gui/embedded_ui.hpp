#pragma once
// The embedded UI views (Task 5 creates the struct EMPTY; Task 7 fills it from
// the generated assets/embedded_gui_assets.hpp). While a view is empty its
// token-prefixed route 404s, so a Task-5-era binary serves the API but no UI.
// Three SEPARATE resources (no inlining) so the HTML CSP stays enforceable.
#include <string_view>

namespace wmr::gui {

struct EmbeddedUi {
    std::string_view html;  // assets/gui/index.html
    std::string_view js;    // assets/gui/app.js
    std::string_view css;   // assets/gui/style.css

    // Per-view served Content-Type.
    static constexpr std::string_view html_mime = "text/html; charset=utf-8";
    static constexpr std::string_view js_mime = "application/javascript; charset=utf-8";
    static constexpr std::string_view css_mime = "text/css; charset=utf-8";
};

}  // namespace wmr::gui
