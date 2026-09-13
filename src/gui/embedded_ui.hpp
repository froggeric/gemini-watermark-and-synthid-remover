#pragma once
// The embedded UI views (Task 5 created the struct EMPTY; Task 7 fills it from
// the generated assets/embedded_gui_assets.hpp). Three SEPARATE resources (no
// inlining) so the HTML CSP stays enforceable.
#include <string_view>

namespace wmr::gui {

struct EmbeddedUi {
    std::string_view html;  // assets/gui/index.html
    std::string_view js;    // assets/gui/app.js
    std::string_view css;   // assets/gui/style.css
    std::string_view favicon32;   // assets/gui/favicon-32.png
    std::string_view favicon96;   // assets/gui/favicon-96.png (HiDPI tabs)
    std::string_view touch_png;   // assets/gui/apple-touch-icon.png (180px)

    // Default construction fills the views over the generated header's
    // inline constexpr arrays (defined in embedded_ui.cpp, the only TU that
    // includes it). The arrays are static storage, so the views are valid for
    // the whole process lifetime, which is what the route lambdas need.
    EmbeddedUi();

    // Per-view served Content-Type.
    static constexpr std::string_view html_mime = "text/html; charset=utf-8";
    static constexpr std::string_view js_mime = "application/javascript; charset=utf-8";
    static constexpr std::string_view css_mime = "text/css; charset=utf-8";
    static constexpr std::string_view png_mime = "image/png";
};

}  // namespace wmr::gui
