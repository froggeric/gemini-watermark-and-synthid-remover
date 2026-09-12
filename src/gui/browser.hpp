#pragma once
#include <string>

namespace wmr::gui {

// Opens the GUI URL in the user's browser. Honors $BROWSER (with an optional
// %s placeholder), then the platform default (open / xdg-open / wslview /
// ShellExecuteW). Returns false when no browser could be launched; callers
// treat that as non-fatal (the URL is on stdout).
bool open_browser(const std::string& url);

}
