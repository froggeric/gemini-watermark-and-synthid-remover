#pragma once
#include <functional>
#include <string>

#include "gui/vendored.hpp"  // httplib (single include point for the vendored headers)

namespace wmr::gui {

struct GuiServerConfig {
    int port = 0;             // 0 = ephemeral port (bind_to_any_port)
    bool open_browser = true; // false for --no-browser and tests
};

// Token-secured 127.0.0.1 server. run() installs the security stack
// (token-prefix gate, Host allowlist, Sec-Fetch-Site check, 1 GiB payload
// cap, nosniff default header), hands the server + token to the route
// registrar, binds, prints the banner, opens the browser, then blocks in
// listen_after_bind() until stop().
class GuiServer {
public:
    // Routes are registered WITH the token prefix (cpp-httplib patterns are
    // full-path): reg builds "/" + token + "/api/jobs" etc.
    using RouteRegistrar = std::function<void(httplib::Server&, const std::string& token)>;

    // Returns 0 on clean stop, 1 on bind failure (stderr message, no banner,
    // no browser).
    int run(const GuiServerConfig& cfg, const RouteRegistrar& reg);

    void stop() { svr_.stop(); }
    std::string url() const;  // http://127.0.0.1:<port>/<token>/
    int bound_port() const { return port_; }
    const std::string& token() const { return token_; }

private:
    httplib::Server svr_;
    std::string token_;
    int port_ = -1;
};

}
