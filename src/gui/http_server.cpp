#include "gui/http_server.hpp"

#include <cstdio>

#if !defined(_WIN32)
#include <sys/socket.h>  // setsockopt/SO_REUSEADDR override below
#endif

#include "gui/browser.hpp"
#include "gui/security.hpp"

#ifndef APP_VERSION
#define APP_VERSION "0.0.0"
#endif

namespace wmr::gui {

int GuiServer::run(const GuiServerConfig& cfg, const RouteRegistrar& reg) {
    token_ = generate_token();
    // cpp-httplib's default socket options set SO_REUSEPORT, which would let a
    // SECOND wmr silently share a pinned port (kernel load-balances between
    // them) instead of failing the "port in use" bind. Plain SO_REUSEADDR
    // keeps the ephemeral fast-restart case working while making the pinned
    // collision an error.
    svr_.set_socket_options([](socket_t sock) {
#if !defined(_WIN32)
        const int one = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#else
        // Windows: keep the httplib default (SO_REUSEADDR via winsock).
        (void)sock;
#endif
    });
    svr_.set_payload_max_length(1ull << 30);  // 1 GiB upload cap
    svr_.set_default_headers({{"X-Content-Type-Options", "nosniff"}});

    svr_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        const std::string want = "/" + token_;
        if (req.path.rfind(want, 0) != 0
            || (req.path.size() > want.size() && req.path[want.size()] != '/')) {
            res.status = 404;  // empty body: probes learn nothing
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!host_allowed(req.get_header_value("Host"), port_)) {
            res.status = 404;  // DNS-rebinding defense
            return httplib::Server::HandlerResponse::Handled;
        }
        if (req.get_header_value("Sec-Fetch-Site") == "cross-site") {
            res.status = 403;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    reg(svr_, token_);

    if (cfg.port > 0) {
        if (!svr_.bind_to_port("127.0.0.1", cfg.port)) {
            std::fprintf(stderr,
                         "wmr gui: port %d in use; choose another or omit --gui-port for an ephemeral one\n",
                         cfg.port);
            return 1;
        }
        port_ = cfg.port;
    } else {
        // NOTE: the 2nd arg of bind_to_any_port is socket_flags, NOT a port.
        port_ = svr_.bind_to_any_port("127.0.0.1");
        if (port_ <= 0) {
            std::fprintf(stderr, "wmr gui: could not bind an ephemeral port on 127.0.0.1\n");
            return 1;
        }
    }

    std::printf("wmr %s GUI\n", APP_VERSION);
    std::printf("Listening on %s\n", url().c_str());
    std::fflush(stdout);

    if (cfg.open_browser) {
        (void)open_browser(url());  // best-effort: a headless box is not fatal
    }
    svr_.listen_after_bind();
    return 0;
}

std::string GuiServer::url() const {
    return "http://127.0.0.1:" + std::to_string(port_) + "/" + token_ + "/";
}

}
