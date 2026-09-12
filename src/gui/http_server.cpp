#include "gui/http_server.hpp"

#include <charconv>
#include <cstdio>

#if !defined(_WIN32)
#include <sys/socket.h>  // setsockopt/SO_REUSEADDR override below (winsock comes via httplib.h)
#endif

#include "gui/browser.hpp"
#include "gui/security.hpp"

#ifndef APP_VERSION
#define APP_VERSION "0.0.0"
#endif

namespace wmr::gui {
namespace {

// The 1 GiB request-body cap, applied twice: the declared-Content-Length
// fast path in the pre-routing handler (an honest client learns the 413
// before a single body byte is buffered) and set_payload_max_length (the
// per-received-byte hard bound that also catches lying and chunked clients).
constexpr unsigned long long kMaxRequestBytes = 1ull << 30;

}  // namespace

int GuiServer::run(const GuiServerConfig& cfg, const RouteRegistrar& reg) {
    token_ = generate_token();
    // cpp-httplib's default socket options set SO_REUSEPORT where defined
    // (POSIX) and plain SO_REUSEADDR on Windows. SO_REUSEPORT would let a
    // SECOND wmr silently share a pinned port (the kernel load-balances
    // between the listeners) instead of failing the "port in use" bind, so we
    // override the default down to SO_REUSEADDR. The override REPLACES the
    // default (httplib does not chain them), so each branch must set the
    // option itself: POSIX SO_REUSEADDR, and on Windows winsock's SO_REUSEADDR
    // (same fast-restart behavior the library default had there).
    svr_.set_socket_options([](socket_t sock) {
        const int one = 1;
#if defined(_WIN32)
        // winsock's setsockopt takes a char* buffer and an int length.
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR,
                   reinterpret_cast<const char*>(&one), static_cast<int>(sizeof one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
#endif
    });
    svr_.set_payload_max_length(kMaxRequestBytes);  // 1 GiB upload cap (hard bound)
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
        // Honest-client fast path: cpp-httplib calls this handler after the
        // headers but before the body, so a DECLARED Content-Length past the
        // cap is refused without buffering a byte of it. Anything the header
        // does not declare (chunked, lying lengths) still hits the
        // per-received-byte hard bound above.
        const std::string len = req.get_header_value("Content-Length");
        if (!len.empty()) {
            unsigned long long declared = 0;
            const auto parsed =
                std::from_chars(len.data(), len.data() + len.size(), declared);
            if (parsed.ec == std::errc{} && parsed.ptr == len.data() + len.size()
                && declared > kMaxRequestBytes) {
                res.status = 413;
                res.set_content(
                    nlohmann::json{{"error",
                                    {{"code", "payload_too_large"},
                                     {"message", "request body exceeds the 1 GiB cap"}}}}
                        .dump(),
                    "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
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
    if (!cfg.banner_tail.empty()) {
        std::printf("%s\n", cfg.banner_tail.c_str());
    }
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
