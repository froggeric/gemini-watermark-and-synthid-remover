#pragma once
#include <string>

namespace wmr::gui {

// 32 lowercase hex chars (16 raw bytes from the platform CSPRNG). The token
// is the server's only auth, so an entropy failure is fatal rather than a
// predictable fallback.
std::string generate_token();

// Host-header allowlist (DNS-rebinding defense). Accepts exactly
// 127.0.0.1:P / localhost:P / [::1]:P for the given port (host part is
// case-insensitive; one trailing dot is tolerated). Anything else (absent,
// empty, foreign host, wrong or missing port) is rejected.
bool host_allowed(const std::string& host_header, int port);

}
