// GUI live-server integration tests (Task 8): boot the REAL GuiServer +
// JobManager stack on an ephemeral port inside the test process and drive it
// over loopback HTTP with cpp-httplib's client. Everything is synthesized in
// memory (no test-images/ fixtures). Covers the security gates (AC3), the
// upload/queue caps (AC4), the job lifecycle (upload -> poll -> download,
// cancel queued/running, list ordering, error codes), and the CLI-policy
// parity invariant (AC7: the downloaded cleaned bytes must equal an
// in-process remove_still + write_still_output run on the same fixture).
//
// Blocking cases inject a processor pinned on a promise (the gui_jobs_test
// pattern). The ReleaseOnExit guard MUST be declared AFTER the ServerCtx so
// it is destroyed (and releases) BEFORE the manager joins its worker: a
// worker still blocked at join time rides the 5 s bounded stop into
// std::_Exit(0) and kills the whole test binary.
#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "gui/vendored.hpp"  // httplib + nlohmann::json (single include point)

#include "core/still_remove.hpp"
#include "core/watermark_engine.hpp"
#include "detection/still_geometry.hpp"  // kStillPresetNames
#include "gui/api.hpp"
#include "gui/embedded_ui.hpp"
#include "gui/http_server.hpp"
#include "gui/jobs.hpp"
#include "gui/options.hpp"

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

using namespace wmr;
using namespace wmr::gui;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// --- fixture helpers (the gui_jobs_test shapes) ---

// RAII temp dir (unique name: pid + counter + steady-clock ns).
struct TempDir {
    fs::path p;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        const auto ns = std::chrono::steady_clock::now().time_since_epoch().count();
        p = fs::temp_directory_path() / ("wmr_gui_srv_" + std::to_string(getpid()) + "_" +
                                         std::to_string(counter.fetch_add(1)) + "_" +
                                         std::to_string(ns));
        fs::create_directories(p);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(p, ec);
    }
};

// Fires a blocking processor's release promise when the test scope exits, even
// on assertion failure (see the file-top comment for the declaration order).
struct ReleaseOnExit {
    std::promise<void>& p;
    ~ReleaseOnExit() {
        try {
            p.set_value();
        } catch (...) {
        }
    }
};

std::string encode_png(const cv::Mat& img) {
    std::vector<unsigned char> buf;
    cv::imencode(".png", img, buf);
    return std::string(buf.begin(), buf.end());
}

std::string small_png_bytes() {  // valid, sniffable, tiny (queue-cap posts)
    return encode_png(cv::Mat(30, 30, CV_8UC3, cv::Scalar(90, 90, 90)));
}

// The canonical detectable fixture (still_remove_test recipe): a V1 mark
// added by the engine at the model position on gray content.
cv::Mat marked_mat() {
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(90, 90, 90));
    WatermarkEngine engine;
    engine.add_watermark(img);
    return img;
}

// Busy synthetic content with no watermark (still_remove_test recipe): the
// detectors must report no-watermark.
cv::Mat busy_mat() {
    cv::Mat img(500, 500, CV_8UC3);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            img.at<cv::Vec3b>(y, x) = {(uchar)((x * 7) % 251), (uchar)((y * 13) % 241),
                                       (uchar)((x * y) % 233)};
    return img;
}

std::string read_file_bytes(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string to_lower(std::string s) {
    for (char& c : s)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    return s;
}

// --- HTTP helpers: a fresh client per call (no stale keep-alive edge cases) ---

httplib::Result http_get(int port, const std::string& path,
                         const httplib::Headers& headers = {}) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(20);
    return cli.Get(path, headers);
}

httplib::Result http_post(int port, const std::string& path) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(20);
    return cli.Post(path);
}

// POST /api/jobs with the given (filename, bytes) file parts. The "options"
// part's filename MUST be the empty string: a part is a FIELD iff its filename
// is empty, so the server's form.get_field("options") sees it.
httplib::Result http_post_job(
    int port, const std::string& token,
    const std::vector<std::pair<std::string, std::string>>& files,
    const std::string& options_json = R"({"denoise":"off"})") {
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(20);
    cli.set_write_timeout(20);
    httplib::UploadFormDataItems items;
    for (const auto& f : files)
        items.push_back({"files", f.second, f.first, "image/png"});
    if (!options_json.empty())
        items.push_back({"options", options_json, "", "application/json"});
    return cli.Post("/" + token + "/api/jobs", items);
}

// Bounded poll until the job reaches a terminal status; returns the final
// snapshot JSON (the detail endpoint's shape).
json wait_job_terminal(int port, const std::string& token, const std::string& id,
                       int timeout_ms = 10000) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    for (;;) {
        auto res = http_get(port, "/" + token + "/api/jobs/" + id);
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 200);
        const auto j = json::parse(res->body);
        const std::string status = j["status"];
        if (status == "done" || status == "failed" || status == "cancelled") return j;
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// Bounded poll until every listed id is terminal via the list endpoint;
// returns the full list array (newest first).
json wait_list_all_terminal(int port, const std::string& token,
                            const std::vector<std::string>& ids, int timeout_ms = 15000) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    for (;;) {
        auto res = http_get(port, "/" + token + "/api/jobs");
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 200);
        const auto arr = json::parse(res->body);
        bool all = true;
        for (const auto& id : ids) {
            bool terminal = false;
            for (const auto& e : arr)
                if (e["job_id"] == id) {
                    const std::string s = e["status"];
                    terminal = s == "done" || s == "failed" || s == "cancelled";
                }
            if (!terminal) all = false;
        }
        if (all) return arr;
        REQUIRE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

// Boots the real stack in-thread: JobManager over a temp run dir + GuiServer
// on its own thread (ephemeral port, no browser). The token is captured in
// the route registrar (which runs inside run(), before the bind), so the main
// thread never guesses it; readiness is then a bounded HTTP poll.
class ServerCtx {
public:
    ServerCtx() {
        mgr_ = std::make_unique<JobManager>(tmp_.p / "run");
        srv_thread_ = std::thread([this] {
            run_rc_ = server_.run(
                GuiServerConfig{0, false, ""},
                [this](httplib::Server& svr, const std::string& t) {
                    token_ = t;
                    register_routes(svr, t, *mgr_, build_api_features(), EmbeddedUi{});
                    token_ready_.store(true);  // release: publishes the token_ write
                });
        });
        if (!wait_ready()) {
            // Teardown BEFORE throwing: a joinable thread member terminating
            // mid-constructor would kill the binary.
            server_.stop();
            if (srv_thread_.joinable()) srv_thread_.join();
            throw std::runtime_error("gui server did not become ready within 10 s");
        }
    }

    ~ServerCtx() { shutdown(); }

    ServerCtx(const ServerCtx&) = delete;
    ServerCtx& operator=(const ServerCtx&) = delete;

    // Idempotent: stop the accept loop, join the server thread. The manager
    // then joins its worker as a member; any blocking processor must already
    // be released (the ReleaseOnExit discipline).
    void shutdown() {
        if (shut_) return;
        shut_ = true;
        server_.stop();
        if (srv_thread_.joinable()) srv_thread_.join();
    }

    JobManager& manager() { return *mgr_; }
    int port() const { return port_; }
    const std::string& token() const { return token_; }
    int run_rc() const { return run_rc_; }

private:
    bool wait_ready() {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!token_ready_.load()) {
            if (std::chrono::steady_clock::now() > deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // bound_port() turns positive between the registrar and listen (the
        // readiness poll below is the real synchronization point).
        while (server_.bound_port() <= 0) {
            if (std::chrono::steady_clock::now() > deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        port_ = server_.bound_port();
        for (;;) {  // the bind is done; the accept loop may not be yet
            auto res = http_get(port_, "/" + token_ + "/api/version");
            if (res && res->status == 200) return true;
            if (std::chrono::steady_clock::now() > deadline) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    TempDir tmp_;
    std::unique_ptr<JobManager> mgr_;  // joined by its own dtor, after the server thread is down
    GuiServer server_;
    std::thread srv_thread_;
    std::string token_;                // written by the server thread, published via token_ready_
    std::atomic<bool> token_ready_{false};
    int port_ = 0;
    int run_rc_ = -1;
    bool shut_ = false;
};

#ifndef _WIN32
// A raw loopback socket for the two cases the client API cannot express: the
// httplib client always sends a Host header and a truthful Content-Length.
// Non-blocking writes + poll, and writes never raise SIGPIPE (SO_NOSIGPIPE on
// macOS, MSG_NOSIGNAL elsewhere), so an early server response cannot be lost
// to a blocked or reset send.
class RawConn {
public:
    explicit RawConn(int port) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        REQUIRE(fd_ >= 0);
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(static_cast<unsigned short>(port));
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        REQUIRE(::connect(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0);
#ifdef SO_NOSIGPIPE
        int one = 1;
        ::setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
        const int flags = ::fcntl(fd_, F_GETFL, 0);
        REQUIRE(flags >= 0);
        REQUIRE(::fcntl(fd_, F_SETFL, flags | O_NONBLOCK) == 0);
    }
    ~RawConn() {
        if (fd_ >= 0) ::close(fd_);
    }
    RawConn(const RawConn&) = delete;
    RawConn& operator=(const RawConn&) = delete;

    bool send_all(const char* data, std::size_t n) {
        std::size_t off = 0;
        while (off < n) {
            const ssize_t w = ::send(fd_, data + off, n - off, send_flags());
            if (w > 0) {
                off += static_cast<std::size_t>(w);
                continue;
            }
            if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                pollfd p{fd_, POLLOUT, 0};
                if (::poll(&p, 1, 5000) <= 0) return false;
                continue;
            }
            return false;  // hung up
        }
        return true;
    }

    // Stream `chunk` until `total` bytes are sent or the server responds /
    // hangs up (checked between writes via poll, never by blocking in send);
    // then read the response.
    std::string flood_then_read(const std::string& chunk, long long sent, long long total,
                                int timeout_s) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
        while (sent < total) {
            pollfd p{fd_, POLLIN | POLLOUT, 0};
            const int r = ::poll(&p, 1, 200);
            if (r < 0) break;
            if (r > 0 && (p.revents & (POLLIN | POLLHUP | POLLERR)) != 0)
                break;  // the response (or hangup) is here: stop writing NOW
            if (r > 0 && (p.revents & POLLOUT) != 0) {
                const ssize_t w = ::send(fd_, chunk.data(), chunk.size(), send_flags());
                if (w > 0) sent += w;
                else if (w < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
                    break;  // hung up mid-flood: fall through to the read
            }
            if (std::chrono::steady_clock::now() > deadline) break;
        }
        return read_all(5000);
    }

    // Read to EOF (our raw requests always send "Connection: close") under a
    // deadline.
    std::string read_all(int timeout_ms) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::milliseconds(timeout_ms);
        std::string out;
        char buf[16384];
        for (;;) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                                       deadline - std::chrono::steady_clock::now())
                                       .count();
            if (remaining <= 0) break;
            pollfd p{fd_, POLLIN, 0};
            if (::poll(&p, 1, static_cast<int>(remaining)) <= 0) break;
            const ssize_t n = ::recv(fd_, buf, sizeof buf, 0);
            if (n > 0) out.append(buf, static_cast<std::size_t>(n));
            else if (n == 0 || errno != EAGAIN)
                break;  // EOF or error: the response is complete
        }
        return out;
    }

private:
    static int send_flags() {
#ifdef MSG_NOSIGNAL
        return MSG_NOSIGNAL;
#else
        return 0;
#endif
    }

    int fd_ = -1;
};
#endif  // !_WIN32

// No response may ever carry a CORS-unlock header (AC3).
void check_no_cors(const httplib::Response& r) {
    for (const auto& kv : r.headers)
        CHECK(to_lower(kv.first).rfind("access-control-allow-", 0) != 0);
}

}  // namespace

TEST_CASE("boots on an ephemeral port and tears down cleanly", "[gui][gui-server]") {
    ServerCtx ctx;
    REQUIRE(ctx.port() > 0);
    REQUIRE(ctx.token().size() == 32);  // 32 lowercase hex chars

    auto res = http_get(ctx.port(), "/" + ctx.token() + "/api/version");
    REQUIRE(res != nullptr);
    REQUIRE(res->status == 200);
    const auto j = json::parse(res->body);
    REQUIRE(j.contains("version"));
    REQUIRE(j.contains("features"));
    REQUIRE(j.contains("presets"));

    ctx.shutdown();
    REQUIRE(ctx.run_rc() == 0);  // a clean stop, not a bind failure
}

TEST_CASE("version endpoint lists the calibrated presets and build features", "[gui][gui-server]") {
    ServerCtx ctx;
    auto res = http_get(ctx.port(), "/" + ctx.token() + "/api/version");
    REQUIRE(res != nullptr);
    REQUIRE(res->status == 200);
    const auto j = json::parse(res->body);
    REQUIRE(j["presets"].size() == std::size(kStillPresetNames));
    for (std::size_t i = 0; i < std::size(kStillPresetNames); ++i)
        REQUIRE(j["presets"][i].get<std::string>() == kStillPresetNames[i]);
    REQUIRE(j["features"]["denoise_ai"].get<bool>() == build_api_features().denoise_ai);
}

TEST_CASE("security gates: un-prefixed paths, bad tokens, hostile Host, cross-site fetches",
          "[gui][gui-server]") {
    ServerCtx ctx;
    const std::string base = "/" + ctx.token();

    // Un-prefixed UI paths: 404 with an EMPTY body (probes learn nothing).
    for (const char* path : {"/", "/favicon.ico"}) {
        auto res = http_get(ctx.port(), path);
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 404);
        REQUIRE(res->body.empty());
        check_no_cors(*res);
    }

    // A well-formed but WRONG token: indistinguishable 404.
    std::string bad_token = ctx.token();
    bad_token.back() = bad_token.back() == '0' ? '1' : '0';
    {
        auto res = http_get(ctx.port(), "/" + bad_token + "/api/version");
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 404);
        check_no_cors(*res);
    }

    // DNS-rebinding defense: a foreign Host is rejected even with the right
    // token (the connection is loopback but the header declares evil.com).
    {
        auto res = http_get(ctx.port(), base + "/api/version",
                            httplib::Headers{{"Host", "evil.com:" + std::to_string(ctx.port())}});
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 404);
        check_no_cors(*res);
    }

    // Cross-site fetches are refused with 403 (everything above is 404).
    {
        auto res = http_get(ctx.port(), base + "/api/version",
                            httplib::Headers{{"Sec-Fetch-Site", "cross-site"}});
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 403);
        check_no_cors(*res);
    }

#ifndef _WIN32
    // A Host-less request (raw bytes; the client API always sends one).
    {
        RawConn conn(ctx.port());
        const std::string req =
            "GET " + base + "/api/version HTTP/1.1\r\nConnection: close\r\n\r\n";
        REQUIRE(conn.send_all(req.data(), req.size()));
        const std::string resp = conn.read_all(5000);
        REQUIRE(resp.rfind("HTTP/1.1 404", 0) == 0);
        REQUIRE(to_lower(resp).find("access-control-allow-") == std::string::npos);
    }
#endif

    // The gates do not over-block: valid token + default Host serves the
    // version and the real embedded UI page.
    {
        auto res = http_get(ctx.port(), base + "/api/version");
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 200);
        check_no_cors(*res);
    }
    {
        auto res = http_get(ctx.port(), base + "/");
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 200);
        REQUIRE(!res->body.empty());  // the embedded UI actually serves
        check_no_cors(*res);
    }
}

#ifndef _WIN32
TEST_CASE("a request body past the 1 GiB cap is refused 413", "[gui][gui-server]") {
    ServerCtx ctx;
    // The cap trips on RECEIVED bytes (the server streams the multipart body
    // and errors once the running total crosses 1 GiB), so the client must
    // actually stream past the cap; it stops the moment the 413 arrives. The
    // server buffers just over 1 GiB of the one part before tripping.
    RawConn conn(ctx.port());
    const std::string boundary = "wmr-test-boundary";
    const long long declared = (1ll << 30) + 4096;
    const std::string head =
        "POST /" + ctx.token() + "/api/jobs HTTP/1.1\r\n"
        "Host: 127.0.0.1:" + std::to_string(ctx.port()) + "\r\n"
        "Content-Type: multipart/form-data; boundary=" + boundary + "\r\n"
        "Content-Length: " + std::to_string(declared) + "\r\n"
        "Connection: close\r\n"
        "\r\n"
        "--" + boundary + "\r\n"
        "Content-Disposition: form-data; name=\"files\"; filename=\"big.png\"\r\n"
        "Content-Type: application/octet-stream\r\n"
        "\r\n";
    REQUIRE(conn.send_all(head.data(), head.size()));
    const std::string chunk(256 * 1024, 'x');
    const std::string resp =
        conn.flood_then_read(chunk, static_cast<long long>(head.size()), declared, 60);
    REQUIRE(!resp.empty());
    REQUIRE(resp.rfind("HTTP/1.1 413", 0) == 0);
}
#endif  // !_WIN32

TEST_CASE("a ninth pending job is refused 429 too_many_pending", "[gui][gui-server]") {
    std::promise<void> release;
    const auto go = release.get_future().share();
    ServerCtx ctx;  // the blocking processor is injected below, BEFORE the first POST
    ctx.manager().set_processor_for_tests([go](GuiFile& f, const JobOptions&, const fs::path&) {
        go.wait();  // every call blocks: 1 running + 7 queued = 8 outstanding
        f.outcome = FileOutcome::Removed;
        f.has_clean = true;
        return f;
    });
    ReleaseOnExit on_fail{release};  // AFTER ctx: releases before the manager joins

    const std::string png = small_png_bytes();
    std::vector<std::string> ids;
    for (int i = 0; i < 8; ++i) {
        INFO("create job " << i);
        auto res = http_post_job(ctx.port(), ctx.token(), {{"a.png", png}});
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 200);
        ids.push_back(json::parse(res->body)["job_id"].get<std::string>());
    }

    // 8 outstanding: the 9th is refused before it is queued.
    auto res = http_post_job(ctx.port(), ctx.token(), {{"a.png", png}});
    REQUIRE(res != nullptr);
    REQUIRE(res->status == 429);
    REQUIRE(json::parse(res->body)["error"]["code"].get<std::string>() == "too_many_pending");

    release.set_value();
    const auto arr = wait_list_all_terminal(ctx.port(), ctx.token(), ids);
    REQUIRE(arr.size() == 8);  // the refused POST never became a job
}

TEST_CASE("happy path: upload a marked image, poll to done, download the cleaned file",
          "[gui][gui-server]") {
    ServerCtx ctx;  // the real engine processor (no injection)
    auto res = http_post_job(ctx.port(), ctx.token(), {{"marked.png", encode_png(marked_mat())}});
    REQUIRE(res != nullptr);
    REQUIRE(res->status == 200);
    const std::string id = json::parse(res->body)["job_id"].get<std::string>();

    const auto job = wait_job_terminal(ctx.port(), ctx.token(), id);
    REQUIRE(job["status"].get<std::string>() == "done");
    REQUIRE(job["files"].size() == 1);
    REQUIRE(job["files"][0]["outcome"].get<std::string>() == "removed");

    auto dl = http_get(ctx.port(),
                       "/" + ctx.token() + "/api/jobs/" + id + "/files/0/image?kind=cleaned");
    REQUIRE(dl != nullptr);
    REQUIRE(dl->status == 200);
    const cv::Mat img = cv::imdecode(
        cv::Mat(1, (int)dl->body.size(), CV_8UC1, (void*)dl->body.data()), cv::IMREAD_COLOR);
    REQUIRE(!img.empty());
    REQUIRE(img.cols == 500);
    REQUIRE(img.rows == 500);
}

TEST_CASE("downloaded cleaned bytes equal the in-process still-remove output",
          "[gui][gui-server]") {
    ServerCtx ctx;
    TempDir scratch;
    const std::string png = encode_png(marked_mat());

    auto res = http_post_job(ctx.port(), ctx.token(), {{"marked.png", png}});
    REQUIRE(res != nullptr);
    REQUIRE(res->status == 200);
    const std::string id = json::parse(res->body)["job_id"].get<std::string>();
    const auto job = wait_job_terminal(ctx.port(), ctx.token(), id);
    REQUIRE(job["files"][0]["outcome"].get<std::string>() == "removed");

    auto dl = http_get(ctx.port(),
                       "/" + ctx.token() + "/api/jobs/" + id + "/files/0/image?kind=cleaned");
    REQUIRE(dl != nullptr);
    REQUIRE(dl->status == 200);

    // The reference run mirrors JobManager::process_file exactly: decode the
    // SAME uploaded bytes, the CLI-verbatim StillRemoveOptions (denoise off),
    // then the shared write tail (PNG, provenance strip).
    cv::Mat img = cv::imdecode(cv::Mat(1, (int)png.size(), CV_8UC1, (void*)png.data()),
                               cv::IMREAD_COLOR);
    WatermarkEngine engine;
    StillRemoveOptions sro;
    const StillRemoveOutcome r = remove_still(engine, img, sro);
    REQUIRE(r.outcome == StillOutcome::Removed);
    const fs::path ref_path = scratch.p / "ref.png";
    REQUIRE(write_still_output(ref_path, img, /*keep_provenance=*/false));
    REQUIRE(dl->body == read_file_bytes(ref_path));
}

TEST_CASE("no-watermark outcome and a sniff-rejected upload row", "[gui][gui-server]") {
    ServerCtx ctx;
    auto res = http_post_job(ctx.port(), ctx.token(),
                             {{"clean.png", encode_png(busy_mat())},
                              {"notes.txt", "this is not an image at all"}});
    REQUIRE(res != nullptr);
    REQUIRE(res->status == 200);  // one bad file never sinks the POST
    const std::string id = json::parse(res->body)["job_id"].get<std::string>();

    const auto job = wait_job_terminal(ctx.port(), ctx.token(), id);
    REQUIRE(job["status"].get<std::string>() == "done");
    REQUIRE(job["files"].size() == 2);
    REQUIRE(job["files"][0]["outcome"].get<std::string>() == "no-watermark");
    REQUIRE(job["files"][1]["outcome"].get<std::string>() == "failed");
    REQUIRE(job["files"][1]["error"].get<std::string>().find("unsupported") !=
            std::string::npos);

    // No cleaned artifact exists for a no-watermark file...
    auto cleaned = http_get(
        ctx.port(), "/" + ctx.token() + "/api/jobs/" + id + "/files/0/image?kind=cleaned");
    REQUIRE(cleaned != nullptr);
    REQUIRE(cleaned->status == 404);
    // ...and no original either for the sniff-rejected row (nothing persisted).
    auto orig = http_get(
        ctx.port(), "/" + ctx.token() + "/api/jobs/" + id + "/files/1/image?kind=original");
    REQUIRE(orig != nullptr);
    REQUIRE(orig->status == 404);
}

TEST_CASE("cancel a queued job: immediate cancel, files never run", "[gui][gui-server]") {
    std::promise<void> first_entered;
    std::promise<void> release;
    const auto go = release.get_future().share();
    ServerCtx ctx;
    std::atomic<int> calls{0};
    ctx.manager().set_processor_for_tests(
        [&first_entered, &calls, go](GuiFile& f, const JobOptions&, const fs::path&) {
            if (calls.fetch_add(1) == 0) {  // block only inside job 1's single file
                first_entered.set_value();
                go.wait();
            }
            f.outcome = FileOutcome::Removed;
            f.has_clean = true;
            return f;
        });
    ReleaseOnExit on_fail{release};

    const std::string png = small_png_bytes();
    auto r1 = http_post_job(ctx.port(), ctx.token(), {{"a.png", png}});
    REQUIRE(r1 != nullptr);
    REQUIRE(r1->status == 200);
    const std::string id1 = json::parse(r1->body)["job_id"].get<std::string>();
    // Bounded: a plain .wait() would hang the whole binary if the worker
    // never entered the processor.
    REQUIRE(first_entered.get_future().wait_for(std::chrono::seconds(10)) ==
            std::future_status::ready);  // the worker is inside job 1's file

    auto r2 = http_post_job(ctx.port(), ctx.token(), {{"b.png", png}});
    REQUIRE(r2 != nullptr);
    REQUIRE(r2->status == 200);
    const std::string id2 = json::parse(r2->body)["job_id"].get<std::string>();

    auto cancel = http_post(ctx.port(), "/" + ctx.token() + "/api/jobs/" + id2 + "/cancel");
    REQUIRE(cancel != nullptr);
    REQUIRE(cancel->status == 200);
    const auto cancelled = json::parse(cancel->body);
    REQUIRE(cancelled["status"].get<std::string>() == "cancelled");  // immediate
    REQUIRE(cancelled["files"][0]["outcome"].get<std::string>() == "not-run");

    release.set_value();
    REQUIRE(wait_job_terminal(ctx.port(), ctx.token(), id1)["status"].get<std::string>() ==
            "done");
    // The cancelled job never ran, even after the worker drained the queue.
    auto after = http_get(ctx.port(), "/" + ctx.token() + "/api/jobs/" + id2);
    REQUIRE(after != nullptr);
    const auto j2 = json::parse(after->body);
    REQUIRE(j2["status"].get<std::string>() == "cancelled");
    REQUIRE(j2["files"][0]["outcome"].get<std::string>() == "not-run");
}

TEST_CASE("cancel a running job: completed file kept and downloadable", "[gui][gui-server]") {
    std::promise<void> f0_in;
    std::promise<void> release;
    const auto go = release.get_future().share();
    ServerCtx ctx;
    const std::string tiny = encode_png(cv::Mat(4, 4, CV_8UC3, cv::Scalar(7, 8, 9)));
    ctx.manager().set_processor_for_tests(
        [&f0_in, go, tiny](GuiFile& f, const JobOptions&, const fs::path& job_dir) {
            f.outcome = FileOutcome::Removed;
            f.has_clean = true;
            // A real artifact on disk: the download route streams from the path.
            std::error_code ec;
            fs::create_directories(job_dir / "out", ec);
            {
                std::ofstream out(job_dir / "out" /
                                      (std::to_string(f.index) + "_clean." + f.orig_ext),
                                  std::ios::binary);
                out << tiny;
            }
            if (f.index == 0) {
                f0_in.set_value();
                go.wait();  // the cancel lands while file 0 is mid-flight
            }
            return f;
        });
    ReleaseOnExit on_fail{release};

    auto res = http_post_job(ctx.port(), ctx.token(),
                             {{"a.png", small_png_bytes()}, {"b.png", small_png_bytes()}});
    REQUIRE(res != nullptr);
    REQUIRE(res->status == 200);
    const std::string id = json::parse(res->body)["job_id"].get<std::string>();
    REQUIRE(f0_in.get_future().wait_for(std::chrono::seconds(10)) ==
            std::future_status::ready);  // the worker is inside file 0

    auto cancel = http_post(ctx.port(), "/" + ctx.token() + "/api/jobs/" + id + "/cancel");
    REQUIRE(cancel != nullptr);
    REQUIRE(cancel->status == 200);  // flag taken; the job is still running here
    REQUIRE(json::parse(cancel->body)["status"].get<std::string>() == "running");

    release.set_value();
    const auto job = wait_job_terminal(ctx.port(), ctx.token(), id);
    REQUIRE(job["status"].get<std::string>() == "cancelled");
    REQUIRE(job["files"][0]["outcome"].get<std::string>() == "removed");  // kept...
    REQUIRE(job["files"][1]["outcome"].get<std::string>() == "not-run");  // never reached

    // ...and the completed file's artifact is downloadable; the not-run file's is not.
    auto dl0 = http_get(
        ctx.port(), "/" + ctx.token() + "/api/jobs/" + id + "/files/0/image?kind=cleaned");
    REQUIRE(dl0 != nullptr);
    REQUIRE(dl0->status == 200);
    REQUIRE(dl0->body == tiny);
    auto dl1 = http_get(
        ctx.port(), "/" + ctx.token() + "/api/jobs/" + id + "/files/1/image?kind=cleaned");
    REQUIRE(dl1 != nullptr);
    REQUIRE(dl1->status == 404);
}

TEST_CASE("invalid options combination (legacy + rect) is 400 invalid_combination",
          "[gui][gui-server]") {
    ServerCtx ctx;
    auto res = http_post_job(ctx.port(), ctx.token(), {{"a.png", small_png_bytes()}},
                             R"({"legacy":true,"rect":[16,16,48,48]})");
    REQUIRE(res != nullptr);
    REQUIRE(res->status == 400);
    REQUIRE(json::parse(res->body)["error"]["code"].get<std::string>() ==
            "invalid_combination");
}

TEST_CASE("unknown job id is 404 unknown_job", "[gui][gui-server]") {
    ServerCtx ctx;
    auto get = http_get(ctx.port(), "/" + ctx.token() + "/api/jobs/deadbeefdead");
    REQUIRE(get != nullptr);
    REQUIRE(get->status == 404);
    REQUIRE(json::parse(get->body)["error"]["code"].get<std::string>() == "unknown_job");

    auto cancel = http_post(ctx.port(), "/" + ctx.token() + "/api/jobs/deadbeefdead/cancel");
    REQUIRE(cancel != nullptr);
    REQUIRE(cancel->status == 404);
    REQUIRE(json::parse(cancel->body)["error"]["code"].get<std::string>() == "unknown_job");
}

TEST_CASE("job list returns this run's jobs newest first", "[gui][gui-server]") {
    ServerCtx ctx;
    ctx.manager().set_processor_for_tests([](GuiFile& f, const JobOptions&, const fs::path&) {
        f.outcome = FileOutcome::Removed;  // instant: no disk work needed here
        f.has_clean = true;
        return f;
    });
    const std::string png = small_png_bytes();
    std::vector<std::string> ids;
    for (const char* name : {"a.png", "b.png", "c.png"}) {
        auto res = http_post_job(ctx.port(), ctx.token(), {{name, png}});
        REQUIRE(res != nullptr);
        REQUIRE(res->status == 200);
        ids.push_back(json::parse(res->body)["job_id"].get<std::string>());
    }
    const auto arr = wait_list_all_terminal(ctx.port(), ctx.token(), ids);
    REQUIRE(arr.size() == 3);
    REQUIRE(arr[0]["job_id"].get<std::string>() == ids[2]);  // newest first
    REQUIRE(arr[1]["job_id"].get<std::string>() == ids[1]);
    REQUIRE(arr[2]["job_id"].get<std::string>() == ids[0]);
    REQUIRE(arr[0].contains("created"));
    REQUIRE(arr[0]["file_count"].get<int>() == 1);
}

TEST_CASE("a pinned port already in use fails the run with rc 1", "[gui][gui-server]") {
    // A scratch listener holds the port (bound is enough: GuiServer's socket
    // sets only SO_REUSEADDR, so the bind cannot share an occupied port).
    httplib::Server scratch;
    const int port = scratch.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);

    GuiServer srv;
    const int rc = srv.run(GuiServerConfig{port, false, ""},
                           [](httplib::Server&, const std::string&) {});
    REQUIRE(rc == 1);  // stderr "port in use" path; no banner, no listen
}
