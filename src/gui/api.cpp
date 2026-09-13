#include "gui/api.hpp"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "detection/still_geometry.hpp"  // kStillPresetNames
#include "gui/update.hpp"                // update_info (page's update notice)
#include "gui/image_sniff.hpp"

#ifndef APP_VERSION
#define APP_VERSION "0.0.0"
#endif

namespace wmr::gui {
namespace {

using nlohmann::json;

// Upload caps (spec "Upload and resource caps"). The 1 GiB request cap itself
// is GuiServer's set_payload_max_length; the queue cap and the aggregate
// storage gate live inside JobManager::create_job.
constexpr std::size_t kMaxFileBytes = 200ull * 1024 * 1024;  // exactly 200 MiB
constexpr int kJobMaxFiles = 100;
constexpr long long kJobMaxBytes = 1024ll * 1024 * 1024;  // 1 GiB per job

// The corrected 5-directive CSP (spec erratum 2026-09-12: without
// connect-src 'self' the fallback to default-src 'none' blocked every
// same-origin API call). Only the HTML document carries it; the CSP has no
// effect on js/css subresources fetched with their own types.
constexpr const char* kCsp =
    "default-src 'none'; img-src 'self'; script-src 'self'; style-src 'self'; "
    "connect-src 'self'";

const char* status_str(JobStatus s) {
    switch (s) {
        case JobStatus::Queued: return "queued";
        case JobStatus::Running: return "running";
        case JobStatus::Done: return "done";
        case JobStatus::Failed: return "failed";
        case JobStatus::Cancelled: return "cancelled";
    }
    return "queued";
}

const char* outcome_str(FileOutcome o) {
    switch (o) {
        case FileOutcome::Pending: return "pending";
        case FileOutcome::Removed: return "removed";
        case FileOutcome::NoWatermark: return "no-watermark";
        case FileOutcome::Failed: return "failed";
        case FileOutcome::NotRun: return "not-run";
    }
    return "pending";
}

// The one error envelope: {"error":{"code":"<stable_snake_case>","message":"..."}}
void send_error(httplib::Response& res, int status, const std::string& code,
                const std::string& message) {
    res.status = status;
    res.set_content(json{{"error", {{"code", code}, {"message", message}}}}.dump(),
                    "application/json");
}

// bbox/score/geometry_source (and variant/error) are null until they exist:
// spec pins them null for pending, no-watermark, failed, and not-run files.
json file_json(const GuiFile& f) {
    json j;
    j["index"] = f.index;
    j["name"] = f.name;
    j["outcome"] = outcome_str(f.outcome);
    j["forced"] = f.forced;  // the UI's "removed (forced)" badge (spec outcome table)
    if (f.bbox)
        j["bbox"] = json::array({f.bbox->x, f.bbox->y, f.bbox->width, f.bbox->height});
    else
        j["bbox"] = nullptr;
    j["score"] = f.outcome == FileOutcome::Removed ? json(f.score) : json(nullptr);
    j["geometry_source"] = f.geometry_source.empty() ? json(nullptr) : json(f.geometry_source);
    j["variant"] = f.variant.empty() ? json(nullptr) : json(f.variant);
    j["error"] = f.error.empty() ? json(nullptr) : json(f.error);
    return j;
}

json snapshot_json(const JobSnapshot& s) {
    json j;
    j["job_id"] = s.id;
    j["status"] = status_str(s.status);
    j["created_at"] = static_cast<std::int64_t>(s.created);
    j["queue_position"] = s.queue_position;
    j["current_file_index"] = s.current_file_index;
    j["files"] = json::array();
    for (const auto& f : s.files) j["files"].push_back(file_json(f));
    return j;
}

// A UI view that is not embedded (pre-Task-7 build state): plain 404.
void serve_view(httplib::Response& res, std::string_view bytes, std::string_view mime,
                bool with_csp) {
    if (bytes.empty()) {
        res.status = 404;
        return;
    }
    res.set_content(std::string(bytes), std::string(mime));
    if (with_csp) res.set_header("Content-Security-Policy", kCsp);
}

// The name with any trailing ".ext" removed ("photo.png" -> "photo";
// ".hidden" keeps its leading dot; no-dot names pass through).
std::string stem_of(const std::string& name) {
    const std::size_t dot = name.rfind('.');
    if (dot == std::string::npos || dot == 0) return name;
    return name.substr(0, dot);
}

// Download name: sanitized stem, then _2/_3/... on a same-stem collision with
// an EARLIER file in the same job, then _clean (cleaned only), then the
// server-side sniffed extension. Spec "Storage naming".
std::string download_name(const JobSnapshot& job, const GuiFile& f, bool cleaned) {
    const std::string stem = stem_of(sanitize_filename(f.name, f.index));
    int same = 0;
    for (const auto& other : job.files) {
        if (other.index == f.index) continue;
        if (other.index < f.index
            && stem_of(sanitize_filename(other.name, other.index)) == stem)
            ++same;
    }
    std::string out = stem;
    if (same > 0) out += "_" + std::to_string(same + 1);
    if (cleaned) out += "_clean";
    out += "." + f.orig_ext;
    return out;
}

}  // namespace

std::string sanitize_filename(const std::string& name, int fallback_index) {
    // Baseline: basename only (both separators; the client controls the whole
    // string and never builds a path on our side, but the download name must
    // not carry one).
    std::size_t base = 0;
    for (std::size_t i = 0; i < name.size(); ++i)
        if (name[i] == '/' || name[i] == '\\') base = i + 1;

    std::string out;
    out.reserve(name.size() - base);
    for (std::size_t i = base; i < name.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        if (c < 0x20 || c == 0x7f) continue;  // control characters
        if (name[i] == '/' || name[i] == '\\' || name[i] == ':' || name[i] == '?'
            || name[i] == '*' || name[i] == '"' || name[i] == '\'' || name[i] == '<'
            || name[i] == '>' || name[i] == '|')
            continue;
        out += name[i];
    }

    // Windows reserved stems (CON, PRN, AUX, NUL, COM1-9, LPT1-9), matched on
    // the stem before the first dot, case-insensitively: suffixed "_".
    static const char* kReserved[] = {"CON",  "PRN",  "AUX",  "NUL",
                                      "COM1", "COM2", "COM3", "COM4", "COM5",
                                      "COM6", "COM7", "COM8", "COM9",
                                      "LPT1", "LPT2", "LPT3", "LPT4", "LPT5",
                                      "LPT6", "LPT7", "LPT8", "LPT9"};
    const std::size_t first_dot = out.find('.');
    const std::string stem = out.substr(0, first_dot);
    std::string upper;
    upper.reserve(stem.size());
    for (char ch : stem)
        upper += static_cast<char>(ch >= 'a' && ch <= 'z' ? ch - 'a' + 'A' : ch);
    for (const char* r : kReserved) {
        if (upper == r) {
            out.insert(first_dot == std::string::npos ? out.size() : first_dot, 1, '_');
            break;
        }
    }

    // Cap at 120 bytes without splitting a UTF-8 sequence.
    if (out.size() > 120) {
        std::size_t cut = 120;
        while (cut > 0 && (static_cast<unsigned char>(out[cut]) & 0xC0) == 0x80) --cut;
        out.resize(cut);
    }

    if (out.empty()) return "image" + std::to_string(fallback_index);
    return out;
}

std::string pct_encode(const std::string& s) {
    static const char* kHex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size());
    for (char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
            || c == '.' || c == '_' || c == '-') {
            out += ch;
        } else {
            out += '%';
            out += kHex[c >> 4];
            out += kHex[c & 0xF];
        }
    }
    return out;
}

std::string mime_for(const std::string& ext) {
    if (ext == "png") return "image/png";
    if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
    if (ext == "webp") return "image/webp";
    return "application/octet-stream";
}

void register_routes(httplib::Server& svr, const std::string& token,
                     JobManager& jobs, const ApiFeatures& features,
                     const EmbeddedUi& ui,
                     const std::function<void()>& request_shutdown) {
    const std::string p = "/" + token;

    // --- UI: three separate resources; 404 until Task 7 embeds them ---
    // ui/features are captured BY VALUE: callers may pass temporaries (a
    // route lambda must never dangle), and both are trivial copies (three
    // string_views into static asset bytes / one bool).
    svr.Get(p + "/", [ui](const httplib::Request&, httplib::Response& res) {
        serve_view(res, ui.html, EmbeddedUi::html_mime, /*with_csp=*/true);
    });
    svr.Get(p + "/app.js", [ui](const httplib::Request&, httplib::Response& res) {
        serve_view(res, ui.js, EmbeddedUi::js_mime, /*with_csp=*/false);
    });
    svr.Get(p + "/style.css", [ui](const httplib::Request&, httplib::Response& res) {
        serve_view(res, ui.css, EmbeddedUi::css_mime, /*with_csp=*/false);
    });

    svr.Get(p + "/api/version",
            [features](const httplib::Request&, httplib::Response& res) {
                json j;
                j["version"] = APP_VERSION;
                j["features"] = json{{"denoise_ai", features.denoise_ai}};
                j["presets"] = json::array();
                for (const char* name : kStillPresetNames)  // single source
                    j["presets"].push_back(name);
                // Update notice data (only when the check is compiled in,
                // not opted out, and was started by run_gui; the test
                // fixture never starts it, so tests see no update key and
                // touch no network).
                if (const UpdateInfo u = update_info(); u.enabled) {
                    j["update"] = json{{"known", u.known},
                                       {"newer", u.newer},
                                       {"current", u.current},
                                       {"latest", u.latest},
                                       {"url", UpdateInfo::url}};
                }
                res.set_content(j.dump(), "application/json");
            });

    svr.Get(p + "/api/jobs", [&jobs](const httplib::Request&, httplib::Response& res) {
        json arr = json::array();  // JobManager::list() is newest-first
        for (const auto& s : jobs.list())
            arr.push_back(json{{"job_id", s.id},
                               {"status", status_str(s.status)},
                               {"created", static_cast<std::int64_t>(s.created)},
                               {"file_count", s.file_count}});
        res.set_content(arr.dump(), "application/json");
    });

    svr.Post(p + "/api/jobs", [&jobs, features](const httplib::Request& req,
                                                httplib::Response& res) {
        if (!req.is_multipart_form_data()) {
            send_error(res, 415, "invalid_content_type",
                       "POST /api/jobs requires multipart/form-data");
            return;
        }

        // Per-file cap, enforced over the parsed multipart parts (naming the
        // offenders) before any sniffing or persistence. get_files returns
        // the parts BY VALUE (its own copy), so parts is non-const and the
        // bytes are MOVED into the PendingUpload below (peak ~= request +
        // this vector, per the spec's 2x budget).
        auto parts = req.form.get_files("files");  // send order
        std::string oversize;
        for (const auto& part : parts) {
            if (part.content.size() <= kMaxFileBytes) continue;
            if (!oversize.empty()) oversize += ", ";
            oversize += part.filename.empty() ? "(unnamed part)" : part.filename;
        }
        if (!oversize.empty()) {
            send_error(res, 413, "payload_too_large",
                       "file(s) exceed the 200 MiB per-file cap: " + oversize);
            return;
        }

        JobOptions opts;
        if (req.form.has_field("options")) {
            // A browser FormData string part carries no filename, so v0.56's
            // reader files it under fields, not files.
            std::string code, msg;
            if (!validate_options_part(req.form.get_field("options"), features, opts,
                                       code, msg)) {
                send_error(res, 400, code, msg);
                return;
            }
        }

        // Sniff each upload: rejects become terminal per-file failed rows via
        // create_job (one bad file never sinks the POST).
        std::vector<PendingUpload> files;
        std::vector<std::pair<std::string, std::string>> failed;
        int part_i = 0;
        for (auto& part : parts) {  // non-const: part.content is moved from below
            const std::string name =
                part.filename.empty() ? "image" + std::to_string(part_i) : part.filename;
            ++part_i;
            const SniffResult s = sniff_image(
                reinterpret_cast<const unsigned char*>(part.content.data()),
                part.content.size());
            if (s.too_large) {
                failed.emplace_back(
                    name, "image too large (" + std::to_string(s.width) + "x"
                              + std::to_string(s.height) + ")");
                continue;
            }
            if (s.format == SniffFormat::Unsupported) {  // includes 0-byte entries
                failed.emplace_back(name,
                                    "unsupported format: " + name + " (use PNG, JPEG, or WebP)");
                continue;
            }
            files.push_back(PendingUpload{name, s.ext, std::move(part.content)});
        }
        if (files.empty()) {
            send_error(res, 400, "no_supported_images", "no supported images");
            return;
        }

        std::string job_id;
        const auto err = jobs.create_job(files, failed, opts, kJobMaxFiles, kJobMaxBytes,
                                         job_id);
        if (err) {
            if (*err == "too_many_pending")
                send_error(res, 429, "too_many_pending",
                           "too many pending jobs; wait for one to finish");
            else if (*err == "payload_too_large")
                send_error(res, 413, "payload_too_large",
                           "job exceeds the 100-file / 1 GiB per-job cap or the 4 GB "
                           "storage bound");
            else  // "storage_error"
                send_error(res, 500, "storage_error", "could not persist the upload");
            return;
        }
        res.set_content(json{{"job_id", job_id}}.dump(), "application/json");
    });

    svr.Get(p + "/api/jobs/(\\w+)",
            [&jobs](const httplib::Request& req, httplib::Response& res) {
                auto snap = jobs.get(req.matches[1].str());
                if (!snap) {
                    send_error(res, 404, "unknown_job", "unknown job id");
                    return;
                }
                res.set_content(snapshot_json(*snap).dump(), "application/json");
            });

    svr.Post(p + "/api/jobs/(\\w+)/cancel",
             [&jobs](const httplib::Request& req, httplib::Response& res) {
                 const std::string id = req.matches[1].str();
                 std::optional<JobStatus> current;
                 std::string code;
                 if (jobs.cancel(id, current, code)) {
                     auto snap = jobs.get(id);  // fresh state post-cancel
                     if (snap) {
                         res.set_content(snapshot_json(*snap).dump(), "application/json");
                         return;
                     }
                     res.set_content(json{{"job_id", id}}.dump(), "application/json");
                     return;
                 }
                 if (code == "unknown_job") {
                     send_error(res, 404, "unknown_job", "unknown job id");
                     return;
                 }
                 // "job_already_finished": 409 with the current state.
                 send_error(res, 409, "job_already_finished",
                            std::string("job already finished (status: ")
                                + (current ? status_str(*current) : "unknown") + ")");
             });

    // The page's Quit button: answer first, then run the Ctrl-C-equivalent
    // graceful path (the callback stops the accept loop; the run-loop caller
    // owns the bounded cancel/join + session-dir cleanup tail). The response
    // is written on this request's own connection, which stop() does not
    // touch (it only closes the listen socket).
    svr.Post(p + "/api/shutdown",
             [request_shutdown](const httplib::Request&, httplib::Response& res) {
                 if (!request_shutdown) {
                     send_error(res, 503, "shutdown_unavailable",
                                "this server was not started with a shutdown hook");
                     return;
                 }
                 res.set_content(R"({"stopping":true})", "application/json");
                 request_shutdown();
             });

    svr.Get(p + "/api/jobs/(\\w+)/files/(\\d+)/image",
            [&jobs](const httplib::Request& req, httplib::Response& res) {
                const std::string id = req.matches[1].str();
                // {i}: from_chars only (never stoi); the route regex already
                // guarantees digits, so this only rejects int overflow.
                const std::string digits = req.matches[2].str();
                int index = 0;
                const auto parsed = std::from_chars(
                    digits.data(), digits.data() + digits.size(), index);
                if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size()) {
                    send_error(res, 404, "unknown_file", "no such file in this job");
                    return;
                }

                const std::string kind = req.get_param_value("kind");
                const bool cleaned = kind.empty() || kind == "cleaned";
                if (!cleaned && kind != "original") {
                    send_error(res, 400, "invalid_option", "kind must be cleaned or original");
                    return;
                }

                auto snap = jobs.get(id);
                if (!snap) {
                    send_error(res, 404, "unknown_job", "unknown job id");
                    return;
                }
                const GuiFile* f = nullptr;
                for (const auto& cand : snap->files)
                    if (cand.index == index) {
                        f = &cand;
                        break;
                    }
                // Paths resolve ONLY through the in-memory map: unknown
                // job/index, pending, no-watermark, or a missing orig all 404.
                auto path = jobs.file_artifact(id, index, cleaned);
                if (!f || !path) {
                    send_error(res, 404, "unknown_file", "no such file in this job");
                    return;
                }

                // Streams from disk (mmap content provider; a vanished file
                // turns into a plain 404 inside httplib).
                res.set_file_content(path->string(), mime_for(f->orig_ext));
                res.set_header("Content-Disposition",
                               "attachment; filename*=UTF-8''"
                                   + pct_encode(download_name(*snap, *f, cleaned)));
            });
}

}  // namespace wmr::gui
