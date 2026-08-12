#include "metadata/provenance.hpp"

#include "metadata/jpeg_markers.hpp"
#include "metadata/png_chunks.hpp"
#include "metadata/provenance_constants.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace wmr::provenance {
namespace {

// ---------- byte / view helpers ----------

bool starts_with_sv(std::span<const std::byte> hay, std::string_view needle) {
    if (hay.size() < needle.size()) return false;
    const auto* h = reinterpret_cast<const unsigned char*>(hay.data());
    for (std::size_t i = 0; i < needle.size(); ++i)
        if (h[i] != static_cast<unsigned char>(needle[i])) return false;
    return true;
}

bool eq_sv_at(std::span<const std::byte> s, std::size_t off,
              std::string_view sv) {
    if (off + sv.size() > s.size()) return false;
    const auto* h = reinterpret_cast<const unsigned char*>(s.data()) + off;
    for (std::size_t i = 0; i < sv.size(); ++i)
        if (h[i] != static_cast<unsigned char>(sv[i])) return false;
    return true;
}

std::string ascii_lower_span(std::span<const std::byte> s) {
    std::string out;
    out.reserve(s.size());
    const auto* p = reinterpret_cast<const unsigned char*>(s.data());
    for (std::size_t i = 0; i < s.size(); ++i) {
        unsigned char c = p[i];
        if (c >= 'A' && c <= 'Z') c = static_cast<unsigned char>(c - 'A' + 'a');
        out.push_back(static_cast<char>(c));
    }
    return out;
}

// Best-effort issuer/tool note (no JUMBF/CBOR parser in v1). Informational and
// clearly labeled as best-effort by the caller.
std::optional<std::string> best_effort_c2pa_note(std::span<const std::byte> in) {
    static constexpr std::string_view kNames[] = {
        "Microsoft",   "Adobe",        "OpenAI",       "Google",
        "Stability AI", "Black Forest Labs", "GPT-4o",  "ChatGPT",
        "Sora",        "DALL-E",       "Imagen",       "Firefly"};
    const std::string hay = ascii_lower_span(in);
    std::string note;
    for (std::string_view name : kNames) {
        const std::string low = ascii_lower_span(
            std::span<const std::byte>(
                reinterpret_cast<const std::byte*>(name.data()), name.size()));
        if (hay.find(low) != std::string::npos) {
            if (!note.empty()) note += ", ";
            note += std::string(name);
        }
    }
    if (note.empty()) return std::nullopt;
    return note;
}

bool is_text_chunk(std::string_view t) {
    return t == kPngText || t == kPngItxt || t == kPngZtxt;
}

// ---------- file IO ----------

std::optional<std::vector<std::byte>> read_file_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    const std::streamoff sz = f.tellg();
    if (sz < 0) return std::nullopt;
    f.seekg(0, std::ios::beg);
    std::vector<std::byte> buf(static_cast<std::size_t>(sz));
    if (sz > 0) {
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        if (!f) return std::nullopt;
    }
    return buf;
}

bool write_file_bytes(const std::filesystem::path& path,
                      std::span<const std::byte> data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) return false;
    if (!data.empty())
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    return static_cast<bool>(f);
}

} // namespace

ContainerFormat sniff_format(std::span<const std::byte> in) {
    if (in.size() >= 8) {
        bool png = true;
        for (int i = 0; i < 8; ++i)
            if (in[i] != kPngSig[i]) {
                png = false;
                break;
            }
        if (png) return ContainerFormat::Png;
    }
    if (in.size() >= 2 && in[0] == kJpegSoi[0] && in[1] == kJpegSoi[1])
        return ContainerFormat::Jpeg;
    if (in.size() >= 12) {
        if (starts_with_sv(in, kWebpRiff) && eq_sv_at(in, 8, kWebpTag))
            return ContainerFormat::WebPRiff;
    }
    if (in.size() >= 8) {
        if (eq_sv_at(in, 4, kIsobmffFtyp)) return ContainerFormat::IsobmffImage;
    }
    return ContainerFormat::Unknown;
}

MetadataReport report_provenance(std::span<const std::byte> in) {
    MetadataReport rep;
    rep.format = sniff_format(in);

    switch (rep.format) {
        case ContainerFormat::Png: {
            rep.supported = true;
            PngReport pr = scan_png(in);
            rep.has_c2pa = pr.has_c2pa;
            rep.c2pa_note = best_effort_c2pa_note(in);
            for (const PngFinding& f : pr.findings) {
                ProvenanceFinding pf;
                pf.format = rep.format;
                pf.where = "PNG:" + f.chunk_type;
                pf.detail = f.detail;
                rep.findings.push_back(std::move(pf));
                if (f.dropped && is_text_chunk(f.chunk_type)) ++rep.ai_text_keys;
            }
            break;
        }
        case ContainerFormat::Jpeg: {
            rep.supported = true;
            JpegReport jr = scan_jpeg(in);
            rep.has_c2pa = jr.has_c2pa;
            rep.c2pa_note = best_effort_c2pa_note(in);
            for (const JpegFinding& f : jr.findings) {
                ProvenanceFinding pf;
                pf.format = rep.format;
                pf.where = "JPEG:" + f.marker;
                pf.detail = f.detail;
                rep.findings.push_back(std::move(pf));
            }
            break;
        }
        default:
            rep.supported = false;
            break;
    }
    return rep;
}

StripResult strip_provenance(std::span<const std::byte> in,
                             const StripOptions& opts) {
    StripResult res;
    res.report = report_provenance(in); // canonical findings + has_c2pa + format

    switch (res.report.format) {
        case ContainerFormat::Png: {
            res.supported = true;
            PngRewriteResult pr = rewrite_png_strip_ai(in, opts.keep_standard);
            if (!pr.ok) {
                res.ok = false; // malformed => caller copies input unchanged
                return res;
            }
            res.ok = true;
            res.items_removed = pr.items_dropped;
            res.report.has_c2pa = res.report.has_c2pa || pr.report.has_c2pa;
            if (!opts.dry_run) res.out = std::move(pr.out);
            return res;
        }
        case ContainerFormat::Jpeg: {
            res.supported = true;
            JpegRewriteResult jr = rewrite_jpeg_strip_ai(in, opts.keep_standard);
            if (!jr.ok) {
                res.ok = false;
                return res;
            }
            res.ok = true;
            res.items_removed = jr.items_dropped;
            res.report.has_c2pa = res.report.has_c2pa || jr.report.has_c2pa;
            if (!opts.dry_run) res.out = std::move(jr.out);
            return res;
        }
        default:
            // Unsupported (WebP/ISOBMFF/Unknown): ok=true, supported=false so the
            // caller copies the input through unchanged.
            res.ok = true;
            res.supported = false;
            return res;
    }
}

PostWriteResult post_write_provenance_strip(const std::string& file_path,
                                            bool keep_standard) {
    PostWriteResult r;
    r.ok = true;
    r.rewritten = false;
    r.items_removed = 0;

    std::optional<std::vector<std::byte>> data_opt = read_file_bytes(file_path);
    if (!data_opt) {
        r.ok = false;
        r.format_note = "read error; file untouched";
        return r;
    }

    StripOptions opts;
    opts.keep_standard = keep_standard;
    opts.dry_run = false;
    const StripResult sr = strip_provenance(*data_opt, opts);

    // Malformed input: leave the original intact (the fail-safe). Not an error
    // for the post-write guarantee (the file is unchanged).
    if (!sr.ok) {
        r.format_note = "malformed; file untouched";
        return r;
    }
    if (!sr.supported) {
        r.format_note = "unsupported format; untouched";
        return r;
    }
    if (sr.items_removed <= 0) {
        r.format_note = "clean (no provenance found)";
        return r; // true no-op: no rewrite, no extra write I/O
    }

    const std::filesystem::path p(file_path);
    const std::filesystem::path dir = p.parent_path();
    const std::string tmp_name = p.filename().string() + ".wmrprov.tmp";
    const std::filesystem::path tmp =
        dir.empty() ? std::filesystem::path(tmp_name) : (dir / tmp_name);

    try {
        if (!write_file_bytes(tmp, sr.out)) {
            std::error_code ec;
            std::filesystem::remove(tmp, ec);
            r.ok = false;
            r.format_note = "temp write error; file untouched";
            return r;
        }
        // Atomic on POSIX when tmp and target share a filesystem (same dir).
        std::filesystem::rename(tmp, p);
    } catch (const std::exception&) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec); // best-effort temp cleanup
        r.ok = false;
        r.format_note = "rename error; file untouched";
        return r;
    }

    r.rewritten = true;
    r.items_removed = sr.items_removed;
    r.format_note = "stripped";
    return r;
}

} // namespace wmr::provenance
