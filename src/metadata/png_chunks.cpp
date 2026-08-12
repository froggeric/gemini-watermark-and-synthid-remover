#include "metadata/png_chunks.hpp"

#include "metadata/provenance_constants.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wmr::provenance {
namespace {

// ---------- byte / view helpers (no <algorithm>, no <utility>) ----------

std::uint32_t be32_at(std::span<const std::byte> s, std::size_t off) {
    return (std::to_integer<std::uint32_t>(s[off]) << 24) |
           (std::to_integer<std::uint32_t>(s[off + 1]) << 16) |
           (std::to_integer<std::uint32_t>(s[off + 2]) << 8) |
           (std::to_integer<std::uint32_t>(s[off + 3]));
}

// 4-byte chunk type as a string_view (caller bounds-checked off+4).
std::string_view type_sv(std::span<const std::byte> s, std::size_t off) {
    return std::string_view(reinterpret_cast<const char*>(s.data() + off), 4);
}

bool starts_with_sv(std::span<const std::byte> hay, std::string_view needle) {
    if (hay.size() < needle.size()) return false;
    const auto* h = reinterpret_cast<const unsigned char*>(hay.data());
    for (std::size_t i = 0; i < needle.size(); ++i)
        if (h[i] != static_cast<unsigned char>(needle[i])) return false;
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

bool contains_ai_substring(std::span<const std::byte> hay) {
    std::string low = ascii_lower_span(hay);
    for (std::string_view needle : kAiSubstrings)
        if (low.find(needle) != std::string::npos) return true;
    return false;
}

bool deny_key_match(std::string_view kw_lower) {
    for (std::string_view k : kAiDenyKeys)
        if (k == kw_lower) return true;
    return false;
}

// Keyword + value of a tEXt/iTXt/zTXt data range. valid=false (=> keep the
// chunk verbatim) when there is no NUL separator (a content anomaly; the
// fail-safe keeps what it cannot prove is AI).
struct TextParts {
    bool valid = false;
    std::string keyword_lower;
    std::span<const std::byte> value; // bytes after the first NUL
};
TextParts parse_text_chunk(std::span<const std::byte> data) {
    TextParts t;
    const std::byte nul{0};
    std::size_t i = 0;
    for (; i < data.size(); ++i)
        if (data[i] == nul) break;
    if (i >= data.size()) return t; // no NUL separator
    t.valid = true;
    t.keyword_lower = ascii_lower_span(data.subspan(0, i));
    t.value = data.subspan(i + 1);
    return t;
}

// ---------- per-chunk decision (single source for scan + rewrite) ----------

enum class Act { Keep, Drop };

struct Decision {
    Act act = Act::Keep;
    bool c2pa = false;
    bool record_finding = false;
    std::string detail;
};

Decision decide_chunk(std::string_view type, std::span<const std::byte> data,
                      bool keep_standard) {
    Decision d;

    if (type == kPngIhdr || type == kPngIdat || type == kPngIend) return d;

    // APNG structural chunks are always preserved, even in strip-all.
    if (type == "acTL" || type == "fcTL" || type == "fdAT") return d;

    // C2PA: caBX chunk, or any chunk whose payload begins with the JUMBF magic.
    const bool is_cabx = (type == kPngC2paChunk);
    const bool jumb_payload = starts_with_sv(data, kJumbfMagic);
    if (is_cabx || jumb_payload) {
        d.act = Act::Drop;
        d.c2pa = true;
        d.record_finding = true;
        d.detail = is_cabx ? "C2PA manifest (caBX chunk)"
                           : "C2PA manifest (jumb payload)";
        return d;
    }

    if (type == kPngText || type == kPngItxt) {
        TextParts tp = parse_text_chunk(data);
        if (!tp.valid) return d; // keep verbatim (content anomaly)
        if (deny_key_match(tp.keyword_lower)) {
            d.act = Act::Drop;
            d.record_finding = true;
            d.detail = "AI key: " + tp.keyword_lower;
            return d;
        }
        // Value scan (case-insensitive). A clean value is kept when
        // keep_standard, dropped as ancillary when strip-all.
        if (contains_ai_substring(tp.value)) {
            d.act = Act::Drop;
            d.record_finding = true;
            d.detail = "AI value in key: " + tp.keyword_lower;
            return d;
        }
        d.act = keep_standard ? Act::Keep : Act::Drop;
        return d;
    }

    if (type == kPngZtxt) {
        // v1 matches by KEY only (no zlib inflate). Phase 2 adds inflate.
        TextParts tp = parse_text_chunk(data);
        if (!tp.valid) return d; // keep verbatim (content anomaly)
        if (deny_key_match(tp.keyword_lower)) {
            d.act = Act::Drop;
            d.record_finding = true;
            d.detail = "AI key: " + tp.keyword_lower;
            return d;
        }
        d.act = keep_standard ? Act::Keep : Act::Drop;
        return d;
    }

    // Any other ancillary/unknown chunk: keep by default, drop in strip-all.
    d.act = keep_standard ? Act::Keep : Act::Drop;
    return d;
}

// ---------- chunk walk (single source for scan + rewrite) ----------

struct ChunkInfo {
    std::size_t header_off = 0; // offset of the 4-byte length field
    std::uint32_t data_len = 0;
    std::string_view type;
    std::span<const std::byte> data;
};

struct PngWalk {
    bool ok = false;
    std::vector<ChunkInfo> chunks;
    std::size_t iend_end = 0; // offset just past IEND's CRC (trailing bytes start)
};

PngWalk walk_png(std::span<const std::byte> in) {
    PngWalk r;
    if (in.size() < 8) return r;
    for (int i = 0; i < 8; ++i)
        if (in[i] != kPngSig[i]) return r;

    std::size_t off = 8;
    bool first = true;
    bool saw_iend = false;
    while (off < in.size()) {
        if (off + 8 > in.size()) return r; // truncated length+type header

        const std::uint32_t len = be32_at(in, off);
        // Single overflow-checked bound: off + (len + 12) <= size. len is
        // uint32 (max ~4.3e9); computed in 64-bit. NO clamp: an over-run is
        // malformed, so fail (caller copies input unchanged).
        const std::uint64_t need =
            static_cast<std::uint64_t>(off) + static_cast<std::uint64_t>(len) + 12u;
        if (need > in.size()) return r;

        const std::string_view type = type_sv(in, off + 4);
        if (first) {
            if (type != kPngIhdr) return r; // IHDR must be first
            first = false;
        }

        ChunkInfo ci;
        ci.header_off = off;
        ci.data_len = len;
        ci.type = type;
        ci.data = in.subspan(off + 8, len);
        r.chunks.push_back(ci);

        off += static_cast<std::size_t>(len) + 12u;

        if (type == kPngIend) {
            saw_iend = true;
            r.iend_end = off;
            break;
        }
    }
    if (!saw_iend) return r; // no IEND => malformed
    r.ok = true;
    return r;
}

void collect_finding(PngReport& rep, const ChunkInfo& c, const Decision& d) {
    if (!d.record_finding) return;
    PngFinding f;
    f.chunk_type = std::string(c.type);
    f.detail = d.detail;
    f.dropped = (d.act == Act::Drop);
    rep.findings.push_back(f);
}

} // namespace

PngReport scan_png(std::span<const std::byte> in) {
    PngReport rep;
    PngWalk w = walk_png(in);
    if (!w.ok) return rep; // ok stays false
    rep.ok = true;
    for (const ChunkInfo& c : w.chunks) {
        Decision d = decide_chunk(c.type, c.data, /*keep_standard=*/true);
        if (d.c2pa) rep.has_c2pa = true;
        collect_finding(rep, c, d);
    }
    return rep;
}

PngRewriteResult rewrite_png_strip_ai(std::span<const std::byte> in,
                                      bool keep_standard) {
    PngRewriteResult res;
    PngWalk w = walk_png(in);
    if (!w.ok) return res; // ok stays false

    res.report.ok = true;
    res.out.reserve(in.size());
    // Signature.
    res.out.insert(res.out.end(), in.begin(), in.begin() + 8);

    for (const ChunkInfo& c : w.chunks) {
        Decision d = decide_chunk(c.type, c.data, keep_standard);
        if (d.c2pa) res.report.has_c2pa = true;
        collect_finding(res.report, c, d);

        const std::size_t total = static_cast<std::size_t>(c.data_len) + 12u;
        if (d.act == Act::Keep) {
            res.out.insert(res.out.end(), in.begin() + c.header_off,
                           in.begin() + c.header_off + total);
        } else {
            ++res.items_dropped;
        }
    }

    // Trailing bytes after IEND (some tools append a trailer): preserve verbatim.
    if (w.iend_end < in.size())
        res.out.insert(res.out.end(), in.begin() + w.iend_end, in.end());

    res.ok = true;
    return res;
}

} // namespace wmr::provenance
