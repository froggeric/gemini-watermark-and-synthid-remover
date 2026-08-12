#include "metadata/jpeg_markers.hpp"

#include "metadata/provenance_constants.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wmr::provenance {
namespace {

// XMP / EXIF APP1 payload signatures (file-local; shared by scan + rewrite).
constexpr std::string_view kXmpNs = "http://ns.adobe.com/xap/";
// "Exif\0\0" (6 bytes, embedded NULs).
constexpr std::string_view kExifSig =
    std::string_view("Exif\0\0", 6);

// ---------- byte / view helpers (no <algorithm>, no <utility>) ----------

std::uint16_t be16_at(std::span<const std::byte> s, std::size_t off) {
    return static_cast<std::uint16_t>(
        (std::to_integer<std::uint32_t>(s[off]) << 8) |
        std::to_integer<std::uint32_t>(s[off + 1]));
}

bool starts_with_sv(std::span<const std::byte> hay, std::string_view needle) {
    if (hay.size() < needle.size()) return false;
    const auto* h = reinterpret_cast<const unsigned char*>(hay.data());
    for (std::size_t i = 0; i < needle.size(); ++i)
        if (h[i] != static_cast<unsigned char>(needle[i])) return false;
    return true;
}

bool starts_with_bytes(std::span<const std::byte> hay, const std::byte* arr,
                       std::size_t n) {
    if (hay.size() < n) return false;
    for (std::size_t i = 0; i < n; ++i)
        if (hay[i] != arr[i]) return false;
    return true;
}

bool bytes_eq_at(std::span<const std::byte> s, std::size_t off,
                 const std::byte* arr, std::size_t n) {
    if (off + n > s.size()) return false;
    for (std::size_t i = 0; i < n; ++i)
        if (s[off + i] != arr[i]) return false;
    return true;
}

bool contains_sv(std::span<const std::byte> hay, std::string_view needle) {
    const std::size_t n = hay.size();
    const std::size_t m = needle.size();
    if (m == 0) return true;
    if (n < m) return false;
    const auto* h = reinterpret_cast<const unsigned char*>(hay.data());
    for (std::size_t i = 0; i + m <= n; ++i) {
        std::size_t j = 0;
        for (; j < m; ++j)
            if (h[i + j] != static_cast<unsigned char>(needle[j])) break;
        if (j == m) return true;
    }
    return false;
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

// APP11 C2PA detection (informational; removal drops ALL APP11 regardless).
// True when the payload carries the JUMBF "JP" envelope with the C2PA type
// UUID at offset [24..40], or any "jumb" magic (content sniff).
bool app11_is_c2pa(std::span<const std::byte> p) {
    const bool ci_ok = p.size() >= 2 &&
                       p[0] == kJpegApp11EnvelopeCI[0] &&
                       p[1] == kJpegApp11EnvelopeCI[1];
    if (ci_ok) {
        if (bytes_eq_at(p, 24, kJpegApp11C2paMagic, 4)) return true;
        if (bytes_eq_at(p, 24, kC2paJumbfTypeUuid16, 16)) return true;
    }
    if (contains_sv(p, kJumbfMagic)) return true;
    return false;
}

std::string app_label(std::uint8_t m) {
    // m in 0xE0..0xEF -> "APP0".."APP15" (decimal of m-0xE0).
    unsigned n = static_cast<unsigned>(m - 0xE0);
    std::string s = "APP";
    if (n >= 10) {
        s.push_back(static_cast<char>('0' + (n / 10)));
        s.push_back(static_cast<char>('0' + (n % 10)));
    } else {
        s.push_back(static_cast<char>('0' + n));
    }
    return s;
}

// ---------- per-marker decision (single source for scan + rewrite) ----------

enum class Act { Keep, Drop };
enum class MarkerKind { Standalone, Lengthmarked };

struct MarkerInfo {
    std::size_t off = 0;   // offset of the 0xFF prefix actually emitted
    std::uint8_t marker = 0;
    MarkerKind kind = MarkerKind::Standalone;
    std::uint16_t seg_len = 0; // includes the 2 length bytes
    std::size_t total = 0;     // bytes consumed: 2 (standalone) or 2+seg_len
    std::span<const std::byte> payload; // length-marked: bytes after the 2-byte length
};

struct JDecision {
    Act act = Act::Keep;
    bool c2pa = false;
    bool record_finding = false;
    std::string marker_label;
    std::string detail;
};

JDecision decide_marker(const MarkerInfo& ji, bool keep_standard) {
    JDecision d;
    const std::uint8_t m = ji.marker;
    const std::span<const std::byte> p = ji.payload;

    // APP11 (0xEB): always dropped. C2PA flag set when the payload matches.
    if (m == 0xEB) {
        d.act = Act::Drop;
        d.marker_label = "APP11";
        if (app11_is_c2pa(p)) {
            d.c2pa = true;
            d.record_finding = true;
            d.detail = "C2PA manifest (APP11/JUMBF)";
        } else {
            d.record_finding = true;
            d.detail = "APP11 segment (dropped; JUMBF/C2PA/JPEG-XS)";
        }
        return d;
    }

    // strip-all: drop ALL APPn and COM (no per-item finding; bulk).
    if (!keep_standard && ((m >= 0xE0 && m <= 0xEF) || m == 0xFE)) {
        d.act = Act::Drop;
        return d;
    }

    if (m == 0xE1) { // APP1
        const bool is_xmp = starts_with_sv(p, kXmpNs);
        const bool is_exif = starts_with_sv(p, kExifSig);
        if (contains_ai_substring(p)) {
            d.act = Act::Drop;
            d.record_finding = true;
            d.marker_label = is_xmp ? "APP1/XMP" : (is_exif ? "APP1/EXIF" : "APP1");
            d.detail = is_xmp ? "AI marker in APP1/XMP"
                              : (is_exif ? "AI marker in APP1/EXIF (dropped whole)"
                                         : "AI marker in APP1");
            return d;
        }
        d.act = Act::Keep; // non-AI XMP / non-AI EXIF kept verbatim
        return d;
    }

    if (m == 0xED) { // APP13 IPTC
        if (contains_ai_substring(p)) {
            d.act = Act::Drop;
            d.record_finding = true;
            d.marker_label = "APP13";
            d.detail = "AI marker in APP13/IPTC";
            return d;
        }
        d.act = Act::Keep;
        return d;
    }

    if (m >= 0xE0 && m <= 0xEF) { // other APPn (APP0, APP2..APPA, APPC..APPF)
        if (contains_ai_substring(p)) {
            d.act = Act::Drop;
            d.record_finding = true;
            d.marker_label = app_label(m);
            d.detail = "AI marker in " + app_label(m);
            return d;
        }
        d.act = Act::Keep;
        return d;
    }

    if (m == 0xFE) { // COM
        if (contains_ai_substring(p)) {
            d.act = Act::Drop;
            d.record_finding = true;
            d.marker_label = "COM";
            d.detail = "AI marker in COM";
            return d;
        }
        d.act = Act::Keep;
        return d;
    }

    // DQT/DHT/SOF/DRI/etc.: always kept verbatim.
    return d;
}

// ---------- marker walk (single source for scan + rewrite) ----------

struct JpegWalk {
    bool ok = false;
    std::vector<MarkerInfo> markers; // pre-SOS markers (standalone + length-marked)
    std::size_t verbatim_off = 0;    // offset where verbatim copy begins (SOS/EOI 0xFF)
};

JpegWalk walk_jpeg(std::span<const std::byte> in) {
    JpegWalk r;
    if (in.size() < 2) return r;
    if (in[0] != kJpegSoi[0] || in[1] != kJpegSoi[1]) return r;

    std::size_t off = 2;
    while (true) {
        if (off + 2 > in.size()) return r; // truncated

        // Skip 0xFF fill bytes: JPEG allows multiple 0xFF padding bytes before
        // the real marker byte. They are semantically null; not emitted.
        while (off + 1 < in.size() && in[off] == std::byte{0xFF} &&
               in[off + 1] == std::byte{0xFF}) {
            ++off;
        }
        if (off + 2 > in.size()) return r; // truncated after fill
        if (in[off] != std::byte{0xFF}) return r;

        const std::uint8_t m = std::to_integer<std::uint8_t>(in[off + 1]);
        if (m == 0x00) return r; // 0xFF00 is entropy stuffing; unexpected pre-SOS

        if (m == 0xD9 || m == 0xDA) { // EOI or first SOS: copy remainder verbatim
            r.verbatim_off = off;
            r.ok = true;
            return r;
        }

        // Standalone (no length): TEM (0x01), RSTn (0xD0..0xD7), SOI (0xD8).
        if (m == 0x01 || m == 0xD8 || (m >= 0xD0 && m <= 0xD7)) {
            MarkerInfo ji;
            ji.off = off;
            ji.marker = m;
            ji.kind = MarkerKind::Standalone;
            ji.total = 2;
            r.markers.push_back(ji);
            off += 2;
            continue;
        }

        // Length-marked marker.
        if (off + 4 > in.size()) return r; // need marker(2) + length(2)
        const std::uint16_t seg = be16_at(in, off + 2);
        if (seg < 2) return r; // malformed length
        if (off + 2 + seg > in.size()) return r; // over-run => malformed (NO clamp)

        MarkerInfo ji;
        ji.off = off;
        ji.marker = m;
        ji.kind = MarkerKind::Lengthmarked;
        ji.seg_len = seg;
        ji.total = static_cast<std::size_t>(2) + seg;
        ji.payload = in.subspan(off + 4, static_cast<std::size_t>(seg) - 2);
        r.markers.push_back(ji);
        off += ji.total;
    }
}

void collect_finding(JpegReport& rep, const JDecision& d) {
    if (!d.record_finding) return;
    JpegFinding f;
    f.marker = d.marker_label;
    f.detail = d.detail;
    f.dropped = (d.act == Act::Drop);
    rep.findings.push_back(f);
}

} // namespace

JpegReport scan_jpeg(std::span<const std::byte> in) {
    JpegReport rep;
    JpegWalk w = walk_jpeg(in);
    if (!w.ok) return rep;
    rep.ok = true;
    for (const MarkerInfo& ji : w.markers) {
        if (ji.kind != MarkerKind::Lengthmarked) continue;
        JDecision d = decide_marker(ji, /*keep_standard=*/true);
        if (d.c2pa) rep.has_c2pa = true;
        collect_finding(rep, d);
    }
    return rep;
}

JpegRewriteResult rewrite_jpeg_strip_ai(std::span<const std::byte> in,
                                        bool keep_standard) {
    JpegRewriteResult res;
    JpegWalk w = walk_jpeg(in);
    if (!w.ok) return res;

    res.report.ok = true;
    res.out.reserve(in.size());
    // SOI.
    res.out.insert(res.out.end(), in.begin(), in.begin() + 2);

    for (const MarkerInfo& ji : w.markers) {
        JDecision d;
        if (ji.kind == MarkerKind::Standalone) {
            d.act = Act::Keep; // TEM/RSTn/SOI always kept
        } else {
            d = decide_marker(ji, keep_standard);
        }
        if (d.c2pa) res.report.has_c2pa = true;
        collect_finding(res.report, d);

        if (d.act == Act::Keep) {
            res.out.insert(res.out.end(), in.begin() + ji.off,
                           in.begin() + ji.off + ji.total);
        } else {
            ++res.items_dropped;
        }
    }

    // From the first SOS (or EOI) to end: copy VERBATIM. Covers progressive and
    // multi-scan JPEG (later DHT/SOS scans and all entropy stay byte-identical).
    if (w.verbatim_off < in.size())
        res.out.insert(res.out.end(), in.begin() + w.verbatim_off, in.end());

    res.ok = true;
    return res;
}

} // namespace wmr::provenance
