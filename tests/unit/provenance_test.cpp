// Unit tests for the provenance public API (wmr::provenance): format sniffing,
// the byte-scan reporter, dispatch routing, dry-run, the detection/removal
// PARITY invariant, and the bit-identical-pixels regression.
//
// Also contains the honesty-lock wording test [metadata][cli][wording], which
// asserts the user-facing provenance help text is factual (no over-claim of
// "removes C2PA", no em dashes) and advertises the --keep-provenance opt-out.
// The help string is the inline wmr::provenance_strip_help_text() defined in
// cli/cli_app.hpp (mirrors the synthid_attack_help_text pattern), so this test
// asserts on it directly without linking cli_app.cpp (which would drag in
// FFmpeg) and without spawning a subprocess.

#include <catch2/catch_test_macros.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "metadata/jpeg_markers.hpp"
#include "metadata/png_chunks.hpp"
#include "metadata/provenance.hpp"
#include "metadata/provenance_constants.hpp"

#include "cli/cli_app.hpp"  // wmr::provenance_strip_help_text() (inline; honesty lock)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using byte = std::byte;

std::vector<byte> str_to_bytes(std::string_view s) {
    const auto* p = reinterpret_cast<const byte*>(s.data());
    return std::vector<byte>(p, p + s.size());
}

bool contains(std::span<const byte> hay, std::string_view needle) {
    if (needle.empty()) return true;
    if (hay.size() < needle.size()) return false;
    const auto* h = reinterpret_cast<const unsigned char*>(hay.data());
    const std::size_t n = hay.size();
    const std::size_t m = needle.size();
    for (std::size_t i = 0; i + m <= n; ++i) {
        bool ok = true;
        for (std::size_t j = 0; j < m; ++j) {
            if (h[i + j] != static_cast<unsigned char>(needle[j])) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

std::string ascii_lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        out.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c);
    }
    return out;
}

// ---------------- PNG chunk helpers (CRC-32 + length/type/CRC) ----------------

std::uint32_t g_crc_table[256];
bool g_crc_ready = false;
void ensure_crc_table() {
    for (std::uint32_t i = 0; i < 256; ++i) {
        std::uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        g_crc_table[i] = c;
    }
    g_crc_ready = true;
}
std::uint32_t crc32_bytes(const byte* p, std::size_t n) {
    if (!g_crc_ready) ensure_crc_table();
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i)
        c = g_crc_table[(c ^ std::to_integer<std::uint8_t>(p[i])) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}
void append_u32_be(std::vector<byte>& v, std::uint32_t x) {
    v.push_back(byte(x >> 24));
    v.push_back(byte(x >> 16));
    v.push_back(byte(x >> 8));
    v.push_back(byte(x));
}
void append_png_chunk(std::vector<byte>& v, std::string_view type,
                      std::span<const byte> data) {
    REQUIRE(type.size() == 4);
    append_u32_be(v, static_cast<std::uint32_t>(data.size()));
    const std::size_t td_start = v.size();
    for (char c : type) v.push_back(byte(static_cast<unsigned char>(c)));
    v.insert(v.end(), data.begin(), data.end());
    append_u32_be(v, crc32_bytes(v.data() + td_start, 4 + data.size()));
}
std::vector<byte> text_data(std::string_view keyword, std::string_view value) {
    std::vector<byte> d;
    for (char c : keyword) d.push_back(byte(static_cast<unsigned char>(c)));
    d.push_back(byte{0});
    for (char c : value) d.push_back(byte(static_cast<unsigned char>(c)));
    return d;
}

// ---------------- JPEG marker helpers ----------------

std::uint16_t be16_at(std::span<const byte> s, std::size_t off) {
    return static_cast<std::uint16_t>(
        (std::to_integer<std::uint32_t>(s[off]) << 8) |
        std::to_integer<std::uint32_t>(s[off + 1]));
}
void append_jpeg_marker(std::vector<byte>& v, std::uint8_t marker,
                        std::span<const byte> payload) {
    REQUIRE(payload.size() <= 65533u);
    v.push_back(byte{0xFF});
    v.push_back(byte{marker});
    const std::uint16_t seg = static_cast<std::uint16_t>(2 + payload.size());
    v.push_back(byte(seg >> 8));
    v.push_back(byte(seg & 0xFF));
    v.insert(v.end(), payload.begin(), payload.end());
}
std::vector<byte> c2pa_app11_payload(std::uint32_t z, std::string_view extra) {
    std::vector<byte> p;
    p.push_back(byte{0x4A}); p.push_back(byte{0x50}); // CI "JP"
    p.push_back(byte{0x00}); p.push_back(byte{0x01}); // En
    p.push_back(byte((z >> 24) & 0xFF));
    p.push_back(byte((z >> 16) & 0xFF));
    p.push_back(byte((z >> 8) & 0xFF));
    p.push_back(byte(z & 0xFF));
    p.push_back(byte{0x00}); p.push_back(byte{0x00}); // LBox (jumb)
    p.push_back(byte{0x00}); p.push_back(byte{0x10});
    for (char c : std::string_view("jumb")) p.push_back(byte(static_cast<unsigned char>(c)));
    p.push_back(byte{0x00}); p.push_back(byte{0x00}); // LBox (jumd)
    p.push_back(byte{0x00}); p.push_back(byte{0x14});
    for (char c : std::string_view("jumd")) p.push_back(byte(static_cast<unsigned char>(c)));
    for (int i = 0; i < 16; ++i) p.push_back(wmr::provenance::kC2paJumbfTypeUuid16[i]);
    for (char c : extra) p.push_back(byte(static_cast<unsigned char>(c)));
    return p;
}

// ---------------- real, decodable images via OpenCV ----------------

cv::Mat make_img() {
    cv::Mat img(8, 8, CV_8UC3);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3b& px = img.at<cv::Vec3b>(y, x);
            px[0] = static_cast<uchar>(x * 30);
            px[1] = static_cast<uchar>(y * 30);
            px[2] = static_cast<uchar>((x + y) * 10);
        }
    return img;
}

std::vector<byte> encode_png() {
    std::vector<uchar> buf;
    REQUIRE(cv::imencode(".png", make_img(), buf));
    const auto* p = reinterpret_cast<const byte*>(buf.data());
    return std::vector<byte>(p, p + buf.size());
}
std::vector<byte> encode_jpeg() {
    std::vector<uchar> buf;
    REQUIRE(cv::imencode(".jpg", make_img(), buf));
    const auto* p = reinterpret_cast<const byte*>(buf.data());
    return std::vector<byte>(p, p + buf.size());
}

std::optional<std::size_t> find_png_chunk_off(std::span<const byte> png,
                                              std::string_view type) {
    if (png.size() < 8) return std::nullopt;
    std::size_t off = 8;
    while (off + 8 <= png.size()) {
        bool match = true;
        for (int i = 0; i < 4; ++i)
            if (std::to_integer<char>(png[off + 4 + i]) != type[i]) { match = false; break; }
        if (match) return off;
        const std::uint32_t len =
            (std::to_integer<std::uint32_t>(png[off]) << 24) |
            (std::to_integer<std::uint32_t>(png[off + 1]) << 16) |
            (std::to_integer<std::uint32_t>(png[off + 2]) << 8) |
            (std::to_integer<std::uint32_t>(png[off + 3]));
        off += static_cast<std::size_t>(len) + 12u;
    }
    return std::nullopt;
}
std::vector<byte> inject_before_idat(std::span<const byte> base,
                                     std::span<const byte> extra) {
    const auto idat = find_png_chunk_off(base, "IDAT");
    REQUIRE(idat.has_value());
    std::vector<byte> out(base.begin(), base.begin() + *idat);
    out.insert(out.end(), extra.begin(), extra.end());
    out.insert(out.end(), base.begin() + *idat, base.end());
    return out;
}
std::size_t after_jpeg_app0(std::span<const byte> j) {
    if (j.size() >= 4 && std::to_integer<std::uint8_t>(j[2]) == 0xFF &&
        std::to_integer<std::uint8_t>(j[3]) == 0xE0) {
        REQUIRE(j.size() >= 6);
        const std::uint16_t seg = be16_at(j, 4);
        return static_cast<std::size_t>(2) + 2 + seg;
    }
    return 2;
}
std::vector<byte> inject_after_app0(std::span<const byte> base,
                                    std::span<const byte> extra) {
    const std::size_t split = after_jpeg_app0(base);
    std::vector<byte> out(base.begin(), base.begin() + split);
    out.insert(out.end(), extra.begin(), extra.end());
    out.insert(out.end(), base.begin() + split, base.end());
    return out;
}

bool pixels_identical(std::span<const byte> a, std::span<const byte> b) {
    const auto* pa = reinterpret_cast<const uchar*>(a.data());
    const auto* pb = reinterpret_cast<const uchar*>(b.data());
    cv::Mat ma = cv::imdecode(cv::Mat(1, static_cast<int>(a.size()), CV_8U, const_cast<uchar*>(pa)), cv::IMREAD_UNCHANGED);
    cv::Mat mb = cv::imdecode(cv::Mat(1, static_cast<int>(b.size()), CV_8U, const_cast<uchar*>(pb)), cv::IMREAD_UNCHANGED);
    if (ma.empty() || mb.empty()) return false;
    if (ma.size() != mb.size() || ma.type() != mb.type()) return false;
    cv::Mat diff;
    cv::absdiff(ma, mb, diff);
    double mx = 0.0;
    cv::minMaxLoc(diff.reshape(1), nullptr, &mx);
    return mx == 0.0;
}

} // namespace

// =====================================================================================

TEST_CASE("provenance sniff and report", "[metadata]") {
    SECTION("sniff_format: PNG / JPEG / WebP / ISOBMFF / unknown") {
        using F = wmr::provenance::ContainerFormat;
        const std::vector<byte> png = encode_png();
        CHECK(wmr::provenance::sniff_format(png) == F::Png);

        const std::vector<byte> jpeg = encode_jpeg();
        CHECK(wmr::provenance::sniff_format(jpeg) == F::Jpeg);

        // WebP: "RIFF" + size(4) + "WEBP"
        std::vector<byte> webp = str_to_bytes("RIFF");
        webp.resize(8, byte{0}); // 4 bytes of "size"
        for (char c : std::string_view("WEBP")) webp.push_back(byte(static_cast<unsigned char>(c)));
        CHECK(wmr::provenance::sniff_format(webp) == F::WebPRiff);

        // ISOBMFF: 4 bytes size + "ftyp" at offset 4.
        std::vector<byte> iso(4, byte{0});
        for (char c : std::string_view("ftyp")) iso.push_back(byte(static_cast<unsigned char>(c)));
        for (int i = 0; i < 8; ++i) iso.push_back(byte{0});
        CHECK(wmr::provenance::sniff_format(iso) == F::IsobmffImage);

        const std::vector<byte> junk = str_to_bytes("just some random text bytes");
        CHECK(wmr::provenance::sniff_format(junk) == F::Unknown);

        // The 12-byte PNG signature prefix must still sniff as PNG, not be
        // confused by content that could resemble ftyp at offset 4.
        CHECK(wmr::provenance::sniff_format(std::span<const byte>(png.data(), 12)) == F::Png);
    }

    SECTION("report_provenance: C2PA + Google issuer + trainedAlgorithmicMedia => has_c2pa, note, IPTC finding") {
        // JPEG carrying: a real C2PA APP11 (with the C2PA UUID + "Google"), and
        // an APP13/IPTC segment carrying "trainedAlgorithmicMedia".
        const std::vector<byte> base = encode_jpeg();
        std::vector<byte> extra;
        append_jpeg_marker(extra, 0xEB, c2pa_app11_payload(1, "Google"));
        std::vector<byte> iptc = str_to_bytes("http://iptc.org/"); // a benign IPTC-ish prefix
        const std::vector<byte> ai = str_to_bytes(
            "<DigitalSourceType>trainedAlgorithmicMedia SENTINEL_REPORT_IPTC</DigitalSourceType>");
        iptc.insert(iptc.end(), ai.begin(), ai.end());
        append_jpeg_marker(extra, 0xED, iptc);
        const std::vector<byte> in = inject_after_app0(base, extra);

        const auto rep = wmr::provenance::report_provenance(in);
        REQUIRE(rep.format == wmr::provenance::ContainerFormat::Jpeg);
        CHECK(rep.supported);
        CHECK(rep.has_c2pa);
        // Best-effort issuer/tool note picked up "Google".
        REQUIRE(rep.c2pa_note.has_value());
        CHECK(contains(str_to_bytes(*rep.c2pa_note), "Google"));

        // The IPTC finding is reported (where starts with "JPEG:APP13").
        bool has_iptc_finding = false;
        for (const auto& f : rep.findings) {
            if (f.where.rfind("JPEG:APP13", 0) == 0) has_iptc_finding = true;
        }
        CHECK(has_iptc_finding);
    }
}

// -------------------------------------------------------------------------------------

TEST_CASE("provenance dispatch routing and dry-run", "[metadata]") {
    using F = wmr::provenance::ContainerFormat;

    SECTION("PNG and JPEG route to the right rewriter; WebP/Unknown pass through") {
        wmr::provenance::StripOptions opts; // keep_standard=true, dry_run=false

        // PNG with a caBX => supported, ok, stripped.
        const std::vector<byte> png_base = encode_png();
        std::vector<byte> png_extra;
        append_png_chunk(png_extra, "caBX", str_to_bytes("jumb SENTINEL_DISPATCH_PNG"));
        const std::vector<byte> png_in = inject_before_idat(png_base, png_extra);
        auto pr = wmr::provenance::strip_provenance(png_in, opts);
        CHECK(pr.report.format == F::Png);
        CHECK(pr.supported);
        CHECK(pr.ok);
        CHECK(pr.items_removed >= 1);
        CHECK_FALSE(contains(pr.out, "SENTINEL_DISPATCH_PNG"));

        // JPEG with an APP11/c2pa => supported, ok, stripped.
        const std::vector<byte> jpeg_base = encode_jpeg();
        std::vector<byte> jpeg_extra;
        append_jpeg_marker(jpeg_extra, 0xEB, c2pa_app11_payload(1, "SENTINEL_DISPATCH_JPEG"));
        const std::vector<byte> jpeg_in = inject_after_app0(jpeg_base, jpeg_extra);
        auto jr = wmr::provenance::strip_provenance(jpeg_in, opts);
        CHECK(jr.report.format == F::Jpeg);
        CHECK(jr.supported);
        CHECK(jr.ok);
        CHECK(jr.items_removed >= 1);
        CHECK_FALSE(contains(jr.out, "SENTINEL_DISPATCH_JPEG"));

        // WebP => unsupported, ok=true, supported=false (caller copies input).
        std::vector<byte> webp = str_to_bytes("RIFF");
        webp.resize(8, byte{0});
        for (char c : std::string_view("WEBP")) webp.push_back(byte(static_cast<unsigned char>(c)));
        auto wr = wmr::provenance::strip_provenance(webp, opts);
        CHECK(wr.report.format == F::WebPRiff);
        CHECK(wr.ok);
        CHECK_FALSE(wr.supported);
        CHECK(wr.out.empty());

        // Unknown => same pass-through contract.
        const std::vector<byte> junk = str_to_bytes("not an image at all");
        auto ur = wmr::provenance::strip_provenance(junk, opts);
        CHECK(ur.report.format == F::Unknown);
        CHECK(ur.ok);
        CHECK_FALSE(ur.supported);
        CHECK(ur.out.empty());
    }

    SECTION("dry-run produces an empty out but a full report") {
        const std::vector<byte> png_base = encode_png();
        std::vector<byte> extra;
        append_png_chunk(extra, "tEXt", text_data("parameters", "SENTINEL_DRYRUN_PARAMS"));
        append_png_chunk(extra, "caBX", str_to_bytes("jumb SENTINEL_DRYRUN_CABX"));
        const std::vector<byte> in = inject_before_idat(png_base, extra);

        wmr::provenance::StripOptions opts;
        opts.dry_run = true; // report only, no output bytes
        const auto r = wmr::provenance::strip_provenance(in, opts);
        REQUIRE(r.ok);
        REQUIRE(r.supported);
        // No output bytes are produced in dry-run.
        CHECK(r.out.empty());
        // But the report is populated and the would-be removals are counted.
        CHECK(r.items_removed >= 2);
        CHECK(r.report.has_c2pa);
        CHECK_FALSE(r.report.findings.empty());
    }
}

// -------------------------------------------------------------------------------------

TEST_CASE("detection-removal parity: every reported finding is removed", "[metadata]") {
    // The central invariant: report_provenance and strip_provenance draw on the
    // SAME constants (provenance_constants.hpp) and the SAME per-chunk/per-marker
    // decision, so the set of findings reported (with dropped=true) must equal
    // the set of items removed. Each fixture embeds a unique sentinel in the
    // droppable payload; after strip the sentinel must be gone, and the count of
    // reported findings must equal items_removed.
    wmr::provenance::StripOptions opts; // keep_standard=true

    const std::vector<byte> png_base = encode_png();
    const std::vector<byte> jpeg_base = encode_jpeg();

    struct Fixture {
        std::vector<byte> bytes;
        std::string sentinel; // must be ABSENT from the stripped output
        std::string label;
    };
    std::vector<Fixture> fixtures;

    // PNG: one deny key.
    {
        std::vector<byte> e;
        append_png_chunk(e, "tEXt", text_data("parameters", "SENTINEL_PAR_DENY"));
        fixtures.push_back({inject_before_idat(png_base, e), "SENTINEL_PAR_DENY", "PNG deny key"});
    }
    // PNG: AI substring in a keep-list key's value.
    {
        std::vector<byte> e;
        append_png_chunk(e, "tEXt", text_data("Comment", "made with stable-diffusion SENTINEL_PAR_SUBSTR"));
        fixtures.push_back({inject_before_idat(png_base, e), "SENTINEL_PAR_SUBSTR", "PNG AI value substring"});
    }
    // PNG: C2PA caBX chunk.
    {
        std::vector<byte> e;
        append_png_chunk(e, "caBX", str_to_bytes("jumb SENTINEL_PAR_CABX"));
        fixtures.push_back({inject_before_idat(png_base, e), "SENTINEL_PAR_CABX", "PNG caBX chunk"});
    }
    // PNG: jumb payload in a non-caBX chunk.
    {
        std::vector<byte> e;
        append_png_chunk(e, "prVn", str_to_bytes("jumb SENTINEL_PAR_JUMB"));
        fixtures.push_back({inject_before_idat(png_base, e), "SENTINEL_PAR_JUMB", "PNG jumb content sniff"});
    }
    // JPEG: C2PA APP11.
    {
        std::vector<byte> e;
        append_jpeg_marker(e, 0xEB, c2pa_app11_payload(1, "SENTINEL_PAR_APP11"));
        fixtures.push_back({inject_after_app0(jpeg_base, e), "SENTINEL_PAR_APP11", "JPEG APP11/C2PA"});
    }
    // JPEG: APP1/XMP-AI.
    {
        std::vector<byte> p = str_to_bytes("http://ns.adobe.com/xap/");
        const std::vector<byte> body = str_to_bytes("trainedAlgorithmicMedia SENTINEL_PAR_XMP");
        p.insert(p.end(), body.begin(), body.end());
        std::vector<byte> e;
        append_jpeg_marker(e, 0xE1, p);
        fixtures.push_back({inject_after_app0(jpeg_base, e), "SENTINEL_PAR_XMP", "JPEG APP1/XMP-AI"});
    }
    // JPEG: APP13/IPTC-AI.
    {
        std::vector<byte> e;
        append_jpeg_marker(e, 0xED, str_to_bytes("tc260:aigc SENTINEL_PAR_IPTC"));
        fixtures.push_back({inject_after_app0(jpeg_base, e), "SENTINEL_PAR_IPTC", "JPEG APP13/IPTC-AI"});
    }

    for (const auto& fx : fixtures) {
        const auto report = wmr::provenance::report_provenance(fx.bytes);
        const auto strip = wmr::provenance::strip_provenance(fx.bytes, opts);

        INFO("fixture: " << fx.label);
        REQUIRE(strip.ok);
        REQUIRE(strip.supported);
        // The reporter must have flagged something for this fixture.
        REQUIRE_FALSE(report.findings.empty());
        // Parity: the number of reported findings equals the number of items
        // removed (under keep_standard, every finding is a drop and every drop
        // has a finding).
        CHECK(report.findings.size() == static_cast<std::size_t>(strip.items_removed));
        // The flagged item is actually gone from the stripped output.
        const std::span<const byte> out(strip.out.data(), strip.out.size());
        CHECK_FALSE(contains(out, fx.sentinel));
    }
}

// -------------------------------------------------------------------------------------

TEST_CASE("bit-identical pixels regression across supported fixtures", "[metadata]") {
    const std::vector<byte> png_base = encode_png();
    const std::vector<byte> jpeg_base = encode_jpeg();

    SECTION("PNG: stripping AI metadata leaves pixels byte-identical") {
        std::vector<byte> extra;
        append_png_chunk(extra, "tEXt", text_data("parameters", "SENTINEL_PIX_PNG_PARAMS"));
        append_png_chunk(extra, "tEXt", text_data("Author", "SENTINEL_PIX_PNG_AUTHOR")); // kept
        append_png_chunk(extra, "caBX", str_to_bytes("jumb SENTINEL_PIX_PNG_CABX"));
        const std::vector<byte> in = inject_before_idat(png_base, extra);

        const auto r = wmr::provenance::strip_provenance(in, {});
        REQUIRE(r.ok);
        REQUIRE(r.supported);
        CHECK(pixels_identical(in, r.out));
    }

    SECTION("JPEG: stripping AI markers leaves pixels byte-identical") {
        std::vector<byte> extra;
        append_jpeg_marker(extra, 0xEB, c2pa_app11_payload(1, "SENTINEL_PIX_JPG_C2PA"));
        std::vector<byte> xmp = str_to_bytes("http://ns.adobe.com/xap/");
        const std::vector<byte> body = str_to_bytes("trainedAlgorithmicMedia SENTINEL_PIX_JPG_XMP");
        xmp.insert(xmp.end(), body.begin(), body.end());
        append_jpeg_marker(extra, 0xE1, xmp);
        const std::vector<byte> in = inject_after_app0(jpeg_base, extra);

        const auto r = wmr::provenance::strip_provenance(in, {});
        REQUIRE(r.ok);
        REQUIRE(r.supported);
        CHECK(pixels_identical(in, r.out));
    }
}

// -------------------------------------------------------------------------------------
// Honesty lock for the provenance / "wmr metadata" help text.
//
// Locks the user-facing description to: (a) carry the factual scope words
// ("lossless", "PNG", "JPEG", "provenance"); (b) NOT make the absolute over-
// claim "removes C2PA" (the v1 strip is lossless on pixels, container-level,
// and covers PNG + JPEG; the help must stay factual per DECISION B); (c) contain
// no em dashes (project prose rule); (d) advertise the --keep-provenance opt-out
// for the remove/synthid default strip. Mirrors the synthid_attack_cli_test.cpp
// pattern: asserts on the inline help function in cli/cli_app.hpp, no subprocess.

TEST_CASE("provenance help text is honest and factual", "[metadata][cli][wording]") {
    const std::string text = wmr::provenance_strip_help_text();

    // (a) The factual scope words MUST be present (case-insensitive).
    const std::string low = ascii_lower(text);
    for (std::string_view word : {"lossless", "png", "jpeg", "provenance"}) {
        INFO("help text must mention: " << word);
        CHECK(low.find(word) != std::string::npos);
    }

    // (b) The text MUST NOT make the absolute over-claim "removes C2PA".
    CHECK(low.find("removes c2pa") == std::string::npos);

    // (c) No em dashes (U+2014, UTF-8 0xE2 0x80 0x94) anywhere in the text.
    const std::string em_dash = "\xE2\x80\x94";
    CHECK(text.find(em_dash) == std::string::npos);

    // (d) The remove-path opt-out --keep-provenance is advertised.
    CHECK(low.find("--keep-provenance") != std::string::npos);
}
