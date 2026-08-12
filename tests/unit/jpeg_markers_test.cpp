// Unit tests for the JPEG marker-level provenance strip (wmr::provenance).
//
// All fixtures are SYNTHESIZED in code. A real, decodable JPEG is produced with
// cv::imencode; AI/C2PA APP markers are then spliced in between APP0 and the
// first DQT (i.e. pre-SOS, where the strip operates). The entropy-coded bytes
// from the first SOS onward are copied verbatim by the rewriter, so the bit-
// identical-pixels gate (cv::imdecode(in) == cv::imdecode(out)) holds and the
// entropy tail is byte-identical. The real C2PA APP11 envelope (ISO/IEC
// 19566-5: CI "JP" + En + Z + jumb/jumd + the C2PA JUMBF type UUID) is built
// exactly per the plan, so has_c2pa is exercised against the genuine layout.

#include <catch2/catch_test_macros.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "metadata/jpeg_markers.hpp"
#include "metadata/provenance_constants.hpp"

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

std::uint16_t be16_at(std::span<const byte> s, std::size_t off) {
    return static_cast<std::uint16_t>(
        (std::to_integer<std::uint32_t>(s[off]) << 8) |
        std::to_integer<std::uint32_t>(s[off + 1]));
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

// Append a length-marked APPn/COM marker: FF <marker> + be16(2+payload) + payload.
void append_marker(std::vector<byte>& v, std::uint8_t marker,
                   std::span<const byte> payload) {
    REQUIRE(payload.size() <= 65533u); // seg_len fits in 16 bits
    v.push_back(byte{0xFF});
    v.push_back(byte{marker});
    const std::uint16_t seg = static_cast<std::uint16_t>(2 + payload.size());
    v.push_back(byte(seg >> 8));
    v.push_back(byte(seg & 0xFF));
    v.insert(v.end(), payload.begin(), payload.end());
}

// Real C2PA APP11 payload per ISO/IEC 19566-5:
//   CI="JP" + En(2) + Z(4 BE) + LBox + "jumb" + LBox + "jumd" + kC2paJumbfTypeUuid16
// payload[24..28] is therefore "c2pa", which is what app11_is_c2pa() checks.
// `z` is the segment sequence number; `extra` is appended (used for the issuer
// note + a unique sentinel).
std::vector<byte> c2pa_app11_payload(std::uint32_t z, std::string_view extra) {
    std::vector<byte> p;
    p.push_back(byte{0x4A}); p.push_back(byte{0x50});              // CI "JP"
    p.push_back(byte{0x00}); p.push_back(byte{0x01});              // En
    p.push_back(byte((z >> 24) & 0xFF));                           // Z (BE)
    p.push_back(byte((z >> 16) & 0xFF));
    p.push_back(byte((z >> 8) & 0xFF));
    p.push_back(byte(z & 0xFF));
    // LBox for the jumb superbox (4 bytes). Pushed explicitly: a string_view
    // built from a NUL-containing literal would truncate at the first NUL.
    p.push_back(byte{0x00}); p.push_back(byte{0x00});
    p.push_back(byte{0x00}); p.push_back(byte{0x10});
    for (char c : std::string_view("jumb")) p.push_back(byte(static_cast<unsigned char>(c)));
    // LBox for the jumd description box (4 bytes).
    p.push_back(byte{0x00}); p.push_back(byte{0x00});
    p.push_back(byte{0x00}); p.push_back(byte{0x14});
    for (char c : std::string_view("jumd")) p.push_back(byte(static_cast<unsigned char>(c)));
    for (int i = 0; i < 16; ++i) p.push_back(wmr::provenance::kC2paJumbfTypeUuid16[i]);
    for (char c : extra) p.push_back(byte(static_cast<unsigned char>(c)));
    return p;
}

// ---------------- real, decodable JPEG via OpenCV ----------------

std::vector<byte> encode_jpeg(const cv::Mat& img) {
    std::vector<uchar> buf;
    REQUIRE(cv::imencode(".jpg", img, buf));
    const auto* p = reinterpret_cast<const byte*>(buf.data());
    return std::vector<byte>(p, p + buf.size());
}

// Offset just past the APP0/JFIF segment (SOI + APP0). If no APP0 is present,
// returns offset 2 (just past SOI) so injected markers still land pre-SOS.
std::size_t after_app0(std::span<const byte> j) {
    if (j.size() >= 4 && std::to_integer<std::uint8_t>(j[2]) == 0xFF &&
        std::to_integer<std::uint8_t>(j[3]) == 0xE0) {
        REQUIRE(j.size() >= 6);
        const std::uint16_t seg = be16_at(j, 4);
        // SOI occupies [0,2). APP0 starts at offset 2; APP0 total = 2 (marker) +
        // seg_len bytes, so the split point = 2 + 2 + seg.
        return static_cast<std::size_t>(2) + 2 + seg;
    }
    return 2;
}

// Offset of the 0xFF prefix of the first SOS (0xDA), scanning pre-SOS markers
// (skipping 0xFF fill bytes like the rewriter does).
std::optional<std::size_t> first_sos_off(std::span<const byte> j) {
    if (j.size() < 2) return std::nullopt;
    std::size_t off = 2;
    while (off + 2 <= j.size()) {
        while (off + 1 < j.size() &&
               std::to_integer<std::uint8_t>(j[off]) == 0xFF &&
               std::to_integer<std::uint8_t>(j[off + 1]) == 0xFF) {
            ++off;
        }
        if (off + 2 > j.size()) return std::nullopt;
        if (std::to_integer<std::uint8_t>(j[off]) != 0xFF) return std::nullopt;
        const std::uint8_t m = std::to_integer<std::uint8_t>(j[off + 1]);
        if (m == 0xDA) return off; // first SOS
        if (m == 0xD9) return std::nullopt; // EOI before any SOS
        if (m == 0x00) return std::nullopt; // unexpected stuffing pre-SOS
        if (m == 0x01 || m == 0xD8 || (m >= 0xD0 && m <= 0xD7)) {
            off += 2;
            continue;
        }
        if (off + 4 > j.size()) return std::nullopt;
        const std::uint16_t seg = be16_at(j, off + 2);
        off += static_cast<std::size_t>(2) + seg;
    }
    return std::nullopt;
}

// Splice `extra` markers right after APP0 (pre-SOS). The entropy tail is
// byte-for-byte the encoded original.
std::vector<byte> inject_after_app0(std::span<const byte> base,
                                    std::span<const byte> extra) {
    const std::size_t split = after_app0(base);
    std::vector<byte> out(base.begin(), base.begin() + split);
    out.insert(out.end(), extra.begin(), extra.end());
    out.insert(out.end(), base.begin() + split, base.end());
    return out;
}

bool pixels_identical(std::span<const byte> a, std::span<const byte> b) {
    const auto* pa = reinterpret_cast<const uchar*>(a.data());
    const auto* pb = reinterpret_cast<const uchar*>(b.data());
    cv::Mat ma = cv::imdecode(cv::Mat(1, static_cast<int>(a.size()), CV_8U, const_cast<uchar*>(pa)), cv::IMREAD_COLOR);
    cv::Mat mb = cv::imdecode(cv::Mat(1, static_cast<int>(b.size()), CV_8U, const_cast<uchar*>(pb)), cv::IMREAD_COLOR);
    if (ma.empty() || mb.empty()) return false;
    if (ma.size() != mb.size() || ma.type() != mb.type()) return false;
    cv::Mat diff;
    cv::absdiff(ma, mb, diff);
    // minMaxLoc needs a single-channel Mat; flatten the (possibly multi-channel)
    // diff. For an unsigned diff, max == 0 iff every pixel is identical.
    double mx = 0.0;
    cv::minMaxLoc(diff.reshape(1), nullptr, &mx);
    return mx == 0.0;
}

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

} // namespace

// =====================================================================================

TEST_CASE("JPEG provenance strip", "[metadata][jpeg]") {
    const cv::Mat img = make_img();
    const std::vector<byte> base = encode_jpeg(img);
    REQUIRE(first_sos_off(base).has_value()); // sanity: base has a SOS

    SECTION("SOI+APP0+APP11(real C2PA)+APP1/XMP-AI+APP13/IPTC-AI+SOS+entropy+EOI") {
        std::vector<byte> extra;
        // Real C2PA APP11 envelope; "Google" rides inside so the issuer note fires.
        append_marker(extra, 0xEB, c2pa_app11_payload(/*z=*/1, "Google SENTINEL_APP11_C2PA"));
        // APP1/XMP carrying an AI marker.
        std::vector<byte> xmp = str_to_bytes("http://ns.adobe.com/xap/");
        const std::vector<byte> xmp_body =
            str_to_bytes("<x:xmpmeta>trainedAlgorithmicMedia SENTINEL_XMP_AI</x:xmpmeta>");
        xmp.insert(xmp.end(), xmp_body.begin(), xmp_body.end());
        append_marker(extra, 0xE1, xmp);
        // APP13/IPTC carrying an AI marker.
        append_marker(extra, 0xED, str_to_bytes("tc260:aigc SENTINEL_IPTC_AI"));

        const std::vector<byte> in = inject_after_app0(base, extra);

        // The reporter must flag C2PA on the unstripped input.
        const auto pre_report = wmr::provenance::scan_jpeg(in);
        REQUIRE(pre_report.ok);
        CHECK(pre_report.has_c2pa);

        const auto r = wmr::provenance::rewrite_jpeg_strip_ai(in, /*keep_standard=*/true);
        REQUIRE(r.ok);
        const std::span<const byte> out(r.out.data(), r.out.size());

        // APP11, XMP-AI and IPTC-AI are dropped.
        CHECK_FALSE(contains(out, "SENTINEL_APP11_C2PA"));
        CHECK_FALSE(contains(out, "SENTINEL_XMP_AI"));
        CHECK_FALSE(contains(out, "SENTINEL_IPTC_AI"));
        // APP0/JFIF survives (the byte "JF" appears in "JFIF").
        CHECK(contains(out, "JFIF"));
        CHECK(r.report.has_c2pa);

        // Entropy tail (from the first SOS onward) is byte-identical.
        const auto sos_in = first_sos_off(in);
        const auto sos_out = first_sos_off(out);
        REQUIRE(sos_in.has_value());
        REQUIRE(sos_out.has_value());
        const std::size_t tail_in = in.size() - *sos_in;
        const std::size_t tail_out = out.size() - *sos_out;
        CHECK(tail_in == tail_out);
        bool tail_eq = (tail_in == tail_out);
        for (std::size_t i = 0; i < tail_in && tail_eq; ++i)
            if (in[*sos_in + i] != out[*sos_out + i]) tail_eq = false;
        CHECK(tail_eq);

        // Bit-identical pixels.
        CHECK(pixels_identical(in, out));
    }

    SECTION("multi-segment C2PA: two APP11 (same En, Z=1 and Z=2) both dropped") {
        std::vector<byte> extra;
        append_marker(extra, 0xEB, c2pa_app11_payload(/*z=*/1, "SENTINEL_APP11_Z1"));
        append_marker(extra, 0xEB, c2pa_app11_payload(/*z=*/2, "SENTINEL_APP11_Z2"));
        const std::vector<byte> in = inject_after_app0(base, extra);

        const auto r = wmr::provenance::rewrite_jpeg_strip_ai(in, true);
        REQUIRE(r.ok);
        CHECK(r.items_dropped >= 2); // both APP11 segments
        const std::span<const byte> out(r.out.data(), r.out.size());
        CHECK_FALSE(contains(out, "SENTINEL_APP11_Z1"));
        CHECK_FALSE(contains(out, "SENTINEL_APP11_Z2"));
        // No APP11 marker remains at all (0xFF 0xEB sequence gone from pre-SOS).
        bool found_app11 = false;
        for (std::size_t i = 0; i + 1 < r.out.size(); ++i) {
            if (std::to_integer<std::uint8_t>(r.out[i]) == 0xFF &&
                std::to_integer<std::uint8_t>(r.out[i + 1]) == 0xEB) {
                found_app11 = true;
                break;
            }
        }
        CHECK_FALSE(found_app11);
    }

    SECTION("APP1/EXIF with an AI marker is dropped whole; without it, kept verbatim") {
        // APP1 "Exif\0\0" + "Software=ComfyUI" -> dropped whole.
        std::vector<byte> exif_ai = str_to_bytes(std::string_view("Exif\0\0", 6));
        const std::vector<byte> ai_field =
            str_to_bytes("Software=ComfyUI SENTINEL_EXIF_AI");
        exif_ai.insert(exif_ai.end(), ai_field.begin(), ai_field.end());
        std::vector<byte> extra_ai;
        append_marker(extra_ai, 0xE1, exif_ai);
        const std::vector<byte> in_ai = inject_after_app0(base, extra_ai);
        const auto r_ai = wmr::provenance::rewrite_jpeg_strip_ai(in_ai, true);
        REQUIRE(r_ai.ok);
        CHECK(r_ai.items_dropped >= 1);
        const std::span<const byte> out_ai(r_ai.out.data(), r_ai.out.size());
        CHECK_FALSE(contains(out_ai, "SENTINEL_EXIF_AI"));

        // APP1 "Exif\0\0" + "Software=Adobe Photoshop" -> kept verbatim.
        std::vector<byte> exif_ok = str_to_bytes(std::string_view("Exif\0\0", 6));
        const std::vector<byte> clean_field =
            str_to_bytes("Software=Adobe Photoshop SENTINEL_EXIF_OK");
        exif_ok.insert(exif_ok.end(), clean_field.begin(), clean_field.end());
        std::vector<byte> extra_ok;
        append_marker(extra_ok, 0xE1, exif_ok);
        const std::vector<byte> in_ok = inject_after_app0(base, extra_ok);
        const auto r_ok = wmr::provenance::rewrite_jpeg_strip_ai(in_ok, true);
        REQUIRE(r_ok.ok);
        const std::span<const byte> out_ok(r_ok.out.data(), r_ok.out.size());
        CHECK(contains(out_ok, "SENTINEL_EXIF_OK"));
    }

    SECTION("progressive/multi-scan: bytes after the first SOS are byte-identical") {
        // Synthesize SOI + DQT + SOF2 + DHT + APP11/c2pa + SOS + <entropy with a
        // second SOS> + EOI. The rewriter drops APP11 and copies everything from
        // the first SOS verbatim (covering multi-scan progressive JPEG).
        std::vector<byte> syn;
        syn.push_back(byte{0xFF}); syn.push_back(byte{0xD8}); // SOI
        append_marker(syn, 0xDB, str_to_bytes("DQT-PAYLOAD-SENTINEL"));        // DQT
        append_marker(syn, 0xC2, str_to_bytes("SOF2-PROGRESSIVE-SENTINEL"));   // SOF2
        append_marker(syn, 0xC4, str_to_bytes("DHT-PAYLOAD-SENTINEL"));        // DHT
        append_marker(syn, 0xEB, c2pa_app11_payload(1, "SENTINEL_PROG_APP11")); // APP11/c2pa
        // First SOS marker + a small header, then entropy that itself contains a
        // second SOS (a later scan) and an EOI. All of this is verbatim-copied.
        append_marker(syn, 0xDA, str_to_bytes("SOS-HDR"));
        for (char c : std::string_view("\x11\x22\x33")) syn.push_back(byte(static_cast<unsigned char>(c)));
        syn.push_back(byte{0xFF}); syn.push_back(byte{0xDA}); // second scan SOS (entropy)
        for (char c : std::string_view("\x44\x55")) syn.push_back(byte(static_cast<unsigned char>(c)));
        syn.push_back(byte{0xFF}); syn.push_back(byte{0xD9}); // EOI

        const auto r = wmr::provenance::rewrite_jpeg_strip_ai(syn, true);
        REQUIRE(r.ok);
        const std::span<const byte> out(r.out.data(), r.out.size());
        // APP11 gone; DQT/SOF2/DHT kept.
        CHECK_FALSE(contains(out, "SENTINEL_PROG_APP11"));
        CHECK(contains(out, "DQT-PAYLOAD-SENTINEL"));
        CHECK(contains(out, "SOF2-PROGRESSIVE-SENTINEL"));
        CHECK(contains(out, "DHT-PAYLOAD-SENTINEL"));

        // Everything from the first SOS onward is byte-identical to the input.
        const auto sos_in = first_sos_off(syn);
        const auto sos_out = first_sos_off(out);
        REQUIRE(sos_in.has_value());
        REQUIRE(sos_out.has_value());
        const std::size_t tail_in = syn.size() - *sos_in;
        CHECK(tail_in == out.size() - *sos_out);
        bool tail_eq = true;
        for (std::size_t i = 0; i < tail_in && tail_eq; ++i)
            if (syn[*sos_in + i] != out[*sos_out + i]) tail_eq = false;
        CHECK(tail_eq);
    }

    SECTION("marker preceded by 0xFF fill bytes parses correctly") {
        // SOI, then 3 fill 0xFF bytes, then APP0 with a sentinel, then DQT + SOS
        // + entropy + EOI. The rewriter skips the fill and keeps APP0.
        std::vector<byte> syn;
        syn.push_back(byte{0xFF}); syn.push_back(byte{0xD8}); // SOI
        syn.push_back(byte{0xFF}); syn.push_back(byte{0xFF}); // fill
        syn.push_back(byte{0xFF});                             // fill
        append_marker(syn, 0xE0, str_to_bytes("JFIF FILL_SENTINEL")); // APP0
        append_marker(syn, 0xDB, str_to_bytes("DQT"));                // DQT
        append_marker(syn, 0xDA, str_to_bytes("SOS"));               // SOS
        for (char c : std::string_view("\x10\x20")) syn.push_back(byte(static_cast<unsigned char>(c)));
        syn.push_back(byte{0xFF}); syn.push_back(byte{0xD9});         // EOI

        const auto r = wmr::provenance::rewrite_jpeg_strip_ai(syn, true);
        REQUIRE(r.ok);
        const std::span<const byte> out(r.out.data(), r.out.size());
        CHECK(contains(out, "FILL_SENTINEL"));
    }

    SECTION("malformed: marker seg_len runs past EOF => ok=false (no clamp)") {
        std::vector<byte> bad;
        bad.push_back(byte{0xFF}); bad.push_back(byte{0xD8}); // SOI
        bad.push_back(byte{0xFF}); bad.push_back(byte{0xE0}); // APP0
        bad.push_back(byte{0xFF}); bad.push_back(byte{0x7F}); // seg_len = 0xFF7F (65407, past EOF)
        bad.push_back(byte{'X'});                             // only one body byte follows

        const auto r = wmr::provenance::rewrite_jpeg_strip_ai(bad, true);
        CHECK_FALSE(r.ok);
        CHECK(r.out.empty());
    }

    SECTION("strip-all drops APPn + COM, keeps DQT/DHT/SOF, stays lossless") {
        std::vector<byte> extra;
        append_marker(extra, 0xE0, str_to_bytes("JFIF STRIPALL_APP0")); // APP0 (non-AI)
        append_marker(extra, 0xFE, str_to_bytes("a plain comment STRIPALL_COM")); // COM (non-AI)
        const std::vector<byte> in = inject_after_app0(base, extra);

        const auto r = wmr::provenance::rewrite_jpeg_strip_ai(in, /*keep_standard=*/false);
        REQUIRE(r.ok);
        const std::span<const byte> out(r.out.data(), r.out.size());
        CHECK_FALSE(contains(out, "STRIPALL_APP0"));
        CHECK_FALSE(contains(out, "STRIPALL_COM"));
        // Entropy tail is still byte-identical (lossless).
        const auto sos_in = first_sos_off(in);
        const auto sos_out = first_sos_off(out);
        REQUIRE(sos_in.has_value());
        REQUIRE(sos_out.has_value());
        bool tail_eq = (in.size() - *sos_in) == (out.size() - *sos_out);
        for (std::size_t i = 0; i < in.size() - *sos_in && tail_eq; ++i)
            if (in[*sos_in + i] != out[*sos_out + i]) tail_eq = false;
        CHECK(tail_eq);
        CHECK(pixels_identical(in, out));
    }
}
