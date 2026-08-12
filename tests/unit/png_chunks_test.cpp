// Unit tests for the PNG chunk-level provenance strip (wmr::provenance).
//
// All fixtures are SYNTHESIZED in code: a real, decodable PNG is produced with
// cv::imencode, then metadata chunks (tEXt, zTXt, caBX, jumb-payload, APNG
// structural) are spliced in before the IDAT. This keeps the IDAT/IHDR/IEND
// bytes valid (so cv::imdecode succeeds) while exercising every branch of
// decide_chunk. No binary blobs, no LFS.
//
// The bit-identical-pixels gate (cv::imdecode(in) == cv::imdecode(out), max abs
// diff 0) is the load-bearing regression: dropping a metadata chunk must never
// touch a pixel.

#include <catch2/catch_test_macros.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "metadata/png_chunks.hpp"
#include "metadata/provenance_constants.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ---------------- byte / view helpers ----------------

using byte = std::byte;

std::vector<byte> str_to_bytes(std::string_view s) {
    const auto* p = reinterpret_cast<const byte*>(s.data());
    return std::vector<byte>(p, p + s.size());
}

std::uint32_t be32_at(std::span<const byte> s, std::size_t off) {
    return (std::to_integer<std::uint32_t>(s[off]) << 24) |
           (std::to_integer<std::uint32_t>(s[off + 1]) << 16) |
           (std::to_integer<std::uint32_t>(s[off + 2]) << 8) |
           (std::to_integer<std::uint32_t>(s[off + 3]));
}

void append_u32_be(std::vector<byte>& v, std::uint32_t x) {
    v.push_back(byte(x >> 24));
    v.push_back(byte(x >> 16));
    v.push_back(byte(x >> 8));
    v.push_back(byte(x));
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

// ---------------- PNG CRC-32 (polynomial 0xEDB88320, init/final 0xFFFFFFFF) ----------------

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

// Append a full PNG chunk (length + type + data + CRC) to v.
void append_chunk(std::vector<byte>& v, std::string_view type,
                  std::span<const byte> data) {
    REQUIRE(type.size() == 4);
    append_u32_be(v, static_cast<std::uint32_t>(data.size()));
    const std::size_t td_start = v.size(); // type+data will be contiguous here
    for (char c : type) v.push_back(byte(static_cast<unsigned char>(c)));
    v.insert(v.end(), data.begin(), data.end());
    const std::uint32_t crc = crc32_bytes(v.data() + td_start, 4 + data.size());
    append_u32_be(v, crc);
}

// tEXt/iTXt/zTXt data = keyword + NUL + value (v1 does not inflate zTXt, so the
// same layout exercises the key-only match).
std::vector<byte> text_data(std::string_view keyword, std::string_view value) {
    std::vector<byte> d;
    for (char c : keyword) d.push_back(byte(static_cast<unsigned char>(c)));
    d.push_back(byte{0}); // NUL separator
    for (char c : value) d.push_back(byte(static_cast<unsigned char>(c)));
    return d;
}

// ---------------- real, decodable PNG via OpenCV ----------------

std::vector<byte> encode_png(const cv::Mat& img) {
    std::vector<uchar> buf;
    REQUIRE(cv::imencode(".png", img, buf));
    const auto* p = reinterpret_cast<const byte*>(buf.data());
    return std::vector<byte>(p, p + buf.size());
}

// Offset of the first chunk whose 4-byte type matches (the offset of its length
// field), or nullopt.
std::optional<std::size_t> find_chunk_off(std::span<const byte> png,
                                          std::string_view type) {
    if (png.size() < 8) return std::nullopt;
    std::size_t off = 8;
    while (off + 8 <= png.size()) {
        bool match = true;
        for (int i = 0; i < 4; ++i) {
            if (std::to_integer<char>(png[off + 4 + i]) != type[i]) {
                match = false;
                break;
            }
        }
        if (match) return off;
        const std::uint32_t len = be32_at(png, off);
        off += static_cast<std::size_t>(len) + 12u;
    }
    return std::nullopt;
}

// Concatenate every IDAT chunk's DATA bytes (the zlib stream). The strip must
// leave this byte-identical.
std::vector<byte> extract_idat_data(std::span<const byte> png) {
    std::vector<byte> out;
    std::size_t off = 8;
    while (off + 8 <= png.size()) {
        const std::string_view type(reinterpret_cast<const char*>(png.data() + off + 4), 4);
        const std::uint32_t len = be32_at(png, off);
        if (off + 8 + len + 4 > png.size()) break;
        if (type == "IDAT") {
            const auto* d = png.data() + off + 8;
            out.insert(out.end(), d, d + len);
        }
        off += static_cast<std::size_t>(len) + 12u;
        if (type == "IEND") break;
    }
    return out;
}

// Splice extra chunks between the header and the first IDAT. Returns a new PNG
// whose IDAT/IEND are byte-for-byte the encoded original.
std::vector<byte> inject_before_idat(std::span<const byte> base,
                                     std::span<const byte> extra) {
    const auto idat = find_chunk_off(base, "IDAT");
    REQUIRE(idat.has_value());
    std::vector<byte> out(base.begin(), base.begin() + *idat);
    out.insert(out.end(), extra.begin(), extra.end());
    out.insert(out.end(), base.begin() + *idat, base.end());
    return out;
}

// Bit-identical-pixels gate: both buffers must decode to equal Mats.
bool pixels_identical(std::span<const byte> a, std::span<const byte> b) {
    const auto* pa = reinterpret_cast<const uchar*>(a.data());
    const auto* pb = reinterpret_cast<const uchar*>(b.data());
    cv::Mat ma = cv::imdecode(cv::Mat(1, static_cast<int>(a.size()), CV_8U, const_cast<uchar*>(pa)), cv::IMREAD_UNCHANGED);
    cv::Mat mb = cv::imdecode(cv::Mat(1, static_cast<int>(b.size()), CV_8U, const_cast<uchar*>(pb)), cv::IMREAD_UNCHANGED);
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

// A small, colorful Mat so a channel swap or shift would be visible (gray
// fixtures can hide such bugs).
cv::Mat make_img() {
    cv::Mat img(8, 8, CV_8UC3);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3b& px = img.at<cv::Vec3b>(y, x);
            px[0] = static_cast<uchar>(x * 30);     // B
            px[1] = static_cast<uchar>(y * 30);     // G
            px[2] = static_cast<uchar>((x + y) * 10); // R
        }
    return img;
}

} // namespace

// =====================================================================================

TEST_CASE("PNG provenance strip", "[metadata][png]") {
    const cv::Mat img = make_img();
    const std::vector<byte> base = encode_png(img);
    // Sanity: the OpenCV-encoded base is a clean IHDR+IDAT+IEND (no ancillary
    // chunks), so any findings/drops below come only from the chunks we inject.
    REQUIRE(find_chunk_off(base, "IDAT").has_value());
    REQUIRE(find_chunk_off(base, "IEND").has_value());

    SECTION("valid PNG with caBX + AI tEXt + IDAT + IEND (decode + drop gate)") {
        std::vector<byte> extra;
        append_chunk(extra, "tEXt", text_data("parameters", "steps=30,sampler=euler,SENTINEL_PARAMS_9"));
        append_chunk(extra, "tEXt", text_data("Author", "me, the author"));
        append_chunk(extra, "caBX", str_to_bytes("jumb SENTINEL_CABX_42 manifest bytes"));

        const std::vector<byte> in = inject_before_idat(base, extra);
        const auto r = wmr::provenance::rewrite_png_strip_ai(in, /*keep_standard=*/true);

        REQUIRE(r.ok);
        REQUIRE(r.items_dropped >= 2); // parameters + caBX (Author kept)
        CHECK(r.report.has_c2pa);

        // caBX chunk type and the parameters key/value are gone; Author survives.
        const std::span<const byte> out(r.out.data(), r.out.size());
        CHECK_FALSE(contains(out, "SENTINEL_PARAMS_9"));
        CHECK_FALSE(contains(out, "caBX"));
        CHECK(contains(out, "the author"));

        // IDAT bytes are byte-identical to the input.
        CHECK(extract_idat_data(in) == extract_idat_data(out));

        // Bit-identical pixels (the load-bearing regression).
        CHECK(pixels_identical(in, out));
    }

    SECTION("denylist key coverage: every kAiDenyKeys element is dropped") {
        for (std::string_view key : wmr::provenance::kAiDenyKeys) {
            std::vector<byte> extra;
            const std::string val = "v_" + std::string(key) + "_SENTINEL";
            append_chunk(extra, "tEXt", text_data(key, val));
            const std::vector<byte> in = inject_before_idat(base, extra);

            const auto r = wmr::provenance::rewrite_png_strip_ai(in, true);
            REQUIRE(r.ok);
            CHECK(r.items_dropped >= 1);
            const std::span<const byte> out(r.out.data(), r.out.size());
            CHECK_FALSE(contains(out, val));
        }
    }

    SECTION("Author/Title/Copyright kept under keep_standard, dropped under strip-all") {
        std::vector<byte> extra;
        append_chunk(extra, "tEXt", text_data("Author", "SENTINEL_KEEP_A"));
        append_chunk(extra, "tEXt", text_data("Title", "SENTINEL_KEEP_T"));
        append_chunk(extra, "tEXt", text_data("Copyright", "SENTINEL_KEEP_C"));
        const std::vector<byte> in = inject_before_idat(base, extra);

        const auto keep = wmr::provenance::rewrite_png_strip_ai(in, /*keep_standard=*/true);
        REQUIRE(keep.ok);
        CHECK(keep.items_dropped == 0);
        const std::span<const byte> keep_out(keep.out.data(), keep.out.size());
        CHECK(contains(keep_out, "SENTINEL_KEEP_A"));
        CHECK(contains(keep_out, "SENTINEL_KEEP_T"));
        CHECK(contains(keep_out, "SENTINEL_KEEP_C"));
        CHECK(pixels_identical(in, keep.out));

        const auto strip = wmr::provenance::rewrite_png_strip_ai(in, /*keep_standard=*/false);
        REQUIRE(strip.ok);
        CHECK(strip.items_dropped >= 3);
        const std::span<const byte> strip_out(strip.out.data(), strip.out.size());
        CHECK_FALSE(contains(strip_out, "SENTINEL_KEEP_A"));
        CHECK_FALSE(contains(strip_out, "SENTINEL_KEEP_T"));
        CHECK_FALSE(contains(strip_out, "SENTINEL_KEEP_C"));
        CHECK(pixels_identical(in, strip.out));
    }

    SECTION("software kept unless its value carries an AI substring") {
        // "Adobe Photoshop" has no AI substring -> kept.
        std::vector<byte> extra_ok;
        append_chunk(extra_ok, "tEXt", text_data("Software", "Adobe Photoshop SENTINEL_SW_PS"));
        const std::vector<byte> in_ok = inject_before_idat(base, extra_ok);
        const auto r_ok = wmr::provenance::rewrite_png_strip_ai(in_ok, true);
        REQUIRE(r_ok.ok);
        CHECK(r_ok.items_dropped == 0);
        const std::span<const byte> out_ok(r_ok.out.data(), r_ok.out.size());
        CHECK(contains(out_ok, "SENTINEL_SW_PS"));

        // "ComfyUI" is an AI substring -> the keep-list key is dropped by value scan.
        std::vector<byte> extra_ai;
        append_chunk(extra_ai, "tEXt", text_data("Software", "ComfyUI SENTINEL_SW_CF"));
        const std::vector<byte> in_ai = inject_before_idat(base, extra_ai);
        const auto r_ai = wmr::provenance::rewrite_png_strip_ai(in_ai, true);
        REQUIRE(r_ai.ok);
        CHECK(r_ai.items_dropped >= 1);
        const std::span<const byte> out_ai(r_ai.out.data(), r_ai.out.size());
        CHECK_FALSE(contains(out_ai, "SENTINEL_SW_CF"));
    }

    SECTION("zTXt matched by KEY only (value not scanned in v1)") {
        // Deny key -> dropped by key.
        std::vector<byte> extra_d;
        append_chunk(extra_d, "zTXt", text_data("parameters", "\x00x\x9cSENTINEL_ZTX_P"));
        const std::vector<byte> in_d = inject_before_idat(base, extra_d);
        const auto rd = wmr::provenance::rewrite_png_strip_ai(in_d, true);
        REQUIRE(rd.ok);
        CHECK(rd.items_dropped >= 1);
        const std::span<const byte> outd(rd.out.data(), rd.out.size());
        CHECK_FALSE(contains(outd, "SENTINEL_ZTX_P"));

        // Non-deny key -> kept under keep_standard (value not inflated/scanned),
        // even though the raw bytes happen to contain an AI marker.
        std::vector<byte> extra_k;
        append_chunk(extra_k, "zTXt", text_data("Author", "\x00comfyui SENTINEL_ZTX_A"));
        const std::vector<byte> in_k = inject_before_idat(base, extra_k);
        const auto rk = wmr::provenance::rewrite_png_strip_ai(in_k, true);
        REQUIRE(rk.ok);
        CHECK(rk.items_dropped == 0);
        const std::span<const byte> outk(rk.out.data(), rk.out.size());
        CHECK(contains(outk, "SENTINEL_ZTX_A"));
    }

    SECTION("malformed: declared chunk length runs past EOF => ok=false (no clamp)") {
        // Signature + valid IHDR, then a chunk header claiming len=0xFFFFFF with
        // no body. The fail-safe must reject the whole file (caller copies input
        // unchanged); it must NOT clamp the length and produce a truncated file.
        std::vector<byte> bad(base.begin(), base.begin() + 8); // signature
        {
            const auto ihdr = find_chunk_off(base, "IHDR");
            REQUIRE(ihdr.has_value());
            const std::size_t ihdr_total = 12 + 13; // IHDR is always 13 bytes
            bad.insert(bad.end(), base.begin() + *ihdr, base.begin() + *ihdr + ihdr_total);
        }
        // Bogus ancillary chunk: length 0x00FFFFFF, type "tEXt", then truncate.
        append_u32_be(bad, 0x00FFFFFFu);
        for (char c : std::string_view("tEXt")) bad.push_back(byte(static_cast<unsigned char>(c)));
        bad.push_back(byte{'X'}); // only one byte of an impossible body

        const auto r = wmr::provenance::rewrite_png_strip_ai(bad, true);
        CHECK_FALSE(r.ok);
        // ok=false => out is empty; the documented contract is the caller copies
        // the input through UNCHANGED (fail-safe: never truncate).
        CHECK(r.out.empty());
    }

    SECTION("malformed: not a PNG => ok=false") {
        const std::vector<byte> junk = str_to_bytes("not a png file at all, just text");
        const auto r = wmr::provenance::rewrite_png_strip_ai(junk, true);
        CHECK_FALSE(r.ok);
        CHECK(r.out.empty());
    }

    SECTION("trailing bytes after IEND are preserved verbatim") {
        std::vector<byte> extra;
        append_chunk(extra, "tEXt", text_data("parameters", "SENTINEL_TRAIL_PARAMS"));
        std::vector<byte> in = inject_before_idat(base, extra);
        in.push_back(byte{'T'}); // trailer after IEND
        in.push_back(byte{'R'});
        in.push_back(byte{'X'});

        const auto r = wmr::provenance::rewrite_png_strip_ai(in, true);
        REQUIRE(r.ok);
        const std::span<const byte> out(r.out.data(), r.out.size());
        CHECK(contains(out, "TRX"));
        CHECK_FALSE(contains(out, "SENTINEL_TRAIL_PARAMS"));
        // The trailer must be the suffix.
        REQUIRE(r.out.size() >= 3);
        CHECK(std::to_integer<char>(r.out[r.out.size() - 3]) == 'T');
        CHECK(std::to_integer<char>(r.out[r.out.size() - 2]) == 'R');
        CHECK(std::to_integer<char>(r.out[r.out.size() - 1]) == 'X');
    }

    SECTION("APNG structural chunks preserved even under strip-all") {
        std::vector<byte> extra;
        // acTL, fcTL, fdAT. Sizes/payloads here are stand-ins (not a spec-valid
        // APNG): the goal is to prove the strip-all policy preserves the APNG
        // structural chunk TYPES, not to render an animation.
        append_chunk(extra, "acTL", str_to_bytes("SENTINEL_APNG_ACTL1234"));
        append_chunk(extra, "fcTL", str_to_bytes("SENTINEL_APNG_FCTL56789012"));
        // fdAT = 4-byte BE sequence number + frame data. Built as explicit bytes
        // (NOT via str_to_bytes of a NUL-led literal, which truncates to length 0
        // because string_view from a const char* stops at the first NUL).
        std::vector<byte> fdat_data = {byte{0x00}, byte{0x00}, byte{0x00}, byte{0x01}};
        const std::vector<byte> fdat_sentinel = str_to_bytes("SENTINEL_APNG_FDAT");
        fdat_data.insert(fdat_data.end(), fdat_sentinel.begin(), fdat_sentinel.end());
        append_chunk(extra, "fdAT", fdat_data);
        append_chunk(extra, "tEXt", text_data("parameters", "SENTINEL_APNG_PARAMS"));
        const std::vector<byte> in = inject_before_idat(base, extra);

        const auto strip = wmr::provenance::rewrite_png_strip_ai(in, /*keep_standard=*/false);
        REQUIRE(strip.ok);
        const std::span<const byte> out(strip.out.data(), strip.out.size());
        // APNG structural chunks survive strip-all (animation must not break).
        CHECK(contains(out, "SENTINEL_APNG_ACTL1234"));
        CHECK(contains(out, "SENTINEL_APNG_FCTL56789012"));
        CHECK(contains(out, "SENTINEL_APNG_FDAT"));
        // The AI tEXt is still dropped under strip-all.
        CHECK_FALSE(contains(out, "SENTINEL_APNG_PARAMS"));
        // Lossless invariant. We cannot use pixels_identical() here: an acTL chunk
        // placed before IDAT (as the stand-in does) is not a valid APNG, so
        // OpenCV's libpng rejects the whole image on decode. The IDAT zlib stream
        // is the load-bearing invariant instead: it must be byte-identical.
        CHECK(extract_idat_data(in) == extract_idat_data(out));
    }

    SECTION("content sniff: non-caBX chunk whose payload starts with jumb is dropped") {
        std::vector<byte> extra;
        // An arbitrary ancillary type "prVn" (not caBX, not a text type) carrying
        // a jumb-payload must be caught by the content sniff and dropped as C2PA.
        std::vector<byte> payload = str_to_bytes("jumb SENTINEL_JUMB_SNIFF");
        append_chunk(extra, "prVn", payload);
        const std::vector<byte> in = inject_before_idat(base, extra);

        const auto r = wmr::provenance::rewrite_png_strip_ai(in, true);
        REQUIRE(r.ok);
        CHECK(r.report.has_c2pa);
        CHECK(r.items_dropped >= 1);
        const std::span<const byte> out(r.out.data(), r.out.size());
        CHECK_FALSE(contains(out, "SENTINEL_JUMB_SNIFF"));
    }
}
