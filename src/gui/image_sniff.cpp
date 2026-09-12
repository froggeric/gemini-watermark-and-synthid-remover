#include "gui/image_sniff.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace wmr::gui {
namespace {

// Decompression-bomb gate: OpenCV would happily allocate w*h*3 bytes for a
// tiny header declaring a huge canvas.
constexpr int64_t kMaxPixels = 120000000;  // ~120 MP
constexpr int64_t kMaxSide = 16384;

uint16_t be16(const unsigned char* p) {
    return static_cast<uint16_t>((p[0] << 8) | p[1]);
}
uint16_t le16(const unsigned char* p) {
    return static_cast<uint16_t>((p[1] << 8) | p[0]);
}
uint32_t le32(const unsigned char* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8)
           | (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

void apply_gate(SniffResult& r) {
    const int64_t w = r.width, h = r.height;
    r.too_large = w * h > kMaxPixels || w > kMaxSide || h > kMaxSide;
}

SniffResult sniff_png(const unsigned char* data, size_t n) {
    static const unsigned char kMagic[] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    if (n < 8 || std::memcmp(data, kMagic, 8) != 0) return {};
    if (n < 24) return {};  // truncated before the IHDR W/H fields
    SniffResult r;
    r.format = SniffFormat::Png;
    r.ext = "png";
    // IHDR: big-endian W at 16-19, H at 20-23. A hostile header can declare
    // up to 0xFFFFFFFF; gate on the raw value and clamp before storing in int.
    const uint32_t w = (static_cast<uint32_t>(data[16]) << 24) | (static_cast<uint32_t>(data[17]) << 16)
                     | (static_cast<uint32_t>(data[18]) << 8) | data[19];
    const uint32_t h = (static_cast<uint32_t>(data[20]) << 24) | (static_cast<uint32_t>(data[21]) << 16)
                     | (static_cast<uint32_t>(data[22]) << 8) | data[23];
    r.width = static_cast<int>(std::min<uint32_t>(w, 0x7FFFFFFFu));
    r.height = static_cast<int>(std::min<uint32_t>(h, 0x7FFFFFFFu));
    apply_gate(r);
    return r;
}

SniffResult sniff_jpeg(const unsigned char* data, size_t n) {
    if (n < 2 || data[0] != 0xFF || data[1] != 0xD8) return {};
    // Walk length-prefixed segments (FF marker len_hi len_lo) until an SOF.
    // seg points at the 0xFF; within an SOF segment: marker(+1), len(+2..3),
    // precision(+4), height(+5..6), width(+7..8), Nf(+9).
    size_t p = 2;
    while (p + 4 <= n) {
        if (data[p] != 0xFF) break;  // garbage
        const unsigned marker = data[p + 1];
        if (marker == 0xDA) break;   // SOS before any SOF: no dimensions
        const bool is_sof = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4
                            && marker != 0xC8 && marker != 0xCC;
        if (is_sof) {
            // Enough bytes for FF+marker+len+precision+H+W+Nf. NOT the full
            // segment length: a truncated-but-sufficient SOF is fine.
            if (p + 10 > n) break;
            SniffResult r;
            r.format = SniffFormat::Jpeg;
            r.ext = "jpg";
            r.height = be16(data + p + 5);  // SOF stores height then width
            r.width = be16(data + p + 7);
            apply_gate(r);
            return r;
        }
        const unsigned len = be16(data + p + 2);
        if (len < 2) break;  // malformed segment length
        p += 2 + len;
    }
    return {};  // ran out of data / garbage: Unsupported
}

SniffResult sniff_webp(const unsigned char* data, size_t n) {
    if (n < 12 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WEBP", 4) != 0) {
        return {};
    }
    const size_t c = 12;  // first chunk header: FourCC(4) + size(4, LE)
    if (c + 8 > n) return {};
    const uint32_t csize = le32(data + c + 4);
    if (static_cast<uint64_t>(c) + 8 + csize > n) return {};  // truncated chunk tree
    const unsigned char* payload = data + c + 8;

    SniffResult r;
    r.format = SniffFormat::Webp;
    r.ext = "webp";
    if (std::memcmp(data + c, "VP8 ", 4) == 0) {
        // Keyframe: 3-byte frame tag + 3-byte start code, then 14-bit
        // width/height (2 upscaling bits each) little-endian.
        if (csize < 10) return {};
        r.width = le16(payload + 6) & 0x3FFF;
        r.height = le16(payload + 8) & 0x3FFF;
    } else if (std::memcmp(data + c, "VP8L", 4) == 0) {
        // Signature byte, then 32 LE bits: 14-bit width-1, 14-bit height-1.
        if (csize < 5) return {};
        const uint32_t bits = le32(payload + 1);
        r.width = static_cast<int>((bits & 0x3FFFu) + 1);
        r.height = static_cast<int>(((bits >> 14) & 0x3FFFu) + 1);
    } else if (std::memcmp(data + c, "VP8X", 4) == 0) {
        // Flags(4), then 24-bit LE canvas width-1 / height-1.
        if (csize < 10) return {};
        r.width = static_cast<int>(
            ((static_cast<uint32_t>(payload[4]) | (static_cast<uint32_t>(payload[5]) << 8)
              | (static_cast<uint32_t>(payload[6]) << 16)) + 1));
        r.height = static_cast<int>(
            ((static_cast<uint32_t>(payload[7]) | (static_cast<uint32_t>(payload[8]) << 8)
              | (static_cast<uint32_t>(payload[9]) << 16)) + 1));
    } else {
        return {};
    }
    apply_gate(r);
    return r;
}

}  // namespace

SniffResult sniff_image(const unsigned char* data, size_t n) {
    if (data == nullptr || n == 0) {
        SniffResult r;
        r.empty = true;
        return r;
    }
    if (SniffResult r = sniff_png(data, n); r.format != SniffFormat::Unsupported) return r;
    if (SniffResult r = sniff_jpeg(data, n); r.format != SniffFormat::Unsupported) return r;
    if (SniffResult r = sniff_webp(data, n); r.format != SniffFormat::Unsupported) return r;
    return {};  // HEIC, PDF, unknown or truncated: Unsupported
}

}
