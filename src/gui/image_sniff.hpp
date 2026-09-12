#pragma once
#include <cstddef>
#include <string>

namespace wmr::gui {

enum class SniffFormat { Png, Jpeg, Webp, Unsupported };

struct SniffResult {
    SniffFormat format = SniffFormat::Unsupported;
    std::string ext;         // "png" / "jpg" / "webp"; empty when unsupported
    int width = 0;
    int height = 0;
    bool too_large = false;  // decompression-bomb gate (kMaxPixels / kMaxSide)
    bool empty = false;      // zero-length input
};

// Classifies upload bytes by magic prefix and extracts dimensions WITHOUT
// decoding. The input is attacker-controlled (truncated multipart content):
// every field read is length-guarded, and a buffer too short for a field
// classifies as Unsupported; nothing ever reads past n.
SniffResult sniff_image(const unsigned char* data, size_t n);

}
