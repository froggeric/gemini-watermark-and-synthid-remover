// JPEG marker-level provenance strip: scan and rewrite. Pure TU (no OpenCV,
// no FFmpeg). Operates on raw container bytes only, never decodes pixels.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace wmr::provenance {

struct JpegFinding {
    std::string marker; // "APP11", "APP1/XMP", "APP13", "APP1/EXIF"
    std::string detail;
    bool dropped;
};

struct JpegReport {
    bool ok = false;
    bool has_c2pa = false;
    std::vector<JpegFinding> findings;
};

struct JpegRewriteResult {
    bool ok = false; // false => malformed => copy input unchanged
    std::vector<std::byte> out;
    int items_dropped = 0;
    JpegReport report;
};

JpegReport scan_jpeg(std::span<const std::byte> in);

// Rewrite: from SOI, walk markers. On the first SOS (0xDA) or EOI (0xD9) copy
// the remainder VERBATIM (lossless; also covers progressive/multi-scan JPEG,
// because every byte after the first SOS is copied unchanged). Before SOS:
//   - drop ALL APP11 (0xEB) (rare; only JUMBF/C2PA/JPEG-XS use it);
//   - drop APP1 (0xE1) whose payload carries an AI marker (covers XMP-AI and
//     EXIF-AI; an AI-carrying APP1/EXIF is dropped WHOLE, no IFD rewrite in v1);
//   - drop APP13/IPTC (0xED) whose payload carries an AI marker;
//   - drop any other APPn whose payload carries a bare AI marker;
//   - drop COM (0xFE) whose bytes carry an AI marker;
//   - copy APP0, DQT, DHT, SOF, DRI and all other markers verbatim.
// keep_standard=false drops ALL APPn and COM (keeps DQT/DHT/SOF/DRI; still
// lossless). 0xFF fill bytes before a marker are skipped (semantically null).
JpegRewriteResult rewrite_jpeg_strip_ai(std::span<const std::byte> in,
                                        bool keep_standard);

} // namespace wmr::provenance
