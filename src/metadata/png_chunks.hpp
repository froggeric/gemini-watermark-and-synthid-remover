// PNG chunk-level provenance strip: scan and rewrite. Pure TU (no OpenCV,
// no FFmpeg). Operates on raw container bytes only, never decodes pixels.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wmr::provenance {

struct PngFinding {
    std::string chunk_type; // "tEXt", "caBX", ...
    std::string detail;     // e.g. "AI key: parameters" or "C2PA manifest"
    bool dropped;           // true if the remover will drop it
};

struct PngReport {
    bool ok = false; // false => malformed; caller copies input unchanged
    bool has_c2pa = false;
    std::vector<PngFinding> findings;
};

struct PngRewriteResult {
    bool ok = false; // false => malformed => caller copies input unchanged
    std::vector<std::byte> out; // the rewritten bytes (valid when ok)
    int items_dropped = 0;
    PngReport report;
};

// Scan only (no rewrite). Used by the reporter. The 'dropped' flag reflects
// the default remover policy (keep_standard=true).
PngReport scan_png(std::span<const std::byte> in);

// Rewrite: drop caBX plus AI tEXt/iTXt/zTXt-by-key (plus everything
// non-essential when !keep_standard). Drops WHOLE chunks only, so surviving
// chunks keep their CRCs (no CRC recompute). IHDR/IDAT/IEND and the APNG
// structural chunks (acTL/fcTL/fdAT) are never touched. Trailing bytes after
// IEND are copied verbatim. keep_standard=false drops every chunk except
// IHDR, IDAT(s), IEND, and the APNG structural chunks.
PngRewriteResult rewrite_png_strip_ai(std::span<const std::byte> in,
                                      bool keep_standard);

} // namespace wmr::provenance
