// Provenance metadata strip: public API and dispatch. Operates on raw
// container bytes (PNG chunks, JPEG markers); never decodes pixels.
//
// v1 scope: PNG strip + JPEG strip + the format-agnostic byte-scan reporter.
// WebP/AVIF/HEIF/JPEG-XL/MP4/MOV are reported as unsupported and passed
// through unchanged (later phases).
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wmr::provenance {

enum class ContainerFormat {
    Png,
    Jpeg,
    WebPRiff,
    IsobmffImage,
    IsobmffVideo,
    Unknown
};

struct ProvenanceFinding {
    ContainerFormat format;
    std::string where; // "PNG:tEXt:parameters", "JPEG:APP11", "PNG:caBX"
    std::string detail;
};

struct MetadataReport {
    ContainerFormat format = ContainerFormat::Unknown;
    bool supported = false; // false for WebP/AVIF/... in v1
    bool has_c2pa = false;
    std::optional<std::string> c2pa_note; // issuer/tool substring if found (best-effort)
    std::vector<ProvenanceFinding> findings;
    int ai_text_keys = 0;
};

struct StripOptions {
    bool keep_standard = true; // default: keep non-AI standard metadata
    bool dry_run = false;      // report only, do not produce output bytes
};

struct StripResult {
    bool ok = false; // false => malformed OR failed => copy input unchanged
    bool supported = false; // false for WebP/AVIF/... (no strip attempted)
    int items_removed = 0;
    std::vector<std::byte> out; // valid when ok && supported && !dry_run
    MetadataReport report;
};

// Identify the container by magic bytes (12 bytes is enough).
ContainerFormat sniff_format(std::span<const std::byte> head);

// Byte-scan report. On a supported format, runs the per-format scan (which
// populates findings + has_c2pa). On an unsupported format, populates format
// and supported=false with no findings.
MetadataReport report_provenance(std::span<const std::byte> in);

// Dispatch by sniffed format. For unsupported formats, returns {ok=true,
// supported=false} so the caller copies the input through unchanged.
StripResult strip_provenance(std::span<const std::byte> in,
                             const StripOptions& opts);

// Post-write provenance guarantee for the remove/synthid path. Reads the
// just-written file, and if it carries any C2PA/AI metadata, strips it in
// place (temp file plus rename for atomicity). On a clean file (the common
// case on OpenCV output, which already strips) the scan finds nothing and the
// file is NOT rewritten (a true no-op, no extra write I/O). On unsupported
// formats it is a no-op. Never throws; on any error the ORIGINAL is left
// intact. Called only when the caller's keep_provenance flag is false.
struct PostWriteResult {
    bool ok = true;
    bool rewritten = false;
    int items_removed = 0;
    std::string format_note;
};
PostWriteResult post_write_provenance_strip(const std::string& file_path,
                                            bool keep_standard);

} // namespace wmr::provenance
