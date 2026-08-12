// Provenance metadata constants: the single source of truth shared by the
// reporter (scan) and the remover (strip), so detection and removal cannot
// drift. Header-only constexpr. This TU is pure: no OpenCV, no FFmpeg.
//
// The C2PA storage constants below were verified three ways: the C2PA
// Technical Specification 2.1 (spec.c2pa.org), the contentauth/c2pa-rs
// reference implementation, and a hexdump of a real C2PA fixture. See
// docs/research/provenance-metadata-strip-plan.md section 5 for the citations.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace wmr::provenance {

// Container signatures (magic bytes), sniffed by content, never by extension.
inline constexpr std::byte kPngSig[8] = {
    std::byte{0x89}, std::byte{0x50}, std::byte{0x4E}, std::byte{0x47},
    std::byte{0x0D}, std::byte{0x0A}, std::byte{0x1A}, std::byte{0x0A}};
inline constexpr std::byte kJpegSoi[2] = {std::byte{0xFF}, std::byte{0xD8}};
inline constexpr std::string_view kWebpRiff = "RIFF";    // offset 0
inline constexpr std::string_view kWebpTag = "WEBP";     // offset 8
inline constexpr std::string_view kIsobmffFtyp = "ftyp"; // offset 4

// C2PA storage. The strip is DUAL: type match plus content sniff, so a manifest
// that uses a variant label is still caught by the content sniff.
//
// PNG: ancillary chunk type "caBX" (63 61 42 58; capital B, capital X).
// The casing is significant: PNG chunk-name case encodes the
// ancillary/private/reserved/safe-to-copy bits.
inline constexpr std::string_view kPngC2paChunk = "caBX";
//
// JPEG: there is NO top-level UUID identifying a C2PA APP11. C2PA rides the
// ISO/IEC 19566-5 JUMBF-in-APP11 envelope. The APP11 payload begins with:
//   CI = "JP" (0x4A 0x50), En = 2 bytes (box instance),
//   Z = 4 bytes BE (segment sequence number 1,2,3,...; NOT a 1-byte count),
// followed by the JUMBF superbox bytes. C2PA is identified by the JUMBF
// Description box content-type UUID at payload offset [24..40]:
//   6332706100110010800000aa00389b71  ("c2pa" plus the ISO JUMBF UUID tail).
// Quick check: payload[24..28] == {0x63,0x32,0x70,0x61} ("c2pa"). The v1
// strip drops ALL APP11 (rare; only C2PA/JUMBF/JPEG-XS use it), so the
// envelope details matter only for the reporter's has_c2pa flag.
inline constexpr std::byte kJpegApp11C2paMagic[4] = {
    std::byte{0x63}, std::byte{0x32}, std::byte{0x70},
    std::byte{0x61}}; // "c2pa"
inline constexpr std::byte kJpegApp11EnvelopeCI[2] = {
    std::byte{0x4A}, std::byte{0x50}}; // "JP"
inline constexpr std::byte kC2paJumbfTypeUuid16[16] = {
    std::byte{0x63}, std::byte{0x32}, std::byte{0x70}, std::byte{0x61},
    std::byte{0x00}, std::byte{0x11}, std::byte{0x00}, std::byte{0x10},
    std::byte{0x80}, std::byte{0x00}, std::byte{0x00}, std::byte{0xAA},
    std::byte{0x00}, std::byte{0x38}, std::byte{0x9B}, std::byte{0x71}};
//
// ISOBMFF (AVIF/HEIF/MOV/MP4): a "uuid" box (type 75 75 69 64) whose 16-byte
// extended type is D8FEC3D6-1B0E-483C-9297-582887EC4881. This is the
// ISOBMFF uuid-box extended type ONLY (it never appears in JPEG).
inline constexpr std::byte kIsobmffC2paUuid16[16] = {
    std::byte{0xD8}, std::byte{0xFE}, std::byte{0xC3}, std::byte{0xD6},
    std::byte{0x1B}, std::byte{0x0E}, std::byte{0x48}, std::byte{0x3C},
    std::byte{0x92}, std::byte{0x97}, std::byte{0x58}, std::byte{0x28},
    std::byte{0x87}, std::byte{0x7E}, std::byte{0xC4}, std::byte{0x81}};
//
// JUMBF box magics (ISO/IEC 19566-5): "jumb" superbox, "jumd" description box.
inline constexpr std::string_view kJumbfMagic = "jumb";   // 6A 75 6D 62
inline constexpr std::string_view kJumbfDescMagic = "jumd"; // 6A 75 6D 64

// PNG chunk types the strip parses (everything else is copied verbatim).
inline constexpr std::string_view kPngText = "tEXt";
inline constexpr std::string_view kPngItxt = "iTXt";
inline constexpr std::string_view kPngZtxt = "zTXt";
inline constexpr std::string_view kPngIhdr = "IHDR";
inline constexpr std::string_view kPngIdat = "IDAT";
inline constexpr std::string_view kPngIend = "IEND";

// tEXt/iTXt keys that are ALWAYS dropped (AI-specific). Lowercase compared.
inline constexpr std::string_view kAiDenyKeys[] = {
    "parameters",        "prompt",          "negative_prompt",
    "workflow",          "comfyui",         "sd-metadata",
    "invokeai_metadata", "generation_data", "ai_metadata",
    "dream",             "sd:prompt",       "sd:negative_prompt",
    "sd:seed",           "sd:steps",        "sd:sampler",
    "sd:cfg_scale",      "sd:model_hash",   "c2pa",
    "c2pa_chunk"};

// AI keyword substrings. Applied to (a) the VALUE of a keep-list key, and
// (b) any APP segment / IPTC / XMP bytes. Case-insensitive.
inline constexpr std::string_view kAiSubstrings[] = {
    "stable_diffusion",                    "stable-diffusion",
    "comfyui",                             "automatic1111",
    "invokeai",                            "midjourney",
    "dall-e",                              "dalle",
    "imagen",                              "synthid",
    "google_ai",                           "openai",
    "c2pa",                                "trainedalgorithmicmedia",
    "compositesynthetic",                  "compositewithtrainedalgorithmicmedia",
    "aisystemused",                        "aipromptinformation",
    "aigc",                                "tc260:aigc"};

// Standard keys KEPT by default (keep_standard=true). 'software' is here AND
// has its value scanned for AI substrings: kept UNLESS its value carries an AI
// marker. Listed for documentation and for the keep/scan decision; the strip
// scan applies to every non-deny key when keep_standard is true.
inline constexpr std::string_view kStandardKeepKeys[] = {
    "author", "title", "description", "copyright", "creation time",
    "software", "comment", "disclaimer", "source", "warning"};

// IPTC "Made with AI" markers and the TC260 AIGC namespace are covered by
// kAiSubstrings above (substring match on the raw APP13 bytes).

} // namespace wmr::provenance
