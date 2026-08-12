# Provenance metadata strip: implementation plan (revised, implementation-ready)

This is the reviewed and corrected plan for native C2PA / AI-provenance metadata
stripping in `wmr`. It supersedes the architect's original blueprint. Every claim
about the codebase below was verified by reading the real files (paths and line
numbers are current as of v1.16.8 on `main`). The two external facts that could
not be read from source were verified empirically (OpenCV `imwrite` behavior) and
against the spec (the C2PA storage constants), and are flagged inline.

Provenance metadata is a distinct concern from the visible Gemini diamond
(`WatermarkEngine` reverse alpha-blend) and from SynthID (`--synthid-attack
regen`). It never decodes pixels. It operates only on container bytes (PNG
chunks, JPEG markers, later ISOBMFF boxes).

## 0. Owner decisions applied

**DECISION A: strip by default on `remove` and `synthid`.** The provenance strip
runs as a guaranteed final step of the output of `wmr remove` and `wmr
synthid`, with an opt-out flag `--keep-provenance`. The mechanism is a
post-write byte pass on the output file (after `cv::imwrite`), not a change to
the OpenCV write path. See section 9 for why it is defensive (today near-no-op)
and section 3 for the re-encode reality. For `video` in v1, the FFmpeg re-encode
already produces a fresh container with no carried-over input boxes; section 11
states the v1 video guarantee precisely.

**DECISION B: no legal disclaimer.** No dedicated ethical/legal notice, no
confirmation prompt. Only a plain factual description in the README and in
`--help` (and one honest wording-lock test, see section 10).

**DECISION C: v1 scope.** v1 = PNG strip + JPEG strip + the format-agnostic
byte-scan reporter, exposed both as the standalone `wmr metadata` subcommand AND
as the default post-write strip on `remove`/`synthid`. WebP RIFF, JPEG IFD
surgical scrub, zTXt inflate, AVIF/HEIF/JPEG-XL ISOBMFF, and MP4/MOV are later
phases (section 11).

## 1. What v1 does and does not guarantee

v1 GUARANTEES, for the standalone `wmr metadata` on a PNG or JPEG input:
- Lossless, container-level removal of: the C2PA manifest chunk/marker, the
  AI-specific `tEXt`/`iTXt` keys (ComfyUI/A1111/InvokeAI/Midjourney/SD
  parameters/workflow, etc.), APP1 XMP carrying AI markers, APP13 IPTC AI
  markers, and (by default-off option) ALL non-essential metadata.
- Bit-identical pixels (PNG IDAT bytes and JPEG entropy bytes are copied
  verbatim; only metadata chunks/markers are dropped).
- Fail-safe: any malformed input is copied through UNCHANGED, never truncated.

v1 GUARANTEES, for the default post-write strip on `remove`/`synthid`:
- The output file is provably provenance-free (the post-write pass scans it; if
  it finds nothing to strip it leaves the file untouched, a true no-op; if it
  finds anything it strips in place). On today's OpenCV output this finds
  nothing (OpenCV already strips; section 3), so the guarantee holds at zero
  cost. The pass exists so the guarantee is independent of the encoder.

v1 does NOT guarantee:
- WebP, AVIF, HEIF, JPEG-XL: pass-through (sniffed, reported as unsupported,
  bytes copied unchanged). Phase 2/3.
- MP4/MOV C2PA: `wmr video` re-encodes via FFmpeg (a fresh container), so input
  C2PA boxes are NOT carried into the output by construction (section 11), but
  v1 does not run an explicit ISOBMFF strip. Phase 4.
- `zTXt` values: matched by KEY only in v1 (no zlib inflate). A `zTXt` with an
  AI key (e.g. `zTXt parameters`) is dropped by key; a `zTXt` with a benign key
  but an AI value is NOT scanned (kept). Phase 2 adds inflate.
- EXIF surgical scrub: in v1 an APP1-EXIF segment that carries an AI marker is
  dropped WHOLE (section 6, the IFD-rewrite landmine). Standard non-AI EXIF is
  kept verbatim.

## 2. Verified integration surface (the codebase facts)

These were confirmed by reading the real files. Line numbers are current.

- `CliMode` enum (`src/cli/cli_app.hpp:20`): `{AutoRemove, Detect, SynthidOnly,
  Video, Cache}`. Add `Metadata`. The dispatch `switch` is in `run_cli`
  (`src/cli/cli_app.cpp:786`). Mode is set from the parsed subcommand
  (`cli_app.cpp:772`); add an `else if (metadata_cmd->parsed())` arm.
- Image output is written at exactly TWO `cv::imwrite` sites:
  - `src/cli/cli_app.cpp:391` inside `process_single_image` (single image).
  - `src/cli/batch_processor.cpp:193` inside `process_single` (batch, one per
    file). The batch entry is `batch_process` (`src/cli/batch_processor.cpp:202`),
    called from `run_cli` at `cli_app.cpp:803` when `is_directory(opts.input_path)`
    and mode is AutoRemove/SynthidOnly.
  - There are no other image-output sites. `regenerator.cpp` operates on a
    `cv::Mat` in memory and returns it; `video_writer.cpp` writes only the MP4
    via FFmpeg (no `imwrite`/`fopen` side images).
- Input is read via `cv::imread(...)` at `cli_app.cpp:108` (detect),
  `cli_app.cpp:165` (remove single), `batch_processor.cpp:47` (batch). The
  metadata layer does NOT use these; it reads raw bytes itself.
- The alpha maps are constexpr PNG byte arrays decoded at runtime via
  `cv::imdecode` (`assets/embedded_assets.hpp`: e.g. `v2_diamond_48_still_png`).
  There is NO chunk-level PNG IO anywhere in the codebase. The metadata layer is
  the first byte-level container parser; it duplicates nothing.
- No existing chunk/EXIF/C2PA/JUMBF/APP11 parser exists (`rg` over `src/`
  confirms). Greenfield.
- `wmr video` is a full FFmpeg re-encode: `video_writer.cpp` calls
  `avcodec_find_encoder_by_name`, `avcodec_send_frame`, `avformat_write_header`
  with a fresh `AVFormatContext`. There is NO stream copy / remux / `AVDISCARD`
  path. A C2PA "uuid"/"jumb" box in the input is therefore NOT copied into the
  output (FFmpeg writes a fresh moov/mdat from the encoded packets).
- Pure-TU precedent (`src/video/notebooklm_gates.{hpp,cpp}`): pure logic, no
  OpenCV/FFmpeg includes, build macros passed IN as bool params so the test
  target (which never defines them) sees identical logic. The metadata TUs are
  pure in the same sense (they include only `<span>`, `<byte>`, `<vector>`,
  `<string>`, `<cstdint>`, `<optional>`, `<algorithm>`, no OpenCV/FFmpeg).
- `src/core/paths.cpp` precedent for the CMake pattern: an UNGATED source that
  is needed by tests is listed in BOTH the top-level `SOURCES`
  (`CMakeLists.txt:17`) AND the test `LIB_SOURCES` (`tests/CMakeLists.txt:18`),
  and is NOT repeated in any feature-gated `target_sources` mirror block
  (that would be a duplicate-source error). The metadata sources follow this
  pattern (ungated, in both lists).
- `tests/integration/visible_pipeline_test.cpp` exercises the engine on
  `cv::Mat` in memory; it does NOT write a file and byte-compare it. No existing
  test byte-compares the output of `remove`/`synthid`. So the post-write strip
  cannot break any existing test fixture.
- zlib is NOT directly linked to `build/wmr`. It is transitively present
  (`libopencv_imgcodecs` links `/usr/lib/libz.1.dylib`). To use zlib from wmr
  code (Phase 2 zTXt inflate) add `find_package(ZLIB REQUIRED)` +
  `target_link_libraries(wmr PRIVATE ZLIB::ZLIB)`. v1 needs no zlib.
- `LICENSE-THIRD-PARTY.md` has per-component sections; `CHANGELOG.md` has an
  empty `[Unreleased]`; `README.md` has a "What it does" table (`README.md:36`)
  and a "Credits" section (`README.md:333`); the "Usage reference" starts at
  `README.md:92`.

## 3. The OpenCV re-encode reality, and why the remove-path strip is defensive

This is the single most important integration fact, verified empirically at the
byte level (standalone OpenCV round-trip AND the real `build/wmr` binary on a
fixture): **`cv::imwrite` already drops ALL container metadata for every format
wmr writes, and injects NOTHING.**

- JPEG: input APP0/JFIF + APP1/EXIF + APP1/XMP + APP13/IPTC + COM -> output has
  only a freshly-emitted default APP0/JFIF (libjpeg-turbo 1.01, density 1x1, no
  units) plus DQT/SOF0/DHT/SOS. All EXIF/XMP/IPTC/comment gone. APP11/C2PA gone.
- PNG: input sRGB/gAMA/cHRM/pHYs/tEXt/zTXt/tIME/iCCP/caBX -> output is exactly
  `IHDR + IDAT(s) + IEND`. Confirmed on the real binary: `wmr remove` on
  `test-images/896x1200-test4-gemini36.png` produced a PNG with chunks
  `[IHDR, IDAT x18, IEND]`, zero ancillary chunks, zero trailing bytes.
- WebP: input with EXIF + XMP flags -> output with neither.
- OpenCV's libpng encoder never calls `png_set_text` on write, so it authors no
  `Software`/`cv:tiff`/tEXt/iTXt/zTXt of its own.

Consequence for DECISION A: the post-write strip on `remove`/`synthid` is, on
today's OpenCV output, a TRUE NO-OP (the scan finds nothing, the file is not
rewritten). Its value is:

1. A GUARANTEE that the output is provenance-free, independent of the encoder.
   If a future OpenCV build, a `cv::imwrite` overload, or a future non-OpenCV
   output path (direct libpng/libjpeg write, a passthrough mode) starts emitting
   any chunk/marker, the post-write pass catches and strips it.
2. Cheap insurance: the no-op path is one file read + a byte scan (no decode, no
   rewrite). For a normal image this is sub-millisecond.

The plan must NOT claim the remove-path strip is doing active forensic work
today. It is honest insurance. The ACTIVE, load-bearing feature is the
standalone `wmr metadata`, which losslessly strips USER input files that DO
carry metadata (no re-encode, no pixel loss). Section 9 specifies the no-op-on-
clean optimization so the guarantee holds at zero cost.

## 4. Source tree and full interfaces

New directory `src/metadata/`, namespace `wmr::provenance`. Five files. The
three `.cpp` go in both `CMakeLists.txt` SOURCES and `tests/CMakeLists.txt`
LIB_SOURCES (ungated; paths.cpp precedent). The two `.hpp` are header-only.

### `src/metadata/provenance_constants.hpp` (header-only, constexpr)
Single source of truth for every constant, used by BOTH the reporter (scan) and
the remover (strip), so detection and removal cannot drift (the parity test,
section 10, asserts this).

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <array>
#include <string_view>

namespace wmr::provenance {

// Container signatures (magic bytes), sniffed by content, never by extension.
inline constexpr std::byte kPngSig[8] = {
    std::byte{0x89}, std::byte{0x50}, std::byte{0x4E}, std::byte{0x47},
    std::byte{0x0D}, std::byte{0x0A}, std::byte{0x1A}, std::byte{0x0A}};
inline constexpr std::byte kJpegSoi[2] = {std::byte{0xFF}, std::byte{0xD8}};
inline constexpr std::string_view kWebpRiff = "RIFF";   // offset 0
inline constexpr std::string_view kWebpTag = "WEBP";    // offset 8
inline constexpr std::string_view kIsobmffFtyp = "ftyp"; // offset 4

// C2PA storage, VERIFIED against the C2PA spec 2.4, the contentauth/c2pa-rs
// reference implementation, and a hexdump of a real C2PA JPEG fixture (see
// section 5 for the citations). The strip is DUAL: type match + content sniff,
// so a manifest that uses a variant label is still caught by the content sniff.
//
// PNG: ancillary chunk type "caBX" (63 61 42 58; capital B, capital X).
inline constexpr std::string_view kPngC2paChunk = "caBX";
//
// JPEG: there is NO top-level UUID identifying a C2PA APP11. C2PA rides the
// ISO/IEC 19566-5 JUMBF-in-APP11 envelope. The APP11 payload begins with:
//   CI = "JP" (0x4A 0x50), En = 2 bytes (box instance), Z = 4 bytes BE
//   (segment sequence number 1,2,3,...; NOT a 1-byte count),
// followed by the JUMBF superbox bytes. C2PA is identified by the JUMBF
// Description box content-type UUID at payload offset [24..40]:
//   6332706100110010800000aa00389b71  ("c2pa" + the ISO JUMBF UUID tail)
// Quick check: payload[24..28] == {0x63,0x32,0x70,0x61} ("c2pa"). The strip in
// v1 drops ALL APP11 (rare; only C2PA/JUMBF/JPEG-XS use it), so the envelope
// details matter only for the REPORTER's has_c2pa flag, not for removal.
inline constexpr std::byte kJpegApp11C2paMagic[4] = {
    std::byte{0x63}, std::byte{0x32}, std::byte{0x70}, std::byte{0x61}}; // "c2pa"
inline constexpr std::byte kJpegApp11EnvelopeCI[2] = {std::byte{0x4A}, std::byte{0x50}}; // "JP"
inline constexpr std::byte kC2paJumbfTypeUuid16[16] = {
    std::byte{0x63},std::byte{0x32},std::byte{0x70},std::byte{0x61},
    std::byte{0x00},std::byte{0x11},std::byte{0x00},std::byte{0x10},
    std::byte{0x80},std::byte{0x00},std::byte{0x00},std::byte{0xAA},
    std::byte{0x00},std::byte{0x38},std::byte{0x9B},std::byte{0x71}};
//
// ISOBMFF (AVIF/HEIF/MOV/MP4): a "uuid" box (type 75 75 69 64) whose 16-byte
// extended type is D8FEC3D6-1B0E-483C-9297-582887EC4881. (THIS is the constant
// the original blueprint cited; it is correct for ISOBMFF and was wrongly also
// attributed to JPEG.)
inline constexpr std::byte kIsobmffC2paUuid16[16] = {
    std::byte{0xD8},std::byte{0xFE},std::byte{0xC3},std::byte{0xD6},
    std::byte{0x1B},std::byte{0x0E},std::byte{0x48},std::byte{0x3C},
    std::byte{0x92},std::byte{0x97},std::byte{0x58},std::byte{0x28},
    std::byte{0x87},std::byte{0x7E},std::byte{0xC4},std::byte{0x81}};
//
// JUMBF box magics (ISO/IEC 19566-5): "jumb" superbox, "jumd" description box.
inline constexpr std::string_view kJumbfMagic   = "jumb";  // 6A 75 6D 62
inline constexpr std::string_view kJumbfDescMagic = "jumd"; // 6A 75 6D 64

// PNG chunk types the strip parses (everything else is copied verbatim).
inline constexpr std::string_view kPngText   = "tEXt";
inline constexpr std::string_view kPngItxt   = "iTXt";
inline constexpr std::string_view kPngZtxt   = "zTXt";
inline constexpr std::string_view kPngIhdr   = "IHDR";
inline constexpr std::string_view kPngIdat   = "IDAT";
inline constexpr std::string_view kPngIend   = "IEND";

// tEXt/iTXt keys that are ALWAYS dropped (AI-specific). Lowercase compared.
inline constexpr std::string_view kAiDenyKeys[] = {
    "parameters", "prompt", "negative_prompt", "workflow", "comfyui",
    "sd-metadata", "invokeai_metadata", "generation_data", "ai_metadata",
    "dream", "sd:prompt", "sd:negative_prompt", "sd:seed", "sd:steps",
    "sd:sampler", "sd:cfg_scale", "sd:model_hash", "c2pa", "c2pa_chunk"
};

// AI keyword substrings. Applied to (a) the VALUE of a keep-list key, and
// (b) any APP segment / IPTC / XMP bytes. Case-insensitive.
inline constexpr std::string_view kAiSubstrings[] = {
    "stable_diffusion", "stable-diffusion", "comfyui", "automatic1111",
    "invokeai", "midjourney", "dall-e", "dalle", "imagen", "synthid",
    "google_ai", "openai", "c2pa", "trainedalgorithmicmedia",
    "compositesynthetic", "compositewithtrainedalgorithmicmedia",
    "aisystemused", "aipromptinformation", "aigc", "tc260:aigc"
};

// Standard keys KEPT by default (keep_standard=true). 'software' is here AND in
// the substring list: keep UNLESS its value carries an AI substring.
inline constexpr std::string_view kStandardKeepKeys[] = {
    "author", "title", "description", "copyright", "creation time",
    "software", "comment", "disclaimer", "source", "warning"
};

// IPTC "Made with AI" markers (byte-scan; substring match on raw bytes).
// (Same strings appear in kAiSubstrings above; listed here for documentation.)
// TC260 AIGC namespace + IPTC 2025.1 AI fields are covered by kAiSubstrings.

} // namespace wmr::provenance
```

### `src/metadata/png_chunks.{hpp,cpp}`
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include <optional>

namespace wmr::provenance {

struct PngFinding {
    std::string chunk_type;     // "tEXt", "caBX", ...
    std::string detail;         // e.g. "AI key: parameters" or "C2PA manifest"
    bool dropped;               // true if the remover will drop it
};

struct PngReport {
    bool ok = false;            // false => malformed; caller copies input unchanged
    bool has_c2pa = false;
    std::vector<PngFinding> findings;
};

struct PngRewriteResult {
    bool ok = false;            // false => malformed => caller copies input unchanged
    std::vector<std::byte> out; // the rewritten bytes (valid when ok)
    int items_dropped = 0;
    PngReport report;
};

// Scan only (no rewrite). Used by the reporter.
PngReport scan_png(std::span<const std::byte> in);

// Rewrite: drop caBX + AI tEXt/iTXt/zTXt-by-key (+ everything non-essential when
// !keep_standard). Drops WHOLE chunks only, so surviving chunks keep their CRCs
// (no CRC recompute). IDAT/IHDR/IEND/APNG chunks are never touched.
// keep_standard=false (strip-all) drops every chunk except IHDR, IDAT(s), IEND.
PngRewriteResult rewrite_png_strip_ai(std::span<const std::byte> in,
                                      bool keep_standard);

} // namespace wmr::provenance
```

### `src/metadata/jpeg_markers.{hpp,cpp}`
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace wmr::provenance {

struct JpegFinding {
    std::string marker;         // "APP11/C2PA", "APP1/XMP", "APP13/IPTC", "APP1/EXIF"
    std::string detail;
    bool dropped;
};

struct JpegReport {
    bool ok = false;
    bool has_c2pa = false;
    std::vector<JpegFinding> findings;
};

struct JpegRewriteResult {
    bool ok = false;            // false => malformed => copy input unchanged
    std::vector<std::byte> out;
    int items_dropped = 0;
    JpegReport report;
};

JpegReport scan_jpeg(std::span<const std::byte> in);

// Rewrite: from SOI, walk markers. On the first SOS (0xDA) or EOI (0xD9) copy
// the remainder VERBATIM (lossless; also covers progressive/multi-scan JPEG,
// because all bytes after the first SOS are copied unchanged). Before SOS:
//   - drop ALL APP11 (0xEB) (rare; only JUMBF/C2PA/JPEG-XS use it; section 6);
//   - drop APP1/XMP (0xE1) whose payload starts with the Adobe XMP namespace;
//   - drop APP13/IPTC (0xED) whose payload carries an AI marker;
//   - drop any APPn whose payload carries a bare AI marker (kAiSubstrings);
//   - APP1/EXIF (starts "Exif\0\0"): if its raw bytes carry an AI marker, drop
//     the WHOLE APP1 (no IFD rewrite in v1; section 6); else keep verbatim.
//   - copy DQT/DHT/SOF/COM(without AI)/APP0/DRI and all other markers verbatim.
// keep_standard=false drops ALL APPn (except what decode needs) and COM.
JpegRewriteResult rewrite_jpeg_strip_ai(std::span<const std::byte> in,
                                        bool keep_standard);

} // namespace wmr::provenance
```

### `src/metadata/provenance.{hpp,cpp}` (public API + dispatch)
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include <optional>

namespace wmr::provenance {

enum class ContainerFormat { Png, Jpeg, WebPRiff, IsobmffImage, IsobmffVideo, Unknown };

struct ProvenanceFinding {
    ContainerFormat format;
    std::string where;      // "PNG:tEXt:parameters", "JPEG:APP11", "PNG:caBX"
    std::string detail;
};

struct MetadataReport {
    ContainerFormat format = ContainerFormat::Unknown;
    bool supported = false;          // false for WebP/AVIF/... in v1
    bool has_c2pa = false;
    std::optional<std::string> c2pa_note; // issuer/tool substring if found (best-effort)
    std::vector<ProvenanceFinding> findings;
    int ai_text_keys = 0;
};

struct StripOptions {
    bool keep_standard = true;       // default: keep non-AI standard metadata
    bool dry_run = false;            // report only, do not produce output bytes
};

struct StripResult {
    bool ok = false;                 // false => malformed OR unsupported => copy input
    bool supported = false;          // false for WebP/AVIF/... (no strip attempted)
    int items_removed = 0;
    std::vector<std::byte> out;      // valid when ok && supported && !dry_run
    MetadataReport report;
};

ContainerFormat sniff_format(std::span<const std::byte> head); // 12 bytes is enough

MetadataReport report_provenance(std::span<const std::byte> in);

// Dispatch by sniffed format. For Unsupported formats, returns {ok=true,
// supported=false} so the caller copies the input through unchanged.
StripResult strip_provenance(std::span<const std::byte> in, const StripOptions& opts);

} // namespace wmr::provenance
```

`provenance.cpp` dispatches on `sniff_format`: Png -> `scan_png`/`rewrite_png_strip_ai`;
Jpeg -> `scan_jpeg`/`rewrite_jpeg_strip_ai`; anything else -> `{ok=true,
supported=false}`. The reporter for unsupported formats still emits a
`MetadataReport` with `supported=false` (so `wmr metadata file.avif` reports
"AVIF detected, not supported in v1").

## 5. The C2PA constants (VERIFIED against spec + reference impl + real fixture)

The C2PA storage constants were cross-verified three ways: the C2PA Technical
Specification 2.4 (spec.c2pa.org), the `contentauth/c2pa-rs` reference
implementation (`sdk/src/asset_handlers/{png_io,jpeg_io,bmff_io}.rs`,
`sdk/src/jumbf/boxes.rs`), and a hexdump of the real C2PA fixture
`sdk/tests/fixtures/CA.jpg`.

Findings against the original blueprint:

- PNG chunk `"caBX"` (bytes `63 61 42 58`, capital B capital X): CORRECT. C2PA
  spec 2.4: "a chunk type of 'caBX'". The casing is significant (PNG chunk-name
  case encodes ancillary/private/reserved/safe-to-copy bits). Now also in the
  W3C PNG spec. Recommended to precede IDAT (we never touch IDAT regardless).
- JUMBF `"jumb"` (`6A 75 6D 62`) and `"jumd"` (`6A 75 6D 64`): CORRECT.
- ISOBMFF C2PA UUID `D8FEC3D6-1B0E-483C-9297-582887EC4881` (bytes
  `d8fec3d61b0e483c92975828877ec481`): CORRECT, and it is the ISOBMFF uuid-box
  extended type ONLY. `c2pa-rs` `bmff_io.rs`: `const C2PA_UUID = [0xd8,0xfe,
  0xc3,0xd6,0x1b,0x0e,0x48,0x3c,0x92,0x97,0x58,0x28,0x87,0x7e,0xc4,0x81]`.
- JPEG: the blueprint was WRONG. It cited `d8fec3d6...` as the JPEG APP11 UUID.
  That UUID never appears in JPEG. A JPEG parser matching it would silently miss
  EVERY real C2PA manifest. JPEG C2PA uses the ISO/IEC 19566-5 JUMBF-in-APP11
  envelope, identified by the JUMBF Description box content-type UUID
  `63327061-0011-0010-8000-00AA00389B71` at APP11 payload offset [24..40]
  (quick check: payload[24..28] == `63 32 70 61` = "c2pa"). The APP11 payload
  layout is: CI="JP" (`4A 50`) + En (2 B, box instance) + Z (4 B BE, segment
  sequence number 1,2,3,...; NOT the 1-byte count the blueprint claimed) +
  JUMBF superbox bytes. Continuation segments duplicate the LBox+TBox (8 B).
  Max ~64000 B per segment, so a manifest spans multiple APP11 markers grouped
  by matching En.
- Full signed manifests are NEVER stored in XMP. XMP carries only a remote
  manifest reference (a URL) via APP1 `http://ns.adobe.com/xap/1.0/` (JPEG) or
  an `iTXt XML:com.adobe.xmp` chunk (PNG). Stripping the manifest
  (APP11/caBX/uuid) invalidates the binding regardless of the XMP URL.

De-risking design (applied regardless, and now well-justified):

1. JPEG removal: the strip drops ALL APP11 (0xEB) markers unconditionally.
   APP11 is rare (only C2PA/JUMBF/JPEG-XS use it; normal camera/phone/OpenCV
   JPEGs have none), so dropping all APP11 removes any multi-segment C2PA
   manifest WITHOUT parsing the JUMBF envelope, the Z counter, or the En
   grouping. This is the right v1 call: it is simple, complete, and the exact
   JUMBF UUID does not gate removal.
2. JPEG reporting (has_c2pa): set `has_c2pa=true` when an APP11 payload carries
   CI=="JP" AND payload[24..28] == `63 32 70 61` (or the full 16-byte
   `kC2paJumbfTypeUuid16` at [24..40]). This is informational; removal already
   dropped the APP11.
3. PNG: drop any chunk whose type equals `kPngC2paChunk` ("caBX") OR whose
   payload starts with the JUMBF `jumb` magic. The content sniff catches a
   manifest even if the chunk type label varies.
4. ISOBMFF (Phase 3/4): drop any `uuid` box whose 16-byte extended type equals
   `kIsobmffC2paUuid16`, AND any box whose type is `jumb` or whose payload
   starts with the JUMBF magic. (The outer C2PA wrapper is always the `uuid`
   box; the `jumb` superbox lives inside it.)

The parity test (section 10) synthesizes a real APP11 envelope (CI="JP" + En +
Z + a `jumb`/`jumd` superbox carrying the C2PA type UUID), a real `caBX` chunk,
and a bare `jumb` payload, and asserts all are removed.

## 6. Algorithms (corrected, with edge cases and the fail-safe)

### Fail-safe rule (supreme; overrides everything)
On ANY structural anomaly that prevents a safe parse (chunk/marker length would
exceed the buffer, truncated header, missing terminator, etc.), the parser sets
`ok=false` and returns. The CALLER copies the input bytes to the output
UNCHANGED. The strip never produces a truncated or partially-edited file. A
per-chunk/per-marker keyword parse anomaly (e.g. a `tEXt` with no NUL) is
handled conservatively at that granularity: KEEP that chunk/marker verbatim
(we cannot prove it is AI, so the safe default is to keep it), and continue.
File-structure anomalies fail the whole file; content anomalies keep the item.

### PNG walk (`scan_png` / `rewrite_png_strip_ai`)
1. Require `in.size() >= 8` and `in[0..8] == kPngSig`. Else `ok=false`.
2. Require the first chunk type to be `IHDR` (PNG spec: IHDR must be first).
   Else `ok=false`. (We never drop IHDR, so it stays first.)
3. Loop from offset 8:
   - Read `len = be32(in + off)`, `type = in[off+4 .. off+8]`. Need `off+8 <=
     size`. Else `ok=false`.
   - Data range is `[off+8, off+8+len)`. CRC is at `[off+8+len, off+8+len+4)`.
     Require `off+8+len+4 <= size` using 64-bit `size_t` arithmetic (len <=
     2^32-1, so len+12 cannot overflow size_t on 64-bit; still computed as
     `off + (len+12)` with a single overflow check). **DO NOT "clamp" len to
     remaining.** If the declared len runs past EOF, that is malformed: set
     `ok=false` and stop. (The blueprint's "clamp len to remaining" violates
     the fail-safe rule: it would drop chunks based on a wrong length and
     produce a corrupted-but-valid-looking file. BLOCKER; fixed here.)
   - Decide per type:
     - `IHDR`, `IDAT`, `IEND`: always keep (copy 4+len+4 bytes verbatim).
       Multiple IDATs are a single zlib stream split across chunks; we never
       touch any IDAT, so the stream stays intact.
     - `caBX` (or type with `jumb` payload): record finding, drop (skip the
       whole chunk incl. CRC). has_c2pa=true.
     - `tEXt`/`iTXt`: parse keyword = bytes before the first NUL within the
       data range. If no NUL in range: KEEP the chunk verbatim (content
       anomaly, conservative keep) and continue. Lowercase the keyword. If
       keyword in `kAiDenyKeys`: drop. Else if keep_standard: scan the value
       (bytes after the NUL) for `kAiSubstrings`; drop if match, else keep.
       Else (!keep_standard): drop.
     - `zTXt` (v1, no inflate): parse keyword before the first NUL (same NUL
       rule). Drop if keyword in `kAiDenyKeys`. Else keep (value not scanned).
       Phase 2 adds inflate + value scan.
     - `eXIf`, `iCCP`, `sRGB`, `gAMA`, `cHRM`, `pHYs`, `tIME`, `acTL`,
       `fcTL`, `fdAT` (APNG), and any other type: if keep_standard, keep
       verbatim; if !keep_standard, drop the ancillary ones (keep only IHDR/
       IDAT/IEND; NEVER drop acTL/fcTL/fdAT even in strip-all, to avoid
       breaking APNG animation. strip-all drops provenance-ish ancillary
       chunks only: tEXt/iTXt/zTXt/eXIf/iCCP/tIME/sRGB/gAMA/cHRM/pHYs/caBX.
       APNG structural chunks are always preserved.)
   - Stop when type == `IEND` (copy IEND incl. CRC, then advance).
4. After IEND, copy any trailing bytes verbatim (some tools append a trailer;
   the fail-safe preserves it). Do not truncate.
5. Because we only drop WHOLE chunks, every surviving chunk's CRC remains valid
   (PNG CRC covers type+data, not the length field). No CRC arithmetic is ever
   needed. This part of the blueprint is correct.

### JPEG walk (`scan_jpeg` / `rewrite_jpeg_strip_ai`)
1. Require `in.size() >= 2` and `in[0..2] == kJpegSoi`. Else `ok=false`.
2. Loop from `off = 2`:
   - Require `off+1 < size` and `in[off] == 0xFF`. Else `ok=false`.
   - `m = in[off+1]`. Skip any `0xFF` fill bytes before a marker (JPEG allows
     multiple 0xFF padding bytes before the real marker byte; loop while
     `in[off]==0xFF` and `in[off+1]==0xFF`, advancing off).
   - Standalone markers (no length): `m == 0x01` or `0xD0..0xD7` (RSTn) or
     `0xD8` (SOI, shouldn't reappear) -> emit 2 bytes, `off += 2`, continue.
     (RSTn only appears inside entropy data, which we copy verbatim after SOS,
     so this branch rarely fires before SOS; it is defensive.)
   - `m == 0xD9` (EOI) or `m == 0xDA` (SOS): copy `in[off .. size)` VERBATIM
     into the output and finish. This is the lossless path. It also covers
     progressive / multi-scan JPEG: every byte after the first SOS (including
     later DHT/SOS scans and their entropy) is copied unchanged, so pixels are
     bit-identical and we only ever strip markers that precede the first SOS.
   - Otherwise (length-marked marker: APPn 0xE0-0xEF, DQT 0xDB, DHT 0xC4,
     SOF 0xC0/0xC1/0xC2, COM 0xFE, etc.):
     - Read `seg_len = be16(in + off + 2)`. Require `off + 2 + seg_len <= size`
       and `seg_len >= 2`. Else `ok=false`.
     - payload = `in[off+4 .. off+2+seg_len)`.
     - Dispatch:
       - `0xEB` APP11: drop (unconditional; section 5). has_c2pa=true if the
         payload carries `jumb` or the C2PA UUID.
       - `0xE1` APP1 and payload starts `"http://ns.adobe.com/xap/"` (XMP) or
         carries any `kAiSubstrings`: drop. Else if payload starts
         `"Exif\0\0"`: if its raw bytes carry an AI substring, drop the WHOLE
         APP1 (no IFD rewrite; section 6 note); else keep verbatim.
       - `0xED` APP13 and payload carries an `kAiSubstrings`: drop. Else keep.
       - Any other APPn (0xE0/APP0, 0xE2..0xEA, 0xEC..0xEF) whose payload
         carries an `kAiSubstrings`: drop. Else keep.
       - `0xFE` COM whose bytes carry an `kAiSubstrings`: drop. Else keep.
       - All others (DQT/DHT/SOF/DRI/SOF2...): keep verbatim.
     - If !keep_standard: drop ALL APPn and COM (keep DQT/DHT/SOF/DRI and the
       decode-critical markers; still lossless).
     - `off += 2 + seg_len`.
3. If the loop ends without seeing SOS or EOI, `ok=false` (truncated).

EXIF / IFD note (why v1 drops APP1-EXIF whole instead of surgical): rewriting a
TIFF IFD in place changes byte offsets. MakerNotes store ABSOLUTE offsets into
the APP1 segment; any IFD edit shifts them and corrupts the MakerNote (a
forensic artifact). A correct surgical scrub must re-base every offset and fix
or drop MakerNotes. That is ~150+ lines of careful code with its own bug
surface. v1 avoids it: if APP1-EXIF carries an AI marker, the whole APP1 is
dropped (pixels unaffected; EXIF is pure metadata). Standard non-AI EXIF is
kept verbatim. Phase 2 can add a real IFD scrub only if a real-world fixture
needs it. This is safer than the blueprint's "~80 lines" estimate.

### Sniff (`sniff_format`)
Inspect the first 12 bytes, by content (never extension):
- starts with `kPngSig` (8 B) -> `Png`
- starts with `kJpegSoi` -> `Jpeg`
- starts with `RIFF` (B 0-3) and bytes 8-11 == `WEBP` -> `WebPRiff`
- bytes 4-7 == `ftyp` (size at 0-3) -> `IsobmffImage` (or `IsobmffVideo` if a
  `moov`/`mdat` box is also present; v1 treats both as unsupported)
- else `Unknown`

### Byte-scan reporter (`report_provenance`)
On a supported format, run the per-format scan (which populates findings +
has_c2pa). On an unsupported format, populate `format`, `supported=false`, no
findings. The C2PA issuer/tool note is a best-effort substring search on the raw
bytes for known issuers (Microsoft, Adobe, OpenAI, Google, Stability AI, Black
Forest Labs) and tools (GPT-4o, ChatGPT, Sora, DALL-E, Imagen, Firefly). There
is NO CBOR/JUMBF parser in v1; the note is informational and clearly labeled.

## 7. CMake changes (exact, verified paths)

### `CMakeLists.txt` (top-level, ungated; ships in every binary incl lean/AI-off)
Add the three `.cpp` to the `SOURCES` set (currently `CMakeLists.txt:17`).
Insert after `src/cli/progress.cpp` (line 34):
```cmake
    src/metadata/provenance.cpp
    src/metadata/png_chunks.cpp
    src/metadata/jpeg_markers.cpp
```
No `find_package` changes (no new deps in v1). No feature option (ungated).
`target_link_libraries` is unchanged (zlib is NOT added in v1).

### `tests/CMakeLists.txt` (mirror; ungated, paths.cpp precedent)
Add the three `.cpp` to the `LIB_SOURCES` set (currently `tests/CMakeLists.txt:18`),
after the `src/cli/progress.cpp` line (line 27). Add the three test files to
`TEST_SOURCES` (currently line 6):
```cmake
    unit/provenance_test.cpp
    unit/png_chunks_test.cpp
    unit/jpeg_markers_test.cpp
```
Do NOT put these in a `target_sources(wmr_tests PRIVATE ...)` mirror block:
provenance is ungated, so it belongs in the always-compiled `LIB_SOURCES` (a
mirror block is for feature-gated sources and would be a no-op here since
`wmr_tests` already exists; the comment at `tests/CMakeLists.txt:60` explains
this). The header-only `provenance_constants.hpp` needs no listing.

### Lean / AI-off build check
The metadata layer includes no OpenCV/FFmpeg/NCNN/ORT headers and is behind no
`WMR_*` macro, so the lean build compiles it unchanged. Verify with:
`cmake --preset mac-homebrew-Debug -DWMR_BUILD_AI_DENOISE=OFF -DWMR_BUILD_REGEN=OFF -B build-off && cmake --build build-off --target wmr`,
then `nm build-off/wmr | grep provenance` (symbols present).

## 8. CLI wiring (exact)

### `src/cli/cli_app.hpp`
- Add `Metadata` to the `CliMode` enum (after `Cache`).
- Add to `CliOptions`:
  ```cpp
  // Provenance / C2PA metadata.
  bool keep_provenance = false;   // remove/synthid/video: opt OUT of the default strip
  // metadata subcommand:
  bool metadata_dry_run = false;  // report only, do not write
  bool metadata_strip_all = false;// drop ALL non-essential metadata (not just AI)
  ```
  (`keep_provenance` defaults false: the default is to strip. The flag is the
  opt-OUT.)
- Add an inline `provenance_strip_help_text()` (mirrors
  `synthid_attack_help_text()`), so the wording test can assert on it without
  linking `cli_app.cpp`. It states plainly: the strip is lossless on pixels,
  container-level, v1 = PNG + JPEG, and the remove/synthid default strip is a
  defensive guarantee (on today's OpenCV output it is a no-op scan).

### `src/cli/cli_app.cpp`
- Register a `metadata` subcommand (after `cache_cmd`, ~line 723):
  ```cpp
  auto* metadata_cmd = app.add_subcommand("metadata",
      "Report and strip C2PA / AI-provenance metadata (lossless on pixels)");
  metadata_cmd->add_option("input", opts.input_path, "Input image or directory")
      ->required()->check(CLI::ExistingPath);
  metadata_cmd->add_option("-o,--output", opts.output_path,
      "Output path (file; for a directory input, defaults to <input>_clean/)");
  metadata_cmd->add_flag("--dry-run", opts.metadata_dry_run,
      "Report findings only; do not write");
  metadata_cmd->add_flag("--strip-all", opts.metadata_strip_all,
      "Drop all non-essential metadata, not just AI markers");
  metadata_cmd->add_flag("-r,--recursive", opts.recursive,
      "Process directories recursively");
  add_common(metadata_cmd);
  ```
- Add `--keep-provenance` to `remove_cmd`, `synthid_cmd`, and `video_cmd`:
  ```cpp
  remove_cmd->add_flag("--keep-provenance", opts.keep_provenance,
      "Keep C2PA/AI-provenance metadata in the output (default: strip it). "
      "v1 strip covers PNG and JPEG; video output is already re-encoded.");
  ```
- Add an `else if (metadata_cmd->parsed()) { opts.mode = CliMode::Metadata; }`
  arm at the mode-setting block (~line 772).
- Add a dispatch arm in the `switch` (~line 786):
  ```cpp
  case CliMode::Metadata:
      rc = process_metadata(opts);
      break;
  ```
- Implement `static int process_metadata(const CliOptions& opts)` (new function;
  template off `process_cache`). It:
  - If `input_path` is a directory: iterate files (recursive when `-r`), sniff
    each, `report_provenance` + (unless `--dry-run`) `strip_provenance` writing
    to `<output>/<rel>` or `<input>_clean/<rel>`. This is a SEPARATE lightweight
    loop, NOT `batch_process` (which loads into a `cv::Mat` and re-encodes; the
    metadata path must be lossless byte IO). Accept any regular file; skip
    unsupported formats with a one-line report.
  - If a file: read whole file as `std::vector<std::byte>`,
    `report_provenance`; print findings; if `--dry-run` exit 0; else
    `strip_provenance`, and if `supported && items_removed>0` write `out` (else
    copy input through). Exit 0.

### The post-write strip (DECISION A): see section 9.

## 9. The post-write strip hook (exact sites, atomic write, no-op-on-clean)

Add one helper in `src/metadata/provenance.hpp` (callable from cli_app.cpp and
batch_processor.cpp):
```cpp
// Post-write provenance guarantee for the remove/synthid path. Reads the just-
// written file, and if it carries any C2PA/AI metadata, strips it in place
// (temp file + rename for atomicity). On a clean file (the common case on
// OpenCV output) the scan finds nothing and the file is NOT rewritten (true
// no-op, no extra write I/O). On unsupported formats (WebP in v1) it is a no-op
// (reported once at debug level). Never throws; returns a one-line summary.
// Called ONLY when opts.keep_provenance is false.
struct PostWriteResult { bool ok; bool rewritten; int items_removed; std::string format_note; };
PostWriteResult post_write_provenance_strip(const std::string& file_path,
                                            bool keep_standard);
```

Hook it at the two `cv::imwrite` sites, guarded by `!opts.keep_provenance` and
by a SUCCESSFUL write:

`src/cli/cli_app.cpp` after line 394 (`Saved: ...`):
```cpp
if (!cv::imwrite(output, image, params)) { /* unchanged error */ }
// Guaranteed provenance-free output (DECISION A). Defensive on today's OpenCV
// output (which already strips); the guarantee holds if the encoder changes.
if (!opts.keep_provenance) {
    auto pr = wmr::provenance::post_write_provenance_strip(output, /*keep_standard=*/true);
    if (pr.rewritten && pr.items_removed > 0)
        spdlog::info("Stripped {} provenance item(s) from output", pr.items_removed);
}
spdlog::info("Saved: {}", output);  // (move the Saved line after the strip, or keep before; pick before so the path is logged regardless)
```
Note: order the `Saved:` log AFTER the strip so the path is logged once, after
the final bytes are on disk.

`src/cli/batch_processor.cpp` after line 193 (`cv::imwrite` success), same guard
on `!opts.keep_provenance`, inside `process_single`.

Atomic write: `post_write_provenance_strip` reads the file, scans; if
`items_removed == 0`, return `{ok=true, rewritten=false}` WITHOUT writing. If
`items_removed > 0`, write `out` to `<file>.wmrprov.tmp` in the same directory,
`std::filesystem::rename` over the original (atomic on POSIX same-filesystem),
and clean up the temp on any error. On any exception, return `{ok=false,...}`
and LEAVE THE ORIGINAL INTACT (the strip never makes the output worse).

Composes with existing flags: `--keep-provenance` is independent of
`--denoise`/`--synthid-attack`/geometry flags (it only gates the post-write
pass). On `video`, add the flag for CLI symmetry but document it as a v1 no-op
(the re-encode already drops input C2PA boxes; section 11).

No clash with the update-check chokepoint (`cli_app.cpp:816`, runs AFTER the
dispatch returns) or the progress reporter (the strip is fast; a rewrite only
happens when there is something to strip, and it logs one line).

## 10. Tests (full plan)

All fixtures are SYNTHESIZED IN-CODE (no LFS, no binary assets). Tests live in
`tests/unit/`.

### `tests/unit/png_chunks_test.cpp` (`[metadata][png]`)
- `SECTION("valid PNG with caBX + AI tEXt + IDAT + IEND"`): synthesize bytes =
  sig + IHDR + `tEXt parameters=...` + `tEXt author=me` + IDAT + `caBX` +
  IEND. Assert `rewrite_png_strip_ai(...,true).ok`, `items_dropped >= 2`
  (parameters + caBX), `author` survives, IDAT bytes byte-identical to input,
  and `cv::imdecode(out) == cv::imdecode(in)` (bit-identical pixels).
- `SECTION("strip-all drops ancillary, keeps APNG structural"`): add `acTL`/
  `fcTL`/`fdAT`; assert they survive even with `keep_standard=false`.
- `SECTION("denylist key coverage"`): one `tEXt` per `kAiDenyKeys` element;
  assert all dropped.
- `SECTION("software kept unless value has AI substring"`): `tEXt
  Software=Photoshop` kept; `tEXt Software=ComfyUI` dropped.
- `SECTION("zTXt matched by key only"`): `zTXt parameters=<compressed>` dropped
  by key; `zTXt Author=<compressed>` kept (value not scanned in v1).
- `SECTION("malformed: declared len runs past EOF"`): assert `ok=false` (the
  fail-safe; verifies the no-clamp fix).
- `SECTION("malformed: not PNG"`): random bytes -> `ok=false`.
- `SECTION("trailing bytes after IEND preserved"`).
- `SECTION("content sniff: chunk named X with jumb payload dropped"`): a
  non-caBX chunk whose payload starts with `jumb` is dropped (de-risking).

### `tests/unit/jpeg_markers_test.cpp` (`[metadata][jpeg]`)
- `SECTION("SOI+APP0+APP11(real C2PA envelope)+APP1/XMP-AI+APP13/IPTC-AI+SOS+entropy+EOI"`:
  synthesize a REAL C2PA APP11 per section 5: payload = CI="JP" + En(2) + Z(4
  BE = 1) + LBox + "jumb" + LBox + "jumd" + the 16-byte
  `kC2paJumbfTypeUuid16`. Assert `rewrite_jpeg_strip_ai` drops the APP11,
  drops XMP-AI and IPTC-AI, keeps APP0, entropy+EOI byte-identical, and
  `cv::imdecode(out)==cv::imdecode(in)`; `report.has_c2pa==true` on the input.
- `SECTION("multi-segment C2PA: two APP11 with same En, Z=1 and Z=2"`: assert
  BOTH dropped (proves the drop-all-APP11 decision covers multi-segment
  manifests without parsing Z/En).
- `SECTION("APP1/EXIF with AI marker dropped whole"`): APP1 starting
  `Exif\0\0` + a `Software=ComfyUI` string in the IFD -> whole APP1 dropped;
  without the marker -> kept verbatim.
- `SECTION("progressive multi-scan: bytes after first SOS identical"`):
  synthesize SOI + DQT + SOF2 + DHT + APP11/c2pa + SOS + <entropy with a
  second SOS> + EOI; assert APP11 gone and everything from the first SOS
  byte-identical.
- `SECTION("marker preceded by 0xFF fill bytes"`): parses correctly.
- `SECTION("malformed: seg_len runs past EOF"`): `ok=false`.
- `SECTION("strip-all drops APPn+COM, keeps DQT/DHT/SOF, lossless"`.

### `tests/unit/provenance_test.cpp` (`[metadata]`)
- `SECTION("sniff: PNG/JPEG/WebP/ISOBMFF/unknown"`): one case each.
- `SECTION("report: C2PA UUID + Google issuer + trainedAlgorithmicMedia"`).
- `SECTION("dispatch routing"`): PNG and JPEG route to the right rewriter;
  WebP/Unknown return `{ok=true,supported=false}`.
- `SECTION("dry-run produces empty out but full report"`).
- `SECTION("detection-removal PARITY"`): the central invariant. For a corpus of
  synthesized fixtures (one per deny key, one per AI substring, the C2PA chunk,
  a `jumb` payload, an XMP-AI APP1, an IPTC-AI APP13), assert that EVERY
  finding `report_provenance` flags with `dropped=true` is ABSENT from
  `strip_provenance`'s `out`. This locks detection and removal to the same
  constants (single source of truth in `provenance_constants.hpp`).
- `SECTION("bit-identical pixels regression"`): for each supported fixture,
  `cv::imdecode(in)==cv::imdecode(out)` exactly (max abs diff 0).

### Wording test (honesty lock; `[metadata][cli][wording]`)
Add to `tests/unit/provenance_test.cpp` (or a new `metadata_cli_test.cpp`):
assert `wmr::provenance_strip_help_text()` contains "lossless", "PNG", "JPEG",
"provenance", and does NOT contain "removes C2PA" as an absolute claim (mirror
of the `synthid_attack_cli_test.cpp` forbidden-claim pattern). This satisfies
DECISION B (factual, no over-claim, no disclaimer).

### Regression check (no existing test breaks)
Confirmed in section 2: `visible_pipeline_test.cpp` operates on `cv::Mat`, not a
written file; no test byte-compares remove/synthid output. The post-write strip
adds no new test dependency. Run `ctest --test-dir build --output-on-failure`
and `./build/tests/wmr_tests` to confirm green.

## 11. Phased scope (v1 bounded)

- v1 (this plan): PNG strip, JPEG strip, byte-scan reporter, standalone `wmr
  metadata`, default post-write strip on `remove`/`synthid`. AVIF/HEIF/JPEG-XL/
  WebP/MP4 unsupported (reported, pass-through).
- Phase 2: WebP RIFF strip; zTXt inflate (link zlib via
  `find_package(ZLIB REQUIRED)`; it is transitively present); JPEG surgical IFD
  scrub with MakerNote re-basing (only if a real fixture needs it); Samsung/
  Apple trailer signatures.
- Phase 3: AVIF/HEIF/JPEG-XL ISOBMFF strip (image; build new buffer from kept
  boxes, fail-safe on any size anomaly).
- Phase 4: MP4/MOV ISOBMFF offset-preserving strip (retype the C2PA uuid box to
  `free` in place, so absolute offsets in the moov atom stay valid). This is
  when `wmr video` would gain an EXPLICIT strip (today its re-encode is the
  implicit one).

### v1 video stance (precise)
`wmr video` re-encodes via FFmpeg (`video_writer.cpp`:
`avcodec_find_encoder_by_name` + `avcodec_send_frame` + a fresh
`AVFormatContext` + `avformat_write_header`). There is NO stream copy / remux.
FFmpeg therefore writes a fresh moov/mdat from the encoded packets and does NOT
copy unknown input boxes. A C2PA "uuid"/"jumb" box present in the input MP4 is
NOT carried into the output. So in v1 the video output is already free of
input C2PA by construction. v1 does NOT run an explicit ISOBMFF strip and does
NOT make a byte-level guarantee (Phase 4). FFmpeg may write its own `udta`
("encoder: Lavf") tag; that is encoder metadata, not C2PA/AI provenance, and is
out of scope. `--keep-provenance` is accepted on `video` for CLI symmetry and is
a documented no-op in v1.

## 12. Docs

### `README.md`
- "What it does" table (`README.md:36`): add a row:
  `| C2PA / AI provenance metadata | AI-generated images | Lossless container-level strip (PNG, JPEG); default on remove/synthid | Lossless on pixels |`
- Add a "Provenance metadata" subsection under "Usage reference" (after
  "cache subcommand", ~line 222). Plain factual description: what `wmr metadata`
  does, the default strip on remove/synthid, `--keep-provenance`, the v1 scope
  (PNG/JPEG; WebP/AVIF/MP4 later), and the honest note that the remove-path
  strip is a defensive guarantee (on today's OpenCV output it is a no-op scan).
  No disclaimer, no ethical framing (DECISION B). No em dashes; plain
  punctuation.
- Update the Table of contents (`README.md:11`) with the new subsection.
- "Credits" (`README.md:333`): add a line attributing
  `wiltodelta/remove-ai-watermarks` (MIT; studied and reimplemented the
  byte-level container approach; no verbatim code copied). Do NOT mention
  `cebeuq/Synthid-Bypass` (no license; nothing copied).

### `CLAUDE.md`
- Add a "Provenance metadata strip" section: design (container-level, lossless
  on pixels, never decodes; the fail-safe rule; the dual type+content C2PA
  detection; v1 = PNG/JPEG; the OpenCV re-encode reality and why the remove-path
  strip is defensive; `--keep-provenance`; parity invariant; zTXt/EXIF
  limitations). Note the metadata layer stays OUT of `WatermarkEngine`. Note
  the CMake pattern (ungated SOURCES + LIB_SOURCES, paths.cpp precedent).

### `LICENSE-THIRD-PARTY.md`
- Attribution section for `wiltodelta/remove-ai-watermarks` (MIT): "Studied and
  reimplemented the byte-level container strip approach. No verbatim code
  copied." with the MIT notice. State explicitly that `cebeuq/Synthid-Bypass`
  has NO license and NOTHING was copied from it.

### `CHANGELOG.md`
- Under `[Unreleased]` -> `### Added`:
  - New `wmr metadata` subcommand (report + lossless strip of C2PA / AI
    provenance metadata from PNG and JPEG; format-agnostic byte-scan reporter).
  - `wmr remove` and `wmr synthid` now strip provenance metadata from the
    output by default (opt out with `--keep-provenance`). Honest note: on
    today's OpenCV output this is a no-op scan (OpenCV already strips all
    metadata on write); the pass guarantees the output stays provenance-free
    independent of the encoder.
  - `--keep-provenance` flag (remove/synthid/video).
  - v1 scope note: PNG + JPEG; WebP/AVIF/HEIF/JPEG-XL/MP4 are later phases.

## 13. Security considerations (parsing untrusted bytes)

- Integer overflow: PNG `len` is 4 bytes (max 2^32-1). All offset arithmetic
  uses 64-bit `size_t` and validates `off + (len + 12) <= size` as a single
  check before reading. ISOBMFF 64-bit `largesize` is validated
  `largesize <= remaining` to reject a hostile 16-EB size.
- OOM: the parser NEVER allocates a buffer sized by an untrusted length. It
  operates in a single streaming pass over the input buffer. The only
  allocation is the output `vector<byte>`, reserved from the input size (a
  known, already-loaded buffer), and it only ever shrinks (we drop, never grow).
  No `resize(untrusted_len)`.
- Zip bomb: v1 does not inflate (zTXt value not scanned). Phase 2 inflate MUST
  cap the decompressed size (e.g. 64 MB per chunk) and bail to keep-verbatim on
  exceed.
- CRC DoS: not applicable (we never compute CRC; dropping whole chunks keeps
  surviving CRCs valid).
- Truncation: the fail-safe rule means a malformed input is copied UNCHANGED,
  never truncated. The worst case for a hostile file is "no strip happens"
  (kept verbatim), which is safe (no data loss, no crash).
- No UB: every read is preceded by a bounds check; `std::span` access is
  bounds-checked in debug builds.

## 14. Issues found in the original blueprint (the review)

Listed by criterion, each with the fix applied above.

### Correctness
1. BLOCKER. "Clamp len to remaining" for the PNG chunk walk. This violates the
   fail-safe rule (it would drop chunks based on a wrong length and produce a
   corrupted-but-valid-looking file). Fix: section 6 step 3, bounds-check and
   set `ok=false` on over-run; never clamp.
2. BLOCKER. The blueprint cited the C2PA UUID `d8fec3d61b0e483c92975828877ec481`
   as a JPEG APP11 identifier. That UUID is REAL but is the ISOBMFF uuid-box
   extended type ONLY; it never appears in JPEG. A JPEG parser matching it would
   silently miss every real C2PA manifest. Verified against the C2PA spec 2.4,
   the `contentauth/c2pa-rs` reference impl, and a real fixture hexdump. Fix:
   section 5, JPEG removal drops ALL APP11 (so the exact constant does not gate
   removal); JPEG reporting detects C2PA via the real JUMBF envelope ("JP" +
   En + Z) and the JUMBF description UUID `63327061-0011-0010-8000-00AA00389B71`
   at payload[24..40]. (PNG `"caBX"` and `jumb` were CORRECT; ISOBMFF UUID
   CORRECT and re-scoped to ISOBMFF.)
3. MAJOR. APP11 multi-segment C2PA manifests: the blueprint claimed a 1-byte
   segment count. The real envelope (ISO/IEC 19566-5) uses a 4-byte big-endian Z
   sequence number (1,2,3,...), grouped by a 2-byte En, with continuation
   segments duplicating LBox+TBox. Fix: section 5/6, drop ALL APP11 (covers any
   multi-segment manifest without parsing Z/En); section 10 multi-segment test.
4. MAJOR. EXIF surgical IFD scrub "~80 lines" underestimates the MakerNote
   offset landmine and its bug surface. Fix: section 6, v1 drops APP1-EXIF whole
   when it carries an AI marker; keep verbatim otherwise; defer surgical scrub
   to Phase 2.
5. MAJOR. JPEG marker walk did not handle 0xFF fill bytes before a marker, nor
   multi-scan progressive JPEG robustly. Fix: section 6, skip 0xFF fill; copy
   everything from the first SOS verbatim (covers multi-scan).
6. MINOR. "Software in denylist AND keep-list" was conflated. Fix: section 4,
   three-tier logic (deny keys always drop; keep keys drop only if value has an
   AI substring; unknown keys keep by default).
7. MINOR. APNG structural chunks (acTL/fcTL/fdAT) must always be preserved,
   even in strip-all. Fix: section 6.
8. MINOR. Trailing bytes after IEND must be copied verbatim. Fix: section 6.

### Regressions
9. MAJOR. The post-write strip changes remove/synthid output bytes. Verified
   (section 2) that NO existing test byte-compares that output, so no fixture
   breaks. The strip is also a no-op on clean output (section 9), so even a
   byte-comparing user sees no change on today's OpenCV output.
10. MINOR. New `CliMode::Metadata` and subcommand must not shift existing
    parsing. Fix: section 8, added as an additive arm; `require_subcommand(0,1)`
    and the default-AutoRemove fallback are unchanged.

### Best practices / security
11. MAJOR. "Clamp" and unspecified OOM handling. Fix: section 13, never allocate
    on an untrusted length; 64-bit size_t arithmetic; fail-safe keep-unchanged.
12. MINOR. Namespace and header hygiene. Fix: `wmr::provenance`; pure TUs
    include only `<span>/<byte>/<vector>/...`, no OpenCV/FFmpeg (section 4).

### Completeness
13. MAJOR. The original plan omitted the DECISION A mechanism entirely (it
    argued AGAINST a remove-path strip). Fix: sections 0, 8, 9 (post-write hook,
    `--keep-provenance`, atomic temp+rename, no-op-on-clean).
14. MAJOR. CMake listings were underspecified. Fix: section 7, exact ungated
    SOURCES + LIB_SOURCES + TEST_SOURCES (paths.cpp precedent; not a mirror
    block).
15. MINOR. Wording/honesty test omitted. Fix: section 10 (provenance wording
    lock, DECISION B).
16. MINOR. The standalone `metadata` directory batch path would have reused
    `batch_process` (which re-encodes via OpenCV, not lossless). Fix: section 8,
    a separate lightweight byte-IO loop.

### Integration
17. MAJOR. The remove-path strip is a defensive no-op today (OpenCV already
    strips everything; section 3). The plan states this honestly instead of
    over-claiming active forensic work.
18. MINOR. `--keep-provenance` composes cleanly (independent of denoise/synthid
    /geometry flags; section 9). No clash with the update-check chokepoint
    (`cli_app.cpp:816`, runs after dispatch) or the progress reporter.
