// GUI options validation (Task 5): the full API options matrix as a pure
// function, plus the download-path helpers (sanitize_filename, pct_encode,
// mime_for). Table-driven per the plan: every valid set, every invalid value,
// every invalid combination, the ai-denoise feature gate, presets checked
// against kStillPresetNames, and the 64 KB options-part cap.
#include <catch2/catch_test_macros.hpp>

#include <string>

#include "detection/still_geometry.hpp"
#include "gui/api.hpp"
#include "gui/options.hpp"

using namespace wmr::gui;
using nlohmann::json;

namespace {

struct Checked {
    bool ok = false;
    std::string code, msg;
    JobOptions opts;
    explicit Checked(const std::string& text, const ApiFeatures& f) {
        ok = validate_job_options(json::parse(text), f, opts, code, msg);
    }
    explicit Checked(const std::string& text, const ApiFeatures& f, int /*text_part*/) {
        ok = validate_options_part(text, f, opts, code, msg);
    }
};

}  // namespace

TEST_CASE("gui options: valid sets", "[gui-options]") {
    const ApiFeatures with_ai{true}, no_ai{false};
    const char* valid[] = {
        "{}",
        R"({"denoise":"off"})",
        R"({"denoise":"soft"})",
        R"({"denoise":"ns"})",
        R"({"denoise":"telea"})",
        R"({"denoise":"ai"})",
        R"({"legacy":true})",
        R"({"forceRemove":true})",
        R"({"keepProvenance":true})",
        // The documented valid pair (CLI: --force --legacy).
        R"({"legacy":true,"forceRemove":true})",
        R"({"rect":[10,20,200,300]})",
        R"({"rect":[0,0,8,8]})",
        R"({"rect":[96,96,48,48]})",
        R"({"geoPreset":"gemini36-portrait"})",
        R"({"geoPreset":"gemini36-large"})",
        R"({"geoPreset":"gemini38-2k-portrait"})",
        // One geometry source at a time composes with the non-geometry flags.
        R"({"denoise":"telea","geoPreset":"gemini38-2k-portrait","keepProvenance":true})",
        R"({"denoise":"ns","rect":[10,10,100,100],"forceRemove":false,"legacy":false})",
    };
    for (const char* text : valid) {
        CAPTURE(text);
        Checked c(text, with_ai);
        INFO("code=" << c.code << " msg=" << c.msg);
        REQUIRE(c.ok);
    }
    // Defaults when nothing is set.
    const Checked defaults("{}", with_ai);
    REQUIRE(defaults.opts.denoise == "off");
    REQUIRE_FALSE(defaults.opts.legacy);
    REQUIRE_FALSE(defaults.opts.geo_preset.has_value());
    REQUIRE_FALSE(defaults.opts.rect.has_value());
    REQUIRE_FALSE(defaults.opts.keep_provenance);
    REQUIRE_FALSE(defaults.opts.force);

    const Checked r(R"({"rect":[10,20,200,300]})", with_ai);
    REQUIRE(r.opts.rect.has_value());
    REQUIRE(*r.opts.rect == cv::Rect(10, 20, 200, 300));

    const Checked preset(R"({"geoPreset":"gemini36-large"})", with_ai);
    REQUIRE(preset.opts.geo_preset == std::optional<std::string>("gemini36-large"));

    const Checked pair(R"({"legacy":true,"forceRemove":true,"denoise":"ai"})", with_ai);
    REQUIRE(pair.ok);
    REQUIRE(pair.opts.legacy);
    REQUIRE(pair.opts.force);
    REQUIRE(pair.opts.denoise == "ai");
}

TEST_CASE("gui options: ai denoise gated on the feature", "[gui-options]") {
    const ApiFeatures no_ai{false};
    const Checked c(R"({"denoise":"ai"})", no_ai);
    REQUIRE_FALSE(c.ok);
    REQUIRE(c.code == "invalid_option");
}

TEST_CASE("gui options: presets validated against kStillPresetNames", "[gui-options]") {
    const ApiFeatures f{true};
    for (const char* name : wmr::kStillPresetNames) {
        CAPTURE(name);
        const Checked good(R"({"geoPreset":")" + std::string(name) + R"("})", f);
        INFO("code=" << good.code << " msg=" << good.msg);
        REQUIRE(good.ok);
        // A corrupted name must never validate (the table, not a prefix match).
        const Checked bad(R"({"geoPreset":")" + std::string(name) + R"(x"})", f);
        REQUIRE_FALSE(bad.ok);
        REQUIRE(bad.code == "invalid_option");
    }
    const Checked not_in_table(R"({"geoPreset":"gemini35-portrait"})", f);
    REQUIRE_FALSE(not_in_table.ok);
    REQUIRE(not_in_table.code == "invalid_option");
}

TEST_CASE("gui options: invalid values", "[gui-options]") {
    const ApiFeatures with_ai{true};
    const char* invalid[] = {
        R"({"denoise":"bogus"})",
        R"({"denoise":"AI"})",
        R"({"denoise":""})",
        R"({"denoise":1})",
        R"({"denoise":null})",
        R"({"geoPreset":42})",
        R"({"geoPreset":null})",
        R"({"rect":[10,10,4,4]})",          // w < 8
        R"({"rect":[10,10,100,4]})",        // h < 8
        R"({"rect":[-1,10,100,100]})",      // x < 0
        R"({"rect":[10,-2,100,100]})",      // y < 0
        R"({"rect":[10,10,100]})",          // 3 elements
        R"({"rect":[10,10,100,100,1]})",    // 5 elements
        R"({"rect":[10,10,100.5,100]})",    // non-integer
        R"({"rect":[10.0,10,100,100]})",    // float, even integral-valued
        R"({"rect":[10,10,"100",100]})",    // string element
        R"({"rect":[2147483648,0,8,8]})",   // overflows int
        R"({"rect":"10,10,100,100"})",
        R"({"rect":{}})",
        R"({"rect":null})",
        R"({"legacy":1})",
        R"({"legacy":"true"})",
        R"({"forceRemove":0})",
        R"({"keepProvenance":"yes"})",
        R"({"bogus":true})",
        R"({"Denoise":"ns"})",              // unknown key (case-sensitive names)
    };
    for (const char* text : invalid) {
        CAPTURE(text);
        const Checked c(text, with_ai);
        REQUIRE_FALSE(c.ok);
        INFO("msg=" << c.msg);
        REQUIRE(c.code == "invalid_option");
        REQUIRE_FALSE(c.msg.empty());
    }
}

TEST_CASE("gui options: invalid combinations", "[gui-options]") {
    const ApiFeatures f{true};
    const char* combos[] = {
        R"({"legacy":true,"rect":[10,10,100,100]})",
        R"({"legacy":true,"geoPreset":"gemini36-portrait"})",
        R"({"forceRemove":true,"rect":[10,10,100,100]})",
        R"({"forceRemove":true,"geoPreset":"gemini36-portrait"})",
        R"({"rect":[10,10,100,100],"geoPreset":"gemini36-portrait"})",
        R"({"legacy":true,"forceRemove":true,"rect":[0,0,8,8]})",  // valid pair + rect still rejected
    };
    for (const char* text : combos) {
        CAPTURE(text);
        const Checked c(text, f);
        REQUIRE_FALSE(c.ok);
        REQUIRE(c.code == "invalid_combination");
        REQUIRE_FALSE(c.msg.empty());
    }
    // The spec pins the legacy message verbatim.
    const Checked legacy_rect(R"({"legacy":true,"rect":[10,10,100,100]})", f);
    REQUIRE(legacy_rect.msg ==
            "rect/geoPreset apply to the V2 profile only; legacy (V1) uses fixed positions");
    const Checked legacy_preset(R"({"legacy":true,"geoPreset":"gemini36-portrait"})", f);
    REQUIRE(legacy_preset.msg ==
            "rect/geoPreset apply to the V2 profile only; legacy (V1) uses fixed positions");
}

TEST_CASE("gui options: options part (raw text entry)", "[gui-options]") {
    const ApiFeatures f{true};
    // Absent/empty part: defaults, valid.
    const Checked empty("", f, 0);
    REQUIRE(empty.ok);
    REQUIRE(empty.opts.denoise == "off");

    // Over the 64 KB cap: rejected as invalid_option BEFORE parsing (this
    // text is also invalid JSON; the cap must win).
    std::string huge(65537, 'a');
    const Checked too_big(huge, f, 0);
    REQUIRE_FALSE(too_big.ok);
    REQUIRE(too_big.code == "invalid_option");

    // Under the cap but not JSON.
    const Checked bad_json("{", f, 0);
    REQUIRE_FALSE(bad_json.ok);
    REQUIRE(bad_json.code == "invalid_option");

    // Valid JSON, wrong shape.
    const Checked not_object(R"([1,2,3])", f, 0);
    REQUIRE_FALSE(not_object.ok);
    REQUIRE(not_object.code == "invalid_option");

    // A normal options part round-trips through the matrix.
    const Checked fine(R"({"denoise":"ns","rect":[10,10,100,100]})", f, 0);
    REQUIRE(fine.ok);
    REQUIRE(fine.opts.denoise == "ns");
    REQUIRE(fine.opts.rect.has_value());
}

TEST_CASE("gui api: sanitize_filename", "[gui-options]") {
    REQUIRE(sanitize_filename("photo.png", 0) == "photo.png");
    REQUIRE(sanitize_filename("/etc/passwd", 0) == "passwd");
    REQUIRE(sanitize_filename("C:\\Users\\x\\img.png", 0) == "img.png");
    REQUIRE(sanitize_filename("../../evil.png", 0) == "evil.png");
    REQUIRE(sanitize_filename("a\"b'c.png", 0) == "abc.png");
    REQUIRE(sanitize_filename("a:b*c?d.png", 0) == "abcd.png");
    REQUIRE(sanitize_filename("a<b>c|d.png", 0) == "abcd.png");
    REQUIRE(sanitize_filename("a\x01\x02" "b.png", 0) == "ab.png");
    // UTF-8 passes through untouched (only ASCII control/forbidden bytes strip).
    REQUIRE(sanitize_filename("h\xc3\xa9llo.png", 0) == "h\xc3\xa9llo.png");
    // Windows reserved stems get a suffix, before the extension when there is one.
    REQUIRE(sanitize_filename("con.png", 0) == "con_.png");
    REQUIRE(sanitize_filename("CON", 3) == "CON_");
    REQUIRE(sanitize_filename("com7.tar.gz", 0) == "com7_.tar.gz");
    REQUIRE(sanitize_filename("lpt1.png", 0) == "lpt1_.png");
    REQUIRE(sanitize_filename("nul.txt", 0) == "nul_.txt");
    REQUIRE(sanitize_filename("aux.png", 0) == "aux_.png");
    REQUIRE(sanitize_filename("prn.png", 0) == "prn_.png");
    // Exact-stem match only.
    REQUIRE(sanitize_filename("console.png", 0) == "console.png");
    // Empty after stripping falls back to image<N>.
    REQUIRE(sanitize_filename("", 5) == "image5");
    REQUIRE(sanitize_filename("/", 2) == "image2");
    REQUIRE(sanitize_filename("\x1f", 7) == "image7");
    // Length cap: 120 bytes, on a UTF-8 boundary.
    const std::string long_name(300, 'x');
    REQUIRE(sanitize_filename(long_name, 0).size() == 120);
    std::string accents;  // 400 bytes of 2-byte UTF-8 chars
    for (int i = 0; i < 200; ++i) accents += "\xc3\xa9";
    const std::string capped = sanitize_filename(accents, 0);
    REQUIRE(capped.size() <= 120);
    REQUIRE((capped.size() % 2) == 0);  // did not split a UTF-8 pair
}

TEST_CASE("gui api: pct_encode (RFC 5987)", "[gui-options]") {
    REQUIRE(pct_encode("photo.png") == "photo.png");
    REQUIRE(pct_encode("a-b_c.d") == "a-b_c.d");
    REQUIRE(pct_encode("A_Z-09.") == "A_Z-09.");
    REQUIRE(pct_encode("my file.png") == "my%20file.png");
    REQUIRE(pct_encode("h\xc3\xa9llo.png") == "h%C3%A9llo.png");
    REQUIRE(pct_encode("a~b.png") == "a%7Eb.png");  // '~' is NOT in the keep set
    REQUIRE(pct_encode("a\"b") == "a%22b");
    REQUIRE(pct_encode("a'b") == "a%27b");
    REQUIRE(pct_encode("100%") == "100%25");
    REQUIRE(pct_encode("") == "");
}

TEST_CASE("gui api: mime_for", "[gui-options]") {
    REQUIRE(mime_for("png") == "image/png");
    REQUIRE(mime_for("jpg") == "image/jpeg");
    REQUIRE(mime_for("jpeg") == "image/jpeg");
    REQUIRE(mime_for("webp") == "image/webp");
    REQUIRE(mime_for("") == "application/octet-stream");
    REQUIRE(mime_for("heic") == "application/octet-stream");
    REQUIRE(mime_for("png2") == "application/octet-stream");
}
