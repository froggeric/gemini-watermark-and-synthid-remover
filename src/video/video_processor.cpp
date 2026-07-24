#include "video/video_processor.hpp"
#include "video/video_reader.hpp"
#include "video/scene_detector.hpp"
#include "video/notebooklm_gates.hpp"
#include "video/geometry_detector.hpp"
#include "core/watermark_engine.hpp"
#include "core/inpaint.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <fmt/format.h>

#if defined(WMR_AI_MIGAN)
// MI-GAN ONNX inpainter for NotebookLM intricate backgrounds (replaces FSR and
// LaMa — sharper on cartoons, ~225 ms CPU, MIT, 27 MB). Guarded so a build
// without MI-GAN (WMR_BUILD_AI_MIGAN=OFF) compiles cleanly and falls back to NS.
#include "core/migan_inpainter.hpp"
#endif

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

namespace wmr {

// ---------------------------------------------------------------------------
// Resolve geometry from config + resolution
// ---------------------------------------------------------------------------
WatermarkPosition VideoProcessor::resolve_geometry(
    const VideoWatermarkConfig& config, int width, int height) const
{
    return get_video_watermark_geometry(config.variant, width, height, config.profile);
}

// ---------------------------------------------------------------------------
// Determine WatermarkSize from geometry
// ---------------------------------------------------------------------------
WatermarkSize VideoProcessor::geometry_to_size(const WatermarkPosition& geo) const {
    return (geo.logo_size > 48) ? WatermarkSize::Large : WatermarkSize::Small;
}

// ---------------------------------------------------------------------------
// Alpha map + default anchor for a resolved geometry. The single helper for the
// pick + position math that detect_in_shot and process() both used to duplicate.
// ---------------------------------------------------------------------------
VideoProcessor::VideoAlphaAnchor VideoProcessor::select_video_alpha(
    const WatermarkEngine& engine, VideoProfile profile,
    const WatermarkPosition& geo, int width, int height) const
{
    VideoAlphaAnchor out;
    // Route the small/large pick through effective_alpha_size so the >48/>68
    // gate lives in exactly one place. The compared Sizes are that function's
    // "large" return values (99x43 Veo / 96x96 Gemini); this is equivalent to
    // >48/>68 for every logo_size the table/auto/rect path can emit
    // (Gemini 44/48/96, Veo 68/99).
    const cv::Size a = effective_alpha_size(profile, geo.logo_size);
    if (profile == VideoProfile::VeoLegacy) {
        out.alpha = (a == cv::Size(99, 43)) ? &engine.get_veo_text_alpha_large()
                                            : &engine.get_veo_text_alpha_small();
    } else {
        if (a == cv::Size(96, 96)) {
            out.alpha = &engine.get_v2_diamond_alpha_large();
        } else {
            // Gemini 3.6 48px: remove with the STILL capture, not the VIDEO 48px
            // capture. Both are the same watermark (shape corr 0.997), but the
            // still alpha was captured on a near-uniform-black background so it
            // needs no background correction, while the video 48px capture sat on
            // a non-black background (~0.09) and was "corrected" with an
            // approximate 25th-percentile estimate that left it ~0.007
            // under-calibrated -> a faint bright residual. Measured on a real
            // Gemini 3.6 video the still alpha cuts the mark-mean residual from
            // 0.0074 to 0.0030 (and |residual| 0.0235 -> 0.0200). Fall back to
            // the video capture only if the still alpha failed to decode.
            const cv::Mat& still = engine.get_v2_diamond_alpha_48_still();
            out.alpha = still.empty() ? &engine.get_v2_diamond_alpha_small() : &still;
        }
    }
    if (out.alpha && !out.alpha->empty()) {
        out.position = {width - geo.margin_right - out.alpha->cols,
                        height - geo.margin_bottom - out.alpha->rows};
        out.region = cv::Rect(out.position.x, out.position.y,
                              out.alpha->cols, out.alpha->rows);
    } else {
        out.position = geo.get_position(width, height);
        out.region = cv::Rect(out.position.x, out.position.y,
                              geo.logo_size, geo.logo_size);
    }
    return out;
}

namespace {
// Min |NCC| for a candidate geometry to count as a detection (same as NotebookLM).
constexpr float kAutoGeometryMinConfidence = 0.45f;
// A raw (off-table) detection must clear this to override the resolution guess,
// so a busy-corner false positive cannot regress a video that already works.
// Snapped (on-table) detections are trusted at kAutoGeometryMinConfidence.
constexpr float kAutoOverrideRawScore = 0.60f;

// Edge-only TELEA cleanup of the reverse-blended diamond's border/halo. Caller guards
// on config.edge_cleanup and profile (Gemini diamond only).
void apply_edge_cleanup(cv::Mat& frame, const cv::Rect& region, const cv::Mat* alpha) {
    if (!alpha || alpha->empty() || region.width < 4 || region.height < 4) return;
    wmr::inpaint_diamond_edges(frame, region, *alpha);
}
}  // namespace

// ---------------------------------------------------------------------------
// Content-based geometry search: sample frames, multi-template match in the
// bottom-right corner, snap to the known table. FFmpeg-linked (needs VideoReader).
// ---------------------------------------------------------------------------
std::optional<SnappedGeometry> VideoProcessor::auto_detect_geometry(
    VideoReader& reader, WatermarkEngine& engine,
    const VideoWatermarkConfig& config, float& score)
{
    score = 0.0f;
    const int64_t total = reader.frame_count();
    if (total < 2) return std::nullopt;

    const int W = reader.width();
    const int H = reader.height();
    const int64_t coverage_end =
        std::min(total, static_cast<int64_t>(static_cast<double>(total) * kShotCoverage));
    const int sample_count = std::min(kShotSampleCount, static_cast<int>(coverage_end));
    if (sample_count < 1) return std::nullopt;

    // Sample grayscale frames over the first 90%.
    std::vector<cv::Mat> frames;
    frames.reserve(sample_count);
    for (int i = 0; i < sample_count; ++i) {
        const int64_t idx = (sample_count <= 1) ? 0 : static_cast<int64_t>(
            static_cast<double>(i) / static_cast<double>(sample_count - 1) *
            static_cast<double>(coverage_end - 1));
        if (!reader.seek(idx)) continue;
        cv::Mat frame;
        if (!reader.next_frame(frame) || frame.empty()) continue;
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        frames.push_back(gray);
    }
    if (frames.empty()) return std::nullopt;

    // Real alpha assets at native size, converted to CV_8U for matching.
    std::vector<cv::Mat> templates;
    auto add_template = [&](const cv::Mat& alpha) {
        if (alpha.empty()) return;
        cv::Mat t;
        alpha.convertTo(t, CV_8U, 255.0);
        templates.push_back(t);
    };
    if (config.profile == VideoProfile::VeoLegacy) {
        add_template(engine.get_veo_text_alpha_small());  // 68x30
        add_template(engine.get_veo_text_alpha_large());  // 99x43
    } else {
        add_template(engine.get_v2_diamond_alpha_small());  // 48x48
        add_template(engine.get_v2_diamond_alpha_large());  // 96x96
    }
    if (templates.empty()) return std::nullopt;

    // Bottom-right corner window, sized for the largest mark (1080p: margin 192 + 96).
    cv::Rect corner;
    if (config.profile == VideoProfile::VeoLegacy) {
        const int x0 = std::max(0, W - 200);
        const int y0 = std::max(0, H - 120);
        corner = cv::Rect(x0, y0, W - x0, H - y0);
    } else {
        const int x0 = std::max(0, W - 320);
        const int y0 = std::max(0, H - 320);
        corner = cv::Rect(x0, y0, W - x0, H - y0);
    }

    auto hit = detect_geometry_in_frames(frames, templates, corner, kAutoGeometryMinConfidence);
    if (!hit) return std::nullopt;
    score = hit->score;
    return snap_geometry_to_known(hit->rect, W, H, config.profile);
}

// ---------------------------------------------------------------------------
// Resolve the effective geometry per the precedence chain (see header).
// ---------------------------------------------------------------------------
VideoProcessor::ResolvedGeometry VideoProcessor::resolve_effective_geometry(
    const VideoWatermarkConfig& config, int W, int H,
    VideoReader& reader, WatermarkEngine& engine)
{
    ResolvedGeometry out;

    const auto use_table_guess = [&]() {
        out.geo = resolve_geometry(config, W, H);
        out.source = config.force ? "force"
                   : (config.variant != VideoVariant::Auto) ? "variant"
                                                            : "resolution";
        out.score = 0.0f;
    };

    // 1. Manual --rect override (any profile).
    if (config.rect) {
        out.geo = rect_to_watermark_position(*config.rect, W, H, config.profile);
        out.source = "rect";
        out.score = 0.0f;
        return out;
    }
    // 2. --force / explicit --variant opt out of the content search.
    if (config.force || config.variant != VideoVariant::Auto) {
        use_table_guess();
        return out;
    }
    // 3. Content-based auto-detect (the new default for Auto).
    if (!config.no_auto_geometry) {
        float score = 0.0f;
        if (auto hit = auto_detect_geometry(reader, engine, config, score)) {
            const auto verdict = decide_auto_geometry(hit->snapped, score, kAutoOverrideRawScore);
            if (verdict == AutoGeometryVerdict::UseSnapped) {
                out.geo = hit->geometry;
                out.source = "auto/snapped";
                out.score = score;
                return out;
            }
            if (verdict == AutoGeometryVerdict::UseRaw) {
                out.geo = hit->geometry;
                out.source = "auto/raw";
                out.score = score;
                return out;
            }
            spdlog::warn("Auto-geometry: raw detection score {:.2f} < {:.2f}; "
                         "keeping the resolution guess",
                         score, kAutoOverrideRawScore);
        } else {
            spdlog::info("Auto-geometry: no confident match; using the resolution guess");
        }
    }
    // 4. Resolution guess (today's behavior).
    use_table_guess();
    return out;
}

// ---------------------------------------------------------------------------
// Shot-level detection: sample frames and establish watermark baseline
// ---------------------------------------------------------------------------
VideoProcessor::ShotDetection VideoProcessor::detect_in_shot(
    VideoReader& reader,
    WatermarkEngine& engine,
    const VideoWatermarkConfig& config,
    WatermarkPosition geo,
    int64_t range_start,
    int64_t range_end,
    int max_samples)
{
    ShotDetection result;

    const int64_t total = reader.frame_count();
    if (total <= 0) {
        spdlog::warn("Video has no frames, shot detection skipped");
        return result;
    }

    // Geometry was resolved upstream by resolve_effective_geometry and passed in.
    auto wsize = geometry_to_size(geo);
    auto anchor = select_video_alpha(engine, config.profile, geo,
                                     reader.width(), reader.height());
    const cv::Mat* video_alpha = anchor.alpha;
    result.position = anchor.position;
    result.region   = anchor.region;
    result.size     = wsize;

    // Determine sample range: use range params if specified, otherwise default coverage
    const int64_t sample_start = (range_start >= 0) ? range_start : 0;
    const int64_t sample_end = (range_end >= 0) ? range_end
        : static_cast<int64_t>(static_cast<double>(total) * kShotCoverage);
    const int64_t coverage_end = std::min(sample_end, total);
    const int samples_available = static_cast<int>(coverage_end - sample_start);

    // Adaptive sample count for scene ranges
    const int sample_count = (range_start >= 0)
        ? std::min(max_samples, std::max(2, samples_available / 30))
        : std::min(max_samples, samples_available);

    if (sample_count <= 0) {
        spdlog::warn("Video too short for shot sampling");
        return result;
    }

    // For a single sample, just check frame 0
    if (sample_count == 1) {
        cv::Mat frame;
        if (reader.seek(0) && reader.next_frame(frame) && !frame.empty()) {
            auto det = engine.detect_watermark(frame, wsize, geo, video_alpha);
            if (det.detected) {
                result.found = true;
                result.position = cv::Point(det.region.x, det.region.y);
                result.size = det.size;
                result.confidence = det.confidence;
                result.region = det.region;
                spdlog::info("Shot detection (single frame): detected at ({},{}) conf={:.2f}",
                             result.position.x, result.position.y, result.confidence);
            }
        }
        return result;
    }

    struct Sample {
        cv::Point position;
        WatermarkSize size;
        float confidence;
        cv::Rect region;
    };

    std::vector<Sample> detections;
    detections.reserve(sample_count);

    spdlog::info("Shot detection: sampling {} frames over first {} of {} total "
                 "(geo: margin={},{} size={})",
                 sample_count, coverage_end, total,
                 geo.margin_right, geo.margin_bottom, geo.logo_size);

    if (range_start >= 0) {
        // --- Scene-range path: sequential decode with PTS verification ---
        // Random seek after backward-flag lands on a keyframe that may be
        // before the scene boundary (e.g. in a concatenated video).  Decode
        // forward and skip frames until PTS confirms we're past range_start.

        auto* stream = reader.format_context()->streams[reader.video_stream_index()];

        auto frame_to_pts = [&](int64_t idx) -> int64_t {
            if (stream->avg_frame_rate.num <= 0 || stream->avg_frame_rate.den <= 0)
                return INT64_MIN;
            return idx *
                   static_cast<int64_t>(stream->time_base.den) *
                   static_cast<int64_t>(stream->avg_frame_rate.den) /
                   (static_cast<int64_t>(stream->time_base.num) *
                    static_cast<int64_t>(stream->avg_frame_rate.num));
        };

        // Pre-compute sample frame indices and their target PTS
        struct Target { int64_t frame_idx; int64_t pts; };
        std::vector<Target> targets;
        targets.reserve(sample_count);
        for (int i = 0; i < sample_count; ++i) {
            int64_t idx = sample_start + static_cast<int64_t>(
                static_cast<double>(i) / static_cast<double>(sample_count - 1) *
                static_cast<double>(coverage_end - sample_start - 1));
            targets.push_back({idx, frame_to_pts(idx)});
        }

        // Seek to first sample point (backward to keyframe)
        reader.seek(targets.front().frame_idx);

        cv::Mat frame;
        size_t next_target = 0;
        // Safety: limit total decoded frames to avoid infinite loop
        int max_decode = static_cast<int>(coverage_end - sample_start + 60);

        while (reader.next_frame(frame) && next_target < targets.size()
               && max_decode-- > 0) {
            if (frame.empty()) continue;

            int64_t cur_pts = reader.last_pts();
            int64_t tgt_pts = targets[next_target].pts;

            // Skip frames until PTS reaches the target
            if (tgt_pts != INT64_MIN && cur_pts != INT64_MIN
                && cur_pts < tgt_pts) {
                continue;
            }

            // Reached (or passed) the target — this is our sample
            auto det = engine.detect_watermark(frame, wsize, geo, video_alpha);
            if (det.detected) {
                detections.push_back({cv::Point(det.region.x, det.region.y),
                                      det.size, det.confidence, det.region});
                spdlog::debug("Scene sample {}: detected at ({},{}) conf={:.2f}",
                              next_target, det.region.x, det.region.y,
                              det.confidence);
            } else {
                spdlog::debug("Scene sample {}: not detected ({:.2f})",
                              next_target, det.confidence);
            }
            ++next_target;
        }

        spdlog::info("Scene-range detection: {}/{} samples collected via sequential decode",
                     detections.size(), sample_count);
    } else {
        // --- Full-video path: random seek per sample (original behavior) ---
        for (int i = 0; i < sample_count; ++i) {
            int64_t frame_idx = sample_start + static_cast<int64_t>(
                static_cast<double>(i) / static_cast<double>(sample_count - 1) *
                static_cast<double>(coverage_end - sample_start - 1));

            if (!reader.seek(frame_idx)) {
                spdlog::debug("Shot sample {}: seek to {} failed", i, frame_idx);
                continue;
            }

            cv::Mat frame;
            if (!reader.next_frame(frame) || frame.empty()) {
                spdlog::debug("Shot sample {}: read at {} failed", i, frame_idx);
                continue;
            }

            auto det = engine.detect_watermark(frame, wsize, geo, video_alpha);
            if (det.detected) {
                detections.push_back({cv::Point(det.region.x, det.region.y),
                                      det.size, det.confidence, det.region});
                spdlog::debug("Shot sample {}: detected at ({},{}) conf={:.2f}",
                              i, det.region.x, det.region.y, det.confidence);
            } else {
                spdlog::debug("Shot sample {}: not detected ({:.2f})",
                              i, det.confidence);
            }
        }
    }

    // Need >50% detection rate to trust the result
    const int threshold = (sample_count + 1) / 2;
    if (static_cast<int>(detections.size()) < threshold) {
        spdlog::info("Shot detection: {}/{} samples detected (< {} majority threshold)",
                     detections.size(), sample_count, threshold);
        result.found = false;
        result.confidence = 0.0f;
        return result;
    }

    // Compute median of detected positions, sizes, and confidences
    std::vector<int> xs, ys;
    std::vector<float> confs;
    int small_count = 0;
    cv::Rect median_region;

    for (const auto& d : detections) {
        xs.push_back(d.position.x);
        ys.push_back(d.position.y);
        confs.push_back(d.confidence);
        if (d.size == WatermarkSize::Small) ++small_count;
    }

    const auto mid = xs.size() / 2;

    std::sort(xs.begin(), xs.end());
    std::sort(ys.begin(), ys.end());
    std::sort(confs.begin(), confs.end());

    result.position = cv::Point(xs[mid], ys[mid]);
    result.confidence = confs[mid];

    // Use the region from the detection closest to median position
    int best_dist = std::numeric_limits<int>::max();
    for (const auto& d : detections) {
        int dist = std::abs(d.position.x - result.position.x) +
                   std::abs(d.position.y - result.position.y);
        if (dist < best_dist) {
            best_dist = dist;
            median_region = d.region;
        }
    }
    result.region = median_region;

    // Majority size
    result.size = (small_count > static_cast<int>(detections.size()) / 2)
                      ? WatermarkSize::Small : WatermarkSize::Large;
    result.found = true;

    spdlog::info("Shot detection: anchor at ({},{}) size={} conf={:.2f} "
                 "({}/{} detections)",
                 result.position.x, result.position.y,
                 result.size == WatermarkSize::Small ? "small" : "large",
                 result.confidence, detections.size(), sample_count);

    return result;
}

// ---------------------------------------------------------------------------
// Main processing loop
// ---------------------------------------------------------------------------
VideoResult VideoProcessor::process(const std::string& input_path,
                                    const std::string& output_path,
                                    const VideoWatermarkConfig& config,
                                    const EncodeOptions& encode_opts)
{
    VideoResult result;
    const auto t_start = std::chrono::steady_clock::now();

    // NotebookLM uses a separate path (temporal detection + NS inpaint)
    if (config.profile == VideoProfile::NotebookLM) {
        return process_notebooklm(input_path, output_path, config, encode_opts);
    }

    // Open input
    VideoReader reader;
    if (!reader.open(input_path)) {
        result.success = false;
        result.message = fmt::format("Failed to open input: {}", input_path);
        spdlog::error("{}", result.message);
        return result;
    }

    spdlog::info("Input: {} ({}x{}, {:.2f} fps, {} frames)",
                 input_path, reader.width(), reader.height(),
                 reader.fps(), reader.frame_count());

    WatermarkEngine engine;

    // Resolve video-specific geometry (--rect > auto-detect > --variant > resolution)
    auto resolved = resolve_effective_geometry(
        config, reader.width(), reader.height(), reader, engine);
    auto geo = resolved.geo;
    auto wsize = geometry_to_size(geo);
    spdlog::info("Geometry: margin={},{} logo_size={} (source={}, score={:.2f})",
                 geo.margin_right, geo.margin_bottom, geo.logo_size,
                 resolved.source, resolved.score);

    // Alpha + default anchor (top-left + bbox), chosen once via the single
    // effective_alpha_size gate; reused by the --force branches below.
    auto anchor = select_video_alpha(engine, config.profile, geo,
                                     reader.width(), reader.height());
    const cv::Mat* video_alpha = anchor.alpha;

    // Shot-level detection
    ShotDetection shot;

    if (config.scenes) {
        // ---- Scene splitting: detect boundaries, split into separate files ----

        // Detect scene boundaries using SceneDetector's own reader
        spdlog::info("Scanning for scene boundaries...");
        SceneDetector detector({config.scene_threshold});
        auto scenes = detector.detect_boundaries(input_path);

        if (scenes.empty()) {
            spdlog::warn("Scene detection returned no results, treating as single scene");
            scenes.push_back({0, reader.frame_count()});
        }

        spdlog::info("Detected {} scene(s)", scenes.size());

        // Single full-video watermark detection
        if (config.force) {
            shot.region     = anchor.region;
            shot.position   = anchor.position;
            shot.found      = false;
            shot.size       = wsize;
            shot.confidence = 1.0f;
            spdlog::info("Force mode: using position ({},{}) size={}",
                         shot.position.x, shot.position.y, geo.logo_size);
        } else {
            shot = detect_in_shot(reader, engine, config, geo);
        }

        reader.seek(0);

        // Output naming: <stem>_<NNN>.mp4
        namespace fs = std::filesystem;
        const std::string stem = fs::path(input_path).stem().string();
        const fs::path out_dir(output_path);
        const int num_scenes = static_cast<int>(scenes.size());
        const int pad_width = std::max(3, static_cast<int>(
            std::ceil(std::log10(static_cast<double>(num_scenes) + 1))));

        // Process each scene sequentially (reader stays in position between scenes)
        for (int si = 0; si < num_scenes; ++si) {
            const auto& scene = scenes[si];
            const int64_t scene_frames = scene.end_frame - scene.start_frame;
            const double start_sec = static_cast<double>(scene.start_frame) / reader.fps();
            const double end_sec = static_cast<double>(scene.end_frame) / reader.fps();

            const std::string out_name = fmt::format("{}_{:0{}}.mp4", stem, si + 1, pad_width);
            const fs::path out_path = out_dir / out_name;

            spdlog::info("Scene {}/{}: {} (frames [{},{}), {:.2f}s)",
                         si + 1, num_scenes, out_name,
                         scene.start_frame, scene.end_frame,
                         static_cast<double>(scene_frames) / reader.fps());

            // Create writer for this scene
            VideoWriter writer;
            if (!writer.open(out_path.string(), reader.width(), reader.height(),
                             reader.fps(), encode_opts, input_path)) {
                spdlog::error("Failed to open output for scene {}: {}",
                              si + 1, out_path.string());
                result.success = false;
                result.message = fmt::format("Failed to open output for scene {}", si + 1);
                reader.close();
                return result;
            }

            // Process frames for this scene (sequential decode, no seeking)
            cv::Mat frame;
            int64_t scene_processed = 0;
            for (int64_t fi = scene.start_frame; fi < scene.end_frame; ++fi) {
                if (!reader.next_frame(frame)) {
                    spdlog::warn("Scene {}: unexpected EOF at frame {}", si + 1, fi);
                    break;
                }

                if (frame.empty()) {
                    spdlog::warn("Scene {}: frame {} empty, skipping", si + 1, fi);
                    ++result.frames_skipped;
                    continue;
                }

                if (frame.cols != reader.width() || frame.rows != reader.height()) {
                    spdlog::warn("Scene {}: frame {} unexpected dimensions {}x{}",
                                 si + 1, fi, frame.cols, frame.rows);
                    ++result.frames_skipped;
                    continue;
                }

                // Apply watermark removal (same logic as non-scene branch)
                if (config.force) {
                    DetectionResult det;
                    det.detected = true;
                    det.confidence = 1.0f;
                    det.region = shot.region;
                    det.size = shot.size;
                    engine.remove_watermark_alpha_only(frame, det, video_alpha);
                    if (config.edge_cleanup && config.profile != VideoProfile::VeoLegacy)
                        apply_edge_cleanup(frame, det.region, video_alpha);
                } else {
                    DetectionResult det;
                    det.detected = true;
                    det.confidence = shot.confidence;
                    det.region = shot.region;
                    det.size = shot.size;

                    auto detection = engine.detect_watermark(frame, wsize, geo, video_alpha);
                    if (detection.detected &&
                        detection.confidence >= kOcclusionGateNcc) {
                        cv::Point detected_pos(detection.region.x, detection.region.y);
                        int dx = std::abs(detected_pos.x - shot.position.x);
                        int dy = std::abs(detected_pos.y - shot.position.y);
                        if (dx <= kPositionTolerancePx && dy <= kPositionTolerancePx) {
                            det.region = detection.region;
                            det.size = detection.size;
                        }
                    }

                    engine.remove_watermark_alpha_only(frame, det, video_alpha);
                    if (config.edge_cleanup && config.profile != VideoProfile::VeoLegacy)
                        apply_edge_cleanup(frame, det.region, video_alpha);
                }

                writer.write_frame(frame);
                ++scene_processed;
                ++result.frames_processed;
            }

            // Copy audio for this scene's time range
            if (!writer.copy_audio_range(start_sec, end_sec)) {
                spdlog::warn("Scene {}: audio range copy failed", si + 1);
            }

            writer.close();

            double scene_dur = static_cast<double>(scene_processed) / reader.fps();
            spdlog::info("Scene {}/{}: {} done ({} frames, {:.2f}s)",
                         si + 1, num_scenes, out_name, scene_processed, scene_dur);
        }

        reader.close();

    } else {
        // ---- Standard single-shot processing (original behavior) ----

        if (config.force) {
            shot.region     = anchor.region;
            shot.position   = anchor.position;
            shot.found      = false;
            shot.size       = wsize;
            shot.confidence = 1.0f;
            spdlog::info("Force mode: using position ({},{}) size={}",
                         shot.position.x, shot.position.y, geo.logo_size);
        } else {
            shot = detect_in_shot(reader, engine, config, geo);
            reader.seek(0);
        }

        // Open output writer (audio streams set up before MP4 header)
        VideoWriter writer;
        if (!writer.open(output_path, reader.width(), reader.height(),
                         reader.fps(), encode_opts, input_path)) {
            result.success = false;
            result.message = fmt::format("Failed to open output: {}", output_path);
            spdlog::error("{}", result.message);
            reader.close();
            return result;
        }

        // Process frames
        cv::Mat frame;
        int64_t frame_idx = 0;
        int64_t last_progress_frame = 0;
        auto t_last_progress = t_start;

        while (reader.next_frame(frame)) {
            if (frame.empty()) {
                spdlog::warn("Frame {}: empty, skipping", frame_idx);
                writer.write_frame(frame);
                ++result.frames_skipped;
                ++frame_idx;
                continue;
            }

            // Guard against corrupt frames from seek artifacts
            if (frame.cols != reader.width() || frame.rows != reader.height()) {
                spdlog::warn("Frame {}: unexpected dimensions {}x{} (expected {}x{}), skipping",
                             frame_idx, frame.cols, frame.rows,
                             reader.width(), reader.height());
                ++result.frames_skipped;
                ++frame_idx;
                continue;
            }

            if (config.force) {
                // Pure reverse alpha blend — no inpaint to avoid blur
                DetectionResult det;
                det.detected = true;
                det.confidence = 1.0f;
                det.region = shot.region;
                det.size = shot.size;
                engine.remove_watermark_alpha_only(frame, det, video_alpha);
                if (config.edge_cleanup && config.profile != VideoProfile::VeoLegacy)
                    apply_edge_cleanup(frame, det.region, video_alpha);
                writer.write_frame(frame);
                ++result.frames_processed;
            } else {
                // Shot detection confirmed the watermark — process every frame.
                // Use per-frame detection for position refinement when available,
                // fall back to shot anchor when detection fails (occluded frames).
                DetectionResult det;
                det.detected = true;
                det.confidence = shot.confidence;
                det.region = shot.region;
                det.size = shot.size;

                auto detection = engine.detect_watermark(frame, wsize, geo, video_alpha);
                if (detection.detected &&
                    detection.confidence >= kOcclusionGateNcc) {
                    // Per-frame detection succeeded — refine position
                    cv::Point detected_pos(detection.region.x, detection.region.y);
                    int dx = std::abs(detected_pos.x - shot.position.x);
                    int dy = std::abs(detected_pos.y - shot.position.y);

                    if (dx <= kPositionTolerancePx && dy <= kPositionTolerancePx) {
                        det.region = detection.region;
                        det.size = detection.size;
                    }
                }

                engine.remove_watermark_alpha_only(frame, det, video_alpha);
                if (config.edge_cleanup && config.profile != VideoProfile::VeoLegacy)
                    apply_edge_cleanup(frame, det.region, video_alpha);
                writer.write_frame(frame);
                ++result.frames_processed;
            }

            // Progress output
            ++frame_idx;
            auto t_now = std::chrono::steady_clock::now();
            double since_last = std::chrono::duration<double>(t_now - t_last_progress).count();

            if (frame_idx - last_progress_frame >= kProgressIntervalFrames ||
                since_last >= 2.0) {
                double elapsed = std::chrono::duration<double>(t_now - t_start).count();
                double proc_fps = static_cast<double>(frame_idx) / std::max(elapsed, 1e-9);
                int64_t remaining = reader.frame_count() - frame_idx;
                double eta = (remaining > 0) ? (static_cast<double>(remaining) / proc_fps) : 0.0;

                spdlog::info("Frame {}/{} ({:.1f} fps, ETA {:.0f}s)",
                             frame_idx, reader.frame_count(), proc_fps, eta);

                last_progress_frame = frame_idx;
                t_last_progress = t_now;
            }
        }

        // Copy audio
        if (!writer.copy_audio()) {
            spdlog::warn("Audio copy failed or no audio stream present");
        }

        writer.close();
        reader.close();
    }

    // Build result
    const auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();
    result.detection_confidence = shot.confidence;
    result.success = true;
    result.message = fmt::format("Processed {} frames ({} skipped) in {:.1f}s",
                                 result.frames_processed, result.frames_skipped,
                                 result.elapsed_seconds);

    spdlog::info("Done: {} processed, {} skipped, detection conf={:.2f}, "
                 "elapsed={:.1f}s",
                 result.frames_processed, result.frames_skipped,
                 result.detection_confidence, result.elapsed_seconds);

    return result;
}


// ---------------------------------------------------------------------------
// Inpaint the watermark ROI on a single frame.
//   "migan" -> MI-GAN ONNX (full-frame, uint8 dynamic-res) when WMR_AI_MIGAN,
//              else fall through to NS.
//   "ns" / anything else -> cv::inpaint Navier-Stokes on a lean padded crop.
// MI-GAN takes the whole frame (mobile-optimized, ~225 ms at 720p) and composites
// the fill itself; NS is a local PDE method that only needs a small radius+4 crop.
static void inpaint_mark_roi(cv::Mat& frame, const cv::Rect& mark_rect,
                             int radius, const std::string& method) {
    const cv::Rect bounds(0, 0, frame.cols, frame.rows);
    cv::Rect mask = mark_rect & bounds;
    if (mask.empty()) return;

    if (method == "migan") {
#if defined(WMR_AI_MIGAN)
        // Leaked process singleton: the Ort::Session races ONNX Runtime global
        // state during static teardown (same rationale as the NCNN denoiser).
        static MiganInpainter* migan = []() {
            auto* p = new MiganInpainter();   // intentionally never deleted
            p->initialize();
            return p;
        }();
        if (migan->is_ready()) {
            migan->inpaint_hole(frame, mask);
            return;
        }
        // Model missing / init failed -> fall through to the NS path below.
#endif
        // MI-GAN not compiled in -> fall through to NS below.
    }

    // NS: a local PDE method — a lean radius+4 padded crop is all it needs.
    const int pad = radius + 4;
    cv::Rect roi = (mask + cv::Size(2 * pad, 2 * pad)) - cv::Point(pad, pad);
    roi &= bounds;
    if (roi.width <= 0 || roi.height <= 0) return;
    cv::Rect local = mask - cv::Point(roi.x, roi.y);
    local &= cv::Rect(0, 0, roi.width, roi.height);
    if (local.empty()) return;

    cv::Mat crop = frame(roi).clone();
    cv::Mat lmask = cv::Mat::zeros(crop.size(), CV_8U);
    lmask(local).setTo(255);
    cv::dilate(lmask, lmask, cv::Mat::ones(5, 5, CV_8U), cv::Point(-1, -1), 1);

    cv::Mat out;
    cv::inpaint(crop, lmask, out, radius, cv::INPAINT_NS);  // the always-available fallback
    out.copyTo(frame(roi));
}

// ---------------------------------------------------------------------------
// NotebookLM processing: per-scene adaptive dispatch.
//   detect bbox → split into scenes → per-scene presence + complexity gates →
//   skip absent-mark scenes; NS-inpaint present scenes (ROI crop). Single-file
//   output (NOT split like the Gemini --scenes branch). Audio copied once.
// ---------------------------------------------------------------------------
VideoResult VideoProcessor::process_notebooklm(const std::string& input_path,
                                               const std::string& output_path,
                                               const VideoWatermarkConfig& config,
                                               const EncodeOptions& encode_opts)
{
    VideoResult result;
    const auto t_start = std::chrono::steady_clock::now();

    // 1. Detect the watermark bbox (whole-video template match + snap to known).
    NotebookLMDetector detector;
    auto det = detector.detect(input_path, config.rect);
    if (!det.found) {
        result.success = false;
        result.message = "NotebookLM watermark not detected (use --rect x,y,w,h to specify manually)";
        spdlog::error("{}", result.message);
        return result;
    }
    spdlog::info("NotebookLM watermark bbox: ({},{},{},{})",
                 det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height);

    // 2. Open the main reader (used for the decode loop).
    VideoReader reader;
    if (!reader.open(input_path)) {
        result.success = false;
        result.message = fmt::format("Failed to open input: {}", input_path);
        spdlog::error("{}", result.message);
        return result;
    }
    spdlog::info("Input: {} ({}x{}, {:.2f} fps, {} frames)",
                 input_path, reader.width(), reader.height(),
                 reader.fps(), reader.frame_count());

    // 3. Scene boundaries (SceneDetector opens its own reader; undisturbs main).
    SceneDetector scene_detector({config.scene_threshold});
    auto scenes = scene_detector.detect_boundaries(input_path);
    if (scenes.empty()) {
        scenes.push_back({0, reader.frame_count()});
    }
    spdlog::info("NotebookLM: {} scene(s)", scenes.size());

    // 4. Resolve the platform default + whether the complexity analysis is needed.
    //    --notebooklm-method {auto|ns|migan}: "auto" defers to the platform
    //    default; "ns"/"migan" force one. The platform default is MI-GAN-everywhere
    //    only on Apple Silicon (the Neural Engine makes MI-GAN fast, ~28 ms/frame);
    //    elsewhere MI-GAN is slow (ORT-CPU ~225 ms, or CoreML without an ANE), so
    //    the complexity gate (NS for uniform, MI-GAN for intricate) is kept. NS is
    //    always the fallback when MI-GAN is unavailable (not built, or model failed
    //    to load — the latter is caught by inpaint_mark_roi's is_ready() check).
    const bool has_migan =
#if defined(WMR_AI_MIGAN)
        true;
#else
        false;
#endif
#if defined(WMR_AI_MIGAN_COREML) && defined(__arm64__)
    // Native Apple Silicon: MI-GAN runs on the ANE. x86_64 — including a Rosetta-
    // translated arm64 binary, which compiles without __arm64__ — keeps the gate.
    static constexpr const char* kPlatformDefault = "migan";
#else
    static constexpr const char* kPlatformDefault = "gated";
#endif
    if (config.notebooklm_method == "migan" && !has_migan) {
        spdlog::warn("--notebooklm-method migan ignored (MI-GAN not built); using NS");
    }

    // The complexity analysis pass (a full sequential decode + Sobel per scene)
    // is only needed when the gate will actually be consulted: auto on a gated
    // platform. Forced ns/migan and arm64-auto skip it entirely.
    const bool need_complexity =
        (config.notebooklm_method == "auto" && std::string(kPlatformDefault) == "gated");

    // 4a. Per-scene complexity (separate temp reader so the main decode reader
    //     stays pristine). Every scene is inpainted — template-matching presence
    //     is unreliable for this semi-transparent, color-adaptive mark (a
    //     faint-but-present mark scores the same ~0.34-0.43 as a genuinely-absent
    //     scene), so skipping risked leaving a visible watermark. Inpainting an
    //     already-clean patch is imperceptible. The complexity score only routes
    //     NS vs MI-GAN (on gated platforms).
    std::vector<float> scene_complexity(scenes.size(), 0.0f);
    if (need_complexity) {
        VideoReader areader;
        const bool analysis_ok = areader.open(input_path);
        if (!analysis_ok) {
            spdlog::warn("NotebookLM: analysis reader failed; treating all scenes as uniform (NS)");
        }
        for (size_t i = 0; i < scenes.size(); ++i) {
            scene_complexity[i] = analysis_ok
                ? detector.background_complexity(areader, scenes[i].start_frame,
                                                 scenes[i].end_frame, det.bbox)
                : 0.0f;
            spdlog::info("NotebookLM scene {}/{}: frames[{},{}] complexity={:.1f}",
                         i + 1, scenes.size(), scenes[i].start_frame, scenes[i].end_frame,
                         scene_complexity[i]);
        }
    } else {
        spdlog::info("NotebookLM: complexity analysis skipped (method={}, platform default={})",
                     config.notebooklm_method, kPlatformDefault);
    }

    // 4b. Resolve the per-scene inpaint method.
    std::vector<std::string> scene_method(scenes.size());
    bool warned_no_migan = false;
    for (size_t i = 0; i < scenes.size(); ++i) {
        scene_method[i] = resolve_inpaint_method(
            scene_complexity[i], config.notebooklm_complexity_threshold,
            has_migan, config.notebooklm_method, kPlatformDefault);
        if (has_migan && scene_method[i] == "ns" &&
            scene_complexity[i] >= static_cast<float>(config.notebooklm_complexity_threshold) &&
            !warned_no_migan) {
            // MI-GAN is built but an intricate scene still resolved to NS ->
            // the model failed to load. Warn once (every other scene is fine).
            spdlog::warn("NotebookLM: MI-GAN model unavailable; using NS");
            warned_no_migan = true;
        }
        if (need_complexity) {
            spdlog::info("NotebookLM scene {}/{}: complexity={:.1f} -> inpaint({})",
                         i + 1, scenes.size(), scene_complexity[i], scene_method[i]);
        } else {
            spdlog::info("NotebookLM scene {}/{}: complexity=n/a -> inpaint({})",
                         i + 1, scenes.size(), scene_method[i]);
        }
    }

    // 5. Open output writer.
    VideoWriter writer;
    if (!writer.open(output_path, reader.width(), reader.height(),
                     reader.fps(), encode_opts, input_path)) {
        result.success = false;
        result.message = fmt::format("Failed to open output: {}", output_path);
        spdlog::error("{}", result.message);
        reader.close();
        return result;
    }

    // 6. Inpaint geometry (clamped bbox + radius).
    cv::Rect mask_rect = det.bbox;
    mask_rect.x = std::max(0, mask_rect.x);
    mask_rect.y = std::max(0, mask_rect.y);
    mask_rect.width = std::min(mask_rect.width, reader.width() - mask_rect.x);
    mask_rect.height = std::min(mask_rect.height, reader.height() - mask_rect.y);
    const int inpaint_radius = std::clamp(mask_rect.height / 4, 3, 8);

    // 7. Main decode: sequential, single writer, per-scene dispatch.
    reader.seek(0);
    cv::Mat frame;
    int64_t frame_idx = 0;
    int64_t last_progress_frame = 0;
    int64_t frames_inpainted = 0;
    size_t cur_scene = 0;
    auto t_last_progress = t_start;

    while (reader.next_frame(frame)) {
        // Advance to the scene containing frame_idx (scenes are contiguous).
        while (cur_scene + 1 < scenes.size() && frame_idx >= scenes[cur_scene].end_frame) {
            ++cur_scene;
        }
        if (frame.empty()) {
            spdlog::warn("Frame {}: empty, skipping", frame_idx);
            writer.write_frame(frame);
            ++result.frames_skipped;
            ++frame_idx;
            continue;
        }
        if (frame.cols != reader.width() || frame.rows != reader.height()) {
            spdlog::warn("Frame {}: unexpected dimensions {}x{}, skipping",
                         frame_idx, frame.cols, frame.rows);
            ++result.frames_skipped;
            ++frame_idx;
            continue;
        }

        // Every scene is inpainted (presence detection is unreliable for this
        // semi-transparent mark — see the analysis-pass note above).
        inpaint_mark_roi(frame, mask_rect, inpaint_radius, scene_method[cur_scene]);
        ++frames_inpainted;
        writer.write_frame(frame);
        ++result.frames_processed;

        ++frame_idx;
        auto t_now = std::chrono::steady_clock::now();
        double since_last = std::chrono::duration<double>(t_now - t_last_progress).count();
        if (frame_idx - last_progress_frame >= kProgressIntervalFrames || since_last >= 2.0) {
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            double proc_fps = static_cast<double>(frame_idx) / std::max(elapsed, 1e-9);
            int64_t remaining = reader.frame_count() - frame_idx;
            double eta = (remaining > 0) ? (static_cast<double>(remaining) / proc_fps) : 0.0;
            spdlog::info("Frame {}/{} ({:.1f} fps, ETA {:.0f}s)",
                         frame_idx, reader.frame_count(), proc_fps, eta);
            last_progress_frame = frame_idx;
            t_last_progress = t_now;
        }
    }

    // 8. Audio + teardown.
    if (!writer.copy_audio()) {
        spdlog::warn("Audio copy failed or no audio stream present");
    }
    writer.close();
    reader.close();

    const auto t_end = std::chrono::steady_clock::now();
    result.elapsed_seconds = std::chrono::duration<double>(t_end - t_start).count();
    result.detection_confidence = det.confidence;
    result.success = true;
    result.message = fmt::format("NotebookLM: {} frames ({} inpainted, {} skipped) "
                                 "in {:.1f}s",
                                 result.frames_processed, frames_inpainted,
                                 result.frames_skipped, result.elapsed_seconds);
    spdlog::info("Done: {} processed ({} inpainted, {} skipped), elapsed={:.1f}s",
                 result.frames_processed, frames_inpainted,
                 result.frames_skipped, result.elapsed_seconds);
    return result;
}

} // namespace wmr
