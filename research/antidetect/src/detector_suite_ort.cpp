#ifdef WMR_ANTIDETECT_ADVERSARIAL
#include "core/detector_suite_ort.hpp"

#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#if defined(__APPLE__)
#include <coreml_provider_factory.h>  // OrtSessionOptionsAppendExecutionProvider_CoreML
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <thread>

namespace wmr::antidetect {
namespace {

struct PreppedTensor {
    std::vector<float> buf;  // planar NCHW
    int h = 0, w = 0;
};

// ResizeCropNorm preprocessing (the manifest's resize_short/crop/mean/std):
// shortest-side resize -> center crop -> BGR->RGB -> [0,1] ->
// per-channel (x-mean)/std -> planar NCHW float buffer.
// The resize MUST be antialiased on downscale (INTER_AREA): the commfor ViT's
// logit depends strongly on the resize kernel, and a plain INTER_LINEAR/CUBIC
// downscale manufactures high-frequency energy it reads as generative (measured
// 2026-09-01, docs/research/antidetect-m0-calibration.md: 2-4 logit shifts
// toward fake on AI fixtures; INTER_AREA tracks the canonical PIL bicubic
// within ~1 logit).
PreppedTensor prep_resize_crop_norm(const cv::Mat& bgr_u8, const SurrogateSpec& s) {
    cv::Mat img = bgr_u8;
    if (s.resize_short > 0) {
        const double scale =
            static_cast<double>(s.resize_short) / std::min(img.rows, img.cols);
        const int interp = scale < 1.0 ? cv::INTER_AREA : cv::INTER_CUBIC;
        cv::resize(img, img, cv::Size(), scale, scale, interp);
    }
    if (s.crop > 0) {
        const int x0 = std::max(0, (img.cols - s.crop) / 2);
        const int y0 = std::max(0, (img.rows - s.crop) / 2);
        img = img(cv::Rect(x0, y0, std::min(s.crop, img.cols),
                           std::min(s.crop, img.rows)));
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> ch;
    cv::split(img, ch);
    const size_t n = static_cast<size_t>(img.total());
    PreppedTensor t;
    t.h = img.rows;
    t.w = img.cols;
    t.buf.resize(n * 3);
    for (int c = 0; c < 3; ++c) {
        cv::Mat normed = (ch[c] - s.mean[c]) / s.stdv[c];
        std::copy(normed.begin<float>(), normed.end<float>(), t.buf.begin() + n * c);
    }
    return t;
}

// BakedGraph contract: the graph does everything except the input-level pixel
// cap; the runtime feeds float RGB [0,1] NCHW at (capped) native size (the
// corvi export bakes ImageNet normalization + the spatial-mean head; dynamic
// H/W). The cap mirrors the calibration harness exactly — sqrt-area bicubic
// downscale, thresholds were measured WITH it (CORVI_MAX_PIXELS = 1.2 MPix;
// detectors.py) — so the C++ scores sit on the calibrated logits.
PreppedTensor prep_baked(const cv::Mat& bgr_u8, const SurrogateSpec& s) {
    cv::Mat img = bgr_u8;
    if (s.max_pixels > 0 &&
        static_cast<size_t>(img.total()) > s.max_pixels) {
        const double scale = std::sqrt(static_cast<double>(s.max_pixels) /
                                       static_cast<double>(img.total()));
        cv::resize(img, img,
                   cv::Size(static_cast<int>(std::lround(img.cols * scale)),
                            static_cast<int>(std::lround(img.rows * scale))),
                   cv::INTER_CUBIC);
    }
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> ch;
    cv::split(rgb, ch);
    const size_t n = static_cast<size_t>(rgb.total());
    PreppedTensor t;
    t.h = rgb.rows;
    t.w = rgb.cols;
    t.buf.resize(n * 3);
    for (int c = 0; c < 3; ++c)
        std::copy(ch[c].begin<float>(), ch[c].end<float>(), t.buf.begin() + n * c);
    return t;
}

} // namespace

bool DetectorSuite::initialize(const std::vector<const SurrogateSpec*>& specs,
                               const std::filesystem::path& models_dir) {
    models_.clear();
    specs_ = specs;
    if (!env_) env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "wmr-antidetect");

    for (const SurrogateSpec* spec : specs) {
        const auto path = models_dir / spec->filename;
        try {
            Ort::SessionOptions opts;
            // corvi (stride-1 ResNet-50 at up to 1.2 MPix) dominates the
            // adversarial loop's cost; half the cores (>=2) roughly halves it
            // vs the earlier conservative 2-thread pin without starving the
            // rest of the pipeline.
            const unsigned hw = std::thread::hardware_concurrency();
            opts.SetIntraOpNumThreads(static_cast<int>(std::min(8u, std::max(2u, hw / 2))));
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#if defined(__APPLE__)
            // The pinned ORT prebuilt ships the CoreML EP; with flags=0 it
            // places nodes on the ANE/GPU where eligible and falls back to
            // CPU inside the EP for the rest (dynamic-shape graphs like
            // corvi's may partition or fall back entirely - never worse than
            // CPU). WMR_AD_DISABLE_COREML=1 is the A/B + determinism escape:
            // scores can differ slightly between providers (different
            // kernels), so pinned comparisons must pin the provider too.
            if (!std::getenv("WMR_AD_DISABLE_COREML")) {
                if (OrtSessionOptionsAppendExecutionProvider_CoreML(
                        opts, /*coreml_flags=*/0) != nullptr) {
                    spdlog::info("Antidetect: CoreML EP unavailable for '{}'; "
                                 "using CPU", spec->key);
                }
            }
#endif
            Model m;
            m.spec = spec;
            m.session = std::make_unique<Ort::Session>(*env_, path.c_str(), opts);
            Ort::AllocatorWithDefaultOptions alloc;
            m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
            m.out_name = m.session->GetOutputNameAllocated(0, alloc).get();
            models_.push_back(std::move(m));
        } catch (const Ort::Exception& e) {
            spdlog::warn("Antidetect: surrogate '{}' failed to load ({}); skipping",
                         spec->key, e.what());
        }
    }
    if (models_.empty()) {
        spdlog::warn("Antidetect: no surrogate detector could be loaded");
        return false;
    }
    return true;
}

std::vector<float> DetectorSuite::score(const cv::Mat& bgr_u8) const {
    std::vector<float> out;
    out.reserve(models_.size());
    for (const Model& m : models_) {
        const PreppedTensor t = (m.spec->prep == SurrogatePrep::BakedGraph)
                                    ? prep_baked(bgr_u8, *m.spec)
                                    : prep_resize_crop_norm(bgr_u8, *m.spec);
        const std::array<int64_t, 4> dims{1, 3, t.h, t.w};
        try {
            const auto tensor = Ort::Value::CreateTensor<float>(
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault),
                const_cast<float*>(t.buf.data()), t.buf.size(), dims.data(), 4);
            const char* in_n[] = {m.in_name.c_str()};
            const char* out_n[] = {m.out_name.c_str()};
            const auto outs = m.session->Run(Ort::RunOptions{nullptr}, in_n,
                                             &tensor, 1, out_n, 1);
            const float logit = outs[0].GetTensorData<float>()[0];
            // Non-sigmoid models (corvi) report a raw logit whose decision
            // boundary is 0 = the flip threshold 0.5. Clamping the raw logit
            // to [0,1] would eat that boundary (a logit of -0.2 clamps to 0.0
            // and could never read as flipped), so squash monotonically with
            // 0 preserved as 0.5 instead. +-3 keeps margin resolution in the
            // report: measured corvi post-pass margins land in -0.6..-1.4, so
            // a +-1 clamp would saturate exactly where the interesting
            // numbers live.
            const float p =
                m.spec->sigmoid_output
                    ? 1.0f / (1.0f + std::exp(-logit))
                    : 0.5f + 0.5f * (std::clamp(logit, -3.0f, 3.0f) / 3.0f);
            out.push_back(std::clamp(p, 0.0f, 1.0f));
        } catch (const Ort::Exception& e) {
            spdlog::warn("Antidetect: surrogate '{}' scoring failed ({})",
                         m.spec->key, e.what());
            out.push_back(1.0f);  // missing score reads as "looks fake"
        }
    }
    return out;
}

} // namespace wmr::antidetect
#endif
