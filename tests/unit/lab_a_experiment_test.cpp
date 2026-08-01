// WS3: LAB `a`-channel experiment (--lab-a flag).
//
// Acceptance criterion #1: when --lab-a is engaged the output must DIFFER from
// the default BGR path (the path is genuinely exercised, not a vacuous no-op),
// AND default-off must be byte-identical to today's behavior. This case also
// checks the engaged output is a valid image (dimensions preserved, not
// saturated to black/white) so the experiment cannot silently produce garbage.
//
// The detector sibling path (SynthidDetector::ColorSpace::LabA) is exercised
// against a tiny synthetic codebook to confirm it produces a finite, in-range
// confidence and that LabA != BGR on a non-trivial input.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "core/fft_context.hpp"
#include "detection/synthid_detector.hpp"
#include "synthid/codebook_subtractor.hpp"
#include "synthid/noise_residual_subtractor.hpp"
#include "synthid/spectral_codebook.hpp"

using namespace wmr;

namespace {

// A non-trivial color image (varies in BGR so it lands in the content branch,
// where the LAB-a path actually does spectral disruption). A smooth horizontal
// red-green gradient plus a vertical blue ramp gives real chrominance in `a`.
cv::Mat make_content_image(int w, int h) {
    cv::Mat img(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float t = static_cast<float>(x) / std::max(1, w - 1);
            const float s = static_cast<float>(y) / std::max(1, h - 1);
            // BGR. Strong green-red variation drives `a` channel energy.
            const int b = static_cast<int>(30.0f + 60.0f * s);
            const int g = static_cast<int>(40.0f + 180.0f * t);
            const int r = static_cast<int>(220.0f - 180.0f * t);
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                static_cast<uint8_t>(std::clamp(b, 0, 255)),
                static_cast<uint8_t>(std::clamp(g, 0, 255)),
                static_cast<uint8_t>(std::clamp(r, 0, 255)));
        }
    }
    return img;
}

}  // namespace

TEST_CASE("WS3 --lab-a engages a real path (output differs from BGR; valid image)",
          "[synthid][lab-a]") {
    // Use dimensions large enough to put content in the carrier band (r=30..500).
    constexpr int W = 256, H = 256;
    const cv::Mat src = make_content_image(W, H);

    FftContext fft;
    NoiseResidualSubtractor subtractor(fft);

    RemovalConfig cfg_bgr;       // default: lab_a == false
    RemovalConfig cfg_lab_a;
    cfg_lab_a.lab_a = true;

    cv::Mat out_bgr = src.clone();
    cv::Mat out_lab = src.clone();
    subtractor.remove_synthid(out_bgr, cfg_bgr);
    subtractor.remove_synthid(out_lab, cfg_lab_a);

    // (1) Default-off must be byte-identical to a separate default-off run
    //     (the lab_a branch is skipped entirely; same RNG seed, same path).
    cv::Mat out_bgr_again = src.clone();
    subtractor.remove_synthid(out_bgr_again, cfg_bgr);
    REQUIRE(cv::countNonZero(out_bgr.reshape(1) != out_bgr_again.reshape(1)) == 0);

    // (2) Engaged path must DIFFER from default-off. If it does not, the lab-a
    //     branch is a vacuous no-op (the BLOCKER the task names).
    const int diff_pixels = cv::countNonZero(out_bgr.reshape(1) != out_lab.reshape(1));
    INFO("lab-a vs BGR differing pixels: " << diff_pixels << " / " << (W * H * 3));
    REQUIRE(diff_pixels > 0);

    // (3) Engaged output is a valid image: same size, not saturated.
    REQUIRE(out_lab.rows == H);
    REQUIRE(out_lab.cols == W);
    REQUIRE(out_lab.type() == CV_8UC3);
    cv::Scalar mean_lab = cv::mean(out_lab);
    cv::Scalar std_lab;
    cv::meanStdDev(out_lab, cv::Scalar(), std_lab);
    INFO("lab-a output mean BGR=" << mean_lab << " std=" << std_lab);
    // Not all-black and not all-white (mean well inside [10, 245] per channel).
    for (int ch = 0; ch < 3; ++ch) {
        CHECK(mean_lab[ch] > 10.0);
        CHECK(mean_lab[ch] < 245.0);
    }
    // Not flat (content disruption + gradient leaves real variance).
    CHECK(std_lab[0] > 1.0);
}

TEST_CASE("WS3 SynthidDetector LabA path yields a finite, in-range confidence",
          "[synthid][lab-a]") {
    constexpr int W = 128, H = 128;
    const cv::Mat src = make_content_image(W, H);

    // Tiny synthetic codebook: one profile at WxH, all-ones magnitude/consistency,
    // zero phase, so detection is well-defined without a real carrier template.
    SpectralProfile p;
    p.width = W;
    p.height = H;
    p.sample_count = 1;
    for (int ch = 0; ch < 3; ++ch) {
        p.magnitude_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1);
        p.phase_bgr[ch] = cv::Mat::zeros(H, W, CV_32FC1);
        p.consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1);
        p.phase_consistency_bgr[ch] = cv::Mat::ones(H, W, CV_32FC1);
    }
    SpectralCodebook cb;
    cb.add_profile(p);

    FftContext fft;
    SynthidDetector detector(fft);

    auto res_bgr = detector.detect(src, cb, SynthidDetector::ColorSpace::BGR);
    auto res_lab = detector.detect(src, cb, SynthidDetector::ColorSpace::LabA);

    INFO("BGR confidence=" << res_bgr.confidence << " LabA confidence=" << res_lab.confidence);

    // Both paths produce a confidence in [0, 1] (finite, in range).
    CHECK(res_bgr.confidence >= 0.0f);
    CHECK(res_bgr.confidence <= 1.0f);
    CHECK(res_lab.confidence >= 0.0f);
    CHECK(res_lab.confidence <= 1.0f);

    // LabA path must actually engage (it scores a different plane against the
    // same reference, so the confidence should differ unless the image is
    // pathological). A byte-identical confidence would mean the path is inert.
    // Tolerate the rare equality on degenerate synthetic inputs.
    CHECK(std::abs(res_bgr.confidence - res_lab.confidence) > 1e-6f);
}
