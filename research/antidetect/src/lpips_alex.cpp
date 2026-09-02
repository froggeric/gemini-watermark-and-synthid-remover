#include "core/lpips_alex.hpp"

#include "lpips_alex_l2.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace wmr::antidetect {
namespace {

using wmr::embedded::kLpipsAlexConv1B;
using wmr::embedded::kLpipsAlexConv1W;
using wmr::embedded::kLpipsAlexConv2B;
using wmr::embedded::kLpipsAlexConv2W;
using wmr::embedded::kLpipsAlexLin2W;

constexpr int kConv1Out = 64, kConv1In = 3, kConv1K = 11, kConv1Stride = 4,
              kConv1Pad = 2;
constexpr int kConv2Out = 192, kConv2In = 64, kConv2K = 5, kConv2Stride = 1,
              kConv2Pad = 2;
constexpr float kPoolK = 3, kPoolStride = 2;
// ImageNet stats in RGB order (the network input after BGR->RGB).
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};

int conv_out_dim(int in, int k, int stride, int pad) {
    return (in + 2 * pad - k) / stride + 1;
}

// Zero-padded strided convolution of a plane stack. Weights are indexed
// [out][in][ky][kx] (torch layout, flat). Written as flat loops over contiguous
// output rows so the compiler vectorizes; a 512px input costs ~0.5 GFLOP.
void conv2d(const std::vector<cv::Mat>& in, int in_c, const float* w,
            const float* bias, int out_c, int k, int stride, int pad,
            std::vector<cv::Mat>& out) {
    const int oh = conv_out_dim(in[0].rows, k, stride, pad);
    const int ow = conv_out_dim(in[0].cols, k, stride, pad);

    // Zero pad each input plane once (torch F.conv2d pads with zeros).
    std::vector<cv::Mat> padded(in_c);
    if (pad > 0) {
        for (int c = 0; c < in_c; ++c)
            cv::copyMakeBorder(in[c], padded[c], pad, pad, pad, pad,
                               cv::BORDER_CONSTANT, cv::Scalar(0));
    } else {
        padded = in;
    }

    out.assign(out_c, {});
    const int span = k * k;
    for (int oc = 0; oc < out_c; ++oc) {
        cv::Mat acc(oh, ow, CV_32F);
        const float* wk = w + static_cast<size_t>(oc) * in_c * span;
        for (int y = 0; y < oh; ++y) {
            float* dst = acc.ptr<float>(y);
            for (int x = 0; x < ow; ++x) dst[x] = bias[oc];
        }
        for (int ic = 0; ic < in_c; ++ic) {
            const float* wic = wk + static_cast<size_t>(ic) * span;
            const cv::Mat& p = padded[ic];
            for (int y = 0; y < oh; ++y) {
                float* dst = acc.ptr<float>(y);
                for (int ky = 0; ky < k; ++ky) {
                    const float* src = p.ptr<float>(y * stride + ky);
                    const float* wrow = wic + ky * k;
                    for (int x = 0; x < ow; ++x) {
                        const int base = x * stride;
                        float s = 0.0f;
                        for (int kx = 0; kx < k; ++kx)
                            s += src[base + kx] * wrow[kx];
                        dst[x] += s;
                    }
                }
            }
        }
        // relu in place (both LPIPS convs are immediately relu'd).
        cv::Mat rel;
        cv::max(acc, 0.0f, rel);
        out[oc] = rel;
    }
}

// 3x3 stride-2 max pool, torch semantics (ceil_mode=False).
void maxpool3x2(const std::vector<cv::Mat>& in, std::vector<cv::Mat>& out) {
    const int oh = (in[0].rows - 3) / 2 + 1;
    const int ow = (in[0].cols - 3) / 2 + 1;
    out.resize(in.size());
    for (size_t c = 0; c < in.size(); ++c) {
        cv::Mat m(oh, ow, CV_32F);
        for (int y = 0; y < oh; ++y) {
            float* dst = m.ptr<float>(y);
            for (int x = 0; x < ow; ++x) {
                float v = -1e30f;
                for (int ky = 0; ky < 3; ++ky) {
                    const float* src = in[c].ptr<float>(y * 2 + ky);
                    for (int kx = 0; kx < 3; ++kx)
                        v = std::max(v, src[x * 2 + kx]);
                }
                dst[x] = v;
            }
        }
        out[c] = m;
    }
}

} // namespace

void LpipsAlex::features(const cv::Mat& bgr_u8, std::vector<cv::Mat>& planes) const {
    // BGR u8 -> RGB float [-1,1] -> ImageNet-normalized planes.
    std::vector<cv::Mat> scaled(3);
    for (int c = 0; c < 3; ++c) {
        cv::Mat f, one;
        cv::extractChannel(bgr_u8, f, 2 - c);  // RGB order
        f.convertTo(one, CV_32F, 1.0 / 255.0);
        one = one * 2.0f - 1.0f;
        one = (one - kMean[c]) / kStd[c];
        scaled[c] = one;
    }

    std::vector<cv::Mat> conv1, pooled, conv2;
    conv2d(scaled, kConv1In, kLpipsAlexConv1W, kLpipsAlexConv1B, kConv1Out,
           kConv1K, kConv1Stride, kConv1Pad, conv1);
    maxpool3x2(conv1, pooled);
    conv2d(pooled, kConv2In, kLpipsAlexConv2W, kLpipsAlexConv2B, kConv2Out,
           kConv2K, kConv2Stride, kConv2Pad, conv2);

    // Per-channel spatial unit normalization (LPIPS unit_normalize).
    planes.assign(conv2.size(), {});
    for (size_t c = 0; c < conv2.size(); ++c) {
        cv::Mat sq;
        cv::multiply(conv2[c], conv2[c], sq);
        const float sum = static_cast<float>(cv::sum(sq)[0]);
        const float inv = 1.0f / std::sqrt(std::max(sum, 1e-8f));
        cv::Mat n;
        conv2[c].convertTo(n, CV_32F, inv);
        planes[c] = n;
    }
}

float LpipsAlex::distance(const cv::Mat& a_bgr_u8, const cv::Mat& b_bgr_u8) const {
    if (a_bgr_u8.size() != b_bgr_u8.size()) return 1.0f;
    std::vector<cv::Mat> fa, fb;
    features(a_bgr_u8, fa);
    features(b_bgr_u8, fb);

    // mean over space of sum_c w_c * (fa_c - fb_c)^2
    double total = 0.0;
    for (size_t c = 0; c < fa.size(); ++c) {
        cv::Mat d;
        cv::subtract(fa[c], fb[c], d);
        cv::multiply(d, d, d);
        total += static_cast<double>(kLpipsAlexLin2W[c]) * cv::sum(d)[0];
    }
    return static_cast<float>(total / (fa.empty() ? 1.0 : fa[0].total()));
}

float LpipsAlex::distance_capped(const cv::Mat& a_bgr_u8, const cv::Mat& b_bgr_u8,
                                 int max_side) const {
    const int m = std::max(a_bgr_u8.rows, a_bgr_u8.cols);
    if (m <= max_side) return distance(a_bgr_u8, b_bgr_u8);
    const double s = static_cast<double>(max_side) / m;
    cv::Mat ra, rb;
    cv::resize(a_bgr_u8, ra, cv::Size(), s, s, cv::INTER_AREA);
    cv::resize(b_bgr_u8, rb, cv::Size(), s, s, cv::INTER_AREA);
    return distance(ra, rb);
}

} // namespace wmr::antidetect
