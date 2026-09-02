#pragma once
// ORT-backed surrogate-detector suite for the anti-detection adversarial stage.
// Only compiled when the adversarial feature pulled in ONNX Runtime
// (WMR_ANTIDETECT_ADVERSARIAL); the manifest it consumes is ORT-free
// (detector_suite.hpp).
#ifdef WMR_ANTIDETECT_ADVERSARIAL

#include "core/detector_suite.hpp"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace wmr::antidetect {

class DetectorSuite {
public:
    // Loads one Ort::Session per spec from models_dir/<filename>. Returns
    // false (with spdlog errors) when nothing could be loaded.
    bool initialize(const std::vector<const SurrogateSpec*>& specs,
                    const std::filesystem::path& models_dir);
    bool is_ready() const { return !models_.empty(); }

    // One p_fake in [0,1] per loaded surrogate, manifest order.
    std::vector<float> score(const cv::Mat& bgr_u8) const;

    const std::vector<const SurrogateSpec*>& specs() const { return specs_; }

private:
    struct Model {
        const SurrogateSpec* spec = nullptr;
        std::unique_ptr<Ort::Session> session;
        std::string in_name;
        std::string out_name;
    };
    // Shared Ort::Env, process lifetime. The suite is held by a deliberately
    // leaked singleton (the NcnnDenoiser rationale: destroying an ORT session
    // during C++ static teardown races the runtime's own teardown).
    std::shared_ptr<Ort::Env> env_;
    std::vector<Model> models_;
    std::vector<const SurrogateSpec*> specs_;
};

} // namespace wmr::antidetect
#endif
