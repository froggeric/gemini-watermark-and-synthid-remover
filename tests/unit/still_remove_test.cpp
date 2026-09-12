#include <catch2/catch_test_macros.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
#include "core/still_remove.hpp"

using namespace wmr;

namespace {
cv::Mat gray_with_v1_mark() {
    cv::Mat img(500, 500, CV_8UC3, cv::Scalar(90, 90, 90));
    WatermarkEngine engine;
    engine.add_watermark(img);              // V1 mark at the model position
    return img;
}
}

TEST_CASE("still-remove detects and removes an engine-added mark", "[still-remove]") {
    cv::Mat img = gray_with_v1_mark();
    WatermarkEngine engine;
    StillRemoveOutcome r = remove_still(engine, img, {});
    REQUIRE(r.outcome == StillOutcome::Removed);
    REQUIRE(r.bbox.has_value());
    REQUIRE(r.score > 0.0f);
    REQUIRE((r.variant == "V1" || r.variant == "V2"));
    REQUIRE((r.geometry_source == "model" || r.geometry_source == "auto/raw" ||
             r.geometry_source == "auto/snapped"));
}

TEST_CASE("still-remove reports NoWatermark on clean content", "[still-remove]") {
    // Busy synthetic content with no watermark: the detectors must not fire.
    cv::Mat img(500, 500, CV_8UC3);
    for (int y = 0; y < img.rows; ++y)
        for (int x = 0; x < img.cols; ++x)
            img.at<cv::Vec3b>(y, x) = {(uchar)((x * 7) % 251), (uchar)((y * 13) % 241), (uchar)((x * y) % 233)};
    WatermarkEngine engine;
    StillRemoveOutcome r = remove_still(engine, img, {});
    REQUIRE(r.outcome == StillOutcome::NoWatermark);
    REQUIRE_FALSE(r.bbox.has_value());
}

TEST_CASE("still-remove force path removes without detection", "[still-remove]") {
    cv::Mat img = gray_with_v1_mark();
    WatermarkEngine engine;
    StillRemoveOptions o; o.force = true;
    StillRemoveOutcome r = remove_still(engine, img, o);
    REQUIRE(r.outcome == StillOutcome::Removed);
    REQUIRE(r.forced);
    REQUIRE(r.geometry_source.empty());
}

TEST_CASE("still-remove maps exceptions to Failed", "[still-remove]") {
    cv::Mat empty;
    WatermarkEngine engine;
    StillRemoveOutcome r = remove_still(engine, empty, {});
    REQUIRE(r.outcome == StillOutcome::Failed);
    REQUIRE_FALSE(r.error.empty());
}

TEST_CASE("still-remove respects an explicit rect override", "[still-remove]") {
    cv::Mat img = gray_with_v1_mark();
    WatermarkEngine engine;
    // The V1 mark sits at margin (64,64) with size 96 at 500x500 (either dim <= 1024 -> 48? no:
    // V1: 48x48 if EITHER dim <= 1024 else 96x96; 500x500 -> 48 at margin (32,32)).
    // Do not hardcode: remove at a deliberately wrong-but-forced rect must still run.
    StillRemoveOptions o;
    o.geometry.rect = cv::Rect(32, 32, 48, 48);
    StillRemoveOutcome r = remove_still(engine, img, o);
    REQUIRE(r.outcome == StillOutcome::Removed);
    REQUIRE(r.geometry_source == "rect");
}

TEST_CASE("write_still_output writes decodable output", "[still-remove]") {
    cv::Mat img(20, 20, CV_8UC3, cv::Scalar(1, 2, 3));
    auto tmp = std::filesystem::temp_directory_path() / ("wmr_sr_" + std::to_string(getpid()) + ".png");
    REQUIRE(write_still_output(tmp, img, false));
    cv::Mat back = cv::imread(tmp.string());
    REQUIRE_FALSE(back.empty());
    REQUIRE(back.size() == img.size());
    std::filesystem::remove(tmp);
}
