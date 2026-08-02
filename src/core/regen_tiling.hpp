#pragma once
#include <opencv2/core.hpp>
#include <vector>
namespace wmr {

struct RegenTile {
    cv::Rect rect;     // pixel rect within the source image (clipped to bounds)
    cv::Mat  weight;   // per-pixel raised-cosine feather [0,1], same size as rect
};

// Tile [0,w)x[0,h) into overlapping square tiles of edge `tile_size` with `overlap`
// pixels of feather between neighbors (MultiDiffusion-style). Edge tiles are clipped
// to the image. Interior tile sides that have a neighbor ramp 0..1 over the overlap;
// sides on the image boundary stay 1 (no neighbor to blend with). Weights are in [0,1]
// and need NOT sum to 1 across tiles: the caller (Regenerator) normalizes by the
// per-pixel weight sum. Pure; OpenCV-only; unit-tested.
std::vector<RegenTile> build_regen_tiles(int width, int height, int tile_size, int overlap);

} // namespace wmr
