#include "core/regen_tiling.hpp"
#include <algorithm>
#include <cmath>
namespace wmr {

namespace {
// Per-axis feather weight for column/row index `i` in a tile of edge `n`.
// `nbr_lo` true means there is a neighbor above/left (feather 0..1 over the first
// `ov` pixels); `nbr_hi` true means a neighbor below/right (feather 1..0 over the
// last `ov` pixels). A side at the image boundary (no neighbor) stays 1. The two
// sides are combined with min: in a tile larger than 2*ov the lo and hi ramps never
// overlap, so min picks the active one; the product form (edge_ramp(i)*edge_ramp(n-1-i))
// was WRONG because each factor independently fired on the opposite margin and zeroed
// the near edge (e.g. factor2 at i=0 sees n-1 and returns 0 -> top row became 0).
float axis_weight(int i, int n, int ov, bool nbr_lo, bool nbr_hi) {
    if (ov <= 0) return 1.0f;
    auto ramp = [](int idx, int o) {
        return 0.5f * (1.0f - static_cast<float>(std::cos(
            static_cast<double>(idx) / o * CV_PI)));
    };
    float w = 1.0f;
    if (nbr_lo && i < ov) w = std::min(w, ramp(i, ov));
    if (nbr_hi) {
        int d = n - 1 - i;
        if (d < ov) w = std::min(w, ramp(d, ov));
    }
    return w;
}
} // namespace

std::vector<RegenTile> build_regen_tiles(int width, int height, int tile_size, int overlap) {
    std::vector<RegenTile> tiles;
    if (width <= 0 || height <= 0 || tile_size <= 0) return tiles;
    if (overlap < 0) overlap = 0;
    if (overlap >= tile_size) overlap = tile_size / 2;

    // Step so adjacent tiles overlap by `overlap`; the final tile is snapped flush to
    // the image edge (which may make its overlap with the neighbor larger than nominal,
    // producing a region where both tiles are ~1 -> a valid weighted average after the
    // caller's per-pixel normalization).
    auto starts = [](int n, int tile, int ov) {
        std::vector<int> s;
        if (n <= tile) { s.push_back(0); return s; }
        int x = 0;
        while (true) {
            s.push_back(x);
            if (x + tile >= n) break;
            x += (tile - ov);
            if (x + tile > n) x = n - tile;  // snap final tile flush to the edge
            if (x <= s.back()) break;        // safety against a pathological loop
        }
        return s;
    };
    auto xs = starts(width,  tile_size, overlap);
    auto ys = starts(height, tile_size, overlap);

    for (int y0 : ys) {
        for (int x0 : xs) {
            int tw = std::min(tile_size, width  - x0);
            int th = std::min(tile_size, height - y0);
            RegenTile t;
            t.rect = cv::Rect(x0, y0, tw, th);
            t.weight = cv::Mat(th, tw, CV_32FC1);
            // Boundary sides (no neighbor) stay at weight 1; interior sides feather
            // 0..1 over `overlap`. Two adjacent tiles therefore ramp 1->0 and 0->1
            // across their shared overlap -> their sum is >0 everywhere (no holes).
            bool nbr_top   = (y0 != ys.front());
            bool nbr_bot   = (y0 + th < height);
            bool nbr_left  = (x0 != xs.front());
            bool nbr_right = (x0 + tw < width);
            for (int y = 0; y < th; ++y) {
                float wy = axis_weight(y, th, overlap, nbr_top, nbr_bot);
                for (int x = 0; x < tw; ++x) {
                    float wx = axis_weight(x, tw, overlap, nbr_left, nbr_right);
                    t.weight.at<float>(y, x) = wx * wy;
                }
            }
            tiles.push_back(std::move(t));
        }
    }
    return tiles;
}

} // namespace wmr
