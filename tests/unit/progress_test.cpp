// Unit tests for the pure progress helpers (format_*, RateEstimator).
// The stateful reporters (ProgressReporter / ByteProgress / Stage) write to a
// stream and are verified manually on a TTY + piped, not here.

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "cli/progress.hpp"

#include <cmath>

using namespace wmr;
using Catch::Approx;

// --- format_bytes ---------------------------------------------------------

TEST_CASE("format_bytes: integer bytes under 1 KiB", "[progress]") {
    REQUIRE(format_bytes(0) == "0 B");
    REQUIRE(format_bytes(1) == "1 B");
    REQUIRE(format_bytes(512) == "512 B");
    REQUIRE(format_bytes(1023) == "1023 B");
}

TEST_CASE("format_bytes: KiB with one decimal", "[progress]") {
    REQUIRE(format_bytes(1024ull) == "1.0 KiB");
    REQUIRE(format_bytes(1024ull * 4) == "4.0 KiB");
    REQUIRE(format_bytes(static_cast<uint64_t>(1024.0 * 1.5)) == "1.5 KiB");
}

TEST_CASE("format_bytes: MiB and above with two decimals", "[progress]") {
    constexpr uint64_t MiB = 1024ull * 1024ull;
    constexpr uint64_t GiB = MiB * 1024ull;
    REQUIRE(format_bytes(MiB) == "1.00 MiB");
    // ~18.20 MiB
    REQUIRE(format_bytes(static_cast<uint64_t>(18.20 * MiB)) == "18.20 MiB");
    // ~3.41 GiB
    REQUIRE(format_bytes(static_cast<uint64_t>(3.41 * GiB)) == "3.41 GiB");
    REQUIRE(format_bytes(GiB) == "1.00 GiB");
}

TEST_CASE("format_bytes: TiB tier", "[progress]") {
    constexpr uint64_t GiB = 1024ull * 1024ull * 1024ull;
    constexpr uint64_t TiB = GiB * 1024ull;
    REQUIRE(format_bytes(TiB) == "1.00 TiB");
    REQUIRE(format_bytes(static_cast<uint64_t>(2.5 * TiB)) == "2.50 TiB");
}

// --- format_eta -----------------------------------------------------------

TEST_CASE("format_eta: zero / negative / NaN -> placeholder", "[progress]") {
    REQUIRE(format_eta(0.0) == "--");
    REQUIRE(format_eta(-1.0) == "--");
    REQUIRE(format_eta(std::nan("")) == "--");
}

TEST_CASE("format_eta: under 10s -> '<10s'", "[progress]") {
    REQUIRE(format_eta(0.5) == "<10s");
    REQUIRE(format_eta(9.99) == "<10s");
}

TEST_CASE("format_eta: seconds", "[progress]") {
    REQUIRE(format_eta(10.0) == "10s");
    REQUIRE(format_eta(45.4) == "45s");
    REQUIRE(format_eta(59.4) == "59s");
}

TEST_CASE("format_eta: minutes+seconds", "[progress]") {
    REQUIRE(format_eta(60.0) == "1m0s");
    REQUIRE(format_eta(167.0) == "2m47s");
    REQUIRE(format_eta(599.0) == "9m59s");
}

TEST_CASE("format_eta: hours+minutes", "[progress]") {
    REQUIRE(format_eta(3600.0) == "1h0m");
    REQUIRE(format_eta(231.0 * 60.0) == "3h51m");  // 231 min = 3h51m
}

// --- format_bar -----------------------------------------------------------

TEST_CASE("format_bar: empty and full", "[progress]") {
    REQUIRE(format_bar(0.0, 10) == "[----------]");
    REQUIRE(format_bar(1.0, 10) == "[##########]");
}

TEST_CASE("format_bar: halves", "[progress]") {
    REQUIRE(format_bar(0.5, 10) == "[#####-----]");
    REQUIRE(format_bar(0.25, 8) == "[##------]");
}

TEST_CASE("format_bar: clamps out-of-range frac", "[progress]") {
    REQUIRE(format_bar(-0.5, 6) == "[------]");
    REQUIRE(format_bar(2.0, 6) == "[######]");
    REQUIRE(format_bar(std::nan(""), 6) == "[------]");
}

TEST_CASE("format_bar: default width 24", "[progress]") {
    REQUIRE(format_bar(0.0).size() == 26);  // 24 + '[' + ']'
    REQUIRE(format_bar(0.5) == "[############" + std::string(12, '-') + "]");
}

TEST_CASE("format_bar: minimum width 2", "[progress]") {
    REQUIRE(format_bar(0.5, 0).size() == 4);  // clamped to width 2
    REQUIRE(format_bar(0.5, 0) == "[#-]");    // 1 of 2 cells filled at 50%
}

// --- RateEstimator --------------------------------------------------------

TEST_CASE("RateEstimator: no samples -> rate 0, eta hidden", "[progress]") {
    RateEstimator e(12);
    REQUIRE(e.sample_count() == 0);
    REQUIRE(e.rate_per_sec() == 0.0);
    REQUIRE(e.eta_seconds(12) == 0.0);
    REQUIRE_FALSE(e.eta_available());
}

TEST_CASE("RateEstimator: under 3 samples -> eta still hidden", "[progress]") {
    RateEstimator e(100);
    e.sample(1.0);
    e.sample(1.0);
    REQUIRE(e.sample_count() == 2);
    REQUIRE(e.rate_per_sec() == Approx(1.0).epsilon(0.01));
    REQUIRE_FALSE(e.eta_available());
    REQUIRE(e.eta_seconds(98) == 0.0);  // hidden despite a valid rate
}

TEST_CASE("RateEstimator: >=3 samples AND >=5% done -> eta available", "[progress]") {
    // total = 100; 5% = 5 units. 3 samples means 3 units done = 3% (< 5%) -> hidden.
    RateEstimator e_small(100);
    e_small.sample(2.0); e_small.sample(2.0); e_small.sample(2.0);
    REQUIRE(e_small.sample_count() == 3);
    REQUIRE_FALSE(e_small.eta_available());  // 3% < 5%

    // total = 60; 5% = 3 units. 3 samples = 5% -> available.
    RateEstimator e(60);
    e.sample(2.0); e.sample(2.0); e.sample(2.0);
    REQUIRE(e.eta_available());
    REQUIRE(e.rate_per_sec() == Approx(0.5).epsilon(0.01));
    // eta = remaining(57) * ewma_unit(2.0) = 114s
    REQUIRE(e.eta_seconds(57) == Approx(114.0).epsilon(0.01));
}

TEST_CASE("RateEstimator: range vs confident estimate", "[progress]") {
    // 3 samples + 5%+ -> available but NOT confident (need >=6 samples + 20%).
    RateEstimator e(30);
    for (int i = 0; i < 3; ++i) e.sample(1.0);
    REQUIRE(e.eta_available());
    REQUIRE_FALSE(e.eta_confident());

    // 6 samples + 20% (6/30 = 20%) -> confident.
    for (int i = 0; i < 3; ++i) e.sample(1.0);
    REQUIRE(e.eta_confident());
}

TEST_CASE("RateEstimator: EWMA tracks the latest samples", "[progress]") {
    RateEstimator e(100);
    e.sample(10.0);            // first sample: ewma = 10s/unit -> 0.1/s
    REQUIRE(e.rate_per_sec() == Approx(0.1).epsilon(0.05));
    e.sample(1.0);             // EWMA(alpha=0.3): 0.3*1 + 0.7*10 = 7.3 -> ~0.137/s
    REQUIRE(e.rate_per_sec() == Approx(0.137).epsilon(0.02));
}

TEST_CASE("RateEstimator: total=0 skips the done-fraction gate", "[progress]") {
    RateEstimator e(0);  // total unknown
    e.sample(1.0); e.sample(1.0); e.sample(1.0);
    REQUIRE(e.eta_available());   // samples-only gate
    REQUIRE_FALSE(e.eta_confident());  // still need >=6 for confidence
}

TEST_CASE("RateEstimator: skips non-positive unit times", "[progress]") {
    RateEstimator e(10);
    e.sample(0.0);   // ignored
    e.sample(-1.0);  // ignored
    REQUIRE(e.sample_count() == 0);
    e.sample(2.0);
    REQUIRE(e.sample_count() == 1);
}
