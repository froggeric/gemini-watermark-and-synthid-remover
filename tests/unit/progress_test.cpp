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

// --- format_byte_line -------------------------------------------------------

// Visible length: byte count excluding ANSI escape sequences (bold labels add
// \033[1m ... \033[0m, which occupy no terminal columns).
static size_t visible_len(const std::string& s) {
    size_t n = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\033' && i + 1 < s.size() && s[i + 1] == '[') {
            size_t j = i + 2;
            while (j < s.size() && (s[j] < 0x40 || s[j] > 0x7e)) ++j;
            i = j;  // j is the final byte of the CSI sequence; the loop's ++i skips it
            continue;
        }
        ++n;
    }
    return n;
}

TEST_CASE("format_byte_line: percent clamps to 100 when done exceeds total", "[progress]") {
    // The live bug: a 1 KB redirect-page total latched, then a 6.9 GB download
    // rendered as "6.46 GiB / 1022 B  678855410%".
    std::string line = format_byte_line("model", 6940000000ull, 1022, 0.0, false, 0);
    REQUIRE(line.find("100%") != std::string::npos);
    REQUIRE(line.find("%") != std::string::npos);
    REQUIRE(line.find("678") == std::string::npos);
}

TEST_CASE("format_byte_line: normal uncapped line", "[progress]") {
    constexpr uint64_t KiB = 1024ull;
    std::string line = format_byte_line("m.bin", KiB, 2 * KiB, 0.0, false, 0);
    REQUIRE(line == "m.bin  1.0 KiB / 2.0 KiB  50%  [############------------]");
}

TEST_CASE("format_byte_line: unknown total shows bytes without percent", "[progress]") {
    std::string line = format_byte_line("m.bin", 2048, 0, 0.0, false, 0);
    REQUIRE(line == "m.bin  2.0 KiB");
    REQUIRE(line.find('%') == std::string::npos);
}

TEST_CASE("format_byte_line: cap shrinks the bar, then the label, never wraps", "[progress]") {
    constexpr uint64_t KiB = 1024ull;
    // Wide terminal: full line, 24-cell bar present.
    std::string wide = format_byte_line("a-very-long-model-name.safetensors", KiB, 2 * KiB,
                                        0.0, false, 100);
    REQUIRE(visible_len(wide) <= 100);
    REQUIRE(wide.find("####") != std::string::npos);   // bar kept (possibly shrunk)

    // Narrow terminal (60 cols): the label alone is 34 cols; the bar must shrink
    // or vanish and the visible line must fit.
    std::string narrow = format_byte_line("a-very-long-model-name.safetensors", KiB, 2 * KiB,
                                          0.0, false, 60);
    REQUIRE(visible_len(narrow) <= 60);
    // Sizes/percent (the information) survive; the label may be truncated.
    REQUIRE(narrow.find("1.0 KiB / 2.0 KiB") != std::string::npos);
    REQUIRE(narrow.find("50%") != std::string::npos);

    // Degenerate 30-col terminal: still no wrap, still shows the percent.
    std::string tiny = format_byte_line("a-very-long-model-name.safetensors", KiB, 2 * KiB,
                                        0.0, false, 30);
    REQUIRE(visible_len(tiny) <= 30);
    REQUIRE(tiny.find("50%") != std::string::npos);
}

TEST_CASE("format_byte_line: bold codes wrap a truncated label and take no columns", "[progress]") {
    constexpr uint64_t KiB = 1024ull;
    std::string line = format_byte_line("a-very-long-model-name.safetensors", KiB, 2 * KiB,
                                        0.0, true, 60);
    REQUIRE(visible_len(line) <= 60);
    // Balanced escapes: bold opens at the start, reset still present after
    // truncation (a cut reset would leave the rest of the terminal bold).
    REQUIRE(line.substr(0, 4) == "\033[1m");
    REQUIRE(line.find("\033[0m") != std::string::npos);
    // The reset must come before the payload (it closes the label), not just
    // anywhere: find the FIRST reset and check the payload follows it.
    size_t reset = line.find("\033[0m");
    REQUIRE(line.compare(reset + 4, 2, "  ") == 0);
}

// --- WindowedRate -----------------------------------------------------------

TEST_CASE("WindowedRate: bursty ticks produce a stable windowed rate", "[progress]") {
    // Per-tick rates alternate 5x above/below the mean (observed live: 5 MiB/s
    // and 303 MiB/s on consecutive ticks during a HuggingFace fetch). The
    // displayed rate must reflect the multi-second average, not the bursts.
    WindowedRate r(4.0);  // 4-second window
    // Two 4s windows, each carrying 40 MiB total => true rate 10 MiB/s.
    // Delivered in alternating bursts of 8 MiB and 0 per 0.4s tick.
    double done = 0.0;
    double t = 0.0;
    for (int tick = 0; tick < 20; ++tick) {          // window 1: t 0.0..7.6
        done += (tick % 2 == 0) ? 8.0 * 1024 * 1024 : 0.0;
        r.sample(static_cast<uint64_t>(done), t);
        t += 0.4;
    }
    // window 1 closed at t=4.0 with 40 MiB; window 2 closes at t=8.0
    for (int tick = 0; tick < 10; ++tick) {          // window 2
        done += (tick % 2 == 0) ? 8.0 * 1024 * 1024 : 0.0;
        r.sample(static_cast<uint64_t>(done), t);
        t += 0.4;
    }
    double rate = r.rate_per_sec();
    REQUIRE(rate == Approx(10.0 * 1024 * 1024).epsilon(0.15));
}

TEST_CASE("WindowedRate: no rate before 2s of data, since-start during first window", "[progress]") {
    WindowedRate r(4.0);
    r.sample(0, 0.0);
    r.sample(2ull * 1024 * 1024, 1.0);   // < 2s: hidden
    REQUIRE(r.rate_per_sec() == 0.0);
    REQUIRE_FALSE(r.full_window());
    r.sample(4ull * 1024 * 1024, 2.5);   // since-start avg = 4 MiB / 2.5s
    REQUIRE(r.rate_per_sec() == Approx(4.0 * 1024 * 1024 / 2.5).epsilon(0.05));
    REQUIRE_FALSE(r.full_window());      // no full window closed yet
}

TEST_CASE("WindowedRate: full window flag flips after TWO windows close", "[progress]") {
    WindowedRate r(4.0);
    uint64_t done = 0;
    double t = 0.0;
    int i = 0;
    for (; i <= 80 && !r.full_window(); ++i) {
        r.sample(done, t);
        done += 1024 * 1024;
        t += 0.25;
    }
    REQUIRE(r.full_window());
    REQUIRE(t >= 8.0);  // two 4s windows
}

TEST_CASE("WindowedRate: a stall decays the rate to zero", "[progress]") {
    WindowedRate r(4.0);
    uint64_t done = 0;
    double t = 0.0;
    for (; t < 8.0; t += 0.4) {           // 10 MiB/s for 8s
        done += static_cast<uint64_t>(10.0 * 1024 * 1024 * 0.4);
        r.sample(done, t);
    }
    REQUIRE(r.rate_per_sec() > 5.0 * 1024 * 1024);
    for (; t < 28.0; t += 0.4) {          // 20s of no progress (5 empty windows)
        r.sample(done, t);
    }
    REQUIRE(r.rate_per_sec() < 1.0 * 1024 * 1024);  // decayed toward 0
}

TEST_CASE("WindowedRate: non-monotonic time is ignored", "[progress]") {
    WindowedRate r(4.0);
    r.sample(1024, 1.0);
    r.sample(4096, 0.5);   // clock went backwards; must not corrupt state
    r.sample(8192, 3.5);
    REQUIRE(r.rate_per_sec() > 0.0);
}

TEST_CASE("format_byte_line: ETA hidden until the rate is trusted", "[progress]") {
    constexpr uint64_t KiB = 1024ull;
    // Rate available but WindowedRate has not closed a full window yet: the
    // rate shows, the ETA does not (a since-start average over <4s of bursty
    // CDN data produced ETAs swinging 15s <-> 15min in the field).
    std::string untrusted = format_byte_line("m.bin", KiB, 2 * KiB,
                                             static_cast<double>(KiB), false, 0, false);
    REQUIRE(untrusted.find("/s") != std::string::npos);
    REQUIRE(untrusted.find("ETA") == std::string::npos);
    std::string trusted = format_byte_line("m.bin", KiB, 2 * KiB,
                                           static_cast<double>(KiB), false, 0, true);
    REQUIRE(trusted.find("ETA") != std::string::npos);
    REQUIRE(trusted.find("ETA <10s") != std::string::npos);  // 1 MiB left at 1 MiB/s
}
