// Task 1 link probe for the SynthID diffusion-regen vendor (Phase 2).
//
// This TU exists ONLY to prove that leejet/stable-diffusion.cpp is vendored,
// builds, and links into the `wmr` target under WMR_BUILD_REGEN. It touches the
// public API surface the later feature tasks will build on (ctx + img-gen param
// init, plus a free-standing utility) so a signature drift in the submodule
// surfaces here rather than in feature code. It is NOT registered as a test:
// `target_sources` pulls it into the wmr executable directly, so a successful
// `WMR_BUILD_REGEN=1` link is the pass signal. No feature sources live under
// src/ yet (Tasks 2-5 append them one-per-task); this file must not be the
// place any of them land.
#include <stable-diffusion.h>

int regen_link_probe() {
    sd_ctx_params_t cp;
    sd_ctx_params_init(&cp);
    sd_img_gen_params_t ip;
    sd_img_gen_params_init(&ip);
    (void)cp;
    (void)ip;
    return (int)sd_get_num_physical_cores();
}
