#include "cli/cli_app.hpp"

#ifdef WMR_BUILD_REGEN
#include <cstdio>   // std::fflush
#include <cstdlib>  // std::_Exit
namespace wmr { bool regenerator_was_used(); }
#endif

int main(int argc, char* argv[]) {
    int rc = wmr::run_cli(argc, argv);
#ifdef WMR_BUILD_REGEN
    // If the regen backend (ggml/leejet stable-diffusion.cpp) was initialized this run,
    // its Metal/CUDA/Vulkan device lives in a ggml STATIC that aborts at process exit
    // (ggml-metal: GGML_ASSERT([rsets->data count] == 0) in ggml_metal_device_free,
    // called from the static vector<unique_ptr<ggml_metal_device>> destructor). The
    // leak of our Regenerator singleton doesn't cause it -- it's ggml's own teardown.
    // std::_Exit skips static destructors entirely (the standard llama.cpp/ggml
    // workaround for these Metal teardown aborts). Flush first so the saved-file +
    // log lines land. Only fires when an sd_ctx was actually created; lean /
    // regen-requested-but-no-model runs return normally.
    if (wmr::regenerator_was_used()) {
        std::fflush(nullptr);
        std::_Exit(rc);
    }
#endif
    return rc;
}
