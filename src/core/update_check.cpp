#ifdef WMR_UPDATE_CHECK
#include "core/update_check.hpp"

namespace wmr {

// Filled in across later tasks. The stub fetch + no-op orchestrator let the
// build harness link before any real logic exists.
FetchResult fetch_latest_release(const std::string& /*etag*/) {
    return {/*ok=*/false, /*http_code=*/0, /*body=*/"", /*etag=*/"", /*error=*/"unimplemented"};
}

// Resolves the default fetch when the caller omits it. Defined out-of-line so
// the header's default argument `FetchFn fetch = {}` can be replaced by a real
// default at the call site in run_cli (which passes fetch_latest_release).
void maybe_check_for_update(bool /*no_update_check*/, FetchFn /*fetch*/) {
    // no-op for now
}

}  // namespace wmr
#endif  // WMR_UPDATE_CHECK
