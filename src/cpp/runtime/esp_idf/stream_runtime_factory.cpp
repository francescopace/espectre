#include "stream_runtime_factory.h"

#ifdef ESPECTRE_ENABLE_STREAM_RUNTIME
#include "stream_esp_idf_runtime.h"
#endif

namespace espectre {

std::unique_ptr<IEspectreRuntime> make_stream_runtime(const RuntimeConfig &config) {
#ifdef ESPECTRE_ENABLE_STREAM_RUNTIME
  return std::make_unique<StreamEspIdfRuntime>(config);
#else
  (void)config;
  return nullptr;
#endif
}

}  // namespace espectre
