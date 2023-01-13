// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/**  Tests attempting to begin recording to a new graph when recording is
 * already in progress on another.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  bool success = false;

  ext::oneapi::experimental::command_graph graphA;
  graphA.begin_recording(testQueue);

  try {
    ext::oneapi::experimental::command_graph graphB;
    graphB.begin_recording(testQueue);
  } catch (sycl::exception &e) {
    auto stdErrc = e.code().value();
    if (stdErrc == static_cast<int>(errc::invalid)) {
      success = true;
    }
  }

  graphA.end_recording();

  return success;
}