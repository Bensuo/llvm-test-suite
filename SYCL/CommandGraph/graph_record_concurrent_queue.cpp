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
  testQueue.begin_recording(graphA);

  try {
    ext::oneapi::experimental::command_graph graphB;
    testQueue.begin_recording(graphB);
  } catch (sycl::exception &e) {
    auto stdErrc = e.code().value();
    if (stdErrc == static_cast<int>(errc::invalid)) {
      success = true;
    }
  }

  testQueue.end_recording();

  return success;
}