// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/**  Tests the ability to finalize a command graph while it is currently being
 * recorded to.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  ext::oneapi::experimental::command_graph graph;
  graph.begin_recording(testQueue);

  try {
    graph.finalize(testQueue.get_context());
  } catch (sycl::exception &e) {
    std::cout << "Exception thrown on finalize."
              << "\n";
    std::abort();
  }

  graph.end_recording();
  return 0;
}