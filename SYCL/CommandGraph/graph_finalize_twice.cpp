// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/**  Tests calling finalize() more than once on the same command_graph.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  ext::codeplay::command_graph<
      ext::oneapi::experimental::graph_state::modifiable>
      graph;
  auto graphExec = graph.finalize(testQueue.get_context());
  auto graphExec2 = graph.finalize(testQueue.get_context());

  return 0;
}