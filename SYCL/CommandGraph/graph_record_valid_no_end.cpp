// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** Tests obtaining a finalized, executable graph from a graph which is
 * currently being recorded to (no end_recording called)
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  ext::oneapi::experimental::command_graph graph;
  {
    queue myQueue;
    myQueue.begin_recording(graph);
  }

  try {
    auto graphExec = graph.finalize(testQueue.get_context());
    testQueue.submit([&](handler &cgh) { cgh.exec_graph(graphExec); });
  } catch (sycl::exception &e) {
    std::cout << "Exception thrown on finalize or submission.\n";
    std::abort();
  }
  testQueue.wait();
  return 0;
}
