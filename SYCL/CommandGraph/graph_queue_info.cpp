// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/**  Tests the return values from queue graph functions which change the
 * internal queue state
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  bool failure = false;
  ext::oneapi::experimental::queue_state state =
      testQueue.get_info<info::queue::state>();
  failure |= (state == ext::oneapi::experimental::queue_state::executing);

  ext::oneapi::experimental::command_graph<
      ext::oneapi::experimental::graph_state::modifiable>
      graph;
  graph.begin_recording(testQueue);
  state = testQueue.get_info<info::queue::state>();
  failure |= (state == ext::oneapi::experimental::queue_state::recording);

  graph.end_recording();
  state = testQueue.get_info<info::queue::state>();
  failure |= (state == ext::oneapi::experimental::queue_state::executing);

  return failure;
}