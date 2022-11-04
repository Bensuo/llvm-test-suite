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
  ext::codeplay::queue_state state = testQueue.get_info<info::queue::state>();
  failure |= (state == ext::oneapi::experimental::queue_state::executing);

  ext::codeplay::command_graph<ext::codeplay::graph_state::recordable> graph;
  testQueue.begin_recording(graph);
  state = testQueue.get_info<info::queue::state>();
  failure |= (state == ext::oneapi::experimental::queue_state::recording);

  testQueue.end_recording();
  state = testQueue.get_info<info::queue::state>();
  failure |= (state == ext::oneapi::experimental::queue_state::executing);

  return failure;
}