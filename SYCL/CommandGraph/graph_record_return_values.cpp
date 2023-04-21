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
  ext::oneapi::experimental::command_graph graph{testQueue.get_context(),
                                                 testQueue.get_device()};

  bool failure = false;

  bool changedState = graph.end_recording();
  failure |= changedState;

  changedState = graph.begin_recording(testQueue);
  failure |= !changedState;

  // Recording to same graph is not an exception
  changedState = graph.begin_recording(testQueue);
  failure |= changedState;

  changedState = graph.end_recording();
  failure |= !changedState;

  changedState = graph.end_recording();
  failure |= changedState;

  return failure;
}