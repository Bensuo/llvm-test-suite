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

  bool changedState = testQueue.end_recording();
  failure |= changedState;

  ext::oneapi::experimental::command_graph graph;
  changedState = testQueue.begin_recording(graph);
  failure |= !changedState;

  // Recording to same graph is not an exception
  changedState = testQueue.begin_recording(graph);
  failure |= changedState;

  changedState = testQueue.end_recording();
  failure |= !changedState;

  changedState = testQueue.end_recording();
  failure |= changedState;

  return failure;
}