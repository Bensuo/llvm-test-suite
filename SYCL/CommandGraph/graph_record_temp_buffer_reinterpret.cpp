// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** This test creates a temporary buffer (which is reinterpreted from the main
 * application buffers) which is used in kernels but destroyed before
 * finalization and execution of the graph. The original buffers lifetime
 * extends until after execution of the graph.
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  calculate_reference_data(iterations, size, referenceA, referenceB,
                           referenceC);

  {
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        graph;
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    testQueue.begin_recording(graph);

    // Create some temporary buffers only for recording
    {
      auto bufferA2 = bufferA.template reinterpret<T, 1>(bufferA.get_range());
      auto bufferB2 = bufferB.template reinterpret<T, 1>(bufferB.get_range());
      auto bufferC2 = bufferC.template reinterpret<T, 1>(bufferC.get_range());

      // Record commands to graph
      run_kernels(testQueue, size, bufferA2, bufferB2, bufferC2);

      testQueue.end_recording();
    }
    auto graphExec = graph.finalize(testQueue.get_context());

    // Execute several iterations of the graph
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit(graphExec);
    }
    // Perform a wait on all graph submissions.
    testQueue.wait();
  }

  bool failed = false;
  failed = referenceA != dataA;
  failed = referenceB != dataB;
  failed = referenceC != dataC;

  return failed;
}