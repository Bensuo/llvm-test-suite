// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** This test creates a temporary buffer which is used in kernels but
 * destroyed before finalization and execution of the graph.
 */

#include "graph_common.hpp"

using namespace sycl;

class vector_add_temp;
class temp_write;

int main() {
  queue testQueue;

  using T = int;

  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceC(dataC);
  for (unsigned n = 0; n < iterations; n++) {
    for (size_t i = 0; i < size; i++) {
      referenceC[i] += (dataA[i] + dataB[i]) + 1;
    }
  }

  {
    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        graph;
    buffer<T> bufferA{dataA.data(), range<1>{dataA.size()}};
    buffer<T> bufferB{dataB.data(), range<1>{dataB.size()}};
    buffer<T> bufferC{dataC.data(), range<1>{dataC.size()}};

    testQueue.begin_recording(graph);

    // Create a temporary output buffer to use between kernels.
    {
      buffer<T> bufferTemp{range<1>{dataA.size()}};

      // Vector add to temporary output buffer
      testQueue.submit([&](handler &cgh) {
        auto ptrA = bufferA.get_access<access::mode::read>(cgh);
        auto ptrB = bufferB.get_access<access::mode::read>(cgh);
        auto ptrOut = bufferTemp.get_access<access::mode::write>(cgh);
        cgh.parallel_for<vector_add_temp>(range<1>(size), [=](item<1> id) {
          ptrOut[id] = ptrA[id] + ptrB[id];
        });
      });

      // Modify temp buffer and write to output buffer
      testQueue.submit([&](handler &cgh) {
        auto ptrTemp = bufferTemp.get_access<access::mode::read>(cgh);
        auto ptrOut = bufferC.get_access<access::mode::write>(cgh);
        cgh.parallel_for<temp_write>(
            range<1>(size), [=](item<1> id) { ptrOut[id] += ptrTemp[id] + 1; });
      });
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
  failed |= referenceC != dataC;

  return failed;
}