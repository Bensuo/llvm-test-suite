// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** Tests recording and submission of a graph containing usm memcpy commands.
 */

#include "graph_common.hpp"

using namespace sycl;

class kernel_mod_a;
class kernel_mod_b;

int main() {
  queue testQueue;

  using T = int;

  const T modValue = 7;
  std::vector<T> dataA(size), dataB(size), dataC(size);

  // Initialize the data
  std::iota(dataA.begin(), dataA.end(), 1);
  std::iota(dataB.begin(), dataB.end(), 10);
  std::iota(dataC.begin(), dataC.end(), 1000);

  // Create reference data for output
  std::vector<T> referenceA(dataA), referenceB(dataB), referenceC(dataC);
  for (size_t i = 0; i < iterations; i++) {
    for (size_t j = 0; j < size; j++) {
      referenceA[j] = referenceB[j];
      referenceA[j] += modValue;
      referenceB[j] = referenceA[j];
      referenceB[j] += modValue;
      referenceC[j] = referenceB[j];
    }
  }

  ext::oneapi::experimental::command_graph<
      ext::oneapi::experimental::graph_state::modifiable>
      graph;

  {
    auto ptrA = malloc_device<T>(dataA.size(), testQueue);
    testQueue.memcpy(ptrA, dataA.data(), dataA.size() * sizeof(T)).wait();
    auto ptrB = malloc_device<T>(dataB.size(), testQueue);
    testQueue.memcpy(ptrB, dataB.data(), dataB.size() * sizeof(T)).wait();
    auto ptrC = malloc_device<T>(dataC.size(), testQueue);
    testQueue.memcpy(ptrC, dataC.data(), dataC.size() * sizeof(T)).wait();

    testQueue.begin_recording(graph);

    // Record commands to graph
    // memcpy from B to A
    testQueue.copy(ptrB, ptrA, size);
    // Read & write A
    testQueue.submit([&](handler &cgh) {
      cgh.parallel_for<kernel_mod_a>(range<1>(size), [=](item<1> id) {
        auto linID = id.get_linear_id();
        ptrA[linID] += modValue;
      });
    });

    // memcpy from A to B
    testQueue.copy(ptrA, ptrB, size);

    // Read and write B
    testQueue.submit([&](handler &cgh) {
      cgh.parallel_for<kernel_mod_b>(range<1>(size), [=](item<1> id) {
        auto linID = id.get_linear_id();
        ptrB[linID] += modValue;
      });
    });

    // memcpy from B to C
    testQueue.copy(ptrB, ptrC, size);

    testQueue.end_recording();
    auto graphExec = graph.finalize(testQueue.get_context());

    // Execute graph over n iterations
    for (unsigned n = 0; n < iterations; n++) {
      testQueue.submit(graphExec);
    }
    // Perform a wait on all graph submissions.
    testQueue.wait();

    testQueue.memcpy(dataA.data(), ptrA, dataA.size() * sizeof(T)).wait();
    testQueue.memcpy(dataB.data(), ptrB, dataB.size() * sizeof(T)).wait();
    testQueue.memcpy(dataC.data(), ptrC, dataC.size() * sizeof(T)).wait();

    free(ptrA, testQueue.get_context());
    free(ptrB, testQueue.get_context());
    free(ptrC, testQueue.get_context());
  }

  bool failed = false;
  failed |= referenceA != dataA;
  failed |= referenceB != dataB;
  failed |= referenceC != dataC;

  return failed;
}