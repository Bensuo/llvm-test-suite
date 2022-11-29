// REQUIRES: level_zero, gpu
// RUN: %clangxx -pthread -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

// Test finalizing and submitting a graph in a threaded situation

#include "graph_common.hpp"

#include <thread>

using namespace sycl;

int main() {
  queue testQueue;

  using T = int;

  const unsigned iterations = std::thread::hardware_concurrency();
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

    // Record commands to graph
    run_kernels(testQueue, size, bufferA, bufferB, bufferC);

    testQueue.end_recording();
    auto finalizeGraph = [&]() {
      auto graphExec = graph.finalize(testQueue.get_context());
      testQueue.submit([&](sycl::handler &cgh) { cgh.exec_graph(graphExec); });
    };

    std::vector<std::thread> threads;
    threads.reserve(iterations);

    for (unsigned i = 0; i < iterations; ++i) {
      threads.emplace_back(finalizeGraph);
    }

    for (unsigned i = 0; i < iterations; ++i) {
      threads[i].join();
    }

    // Perform a wait on all graph submissions.
    testQueue.wait();
  }

  bool failed = false;
  failed |= referenceA != dataA;
  failed |= referenceB != dataB;
  failed |= referenceC != dataC;

  return failed;
}
