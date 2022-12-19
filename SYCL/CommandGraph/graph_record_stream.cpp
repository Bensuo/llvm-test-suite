// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/** This test checks that we can use a stream within a command_graph recording
 */

// TODO: Output should be validated using lit mechanisms but this can be done
// at a later point.

#include "graph_common.hpp"

using namespace sycl;

class stream_kernel;

int main() {
  queue testQueue;

  using T = int;
  std::vector<T> dataIn(size);

  // Initialize the data
  std::iota(dataIn.begin(), dataIn.end(), 1);

  {

    ext::oneapi::experimental::command_graph<
        ext::oneapi::experimental::graph_state::modifiable>
        graph;
    buffer<T> bufferIn{dataIn.data(), range<1>{dataIn.size()}};

    testQueue.begin_recording(graph);

    // Vector add to temporary output buffer
    testQueue.submit([&](handler &cgh) {
      auto accIn = bufferIn.get_access<access::mode::read>(cgh);
      sycl::stream out(16 * 16, 16, cgh);
      cgh.parallel_for<stream_kernel>(range<1>(size), [=](item<1> id) {
        out << "Val: " << accIn[id.get_linear_id()] << sycl::endl;
      });
    });
    testQueue.end_recording();

    auto graphExec = graph.finalize(testQueue.get_context());

    testQueue.submit([&](handler &cgh) { graphraph(graphExec); });

    // Perform a wait on all graph submissions.
    testQueue.wait();
  }

  return 0;
}
