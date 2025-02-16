// Common header for SYCL command_graph tests

#include <sycl/sycl.hpp>

#include <sycl/ext/oneapi/experimental/graph.hpp>

#include <numeric>

// Some test constants
constexpr size_t size = 1024;
constexpr unsigned iterations = 5;

// Kernel declarations for use in run_kernels()
class increment_kernel;
class add_kernel;
class subtract_kernel;
class decrement_kernel;

// Kernel declarations for use in run_kernels_usm()
class increment_kernel_usm;
class add_kernel_usm;
class subtract_kernel_usm;
class decrement_kernel_usm;

using namespace sycl;

/** Runs a series of kernels based on the examples from the initial Graphs
 * sample code repo.
 *
 */
template <typename T>
void run_kernels(sycl::queue q, const size_t size, buffer<T> dataA,
                 buffer<T> dataB, buffer<T> dataC) {
  // Read & write Buffer A
  q.submit([&](handler &cgh) {
    auto pData = dataA.template get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<increment_kernel>(range<1>(size),
                                       [=](item<1> id) { pData[id]++; });
  });

  // Reads Buffer A
  // Read & Write Buffer B
  q.submit([&](handler &cgh) {
    auto pData1 = dataA.template get_access<access::mode::read>(cgh);
    auto pData2 = dataB.template get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<add_kernel>(range<1>(size),
                                 [=](item<1> id) { pData2[id] += pData1[id]; });
  });

  // Reads Buffer A
  // Read & writes Buffer C
  q.submit([&](handler &cgh) {
    auto pData1 = dataA.template get_access<access::mode::read>(cgh);
    auto pData2 = dataC.template get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<subtract_kernel>(
        range<1>(size), [=](item<1> id) { pData2[id] -= pData1[id]; });
  });

  // Read & write Buffers B and C
  q.submit([&](handler &cgh) {
    auto pData1 = dataB.template get_access<access::mode::read_write>(cgh);
    auto pData2 = dataC.template get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<decrement_kernel>(range<1>(size), [=](item<1> id) {
      pData1[id]--;
      pData2[id]--;
    });
  });
}

/** Runs a series of kernels based on the examples from the initial Graphs
 * sample code repo.
 *
 */
template <typename T>
void run_kernels_usm(sycl::queue q, const size_t size, T *dataA, T *dataB,
                     T *dataC) {
  // Read & write Buffer A
  q.submit([&](handler &cgh) {
    cgh.parallel_for<increment_kernel_usm>(range<1>(size), [=](item<1> id) {
      auto linID = id.get_linear_id();
      dataA[linID]++;
    });
  });

  // Reads Buffer A
  // Read & Write Buffer B
  q.submit([&](handler &cgh) {
    cgh.parallel_for<add_kernel_usm>(range<1>(size), [=](item<1> id) {
      auto linID = id.get_linear_id();
      dataB[linID] += dataA[linID];
    });
  });

  // Reads Buffer A
  // Read & writes Buffer C
  q.submit([&](handler &cgh) {
    cgh.parallel_for<subtract_kernel_usm>(range<1>(size), [=](item<1> id) {
      auto linID = id.get_linear_id();
      dataC[linID] -= dataA[linID];
    });
  });

  // Read & write Buffers B and C
  q.submit([&](handler &cgh) {
    cgh.parallel_for<decrement_kernel_usm>(range<1>(size), [=](item<1> id) {
      auto linID = id.get_linear_id();
      dataB[linID]--;
      dataC[linID]--;
    });
  });
}

/** Calculates reference data on the host for a given number of executions of
 * the kernels in run_kernels()
 *
 */
template <typename T>
void calculate_reference_data(size_t iterations, size_t size,
                              std::vector<T> &referenceA,
                              std::vector<T> &referenceB,
                              std::vector<T> &referenceC) {
  for (unsigned n = 0; n < iterations; n++) {
    for (size_t i = 0; i < size; i++) {
      referenceA[i]++;
      referenceB[i] += referenceA[i];
      referenceC[i] -= referenceA[i];
      referenceB[i]--;
      referenceC[i]--;
    }
  }
}