// REQUIRES: level_zero, gpu
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

/**  Tests the existence of the vendor test macro for graphs
 */

#include "graph_common.hpp"

using namespace sycl;

int main() {
#ifndef SYCL_EXT_ONEAPI_GRAPH
  std::cout << "SYCL_EXT_ONEAPI_GRAPH vendor test macro not defined\n";
  std::abort();
#else
  return SYCL_EXT_ONEAPI_GRAPH == 0;
#endif
}