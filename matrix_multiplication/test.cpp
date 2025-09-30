#include "cxxopts.hpp"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "common.h"

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
#ifndef DTYPE_IN
#define DTYPE_IN std::bfloat16_t
#endif
#ifndef DTYPE_OUT
#define DTYPE_OUT std::bfloat16_t
#endif
#ifndef DTYPE_ACC
#define DTYPE_ACC float
#endif
using A_DATATYPE = DTYPE_IN;
using B_DATATYPE = DTYPE_IN;
using C_DATATYPE = DTYPE_OUT;
using ACC_DATATYPE = DTYPE_ACC;
#endif


struct TimingResult {
    double copy_to_us;
    double kernel_us;
    double copy_from_us;
    double total_us;
    bool success;
    std::chrono::high_resolution_clock::time_point kernel_start_time;
    std::chrono::high_resolution_clock::time_point kernel_end_time;
};


TimingResult run_matmul_once(
    xrt::kernel &kernel,
    xrt::bo &bo_instr,
    xrt::bo &bo_a,
    xrt::bo &bo_b,
    xrt::bo &bo_out,
    xrt::bo &bo_tmp1,
    xrt::bo &bo_trace,
    unsigned int opcode) {

    TimingResult res;
    res.success = false; 
    try {
        auto total_start = std::chrono::high_resolution_clock::now();

        // --- Copy to device ---
        auto copy_to_start = std::chrono::high_resolution_clock::now();
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_out.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (bo_trace.size()) bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto copy_to_end = std::chrono::high_resolution_clock::now();

        // --- Kernel execution ---
        res.kernel_start_time = std::chrono::high_resolution_clock::now();
        auto run = kernel(opcode, bo_instr, bo_instr.size() / sizeof(int),
                          bo_a, bo_b, bo_out, bo_tmp1, bo_trace);
        ert_cmd_state r = run.wait();
        res.kernel_end_time = std::chrono::high_resolution_clock::now();

        if (r != ERT_CMD_STATE_COMPLETED) {
            std::cerr << "[ERROR] Kernel did not complete successfully. State: " << r << std::endl;
            res.success = false;
        } else {
            res.success = true;
        }

        // --- Copy from device ---
        auto copy_from_start = std::chrono::high_resolution_clock::now();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto copy_from_end = std::chrono::high_resolution_clock::now();

        auto total_end = std::chrono::high_resolution_clock::now();

        res.copy_to_us = std::chrono::duration_cast<std::chrono::microseconds>(copy_to_end - copy_to_start).count();
        res.kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(res.kernel_end_time - res.kernel_start_time).count();
        res.copy_from_us = std::chrono::duration_cast<std::chrono::microseconds>(copy_from_end - copy_from_start).count();
        res.total_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();

    } catch (const std::exception& ex) {
        std::cerr << "[ERROR] Exception in run_matmul_once: " << ex.what() << std::endl;
        res.success = false;
    }
    return res;
}

// ============================================================================
// Golden Model and Verification Functions
// ============================================================================

void matmul_golden(
    const std::vector<A_DATATYPE>& A,
    const std::vector<B_DATATYPE>& B,
    std::vector<C_DATATYPE>& C_golden,
    int M, int K, int N)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            ACC_DATATYPE acc = 0;
            for (int k = 0; k < K; ++k) {
                acc += static_cast<ACC_DATATYPE>(A[m * K + k]) * static_cast<ACC_DATATYPE>(B[k * N + n]);
            }
            C_golden[m * N + n] = static_cast<C_DATATYPE>(acc);
        }
    }
}

int verify_results(
    const std::vector<C_DATATYPE>& C_npu,
    const std::vector<C_DATATYPE>& C_golden,
    int max_errors_to_print = 10,
    float tolerance = 1e-3)
{
    int errors = 0;
    for (size_t i = 0; i < C_golden.size(); ++i) {
        ACC_DATATYPE val_npu = static_cast<ACC_DATATYPE>(C_npu[i]);
        ACC_DATATYPE val_golden = static_cast<ACC_DATATYPE>(C_golden[i]);
        if (std::abs(val_npu - val_golden) > tolerance) {
            if (errors < max_errors_to_print) {
                std::cerr << "[ERROR] Mismatch at index " << i
                          << ": NPU=" << val_npu
                          << ", Golden=" << val_golden << std::endl;
            }
            errors++;
        }
    }
    return errors;
}


int main(int argc, const char *argv[]) {
  cxxopts::Options options("Matrix Matrix Multiplication Benchmark (3-thread parallel)");
  matmul_common::add_default_options(options);
  options.add_options()
    ("no-verify", "Skip the final golden model verification")
    ("a-file", "Input file for matrix A", cxxopts::value<std::string>())
    ("b-file", "Input file for matrix B", cxxopts::value<std::string>())
    ("c-file", "Output file for matrix C", cxxopts::value<std::string>());

  auto vm = options.parse(argc, argv);

  int verbosity = vm["verbosity"].as<int>();
  int trace_size = vm["trace_sz"].as<int>();
  bool skip_verify = vm["no-verify"].as<bool>();

 
  int n_iterations = 20;
  int n_warmup_iterations = 5;

  int M = vm["M"].as<int>();
  int K = vm["K"].as<int>();
  int N = vm["N"].as<int>();

  size_t A_SIZE = M * K * sizeof(A_DATATYPE);
  size_t B_SIZE = K * N * sizeof(B_DATATYPE);
  size_t C_SIZE = M * N * sizeof(C_DATATYPE);

  std::cout << "Loading device and xclbin...\n";
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());

  std::string kernel_name = vm["kernel"].as<std::string>();
  xrt::kernel kernel1(context, kernel_name);
  xrt::kernel kernel2(context, kernel_name);
  xrt::kernel kernel3(context, kernel_name);

  std::vector<uint32_t> instr_v = test_utils::load_instr_binary(vm["instr"].as<std::string>());
  if (instr_v.empty()) {
      throw std::runtime_error("Instruction file is empty or could not be read.");
  }
  std::vector<xrt::bo> bo_instr_list;
  for (int i = 0; i < 3; i++) {
      bo_instr_list.emplace_back(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel1.group_id(1));
      memcpy(bo_instr_list[i].map<void*>(), instr_v.data(), instr_v.size() * sizeof(int));
  }

  std::vector<A_DATATYPE> AVec(M*K);
  std::vector<B_DATATYPE> BVec(K*N);

  if (vm.count("a-file") && vm.count("b-file")) {
      std::cout << "Reading matrix A from: " << vm["a-file"].as<std::string>() << std::endl;
      std::cout << "Reading matrix B from: " << vm["b-file"].as<std::string>() << std::endl;
      std::ifstream f_in_a(vm["a-file"].as<std::string>(), std::ios::binary);
      std::ifstream f_in_b(vm["b-file"].as<std::string>(), std::ios::binary);
      if (!f_in_a || !f_in_b) {
          throw std::runtime_error("Could not open input matrix files.");
      }
      f_in_a.read(reinterpret_cast<char*>(AVec.data()), A_SIZE);
      f_in_b.read(reinterpret_cast<char*>(BVec.data()), B_SIZE);
  } else {
      std::cout << "Generating random matrices (no input files provided).\n";
      for (int i = 0; i < M*K; i++) AVec[i] = matmul_common::get_random<A_DATATYPE>();
      for (int i = 0; i < K*N; i++) BVec[i] = matmul_common::get_random<B_DATATYPE>();
  }

  std::vector<xrt::bo> bo_a_list, bo_b_list, bo_out_list, bo_tmp1_list, bo_trace_list;
  for (int i = 0; i < 3; i++) {
      bo_a_list.emplace_back(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(3));
      bo_b_list.emplace_back(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(4));
      bo_out_list.emplace_back(device, C_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(5));
      bo_tmp1_list.emplace_back(device, 1, XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(6));
      bo_trace_list.emplace_back(device, std::max(trace_size,1) * 4, XRT_BO_FLAGS_HOST_ONLY, kernel1.group_id(7));

      memcpy(bo_a_list[i].map<void*>(), AVec.data(), A_SIZE);
      memcpy(bo_b_list[i].map<void*>(), BVec.data(), B_SIZE);
  }

  unsigned int opcode = 3;

  double sum_copy_to = 0.0, sum_kernel = 0.0, sum_copy_from = 0.0, sum_total = 0.0;
  double sum_wall_clock = 0.0;
  double sum_parallel_kernel = 0.0;
  double sum_parallel_overhead = 0.0;
  double sum_cpu_time = 0.0;
  int total_iterations = n_iterations + n_warmup_iterations;

  std::cout << "Starting benchmark (" << n_warmup_iterations << " warmup + " << n_iterations << " iterations)...\n";
  for (int iter = 0; iter < total_iterations; iter++) {
    std::vector<TimingResult> results(3);

    auto start_wall = std::chrono::high_resolution_clock::now();
    std::thread t1([&]{ results[0] = run_matmul_once(kernel1, bo_instr_list[0], bo_a_list[0], bo_b_list[0], bo_out_list[0], bo_tmp1_list[0], bo_trace_list[0], opcode); });
    std::thread t2([&]{ results[1] = run_matmul_once(kernel2, bo_instr_list[1], bo_a_list[1], bo_b_list[1], bo_out_list[1], bo_tmp1_list[1], bo_trace_list[1], opcode); });
    std::thread t3([&]{ results[2] = run_matmul_once(kernel3, bo_instr_list[2], bo_a_list[2], bo_b_list[2], bo_out_list[2], bo_tmp1_list[2], bo_trace_list[2], opcode); });

    t1.join();
    t2.join();
    t3.join();
    auto end_wall = std::chrono::high_resolution_clock::now();

    for(int i=0; i<3; ++i) {
        if (!results[i].success) {
            throw std::runtime_error("A kernel execution failed in thread " + std::to_string(i+1) + ". Aborting benchmark.");
        }
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    std::vector<C_DATATYPE> C_golden_iter(M * N);
    matmul_golden(AVec, BVec, C_golden_iter, M, K, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();

    if (iter < n_warmup_iterations) {
        if (verbosity > 0) std::cout << "Warmup iteration " << iter << " completed.\n";
        continue;
    }

    double total_wall = std::chrono::duration_cast<std::chrono::microseconds>(end_wall - start_wall).count();
    sum_wall_clock += total_wall;

    auto first_kernel_start = std::min({results[0].kernel_start_time, results[1].kernel_start_time, results[2].kernel_start_time});
    auto last_kernel_end = std::max({results[0].kernel_end_time, results[1].kernel_end_time, results[2].kernel_end_time});
    double parallel_kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(last_kernel_end - first_kernel_start).count();
    sum_parallel_kernel += parallel_kernel_us;

    double parallel_overhead_us = total_wall - parallel_kernel_us;
    sum_parallel_overhead += parallel_overhead_us;
    
    double cpu_time_iter = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count();
    sum_cpu_time += cpu_time_iter;

    double copy_to_sum_iter = 0, kernel_sum_iter = 0, copy_from_sum_iter = 0, total_sum_iter = 0;
    for (int i = 0; i < 3; i++) {
        copy_to_sum_iter += results[i].copy_to_us;
        kernel_sum_iter += results[i].kernel_us;
        copy_from_sum_iter += results[i].copy_from_us;
        total_sum_iter += results[i].total_us;
    }

    sum_copy_to += copy_to_sum_iter / 3.0;
    sum_kernel += kernel_sum_iter / 3.0;
    sum_copy_from += copy_from_sum_iter / 3.0;
    sum_total += total_sum_iter / 3.0;

    int real_iter = iter - n_warmup_iterations;
    if (verbosity > 0) {
        std::cout << "Iteration " << real_iter << " timings (us):\n";
        for (int i = 0; i < 3; i++) {
            std::cout << "  Thread" << i+1 << ": copy_to=" << results[i].copy_to_us
                      << ", kernel=" << results[i].kernel_us
                      << ", copy_from=" << results[i].copy_from_us
                      << ", total=" << results[i].total_us << "\n";
        }
        std::cout << "  Parallel Overhead (copies, etc): " << parallel_overhead_us << " us\n";
        std::cout << "  Parallel kernel total:           " << parallel_kernel_us << " us\n";
        std::cout << "  Wall-clock total for 3 threads:  " << total_wall << " us\n";
        std::cout << "  CPU golden execution:            " << cpu_time_iter << " us\n";
    }
  }

  std::cout << "\n=== Average timings over " << n_iterations << " iterations (us) ===\n";
  std::cout << "--- NPU (per thread) ---\n";
  std::cout << "  Avg copy to device:   " << sum_copy_to / n_iterations << "\n";
  std::cout << "  Avg kernel execution: " << sum_kernel / n_iterations << "\n";
  std::cout << "  Avg copy from device: " << sum_copy_from / n_iterations << "\n";
  std::cout << "  Avg total time:       " << sum_total / n_iterations << "\n";
  std::cout << "--- NPU & CPU Total ---\n";
  std::cout << "  Avg NPU Parallel Overhead:           " << sum_parallel_overhead / n_iterations << "\n";
  std::cout << "  Avg NPU parallel kernel (3 threads): " << sum_parallel_kernel / n_iterations << "\n";
  std::cout << "  Avg NPU wall-clock (3 threads):      " << sum_wall_clock / n_iterations << "\n";
  std::cout << "  Avg CPU execution (1 thread):        " << sum_cpu_time / n_iterations << "\n";

  if (vm.count("c-file")) {
      // Only write the final result after all iterations are done
      if(vm.count("a-file")) { // Check if called from JAX
          std::cout << "Writing result matrix to: " << vm["c-file"].as<std::string>() << std::endl;
          std::vector<C_DATATYPE> C_npu(M * N);
          bo_out_list[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
          memcpy(C_npu.data(), bo_out_list[0].map<void*>(), C_SIZE);
          std::ofstream f_out(vm["c-file"].as<std::string>(), std::ios::binary);
          if (!f_out) {
              throw std::runtime_error("Could not open output matrix file for writing.");
          }
          f_out.write(reinterpret_cast<const char*>(C_npu.data()), C_SIZE);
      }
  }

  if (!skip_verify) {
      std::cout << "\n=== Verifying results... ===\n";

      std::vector<C_DATATYPE> C_npu(M * N);
      bo_out_list[0].sync(XCL_BO_SYNC_BO_FROM_DEVICE);
      memcpy(C_npu.data(), bo_out_list[0].map<void*>(), C_SIZE);

      std::vector<C_DATATYPE> C_golden(M * N);
      matmul_golden(AVec, BVec, C_golden, M, K, N);

      int errors = verify_results(C_npu, C_golden);
      if (errors == 0) {
          std::cout << ">>> TEST PASSED <<<\n";
      } else {
          std::cout << "!!! TEST FAILED with " << errors << " errors !!!\n";
          return 1;
      }
  } else {
      std::cout << "\n=== Verification skipped ===\n";
  }

  return 0;
}
