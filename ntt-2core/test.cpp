//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "test_utils.h"
#include "xrt/xrt_bo.h"

const int scaleFactor = 2;

namespace po = boost::program_options;

int64_t modPow(int64_t x, int64_t n, int64_t mod) {
    int64_t ret;
    if (n == 0) {
        ret = 1;
    } else if (n % 2 == 1) {
        ret = (x * modPow((x * x) % mod, n / 2, mod)) % mod;
    } else {
        ret = modPow((x * x) % mod, n / 2, mod);
    }
    return ret;
}

void make_roots(int32_t n, std::vector<int32_t> &roots, int64_t p, int64_t g){
  int64_t w = modPow(g, (p - 1) / n, p);
  for (int i = 1; i < n; i++) {
        roots[i] = (uint32_t)(((uint64_t)roots[i - 1] * w) % p);
  }
}

int main(int argc, const char *argv[]) {
  constexpr int64_t p = 3329;
  constexpr int64_t g = 3;
  constexpr int64_t n = 8;
  constexpr int64_t trace_size = 8192;
  int IN_VOLUME = 1 << n;
  int OUT_VOLUME = IN_VOLUME + trace_size;
  constexpr bool VERIFY = true;
  
  // Program arguments parsing
  po::options_description desc("Allowed options");
  po::variables_map vm;
  test_utils::add_default_options(desc);

  test_utils::parse_options(argc, argv, desc, vm);
  int verbosity = vm["verbosity"].as<int>();

  int IN_SIZE = IN_VOLUME * sizeof(int32_t);
  int OUT_SIZE = OUT_VOLUME * sizeof(int32_t) + trace_size;

  // Load instruction sequence
  std::vector<uint32_t> instr_v =
      test_utils::load_instr_sequence(vm["instr"].as<std::string>());

  std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT context and load the kernel
  xrt::device device;
  xrt::kernel kernel;

  test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                   vm["xclbin"].as<std::string>(),
                                   vm["kernel"].as<std::string>());

  // set up the buffer objects
  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
  auto bo_inA = xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_root = xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_prime = xrt::bo(device, 1 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_outC = xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  std::cout << "Writing data into buffer objects.\n";

  // Copy instruction stream to xrt buffer object
  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  // Initialize root
  std::vector<int32_t> root(IN_VOLUME);
  root[0] = 1;
  make_roots(IN_VOLUME, root, p, g);
  std::cout << "roots: ";
  for (int i = IN_VOLUME / 2; i < IN_VOLUME; i++){
      std::cout << root[i] << " ";
  }
  std::cout << std::endl;
  
  // Initialize buffer
  int32_t *bufInA = bo_inA.map<int32_t *>();
  int32_t *bufRoot = bo_root.map<int32_t *>();
  int32_t *bufInFactor = bo_prime.map<int32_t *>();
  int32_t *bufOut = bo_outC.map<int32_t *>();
  for (int i = 0; i < IN_VOLUME; i++){
    bufInA[i] = i;
    bufRoot[i] = root[i];
    bufOut[i] = 0;
  }
  *bufInFactor = (int32_t)scaleFactor;

  // sync host to device memories
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_root.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  // bo_prime.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // Execute the kernel and wait to finish
  std::cout << "Running Kernel.\n";
  auto start = std::chrono::high_resolution_clock::now();
  auto run = kernel(bo_instr, instr_v.size(), bo_inA, bo_root, bo_outC);
  ert_cmd_state r = run.wait();
  auto stop = std::chrono::high_resolution_clock::now();
  if (r != ERT_CMD_STATE_COMPLETED) {
      std::cout << "kernel did not complete. returned status: " << r << "\n";
      return 1;
  }

  // Sync device to host memories
  bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // Compare out to golden
  std::string filename = std::format("../../data/ans_q{}_n{}.txt", p, n);
  std::ifstream ansFile(filename);
  if (!ansFile) {
      std::cerr << "Error opening file" << std::endl;
      return 1;
  }
  std::vector<int32_t> answers;
  int ans;
  while (ansFile >> ans) {
      answers.push_back(ans);
  }

  int n_row = 2;
  int n_percore = IN_VOLUME / n_row;
  for (int32_t i = 0; i < n_row; i++){
    for (int j = 0; j < n_percore; j++){
      int idx = i * n_percore + j;
      answers[idx] = idx + i; 
    }
  }
  
  int errors = 0;
  std::cout << "Verifying results ..." << std::endl;
  
  for (int32_t i = 0; i < IN_VOLUME; i++) {
    int32_t ref = answers[i];
    int32_t test = bufOut[i];
    if (test != ref) {
      std::cout << "[" << i << "] Error " << test << " != " << ref << std::endl;
      errors++;
    } else {
      std::cout << "[" << i << "] Correct " << test << " == " << ref << std::endl;
    }
  }

  if (trace_size > 0) {
    test_utils::write_out_trace(((char *)bufOut) + IN_SIZE, trace_size,
                                vm["trace_file"].as<std::string>());
  }

  float npu_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
  std::cout << std::endl
            << "Avg NPU NTT time: " << npu_time << "us."
            << std::endl;

  // Print Pass/Fail result of our test
  if (!errors) {
    std::cout << std::endl << "PASS!" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << std::endl
              << errors << " mismatches." << std::endl
              << std::endl;
    std::cout << std::endl << "fail." << std::endl << std::endl;
    return 1;
  }
}
