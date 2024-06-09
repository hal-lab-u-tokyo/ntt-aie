//===- vector_scaler_mul.cc -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

// Vectorized scale template (general case)
// Assume N is multiple of 16
template <typename T>
void scale_vectorized(T *a, T *c, int32_t prime, const int32_t N) {
  event0();
  constexpr int vec_prime = 32;
  T *__restrict pA1 = a;
  T *__restrict pC1 = c;
  const int F = N / vec_prime;
  T fac = prime;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<T, vec_prime> A0 = aie::load_v<vec_prime>(pA1);
      pA1 += vec_prime;
      aie::accum<acc32, vec_prime> cout = aie::mul(A0, fac);
      aie::store_v(pC1, cout.template to_vector<T>(0));
      pC1 += vec_prime;
    }
  event1();
}


// Vectorized scale template (int32_t case, acc64 used)
// Assume N is multiple of 16
template <>
void scale_vectorized<int32_t>(int32_t *a, int32_t *c, int32_t prime,
                               const int32_t N) {
  event0();
  constexpr int vec_prime = 32;
  int32_t *__restrict pA1 = a;
  int32_t *__restrict pC1 = c;
  const int F = N / vec_prime;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<int32_t, vec_prime> A0 = aie::load_v<vec_prime>(pA1);
      pA1 += vec_prime;
      aie::accum<acc64, vec_prime> cout = aie::mul(A0, prime);
      aie::store_v(pC1, cout.template to_vector<int32_t>(0));
      pC1 += vec_prime;
    }
  event1();
}

extern "C" {

void vector_scalar_mul_scalar(int32_t *a, int32_t *c, int32_t *prime, int32_t N) {
  for (int i = 0; i < N; i++) {
    c[i] = *prime * a[i];
  }
}

void vector_scalar_mul_vectorized_int32(int32_t *a_in, int32_t *c_out, int32_t *prime, int32_t N) {
  scale_vectorized<int32_t>(a_in, c_out, *prime, N);
}

void ntt_stage0_to_Nminus5(int32_t *a_in, int32_t *c_out, int32_t *prime, int32_t N) {
  const int N_half = N / 2;

  // Stage 0
  for (int k = 0; k < N_half; k++){
    int i = 2 * k;
    int j = i + 1;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    //a_in[i] = v1;
    //a_in[j] = v0;
  }

  // Stage 1
  for (int k = 0; k < N_half; k++){
    int i = (k / 2) * 4 + k % 2;
    int j = i + 2;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    //a_in[i] = v1;
    //a_in[j] = v0; 
  }

  // Stage 3
  for (int k = 0; k < N_half; k++){
    int i = (k / 4) * 8 + k % 4;
    int j = i + 4;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    //a_in[i] = v1;
    //a_in[j] = v0; 
  }

  // After Stage 4
  constexpr int vec_prime = 8;
  int32_t *__restrict pA1 = a_in;
  const int F = N_half / vec_prime;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {
      aie::vector<int32_t, vec_prime> v0 = aie::load_v<vec_prime>(pA1);
      pA1 += vec_prime;
      aie::vector<int32_t, vec_prime> v1 = aie::load_v<vec_prime>(pA1);
      aie::vector<int32_t, vec_prime> cout = aie::add(v0, v1);
      pA1 -= vec_prime;
      aie::store_v(pA1, cout);
      pA1 += vec_prime;
      aie::store_v(pA1, cout);
      pA1 += vec_prime;
  }

  for (int i = 0; i < N; i++){
    c_out[i] = a_in[i];
  }
}

} // extern "C"
