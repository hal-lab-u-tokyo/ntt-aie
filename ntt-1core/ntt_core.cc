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

int32_t modadd(int32_t a, int32_t b, int32_t q){
  int ret = a + b;
  if (ret >= q){
    return ret - q;
  }
  return ret;
}

int32_t modsub(int32_t a, int32_t b, int32_t q){
  int ret = a + q - b;
  if (ret >= q){
    return ret - q;
  }
  return ret;
}

int32_t barrett_2k(int32_t a, int32_t b, int32_t q, int32_t w, int32_t u){
	int64_t t = (int64_t)a *(int64_t) b;
	int64_t x_1 = t >> (w - 2);
	int64_t x_2 = u * x_1;
	int64_t s = x_2 >> (w + 2);
	int64_t r = s * q;
	int64_t c = t - r;
	if (c >= q) {
		return c - q;
	}else {
		return c;
	}
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

void ntt_stage0_to_Nminus5(int32_t *a_in, int32_t *root_in, int32_t *c_out, int32_t N, int32_t logN, int32_t p, int32_t w, int32_t u) {
  const int N_half = N / 2;
  int root_idx = N_half;

  // Stage 0
  for (int k = 0; k < N_half; k++){
    int i = 2 * k;
    int j = i + 1;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    int32_t root = root_in[root_idx + k];
    a_in[i] = modadd(v0, v1, p);
    a_in[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  root_idx /= 2;
  // Stage 1
  for (int k = 0; k < N_half; k++){
    int i = (k / 2) * 4 + k % 2;
    int j = i + 2;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    int32_t root = root_in[root_idx + k / 2];
    a_in[i] = modadd(v0, v1, p);
    a_in[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  root_idx /= 2;

  // Stage 2
  for (int k = 0; k < N_half; k++){
    int i = (k / 4) * 8 + k % 4;
    int j = i + 4;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    int32_t root = root_in[root_idx + k / 4];
    a_in[i] = modadd(v0, v1, p);
    a_in[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  root_idx /= 2;
  
  // Stage 3 to Stage N-1
  constexpr int vec_prime = 8;
  const int F = N_half / vec_prime;
  int bf_width = 8;
  int32_t *__restrict pA1 = a_in;
    
  for (int stage = 0; stage < logN - 3; stage++){
    for (int i = 0; i < F; i++){
        int32_t cycle = bf_width / vec_prime;
        int32_t *__restrict pA1_i = pA1 + (i / cycle) * bf_width * 2 + (i % cycle) * vec_prime;
        int32_t root = root_in[root_idx + i];
        aie::vector<int32_t, vec_prime> v0 = aie::load_v<vec_prime>(pA1_i);
        aie::vector<int32_t, vec_prime> v1 = aie::load_v<vec_prime>(pA1_i + bf_width);
        aie::vector<int32_t, vec_prime> p_vector = aie::broadcast<int32_t, vec_prime>(p);
        aie::vector<int32_t, vec_prime> root_vector = aie::broadcast<int32_t, vec_prime>(root);
        aie::vector<int32_t, vec_prime> u_vector = aie::broadcast<int32_t, vec_prime>(u);

        // modadd(v0, v1, p)
        aie::vector<int32_t, vec_prime> v2 = aie::add(v0, v1);
        aie::mask<vec_prime> mask_v2_lt_p = aie::lt(v2, p_vector);
        aie::vector<int32_t, vec_prime> over_v2 = aie::select(p, 0, mask_v2_lt_p);
        aie::vector<int32_t, vec_prime> modadd = aie::sub(v2, over_v2);

        // modsub(v0, v1, p)
        aie::vector<int32_t, vec_prime> v0_plus_p = aie::add(v0, p_vector);
        aie::vector<int32_t, vec_prime> v3 = aie::sub(v0_plus_p, v1);
        aie::mask<vec_prime> mask_v3_lt_p = aie::lt(v3, p_vector);
        aie::vector<int32_t, vec_prime> over_v3 = aie::select(p, 0, mask_v3_lt_p);
        aie::vector<int32_t, vec_prime> modsub = aie::sub(v3, over_v3);
        
        // barrett_2k(modsub(v0, v1, p), root, p, w, u);
        aie::accum<acc64, vec_prime> t = aie::mul(modsub, root_vector);
        aie::vector<int32_t, vec_prime> x_1 = t.template to_vector<int32_t>(w - 2);
        aie::accum<acc64, vec_prime> x_2 = aie::mul(x_1, u_vector);
        aie::vector<int32_t, vec_prime> s = x_2.template to_vector<int32_t>(w + 2);
        aie::accum<acc64, vec_prime> r = aie::mul(s, p_vector);
        aie::vector<int32_t, vec_prime> tt = t.template to_vector<int32_t>(0);
        aie::vector<int32_t, vec_prime> rr = r.template to_vector<int32_t>(0);
        aie::vector<int32_t, vec_prime> c = aie::sub(tt, rr);
        aie::mask<vec_prime> mask_c_lt_p = aie::lt(c, p_vector);
        aie::vector<int32_t, vec_prime> over_c = aie::select(p, 0, mask_c_lt_p);
        aie::vector<int32_t, vec_prime> barrett = aie::sub(c, over_c);
    
        aie::store_v(pA1_i, modadd);
        aie::store_v(pA1_i + bf_width, barrett);
    }
    bf_width *= 2;
    root_idx /= 2;
  }
  
  for (int i = 0; i < N; i++){
    c_out[i] = a_in[i];
  }
}

} // extern "C"
