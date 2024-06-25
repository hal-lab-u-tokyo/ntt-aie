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

#define VEC_NUM 8

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

aie::vector<int32_t, VEC_NUM> vector_modadd(aie::vector<int32_t, VEC_NUM> v0, aie::vector<int32_t, VEC_NUM> v1, aie::vector<int32_t, VEC_NUM> p){
  aie::vector<int32_t, VEC_NUM> v2 = aie::add(v0, v1);
  aie::mask<VEC_NUM> mask_v2_lt_p = aie::lt(v2, p);
  aie::vector<int32_t, VEC_NUM> over_v2 = aie::select(p, 0, mask_v2_lt_p);
  aie::vector<int32_t, VEC_NUM> modadd = aie::sub(v2, over_v2);
  return modadd;
}

aie::vector<int32_t, VEC_NUM> vector_modsub(aie::vector<int32_t, VEC_NUM> v0, aie::vector<int32_t, VEC_NUM> v1, aie::vector<int32_t, VEC_NUM> p){
  aie::vector<int32_t, VEC_NUM> v0_plus_p = aie::add(v0, p);
  aie::vector<int32_t, VEC_NUM> v3 = aie::sub(v0_plus_p, v1);
  aie::mask<VEC_NUM> mask_v3_lt_p = aie::lt(v3, p);
  aie::vector<int32_t, VEC_NUM> over_v3 = aie::select(p, 0, mask_v3_lt_p);
  aie::vector<int32_t, VEC_NUM> modsub = aie::sub(v3, over_v3);
  return modsub;
}

aie::vector<int32_t, VEC_NUM> vector_barrett(aie::vector<int32_t, VEC_NUM> v, aie::vector<int32_t, VEC_NUM> p_vec, aie::vector<int32_t, VEC_NUM> root_vec, aie::vector<int32_t, VEC_NUM> u_vec, int32_t w){
  aie::accum<acc64, VEC_NUM> t = aie::mul(v, root_vec);
  aie::vector<int32_t, VEC_NUM> x_1 = t.template to_vector<int32_t>(w - 2);
  aie::accum<acc64, VEC_NUM> x_2 = aie::mul(x_1, u_vec);
  aie::vector<int32_t, VEC_NUM> s = x_2.template to_vector<int32_t>(w + 2);
  aie::accum<acc64, VEC_NUM> r = aie::mul(s, p_vec);
  aie::vector<int32_t, VEC_NUM> tt = t.template to_vector<int32_t>(0);
  aie::vector<int32_t, VEC_NUM> rr = r.template to_vector<int32_t>(0);
  aie::vector<int32_t, VEC_NUM> c = aie::sub(tt, rr);
  aie::mask<VEC_NUM> mask_c_lt_p = aie::lt(c, p_vec);
  aie::vector<int32_t, VEC_NUM> over_c = aie::select(p_vec, 0, mask_c_lt_p);
  aie::vector<int32_t, VEC_NUM> barrett = aie::sub(c, over_c);
  return barrett;
}

void ntt_stage_parallel8(int32_t N, int32_t core_idx, int32_t *pA1, int32_t *root_in, int32_t bf_width, int32_t root_idx, int32_t F, int32_t p, int32_t w, int32_t u){
    for (int i = 0; i < F; i++){
        int32_t cycle = bf_width / VEC_NUM;
        int32_t idx_base = (i / cycle) * bf_width * 2 + (i % cycle) * VEC_NUM;
        int32_t *__restrict pA1_i = pA1 + idx_base;
        int32_t root = root_in[root_idx + (core_idx * F + i) / cycle];
        aie::vector<int32_t, VEC_NUM> v0 = aie::load_v<VEC_NUM>(pA1_i);
        aie::vector<int32_t, VEC_NUM> v1 = aie::load_v<VEC_NUM>(pA1_i + bf_width);
        aie::vector<int32_t, VEC_NUM> p_vector = aie::broadcast<int32_t, VEC_NUM>(p);
        aie::vector<int32_t, VEC_NUM> root_vector = aie::broadcast<int32_t, VEC_NUM>(root);
        aie::vector<int32_t, VEC_NUM> u_vector = aie::broadcast<int32_t, VEC_NUM>(u);
       
        // modadd(v0, v1, p)
        aie::vector<int32_t, VEC_NUM> modadd = vector_modadd(v0, v1, p_vector);

        // modsub(v0, v1, p)
        aie::vector<int32_t, VEC_NUM> modsub = vector_modsub(v0, v1, p_vector);
        
        // barrett_2k(modsub(v0, v1, p), root, p, w, u);
        aie::vector<int32_t, VEC_NUM> barrett = vector_barrett(modsub, p_vector, root_vector, u_vector, w);
            
        aie::store_v(pA1_i, modadd);
        aie::store_v(pA1_i + bf_width, barrett);
    }
}
  
void ntt_stageN_1_parallel8(int32_t N, int32_t *out0, int32_t *out1, int32_t *in_a0, int32_t *in_a1, int32_t *in_root, int32_t bf_width, int32_t root_idx, int32_t F, int32_t p, int32_t w, int32_t u){
  for (int i = 0; i < F; i++){
        int32_t cycle = bf_width / VEC_NUM;
        int32_t idx_base = (i / cycle) * bf_width * 2 + (i % cycle) * VEC_NUM;
        int32_t *__restrict pA1_i0 = in_a0 + idx_base;
        int32_t *__restrict pA1_i1 = in_a1 + idx_base;
        int32_t *__restrict pOut0_i = out0 + idx_base;
        int32_t *__restrict pOut1_i = out1 + idx_base;
        int32_t root = in_root[root_idx];
        aie::vector<int32_t, VEC_NUM> v0 = aie::load_v<VEC_NUM>(pA1_i0);
        aie::vector<int32_t, VEC_NUM> v1 = aie::load_v<VEC_NUM>(pA1_i1);
        aie::vector<int32_t, VEC_NUM> p_vector = aie::broadcast<int32_t, VEC_NUM>(p);
        aie::vector<int32_t, VEC_NUM> root_vector = aie::broadcast<int32_t, VEC_NUM>(root);
        aie::vector<int32_t, VEC_NUM> u_vector = aie::broadcast<int32_t, VEC_NUM>(u);
       
        // modadd(v0, v1, p)
        aie::vector<int32_t, VEC_NUM> modadd = vector_modadd(v0, v1, p_vector);

        // modsub(v0, v1, p)
        aie::vector<int32_t, VEC_NUM> modsub = vector_modsub(v0, v1, p_vector);
        
        // barrett_2k(modsub(v0, v1, p), root, p, w, u);
        aie::vector<int32_t, VEC_NUM> barrett = vector_barrett(modsub, p_vector, root_vector, u_vector, w);
            
        aie::store_v(pOut0_i, modadd);
        aie::store_v(pOut1_i, barrett);
    }
}

extern "C" {

void ntt_stage_N_1(int32_t N, int32_t *out0, int32_t *out1, int32_t *in_a0, int32_t *in_a1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
  // Stage N-1
  event0();
  const int N_half = N / 2;
  const int F = N_half / VEC_NUM;
  const int root_idx = 1;
  const int bf_width = N;
  ntt_stageN_1_parallel8(N, out0, out1, in_a0, in_a1, in_root, bf_width, root_idx, F, p, w, u);
  event1();
}

void ntt_stage_0_to_N_2(int32_t N_all, int32_t N, int32_t logN, int32_t core_idx, int32_t *a_in, int32_t *root_in, int32_t *c_out0, int32_t *c_out1, int32_t p, int32_t w, int32_t u) {
  const int N_half = N / 2;
  int32_t root_idx = N_all / 2;
  int32_t bf_width = 1;
  int32_t cycle = 1;
  int32_t *__restrict pA1 = a_in;
  int32_t *__restrict pRoot = root_in;
  
  // Stage 0
  event0();
  for (int k = 0; k < N_half; k++){
    int i = 2 * k;
    int j = i + 1;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    int32_t root = root_in[root_idx + core_idx * N_half / bf_width + k];
    a_in[i] = modadd(v0, v1, p);
    a_in[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  event1();

  // Stage 1
  event0();
  bf_width *= 2;
  root_idx /= 2;
  for (int k = 0; k < N_half; k++){
    int i = (k / bf_width) * 2 * bf_width + k % bf_width;
    int j = i + bf_width;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    int32_t root = root_in[root_idx + (core_idx * N_half + k) / bf_width];
    a_in[i] = modadd(v0, v1, p);
    a_in[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  event1();

  // Stage 2
  event0();
  bf_width *= 2;
  root_idx /= 2;
  for (int k = 0; k < N_half; k++){
    int i = (k / bf_width) * 2 * bf_width + k % bf_width;
    int j = i + bf_width;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    int32_t root = root_in[root_idx + (core_idx * N_half + k) / bf_width];
    a_in[i] = modadd(v0, v1, p);
    a_in[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  event1();

  // Stage 3 to Stage N-2
  event0();
  const int F = N_half / VEC_NUM;
  for (int stage = 3; stage < logN; stage++){
    bf_width *= 2;
    root_idx /= 2;
    ntt_stage_parallel8(N, core_idx, a_in, root_in, bf_width, root_idx, F, p, w, u);     
  }
  event1();

  // Write back
  for (int i = 0; i < N_half; i++){
    c_out0[i] = a_in[i];  
    c_out1[i] = a_in[i + N_half];
  }
}

} // extern "C"
