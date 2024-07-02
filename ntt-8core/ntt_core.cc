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

aie::vector<int32_t, VEC_NUM> vector_modadd(aie::vector<int32_t, VEC_NUM> &v0, aie::vector<int32_t, VEC_NUM> &v1, aie::vector<int32_t, VEC_NUM> &p){
  aie::vector<int32_t, VEC_NUM> v2 = aie::add(v0, v1);
  aie::mask<VEC_NUM> mask_v2_lt_p = aie::lt(v2, p);
  aie::vector<int32_t, VEC_NUM> over_v2 = aie::select(p, 0, mask_v2_lt_p);
  aie::vector<int32_t, VEC_NUM> modadd = aie::sub(v2, over_v2);
  return modadd;
}

aie::vector<int32_t, VEC_NUM> vector_modsub(aie::vector<int32_t, VEC_NUM> &v0, aie::vector<int32_t, VEC_NUM> &v1, aie::vector<int32_t, VEC_NUM> &p){
  aie::vector<int32_t, VEC_NUM> v0_plus_p = aie::add(v0, p);
  aie::vector<int32_t, VEC_NUM> v3 = aie::sub(v0_plus_p, v1);
  aie::mask<VEC_NUM> mask_v3_lt_p = aie::lt(v3, p);
  aie::vector<int32_t, VEC_NUM> over_v3 = aie::select(p, 0, mask_v3_lt_p);
  aie::vector<int32_t, VEC_NUM> modsub = aie::sub(v3, over_v3);
  return modsub;
}

aie::vector<int32_t, VEC_NUM> vector_barrett(aie::vector<int32_t, VEC_NUM> &v, aie::vector<int32_t, VEC_NUM> &p_vec, aie::vector<int32_t, VEC_NUM> &root_vec, aie::vector<int32_t, VEC_NUM> &u_vec, int32_t w){
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

void ntt_stage_parallel8_internal(int32_t *out_ptr0, int32_t *out_ptr1, int32_t *in_ptr0, int32_t *in_ptr1, aie::vector<int32_t, VEC_NUM> &root_vector, aie::vector<int32_t, VEC_NUM> &p_vector, int32_t w, aie::vector<int32_t, VEC_NUM> &u_vector){
    aie::vector<int32_t, VEC_NUM> v0 = aie::load_v<VEC_NUM>(in_ptr0);
    aie::vector<int32_t, VEC_NUM> v1 = aie::load_v<VEC_NUM>(in_ptr1);
    
    // modadd(v0, v1, p)
    aie::vector<int32_t, VEC_NUM> modadd = vector_modadd(v0, v1, p_vector);

    // modsub(v0, v1, p)
    aie::vector<int32_t, VEC_NUM> modsub = vector_modsub(v0, v1, p_vector);
    
    // barrett_2k(modsub(v0, v1, p), root, p, w, u);
    aie::vector<int32_t, VEC_NUM> barrett = vector_barrett(modsub, p_vector, root_vector, u_vector, w);
        
    aie::store_v(out_ptr0, modadd);
    aie::store_v(out_ptr1, barrett);
}

extern "C" {

void ntt_1stage(int32_t idx_stage, int32_t N, int32_t core_idx, int32_t n_core, int32_t *out0, int32_t *out1, int32_t *in0, int32_t *in1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
  // Stage N - 1 - idx_stage
  // idx_stage : 0, 1, 2, ...
  event0();
  const int N_half = N / 2;
  const int F = N_half / VEC_NUM;
  const int root_base = 1 << idx_stage;
  const int root_num = 1 << idx_stage;
  const int root_idx = root_base + core_idx / (n_core / root_num);
  int32_t root = in_root[root_idx];
  aie::vector<int32_t, VEC_NUM> root_vector = aie::broadcast<int32_t, VEC_NUM>(root);
  aie::vector<int32_t, VEC_NUM> p_vector = aie::broadcast<int32_t, VEC_NUM>(p);
  aie::vector<int32_t, VEC_NUM> u_vector = aie::broadcast<int32_t, VEC_NUM>(u);
  for (int i = 0; i < F; i++){
    int32_t idx_base = i * VEC_NUM;
    int32_t *__restrict pIn0_i = in0 + idx_base;
    int32_t *__restrict pIn1_i = in1 + idx_base;
    int32_t *__restrict pOut0_i = out0 + idx_base;
    int32_t *__restrict pOut1_i = out1 + idx_base;
    ntt_stage_parallel8_internal(pOut0_i, pOut1_i, pIn0_i, pIn1_i, root_vector, p_vector, w, u_vector);
  }
  event1();
}

void ntt_stage_0_to_logN(int32_t N_all, int32_t N, int32_t logN, int32_t core_idx, int32_t *in_a, int32_t *root_in, int32_t *out0, int32_t *out1, int32_t p, int32_t w, int32_t u) {
  const int N_half = N / 2;
  int32_t root_idx = N_all / 2;
  int32_t bf_width = 1;
  int32_t cycle = 1;
  int32_t *__restrict pRoot = root_in;
  
  // Stage 0
  event0();
  for (int k = 0; k < N_half; k++){
    int i = 2 * k;
    int j = i + 1;
    int32_t v0 = in_a[i];
    int32_t v1 = in_a[j];
    int32_t root = root_in[root_idx + core_idx * N_half / bf_width + k];
    in_a[i] = modadd(v0, v1, p);
    in_a[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  event1();

  // Stage 1
  event0();
  bf_width *= 2;
  root_idx /= 2;
  for (int k = 0; k < N_half; k++){
    int i = (k / bf_width) * 2 * bf_width + k % bf_width;
    int j = i + bf_width;
    int32_t v0 = in_a[i];
    int32_t v1 = in_a[j];
    int32_t root = root_in[root_idx + (core_idx * N_half + k) / bf_width];
    in_a[i] = modadd(v0, v1, p);
    in_a[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  event1();

  // Stage 2
  event0();
  bf_width *= 2;
  root_idx /= 2;
  for (int k = 0; k < N_half; k++){
    int i = (k / bf_width) * 2 * bf_width + k % bf_width;
    int j = i + bf_width;
    int32_t v0 = in_a[i];
    int32_t v1 = in_a[j];
    int32_t root = root_in[root_idx + (core_idx * N_half + k) / bf_width];
    in_a[i] = modadd(v0, v1, p);
    in_a[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  event1();

  // Stage 3 to Stage N-3
  event0();
  const int F = N_half / VEC_NUM;
  aie::vector<int32_t, VEC_NUM> p_vector = aie::broadcast<int32_t, VEC_NUM>(p);
  aie::vector<int32_t, VEC_NUM> u_vector = aie::broadcast<int32_t, VEC_NUM>(u);
  for (int stage = 3; stage < logN; stage++){
    bf_width *= 2;
    root_idx /= 2;
    for (int i = 0; i < F; i++){
      int32_t cycle = bf_width / VEC_NUM;
      int32_t idx_base = (i / cycle) * bf_width * 2 + (i % cycle) * VEC_NUM;
      int32_t *__restrict pA_i = in_a + idx_base;
      int32_t root = root_in[root_idx + (core_idx * F + i) / cycle];
      aie::vector<int32_t, VEC_NUM> root_vector = aie::broadcast<int32_t, VEC_NUM>(root);
      int32_t *__restrict pIn0_i = pA_i;
      int32_t *__restrict pIn1_i = pA_i + bf_width;
      int32_t *__restrict pOut0_i = pIn0_i;
      int32_t *__restrict pOut1_i = pIn1_i;
      ntt_stage_parallel8_internal(pIn0_i, pIn1_i, pOut0_i, pOut1_i, root_vector, p_vector, w, u_vector);
    }
  }
  event1();

  // Write back
  for (int i = 0; i < N_half; i++){
    out0[i] = in_a[i];  
    out1[i] = in_a[i + N_half];
  }
}

} // extern "C"
