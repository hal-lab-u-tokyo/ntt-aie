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

void ntt_stage_parallel4(int32_t *pA1, int32_t *root_in, int32_t bf_width, int32_t root_idx, int32_t F, int32_t p, int32_t w, int32_t u){
    const int32_t vec_prime = 4;
    for (int i = 0; i < F; i++){
        int32_t cycle = bf_width / vec_prime;
        int32_t idx_base = (i / cycle) * bf_width * 2 + (i % cycle) * vec_prime;
        int32_t *__restrict pA1_i = pA1 + idx_base;
        int32_t root = root_in[root_idx + i / cycle];
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

void ntt_stage_parallel8(int32_t *pA1, int32_t *root_in, int32_t bf_width, int32_t root_idx, int32_t F, int32_t p, int32_t w, int32_t u){
    for (int i = 0; i < F; i++){
        int32_t cycle = bf_width / VEC_NUM;
        int32_t idx_base = (i / cycle) * bf_width * 2 + (i % cycle) * VEC_NUM;
        int32_t *__restrict pA1_i = pA1 + idx_base;
        int32_t root = root_in[root_idx + i / cycle];
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

extern "C" {

void ntt_stage0_to_Nminus5(int32_t *a_in, int32_t *root_in, int32_t *c_out, int32_t N, int32_t logN, int32_t p, int32_t w, int32_t u) {
  const int N_half = N / 2;
  int32_t root_idx = N_half;
  int32_t bf_width = 1;
  int32_t vec_prime = 1;
  int32_t F = N_half / vec_prime;
  int32_t *__restrict pA1 = a_in;
  int32_t *__restrict pC = c_out;
  int32_t *__restrict pRoot = root_in;

  // Mask vector on scalar
  for (int i = 0; i < N / 2; i++){
    c_out[i * 2] = 0;
    c_out[i * 2 + 1] = 1;
  }

  // Stage 0 on Vector
  event0();
  F = N_half / VEC_NUM;
  aie::vector<int32_t, VEC_NUM> p_vector = aie::broadcast<int32_t, VEC_NUM>(p);     
  aie::vector<int32_t, VEC_NUM> u_vector = aie::broadcast<int32_t, VEC_NUM>(u);     
  for (int i = 0; i < F; i++){
    // v0
    // v1_left = v0 << 1
    // v1_right = v0 >> 1
    int32_t *__restrict pA1_i = pA1 + i * VEC_NUM;
    int32_t *__restrict pC_i = pC + i * VEC_NUM;
    int32_t *__restrict pRoot_i = pRoot + i * VEC_NUM;
    aie::vector<int32_t, VEC_NUM> v0 = aie::load_v<VEC_NUM>(pA1_i);
    aie::vector<int32_t, VEC_NUM> v1_left = aie::shuffle_down(v0, 1);
    aie::vector<int32_t, VEC_NUM> v1_right = aie::shuffle_up(v0, 1);
    aie::vector<int32_t, VEC_NUM> mask = aie::load_v<VEC_NUM>(pC_i);

    // vadd = modadd(v0, v1_left, p)
    // select idx %2 == 0 from vadd
    aie::vector<int32_t, VEC_NUM> vadd = vector_modadd(v0, v1_left, p_vector);
    aie::mask<VEC_NUM> mask_select_even = aie::eq(mask, 1);
    aie::vector<int32_t, VEC_NUM> vadd_even = aie::select(vadd, 0, mask_select_even);

    // vsub = modsub(v0, v1_right, p)
    // vsub_root = barrett_2k(modsub(v0, v1, p), root, p, w, u);
    // select idx %2 == 1 from vsub_root
    aie::vector<int32_t, VEC_NUM> vsub = vector_modsub(v0, v1_right, p_vector);
    aie::vector<int32_t, VEC_NUM> root_vector = aie::load_v<VEC_NUM>(pRoot_i);
    aie::vector<int32_t, VEC_NUM> barrett = vector_barrett(vsub, p_vector, root_vector, u_vector, w);
    aie::mask<VEC_NUM> mask_select_odd = aie::eq(mask, 0);
    aie::vector<int32_t, VEC_NUM> barrett_odd = aie::select(barrett, 0, mask_select_odd);

    // store
    aie::vector<int32_t, VEC_NUM> ret = aie::add(vadd_even, barrett_odd);
    aie::store_v(pA1_i, ret);
  }
  /*
  for (int k = 0; k < N_half; k++){
    int i = 2 * k;
    int j = i + 1;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    int32_t root = root_in[root_idx + k];
    a_in[i] = modadd(v0, v1, p);
    a_in[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  */
  event1();

  // Stage 1
  event0();
  bf_width *= 2;
  root_idx /= 2;
  for (int k = 0; k < N_half; k++){
    int i = (k / 2) * 4 + k % 2;
    int j = i + 2;
    int32_t v0 = a_in[i];
    int32_t v1 = a_in[j];
    int32_t root = root_in[root_idx + k / 2];
    a_in[i] = modadd(v0, v1, p);
    a_in[j] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
  }
  event1();

  // Stage 2
  event0();
  bf_width *= 2;
  root_idx /= 2;
  vec_prime = 4;
  F = N_half / vec_prime;
  ntt_stage_parallel4(a_in, root_in, bf_width, root_idx, F, p, w, u); 
  event1();

  // Stage 3 to Stage N-1
  event0();
  vec_prime = 8;
  F = N_half / vec_prime;
  for (int stage = 3; stage < logN; stage++){
    bf_width *= 2;
    root_idx /= 2;
    ntt_stage_parallel8(a_in, root_in, bf_width, root_idx, F, p, w, u);     
  }
  event1();
  
  // Write back
  for (int i = 0; i < N; i++){
    c_out[i] = a_in[i];  
  }
}

} // extern "C"
