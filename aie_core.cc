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
    
const int32_t vec_prime = 16;
const int32_t vec_prime_half = vec_prime / 2;

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

aie::vector<int32_t, vec_prime> vector_modadd(aie::vector<int32_t, vec_prime> &v0, aie::vector<int32_t, vec_prime> &v1, aie::vector<int32_t, vec_prime> &p_vector){
  aie::vector<int32_t, vec_prime> v2 = aie::add(v0, v1);
  aie::mask<vec_prime> mask_v2_lt_p = aie::lt(v2, p_vector);
  aie::vector<int32_t, vec_prime> over_v2 = aie::select(p_vector, 0, mask_v2_lt_p);
  aie::vector<int32_t, vec_prime> modadd = aie::sub(v2, over_v2);
  return modadd;
}

aie::vector<int32_t, vec_prime> vector_modsub(aie::vector<int32_t, vec_prime> &v0, aie::vector<int32_t, vec_prime> &v1, aie::vector<int32_t, vec_prime> &p_vector){
  aie::vector<int32_t, vec_prime> v0_plus_p = aie::add(v0, p_vector);
  aie::vector<int32_t, vec_prime> v3 = aie::sub(v0_plus_p, v1);
  aie::mask<vec_prime> mask_v3_lt_p = aie::lt(v3, p_vector);
  aie::vector<int32_t, vec_prime> over_v3 = aie::select(p_vector, 0, mask_v3_lt_p);
  aie::vector<int32_t, vec_prime> modsub = aie::sub(v3, over_v3);
  return modsub;
}

aie::vector<int32_t, vec_prime> vector_barrett(aie::vector<int32_t, vec_prime> &v, aie::vector<int32_t, vec_prime> &p_vec, aie::vector<int32_t, vec_prime> &root_vec, aie::vector<int32_t, vec_prime> &u_vec, int32_t w){
  aie::accum<acc64, vec_prime> t = aie::mul(v, root_vec);
  aie::vector<int32_t, vec_prime> x_1 = t.template to_vector<int32_t>(w - 2);
  aie::accum<acc64, vec_prime> x_2 = aie::mul(x_1, u_vec);
  aie::vector<int32_t, vec_prime> s = x_2.template to_vector<int32_t>(w + 2);
  aie::accum<acc64, vec_prime> r = aie::mul(s, p_vec);
  aie::vector<int32_t, vec_prime> tt = t.template to_vector<int32_t>(0);
  aie::vector<int32_t, vec_prime> rr = r.template to_vector<int32_t>(0);
  aie::vector<int32_t, vec_prime> c = aie::sub(tt, rr);
  aie::mask<vec_prime> mask_c_lt_p = aie::lt(c, p_vec);
  aie::vector<int32_t, vec_prime> over_c = aie::select(p_vec, 0, mask_c_lt_p);
  aie::vector<int32_t, vec_prime> barrett = aie::sub(c, over_c);
  return barrett;
}

aie::vector<int32_t, vec_prime_half> vector_barrett(aie::vector<int32_t, vec_prime_half> &v, aie::vector<int32_t, vec_prime_half> &p_vec, aie::vector<int32_t, vec_prime_half> &root_vec, aie::vector<int32_t, vec_prime_half> &u_vec, int32_t w){
  aie::accum<acc64, vec_prime_half> t = aie::mul(v, root_vec);
  aie::vector<int32_t, vec_prime_half> x_1 = t.template to_vector<int32_t>(w - 2);
  aie::accum<acc64, vec_prime_half> x_2 = aie::mul(x_1, u_vec);
  aie::vector<int32_t, vec_prime_half> s = x_2.template to_vector<int32_t>(w + 2);
  aie::accum<acc64, vec_prime_half> r = aie::mul(s, p_vec);
  aie::vector<int32_t, vec_prime_half> tt = t.template to_vector<int32_t>(0);
  aie::vector<int32_t, vec_prime_half> rr = r.template to_vector<int32_t>(0);
  aie::vector<int32_t, vec_prime_half> c = aie::sub(tt, rr);
  aie::mask<vec_prime_half> mask_c_lt_p = aie::lt(c, p_vec);
  aie::vector<int32_t, vec_prime_half> over_c = aie::select(p_vec, 0, mask_c_lt_p);
  aie::vector<int32_t, vec_prime_half> barrett = aie::sub(c, over_c);
  return barrett;
}

void ntt_stage_parallel8(int32_t *pOut_i0,
                          int32_t *pOut_i1,
                          int32_t *pIn_i0,
                          int32_t *pIn_i1,
                          aie::vector<int32_t, vec_prime> &p_vector,
                          aie::vector<int32_t, vec_prime> &root_vector,
                          aie::vector<int32_t, vec_prime> &u_vector,
                          int32_t p,
                          int32_t w){
    aie::vector<int32_t, vec_prime> v0 = aie::load_v(pIn_i0);
    aie::vector<int32_t, vec_prime> v1 = aie::load_v(pIn_i1);
    
    // modadd(v0, v1, p)
    aie::vector<int32_t, vec_prime> modadd = vector_modadd(v0, v1, p_vector);

    // modsub(v0, v1, p)
    aie::vector<int32_t, vec_prime> modsub = vector_modsub(v0, v1, p_vector);
    
    // barrett_2k(modsub(v0, v1, p), root, p, w, u);
    aie::vector<int32_t, vec_prime> barrett = vector_barrett(modsub, p_vector, root_vector, u_vector, w);

    aie::store_v(pOut_i0, modadd);
    aie::store_v(pOut_i1, barrett);
}

extern "C" {

void swap_buff(int32_t *a, int32_t *b, int32_t N) {
  const int F = N / vec_prime;
  for (int i = 0; i < F; i++){
    int32_t *__restrict pa_i = a + i * vec_prime;
    int32_t *__restrict pb_i = b + i * vec_prime;
    aie::vector<int32_t, vec_prime> va_i = aie::load_v(pa_i);
    aie::vector<int32_t, vec_prime> vb_i = aie::load_v(pb_i);
    aie::store_v(pa_i, vb_i);
    aie::store_v(pb_i, va_i);
  }
}

void ntt_1stage(int32_t idx_stage, int32_t N, int32_t core_idx, int32_t n_core, int32_t *out0, int32_t *out1, int32_t *in0, int32_t *in1, int32_t *in_root, int32_t p, int32_t w, int32_t u) {
  // Stage N - 1 - idx_stage
  // idx_stage : 0, 1, 2, ...
  event0();
  const int N_half = N / 2;
  const int F = N_half / vec_prime;
  const int root_base = 1 << idx_stage;
  const int root_num = 1 << idx_stage;
  const int root_idx = root_base + core_idx / (n_core / root_num);
  int32_t root = in_root[root_idx];
  aie::vector<int32_t, vec_prime> root_vector = aie::broadcast<int32_t, vec_prime>(root);
  aie::vector<int32_t, vec_prime> p_vector = aie::broadcast<int32_t, vec_prime>(p);
  aie::vector<int32_t, vec_prime> u_vector = aie::broadcast<int32_t, vec_prime>(u);
  for (int i = 0; i < F; i++){
    int32_t idx_base = i * vec_prime;
    int32_t *__restrict pIn0_i = in0 + idx_base;
    int32_t *__restrict pIn1_i = in1 + idx_base;
    int32_t *__restrict pOut0_i = out0 + idx_base;
    int32_t *__restrict pOut1_i = out1 + idx_base;
    ntt_stage_parallel8(pOut0_i, pOut1_i, pIn0_i, pIn1_i, p_vector, root_vector, u_vector, p, w);
  }
  event1();
}

void ntt_stage0_to_Nminus5(int32_t *a_in, int32_t *root_in, int32_t *c_out0, int32_t *c_out1, int32_t N, int32_t logN, int32_t N_all, int32_t core_idx, int32_t p, int32_t w, int32_t u) {
  const int N_half = N / 2;
  const int32_t F = N_half / vec_prime;
  int32_t root_idx = N_all / 2;
  int32_t bf_width = 1;
  aie::vector<int32_t, vec_prime> p_vector = aie::broadcast<int32_t, vec_prime>(p);
  aie::vector<int32_t, vec_prime> u_vector = aie::broadcast<int32_t, vec_prime>(u);
  aie::vector<int32_t, vec_prime_half> p_vector_half = aie::broadcast<int32_t, vec_prime_half>(p);
  aie::vector<int32_t, vec_prime_half> u_vector_half = aie::broadcast<int32_t, vec_prime_half>(u);
  aie::vector<int32_t, vec_prime> zero_vector = aie::zeros<int32_t, vec_prime>();
  aie::vector<int32_t, vec_prime_half> zero_vector_half = aie::zeros<int32_t, vec_prime_half>();
  aie::vector<int32_t, 4> zero_vector4 = aie::zeros<int32_t, 4>();

  // Stage 0
  event0();
  event1();
  event0();
  for (int i = 0; i < N / vec_prime; i++){
    int32_t *__restrict pA_i = a_in + i * vec_prime;
    aie::vector<int32_t, vec_prime> v0 = aie::load_v<vec_prime>(pA_i);
    aie::vector<int32_t, vec_prime> v0_l = aie::shuffle_down(v0, 1);
    v0_l = vector_modadd(v0, v0_l, p_vector);
    aie::vector<int32_t, vec_prime_half> v0_l_half = aie::filter_even(v0_l);
    v0_l = aie::concat(v0_l_half, zero_vector_half);
    
    aie::vector<int32_t, vec_prime> v0_r = aie::shuffle_up(v0, 1);
    v0_r = vector_modsub(v0_r, v0, p_vector);
    aie::vector<int32_t, vec_prime_half> v0_r_half = aie::filter_odd(v0_r);
    int32_t *__restrict pRoot_i = root_in + root_idx + core_idx * N_half / bf_width + i * vec_prime/2;
    aie::vector<int32_t, vec_prime_half> root_vector_half = aie::load_v<vec_prime_half>(pRoot_i);
    v0_r_half = vector_barrett(v0_r_half, p_vector_half, root_vector_half, u_vector_half, w);
    v0_r = aie::concat(v0_r_half, zero_vector_half);

    auto [res, res2] = aie::interleave_zip(v0_l, v0_r, 1);
    aie::store_v(pA_i, res);
  }

  // Stage 1
  bf_width *= 2;
  root_idx /= 2;
  for (int i = 0; i < N / vec_prime; i++){
    int32_t *__restrict pA_i = a_in + i * vec_prime;
    aie::vector<int32_t, vec_prime> v0 = aie::load_v<vec_prime>(pA_i);
    aie::vector<int32_t, vec_prime> v0_l = aie::shuffle_down(v0, 2);
    v0_l = vector_modadd(v0, v0_l, p_vector);
    auto [v0_l0, v0_l1] = aie::interleave_unzip(v0_l, zero_vector, 2);
    
    aie::vector<int32_t, vec_prime> v0_r = aie::shuffle_up(v0, 2);
    v0_r = vector_modsub(v0_r, v0, p_vector);
    int32_t *__restrict pRoot_i = root_in + root_idx + core_idx * N_half / bf_width + i * vec_prime/4;
    // Case vec_prime = 16
    aie::vector<int32_t, 4> root0_vector_half = aie::broadcast<int32_t, 4>(pRoot_i[0]);
    aie::vector<int32_t, 4> root1_vector_half = aie::broadcast<int32_t, 4>(pRoot_i[1]);
    aie::vector<int32_t, 4> root2_vector_half = aie::broadcast<int32_t, 4>(pRoot_i[2]);
    aie::vector<int32_t, 4> root3_vector_half = aie::broadcast<int32_t, 4>(pRoot_i[3]);
    root0_vector_half = aie::shuffle_up_fill(root0_vector_half, zero_vector4, 2);
    root1_vector_half = aie::shuffle_up_fill(root1_vector_half, zero_vector4, 2);
    root2_vector_half = aie::shuffle_up_fill(root2_vector_half, zero_vector4, 2);
    root3_vector_half = aie::shuffle_up_fill(root3_vector_half, zero_vector4, 2);
    aie::vector<int32_t, 8> root_vector01 = aie::concat(root0_vector_half, root1_vector_half);
    aie::vector<int32_t, 8> root_vector23 = aie::concat(root2_vector_half, root3_vector_half);
    aie::vector<int32_t, 16> root_vector = aie::concat(root_vector01, root_vector23);
    v0_r = vector_barrett(v0_r, p_vector, root_vector, u_vector, w);
    auto [v0_r0, v0_r1] = aie::interleave_unzip(v0_r, zero_vector, 2);

    auto [res, res2] = aie::interleave_zip(v0_l0, v0_r1, 2);
    aie::store_v(pA_i, res);
  }
  // Stage 2
  bf_width *= 2;
  root_idx /= 2;
  for (int i = 0; i < N / vec_prime; i++){
    int32_t *__restrict pA_i = a_in + i * vec_prime;
    aie::vector<int32_t, vec_prime> v0 = aie::load_v<vec_prime>(pA_i);
    aie::vector<int32_t, vec_prime> v0_l = aie::shuffle_down(v0, 4);
    v0_l = vector_modadd(v0, v0_l, p_vector);
    auto [v0_l0, v0_l1] = aie::interleave_unzip(v0_l, zero_vector, 4);
    
    aie::vector<int32_t, vec_prime> v0_r = aie::shuffle_up(v0, 4);
    v0_r = vector_modsub(v0_r, v0, p_vector);
    int32_t *__restrict pRoot_i = root_in + root_idx + core_idx * N_half / bf_width + i * vec_prime/8;
    // Case vec_prime = 16
    aie::vector<int32_t, vec_prime_half> root0_vector_half = aie::broadcast<int32_t, vec_prime_half>(pRoot_i[0]);
    aie::vector<int32_t, vec_prime_half> root1_vector_half = aie::broadcast<int32_t, vec_prime_half>(pRoot_i[1]);
    root0_vector_half = aie::shuffle_up_fill(root0_vector_half, zero_vector_half, 4);
    root1_vector_half = aie::shuffle_up_fill(root1_vector_half, zero_vector_half, 4);
    aie::vector<int32_t, vec_prime> root_vector = aie::concat(root0_vector_half, root1_vector_half);
    v0_r = vector_barrett(v0_r, p_vector, root_vector, u_vector, w);
    auto [v0_r0, v0_r1] = aie::interleave_unzip(v0_r, zero_vector, 4);

    auto [res, res2] = aie::interleave_zip(v0_l0, v0_r1, 4);
    aie::store_v(pA_i, res);
  }

  // Stage 3
  bf_width *= 2;
  root_idx /= 2;
  for (int i = 0; i < N / vec_prime; i++){
    int32_t *__restrict pA_i = a_in + i * vec_prime;
    aie::vector<int32_t, vec_prime> v0 = aie::load_v<vec_prime>(pA_i);
    aie::vector<int32_t, vec_prime> v0_l = aie::shuffle_down(v0, 8);
    v0_l = vector_modadd(v0, v0_l, p_vector);
    
    aie::vector<int32_t, vec_prime> v0_r = aie::shuffle_up(v0, 8);
    v0_r = vector_modsub(v0_r, v0, p_vector);
    int32_t *__restrict pRoot_i = root_in + root_idx + core_idx * N_half / bf_width + i * vec_prime/16;
    // Case vec_prime = 16
    aie::vector<int32_t, vec_prime> root_vector = aie::broadcast<int32_t, vec_prime>(pRoot_i[0]);
    v0_r = vector_barrett(v0_r, p_vector, root_vector, u_vector, w);
    v0_r = aie::shuffle_down(v0_r, 8);

    auto [res, res2] = aie::interleave_zip(v0_l, v0_r, 8);
    aie::store_v(pA_i, res);
  }

  // Stage 4 to Stage N-1
  for (int stage = 4; stage < logN; stage++){
    bf_width *= 2;
    root_idx /= 2;
    for (int i = 0; i < F; i++)
      chess_prepare_for_pipelining chess_loop_range(64, ) {
      int32_t cycle = bf_width / vec_prime;
      int32_t idx_base = (i / cycle) * bf_width * 2 + (i % cycle) * vec_prime;
      int32_t *__restrict pA_i = a_in + idx_base;
      int32_t root = root_in[root_idx + core_idx * N_half / bf_width + i / cycle];
      aie::vector<int32_t, vec_prime> root_vector = aie::broadcast<int32_t, vec_prime>(root);
      if (stage == logN - 1){
        int32_t *__restrict pC_i0 = c_out0 + idx_base;
        int32_t *__restrict pC_i1 = c_out1 + idx_base;
        ntt_stage_parallel8(pC_i0, pC_i1, pA_i, pA_i + bf_width, p_vector, root_vector, u_vector, p, w);     
      }else {
        ntt_stage_parallel8(pA_i, pA_i + bf_width, pA_i, pA_i + bf_width, p_vector, root_vector, u_vector, p, w);     
      }
    }
  }
  event1();
}

} // extern "C"
