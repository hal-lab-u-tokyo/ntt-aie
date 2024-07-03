#include <cassert>
#include <cstdint>
#include <iostream>
#include <cmath>

uint32_t barrett(uint32_t a, uint32_t b, uint32_t q, uint32_t w, uint32_t u){
	uint64_t t = a * b;
	uint64_t s = (t * u) >> (2 * w);
	uint64_t r = s * q;
	uint64_t c = t - r;
	if (c >= q) {
		return c - q;
	}else {
		return c;
	}
}

uint32_t barrett_2k(uint32_t a, uint32_t b, uint32_t q, uint32_t w, uint32_t u){
	uint64_t t = (uint64_t)a *(uint64_t) b;
	uint64_t x_1 = t >> (w - 2);
	uint64_t x_2 = u * x_1;
	uint64_t s = x_2 >> (w + 2);
	uint64_t r = s * q;
	uint64_t c = t - r;
	if (c >= q) {
		return c - q;
	}else {
		return c;
	}
}

uint64_t naive(uint32_t a, uint32_t b, uint32_t q){
	return ((uint64_t)a * (uint64_t)b ) % (uint64_t)q;
}

int main(){
	uint32_t q = 998244353;
    // 30-bit design
    // Ref: https://eprint.iacr.org/2021/124.pdf
	uint64_t all = (1 << 30) - 1;
    uint32_t w = std::ceil(std::log2(q));
    uint32_t u = std::floor(std::pow(2, 2 * w) / q);

	for (uint64_t i = 0; i < all; i++){
		uint32_t a = i;
		uint32_t b = i;
        	
		uint32_t nai = naive(a, b, q);
		uint32_t brt = barrett_2k(a, b, q, w, u);
		if (nai != brt){
			printf("(%d, %d, %d) = naive:%d vs barrett:%d\n", a, b, q, nai, brt);
			printf("Passed %ld / %ld\n", i + 1, all);
			return 1;
		}
	
	}
	printf("Passed all %ld\n", all);
	return 0;
}
