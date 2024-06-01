#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

void bit_reversal(uint64_t n, std::vector<uint64_t> &res) {
    for (int i = 0; i < n; i++) {
        res[i] = 0;
    }

    uint64_t d = n / 2;
    for (int u = 1; u < n; u <<= 1) {
        for (int k = 0; k < u; k++) {
						std::cout << "(k,u)=(" << k << "," << u << ")" << "res[k]=" << res[k] << ", d=" << d << std::endl;
            res[k | u] = res[k] | d;
        }
        d >>= 1;
    }
}

uint64_t modPow(uint64_t x, uint64_t n, uint64_t mod) {
    uint64_t ret;
    if (n == 0) {
        ret = 1;
    } else if (n % 2 == 1) {
        ret = (x * modPow((x * x) % mod, n / 2, mod)) % mod;
    } else {
        ret = modPow((x * x) % mod, n / 2, mod);
    }
    return ret;
}

void ntt(std::vector<uint64_t> &a, uint64_t n,
          std::vector<uint64_t> &roots_rev, uint64_t p) {
    uint64_t t = 1;
    uint64_t j1, j2, h;
    for (int m = n; m > 1; m >>= 1) {
        j1 = 0;
        h = m / 2;
        for (int i = 0; i < h; i++) {
            j2 = j1 + t - 1;
            for (int j = j1; j <= j2; j++) {
                //std::cout << j << ", " << j + t << ", root=" << h + i << std::endl;
                uint64_t root = roots_rev[h + i];
                uint64_t v0 = a[j];
                uint64_t v1 = a[j + t];
                a[j] = (v0 + v1) % p;
                a[j + t] = (((v0 - v1 + p) % p) * root) % p;
            }
            j1 += 2 * t;
        }
        t <<= 1;
    }
}

void intt(std::vector<uint64_t> &a, uint64_t n, std::vector<uint64_t> &roots_rev,
         uint64_t p, uint64_t n_inv) {
    uint64_t t = n;
    uint64_t j1, j2;
    for (int m = 1; m < n; m <<= 1) {
        t >>= 1;
        for (int i = 0; i < m; i++) {
            j1 = 2 * i * t;
            j2 = j1 + t - 1;
            for (int j = j1; j <= j2; j++) {
                //std::cout << j << ", " << j + t << ", root=" << m + i << std::endl;
                uint64_t root = roots_rev[m + i];
                uint64_t v0 = a[j];
                uint64_t v1 = (a[j + t] * root) % p;
                a[j] = (v0 + v1) % p;
                a[j + t] = (v0 - v1 + p) % p;
            }
        }
    }
     for (int i = 0; i < n; i++) {
        a[i] = (a[i] * n_inv) % p;
    }
}

/*
    Debug
*/
void debug_vector(std::vector<uint64_t> &poly) {
    for (uint64_t e : poly) {
        std::cout << e << " ";
    }
    std::cout << std::endl;
}

void is_equal_polynomial(std::vector<uint64_t> &a, std::vector<uint64_t> &b) {
    std::cout << "TEST";
    assert(a.size() == b.size());
    for (int i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) {
            std::cerr << "...failed" << std::endl;
            std::cout << a[i] << " vs " << b[i] << std::endl;
            return;
        }
    }
    std::cout << "...success!" << std::endl;
}

int main() {
    std::cout << "DFT" << std::endl;
    // Parameters
    uint64_t n = 1 << 4;
    uint64_t p = 998244353;
    uint64_t g = 3;

    uint64_t w = modPow(g, (p - 1) / n, p);
    uint64_t n_inv = modPow(n, p - 2, p);
    std::cout << "(w, n_inv) = (" << w << ", " << n_inv << ")" << std::endl;

    std::vector<uint64_t> roots(n);
    std::vector<uint64_t> invroots(n);
    roots[0] = 1;
    invroots[0] = 1;
    // make roots
    for (int i = 1; i < n; i++) {
        roots[i] = (roots[i - 1] * w) % p;
        invroots[i] = (modPow(roots[i], p - 2, p));
    }
   
    //std::cout << "ROOTS" << std::endl;
    //debug_vector(roots);

    // Input
    std::vector<uint64_t> input(n, 0);
    std::vector<uint64_t> a(n, 0);
    for (int i = 0; i < n; i++) {
        input[i] = i;
        a[i] = i;
    }

    std::cout << "INPUT" << std::endl;
    debug_vector(a);

    // The result is bit-reversed order
    // Use roots_rev and invroots_rev for the correct order
    /*
    std::vector<uint64_t> roots_rev(n);
    std::vector<uint64_t> invroots_rev(n);
    std::vector<uint64_t> bitrev(n, 0);
    bit_reversal(n, bitrev);
    for (int i = 0; i < n; i++) {
        roots_rev[bitrev[i]] = roots[i];
        invroots_rev[bitrev[i]] = invroots[i];
    }
    */


    std::cout << "========= ntt ===========" << std::endl;
    ntt(a, n, roots, p);
    debug_vector(a);

    std::cout << "========= intt ============" << std::endl;
    intt(a, n, invroots, p, n_inv);
    debug_vector(a);

    is_equal_polynomial(input, a);
}