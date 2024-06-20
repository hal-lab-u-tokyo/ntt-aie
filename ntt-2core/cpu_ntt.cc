#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream> 
#include <iostream>
#include <string>
#include <vector>

void bit_reversal(int64_t n, std::vector<int64_t> &res) {
    for (int i = 0; i < n; i++) {
        res[i] = 0;
    }

    int64_t d = n / 2;
    for (int u = 1; u < n; u <<= 1) {
        for (int k = 0; k < u; k++) {
			//std::cout << "(k,u)=(" << k << "," << u << ")" << "res[k]=" << res[k] << ", d=" << d << std::endl;
            res[k | u] = res[k] | d;
        }
        d >>= 1;
    }
}

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

int32_t modadd(int32_t a, int32_t b, int32_t q){
    int32_t ret = a + b;
    if (ret >= q){
        return ret - q;
    }
    return ret;
}

int32_t modsub(int32_t a, int32_t b, int32_t q){
    int32_t ret = a + q - b;
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
    //std::cout << "log2(t): " << std::log2(t) << ", log2(x_2): " << std::log2(x_2) << ", log2(r): " << std::log2(r) << std::endl;
	int64_t c = t - r;
	if (c >= q) {
		return c - q;
	}else {
		return c;
	}
}

void ntt(std::vector<int64_t> &a, int64_t n,
          std::vector<int64_t> &roots_rev, int64_t p, int32_t w, int32_t u) {
    int64_t t = 1;
    int64_t j1, j2, h;
    int idx = 0;
    for (int m = n; m > 1; m >>= 1) {
        j1 = 0;
        h = m / 2;
        for (int i = 0; i < h; i++) {
            j2 = j1 + t - 1;
            for (int j = j1; j <= j2; j++) {
                int64_t root = roots_rev[h + i];
                //std::cout << j << ", " << j + t << ", root[" << h + i << "]=" << root <<std::endl;
                int64_t v0 = a[j];
                int64_t v1 = a[j + t];
                a[j] = modadd(v0, v1, p);
                a[j + t] = barrett_2k(modsub(v0, v1, p), root, p, w, u);
                //std::cout << "modsub = " << modsub(v0, v1, p) << std::endl;
            }
            j1 += 2 * t;
        }
        t <<= 1;
        idx += 1;
    }
}

void intt(std::vector<int64_t> &a, int64_t n, std::vector<int64_t> &roots_rev,
         int64_t p, int64_t n_inv) {
    int64_t t = n;
    int64_t j1, j2;
    for (int m = 1; m < n; m <<= 1) {
        t >>= 1;
        for (int i = 0; i < m; i++) {
            j1 = 2 * i * t;
            j2 = j1 + t - 1;
            for (int j = j1; j <= j2; j++) {
                //std::cout << j << ", " << j + t << ", root=" << m + i << std::endl;
                int64_t root = roots_rev[m + i];
                int64_t v0 = a[j];
                int64_t v1 = (a[j + t] * root) % p;
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
void debug_vector(std::vector<int64_t> &poly) {
    for (int64_t e : poly) {
        std::cout << e << std::endl;
    }
    //std::cout << std::endl;
}

void is_equal_polynomial(std::vector<int64_t> &a, std::vector<int64_t> &b) {
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

void export_vector_to_file(std::vector<int64_t> &poly, int32_t q, int32_t n){
    std::ostringstream oss;
    oss << "../data/ans_q" << q << "_n" << n << ".txt";
    std::string filename = oss.str();
    std::ofstream logFile(filename);
    if (!logFile) {
        std::cerr << "Failed to open file " << filename << std::endl;
        return;
    }
    for (int64_t e : poly) {
        logFile << e << std::endl;
    }
    logFile.close();
    std::cout << "Write to " << filename << std::endl;
}

int main() {
    std::cout << "DFT" << std::endl;
    // Parameters
    int32_t logn = 11;
    int64_t n = 1 << logn;
    //int64_t p = 998244353;
    //int64_t p = 65537;
    int64_t p = 3329;
    //int64_t p = 1097;
    int64_t g = 3;
    int64_t w = modPow(g, (p - 1) / n, p);
    int64_t n_inv = modPow(n, p - 2, p);
    int32_t barrett_w = std::ceil(std::log2(p));
    int32_t barrett_u = std::floor(std::pow(2, 2 * barrett_w) / p);
    //std::cout << "(w, n_inv) = (" << w << ", " << n_inv << ")" << std::endl;

    std::vector<int64_t> roots(n);
    std::vector<int64_t> invroots(n);
    roots[0] = 1;
    invroots[0] = 1;
    // make roots
    for (int i = 1; i < n; i++) {
        roots[i] = (roots[i - 1] * w) % p;
        invroots[i] = (modPow(roots[i], p - 2, p));
    }   

    // Input
    std::vector<int64_t> input(n, 0);
    std::vector<int64_t> a(n, 0);
    for (int i = 0; i < n; i++) {
        input[i] = i;
        a[i] = i;
    }

/*
    std::cout << "INPUT" << std::endl;
    debug_vector(a);
    std::cout << "ROOTS" << std::endl;
    debug_vector(roots);

    std::vector<int64_t> roots_rev(n);
    std::vector<int64_t> invroots_rev(n);
    std::vector<int64_t> bitrev(n, 0);
    bit_reversal(n, bitrev);
    for (int i = 0; i < n; i++) {
        roots_rev[bitrev[i]] = roots[i];
        invroots_rev[bitrev[i]] = invroots[i];
    }
*/

    std::cout << "========= ntt ===========" << std::endl;
    ntt(a, n, roots, p, barrett_w, barrett_u);
    export_vector_to_file(a, p, logn);

    //std::cout << "========= intt ============" << std::endl;
    //intt(a, n, invroots, p, n_inv);
    //debug_vector(a);

    //is_equal_polynomial(input, a);
}