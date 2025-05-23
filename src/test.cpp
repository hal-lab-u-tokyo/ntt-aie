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

int32_t modPow(int32_t x, int32_t n, int32_t mod) {
    int32_t ret;
    if (n == 0) {
        ret = 1;
    } else if (n % 2 == 1) {
        ret = (x * modPow((x * x) % mod, n / 2, mod)) % mod;
    } else {
        ret = modPow((x * x) % mod, n / 2, mod);
    }
    return ret;
}

void make_roots(int32_t n, std::vector<int32_t> &roots, int32_t p, int32_t g) {
    int32_t w = modPow(g, (p - 1) / n, p);
    for (int i = 1; i < n; i++) {
        roots[i] = (uint32_t) (((uint32_t) roots[i - 1] * w) % p);
    }
}

void ntt(std::vector<int32_t> &a, int32_t n, std::vector<int32_t> &roots_rev,
         int32_t p, int32_t stage) {
    int32_t t = 1;
    int32_t j1, j2, h;
    int idx = 0;
    for (int m = n; m > 1; m >>= 1) {
        j1 = 0;
        h = m / 2;
        for (int i = 0; i < h; i++) {
            j2 = j1 + t - 1;
            for (int j = j1; j <= j2; j++) {
                int32_t root = roots_rev[h + i];
                int32_t v0 = a[j];
                int32_t v1 = a[j + t];
                a[j] = (v0 + v1) % p;
                a[j + t] =
                    (static_cast<uint64_t>((v0 + p - v1) % p) * root) % p;
            }
            j1 += 2 * t;
        }
        t <<= 1;
        if (idx == stage) {
            return;
        }
        idx += 1;
    }
}

int main(int argc, const char *argv[]) {
    // ============================
    // Test Parameters
    // ============================
    constexpr int32_t n = 11;
    constexpr int32_t test_stage = n - 1;

    const int block_num = 16;
    std::array<int, block_num> ans_order = {0, 2, 1, 3, 8,  10, 9,  11,
                                            4, 6, 5, 7, 12, 14, 13, 15};

    // ============================
    // Constants
    // ============================
    constexpr int32_t p = 3329;
    constexpr int32_t g = 3;
    constexpr bool VERIFY = true;
    int IN_VOLUME = 1 << n;
    int OUT_VOLUME = IN_VOLUME;

    // ============================
    // Program arguments parsing
    // ============================
    po::options_description desc("Allowed options");
    po::variables_map vm;
    test_utils::add_default_options(desc);

    test_utils::parse_options(argc, argv, desc, vm);
    int verbosity = vm["verbosity"].as<int>();
    int trace_size = vm["trace_sz"].as<int>();
    printf("trace size: %d\n", trace_size);

    int IN_SIZE = IN_VOLUME * sizeof(int32_t);
    int OUT_SIZE = OUT_VOLUME * sizeof(int32_t) + trace_size;

    // ============================
    // Load instructions and data
    // ============================
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_sequence(vm["instr"].as<std::string>());

    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    // Start the XRT context and load the kernel
    xrt::device device;
    xrt::kernel kernel;

    std::cout << "Init xrt load kernel: " << std::endl;
    test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                     vm["xclbin"].as<std::string>(),
                                     vm["kernel"].as<std::string>());

    // set up the buffer objects
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
    auto bo_inA =
        xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
    auto bo_root =
        xrt::bo(device, IN_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_prime = xrt::bo(device, 1 * sizeof(int32_t), XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(4));
    auto bo_outC =
        xrt::bo(device, OUT_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

    std::cout << "Writing data into buffer objects.\n";

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    // Initialize buffer
    int32_t *bufInA = bo_inA.map<int32_t *>();
    int32_t *bufRoot = bo_root.map<int32_t *>();
    int32_t *bufInFactor = bo_prime.map<int32_t *>();
    int32_t *bufOut = bo_outC.map<int32_t *>();
    std::vector<int32_t> root(IN_VOLUME);
    root[0] = 1;
    make_roots(IN_VOLUME, root, p, g);
    for (int i = 0; i < IN_VOLUME; i++) {
        bufInA[i] = i;
        bufRoot[i] = root[i];
        bufOut[i] = 0;
    }
    *bufInFactor = (int32_t) scaleFactor;

    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_root.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // ============================
    // Execute the kernel 10 times
    // ============================
    std::cout << "Running Kernel.\n";
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto run = kernel(bo_instr, instr_v.size(), bo_inA, bo_root, bo_outC);
        ert_cmd_state r = run.wait();
        auto stop = std::chrono::high_resolution_clock::now();
        if (r != ERT_CMD_STATE_COMPLETED) {
            std::cout << "kernel did not complete. returned status: " << r
                      << "\n";
            return 1;
        }
        // Sync device to host memories
        bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        // Time
        float npu_time =
            std::chrono::duration_cast<std::chrono::microseconds>(stop - start)
                .count();
        std::cout << npu_time << std::endl;
    }

    // ============================
    // Execute the kernel for test
    // ============================
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

    std::cout << "=================================" << std::endl;
    if (trace_size > 0) {
        std::cout << "Writing trace output to "
                  << vm["trace_file"].as<std::string>() << std::endl;
        test_utils::write_out_trace(((char *) bufOut) + IN_SIZE, trace_size,
                                    vm["trace_file"].as<std::string>());
    }

    // ============================
    // CPU Reference
    // ============================
    std::vector<int32_t> a_ref(IN_VOLUME);
    for (int i = 0; i < IN_VOLUME; i++) {
        a_ref[i] = i;
    }
    ntt(a_ref, IN_VOLUME, root, p, test_stage);

    // ============================
    // Veryfy Results
    // ============================
    std::vector<int32_t> answers(IN_VOLUME);
    int block_size = IN_VOLUME / block_num;
    for (int i = 0; i < block_num; i++) {
        int base_i = ans_order[i] * block_size;
        for (int j = 0; j < block_size; j++) {
            answers[base_i + j] = a_ref[i * block_size + j];
        }
    }

    int errors = 0;
    std::cout << "Verifying results" << std::endl;

    for (int32_t i = 0; i < IN_VOLUME; i++) {
        int32_t ref = answers[i];
        int32_t test = bufOut[i];
        if (test != ref) {
            // std::cout << "[" << i << "] Error " << test << " != " << ref <<
            // std::endl;
            errors++;
        } else {
            // std::cout << "[" << i << "] Correct " << test << " == " << ref <<
            // std::endl;
        }
    }

    std::cout << "  logN: " << n << std::endl;
    std::cout << "  p: " << p << std::endl;

    if (!errors) {
        std::cout << "  PASS!" << std::endl;
        return 0;
    } else {
        std::cout << "  mismatches: " << errors << std::endl;
        std::cout << "  FAIL." << std::endl << std::endl;
        return 1;
    }
}
