#include <stdio.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>
// #include "omp.h"
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

#include <fstream>
#include <sstream>
#include <type_traits>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

static void HandleError(cudaError_t err,
    const char* file,
    int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

/*
specifically designed for lattice_white and lattice_black
*/

const int LX = 1024;  // Size of the lattice
const int LY = 1024;
const int L = 1024;

const int ITERATIONS = 3000;

__device__ __constant__ int d_J = 1.0f;
__device__ __constant__ int d_LX = LX;
__device__ __constant__ int d_LY = LY;
__device__ __constant__ int d_SubL = LX / 2;

__device__ float d_beta = 1.0f;

/* host side arrays*/
int* h_l;
int* h_lb;
int* h_lw;

/* device side arrays*/
int* d_l;
int* d_lb;
int* d_lw;

curandState* devStates_b, * devStates_w;

void initialize(int* h_l);
void matrix_decomposition(int* lattice, int* even_elements, int* odd_elements, int size);

void reconstruct_full_lattice(int* h_l, int* h_lb, int* h_lw) {
    int even_idx = 0;
    int odd_idx = 0;

    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            if ((i + j) % 2 == 0) {
                h_l[j + i * L] = h_lb[even_idx++];
            }
            else {
                h_l[j + i * L] = h_lw[odd_idx++];
            }
        }
    }
}

// Save lattice as a grayscale image (PGM format)
void save_pgm(const char* filename, int* lattice, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << width << " " << height << "\n255\n";

    for (int i = 0; i < width * height; ++i) {
        unsigned char pixel = (lattice[i] == 1) ? 255 : 0;
        file.write(reinterpret_cast<char*>(&pixel), 1);
    }

    file.close();
}

__global__ void monteCarloStep_white(int* lattice_black, int* lattice_white, curandState* devStates_w);
__global__ void monteCarloStep_black(int* lattice_black, int* lattice_white, curandState* devStates_b);
__global__ void setup_kernel_b(curandState* state_b, unsigned long long seed1);
__global__ void setup_kernel_w(curandState* state_w, unsigned long long seed2);

__global__ void unique(int* input) {
    int threadid = threadIdx.x;
    printf("threeadIdx : %d, value : %d \n", threadid, input[threadid]);
}

int magnetization(int* d_lb, int* d_lw) {
    int sum = 0;
    for (int i = 0; i < L * L / 2; i++) {
        sum += (d_lw[i] + d_lb[i]);
    }
    return sum;
}

int main() {
    size_t size = LX * LY;

    h_l = (int*)malloc(size * sizeof(int));
    h_lb = (int*)malloc(size / 2 * sizeof(int));
    h_lw = (int*)malloc(size / 2 * sizeof(int));

    HANDLE_ERROR(cudaMalloc(&d_l, size * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&d_lb, size * sizeof(int) / 2));
    HANDLE_ERROR(cudaMalloc(&d_lw, size * sizeof(int) / 2));

    // ----------------------------------
    // Allocate memory for device states
    cudaMalloc((void**)&devStates_b, size / 2 * sizeof(curandState));
    cudaMalloc((void**)&devStates_w, size / 2 * sizeof(curandState));
    // ----------------------------------

    initialize(h_l);
    matrix_decomposition(h_l, h_lb, h_lw, size);

    HANDLE_ERROR(cudaMemcpy(d_l, h_l, size * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lb, h_lb, size * sizeof(int) / 2, cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_lw, h_lw, size * sizeof(int) / 2, cudaMemcpyHostToDevice));

    unsigned long long seed1 = 1234;
    unsigned long long seed2 = 5678;

    dim3 block(128, 2, 1);
    dim3 grid(LX / 2 / block.x, LY / block.y, 1);

    // std::ostringstream filename;
    // filename << "mag_results/magnetization" << std::fixed << std::setprecision(0) << (beta * 100) << "txt";

    // std::ofstream outfile(filename.str(), std::ios::app);
    // if (!outfile.is_open()) {
    //     std::cerr << "Unable to open file for writing." << std::endl;
    //     return EXIT_FAILURE;
    // }
    // 
    // 
    // Setup kernel for initializing device states
    setup_kernel_b << <grid, block >> > (devStates_b, seed1);
    setup_kernel_w << <grid, block >> > (devStates_w, seed2);
    // --------------------


    auto tstart = std::chrono::high_resolution_clock::now();

    for (float beta = 0.1f; beta <= 1.01f; beta += 0.05f) {
        cudaMemcpyToSymbol(d_beta, &beta, sizeof(float));

        for (int t = 1; t <= ITERATIONS; t++) {
            // --------------------
            monteCarloStep_black << <grid, block >> > (d_lb, d_lw, devStates_b);
            monteCarloStep_white << <grid, block >> > (d_lb, d_lw, devStates_w);
        }
        HANDLE_ERROR(cudaDeviceSynchronize());

        //Saving PGM snaps

        HANDLE_ERROR(cudaMemcpy(h_lb, d_lb, size / 2 * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(h_lw, d_lw, size / 2 * sizeof(int), cudaMemcpyDeviceToHost));
        reconstruct_full_lattice(h_l, h_lb, h_lw);

        std::ostringstream filename;
        filename << "frames/snapshot_" << beta << ".pgm";
        save_pgm(filename.str().c_str(), h_l, LX, LY);
    }

    auto tend = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(tend - tstart).count();
    int beta_steps = static_cast<int>((1.0f - 0.1f) / 0.05f) + 1;
    double total_updates = static_cast<double>(ITERATIONS) * beta_steps * (LX * LY);
    double seconds = duration / 1e6;
    double MLUPS = total_updates / (seconds * 1e6);

    printf("performance : %f MLUPS \n", MLUPS);

    // Free host memory
    free(h_l);
    free(h_lb);
    free(h_lw);

    // Free device memory
    cudaFree(d_l);
    cudaFree(d_lb);
    cudaFree(d_lw);

    // Free device memory for device states
    cudaFree(devStates_b);
    cudaFree(devStates_w);
    std::cout << "Elapsed time:" << std::setw(9) << std::setprecision(4)
        << (tend - tstart).count() * 1e-9 << "\n";

    return 0;
}

void initialize(int* h_l) {
    for (int i = 0; i < LX; i++) {
        for (int j = 0; j < LY; j++) {
            h_l[i + j * LX] = 1;
        }
    }
}

__device__ inline int index_white(int i, int j) {
    return (i * d_LX + j) / 2;
}

__global__ void monteCarloStep_black(int* lattice_black, int* lattice_white, curandState* devStates_b) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;  // column in sublattice
    int y = threadIdx.y + blockIdx.y * blockDim.y;  // row
    int ID = x + y * d_SubL;

    if (x >= d_SubL || y >= d_LY) return;

    // Convert to full-lattice coordinates (i,j)
    int i = y;
    int j = 2 * x;

    int ip = (i + 1) % d_LY;
    int im = (i - 1 + d_LY) % d_LY;
    int jp = (j + 1) % d_LX;
    int jm = (j - 1 + d_LX) % d_LX;

    curandState localState = devStates_b[ID];

    int sum_neighbors =
        lattice_white[index_white(i, jp)] +
        lattice_white[index_white(i, jm)] +
        lattice_white[index_white(ip, j)] +
        lattice_white[index_white(im, j)];

    int deltaE = 2 * lattice_black[ID] * sum_neighbors;

    float randnum = curand_uniform(&localState);
    if (deltaE <= 0 || randnum < expf(-d_beta * deltaE)) {
        lattice_black[ID] *= -1;
    }

    devStates_b[ID] = localState;
}

__device__ inline int index_black(int i, int j) {
    return (i * d_LX + j) / 2;
}

__global__ void monteCarloStep_white(int* lattice_black, int* lattice_white, curandState* devStates_w) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int ID = x + y * d_SubL;

    if (x >= d_SubL || y >= d_LY) return;

    int i = y;
    int j = 2 * x + 1;

    int ip = (i + 1) % d_LY;
    int im = (i - 1 + d_LY) % d_LY;
    int jp = (j + 1) % d_LX;
    int jm = (j - 1 + d_LX) % d_LX;

    curandState localState = devStates_w[ID];

    int sum_neighbors =
        lattice_black[index_black(i, jp)] +
        lattice_black[index_black(i, jm)] +
        lattice_black[index_black(ip, j)] +
        lattice_black[index_black(im, j)];

    int deltaE = 2 * lattice_white[ID] * sum_neighbors;

    float randnum = curand_uniform(&localState);
    if (deltaE <= 0 || randnum < expf(-d_beta * deltaE)) {
        lattice_white[ID] *= -1;
    }

    devStates_w[ID] = localState;
}

void matrix_decomposition(int* h_l, int* h_lb, int* h_lw, int size) {
    int even_count = 0, odd_count = 0;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            if ((i + j) % 2 == 0) {
                h_lb[even_count++] = h_l[j + i * L];
            }
            else {
                h_lw[odd_count++] = h_l[j + i * L];
            }
        }
    }
}

// ----------------------------------------------------
__global__ void setup_kernel_b(curandState* state_b, unsigned long long seed1) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int ID = x + y * d_SubL;

    curand_init(seed1, ID, 0, &state_b[ID]);
}

__global__ void setup_kernel_w(curandState* state_w, unsigned long long seed2) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int ID = x + y * d_SubL;

    curand_init(seed2, ID, 0, &state_w[ID]);
}