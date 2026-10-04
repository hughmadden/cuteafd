// Register-only SM120 MMA throughput. This is an instruction ceiling, not GEMM
// throughput: no trellis decode, rotations, operand loads or quantization.
// nvcc -O3 -gencode arch=compute_120,code=sm_120 exl3_mma_peak.cu -o mma_peak
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

constexpr int iterations = 4096;
constexpr int chains = 8;
template<int kind> __global__ void mma_peak(float* output) {
    float accum[chains][4] = {};
    int integer[chains][4] = {};
    // Distinct seeds preserve eight independent dependency chains in SASS.
    #pragma unroll
    for (int c = 0; c < chains; ++c)
        for (int j = 0; j < 4; ++j) {
            accum[c][j] = float(c + 1) * 0.0001f;
            integer[c][j] = kind == 5 ? c + 1 : 0x10001000 + c * 0x00010001;
        }
    // Finite small values for F16/BF16/FP8; bounded signed-byte INT8 inputs.
    unsigned a = kind == 1 ? 0x38003800 : 0x14001400;
    if constexpr (kind >= 3) a = 0x01010101;
    #pragma unroll 1
    for (int it = 0; it < iterations; ++it) {
        #pragma unroll
        for (int c = 0; c < chains; ++c) {
            if constexpr (kind == 0)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(accum[c][0]), "+f"(accum[c][1]), "+f"(accum[c][2]), "+f"(accum[c][3])
                    : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
            else if constexpr (kind == 1)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(accum[c][0]), "+f"(accum[c][1]), "+f"(accum[c][2]), "+f"(accum[c][3])
                    : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
            else if constexpr (kind == 2)
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                    "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                    : "+r"(integer[c][0]), "+r"(integer[c][1])
                    : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
            else if constexpr (kind == 3)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+f"(accum[c][0]), "+f"(accum[c][1]), "+f"(accum[c][2]), "+f"(accum[c][3])
                    : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
            else if constexpr (kind == 4)
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                    "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};"
                    : "+r"(integer[c][0]), "+r"(integer[c][1])
                    : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
            else
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+r"(integer[c][0]), "+r"(integer[c][1]), "+r"(integer[c][2]), "+r"(integer[c][3])
                    : "r"(a), "r"(a), "r"(a), "r"(a), "r"(a), "r"(a));
        }
    }
    float sum = 0;
    #pragma unroll
    for (int c = 0; c < chains; ++c)
        for (int j = 0; j < 4; ++j) sum += accum[c][j] + float(integer[c][j]);
    output[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

template<int kind> void measure(const char* name, int sms, float* output) {
    constexpr int threads = 256, replays = 5;
    int blocks = sms * 4;
    for (int i = 0; i < 3; ++i) mma_peak<kind><<<blocks, threads>>>(output);
    check(cudaGetLastError());
    check(cudaDeviceSynchronize());
    cudaEvent_t begin, end;
    check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
    float times[3];
    for (float& ms : times) {
        check(cudaEventRecord(begin));
        for (int i = 0; i < replays; ++i) mma_peak<kind><<<blocks, threads>>>(output);
        check(cudaEventRecord(end)); check(cudaEventSynchronize(end));
        check(cudaEventElapsedTime(&ms, begin, end));
        check(cudaGetLastError());
    }
    std::sort(times, times + 3);
    double ops = double(replays) * blocks * (threads / 32) * iterations * chains
                 * 2 * 16 * 8 * (kind >= 3 ? 32 : 16);
    std::printf("%s,%.3f,%.3f\n", name, times[1] / replays, ops / times[1] / 1e9);
    check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
}

int main() {
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0));
    if (props.major != 12 || props.minor != 0) return 2;
    std::printf("# %s; %d SMs; 4x256 threads/SM; 8 chains; %d iterations\n",
                props.name, props.multiProcessorCount, iterations);
    std::puts("mma,median_ms,tops");
    float* output;
    check(cudaMalloc(&output, props.multiProcessorCount * 4 * 256 * sizeof(float)));
    measure<0>("f16-f32acc", props.multiProcessorCount, output);
    measure<1>("bf16-f32acc", props.multiProcessorCount, output);
    measure<2>("f16-f16acc", props.multiProcessorCount, output);
    measure<3>("e4m3-f32acc", props.multiProcessorCount, output);
    measure<4>("e4m3-f16acc", props.multiProcessorCount, output);
    measure<5>("int8-s32acc", props.multiProcessorCount, output);
    check(cudaFree(output));
}
