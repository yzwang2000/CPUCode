#include <iostream>
#include <immintrin.h>
#include <stdlib.h>
#include <chrono>

void sumArrayNormal(float* A, float* B, float* C, int N)
{
    for(int i=0; i<N; ++i)
    {
        A[i] = B[i] + C[i];
    }
}

void sumArraySimd(float* A, float* B, float* C, int N)
{
    // 先处理能够规约的部分
    int i=0;
    for(; i<=N-8; i+=8)
    {
        _mm256_store_ps(C+i, _mm256_add_ps(_mm256_load_ps(A+i), _mm256_load_ps(B+i)));
    }

    for(; i<N; ++i)
    {
        C[i] = A[i] + B[i];
    }
}

// 检查结果是否一致
bool checkResult(float* C1, float* C2, int size) {
    for (int i = 0; i < size; i++) {
        if (C1[i] != C2[i]) {
            return false;
        }
    }
    return true;
}

int main(){
    constexpr int N = 1000001;  // 元素个数 
    float *A  = (float*)aligned_alloc(32, sizeof(float)*N);
    float *B  = (float*)aligned_alloc(32, sizeof(float)*N);
    float *C  = (float*)aligned_alloc(32, sizeof(float)*N);
    float *C2 = (float*)aligned_alloc(32, sizeof(float)*N);

    auto start_time = std::chrono::high_resolution_clock::now();
    sumArrayNormal(A, B, C, N);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_normal = std::chrono::duration<double>(end_time-start_time);

    start_time = std::chrono::high_resolution_clock::now();
    sumArraySimd(A, B, C2, N);
    end_time = std::chrono::high_resolution_clock::now();
    auto duration_simd = std::chrono::duration<double>(end_time-start_time);

    bool result = checkResult(C, C2, N);
    if(result)
    {
        std::cout << "Sucess" << std::endl;
    }else{
        std::cout << "Failed" << std::endl;
    }

    std::cout << "普通运算用时: " << duration_normal.count() << std::endl;
    std::cout << "SIMD运算用时: " << duration_simd.count() << std::endl;

    free(A);
    free(B);
    free(C);
    free(C2);

    return 0;
}