#include <chrono>
#include <iostream>
#include <immintrin.h>
#include <stdlib.h>

int reduce_sum_normal(int *array, int N)
{
    int sum = 0;
    for(int i=0; i<N; ++i)
    {
        sum += array[i];
    }
    return sum;
}

// 每次读取 8 个整数, 但是为了防止不对齐的现象, 将最开始直接减少 8
int reduce_sum_simd(int *array, int N)
{
    // 使用向量化的处理能够对齐的部分
    __m256i result = _mm256_setzero_si256();
    int i=0;
    for(; i<=N-8; i+=8)
    {
        result = _mm256_add_epi32(result, _mm256_loadu_si256((__m256i*)&array[i]));
    }

    // 将向量化的结果进行规约
    int sum_vec[8];
    int scalar_result = 0;
    _mm256_storeu_si256((__m256i*)sum_vec, result);
    #pragma GCC unroll 4
    for(int j=0; j<8; ++j)
    {
        scalar_result+=sum_vec[j];
    }

    // 将剩余部分进行相加
    #pragma GCC unroll 4
    for(; i<N; ++i)
    {
        scalar_result+=array[i];
    }

    return scalar_result;
}

int main(){
    constexpr int N = 1000000;  // 数组的元素个数
    int *array = (int*)malloc(sizeof(int)*N);  // 开辟空间
    for(int i=0; i<N; ++i)
    {
        array[i] = rand() % 20;
    }

    auto start_time = std::chrono::high_resolution_clock::now(); 
    int sum_normal = reduce_sum_normal(array, N);
    auto end_time = std::chrono::high_resolution_clock::now(); 
    auto duration_time_normal = std::chrono::duration<double>(end_time-start_time);

    start_time = std::chrono::high_resolution_clock::now(); 
    int sum_simd = reduce_sum_simd(array, N);
    end_time = std::chrono::high_resolution_clock::now(); 
    auto duration_time_simd = std::chrono::duration<double>(end_time-start_time);

    std::cout << "普通规约结果: " << sum_normal << " ,用时: " << duration_time_normal.count() << " s" << std::endl;
    std::cout << "SIMD规约结果: " << sum_simd << " ,用时: " << duration_time_simd.count() << " s" << std::endl;

    return 0;
}