#include <iostream>
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

void init_matrix(int **src, int val, int row, int col)
{
    for(int i=0; i<row; ++i)
    {
        for(int j=0; j<col; ++j)
        {
            src[i][j] = val;
        }
    }
}

// void matrixMul_normal(int **A, int **B, int **C, int M, int N, int K)
// {
//     for(int i=0; i<M; ++i)
//     {
//         for(int k=0; k<K; ++k)
//         {
//             for(int j=0; j<N; ++j)
//             {
//                 C[i][j] += A[i][k] * B[k][j];
//             }
//         }
//     }
// }

void matrixMul_normal(int **A, int **B, int **C, int M, int N, int K)
{
    for(int i=0; i<M; ++i)
    {
        for(int j=0; j<N; ++j)
        {
            for(int k=0; k<K; ++k)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void matrixMul_simd(int **A, int **B, int **C, int M, int N, int K){
    #pragma omp parallel for num_threads(8)
    for(int i=0; i<M; ++i)
    {
        for(int k=0; k<K; ++k)
        {
            int j = 0;
            for(; j<=N-8; j+=8)
            {
                _mm256_store_si256((__m256i*)(C[i]+j), _mm256_add_epi32(_mm256_load_si256((__m256i*)(C[i]+j)),
                                 _mm256_mullo_epi32(_mm256_set1_epi32(A[i][k]), _mm256_load_si256((__m256i*)(B[k]+j)))));
            }

            for(;j<N; ++j)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

bool compare_matrices(int **C1, int **C2, int M, int N)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            if(C1[i][j] != C2[i][j])
            {
                return false;
            }
        }
    }
    return true;
}

int main(){
    // 矩阵 A 的大小为 M*K, 矩阵 B 的大小为 K*N, 矩阵 C 的大小为 M*N
    constexpr int M=1024, N=1024, K=1024;
    int** A  = (int**)malloc(M*sizeof(int*));
    int** B  = (int**)malloc(K*sizeof(int*));
    int** C1 = (int**)malloc(M*sizeof(int*));
    int** C2 = (int**)malloc(M*sizeof(int*));

    // A 是一个指针数组, 其中的元素是指针, 指向每一行的数据
    for (int i = 0; i < M; ++i){
        A[i] = (int*)aligned_alloc(32, sizeof(int)*K);
    }
    for (int i = 0; i < K; ++i){
        B[i] = (int*)aligned_alloc(32, sizeof(int)*N);
    }
    for (int i = 0; i < M; ++i){
        C1[i] = (int*)aligned_alloc(32, sizeof(int)*N);
        C2[i] = (int*)aligned_alloc(32, sizeof(int)*N);
    }

    init_matrix(A, 1, M, K);
    init_matrix(B, 1, K, N);
    init_matrix(C1, 0, M, N);
    init_matrix(C2, 0, M, N);

    auto start_time = std::chrono::high_resolution_clock::now();
    matrixMul_normal(A, B, C1, M, N, K);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_normal = std::chrono::duration<double>(end_time-start_time);

    start_time = std::chrono::high_resolution_clock::now();
    matrixMul_simd(A, B, C2, M, N, K);
    end_time = std::chrono::high_resolution_clock::now();
    auto duration_simd = std::chrono::duration<double>(end_time-start_time);

    std::cout << "普通矩阵乘法的运行时间: " << duration_normal.count() << " s" << std::endl;
    std::cout << "SIMD矩阵乘法的运行时间: " << duration_simd.count() << " s" << std::endl;

    if(compare_matrices(C1, C2, M, N))
    {
        std::cout << "两种计算方法的结果一致。" << std::endl;
    }
    else
    {
        std::cout << "两种计算方法的结果不一致！" << std::endl;
    }

    for (int i = 0; i < M; ++i) {
        free(A[i]);
    }
    for (int i = 0; i < K; ++i) {
        free(B[i]);
    }
    for (int i = 0; i < M; ++i) {
        free(C1[i]);
        free(C2[i]);
    }

    free(A);
    free(B);
    free(C1);
    free(C2);
    
    return 0;
}