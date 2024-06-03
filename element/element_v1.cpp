#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <immintrin.h>
#include <stdbool.h>

#define SIZE 1000000

// SIMD 数组相加函数
void simdArrayAdd(float* A, float* B, float* C) {
    #pragma GCC unroll 4
    for (int i = 0; i < SIZE; i += 8) { // 循环跨度修改为 8
        _mm256_store_ps(C + i, _mm256_add_ps(_mm256_load_ps(A + i), _mm256_load_ps(B + i)));
    }
}

// 普通数组相加函数
void normalArrayAdd(float* A, float* B, float* C) {
    #pragma GCC unroll 4
    for (int i = 0; i < SIZE; i++) {
        C[i] = A[i] + B[i];
    }
}

// 检查结果是否一致
bool checkResult(float* C1, float* C2) {
    for (int i = 0; i < SIZE; i++) {
        if (C1[i] != C2[i]) {
            return false;
        }
    }
    return true;
}

int main() {
    float* A  = (float*)aligned_alloc(32, SIZE * sizeof(float));
    float* B  = (float*)aligned_alloc(32, SIZE * sizeof(float));
    float* C  = (float*)aligned_alloc(32, SIZE * sizeof(float));
    float* C2 = (float*)aligned_alloc(32, SIZE * sizeof(float));

    // 随机初始化数组
    for (int i = 0; i < SIZE; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    clock_t startSIMD, endSIMD, startNormal, endNormal;

    startSIMD = clock();
    simdArrayAdd(A, B, C);
    endSIMD = clock();

    startNormal = clock();
    normalArrayAdd(A, B, C2);
    endNormal = clock();

    printf("SIMD 加法时间: %f 毫秒\n", (double)(endSIMD - startSIMD) * 1000.0 / CLOCKS_PER_SEC);
    printf("普通加法时间: %f 毫秒\n", (double)(endNormal - startNormal) * 1000.0 / CLOCKS_PER_SEC);

    if (checkResult(C, C2)) {
        printf("结果一致\n");
    } else {
        printf("结果不一致\n");
    }

    free(A);
    free(B);
    free(C);
    free(C2);

    return 0;
}