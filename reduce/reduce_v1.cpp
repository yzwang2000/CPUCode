#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

// 普通方法进行规约求和
int sum_normal(int *array, int size) {
    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }
    return sum;
}

// 使用AVX2进行规约求和
int sum_avx2(int *array, int size){
    __m256i sum_vec = _mm256_setzero_si256();
    int i;
    for (i = 0; i <= size - 8; i += 8) {
        __m256i vec = _mm256_load_si256((__m256i*)&array[i]);
        sum_vec = _mm256_add_epi32(sum_vec, vec);
    }

    // 将向量中的各个部分规约到一个标量
    int temp[8] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i*)temp, sum_vec);

    int sum = 0;
    for (int j = 0; j < 8; j++) {
        sum += temp[j];
    }

    // 处理剩余元素
    for (; i < size; i++) {
        sum += array[i];
    }

    return sum;
}

int main() {
    int size = 1000000;
    int *array;
    posix_memalign((void**)&array, 32, size * sizeof(int)); // 对齐内存分配
    
    // 初始化数组
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 100;
    }

    // 测试普通方法
    clock_t start_normal = clock();
    int sum1 = sum_normal(array, size);
    clock_t end_normal = clock();
    double time_normal = (double)(end_normal - start_normal) / CLOCKS_PER_SEC;

    // 测试AVX2方法
    clock_t start_avx2 = clock();
    int sum2 = sum_avx2(array, size);
    clock_t end_avx2 = clock();
    double time_avx2 = (double)(end_avx2 - start_avx2) / CLOCKS_PER_SEC;

    // 输出结果
    printf("普通方法求和: %d\n", sum1);
    printf("AVX2方法求和: %d\n", sum2);
    printf("普通方法时间: %f秒\n", time_normal);
    printf("AVX2方法时间: %f秒\n", time_avx2);

    free(array);
    return 0;
}