#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

void MatrixMulNormal(int **A, int **B, int **C, int size)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

void MatrixMul(int **A, int **B, int **C, int size)
{
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
        {
            int j = 0;
            __m256i ra = _mm256_set1_epi32(A[i][k]);
            for (; j <= size - 8; j += 8)
            {
                *(__m256i *)(C[i] + j) = _mm256_add_epi32(*(__m256i *)(C[i] + j), _mm256_mullo_epi32(*(__m256i *)(B[k] + j), ra));
            }
            for (; j < size; j++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

void initializeMatrix(int **matrix, int size, int value)
{
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            matrix[i][j] = value;
}

int main()
{
    int size = 1024; // 矩阵的行数和列数

    // A, B, C1, C2 的矩阵大小都是 1024*1024
    int **A = (int **)malloc(size * sizeof(int *));
    int **B = (int **)malloc(size * sizeof(int *));
    int **C1 = (int **)malloc(size * sizeof(int *));
    int **C2 = (int **)malloc(size * sizeof(int *));
    for (int i = 0; i < size; i++)
    {
        A[i] = (int *)aligned_alloc(32, size * sizeof(int));
        B[i] = (int *)aligned_alloc(32, size * sizeof(int));
        C1[i] = (int *)aligned_alloc(32, size * sizeof(int));
        C2[i] = (int *)aligned_alloc(32, size * sizeof(int));
    }

    initializeMatrix(A, size, 1);
    initializeMatrix(B, size, 1);
    initializeMatrix(C1, size, 0);
    initializeMatrix(C2, size, 0);

    clock_t start, end;

    start = clock();
    MatrixMulNormal(A, B, C1, size);
    end = clock();
    printf("Normal matrix multiplication took %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    MatrixMul(A, B, C2, size);
    end = clock();
    printf("Optimized matrix multiplication took %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    // 验证结果是否一致
    int match = 1;
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            if (C1[i][j] != C2[i][j])
            {
                match = 0;
                break;
            }

    if (match)
        printf("Results match!\n");
    else
        printf("Results do not match!\n");

    for (int i = 0; i < size; i++)
    {
        free(A[i]);
        free(B[i]);
        free(C1[i]);
        free(C2[i]);
    }
    free(A);
    free(B);
    free(C1);
    free(C2);

    return 0;
}