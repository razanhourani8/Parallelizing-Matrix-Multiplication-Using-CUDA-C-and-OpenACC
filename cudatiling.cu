#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void matrixMultiplication(const float *inputMatrixA, const float *inputMatrixB, float *outputMatrixC, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    float result = 0;

    for (int tileIdx = 0; tileIdx < (colsA + TILE_SIZE - 1) / TILE_SIZE; tileIdx++)
    {
        if (row < rowsA && tileIdx * TILE_SIZE + threadIdx.x < colsA)
        {
            A_tile[threadIdx.y][threadIdx.x] = inputMatrixA[row * colsA + tileIdx * TILE_SIZE + threadIdx.x];
        }
        else
        {
            A_tile[threadIdx.y][threadIdx.x] = 0;
        }

        if (col < colsB && tileIdx * TILE_SIZE + threadIdx.y < colsA)
        {
            B_tile[threadIdx.y][threadIdx.x] = inputMatrixB[(tileIdx * TILE_SIZE + threadIdx.y) * colsB + col];
        }
        else
        {
            B_tile[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++)
        {
            result += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < rowsA && col < colsB)
    {
        outputMatrixC[row * colsB + col] = result;
    }
}

void displayMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    // default matrix dimensions
    int rowsA = 10000;
    int colsA = 30000;
    int colsB = 20000;

    // Read matrix dimensions from command line arguments if provided
    if (argc >= 4)
    {
        rowsA = atoi(argv[1]);
        colsA = atoi(argv[2]);
        colsB = atoi(argv[3]);
    }

    // Allocate memory on the host for input and output matrices
    float *matrixA, *matrixB, *matrixC;
    matrixA = (float *)malloc(rowsA * colsA * sizeof(float));
    matrixB = (float *)malloc(colsA * colsB * sizeof(float));
    matrixC = (float *)malloc(rowsA * colsB * sizeof(float));

    // Initialize input matrices with random values
    for (int i = 0; i < rowsA * colsA; i++)
        matrixA[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < colsA * colsB; i++)
        matrixB[i] = rand() / (float)RAND_MAX;

    // Allocate memory on the device for input and output matrices
    float *deviceMatrixA, *deviceMatrixB, *deviceMatrixC;
    cudaMalloc((void **)&deviceMatrixA, rowsA * colsA * sizeof(float));
    cudaMalloc((void **)&deviceMatrixB, colsA * colsB * sizeof(float));
    cudaMalloc((void **)&deviceMatrixC, rowsA * colsB * sizeof(float));

    // Copy input matrices from host to device memory
    cudaMemcpy(deviceMatrixA, matrixA, rowsA * colsA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, matrixB, colsA * colsB * sizeof(float), cudaMemcpyHostToDevice);

    // Set block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);

    // Create and record events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch kernel on the device
    matrixMultiplication<<<gridDim, blockDim>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, rowsA, colsA, colsB);

    // Record and synchronize events
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Execution Time: %f ms\n", elapsedTime);

    // Copy output matrix from device to host memory
    cudaMemcpy(matrixC, deviceMatrixC, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result matrix (uncomment this section to display result matrix)
    // printf("Result Matrix:\n");
    // displayMatrix(matrixC, rowsA, colsB);

    // Free device memory
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    // Free host memory
    free(matrixA);
    free(matrixB);
    free(matrixC);

    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
