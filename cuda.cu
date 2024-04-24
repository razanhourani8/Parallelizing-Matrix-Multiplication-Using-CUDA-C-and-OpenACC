#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix multiplication
__global__ void multiplyMatrices(const float *inputMatrixA, const float *inputMatrixB, float *outputMatrixC, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure that the thread is within the valid matrix dimensions
    if (row < rowsA && col < colsB)
    {
        float result = 0;

        // Perform dot product for the given element of the output matrix
        for (int i = 0; i < colsA; i++)
        {
            result += inputMatrixA[row * colsA + i] * inputMatrixB[i * colsB + col];
        }

        // Save the result to the output matrix
        outputMatrixC[row * colsB + col] = result;
    }
}

// Display a 2D matrix
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
    // Default matrix dimensions
    int rowsA = 10000;
    int colsA = 30000;
    int colsB = 20000;

    // Read matrix dimensions from command line arguments if provided
    if (argc >= 5)
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

    // Set block and grid dimensions for CUDA kernel execution
    dim3 blockDim(32, 32);
    dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, (rowsA + blockDim.y - 1) / blockDim.y);

    // Create and record events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Launch CUDA kernel for matrix multiplication
    multiplyMatrices<<<gridDim, blockDim>>>(deviceMatrixA, deviceMatrixB, deviceMatrixC, rowsA, colsA, colsB);

    // Stop and destroy timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time and print it
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Execution Time: %f ms\n", elapsedTime);

    // Copy output matrix from device to host memory
    cudaMemcpy(matrixC, deviceMatrixC, rowsA * colsB * sizeof(float), cudaMemcpyDeviceToHost);

    // Display the result matrix
    printf("Result Matrix:\n");
    displayMatrix(matrixC, rowsA, colsB);

    // Free device memory
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixC);

    // Free host memory
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
