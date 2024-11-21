#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include <matio.h>
#include "../utils/mat_loader.h"
#include "knn.h"

#define NUM_THREADS 4

typedef struct {
    double *C;
    int d, k, numBlocks, blockSize;
    int *shuffledIndices;
    Neighbor *nearestNeighbors;
    int startBlock;
    int endBlock;
    int startPair;
    int endPair;
    float subBlockRatio;
} ThreadData;

void computeDistances(const double *C, const double *Q, double *D, int m, int n, int d) {
    double *C_squared = (double *)malloc(m * sizeof(double));
    double *Q_squared = (double *)malloc(n * sizeof(double));

    // Calculate C_squared
    for (int i = 0; i < m; i++) {
        C_squared[i] = 0;
        for (int j = 0; j < d; j++) {
            C_squared[i] += C[i * d + j] * C[i * d + j];
        }
    }

    // Calculate Q_squared
    for (int i = 0; i < n; i++) {
        Q_squared[i] = 0;
        for (int j = 0; j < d; j++) {
            Q_squared[i] += Q[i * d + j] * Q[i * d + j];
        }
    }

    double *CQ = (double *)malloc(m * n * sizeof(double));

    // Compute the -2*C*Q_T product using openblas
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2.0, C, d, Q, d, 0.0, CQ, n);

    // Calculate the distances
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            D[i * n + j] = sqrt(C_squared[i] + Q_squared[j] + CQ[i * n + j]);
        }
    }

    free(C_squared);
    free(Q_squared);
    free(CQ);
}

void swap(Neighbor *arr, int i, int j) {
    Neighbor temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

int partition(Neighbor *arr, int left, int right) {
    double pivot = arr[right].distance;
    int pivotIndex = left;
    for (int i = left; i < right; i++) {
        if (arr[i].distance < pivot) {
            swap(arr, i, pivotIndex);
            pivotIndex++;
        }
    }
    swap(arr, right, pivotIndex);
    return pivotIndex;
}

void quickSelect(Neighbor *arr, int left, int right, int k) {
    if (left == right) return;

    int pivotIndex = partition(arr, left, right);

    if (k == pivotIndex) {
        return;
    } else if (k < pivotIndex) {
        quickSelect(arr, left, pivotIndex - 1, k);
    } else {
        quickSelect(arr, pivotIndex + 1, right, k);
    }
}

void selectRandomPoints(int *indices, int blockSize, int sampleSize) {
    for (int i = 0; i < blockSize; i++) indices[i] = i;
    for (int i = 0; i < sampleSize; i++) {
        int randIndex = i + rand() % (blockSize - i);
        int temp = indices[i];
        indices[i] = indices[randIndex];
        indices[randIndex] = temp;
    }
}

void updateKNearestNeighbors(Neighbor *neighbors, Neighbor *nearestNeighbors, int globalIndex, int k) {
    for (int j = 0; j < k; j++) {
        // Find the position to insert the new neighbor
        int maxIndex = -1;
        double maxDistance = -1.0;
        for (int l = 0; l < k; l++) {
            if (nearestNeighbors[globalIndex * k + l].distance > maxDistance) {
                maxDistance = nearestNeighbors[globalIndex * k + l].distance;
                maxIndex = l;
            }
        }
        // If the new neighbor is closer, replace the farthest neighbor
        if (neighbors[j].distance < maxDistance) {
            nearestNeighbors[globalIndex * k + maxIndex] = neighbors[j];
        }
    }
}

void shuffleIndices(int *indices, int size) {
    for (int i = 0; i < size; i++) {
        int randIndex = i + rand() % (size - i);
        int temp = indices[i];
        indices[i] = indices[randIndex];
        indices[randIndex] = temp;
    }
}

void processBlockDistances(const double *D, int *shuffledIndices, int blockSize, int block, int k, Neighbor *nearestNeighbors) {
    for (int i = 0; i < blockSize; i++) {
        Neighbor *neighbors = (Neighbor *)malloc(blockSize * sizeof(Neighbor));
        int globalIndex = shuffledIndices[block * blockSize + i];
        for (int j = 0; j < blockSize; j++) {
            neighbors[j].distance = D[i * blockSize + j];
            neighbors[j].index = shuffledIndices[block * blockSize + j];
        }

        quickSelect(neighbors, 0, blockSize - 1, k);
        updateKNearestNeighbors(neighbors, nearestNeighbors, globalIndex, k);
        free(neighbors);
    }
}

void processBlockPairDistances(const double *D, int *shuffledIndices, int blockSize, int sampleSize, int block1, int block2, int k, Neighbor *nearestNeighbors, int *indices1, int *indices2) {
    for (int i = 0; i < sampleSize; i++) {
        Neighbor *neighbors1 = (Neighbor *)malloc(sampleSize * sizeof(Neighbor));
        Neighbor *neighbors2 = (Neighbor *)malloc(sampleSize * sizeof(Neighbor));
        int globalIndex1 = shuffledIndices[block1 * blockSize + indices1[i]];
        int globalIndex2 = shuffledIndices[block2 * blockSize + indices2[i]];
        for (int j = 0; j < sampleSize; j++) {
            neighbors1[j].distance = D[i * sampleSize + j];
            neighbors1[j].index = shuffledIndices[block2 * blockSize + indices2[j]];
            neighbors2[j].distance = D[j * sampleSize + i];
            neighbors2[j].index = shuffledIndices[block1 * blockSize + indices1[j]];
        }

        quickSelect(neighbors1, 0, sampleSize - 1, k);
        updateKNearestNeighbors(neighbors1, nearestNeighbors, globalIndex1, k);
        quickSelect(neighbors2, 0, sampleSize - 1, k);
        updateKNearestNeighbors(neighbors2, nearestNeighbors, globalIndex2, k);
        free(neighbors1);
        free(neighbors2);
    }
}

void *processBlocks(void *arg) {
    ThreadData *data = (ThreadData *)arg;
    double *C = data->C;
    int d = data->d;
    int k = data->k;
    int numBlocks = data->numBlocks;
    int blockSize = data->blockSize;
    int *shuffledIndices = data->shuffledIndices;
    Neighbor *nearestNeighbors = data->nearestNeighbors;
    int startBlock = data->startBlock;
    int endBlock = data->endBlock;
    int startPair = data->startPair;
    int endPair = data->endPair;
    float subBlockRatio = data->subBlockRatio;

    // Process the points inside each block
    for (int block = startBlock; block < endBlock; block++) {
        double *D = (double *)malloc(blockSize * blockSize * sizeof(double));
        double *C_block = (double *)malloc(blockSize * d * sizeof(double));

        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < d; j++) {
                C_block[i * d + j] = C[shuffledIndices[block * blockSize + i] * d + j];
            }
        }

        computeDistances(C_block, C_block, D, blockSize, blockSize, d);
        
        printf("Computed distances for block %d\n", block);

        processBlockDistances(D, shuffledIndices, blockSize, block, k, nearestNeighbors);

        free(D);
        free(C_block);
    }

    // Process the points between the blocks in pairs
    int pairIndex = 0;
    for (int block1 = 0; block1 < numBlocks; block1++) {
        for (int block2 = block1 + 1; block2 < numBlocks; block2++) {
            if(pairIndex < startPair) {
                pairIndex++;
                continue;
            }
            if(pairIndex >= endPair) {
                break;
            }
            int sampleSize = blockSize * subBlockRatio;
            double *D = (double *)malloc(sampleSize * sampleSize * sizeof(double));
            double *C_block1 = (double *)malloc(sampleSize * d * sizeof(double));
            double *C_block2 = (double *)malloc(sampleSize * d * sizeof(double));

            int *indices1 = (int *)malloc(blockSize * sizeof(int));
            int *indices2 = (int *)malloc(blockSize * sizeof(int));
            selectRandomPoints(indices1, blockSize, sampleSize);
            selectRandomPoints(indices2, blockSize, sampleSize);

            for (int i = 0; i < sampleSize; i++) {
                for (int j = 0; j < d; j++) {
                    C_block1[i * d + j] = C[shuffledIndices[block1 * blockSize + indices1[i]] * d + j];
                }
            }
            for (int i = 0; i < sampleSize; i++) {
                for (int j = 0; j < d; j++) {
                    C_block2[i * d + j] = C[shuffledIndices[block2 * blockSize + indices2[i]] * d + j];
                }
            }
            computeDistances(C_block1, C_block2, D, sampleSize, sampleSize, d);

            printf("Computed distances for blocks (%d, %d)\n", block1, block2);

            processBlockPairDistances(D, shuffledIndices, blockSize, sampleSize, block1, block2, k, nearestNeighbors, indices1, indices2);

            free(D);
            free(C_block1);
            free(C_block2);
            free(indices1);
            free(indices2);

            pairIndex++;
        }
    }

    return NULL;
}

void kNN(double *C, int n, int d, int k, Neighbor *nearestNeighbors, int numBlocks, float subBlockRatio) {
    srand(time(NULL));

    // Initialize nearestNeighbors
    for (int i = 0; i < n * k; i++) {
        nearestNeighbors[i].distance = INFINITY;
        nearestNeighbors[i].index = -1;
    }

    // Shuffle indices
    int *shuffledIndices = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) shuffledIndices[i] = i;
    shuffleIndices(shuffledIndices, n);

    int blockSize = n / numBlocks;
    pthread_t threads[NUM_THREADS];
    ThreadData threadData[NUM_THREADS];

    // Calculate the number of blocks and pairs each thread has to process
    int blocksPerThread = numBlocks / NUM_THREADS;
    int totalPairs = (numBlocks * (numBlocks - 1)) / 2;
    int pairsPerThread = totalPairs / NUM_THREADS + 1; // Plus 1 so we can round up

    for (int i = 0; i < NUM_THREADS; i++) {
        threadData[i] = (ThreadData){
            C, d, k, numBlocks, blockSize, shuffledIndices, nearestNeighbors,
            i * blocksPerThread,
            (i + 1) * blocksPerThread,
            i * pairsPerThread,
            (i + 1) * pairsPerThread,
            subBlockRatio
        };
        pthread_create(&threads[i], NULL, processBlocks, &threadData[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    free(shuffledIndices);
}

int main(int argc, char *argv[]) {
    int n = 1000000; // Number of points in Q
    int m = 1000000; // Number of points in C
    int d = 128; // Number of dimensions
    int k = 100; // Number of nearest neighbors
    int numBlocks = 100;
    float subBlockRatio = 0.05;

    // Set the environment variable for OpenBLAS
    setenv("OPENBLAS_NUM_THREADS", "4", 1);

    double *C = (double *)malloc(m * d * sizeof(double));
    Neighbor *nearestNeighbors = (Neighbor *)malloc(n * k * sizeof(Neighbor));

    // Load .mat file
    loadMatFile("data/train_data.mat", "train_data", C, d, m, "float");
    
    // Replace omp_get_wtime with standard time functions
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    kNN(C, n, d, k, nearestNeighbors, numBlocks, subBlockRatio);

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsedTime = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    int numTestQueries = 10000; // First 10k queries
    double *groundTruth = (double *)malloc(numTestQueries * k * sizeof(double));
    loadMatFile("data/knn_neighbors.mat", "knn_neighbors", groundTruth, k, numTestQueries, "double");

    int correctNeighbors = 0;
    for (int i = 0; i < numTestQueries; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                if (nearestNeighbors[i * k + j].index == (int)groundTruth[i * k + l] - 1) {
                    correctNeighbors++;
                    break;
                }
            }
        }
    }
    int totalNeighbors = numTestQueries * k;
    double recall = (double)correctNeighbors / totalNeighbors * 100.0;
    double queriesPerSecond = n / elapsedTime;

    printf("\nRecall: %.4f%%\n", recall);
    printf("Queries per second: %.2f\n", queriesPerSecond);

    free(C);
    free(nearestNeighbors);
    free(groundTruth);

    return 0;
}