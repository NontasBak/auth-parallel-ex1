#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <matio.h>
#include "../utils/mat_loader.h"
#include "knn.h"

typedef struct {
    double distance;
    int index;
} Neighbor;

void computeDistances(const double *C, const double *Q, double *D, int m, int n, int d);
void printMatrix(const double *A, int m, int n);
void printMatrixInt(const int *A, int m, int n);
void quickSelect(Neighbor *arr, int left, int right, int k);
int partition(Neighbor *arr, int left, int right);
void swap(Neighbor *arr, int i, int j);
void selectRandomPoints(int *indices, int blockSize, int sampleSize);
void updateKNearestNeighbors(Neighbor *neighbors, Neighbor *nearestNeighbors, int globalIndex, int k);
void shuffleIndices(int *indices, int size);

void kNN(double *C, int n, int d, int k, double *dist, int *idx, int numBlocks, float subBlockRatio) {
    srand(time(NULL));

    Neighbor *nearestNeighbors = (Neighbor *)malloc(n * k * sizeof(Neighbor));

    // Initialize nearestNeighbors
    for (int i = 0; i < n * k; i++) {
        nearestNeighbors[i].distance = INFINITY;
        nearestNeighbors[i].index = -1;
    }

    // Shuffle indices
    int *shuffledIndices = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) shuffledIndices[i] = i;
    shuffleIndices(shuffledIndices, n);

    // Process blocks of C and Q
    int blockSize = n / numBlocks;
    #pragma omp parallel for
    for (int block = 0; block < numBlocks; block++) {
        double *D = (double *)malloc(blockSize * blockSize * sizeof(double));

        // Allocate memory for C_block and use shuffled indices
        double *C_block = (double *)malloc(blockSize * d * sizeof(double));
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < d; j++) {
                C_block[i * d + j] = C[shuffledIndices[block * blockSize + i] * d + j];
            }
        }

        computeDistances(C_block, C_block, D, blockSize, blockSize, d);
        printf("Computed distances for block %d\n", block);

        // Find the k nearest neighbors for each point in the block
        #pragma omp parallel for
        for (int i = 0; i < blockSize; i++) {
            Neighbor *neighbors = (Neighbor *)malloc(blockSize * sizeof(Neighbor));
            for (int j = 0; j < blockSize; j++) {
                neighbors[j].distance = D[i * blockSize + j];
                neighbors[j].index = shuffledIndices[block * blockSize + j];
            }

            quickSelect(neighbors, 0, blockSize - 1, k);

            updateKNearestNeighbors(neighbors, nearestNeighbors, shuffledIndices[block * blockSize + i], k);
            free(neighbors);
        }
        free(D);
        free(C_block);
    }

    // Improve the solution by finding distances between points in different blocks using a percentage of the points
    #pragma omp parallel for collapse(2)
    for (int block1 = 0; block1 < numBlocks; block1++) {
        for (int block2 = block1 + 1; block2 < numBlocks; block2++) {
            int sampleSize = blockSize * subBlockRatio;

            double *D = (double *)malloc(sampleSize * sampleSize * sizeof(double));
            double *C_block1 = (double *)malloc(sampleSize * d * sizeof(double));
            double *C_block2 = (double *)malloc(sampleSize * d * sizeof(double));

            // Select random points for the sample
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

            // Find the k nearest neighbors for each point in the sample of block1
            #pragma omp parallel for
            for (int i = 0; i < sampleSize; i++) {
                Neighbor *neighbors = (Neighbor *)malloc(sampleSize * sizeof(Neighbor));
                for (int j = 0; j < sampleSize; j++) {
                    neighbors[j].distance = D[i * sampleSize + j];
                    neighbors[j].index = shuffledIndices[block2 * blockSize + indices2[j]];
                }

                quickSelect(neighbors, 0, sampleSize - 1, k);

                updateKNearestNeighbors(neighbors, nearestNeighbors, shuffledIndices[block1 * blockSize + indices1[i]], k);
                free(neighbors);
            }

            // Find the k nearest neighbors for each point in the sample of block2
            #pragma omp parallel for
            for (int i = 0; i < sampleSize; i++) {
                Neighbor *neighbors = (Neighbor *)malloc(sampleSize * sizeof(Neighbor));
                for (int j = 0; j < sampleSize; j++) {
                    neighbors[j].distance = D[j * sampleSize + i];
                    neighbors[j].index = shuffledIndices[block1 * blockSize + indices1[j]];
                }

                quickSelect(neighbors, 0, sampleSize - 1, k);

                updateKNearestNeighbors(neighbors, nearestNeighbors, shuffledIndices[block2 * blockSize + indices2[i]], k);
                free(neighbors);
            }

            free(D);
            free(C_block1);
            free(C_block2);
            free(indices1);
            free(indices2);
        }
    }

    // Copy the results to the output arrays
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            dist[i * k + j] = nearestNeighbors[i * k + j].distance;
            idx[i * k + j] = nearestNeighbors[i * k + j].index;
        }
    }

    free(shuffledIndices);
    free(nearestNeighbors);
}

int main(int argc, char *argv[]) {
    int n = 1000000; // Number of points in Q
    int m = 1000000; // Number of points in C
    int d = 128;
    int k = 100; // Number of nearest neighbors
    int numBlocks = 100;
    float subBlockRatio = 0.05;
    int numThreads = 4;

    omp_set_num_threads(numThreads);

    double *C = (double *)malloc(m * d * sizeof(double));
    double *dist = (double *)malloc(n * k * sizeof(double));
    int *idx = (int *)malloc(n * k * sizeof(int));

    // Load .mat file
    loadMatFile("data/train_data.mat", "train_data", C, d, m, "float");
    
    double startTime = omp_get_wtime();

    kNN(C, n, d, k, dist, idx, numBlocks, subBlockRatio);

    double endTime = omp_get_wtime();
    double elapsedTime = endTime - startTime;

    int numTestQueries = 10000; // First 10k queries
    double *groundTruth = (double *)malloc(numTestQueries * k * sizeof(double));
    loadMatFile("data/knn_neighbors.mat", "knn_neighbors", groundTruth, k, numTestQueries, "double");

    int correctNeighbors = 0;
    for (int i = 0; i < numTestQueries; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < k; l++) {
                if (idx[i * k + j] == (int)groundTruth[i * k + l] - 1) {
                    correctNeighbors++;
                    break;
                }
            }
        }
    }
    int totalNeighbors = numTestQueries * k;
    double recall0 = (double)correctNeighbors / totalNeighbors * 100.0;
    double queriesPerSecond = n / elapsedTime;

    printf("\nRecall: %.4f%%\n", recall0);
    printf("Queries per second: %.2f\n", queriesPerSecond);

    free(C);
    free(dist);
    free(idx);
    free(groundTruth);

    return 0;
}

void computeDistances(const double *C, const double *Q, double *D, int m, int n, int d) {
    double *C_squared = (double *)malloc(m * sizeof(double));
    double *Q_squared = (double *)malloc(n * sizeof(double));

    // Calculate C_squared
    #pragma omp parallel for
    for (int i = 0; i < m; i++) {
        C_squared[i] = 0;
        for (int j = 0; j < d; j++) {
            C_squared[i] += C[i * d + j] * C[i * d + j];
        }
    }

    // Calculate Q_squared
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        Q_squared[i] = 0;
        for (int j = 0; j < d; j++) {
            Q_squared[i] += Q[i * d + j] * Q[i * d + j];
        }
    }

    // Allocate memory for CQ
    double *CQ = (double *)malloc(m * n * sizeof(double));

    // Compute the -2*C*Q_T product using cblas_dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2.0, C, d, Q, d, 0.0, CQ, n);

    // Calculate the distances
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            D[i * n + j] = sqrt(C_squared[i] + Q_squared[j] + CQ[i * n + j]);
        }
    }

    // Free allocated memory
    free(C_squared);
    free(Q_squared);
    free(CQ);
}

void printMatrix(const double *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", A[i * n + j]);
        }
        printf("\n");
    }
}

void printMatrixInt(const int *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", A[i * n + j]);
        }
        printf("\n");
    }
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

void swap(Neighbor *arr, int i, int j) {
    Neighbor temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
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