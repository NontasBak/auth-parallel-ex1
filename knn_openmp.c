#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <omp.h> // Include OpenMP header
#include <time.h> // Include time header for random number generation

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

int main(int argc, char *argv[]) {
    int n = 100000; // Number of points in Q
    int m = 100000; // Number of points in C
    int d = 2;
    int k = 3; // Number of nearest neighbors
    int numBlocks = 100; // Number of blocks

    // Initialize random seed
    srand(time(NULL));

    // Initialize C and Q (since C == Q)
    double *C = (double *)malloc(m * d * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
            C[i * d + j] = i + j;
        }
    }

    printf("Initialized C matrix with %d points\n", m);

    // Allocate memory for nearest neighbors
    Neighbor *nearestNeighbors = (Neighbor *)malloc(n * k * sizeof(Neighbor));

    // Initialize nearestNeighbors with large distances
    for (int i = 0; i < n * k; i++) {
        nearestNeighbors[i].distance = INFINITY;
        nearestNeighbors[i].index = -1;
    }

    printf("Initialized nearest neighbors array\n");

    // Shuffle indices
    int *shuffledIndices = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) shuffledIndices[i] = i;
    shuffleIndices(shuffledIndices, n);

    // Process blocks of C and Q
    #pragma omp parallel for // Parallelize the outer loop
    for (int block = 0; block < numBlocks; block++) {
        int currentBlockSize = (n + numBlocks - 1) / numBlocks;

        // Allocate memory for D block
        double *D = (double *)malloc(currentBlockSize * currentBlockSize * sizeof(double));

        // Allocate memory for C_block and use shuffled indices
        double *C_block = (double *)malloc(currentBlockSize * d * sizeof(double));
        for (int i = 0; i < currentBlockSize; i++) {
            for (int j = 0; j < d; j++) {
                C_block[i * d + j] = C[shuffledIndices[block * currentBlockSize + i] * d + j];
            }
        }

        computeDistances(C_block, C_block, D, currentBlockSize, currentBlockSize, d);

        printf("Computed distances for block %d\n", block);

        // Find the k nearest neighbors for each point in the block
        #pragma omp parallel for // Parallelize the loop over points in the block
        for (int i = 0; i < currentBlockSize; i++) {
            Neighbor *neighbors = (Neighbor *)malloc(currentBlockSize * sizeof(Neighbor));
            for (int j = 0; j < currentBlockSize; j++) {
                neighbors[j].distance = D[i * currentBlockSize + j];
                neighbors[j].index = shuffledIndices[block * currentBlockSize + j];
            }

            quickSelect(neighbors, 0, currentBlockSize - 1, k);

            updateKNearestNeighbors(neighbors, nearestNeighbors, shuffledIndices[block * currentBlockSize + i], k);
            free(neighbors);
        }

        // Free allocated memory for D block
        free(D);
        free(C_block);
    }

    // Improve the solution by finding distances between points in different blocks using 50% of the points
    #pragma omp parallel for collapse(2) // Parallelize the outer loops
    for (int block1 = 0; block1 < numBlocks; block1++) {
        for (int block2 = block1 + 1; block2 < numBlocks; block2++) {
            int currentBlockSize1 = (n + numBlocks - 1) / numBlocks;
            int currentBlockSize2 = (n + numBlocks - 1) / numBlocks;

            int sampleSize1 = currentBlockSize1 / 2;
            int sampleSize2 = currentBlockSize2 / 2;

            // Allocate memory for D block
            double *D = (double *)malloc(sampleSize1 * sampleSize2 * sizeof(double));

            // Compute the distances using 50% of the points in each block
            double *C_block1 = (double *)malloc(sampleSize1 * d * sizeof(double));
            double *C_block2 = (double *)malloc(sampleSize2 * d * sizeof(double));

            // Select random points for the sample
            int *indices1 = (int *)malloc(currentBlockSize1 * sizeof(int));
            int *indices2 = (int *)malloc(currentBlockSize2 * sizeof(int));
            selectRandomPoints(indices1, currentBlockSize1, sampleSize1);
            selectRandomPoints(indices2, currentBlockSize2, sampleSize2);

            for (int i = 0; i < sampleSize1; i++) {
                for (int j = 0; j < d; j++) {
                    C_block1[i * d + j] = C[shuffledIndices[block1 * currentBlockSize1 + indices1[i]] * d + j];
                }
            }
            for (int i = 0; i < sampleSize2; i++) {
                for (int j = 0; j < d; j++) {
                    C_block2[i * d + j] = C[shuffledIndices[block2 * currentBlockSize2 + indices2[i]] * d + j];
                }
            }
            computeDistances(C_block1, C_block2, D, sampleSize1, sampleSize2, d);

            printf("Computed distances for blocks (%d, %d)\n", block1, block2);

            // Find the k nearest neighbors for each point in the sample of block1
            #pragma omp parallel for // Parallelize the loop over points in the sample of block1
            for (int i = 0; i < sampleSize1; i++) {
                Neighbor *neighbors = (Neighbor *)malloc(sampleSize2 * sizeof(Neighbor));
                for (int j = 0; j < sampleSize2; j++) {
                    neighbors[j].distance = D[i * sampleSize2 + j];
                    neighbors[j].index = shuffledIndices[block2 * currentBlockSize2 + indices2[j]]; // Non-sequential points
                }

                quickSelect(neighbors, 0, sampleSize2 - 1, k);

                updateKNearestNeighbors(neighbors, nearestNeighbors, shuffledIndices[block1 * currentBlockSize1 + indices1[i]], k);
                free(neighbors);
            }

            // Find the k nearest neighbors for each point in the sample of block2
            #pragma omp parallel for // Parallelize the loop over points in the sample of block2
            for (int i = 0; i < sampleSize2; i++) {
                Neighbor *neighbors = (Neighbor *)malloc(sampleSize1 * sizeof(Neighbor));
                for (int j = 0; j < sampleSize1; j++) {
                    neighbors[j].distance = D[j * sampleSize2 + i];
                    neighbors[j].index = shuffledIndices[block1 * currentBlockSize1 + indices1[j]]; // Non-sequential points
                }

                quickSelect(neighbors, 0, sampleSize1 - 1, k);

                updateKNearestNeighbors(neighbors, nearestNeighbors, shuffledIndices[block2 * currentBlockSize2 + indices2[i]], k);
                free(neighbors);
            }

            // Free allocated memory for D block
            free(D);
            free(C_block1);
            free(C_block2);
            free(indices1);
            free(indices2);
        }
    }

    // Print the nearest neighbors
    printf("\nNearest neighbors matrix:\n");
    for (int i = 0; i < n; i++) {
        printf("Point %d: " , i);
        for (int j = 0; j < k; j++) {
            printf("(%d, %.2f) ", nearestNeighbors[i * k + j].index, nearestNeighbors[i * k + j].distance);
        }
        printf("\n");
    }

    // Free allocated memory
    free(shuffledIndices);
    free(C);
    free(nearestNeighbors);

    return 0;
}

void computeDistances(const double *C, const double *Q, double *D, int m, int n, int d) {
    // Allocate memory for C_squared
    double *C_squared = (double *)malloc(m * sizeof(double));

    // Allocate memory for Q_squared
    double *Q_squared = (double *)malloc(n * sizeof(double));

    // Calculate C_squared
    #pragma omp parallel for // Parallelize the loop over points in C
    for (int i = 0; i < m; i++) {
        C_squared[i] = 0;
        for (int j = 0; j < d; j++) {
            C_squared[i] += C[i * d + j] * C[i * d + j];
        }
    }

    // Calculate Q_squared
    #pragma omp parallel for // Parallelize the loop over points in Q
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
    #pragma omp parallel for collapse(2) // Parallelize the nested loops
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