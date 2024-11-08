#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include <omp.h> // Include OpenMP header

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

int main(int argc, char *argv[]) {
    int n = 10000; // Number of points in Q
    int m = 10000; // Number of points in C
    int d = 2;
    int k = 3; // Number of nearest neighbors
    int numBlocks = 10; // Number of blocks

    // Initialize C and Q (since C == Q)
    double *C = (double *)malloc(m * d * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
            C[i * d + j] = i + j;
        }
    }

    // Print C
    // printf("\nC matrix:\n");
    // printMatrix(C, m, d);

    // Allocate memory for nearest neighbors
    Neighbor *nearestNeighbors = (Neighbor *)malloc(n * k * sizeof(Neighbor));

    // Initialize nearestNeighbors with large distances
    for (int i = 0; i < n * k; i++) {
        nearestNeighbors[i].distance = INFINITY;
        nearestNeighbors[i].index = -1;
    }

    // Process blocks of C and Q
    #pragma omp parallel for collapse(2) // Parallelize the outer loops
    for (int block1 = 0; block1 < numBlocks; block1++) {
        for (int block2 = block1; block2 < numBlocks; block2++) {
            int currentBlockSize1 = (n + numBlocks - 1) / numBlocks;
            int currentBlockSize2 = (n + numBlocks - 1) / numBlocks;

            // Use only 50% of the points in each block
            int sampleSize1 = currentBlockSize1 / 2;
            int sampleSize2 = currentBlockSize2 / 2;

            // Allocate memory for D block
            double *D = (double *)malloc(sampleSize1 * sampleSize2 * sizeof(double));

            // Compute the distances
            computeDistances(C + block1 * currentBlockSize1 * d, C + block2 * currentBlockSize2 * d, D, sampleSize1, sampleSize2, d);

            // Find the k nearest neighbors for each point in Q1 and Q2 blocks
            #pragma omp parallel for // Parallelize the loop over points in Q1 block
            for (int i = 0; i < sampleSize1; i++) {
                Neighbor *neighbors = (Neighbor *)malloc(sampleSize2 * sizeof(Neighbor));
                for (int j = 0; j < sampleSize2; j++) {
                    neighbors[j].distance = D[i * sampleSize2 + j];
                    neighbors[j].index = block2 * currentBlockSize2 + j * 2; // Non-sequential points
                }

                quickSelect(neighbors, 0, sampleSize2 - 1, k);

                for (int j = 0; j < k; j++) {
                    if (neighbors[j].distance < nearestNeighbors[(block1 * currentBlockSize1 + i * 2) * k + j].distance) {
                        nearestNeighbors[(block1 * currentBlockSize1 + i * 2) * k + j] = neighbors[j];
                    }
                }
                free(neighbors);
            }

            #pragma omp parallel for // Parallelize the loop over points in Q2 block
            for (int i = 0; i < sampleSize2; i++) {
                Neighbor *neighbors = (Neighbor *)malloc(sampleSize1 * sizeof(Neighbor));
                for (int j = 0; j < sampleSize1; j++) {
                    neighbors[j].distance = D[j * sampleSize2 + i];
                    neighbors[j].index = block1 * currentBlockSize1 + j * 2; // Non-sequential points
                }

                quickSelect(neighbors, 0, sampleSize1 - 1, k);

                for (int j = 0; j < k; j++) {
                    if (neighbors[j].distance < nearestNeighbors[(block2 * currentBlockSize2 + i * 2) * k + j].distance) {
                        nearestNeighbors[(block2 * currentBlockSize2 + i * 2) * k + j] = neighbors[j];
                    }
                }
                free(neighbors);
            }

            // Free allocated memory for D block
            free(D);
        }
    }

    // Print the nearest neighbors
    printf("\nNearest neighbors matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            printf("(%d, %.2f) ", nearestNeighbors[i * k + j].index, nearestNeighbors[i * k + j].distance);
        }
        printf("\n");
    }

    // Free allocated memory
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