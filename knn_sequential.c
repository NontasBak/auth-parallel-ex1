#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

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
    int n = 100; // Number of points in Q
    int m = 100; // Number of points in C
    int d = 2;
    int k = 3; // Number of nearest neighbors
    int blockSize = 10; // Size of each block

    // Initialize Q
    double *Q = (double *)malloc(n * d * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            Q[i * d + j] = i + j;
        }
    }

    // Print Q
    printf("\nQ matrix:\n");
    printMatrix(Q, n, d);

    // Allocate memory for nearest neighbors
    Neighbor *nearestNeighbors = (Neighbor *)malloc(n * k * sizeof(Neighbor));

    // Initialize nearestNeighbors with large distances
    for (int i = 0; i < n * k; i++) {
        nearestNeighbors[i].distance = INFINITY;
        nearestNeighbors[i].index = -1;
    }

    // Process blocks of C
    for (int blockStart = 0; blockStart < m; blockStart += blockSize) {
        int currentBlockSize = (blockStart + blockSize > m) ? (m - blockStart) : blockSize;

        // Initialize C block
        double *C = (double *)malloc(currentBlockSize * d * sizeof(double));
        for (int i = 0; i < currentBlockSize; i++) {
            for (int j = 0; j < d; j++) {
                C[i * d + j] = (blockStart + i) + j;
            }
        }

        // Print C block
        printf("\nC block matrix:\n");
        printMatrix(C, currentBlockSize, d);

        // Allocate memory for D block
        double *D = (double *)malloc(currentBlockSize * n * sizeof(double));

        // Compute the distances
        computeDistances(C, Q, D, currentBlockSize, n, d);

        // Print the distances
        printf("\nD block matrix:\n");
        printMatrix(D, currentBlockSize, n);

        // Find the k nearest neighbors for each point in Q
        for (int i = 0; i < n; i++) {
            Neighbor *neighbors = (Neighbor *)malloc(currentBlockSize * sizeof(Neighbor));
            for (int j = 0; j < currentBlockSize; j++) {
                neighbors[j].distance = D[j * n + i];
                neighbors[j].index = blockStart + j;
            }

            quickSelect(neighbors, 0, currentBlockSize - 1, k);

            for (int j = 0; j < k; j++) {
                if (neighbors[j].distance < nearestNeighbors[i * k + j].distance) {
                    nearestNeighbors[i * k + j] = neighbors[j];
                }
            }
            free(neighbors);
        }

        // Free allocated memory for C block and D block
        free(C);
        free(D);
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
    free(Q);
    free(nearestNeighbors);

    return 0;
}

void computeDistances(const double *C, const double *Q, double *D, int m, int n, int d) {
    // Allocate memory for C_squared
    double *C_squared = (double *)malloc(m * sizeof(double));

    // Allocate memory for Q_squared
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

    // Allocate memory for CQ
    double *CQ = (double *)malloc(m * n * sizeof(double));

    // Compute the -2*C*Q_T product using cblas_dgemm
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, d, -2.0, C, d, Q, d, 0.0, CQ, n);

    // Calculate the distances
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