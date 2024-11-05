#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

void computeDistances(const double *C, const double *Q, double *D, int m, int n, int d);
void printMatrix(const double *A, int m, int n);
void printMatrixInt(const int *A, int m, int n);
void quickSelect(double *arr, int *indices, int left, int right, int k);
int partition(double *arr, int *indices, int left, int right);
void swap(double *arr, int *indices, int i, int j);

int main(int argc, char *argv[]) {
    int n = 5; // Number of points in Q
    int m = 10; // Number of points in C
    int d = 2;
    int k = 3; // Number of nearest neighbors

    // Initialize C
    double *C = (double *)malloc(m * d * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
            C[i * d + j] = i + j;
        }
    }

    // Print C
    printf("C matrix:\n");
    printMatrix(C, m, d);

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

    // Allocate memory for D
    double *D = (double *)malloc(m * n * sizeof(double));

    // Compute the distances
    computeDistances(C, Q, D, m, n, d);

    // Print the distances
    printf("\nD matrix:\n");
    printMatrix(D, m, n);

    // Allocate memory for nearest neighbors
    int *nearestNeighbors = (int *)malloc(n * k * sizeof(int));

    // Find the k nearest neighbors for each point in Q
    for (int i = 0; i < n; i++) {
        double *distances = (double *)malloc(m * sizeof(double));
        int *indices = (int *)malloc(m * sizeof(int));
        for (int j = 0; j < m; j++) {
            distances[j] = D[j * n + i];
            indices[j] = j;
        }

        // // Print distances and indices before quick select
        // printf("\nDistances before quick select for point %d:\n", i);
        // printMatrix(distances, 1, m);
        // printf("Indices before quick select for point %d:\n", i);
        // printMatrixInt(indices, 1, m);

        quickSelect(distances, indices, 0, m - 1, k);

        // // Print distances and indices after quick select
        // printf("\nDistances after quick select for point %d:\n", i);
        // printMatrix(distances, 1, m);
        // printf("Indices after quick select for point %d:\n", i);
        // printMatrixInt(indices, 1, m);

        for (int j = 0; j < k; j++) {
            nearestNeighbors[i * k + j] = indices[j];
        }
        free(distances);
        free(indices);
    }

    // Print the nearest neighbors
    printf("\nNearest neighbors matrix:\n");
    printMatrixInt(nearestNeighbors, n, k);

    // Free allocated memory
    free(C);
    free(Q);
    free(D);
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

void quickSelect(double *arr, int *indices, int left, int right, int k) {
    if (left == right) return;

    int pivotIndex = partition(arr, indices, left, right);

    if (k == pivotIndex) {
        return;
    } else if (k < pivotIndex) {
        quickSelect(arr, indices, left, pivotIndex - 1, k);
    } else {
        quickSelect(arr, indices, pivotIndex + 1, right, k);
    }
}

int partition(double *arr, int *indices, int left, int right) {
    double pivot = arr[right];
    int pivotIndex = left;
    for (int i = left; i < right; i++) {
        if (arr[i] < pivot) {
            swap(arr, indices, i, pivotIndex);
            pivotIndex++;
        }
    }
    swap(arr, indices, right, pivotIndex);
    return pivotIndex;
}

void swap(double *arr, int *indices, int i, int j) {
    double temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;

    int tempIndex = indices[i];
    indices[i] = indices[j];
    indices[j] = tempIndex;
}