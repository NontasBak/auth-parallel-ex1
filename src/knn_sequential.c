#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>
#include "knn.h"

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
        // Sort the k nearest neighbors
        for (int i = left; i < k; i++) {
            for (int j = i + 1; j <= k; j++) {
                if (arr[i].distance > arr[j].distance) {
                    swap(arr, i, j);
                }
            }
        }
        return;
    } else if (k < pivotIndex) {
        quickSelect(arr, left, pivotIndex - 1, k);
    } else {
        quickSelect(arr, pivotIndex + 1, right, k);
    }
}

void kNNsearch(double *C, double *Q, int m, int n, int d, int k, double *dist, int *idx, int numBlocks) {
    Neighbor *nearestNeighbors = (Neighbor *)malloc(n * k * sizeof(Neighbor));

    // Initialize nearestNeighbors
    for (int i = 0; i < n * k; i++) {
        nearestNeighbors[i].distance = INFINITY;
        nearestNeighbors[i].index = -1;
    }

    // Process blocks of Q
    int blockSize = n / numBlocks;
    for (int blockStart = 0; blockStart < n; blockStart += blockSize) {
        double *D = (double *)malloc(m * blockSize * sizeof(double));

        computeDistances(C, Q + blockStart * d, D, m, blockSize, d);

        // Find the k nearest neighbors for each point in the current block of Q
        for (int i = 0; i < blockSize; i++) {
            Neighbor *neighbors = (Neighbor *)malloc(m * sizeof(Neighbor));
            for (int j = 0; j < m; j++) {
                neighbors[j].distance = D[j * blockSize + i];
                neighbors[j].index = j;
            }

            quickSelect(neighbors, 0, m - 1, k);

            for (int j = 0; j < k; j++) {
                if (neighbors[j].distance < nearestNeighbors[(blockStart + i) * k + j].distance) {
                    nearestNeighbors[(blockStart + i) * k + j] = neighbors[j];
                }
            }
            free(neighbors);
        }
        free(D);
    }

    // Copy results to dist and idx
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            dist[i * k + j] = nearestNeighbors[i * k + j].distance;
            idx[i * k + j] = nearestNeighbors[i * k + j].index;
        }
    }
    free(nearestNeighbors);
}

int main(int argc, char *argv[]) {
    int n = 1000; // Number of points in Q
    int m = 1000; // Number of points in C
    int d = 2;
    int k = 3; // Number of nearest neighbors
    int numBlocks = 10;

    // Initialize Q
    double *Q = (double *)malloc(n * d * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            Q[i * d + j] = i + j;
        }
    }

    // Initialize C
    double *C = (double *)malloc(m * d * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
            C[i * d + j] = i + j;
        }
    }

    double *dist = (double *)malloc(n * k * sizeof(double));
    int *idx = (int *)malloc(n * k * sizeof(int));

    kNNsearch(C, Q, m, n, d, k, dist, idx, numBlocks);

    // Print the nearest neighbors
    printf("\nNearest neighbors matrix:\n");
    for (int i = 0; i < n; i++) {
        printf("Point %d: ", i);
        for (int j = 0; j < k; j++) {
            printf("(%d, %.2f) ", idx[i * k + j], dist[i * k + j]);
        }
        printf("\n");
    }

    free(Q);
    free(C);
    free(dist);
    free(idx);

    return 0;
}