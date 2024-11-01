#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <math.h>

void compute_distances(const double *C, const double *Q, double *D, int m, int n, int d);

int main(int argc, char *argv[]) {
    int n = 10;
    int m = 5;
    int d = 2;

    // Initialize C
    double *C = (double *)malloc(m * d * sizeof(double));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
            C[i * d + j] = i + j;
        }
    }

    // Print C
    printf("C matrix:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < d; j++) {
            printf("%.2f ", C[i * d + j]);
        }
        printf("\n");
    }

    // Initialize Q
    double *Q = (double *)malloc(n * d * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            Q[i * d + j] = i + j;
        }
    }

    // Print Q
    printf("\nQ matrix:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            printf("%.2f ", Q[i * d + j]);
        }
        printf("\n");
    }

    // Allocate memory for D
    double *D = (double *)malloc(m * n * sizeof(double));

    // Compute the distances
    compute_distances(C, Q, D, m, n, d);

    // Print the distances
    printf("\nD matrix:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", D[i * n + j]);
        }
        printf("\n");
    }


    // Free allocated memory
    free(C);
    free(Q);
    free(D);

    return 0;
}

void compute_distances(const double *C, const double *Q, double *D, int m, int n, int d) {
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