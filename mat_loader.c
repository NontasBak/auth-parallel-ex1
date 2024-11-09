#include "mat_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int loadMatFile(const char *filename, const char *varname, double *C, int d, int m, const char *datatype)
{
    // Open MAT file
    printf("Opening MAT file\n");
    fflush(stdout);
    mat_t *matfp = Mat_Open(filename, MAT_ACC_RDONLY);
    if (matfp == NULL)
    {
        fprintf(stderr, "Error opening MAT file\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }
    printf("MAT file opened successfully\n");
    fflush(stdout);

    // Read data from MAT file
    printf("Reading data from MAT file\n");
    fflush(stdout);
    matvar_t *matvar = Mat_VarRead(matfp, varname);
    if (matvar == NULL)
    {
        fprintf(stderr, "Error reading variable from MAT file\n");
        fflush(stderr);
        Mat_Close(matfp);
        return EXIT_FAILURE;
    }
    printf("Data read successfully from MAT file\n");
    fflush(stdout);

    // Check the dimensions of the data
    if (matvar->rank != 2 || matvar->dims[0] != d || matvar->dims[1] != m)
    {
        fprintf(stderr, "Unexpected dimensions in MAT file\n");
        fflush(stderr);
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return EXIT_FAILURE;
    }
    printf("Data dimensions are correct\n");
    fflush(stdout);

    void *data = matvar->data;
    if (data == NULL)
    {
        fprintf(stderr, "Error accessing data in MAT file\n");
        fflush(stderr);
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return EXIT_FAILURE;
    }
    printf("Data accessed successfully\n");
    fflush(stdout);

    // Initialize C with data from MAT file
    printf("Initializing matrix\n");
    fflush(stdout);
    if (strcmp(datatype, "int") == 0)
    {
        int *intData = (int *)data;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < d; j++)
            {
                C[i * d + j] = (double)intData[i * d + j]; // Accessing data in column-major order and converting to double
            }
        }
    }
    else if (strcmp(datatype, "double") == 0)
    {
        double *doubleData = (double *)data;
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < d; j++)
            {
                C[i * d + j] = doubleData[i * d + j]; // Accessing data in column-major order
            }
        }
    }
    else
    {
        fprintf(stderr, "Unsupported data type\n");
        fflush(stderr);
        Mat_VarFree(matvar);
        Mat_Close(matfp);
        return EXIT_FAILURE;
    }
    printf("Matrix initialized with %d points\n", m);
    fflush(stdout);

    Mat_VarFree(matvar);
    Mat_Close(matfp);

    return EXIT_SUCCESS;
}