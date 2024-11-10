#ifndef MAT_LOADER_H
#define MAT_LOADER_H

#include <matio.h>

/**
 * @brief Load a .mat matrix and store it in an array
 * 
 * @param filename The name of the file (ends with .mat) 
 * @param varname The variable to read from the file
 * @param arr Array to store the data
 * @param d Number of columns
 * @param m Number of rows
 * @param datatype Datatype of the matrix (float or double)
 * @return int 0 if successful, -1 otherwise
 */
int loadMatFile(const char *filename, const char *varname, double *arr, int d, int m, const char *datatype);

/**
 * @brief Save a matrix to a .mat file
 * 
 * @param filename The name of the file (ends with .mat)
 * @param varname The variable to save to the file
 * @param arr Array to save to the file
 * @param d Number of columns
 * @param m Number of rows
 * @return int 0 if successful, -1 otherwise
 */
int saveMatFile(const char *filename, const char *varname, double *arr, int d, int m);

#endif