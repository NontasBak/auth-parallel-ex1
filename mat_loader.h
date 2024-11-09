#ifndef MAT_LOADER_H
#define MAT_LOADER_H

#include <matio.h>

int loadMatFile(const char *filename, const char *varname, double *C, int d, int m, const char *datatype);

#endif