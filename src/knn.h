#ifndef KNN
#define KNN

/**
 * @brief Find the exact solution of the k-Nearest-Neighbors problem
 * 
 * @param C Corpus data points [m by d]
 * @param Q Query data points [n by d]
 * @param m Number of corpus points
 * @param n Number of query points 
 * @param d Number of dimensions
 * @param k Number of neighbors
 * @param dist Distance of nearest neighbors [n by k]
 * @param idx Indeces of nearest neighbors [n by k]
 * @param numBlocks Number of blocks to split the points
 */
void kNNsearch(double *C, double *Q, int m, int n, int d, int k, double *dist, int *idx, int numBlocks);

/**
 * @brief Find the approximate solution of the k-Nearest-Neighbors problem for C==Q
 * 
 * @param C Corpus data points == Query data points [n by d]
 * @param n Number of points 
 * @param d Number of dimensions
 * @param k Number of neighbors
 * @param dist Distance of nearest neighbors [n by k]
 * @param idx Indeces of nearest neighbors [n by k]
 * @param numBlocks Number of blocks to split the points
 * @param subBlockRatio Size of each sub block when comparing blocks with each other (0, 1]
 */
void kNN(double *C, int n, int d, int k, double *dist, int *idx, int numBlocks, float subBlockRatio);

#endif