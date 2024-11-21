#ifndef KNN
#define KNN

typedef struct {
    double distance;
    int index;
} Neighbor;

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
 * @param nearestNeighbors Array of nearest neighbors
 * @param numBlocks Number of blocks to split the points
 * @param subBlockRatio Size of each sub block when comparing blocks with each other (0, 1]
 */
void kNN(double *C, int n, int d, int k, Neighbor *nearestNeighbors, int numBlocks, float subBlockRatio);

/**
 * @brief Compute the distances between corpus and query points
 * 
 * @param C Corpus data points [m by d]
 * @param Q Query data points [n by d]
 * @param D Distance matrix [m by n]
 * @param m Number of corpus points
 * @param n Number of query points
 * @param d Number of dimensions
 */
void computeDistances(const double *C, const double *Q, double *D, int m, int n, int d);

/**
 * @brief Swap two neighbors in an array
 * 
 * @param arr Array of neighbors
 * @param i Index of the first neighbor
 * @param j Index of the second neighbor
 */
void swap(Neighbor *arr, int i, int j);

/**
 * @brief Partition the array for quickselect
 * 
 * @param arr Array of neighbors
 * @param left Left index
 * @param right Right index
 * @return int Pivot index
 */
int partition(Neighbor *arr, int left, int right);

/**
 * @brief Perform quickselect to find the k nearest neighbors
 * 
 * @param arr Array of neighbors
 * @param left Left index
 * @param right Right index
 * @param k number of nearest neighbors to find
 */
void quickSelect(Neighbor *arr, int left, int right, int k);

/**
 * @brief Select random points from a block, used when creating sub blocks to compare block pairs
 * 
 * @param indices Array of indices
 * @param blockSize Size of the block
 * @param sampleSize Size of the sample (sampleSize <= blockSize)
 */
void selectRandomPoints(int *indices, int blockSize, int sampleSize);

/**
 * @brief Update the k-nearest neighbors
 * 
 * @param neighbors Array of neighbors
 * @param nearestNeighbors Array of nearest neighbors
 * @param globalIndex Global index of the point (due to shuffling)
 * @param k Number of neighbors
 */
void updateKNearestNeighbors(Neighbor *neighbors, Neighbor *nearestNeighbors, int globalIndex, int k);

/**
 * @brief Shuffle the indices array
 * 
 * @param indices Array of indices
 * @param size Size of the array
 */
void shuffleIndices(int *indices, int size);

/**
 * @brief Process distances within a block
 * 
 * @param D Distance matrix
 * @param shuffledIndices Shuffled indices
 * @param blockSize Size of the block
 * @param block Block index
 * @param k Number of neighbors
 * @param nearestNeighbors Array of nearest neighbors
 */
void processBlockDistances(const double *D, int *shuffledIndices, int blockSize, int block, int k, Neighbor *nearestNeighbors);

/**
 * @brief Process distances between two blocks
 * 
 * @param D Distance matrix
 * @param shuffledIndices Shuffled indices
 * @param blockSize Size of the block
 * @param sampleSize Size of the sample
 * @param block1 First block index
 * @param block2 Second block index
 * @param k Number of neighbors
 * @param nearestNeighbors Array of nearest neighbors
 * @param indices1 Points to select from the 1st block
 * @param indices2 Points to select from the 2nd block
 */
void processBlockPairDistances(const double *D, int *shuffledIndices, int blockSize, int sampleSize, int block1, int block2, int k, Neighbor *nearestNeighbors, int *indices1, int *indices2);

#endif