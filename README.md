# auth-parallel-ex1
Parallel implementation of the k-NN algorithm using OpenMP, Pthreads and OpenCilk.

**Parallel Results:**

| OpenMP         | Test #1 | Test #2 | Test #3 | Test #4 |
|----------------|---------|---------|---------|---------|
| Recall         | 2.24%   | 8.12%   | 26.52%  | 57.19%  |
| Queries/second | 12064   | 2504    | 905     | 460     |

| OpenCilk       | Test #1 | Test #2 | Test #3 | Test #4 |
|----------------|---------|---------|---------|---------|
| Recall         | 2.24%   | 8.09%   | 26.54%  | 57.29%  |
| Queries/second | 11846   | 2542    | 915     | 462     |

| Pthreads       | Test #1 | Test #2 | Test #3 | Test #4 |
|----------------|---------|---------|---------|---------|
| Recall         | 2.22%   | 8.13%   | 25.34%  | 57.36%  |
| Queries/second | 11995   | 2516    | 938     | 446     |

**Sequential Results:**

| Sequential     | Test #1 | Test #2 | Test #3 | Test #4 |
|----------------|---------|---------|---------|---------|
| Recall         | 2.23%   | 8.08%   | 26.39%  | 57.21%  |
| Queries/second | 4251    | 897     | 278     | 135     |

PC specs: i5-11400F (6 cores, 12 threads), 16GB RAM, Linux Mint

Test data: SIFT-128-euclidean

All parallel tests run on **4 threads**

### How it works
1. Split the data points in *blocks* (ex. 100 blocks, with 10k points each if we have 1M points)
2. *Within* each block, calculate the distances between the points and find the k-NN
3. Find new neighbors by taking a *subset* of the points from 2 blocks (ex. 50%), calculate the new distances and update the k-NN if necessary
4. Do this for all block pairs

> [!NOTE]
> In step 1 the points in each block are chosen **randomly** and not sequential. In step 3 the points in each subset are chosen **randomly**.

### How to run 

1. Install the necessary dependencies:
```bash
sudo apt-get install libopenblas-dev libmatio-dev
```

2. Move the **sift-128-euclidean.hdf5** file inside the directory `data`
3. Run the Matlab code **knn_data.m** to convert it to a .mat file.
4. Change inside the Makefile the variable `OPENCILK_COMPILER` [Learn more](https://www.opencilk.org/doc/users-guide/install/)
5. Run `make`
6. Run any of the binaries:
```
./knn_sequential
./knn_openmp
./knn_pthreads
CILK_NWORKERS=4 ./knn_opencilk
```
7. Run `make clean` to delete the binaries.

### Modifications
- To change the number of threads (default: 4), change the constant `NUM_THREADS`
- To change the number of blocks (default: 100), go to the `main` function and change the variable `numBlocks`.
- To change the sub-block ratio (default: 0.05 = 5%), change the variable `subBlockRatio`

If your computer is struggling with memory usage, lower the number of threads and increase the number of blocks.

If you want a higher **Recall**, increase the `subBlockRatio` (max = 1).

### About this project
1st exercise of class Parallel and Distributed Systems @AUTH

Written by Epameinondas Bakoulas and Maria Sotiria Kostomanolaki on November 2024
