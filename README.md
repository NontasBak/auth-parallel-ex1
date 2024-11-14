# auth-parallel-ex1
1st exercise of class Parallel and Distributed Systems @AUTH

Written by Epaminontas Bakoulas and Maria Sotiria Kostomanolaki on November 2024

|                | Pthreads (1) | Pthreads (2) | OpenMP (1) | OpenMP (2) | OpenCilk (1) | OpenCilk (2) |
|----------------|--------------|--------------|------------|------------|--------------|--------------|
| Threads        | 4            | 4            | 4          | 4          | 4            | 4            |
| Recall         | 25.34%       | 2.22%        | 26.52%     | 2.24%      | 26.54%       | 2.24%        |
| Queries/second | 938          | 11995        | 905        | 12064      | 915          | 11846        |

PC specs: i5-11400F (6 cores, 12 threads), 16GB RAM, Linux Mint

Test data: SIFT-128-euclidean

### How it works
1. Split the data points in *blocks* (ex. 100 blocks, with 10k points each if we have 1M points)
2. *Within* each block, calculate the distances between the points and find the k-NN
3. Find new neighbors by taking a *subset* of the points from 2 blocks (ex. 50%), calculate the new distances and update the k-NN if necessary
4. Do this for all block pairs

> [!NOTE]
> In step 1 the points in each block are chosen **randomly** and not sequential. In step 3 the points in each subset are chosen **randomly**.
