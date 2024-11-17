# auth-parallel-ex1
1st exercise of class Parallel and Distributed Systems @AUTH

Written by Epameinondas Bakoulas and Maria Sotiria Kostomanolaki on November 2024

| OpenMP         | Test #1 | Test #2 | Test #3 | Test #4 |
|----------------|---------|---------|---------|---------|
| Recall         | 2.24%   | 8.12%   | 26.52%  | 57.19%  |
| Queries/second | 12064   | 2504    | 905     | 460     |

| OpenCilk         | Test #1 | Test #2 | Test #3 | Test #4 |
|----------------|---------|---------|---------|---------|
| Recall         | 2.24%   | 8.09%   | 26.54%  | 57.29%  |
| Queries/second | 11846   | 2542    | 915     | 462     |

| Pthreads         | Test #1 | Test #2 | Test #3 | Test #4 |
|----------------|---------|---------|---------|---------|
| Recall         | 2.22%   | 8.13%   | 25.34%  | 57.36%  |
| Queries/second | 11995   | 2516    | 938     | 446     |

| Sequential         | Test #1 | Test #2 | Test #3 | Test #4 |
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
