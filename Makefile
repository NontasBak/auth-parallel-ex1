CC = gcc
CFLAGS = -I. -Wall

#Add knn_pthreads knn_openmp knn_opencilk here when they're implemented
all: knn_sequential

knn_sequential.o: knn_sequential.c
	$(CC) $(CFLAGS) -o $@ -c $<

knn_sequential: knn_sequential.o
	$(CC) $(CFLAGS) -o $@ $^

# knn_pthreads.o: knn_pthreads.c
# 	$(CC) $(CFLAGS) -o $@ -c $<

# knn_pthreads: knn_pthreads.o
# 	$(CC) $(CFLAGS) -o $@ $^ -lpthread

# knn_openmp.o: knn_openmp.c
# 	$(CC) $(CFLAGS) -fopenmp -o $@ -c $<

# knn_openmp: knn_openmp.o
# 	$(CC) $(CFLAGS) -fopenmp -o $@ $^

# knn_opencilk.o: knn_opencilk.c
# 	$(CC) $(CFLAGS) -fcilkplus -o $@ -c $<

# knn_opencilk: knn_opencilk.o
# 	$(CC) $(CFLAGS) -fcilkplus -o $@ $^

#Add knn_pthreads knn_openmp knn_opencilk here when they're implemented
clean:
	rm -f *.o knn_sequential

