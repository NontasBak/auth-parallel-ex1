CC = gcc
CFLAGS = -I. -Wall

knn_sequential.o: knn_sequential.c knn.h
	$(CC) $(CFLAGS) -o $@ -c $<

knn_pthreads: knn_pthreads.c knn.h
	$(CC) $(CFLAGS) -o $@ -c $<

knn_openmp: knn_openmp.c knn.h
	$(CC) $(CFLAGS) -fopenmp -o $@ -c $<

knn_opencilk: knn_opencilk.c knn.h
	$(CC) $(CFLAGS) -fcilkplus -o $@ -c $<