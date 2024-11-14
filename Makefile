CC = gcc
CFLAGS = -I. -Wall -O3
OPENCILK_COMPILER = /home/nontas/OpenCilk/bin/clang

# Add knn_pthreads knn_openmp knn_opencilk here when they're implemented
all: knn_sequential knn_openmp knn_pthreads knn_opencilk

knn_sequential.o: src/knn_sequential.c src/knn.h
	$(CC) $(CFLAGS) -o $@ -c $<

knn_sequential: knn_sequential.o
	$(CC) $(CFLAGS) -o $@ $^ -lopenblas -lm

knn_pthreads.o: src/knn_pthreads.c src/knn.h
	$(CC) $(CFLAGS) -o $@ -c $<

knn_pthreads: knn_pthreads.o mat_loader.o
	$(CC) $(CFLAGS) -o $@ $^ -lpthread -lopenblas -lm -lmatio

knn_openmp.o: src/knn_openmp.c src/knn.h
	$(CC) $(CFLAGS) -fopenmp -o $@ -c $<

knn_openmp: knn_openmp.o mat_loader.o
	$(CC) $(CFLAGS) -fopenmp -o $@ $^ -lopenblas -lm -lmatio

knn_opencilk.o: src/knn_opencilk.c src/knn.h
	$(OPENCILK_COMPILER) $(CFLAGS) -fopencilk -o $@ -c $<

knn_opencilk: knn_opencilk.o mat_loader.o
	$(OPENCILK_COMPILER) $(CFLAGS) -fopencilk -o $@ $^ -lopenblas -lm -lmatio

mat_loader.o: utils/mat_loader.c
	$(CC) $(CFLAGS) -o $@ -c $<

# Add knn_pthreads knn_openmp knn_opencilk here when they're implemented
clean:
	rm -f *.o knn_sequential knn_openmp knn_pthreads knn_opencilk

