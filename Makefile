CC = gcc
CFLAGS = -I. -Wall -O3

MATLAB_ROOT = /usr/local/MATLAB/R2024b
MATLAB_INCLUDES = -I$(MATLAB_ROOT)/extern/include
MATLAB_LIBS = -L$(MATLAB_ROOT)/bin/glnxa64 -lmat -lmx -leng -lmex

SYSTEM_LIBSTDCXX = /usr/lib/x86_64-linux-gnu/libstdc++.so.6

#Add knn_pthreads knn_openmp knn_opencilk here when they're implemented
all: knn_sequential knn_openmp

knn_sequential.o: knn_sequential.c
	$(CC) $(CFLAGS) -o $@ -c $<

knn_sequential: knn_sequential.o
	$(CC) $(CFLAGS) -o $@ $^ -lopenblas -lm

# knn_pthreads.o: knn_pthreads.c
# 	$(CC) $(CFLAGS) -o $@ -c $<

# knn_pthreads: knn_pthreads.o
# 	$(CC) $(CFLAGS) -o $@ $^ -lpthread

knn_openmp.o: knn_openmp.c
	$(CC) $(CFLAGS) $(MATLAB_INCLUDES) -fopenmp -o $@ -c $<

knn_openmp: knn_openmp.o mat_loader.o
	LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$(MATLAB_ROOT)/bin/glnxa64 $(CC) $(CFLAGS) -fopenmp -o $@ $^ -lopenblas -lm -lmatio

mat_loader.o: mat_loader.c
	$(CC) $(CFLAGS) $(MATLAB_INCLUDES) -o $@ -c $<

# knn_opencilk.o: knn_opencilk.c
# 	$(CC) $(CFLAGS) -fcilkplus -o $@ -c $<

# knn_opencilk: knn_opencilk.o
# 	$(CC) $(CFLAGS) -fcilkplus -o $@ $^

#Add knn_pthreads knn_openmp knn_opencilk here when they're implemented
clean:
	rm -f *.o knn_sequential knn_openmp

