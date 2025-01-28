CXX = /opt/homebrew/Cellar/llvm/19.1.6/bin/clang++
CXXFLAGS = -std=c++23 -O3 -ffast-math

# OpenBLAS comes with LAPACK.
BLAS_LAPACK_INCLUDE = /opt/homebrew/Cellar/openblas/0.3.28/include
BLAS_LAPACK_LIB = /opt/homebrew/Cellar/openblas/0.3.28/lib
GSL_INCLUDE = /opt/homebrew/Cellar/gsl/2.8/include
GSL_LIB = /opt/homebrew/Cellar/gsl/2.8/lib
# OpenBLAS does a bit of internal parallelization, although most of what we do
# is not meaningfully parallelized, we have many small matrix operations.
# Anyway, statically linking OpenBLAS means OpenMP must be manually included.
OMP_LIB = /opt/homebrew/Cellar/libomp/19.1.3/lib

.PHONY: all clean

binaries = test_permutations test_sum test_basic test_MCD

all: $(binaries)

clean:
	rm -f $(binaries)

test_permutations:
	$(CXX) $(CXXFLAGS) -o test_permutations test_permutations.cpp -I$(BLAS_LAPACK_INCLUDE) -I$(GSL_INCLUDE) $(BLAS_LAPACK_LIB)/libopenblas.a $(GSL_LIB)/libgsl.a $(GSL_LIB)/libgslcblas.a -L$(OMP_LIB) -lomp

test_sum:
	$(CXX) $(CXXFLAGS) -o test_sum test_sum.cpp -I$(BLAS_LAPACK_INCLUDE) -I$(GSL_INCLUDE) $(BLAS_LAPACK_LIB)/libopenblas.a $(GSL_LIB)/libgsl.a $(GSL_LIB)/libgslcblas.a -L$(OMP_LIB) -lomp

test_basic:
	$(CXX) $(CXXFLAGS) -o test_basic test_basic.cpp -I$(BLAS_LAPACK_INCLUDE) -I$(GSL_INCLUDE) $(BLAS_LAPACK_LIB)/libopenblas.a $(GSL_LIB)/libgsl.a $(GSL_LIB)/libgslcblas.a -L$(OMP_LIB) -lomp

test_MCD:
	$(CXX) $(CXXFLAGS) -o test_MCD test_MCD.cpp -I$(BLAS_LAPACK_INCLUDE) -I$(GSL_INCLUDE) $(BLAS_LAPACK_LIB)/libopenblas.a $(GSL_LIB)/libgsl.a $(GSL_LIB)/libgslcblas.a -L$(OMP_LIB) -lomp
