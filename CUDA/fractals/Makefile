COMPILER_OPTIONS := -Wall -Wextra
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

julia: *.cpp *.cu
	clang ${COMPILER_OPTIONS} -c file_output.cpp -o file_output.o
	nvcc ${CUDA_COMPILER_OPTIONS} julia.cu file_output.o -o julia
