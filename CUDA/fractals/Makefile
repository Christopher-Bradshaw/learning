COMPILER_OPTIONS := -Wall -Wextra -x c++
IMAGEMAGICK_HEADERS := `/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.10/bin-q16/Magick++-config --cppflags --cxxflags`
IMAGEMAGICK_LIBS := `/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.10/bin-q16/Magick++-config --ldflags --libs`
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

julia: *.cpp
	nvcc ${CUDA_COMPILER_OPTIONS} -c julia.cu -o julia.o
	clang++ ${COMPILER_OPTIONS} ${IMAGEMAGICK_HEADERS} -c file_output.cpp
	clang++ ${COMPILER_OPTIONS} -c main.cpp -o main.o
	clang++ ${IMAGEMAGICK_LIBS} -o main main.o julia.o file_output.o -lcudart
