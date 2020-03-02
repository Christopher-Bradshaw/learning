.DEFAULT_GOAL := julia

COMPILER_OPTIONS := -Wall -Wextra -x c++ -g
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option)) -lineinfo

IMAGEMAGICK_HEADERS := `/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.10/bin-q16/Magick++-config --cppflags --cxxflags`
IMAGEMAGICK_LIBS := `/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.10/bin-q16/Magick++-config --ldflags --libs`

julia.o: julia.cu *.hpp
	nvcc ${CUDA_COMPILER_OPTIONS} -c julia.cu -o julia.o

main.o: main.cpp *.hpp
	clang++ ${COMPILER_OPTIONS} -c main.cpp -o main.o
file_output.o: file_output.cpp *.hpp
	clang++ ${COMPILER_OPTIONS} ${IMAGEMAGICK_HEADERS} -c file_output.cpp

julia: julia.o main.o file_output.o
	clang++ ${IMAGEMAGICK_LIBS} -o main main.o julia.o file_output.o -lcudart

