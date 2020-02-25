COMPILER_OPTIONS := -Wall -Wextra -x c++
IMAGEMAGICK_OPTIONS := `/usr/lib/x86_64-linux-gnu/ImageMagick-6.9.10/bin-q16/Magick++-config --cppflags --cxxflags --ldflags --libs`
CUDA_COMPILER_OPTIONS := $(foreach option, $(COMPILER_OPTIONS), --compiler-options $(option))

julia: *.cpp
	clang++ ${COMPILER_OPTIONS} ${IMAGEMAGICK_OPTIONS} julia.cpp file_output.cpp -o julia
