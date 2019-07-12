.PHONY: clean

main: *.c *.h
	gcc -c c_funcs.c
	gcc -c main.c
	gcc -o main -lm main.o c_funcs.o

build: clean
	python3 setup.py build_ext --inplace

clean:
	-rm -r build
	-rm *.o
	-rm *.so
	-rm wrapper.c
