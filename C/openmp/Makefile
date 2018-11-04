CC = gcc
hello: hello.c
	${CC} $^ -o $@ -fopenmp

integrate: integrate.c
	${CC} $^ -o $@ -fopenmp -lm

pi_estimation: pi_estimation.c
	${CC} $^ -o $@ -fopenmp

bad_cache: bad_cache.c
	${CC} $^ -o $@ -fopenmp

.PHONY: clean
clean:
	rm bad_cache integrate pi_estimation hello
