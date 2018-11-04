CC=gcc
CFLAGS=-I.

HEADERS = *.h
SIMPLE_OBJ = simple_main.o simple_lib.o
MATHY_OBJ = mathy_main.o

all: $(SIMPLE_OBJ) $(MATHY_OBJ) custom_math
	$(CC) -o simple $(SIMPLE_OBJ) $(CFLAGS)
	$(CC) -o mathy $(MATHY_OBJ) $(CFLAGS) -lm -Lcustom_math_module -lcustom_math
	# -L/full/path or -L./local/path also work


%.o: %.c $(HEADERS)
	echo ""
	echo $<
	echo ""
	# $(CC) -c -o $@ $< $(CFLAGS)

CMM = custom_math_module/
custom_math:
	$(CC) -c -o $(CMM)m1.o $(CMM)m1.c
	$(CC) -c -o $(CMM)m2.o $(CMM)m2.c
	ar rcs $(CMM)libcustom_math.a $(CMM)m1.o $(CMM)m2.o
	# $(CC) -static -o $(CMM)libcustom_math.a $(CMM)m1.o $(CMM)m2.o


clean:
	rm *.o a.out simple mathy
