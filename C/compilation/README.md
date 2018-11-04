# What is happening when you compile something?


## Compiling a simple library

Look at `simple_lib.[c,h]`. We can compile this with `gcc -c simple_lib.c` to get `simple_lib.o`. But what is in this .o file?

We can `objdump -d simple_lib.o` to get:
```
0000000000000000 <doubler>:
   0:   55                      push   %rbp
   1:   48 89 e5                mov    %rsp,%rbp
   4:   89 7d fc                mov    %edi,-0x4(%rbp)
   7:   8b 45 fc                mov    -0x4(%rbp),%eax
   a:   01 c0                   add    %eax,%eax
   c:   5d                      pop    %rbp
   d:   c3                      retq

000000000000000e <ten_xer>:
   e:   55                      push   %rbp
... omitted ...

0000000000000023 <_mult_by_y>:
... omitted ...
```

Which shows the assembly in the object file. We see the 3 functions, their locations in the file and the assembly code that they have been compiled down to.

We can also `objdump -t simple_lib.o` to just see the symbol table entries:
```
SYMBOL TABLE:
0000000000000000 l    df *ABS*  0000000000000000 simple_lib.c
0000000000000023 l     F .text  0000000000000013 _mult_by_y
0000000000000000 g     F .text  000000000000000e doubler
000000000000000e g     F .text  0000000000000015 ten_xer
ADDRESS         |  FLAGS | Sect | Size       | Name
```
I've deleted a bunch of stuff that seems to be common to all obj files. We see the same addresses and sizes as with -d. The flags F (function), f (file), d (debugging), g (global), l (local) make sense. ABS means absolute, while all others are in the .text section. I'm not fully sure what the section means.

But basically our .c (english like file) has been compiled down to assembly, with some metadata describing the names and types of things defined.

## Compiling a simple main

Using `simple_main.c`, let's start with `objdump -t simple_main.o`.

```
SYMBOL TABLE:
0000000000000000 l    df *ABS*  0000000000000000 simple_main.c
0000000000000000 g     F .text  0000000000000031 main
0000000000000000         *UND*  0000000000000000 doubler
0000000000000000         *UND*  0000000000000000 printf
```

We see a main function and two `*UND*` things. `*UND*` means that they are referenced in this file but not defined here. However, even though they are undefined, the compiler knows that it will be OK once we find them because we included the header file which declares these functions - it knows that we are using them in an OK way. **Header files describe the types/signatures of things in the implementation so that we can compile things piecemeal**.

## Linking

We now have a compiled library, and a compiled main that relys on that library and on the stdlib. We still don't have anything that we can run. To get there we need to link these things together. Run `gcc simple_main.o simple_lib.o`. We now have `a.out` which should run! But what is in it?
```
SYMBOL TABLE:
0000000000000000 l    df *ABS*	0000000000000000              simple_lib.c
0000000000400509 l     F .text	0000000000000013              _mult_by_y
0000000000000000 l    df *ABS*	0000000000000000              simple_main.c
0000000000600e20 l     O .dynamic	0000000000000000              _DYNAMIC
0000000000601000 l     O .got.plt	0000000000000000              _GLOBAL_OFFSET_TABLE_
0000000000000000       F *UND*	0000000000000000              printf@@GLIBC_2.2.5
0000000000000000       F *UND*	0000000000000000              __libc_start_main@@GLIBC_2.2.5
000000000040051c g     F .text	0000000000000031              main
00000000004004f4 g     F .text	0000000000000015              ten_xer
00000000004003c8 g     F .init	0000000000000000              _init
00000000004004e6 g     F .text	000000000000000e              doubler
```

I've cut away a lot, but we can see that our library functions are now defined and have locations. If we run a.out we will also see that, as expected, the variable `doubler` is just a pointer to the location of the assembly that is the doubler function!

Interestingly there is no pruning - even though we don't use `ten_xer` or `_mult_by_y` they are still there. I suspect there is probably some optimization that I could turn on that would do this.

More concerning though, `printf` is still undefined! This is ok though - it will be dynamically linked. Rather than have ever program carry around its own copy of libc, it will find the systems version of it at run time. We can see the dynamic symbol table with `-T`:

```
0000000000000000      DF *UND*	0000000000000000  GLIBC_2.2.5 printf
0000000000000000      DF *UND*	0000000000000000  GLIBC_2.2.5 __libc_start_main
0000000000000000  w   D  *UND*	0000000000000000              __gmon_start__
```

There is printf, waiting to be linked in!

But what if we want a fully independent binary? We can compile with `gcc -static simple_main.o simple_lib.o` (just add -static) though you might need to install something like `glibc-static`. The linker then goes though the statically compiled libraries and puts everything into the binary. This symbol table is huge so I haven't added it here. The full file is 740K vs 8K for the dynamically linked one.

## Making a case for a build tool

So compilation is easy! You just create object files and then link them:
```
$ gcc -c simple_lib.c
$ gcc -c simple_main.c
$ gcc simple_main.o simple_lib.o
```

Actually this could get annoying if you have many .c files. I suppose you could do this in one go:
```
gcc simple_lib.c simple_main.c
```

But this will recompile all .c files, even if only one of them has changed.

What we really need is a tool that allows you to specify the dependency graph, works out what has changed and what needs to be updated and then does the minimum amount of compilation/linking to get that done. Oh and it should have a nice, user friendly command. I wonder if anyone has made something like that?

## Make

http://www.cs.colby.edu/maxwell/courses/tutorials/maketutor/

Generally, each .c file maps to a .o file and so we want a rule like:
```
CC=gcc
CFLAGS=-I.

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)
```

To see the list of automatic variables ($@, $<, etc) see https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html.

This is not perfect because some c files may rely on some header files. You can try and ensure that each .o file has the correct header deps but it is probably easier to do something like:
```
HEADERS=*.h
%.o: %.c ${HEADERS}
    ...
```
and just suck up the increased compilation time when you modify a header.

This gets us all the .o files. If we assume that our final binary is just built from all the .o files we can do:
```
main: *.o
    $(CC) -o $@ $^ $(CFLAGS)
```

## External libraries

So far we have used a local library (simple_lib) and part of the standard library (stdio). We have seen that we need to manually specify local libraries in the linking stage though `gcc -o main main.o local_library.o`. The standard library is by default linked, and so we didn't need to explicitly specify that we were using functions from that.

However, that does not mean that the compiler "knew" that it needed to link this external library. It just linked `libc` by default. It we want to use any other external libraries we need to specify them.

Consider `mathy_main.c` which includes the sqrt function from math.h. It is first worth asking where the preprocessor looks for math.h. You can find this with `gcc -E -Wp,-v -`. On my system it is the reasonable:
```
/usr/lib/gcc/x86_64-redhat-linux/8/include
/usr/local/include
/usr/include
```
For interest sake, math.h is at `/usr/include/math.h`. Having found that and replaced the include with the .h file, the compiler is then happy that we are using sqrt correctly (correct args, correct return value) and so happily lets us compile `mathy_main.o`.

However, if we then try to create the usable binary we get an error `undefined reference to sqrt`. What is going on here?

The compiler knew the function signature of sqrt (because the header file is in the include path), but when trying to build the final binary the compiler can't find a function with the name `sqrt` in the objects it has.
It is as if we wrote the header file but forgot to write the implementation. Or course, we didn't write any of this and the issue is that the linker can't find the implementation. It lives in `/usr/lib64/libm.so` and so we need to tell gcc to include this library. We do this with a `-lm`.

We can do this easily with just `-lm` because `libm.so` is on the search path for libraries. We can see this with:
```
$ gcc mathy_main.o -Wl,--verbose
...
SEARCH_DIR("=/usr/x86_64-redhat-linux/lib64"); SEARCH_DIR("=/usr/lib64"); SEARCH_DIR("=/usr/local/lib64") ; SEARCH_DIR("=/lib64"); SEARCH_DIR("=/usr/x86_64-redhat-linux/lib"); SEARCH_DIR("=/usr/local/lib"); SEARCH_DIR("=/lib"); SEARCH_DIR("=/usr/lib");
```

Sidebar, to find where a file is use:

```
$ locate libm.so
/usr/lib64/libm.so
```

## User built shared libraries

The previous section shows how to use system built `.so` files using `-l<libname>`. But what about user built `.so` files?

Consider the `custom_math_module` that contains a number of different .c files that we want to build into a single large shared object file. The makefile shows how to do this. However, if we want to use these functions in mathy_main it again can't find them - ` undefined reference to m1`. This .so file is not in the local directory, nor in the default search path.

We could find the include file because we specified its location from the local dir. If we had just called it `custom_math.h` we would need to add to the list of searched include dirs too.

To do this with a static built library we just need to specify the directory with -L and the library name with -l. This will include this in the final built binary.

Doing this with a shared object is a bit more of a schlep so see docs here https://www.cprogramming.com/tutorial/shared-libraries-linux-gcc.html

## An example

Redis (https://github.com/antirez/redis) is a largeish project written in C with a reputation of being well written. They should have a nice Makefile, and I think they do. See https://github.com/antirez/redis/blob/unstable/src/Makefile.

We'll just pull our the interesting features

A convenient way to make all components of redis.
```
all: $(REDIS_SERVER_NAME) $(REDIS_SENTINEL_NAME) $(REDIS_CLI_NAME) $(REDIS_BENCHMARK_NAME) $(REDIS_CHECK_RDB_NAME) $(REDIS_CHECK_AOF_NAME)
	@echo ""
	@echo "Hint: It's a good idea to run 'make test' ;)"
	@echo ""
```

Let's just focus on building the server:
```
REDIS_SERVER_NAME=redis-server
$(REDIS_SERVER_NAME): $(REDIS_SERVER_OBJ)
	$(REDIS_LD) -o $@ $^ ../deps/hiredis/libhiredis.a ../deps/lua/src/liblua.a $(FINAL_LIBS)
```

Where these things are defined:
```
REDIS_SERVER_OBJ=adlist.o quicklist.o ae.o anet.o dict.o server.o sds.o etc...
REDIS_LD=$(QUIET_LINK)$(CC) $(FINAL_LDFLAGS) # Not actually 100% sure where these are defined
FINAL_LIBS=-lm
```

So we have a rule that builds the server `redis-server` from the .o files and some statically built deps. This requires libm so we link that with the FINAL_LIBS. Where do we get the .o files from?

```
REDIS_CC=$(QUIET_CC)$(CC) $(FINAL_CFLAGS)
%.o: %.c .make-prerequisites
	$(REDIS_CC) -c $<

# Prerequisites target
.make-prerequisites:
	@touch $@
```

.make-prerequisites is a file. Whenever we touch one of the deps (e.g. those statically built libraries) we should touch this file. It will then be newer than all the .o files and they will all be recompiled. Looking in the deps Makefile we see that when we recompile the deps we touch this file which is good!

But what about changes to .h files? At the moment it doesn't look like we handle this. However from some tests (touch a .h file and rebuild) we appear to. What is going on?

What is this doing?
```
Makefile.dep:
	-$(REDIS_CC) -MM *.c > Makefile.dep 2> /dev/null || true
```

from the gcc docs:
```
-M  Instead of outputting the result of preprocessing, output a rule suitable for make describing the dependencies of the main source file.  The preprocessor outputs one make rule containing the
    object file name for that source file, a colon, and the names of all the included files, including those coming from -include or -imacros command-line options.
-MM Similar but ignores header files in system dirs.
```

This is fantastically powerful! It builds up the graph of deps to all of the .o files! The output looks something like:
```
$ gcc -MM mathy_main.c
mathy_main.o: mathy_main.c custom_math_module/custom_math.h
```

So actually, there are two places that can build .o files. 1) What we showed above which will catch changes to the pre-requisites. 2) The output of the `gcc -MM` which will catch changes to `.c` and **all** `.h` dependencies!

```
Makefile.dep:
	-$(REDIS_CC) -MM *.c > Makefile.dep 2> /dev/null || true

ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS))))
-include Makefile.dep # This will run all the make commands generated in the deps file!
endif
```
