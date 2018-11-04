# Assembly

In the main compilation readme I included some assembly to show that that was what the object code was made up of. This got me wondering what that meant.

Useful links:
* https://en.wikibooks.org/wiki/X86_Assembly/GAS_Syntax
* http://flint.cs.yale.edu/cs421/papers/x86-asm/asm.html
* https://www.recurse.com/blog/7-understanding-c-by-learning-assembly
* https://en.wikipedia.org/wiki/X86_calling_conventions

## Doubler

A function in C that returns twice its only argument:
```
int doubler(int x) {
    return x * 2;
}
```


Compiles to:
```
   0:   55                      push   %rbp
   1:   48 89 e5                mov    %rsp,%rbp
   4:   89 7d fc                mov    %edi,-0x4(%rbp)
   7:   8b 45 fc                mov    -0x4(%rbp),%eax
   a:   01 c0                   add    %eax,%eax
   c:   5d                      pop    %rbp
   d:   c3                      retq
```

Which means ... well absolutely nothing at the moment.

## Intro to Registers

All of those `%xxx` things are registers. These are data storage locations in the CPU. There are a couple of classes of registers:

1. General purpose 32 bit registers: `E[A,B,C,D]X`. On 64 bit machines it sounds like these are just a subset of a 64 bit register `R[A,B,C,D]X`.
2. Stack pointers: `rbp` which points to the base of the current stack and `rsp` which points to the extent of the current stack. Remember that the stack grows down so rbp > rsp.



## Doubler

With that brief brief into, we can explain doubler

```
# Push what is currently in rbp (the previous frame's base) onto the stack.
   0:   55                      push   %rbp
# Copy what is currently in rsp (the extent of the previous stack) into rbp.
# Set the base of our stack to be the bottom of the previous function's stack.
   1:   48 89 e5                mov    %rsp,%rbp
# Move whatever is contained in %edi to location (%rbp - 4).
# In 0 we pushed what was in rbp onto the stack (which starts at %rbp). I guess
# that was 32 bits because we now push what was in %edi 4 bytes further down.
   4:   89 7d fc                mov    %edi,-0x4(%rbp)
# Move whataver we just put at (%rbp - 4) into %eax
   7:   8b 45 fc                mov    -0x4(%rbp),%eax
# To %eax, add %eax. (the result is stored inplace)
   a:   01 c0                   add    %eax,%eax
# Pop what is currently on the stack (the previous frame's base) into %rbp.
   c:   5d                      pop    %rbp
# Return
   d:   c3                      retq
```

A high level explanation:

* 0: Store the previous stack base pointer on the stack
* 1: Setup our stack base pointer
* 4: Move the arguments into our stack
* 5: Move the arguments into %eax
* a: Double %eax and leave it there
* c: Pop the previous stack base pointer back into %rbp
* d: Return. By the calling convention the return values will be read from %eax.

## Ten x er

```
int ten_xer(int x) {
    return x * 10;
}
```

```
   e:   55                      push   %rbp
   f:   48 89 e5                mov    %rsp,%rbp
  12:   89 7d fc                mov    %edi,-0x4(%rbp)
  15:   8b 55 fc                mov    -0x4(%rbp),%edx
# Pretty standard setup so far. Our argument is in %edx now

# Copy %edx into %eax
  18:   89 d0                   mov    %edx,%eax
# Shift %eax left by 2. Equivalent to multiplying by 4.
  1a:   c1 e0 02                shl    $0x2,%eax

# Add %edx to %eax. %eax is now 5x the input
  1d:   01 d0                   add    %edx,%eax
# Add %eax to itself. It is now 10x the input.
  1f:   01 c0                   add    %eax,%eax
# Cleanup and return
  21:   5d                      pop    %rbp
  22:   c3                      retq
```

## Mult by y

```
static int _mult_by_y(int x, int y) {
    return x * y;
}
```

```
  23:   55                      push   %rbp
  24:   48 89 e5                mov    %rsp,%rbp
  27:   89 7d fc                mov    %edi,-0x4(%rbp)
  2a:   89 75 f8                mov    %esi,-0x8(%rbp)
# Setup and two arguments are on the stack

# Put one arg in %eax
  2d:   8b 45 fc                mov    -0x4(%rbp),%eax
# integer multiply %eax by the other arg
  30:   0f af 45 f8             imul   -0x8(%rbp),%eax
# Cleanup and return
  34:   5d                      pop    %rbp
  35:   c3                      retq
```

