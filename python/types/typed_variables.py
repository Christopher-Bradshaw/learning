# Python also allows typed variables (not just functions args)
# The original PEP 484 allowed these as comments

# mypy complains about both the definition of b and the addition of str, int
a = 1 # type: int
b = 2 # type: str
print(a + b) 


# But PEP 526 moves these out of comments by adding python syntax
# Same mypy errors as before
c: int = 1
d: str = 2
print(c + d)


# Ok great, this is pretty simple. What about the details?

name: str # note, this is not defined! `print(name)` gives a NameError. From the PEP
# Type annotations should not be confused with variable declarations in statically typed languages

# Why is this annotation without initialization useful? branches
if 1 > 0:
    name = "cb"
else:
    name = "someone else"
