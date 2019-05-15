from typeguard import typechecked

# We can add type hints to functions like so
def add(x: int, y: int) -> int:
    return x + y

print(add(1, 2))
# But this doesn't actually enforce them!
print(add("hello ", "world!")) # happily prints 'hello world!'

# So what value do we get?
# Well if we are running mypy, it will flag this for us!
# Either `mypy main.py` or as a editor extension


# If we really want to enforce it, we can using external libraries
# Though there may be a preformance penalty to this
# Also note that mypy doesn't check this!
@typechecked
def minus(x: int, y: int) -> int:
    return x - y

print(minus(2, 1))
try:
    print(minus("a", "b"))
except TypeError as e:
    print(e)
