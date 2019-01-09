import typing

def returns_nothing(name: str) -> None:
    print("Hello, {}!".format(name))
returns_nothing("cb")

# Note these definitions need to be on separate lines, see https://github.com/python/mypy/issues/4794
Greeting = str
Name = str
def type_alias(name: Name) -> Greeting:
    return "Hello, {}!".format(name)
print(type_alias("cb (again)"))


def any_func(a: int, b: typing.Any) -> None:
    print("I got {} problems but {} ain't one".format(a, b))
any_func(99, "typing")
any_func(12, {"look": "ups"})


T = typing.TypeVar("T") # the argument to TypeVar just needs to match the variable name
# cat = typing.TypeVar("cat")
def generic_func(l: typing.Sequence[T]) -> T:
    return l[0] # We know that regardless of T the sequence supports indexing so this is OK
generic_func([1, 2, 3])


def dict_func(d: typing.Dict[str, int]):
    for k, v in d.items():
        print("We have {} {}".format(v, k))

dict_func({"cats": 2, "dogs": 3})
# dict_func({3: "dogs"}) # this fails static analysis


# Float is a superset of int and so if the type annotated is float, ints are acceptable.
# Similarly if the type is complex both floats and ints are good.
# See the numeric tower https://www.python.org/dev/peps/pep-0484/#id27
def numeric_add(a: complex, b: complex) -> complex:
    return a + b
print(numeric_add(1, 2))

def unions(a: typing.Union[int, typing.List[int]]) -> typing.Union[int, typing.List[int]]:
    if isinstance(a, list):
        return [i*2 for i in a]
    else:
        return a * 2
print(unions(1))
print(unions([1, 2, 3]))


def this_should_never_return() -> typing.NoReturn:
    raise Exception("eek!")


def optional(a: typing.Optional[int]) -> None: # Optional[t1] == Union[t1, None]
    print(a)

optional(1)
optional(None)
