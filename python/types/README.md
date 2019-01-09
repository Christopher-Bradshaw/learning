# Typing

See [PEP 483 - mostly theory](https://www.python.org/dev/peps/pep-0483/), [PEP 484](https://www.python.org/dev/peps/pep-0484/), [PRP 526](https://www.python.org/dev/peps/pep-0526/), [PEP 561](https://www.python.org/dev/peps/pep-0561/), [Typing module](https://docs.python.org/3/library/typing.html)

## Some type theory

#### How can we define a type?
* Explicitly enumerate all values - e.g. `True`, `False`.
* Specify functions that can be used on a type - e.g. `len(x)`. Duck Typing.
* Define a class - e.g. `class UserID(int)` then all instances have that type.
* More complicated ways?


#### How do we know if we can assign `v1` of type `t1` to `v2` of type `t2`?

Or said another way, for which values of `t1` and `t2` does `v1 = v2` make sense?

A strong criterion might be that we can only do this if **all** possible values for `v2` are possible values for `v1` (e.g. int and float) and that all functions that can be applied to `t1` can also be applied to `t2`.

This is interesting as the set of values becomes smaller in the subtyping but the set of functions grows.


#### Gradual Typing

http://wphomes.soic.indiana.edu/jsiek/what-is-gradual-typing/

#### Types vs Classes

Every class is a type (e.g. `int`, `UserID`). However there are also types that are not classes (e.g `Union[str, int]`).


## Python Implementation

```
def greeting(name: str) -> str:
    return 'Hello ' + name
```

Which means that this function should be called with a string (or an object whose type is a subtype of str) and will return a string. However, this **is not enforced at runtime** it is there for static analysis tools (e.g. [mypy](https://github.com/python/mypy)) to catch pre-run. So with the types python is still dynamically typed (types need to be valid for the operations you do on them at runtime) but it allows static analysis before too. A useful static-dynamic [primer](https://hackernoon.com/i-finally-understand-static-vs-dynamic-typing-and-you-will-too-ad0c2bd0acc7).

What are the valid options for the types?
* A class (e.g. `int`, `UserID`)
* An alias for a class (e.g. `typing.List`, or user define `Name = str`)
* A special type from the typing module (e.g. `Any`, `Callable`)
