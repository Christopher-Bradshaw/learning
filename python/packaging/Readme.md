# Packaging

[WIP]

Or, I've written a library. How do I make it easy for others to use?

## Project structure

Without going into too much detail, most projects (e.g. [numpy](https://github.com/numpy/numpy), [halotools](https://github.com/astropy/halotools)) have a structure that looks something like:

```
. # the root of the project is probably also called my_library which might seem weird
├── my_library
│   ├── module1
│   │   ├── __init__.py
│   │   ├── code.py
│   │   ├── code_test.py
│   ├── module2
│   │   ├── etc
├── README.md
├── <a bunch of test/version control/changelog stuff>
└── setup.py
```

Hopefully the rest of this doc will make clear why this structure is good. The most important file we will focus on is the `setup.py`.

## Setuptools

According to their [docs](https://setuptools.readthedocs.io/en/latest/setuptools.html) (which are actually very clear),

*Setuptools is a collection of enhancements to the Python distutils that allow developers to more easily build and distribute Python packages, especially ones that have dependencies on other packages.*

I have never used distutils, but is seems like setuptools is designed to be a nice API over them. It takes some code (e.g. a library) and, given some extra information (metadata, dependencies, etc), can install the library/build a distributable version (i.e. make it easy to use).

The extra information to do this lives in `setup.py`.

### setup.py

The simplest setup.py looks like this,

```
from setuptools import setup

setup(
        name="my_library",
        version="0.0.1",
        packages=["my_library"],
)
```

We can use this simple setup.py to install our library with just,

```
python3 setup.py install
```

This will install the library in your site-packages, and it can now be imported in the same way as any other library `import my_library`.


#### Dependencies

If your library requires other code (it probably does...) you can make that clear in the setup using an `install_requires` argument to setup, e.g.

```
setup(
    ...
    install_requires=["numpy", "scipy>=1.2"],
    ...
)
```

There are a couple of other `*_requires` options,
* `python_requires` - if the code requires a specific version of python
* `setup_requires` - if running the installer needs some library installed
* `tests_require` - if the tests needs some extra library (e.g. pytest)


### Eggs

As the [docs](https://setuptools.readthedocs.io/en/latest/formats.html) explain, an egg is,

*A directory or zipfile containing the project’s code and resources, along with an EGG-INFO subdirectory that contains the project’s metadata*

When creating an egg, a `.egg-info` directory might also be created which is,

*a file or directory placed adjacent to the project’s code and resources, that directly contains the project’s metadata.*

An egg is basically a self-contained, ready to be distributed, version of the library.

However, eggs are somewhat deprecated, intended to be replaced by wheels.

### Wheel

[Wheels](https://www.python.org/dev/peps/pep-0427/) are a binary package format

## Pip

Pip is a package manger. It uses Setuptools for some underlying functionality, but adds features on top (e.g. package uninstall!)


## Layers of package installation

We have talked about,
* Ways to convert code into packages: distutils, setuptools
* Formats for those packages: egg, wheel
* A package manager (lets you install/uninstall/etc) packages: pip
