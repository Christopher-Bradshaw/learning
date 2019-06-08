# Packaging

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

See [here](https://docs.python-guide.org/writing/structure/) for some justification of a very similar structure to this. For packaging, we care about how we get `my_library` into something usable. The most important file for that is `setup.py`.

## Setuptools

According to their [docs](https://setuptools.readthedocs.io/en/latest/setuptools.html) (which are actually very clear),

*Setuptools is a collection of enhancements to the Python distutils that allow developers to more easily build and distribute Python packages, especially ones that have dependencies on other packages.*

I have never used distutils (though a quick scan through [the introduction](https://docs.python.org/3/distutils/introduction.html) makes it look pretty similar to setuptools), but is seems like setuptools is designed to be a nice API with some extensions over it. It takes some code (e.g. a library) and, given some extra information (metadata, dependencies, etc), can install the library/build a distributable version (i.e. make it easy to use).

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

#### Extension modules

If your library contains non-python extensions modules (e.g. written in C, C++, cython, etc), these can be compiled with setuptools using the `ext_modules` option. See [the cython docs](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#basic-setup-py) for some example setup.py do to this.

According to the [distutils docs](https://docs.python.org/3.7/distutils/apiref.html), the `ext_modules` takes as an argument a list of `distutils.core.Extension`. In practice, we will use setuptools, which has a `setuptools.extension.Extension` which [subclasses the distutils Extension](https://github.com/pypa/setuptools/blob/d36295aae2d6d2238546309a0dc4043f6e27e78a/setuptools/extension.py#L29-L32).

The distutils/setuptools Extension takes a [number of arguments](https://docs.python.org/3.7/distutils/apiref.html#distutils.core.Extension). Creating the most general extension might look something like,

```
from setuptools.extension import Extension

my_ext = Extension(
    name=my_lib, # The full name of how you want to import the file
    sources=["my_library/my_lib.c"],
    include_dirs= [], # Where to look for C/C++ headers
    library_dirs= [], # Where to look for C/C++ libraries that will be linked in
    libraries= [], # The names (not filenames) of libraries to be linked in
    language="c", # The language that the extension is in (will be detected if not provided)
)
```

My understanding is that setuptools will then pass this off to the appropriate compiler (e.g. gcc) with the provided args, macros, etc to generate the extension module. Note that this doesn't work with cython code - setuptools doesn't know what to do with it. However, the additions are minor.

```
from setuptools.extension import Extension
from Cython.Build import cythonize # Important to do this after the setuptools import, see cython docs

my_ext = cythonize(Extension( ... ), Extension( ... ))
```

[cythonize](https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#cythonize-arguments) compiles the `.pyx` files into `.c` ones, and returns an list of `Extension` objects, modified (e.g. sources renamed to the `.c` compiled version) to be ready for the C compiler.

This does require that users have cython installed, and provides another place that things can go wrong (imagine they have a different version of cython to the one you have tested with). For that reason, you might want to build the `.c` files before creating the distribution, and then have the setup.py just compile those.

#### Other arguments to setup

For a full list of arguments, first look at [the arguments to setup in distutils](https://docs.python.org/3.7/distutils/apiref.html). On top of this, setuptools [adds/modifies some of these](https://setuptools.readthedocs.io/en/latest/setuptools.html#new-and-changed-setup-keywords).

Most of these are just metadata, e.g. author name/email, url for the homepage, descriptions, etc.


#### Other ways to call setup.py

Until now, I have only shown how to install the library into the site-packages (`python3 setup.py install`). There are other things you might want to do,

* Build extensions (`build_ext`): Build all the extension modules (by default in the `build` directory, but optionally inplace with `--inplace`). Useful for testing.
* [Development mode](https://setuptools.readthedocs.io/en/latest/setuptools.html#development-mode) (`develop`): "install" the module in development mode. This builds c/pyx extensions but doesn't move the code to the deployment directory (e.g. site-packages). Instead, it puts `my-library.egg-link` (note dashes rather than underscores) there which links back to the development directory. Any changes (apart from obviously C extensions which need to be rebuilt) are therefore immediately reflected in the "installed" library! Note that this puts an `my_library.egg-info` in your development directory. I'm not 100% sure why, but my guess is that this is needed in the installed directory.
* [Source distribution](https://packaging.python.org/glossary/#term-source-distribution-or-sdist) (`sdist`): Build a source distribution (this is one possible format of libraries on pypi) that can be installed with a tool like pip.
* [Built distribution](https://packaging.python.org/glossary/#term-built-distribution) (`bdist`): Build a built distribution (this is the preferred format of libraries on pypi). This does not need to be installed, only copied to an appropriate location.


### Built distributions

#### Eggs

As the [docs](https://setuptools.readthedocs.io/en/latest/formats.html) explain, an egg is,

*A directory or zipfile containing the project’s code and resources, along with an EGG-INFO subdirectory that contains the project’s metadata*

When creating an egg, a `.egg-info` directory might also be created which is,

*a file or directory placed adjacent to the project’s code and resources, that directly contains the project’s metadata.*

An egg is basically a self-contained, ready to be distributed, version of the library. As a built distribution, it doesn't need any installation, it just needs to be moved to an appropriate location. However, eggs are somewhat deprecated, intended to be replaced by wheels.

#### Wheel

[Wheels](https://pythonwheels.com/) (from [pep-0427](https://www.python.org/dev/peps/pep-0427/)) are a binary package format. For pure python packages they seem to be pretty similar to eggs. However, they can install C extensions without compilation as they include the built C extensions for many platforms (e.g. [numpy](https://pypi.org/project/numpy/#files) provides many different wheels, with a fallback source distribution).

## Package managers

### Pip

Pip is a package manger. It uses Setuptools for some underlying functionality, but adds features on top (e.g. package uninstall). Some examples:

* `pip install <somthing>`: This effectively pulls the package (if it is remote), then likely executes something very similar to `python3 setup.py install`.
* `pip uninstall`: Uninstalls the package. No equivalent in setuptools.
* `pip install -e <some local directory>`: Equivalent to `python3 setup.py develop`

### Conda

I don't use Conda, so this might be subtly wrong but...

The packages in the [Anaconda repo](https://repo.anaconda.com/) are all binaries. Thus, installation is very simple - just download and put in site-packages. Package creation is probably a bit harder (though I am sure there is good tooling) but I don't know how it happens. Conda also apparently does better checks for dependencies, though doesn't have access to as many packages as pip, but these missing packages can be installed by pip.

## Layers of package installation

We have talked about,
* Ways to convert code into distributable packages: distutils, setuptools
* Formats for those packages: egg, wheel
* A package manager (lets you install/uninstall/etc packages from many locations): pip
