# Imports

Let's start with the happy case. You have install a package (e.g. numpy) using your package manager (e.g. pip) and now you want to import it. You can do this with:

```
import numpy as np
```

How does this work? How did Python find numpy?


## [Modules](https://docs.python.org/3/tutorial/modules.html#modules)

There might a slightly more complicated explanation somewhere, but basically *a module is a python file*. The name of the module is the filename without the `.py` suffix. A module/file contains definitions and statements.

The simplest possibly module is just an empty file. However, even this has some properties:

```
$ python3 main_module_attrs.py
__cached__ /home/christopher/code/learning/python/imports/__pycache__/my_module.cpython-37.pyc
__doc__ None
__file__ /home/christopher/code/learning/python/imports/my_module.py
__loader__ <_frozen_importlib_external.SourceFileLoader object at 0x7f8d992ef5f8>
__name__ my_module
__package__
__spec__ ModuleSpec(name='my_module', loader=<_frozen_importlib_external.SourceFileLoader object at 0x7f8d992ef5f8>, origin='/home/christopher/code/learning/python/imports/my_module.py')
```

We can define more interesting modules. But how does python find them?

### [The module search path](https://docs.python.org/3/tutorial/modules.html#the-module-search-path)

When a module is imported, the interpreter searches for it in the `sys.path`. This is populated by a couple of things (e.g. the `PYTHONPATH` environment variable) and also includes the directory that the script lives in (*n.b.* not the cwd when the interpreter was invoked! Also note *script*, some lower level module can't import it the same way).

Prove this to yourself by running:

```
$ python3 main1.py
2
$ python3 nested_scripts/main1.py
ModuleNotFoundError: No module named 'lib1'
```

Note that this search path prefers things in the current directory. To see this, run `python3 -c "import pandas; print(pandas)"`

## Packages

We have modules, but these are just python files. You don't want to stuff everything in one big file. Packages are the next level of organization!

In the past, adding an `__init__.py` file to a directory made it into a package. As of [this pep](https://www.python.org/dev/peps/pep-0420/#specification), I don't think that is needed anymore...

While in a package, you can import other things within the package using either relative imports. See `my_collection_of_modules/mod2.py`.

```
from . import mod1 # relative
import my_collection_of_modules.mod1 as mod1_2 # absolute
assert mod1 is mod1_2
```

However, this is only true if that modules is being loaded as part of the package. If it is loaded as `__main__` then it is not part of the package!
This is because we need to know the module's location in the package (`__name__`) to load other things. If we don't have that, we can't do imports.
See [this](https://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time/14132912#14132912)

The summary of this, is that you cannot easily (without hacks) import local modules that are not below the script that is run as main in the file hierarchy.

Why is this so? [Guido doesn't think you should run scripts from within a package](https://www.python.org/dev/peps/pep-3122/#guido-rejection). That does kinda make sense... The place where this breaks for me is I want to have:

```
.
├── visualizations
│   ├── vis1.py
│   ├── vis2.py
├── analysis
│   └── analysis1.py
├── library_code
│   ├── find_xyz.py
│   ├── compute_abc.py
└── README.md
```

basically I want a couple different classes of top level scripts (in their own directories) that all rely on a shared package.

The other failure case is:

```
.
├── lib
│   ├── l1.py
│   ├── l2.py
├── tests
│   ├── t1.py
│   ├── t2.py
└── README.md
```

And I want to run `python3 t1.py` which imports `lib.l1`.

### Hacks

* Run the module as part of the packages `python3 -m package.module`
* Manually change `sys.path` to include the root of the package


### Install method

https://python-packaging.readthedocs.io/en/latest/minimal.html

I actually really like this. Though, you need to be careful about autoreload failures


## Autoreload Failures

Imagine you have this nested hierarchy:

```
.
├── main.ipynb
└── pkg
    ├── __init__.py # from . import t1
    └── t1
        ├── f.py # a bunch of functions
        ├── __init__.py # from .f import *
```

We expect changes to the function in `f` to be autoreloaded in the notebook where we:
```
import pkg
print(pkg.t1.one_of_those_functions_in_f())
```

However, if we define a new function, `t1` doesn't find out about it. The `__init__.py` hasn't changed and so isn't reloaded!
Changes to functions work fine as their code object is just replaced.
But to get the new function, you need to resave the `__init__.py`.

Solution? Don't use wildcard imports. This forces you to edit `__init__.py` when you make a change.
