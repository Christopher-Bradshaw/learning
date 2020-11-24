# WASM

## Compilation

Emscripten is a backend to LLVM that outputs [asm.js](https://en.wikipedia.org/wiki/Asm.js) (which is deprecated and so we don't want to use) or [WebAssembly (wasm)](https://en.wikipedia.org/wiki/WebAssembly). What this means is that, if we have some code that we can compile to LLVM (e.g. C, Rust, etc) emscripten can take that code to wasm.

Clang also has wasm as a target. I needed to install `lld` to get `wasm-ld` to compile with clang.

However, code often relys on being run on an environment that has certain things. Things like `malloc`. Part of enscripten is to mock out those things. So, while a simple library can be compiled with clang, more complicated things probably require emscripten. You can't just run a fully standalone wasm file -- you need to JS that goes with it.

From [their website](https://emscripten.org/docs/introducing_emscripten/about_emscripten.html)
```
[Emscripten] also emits JavaScript that provides API support to the compiled code.
```

### Emscripten Setup

Install as in [here](https://emscripten.org/docs/getting_started/downloads.html)


## Examples

### Hello world (C)

Let's compile a little math library, `math.c`. This is done with,

```
clang --target=wasm32 --no-standard-libraries -Wl,--export-all -Wl,--no-entry -o math_c.wasm math.c
```

We can include this in a simple html page, as in `math_c.html`. And it just works!

We can do this a number of ways in webpack, first we could add wasm to a file loader and do exactly the same as in the direct html.

```
import math from "./math.wasm";
const response = await fetch(math);
const bytes = await response.arrayBuffer();
const { instance } = await WebAssembly.instantiate(bytes);
console.log('The answer is: ' + instance.exports.add(1, 2));
```

This is not ideal - it requires another round trip to the server.

We could also use the [wasm-loader](https://github.com/ballercat/wasm-loader). Then we can do,

```
import math from "./math.wasm";
math().then(math => {
    const add = math.instance.exports.add;
    console.log(add(1, 2));
});
```


Maybe a nicer way to do this is,

```
async function getExports(lib_) {
    const lib = await lib_();
    return lib.instance.exports;
}

import math from "./math.wasm";
const { add } = await getExports(math);
console.log(add(1, 2));
```

### Hello world (C++)

It is basically the same as above. Compile with,

```
clang --target=wasm32 --no-standard-libraries -Wl,--export-all -Wl,--no-entry -o math_cpp.wasm math.cpp
```

The main change is that the outputted function name will be mangled. To access it in JS, we need to work out what that mangled name is.


### More complicated library

Let's say we want access to something from a big, bad library. Say the GSL. Say the spherical harmonic functions. Say, [gsl_sf_legendre_Plm](https://www.gnu.org/software/gsl/doc/html/specfunc.html#c.gsl_sf_legendre_Plm).

To start with, we need to compile the library.

```
./autogen.sh
./configure
make
make install
```

We should be able to find the installed library with `locate libgsl.a`. We can now compile our own code that links to this library. See for example `gsl.c` which we can compile and run with,

```
clang -lgsl -lgslcblas -lm gsl.c -o gsl
./gsl
```

We now just need to write our own library that defines functions + links into the gsl. However, we probably can't use clang for this project. I don't know exactly what the GSL is using under the hood, but suspect it has some requirements (libc, syscalls?) that aren't available on the web.
