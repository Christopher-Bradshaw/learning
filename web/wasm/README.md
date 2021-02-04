# WASM

## Compilation

Emscripten is a backend to LLVM that outputs [asm.js](https://en.wikipedia.org/wiki/Asm.js) (which is deprecated and so we don't want to use) or [WebAssembly (wasm)](https://en.wikipedia.org/wiki/WebAssembly). What this means is that, if we have some code that we can compile to LLVM (e.g. C, Rust, etc) emscripten can take that code to wasm.
Pretty much everything I do here assumes you are starting from C++.

Clang also has wasm as a target. I needed to install `lld` to get `wasm-ld` to compile with clang.

However, code often relys on being run on an environment that has certain things. Things like `malloc`. Part of enscripten is to mock out those things. So, while a simple library can be compiled with clang, more complicated things probably require emscripten. When libraries get complicated (e.g. they have external calls) you can't just run a fully standalone wasm file -- you need to JS that goes with it.

From [their website](https://emscripten.org/docs/introducing_emscripten/about_emscripten.html)
> [Emscripten] also emits JavaScript that provides API support to the compiled code.

This is also a nice description explaining why you might need different supporting code for the browser vs node, from [this issue](https://github.com/rustwasm/wasm-bindgen/issues/1627)

> Even though all those platforms [linux, windows, etc] use the x86-64 instruction set, they still need separate versions [compiled] because of the host APIs and OS quirks. WebAssembly is the equivalent of a CPU instruction set (like x86-64). It doesn't try to smooth over APIs.

I'll use both Emscripten and Clang here just because Clang is slightly simpler to use so might be useful for simple projects.

### Emscripten Setup/Docs

Install as in [here](https://emscripten.org/docs/getting_started/downloads.html)

See the arguments to `emcc` [here](https://emscripten.org/docs/tools_reference/emcc.html#emccdoc)

## Examples

Note that for each of these we generally have,

* A c/cpp file
* A js file (if compiled with Emscripten)
* A wasm file
* An html file that uses the js/wasm

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

The main change is that the outputted function name will be mangled. To access it in JS, we need to work out what that mangled name is. See `math_cpp.html`.

### Hello world (C, Emscripten)

Compile with,

```
emcc math.c -o math_c_ems.js -s EXPORTED_FUNCTIONS='["_add"]' -s EXPORTED_RUNTIME_METHODS='["cwrap"]'
```

where we explicity export `add` (no idea why we need the underscore and is not explained in the docs). We also need `cwrap` to use the function in our JS (we could also use `ccall` but lets just use `cwrap` for now).

To run this code, see [here](https://emscripten.org/docs/porting/connecting_cpp_and_javascript/Interacting-with-code.html), but basically we can do

```
<script src="./math_c_ems.js"></script>
<script>
    setTimeout(() => { // Timeout so that the ^ has time to load
            const add = Module.cwrap("add", "number", ["number", "number"]);
            console.log(add(1, 2));
        }, 1000)
</script>
```

Note, that we need to first load the javascript, which then loads the wasm. Only once that is loaded/compiled can we access its functions.


### Hello world (C, Emscripten, bindings)

The above is ... messy. [Embind](https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html#embind) let's us clean things up a bit.

We can define (in the cpp) what we want to export. We can then compile with,

```
emcc --bind -o math_bindings.js math_bindings.cpp
```

This let's us write, in the JS,

```
<script src="./math_bindings.js"></script>
<script>
    setTimeout( () => {
            const { add } = Module;
            console.log(add(1, 2));
        }, 1000)
</script>
```

This is the way to go for pulling in subsets of a big, bad library.

### Big, bad, library

Let's say we want access to something from a big, bad library. Say the GSL.

To start with, let's just get a basic understanding of working with this library. Let's clone and install it.

```
git clone git://git.savannah.gnu.org/gsl.git gsl-install
cd gsl-install
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

Sweet! We have an output we can sanity check against. To compile with emscripten, we need to follow a fairly similar procedure. However, we need to compile everything with emscripten (note that emscripten seems to be 32 bit and so I needed the 32 glibc libraries). So, again,

```
git clone git://git.savannah.gnu.org/gsl.git gsl-lib
cd gsl-lib
./autogen.sh
emconfigure ./configure
emmake make
```

Note that we don't install - we will reference the built object files. To compile,

```
emcc gsl-lib/specfunc/bessel*.o gsl-lib/err/*.o --bind -I./gsl-lib -o gsl_bindings.js gsl_bindings.cpp
```

Working out exactly which object files are needed was a bit of guesswork. But we can now verify that our `gsl_bindings.js` runs as expected in `gsl_bindings.html`. Success!

To extend this to further items in the GSL we just need to,

* Include those functions in `gsl_bindings.cpp`.
* Possibly include the implementations (`*.o` files) in the compile step.

There are probably many other issues that I haven't yet run into. I might cover them when I do!

#### Modularize

We can add (to the compile step)
```
-s MODULARIZE=1 -s 'EXPORT_NAME="gsl"'
```

We can then load our function with,

```
const { bessel_J0 } = await gsl();
```

This is useful if we have multiple libraries!

## Webpack

See [here](https://github.com/webpack/webpack/issues/7352) and [the linked gist](https://gist.github.com/surma/b2705b6cca29357ebea1c9e6e15684cc) for the basic structure I used.

We need to use the modularized emscripten (taking note of the export name) and configure webpack to use a `file-loader` on wasm files. Then we can do,


```
import gsl from "./gsl_bindings_modularize.js";
import gslWasm from "./gsl_bindings_modularize.wasm";
const gslModule = await gsl({
    locateFile(path) {
        if(path.endsWith(".wasm")) {
          return gslWasm;
        }
        return path;
    }
});
console.log(gslModule.bessel_J0(5));
```

## Optimizations

We should think about this!

## Arrays

These are a bit complicated. By default, emscripten only knows about [certain basic types](https://emscripten.org/docs/porting/connecting_cpp_and_javascript/embind.html#built-in-type-conversions). Passing these around is fairly simple as they are passed by value. Passing arrays by reference (pointer) brings up questions to memory ownership. I also don't think that wasm has access to the general javascript memory space.
I'm still not sure on all the details here. But here are some things that work.


### Returning an array

If we want to allocate and return a typed array from C++, we can do it like this.

```
#include <emscripten/val.h>

emscripten::val floatArrayGen() {
    size_t bufferLength = 10;
    float *buf = (float *)malloc(bufferLength);

    for (int i = 0; i < bufferLength; ++i) {
        buf[i] = i;
    }

    return emscripten::val(emscripten::typed_memory_view(bufferLength, buf));
}
```

### Passing an array

First we need to construct the array in JS and place it in the WASM heap.

```
// Setup array in javascript
const bytesPerElem = 4;
const arr = new Float32Array([1,2,3]);

// Copy array into the wasm heap. This pointer is the number of *bytes* into the
// correct array the memory will be stored at. I am pretty sure that WASM just has
// a single heap and HEAPF32, HEAP8, HEAP16, etc are just views into this heap.
// So, malloc doesn't know which of these to allocate into. It just knows what
// part of the underlying memory is free.
const bufferPtr = gsl._malloc(arr.length * bytesPerElem);
gsl.HEAPF32.set(arr, bufferPtr / bytesPerElem);

// Sanity check that it is there!
console.log(gsl.HEAPF32.slice(bufferPtr / bytesPerElem, arr.length + bufferPtr / bytesPerElem));

// Call a function
useFloatArray(bufferPtr, bufferPtr + arr.length);
```

Then we need to access this memory in the C++.

```
void useFloatArray(int bufferPtr, int bufferSize) {
    // This was total black magic when I first saw it. But it actually makes sense.
    // bufferPtr is, in JS land, an integer index of the number of bytes into the
    // heap our data is stored. We can't pass this as a float* for reasons I don't
    // fully understand, but we get an error. However, here we reinterpret that
    // integer as a float*! We also pass the bufferSize, else we would have no idea
    // how much memory we have allocated!
    float *buf = reinterpret_cast<float *>(bufferPtr);

    // Do stuff with buf

    // We could return a typed array here as shown previously. Or we can do that
    // once we are back in the JS
}
```

### Shared vector

I don't understand this! It was in some github issue.
```
// Creates a vec whose memory looks at the input arr? I think?
// https://github.com/emscripten-core/emscripten/issues/5519
template<typename T>
void typedArrayToVector(const emscripten::val arr, std::vector<T> &vec) {
    unsigned int length = arr["length"].as<unsigned int>();
    emscripten::val memory = emscripten::val::module_property("buffer");

    vec.reserve(length);

    emscripten::val memoryView = arr["constructor"].new_(memory, reinterpret_cast<uintptr_t>(vec.data()), length);

    memoryView.call<void>("set", arr);
}
```
