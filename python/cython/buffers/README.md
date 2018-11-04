# Numpy and Cython

## Buffer Protocol

https://www.python.org/dev/peps/pep-3118/ defines the "new" buffer protocol (see also https://docs.python.org/3/c-api/buffer.html#bufferobjects). This (as I understand it) is a C-API that defines how python objects can share memory.

Certain objects (e.g. numpy ndarray, byte arrays) support the buffer protocol. This means that you can request a view of their memory through a getbuffer function https://docs.python.org/3/c-api/typeobj.html#buffer-structs, https://github.com/python/cpython/blob/aee5df5e16ec20e94d4315701315c32edae752f5/Include/object.h#L181-L232. They should return a buffer object which is a struct that contains info with a pointer to the start of the memory block + some info about how to access it (item size, stride, total length, etc). *N.B.* this buffer protocol does not copy the data - we just have a view.

The design of this buffer protocol was heavily influenced by numpy (the coauthor of the PEP is the numpy author) so it isn't surprising that this works well with numpy - numpy can import/export buffers. So can things like bytearrays and python arrays (*n.b. not lists!*).

## Memoryviews

The class `memoryview` is the simplest representation of a C buffer at the python level. To create a memoryview, just pass it something that implements the buffer protocol. Let's play with a memoryview *without introducing any numpy/cython complexity* just to learn about it. See `memory_views.py`. This shows us how a memory view is just a description of how to walk through the memory of the original data.

That example was fairly simple though. Let's try something a bit more complicated. See `memory_views_with_numpy.py`. And then a bit more complicated `memory_view_with_structured_numpy.py`. This shows us how a memory view works for 2d numpy arrays and structured arrays.

## Cython typed memoryviews

The memoryviews we have been dealing with are python objects. However they are pretty close to C - they are minimalist (method wise) and are close to the raw data with little overhead. This sounds reasonable for cython and so cython implements a "Typed memoryview" object that is conceptually very similar.

Typed memoryviews have a very similar interface and also implement the buffer protocol. This means that they can be assigned to by anything that also implements the buffer protocol. The cython typed memory view will then point to a view of the data owned by the original object. See `main_typed_memory_views.py` and `typed_memory_views.pyx`. This shows a couple of features:

* `summer` function coverts a bytearray to a typed memoryview (by getting a view of the data) mv and returns the sum
* `mean` takes a c_contiguous, 2d, double array, converts it to a typed memoryview and returns the mean
* `generic_mean` uses a fused type (http://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html) to build a generic mean function. I think how this is implemented is that it creates 5 functions and a wrapper that then tests the type and calls one of them. Grep for `__Pyx_PyObject_to_MemoryviewSlice_ds` to see the checks for different types.
* `minimum` shows how we can define a typed memory view as various slices on a buffer object.
* `doubler_structured_memory_views` shows how we can work with numpy structured arrays


## C pointers

There is another option when dealing with array like things - just use C arrays. Examples of this are in `main_c_pointers.py` and `c_pointers.pyx`. I think you don't generally want to do this:

* C pointers to the stack will go out of scope once the function ends
* Freeing C pointers to the heap is a bit of a schlep (not really, I suppose you could just have the `memory_owner` as a helper class somewhere)

There are some cases where this may be the best option - I don't think there is a good cythonic way to resize memoryviews without dropping out into python e.g. `np.resize`.


## Cython arrays

This is another option that I have seen in the docs but not investigated at all.
