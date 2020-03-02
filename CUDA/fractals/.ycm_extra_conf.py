def FlagsForFile(filename, **kwargs):
    t = "c++"
    if filename.endswith(".cu"):
        t = "cuda"
    return {
        "flags": [
            "-x",
            t,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-std=c++11",
            "-I/usr/include/x86_64-linux-gnu/ImageMagick-6/",
            "-I/usr/include/ImageMagick-6/",
            "-DMAGICKCORE_HDRI_ENABLE=0",
            "-DMAGICKCORE_QUANTUM_DEPTH=16",
        ]
    }
