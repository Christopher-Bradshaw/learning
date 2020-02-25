# OpenGL

OpenGL is an API that describes how to render graphics. However, it is designed to be implemented mostly in hardware.
So, the idea is that your program will send openGL commands to the GPU which has specific hardware to execute them.
OpenGL libraries facilitate this - they provide high level abstractions (e.g. create a window) and then pass the low level requirements off to the GPU to actually do this.

So, there are two levels. The low level openGL, which GPUs power. The high level library, which contains high level abstractions that are useful to programmers.


## Hello world

First, install `freeglut`. This is a C library that implements the openGL API. I'm not really sure...

```
apt install freeglut3-dev
```

Then compile, linking in the new libraries

```
g++ hello_world.cpp -o hello_world -lglut -lGL
```

