# OpenGL

OpenGL is an API that describes how to render graphics. However, it is designed to be implemented mostly in hardware.
So, the idea is that your program will send openGL commands to the GPU which has specific hardware to execute them.
OpenGL libraries facilitate this - they provide high level abstractions (e.g. create a window) and then pass the low level requirements off to the GPU to actually do this.

So, there are two levels. The low level openGL, which GPUs power. The high level library, which contains high level abstractions that are useful to programmers.

Check your OpenGL version with,
```
glxinfo | grep version
```

## Related toolkits

See [here](https://www.khronos.org/opengl/wiki/Related_toolkits_and_APIs)

### Context/Window toolkits

OpenGL provides ways to draw to an open window. But doesn't actually let you open a window! To do that there are a couple of libraries. See `freeglut` (the replacement for `glut` which is old) or `GLFW`.

To install on debian, `apt install freeglut3-dev` or `apt install libglfw3-dev`.

I'm going to use GLFW because apparently it is more fully features. I probably won't nede these so this might be silly. But their [docs](https://www.glfw.org/docs/latest/) also look pretty good.

### Loading libraries

IF you want more than just the core GL functionality, the ext are in separate headers/includes. Use `GLEW` to pull in everything that your system is capable of using.

`apt install libglew-dev`


## Open GL

See [docs](https://www.khronos.org/registry/OpenGL-Refpages/gl4/) and the best [tutorial](https://learnopengl.com/)

## What I want

I don't actually care about most of what OpenGL does. I just want to work out how to paint a 2d array onto the screen...

