// g++ hello_world.cpp -o hello_world -lGL -lglfw -lGLEW
#include <iostream>
#include <assert.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#define GLFW_INCLUDE_NONE


void resize_callback(GLFWwindow *window, int width, int height) {
    std::cout << width << " " << height << std::endl;
    glViewport(0, 0, width, height);
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods) {
    // action is GLFW_[PRESS,RELEASE,REPEAT]
    // mods is a bitfield indicating which modifier keys were held
    std::cout << key << " " << scancode << std::endl;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS){
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

int main(){
    int width, height;
    width = 640;
    height = 480;
    // Initialize
    assert(glfwInit());

    // Create the window and set it to be current
    GLFWwindow* window = glfwCreateWindow(width, height, "Title!", NULL, NULL);
    assert(window != NULL);
    glfwMakeContextCurrent(window);
    // Tell glew what window we are in
    glewInit();

    // Other things!
    glfwSwapInterval(1);


    // Register callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetFramebufferSizeCallback(window, resize_callback);

    // Open GL things
    // Set the viewport the same size as the GLFW one. It could be smaller
    glViewport(0, 0, width, height);
    // We want to generate images by manually writing a 2d matrix to a
    // buffer, rather than specifying shapes etc.
    char *init_data = (char *)malloc(width * height * sizeof(char));
    std::cout << init_data << std::endl;

    GLuint pixel_buffer;
    glGenBuffers(1, &pixel_buffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pixel_buffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(char),
            (void *)init_data, GL_DYNAMIC_DRAW_ARB);

    std::cout << pixel_buffer << std::endl << std::flush;
    // We can get the buffer data
    char *pixel_data;
    glGetBufferPointerv(GL_PIXEL_UNPACK_BUFFER_ARB, GL_BUFFER_MAP_POINTER_ARB, (void **)&pixel_data);
    assert(pixel_data != NULL);

    std::cout << pixel_data << std::endl << std::flush;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Do all rendering

        // Set the color and then set the screen to the previously set color
        /* glClearColor(0.2f, 0.3f, 0.3f, 1.0f); */
        /* glClear(GL_COLOR_BUFFER_BIT); */

        // Switch buffers. GLFW does double buffering
        glfwSwapBuffers(window);

        // Wait until an event comes along to re-render
        /* glfwWaitEvents(); */
        // To continuously re-render
        glfwPollEvents();
    }




    // Teardown
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}

