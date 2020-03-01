// Compile with:
// g++ hello_world_glut.cpp -o hello_world_glut -lglut -lGL
#include <iostream>
#include <GL/glut.h>

void displayMe(void) {
    std::cout << "Call display func!" << std::endl;
    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_POLYGON);
    // X, Y, Z
    glVertex3f(0.0, 0.0, 0.0);
    glVertex3f(1.0, 0.0, 0.0);
    glVertex3f(1.0, 1.0, 0.0);
    glVertex3f(0.0, 1.0, 0.0);

    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, 'b');
    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, 'c');
    glEnd();
    glFlush();
}

void keyCallback(unsigned char key, int x, int y) {
    std::cout << "Pressed (" << (int)key << ") ";
    std::cout << "at location x:" << x << " ";
    std::cout << "y: " << y << " ";
    std::cout << "from top left" << std::endl;

    if (key == 27) { // ESC
        std::exit(EXIT_SUCCESS);
    }

}


void mouseCallback(int button, int state, int x, int y) {
    std::cout << "Button " << button << " ";
    std::cout << "in state " << state << " ";
    std::cout << "at location x:" << x << " ";
    std::cout << "y: " << y << " ";
    std::cout << "from top left" << std::endl;
}

int main(int argc, char** argv)
{
    // Init - this pulls args from argc/argv. We also explicitly set some things
    glutInit(&argc, argv);
    glutInitWindowSize(800, 800);
    glutInitWindowPosition(100, 100);
    glutInitDisplayMode(GLUT_SINGLE);

    // Create the window, with the title text
    glutCreateWindow("Hello world :D");
    glutFullScreen();


    // Register all the callback functions that GLUT needs
    // This sets the function that is called to display the normal plane
    glutDisplayFunc(displayMe);
    glutKeyboardFunc(keyCallback);
    glutMouseFunc(mouseCallback);

    // This will never return!
    // It will call as necessary any callbacks that have been registered.
    glutMainLoop();
    return 0;
}
