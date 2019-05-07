#include <stdio.h>

// Enums are a way to define constants

// By default they are the integers (starting with 0)
enum {
    ZERO, ONE, TWO,
};

// Starts at zero, then increments by 1 after your definition
enum {
    A, // 0
    B = 3, // 3
    C, // 4
};

// You can also define all of them
enum {
    BACKSPACE = '\b',
    BELL = '\a',
};

int main() {
    printf("%d is 1\n", ONE);

    printf("%d, %d, %d\n", A, B, C);

    // But values from different enums can be equal!
    // Note this is different from e.g. go
    // I think this is because go enums are typed
    printf("%d\n", A == ZERO);
}
