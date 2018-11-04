#include <iostream>

class Random {
    // Until we get to the public section things are private
    int seed;

public:
    // Member initialization https://stackoverflow.com/a/8523361
    Random(int s) : seed(s) {
        std::cout << "Random init with seed " << seed << "\n";
    };
    Random() {std::cout << "Random init without seed\n";};
    // This is a destructor. I guess required to free any memory malloced during
    // the lifespan of the object.
    ~Random();

    void updateSeed(int seed);
    int diceRoll() const; // Func doesn't modify the object so should be marked const
    float random() const;
    // Can also put function definitions here. These will be inlined.
    void promise() const {std::cout << "I promise this is all random\n";};

};
