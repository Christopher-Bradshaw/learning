#include "classes.h"
#include <iostream> // cout etc live in this

using namespace std;

Random::~Random() {
    cout << "Bye-bye random\n";
}

int Random::diceRoll() const {
    return 4; // chosen by fair dice roll.
              // guaranteed to be random.
}

// "random" number in [0, 1)
float Random::random() const {
    cout << seed << "\n";
    return 1./seed;
}
