#include "classes.h"
#include <iostream> // cout etc live in this

// Without this we would have to std::xyz for everything in std
// Things in std include cout
using namespace std;

int main() {
    cout << "Hello world!\n";

    Random r;
    cout << r.diceRoll() << "\n";
    r.promise();

    Random r_w_seed(4);
    cout << r_w_seed.random() << "\n";

    return 0;
}
