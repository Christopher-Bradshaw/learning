// If we remove these guards, parent imports "item",
// then if we import both gparent and parent we get "item" twice.

#ifndef _B_
#define _B_
struct item {
    int member;
};
int gp();
#endif
