# Visitor Pattern

See http://craftinginterpreters.com/representing-code.html and https://en.wikipedia.org/wiki/Visitor_pattern.

In a OOP codebase it is easy to add a new class, but hard to add a new method to all members of a family of classes. Doing the latter requires opening up and modifying all those currently implemented classes.

In functional programming languages (which I don't know) the opposite is hard. Adding a new function is easy - you just write a new function that works for the required classes! Adding a new classes requires that you modify all previous functions so that they accept that new data type.

The visitor framework makes adding a new method to a family of classes in an OOP framework easy.

On each class simply add a method (by convention called `accept`) that looks like:

```
class Dog:
    def accept(self, visitor):
        visitor.visitDog(self)

class Cat:
    def accept(self, visitor):
        visitor.visitCat(self)
```

Then, when you want to add a method to each of these classes, create a visitor class:

```
class AnimalSpeakingVisitor:
    def visitDog(self, element):
        print("Woof")

    def visitCat(self, element):
        print("Meow")
```

Then

```
cat = Cat()
cat.accept(AnimalSpeakingVisitor()) #  prints "Meow"
```
