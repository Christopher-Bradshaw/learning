class base:
    def printName(self):
        print(self.name)


class derived1(base):
    def __init__(self, name):
        self.name = name
        self.printName()


def main():
    derived1("name1")


if __name__ == "__main__":
    main()
