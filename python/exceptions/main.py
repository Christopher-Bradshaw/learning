import random


def normal_except():
    try:
        raise ValueError("HAHA")
    except ValueError as e:
        print("ValueError:", e)


def multiple_possible_except_handled_similarly():
    try:
        random_error()
    except Exception as e:
        print("RandomError:", e)


def multiple_possible_except_handled_differently():
    try:
        random_error()
    except ValueError as e:
        print("ValueError:", e)
    except SyntaxError as e:
        print("SyntaxError:", e)


def random_error():
    r = random.random()
    if r < 0.5:
        raise SyntaxError("Yolo")
    else:
        raise ValueError("Haha")


def main():
    normal_except()
    multiple_possible_except_handled_similarly()
    multiple_possible_except_handled_differently()


main()
