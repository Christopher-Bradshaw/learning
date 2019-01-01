import preconditions
# Another library option. This doesn't allow a list of preconditions (which I like).
# But is does have postconditions.
# from covenant import pre, post


@preconditions.preconditions(
    lambda a: isinstance(a, int),
    lambda a: a > 0,
    lambda b: isinstance(b, int),
    lambda b: b > 0,
)
def addPositiveInts(a, b):
    return a + b

def main():
    print(addPositiveInts(1, 2))

    try:
        addPositiveInts(-1, 2)
    except preconditions.PreconditionError as e:
        print(e)

    try:
        addPositiveInts(1., 2)
    except preconditions.PreconditionError as e:
        print(e)

    print("done!")

if __name__ == "__main__":
    main()
