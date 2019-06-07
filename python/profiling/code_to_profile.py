def fibonacci(n):
    if n < 1:
        raise Exception("You fool")
    if n in (1, 2):
        return 1
    return fibonacci(n-1) + fibonacci(n-2)

if __name__ == "__main__":
    for i in range(1, 30):
        print(fibonacci(i))
