# /usr/bin/env python

import sys
from .scanner import Scanner


def main():
    args = sys.argv[1:]

    if len(args) == 0:
        runPrompt()
    elif len(args) == 1:
        runFile(sys.argv[1])
    else:
        print("Usage: pylox [script]")
        exit(1)


def runPrompt():
    while True:
        print("> ", end="")
        line = input()
        run(line)


def run(line):
    s = Scanner(line)
    tokens = s.scanTokens()

    print(tokens)
    for token in tokens:
        print(token)


def runFile():
    pass


# if __name__ == "__main__":
main()
