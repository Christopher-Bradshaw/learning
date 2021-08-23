# /usr/bin/env python

import sys
from scanner import Scanner
from parser import Parser


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
        try:
            run(line)
        except Exception as e:
            print(e)


def run(line):
    s = Scanner(line)
    tokens = s.scanTokens()
    print(tokens)

    p = Parser(tokens)
    tree = p.parse()
    print(tree)


def runFile(file):
    pass


if __name__ == "__main__":
    main()
