from dataclasses import dataclass
from typing import Any
from enum import Enum, auto


class TokenType(Enum):
    # Immediately identifiable as single char
    LEFT_PAREN = auto()
    RIGHT_PAREN = auto()
    LEFT_BRACE = auto()
    RIGHT_BRACE = auto()
    COMMA = auto()
    DOT = auto()
    MINUS = auto()
    PLUS = auto()
    SEMICOLON = auto()
    STAR = auto()

    # One or two char
    BANG = auto()
    BANG_EQUAL = auto()
    EQUAL = auto()
    EQUAL_EQUAL = auto()
    LESS_EQUAL = auto()
    LESS = auto()
    GREATER_EQUAL = auto()
    GREATER = auto()
    SLASH = auto()  # This might be a comment which we then strip

    # Literals
    IDENTIFIER = auto()
    STRING = auto()
    NUMBER = auto()

    # Keywords
    AND = auto()
    OR = auto()
    CLASS = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    TRUE = auto()
    FALSE = auto()
    FUN = auto()
    NIL = auto()
    PRINT = auto()
    RETURN = auto()
    SUPER = auto()
    THIS = auto()
    VAR = auto()
    WHILE = auto()

    EOF = auto()


@dataclass
class Token:
    tokenType: TokenType
    lexeme: str
    literal: Any  # str or float
    lineNumber: int

    def __repr__(self):
        return f"({self.tokenType} {self.lexeme} {self.literal})"


class Scanner:
    def __init__(self, source):
        self.source = source
        self.tokens = []

        self.start = 0  # Start of current lexeme
        self.current = 0  # Current char being considered
        # Note that source is just a string (that might contain newlines)
        # So we just keep track of this for error reporting
        self.line = 1

    def scanTokens(self):
        while not self.isAtEnd():
            self.scanNextToken()

        self.addToken(TokenType.EOF)
        return self.tokens

    def scanNextToken(self):
        c = self.getNextChar()
        # breakpoint()

        # Handle single char tokens
        if c == "(":
            self.addToken(TokenType.LEFT_PAREN)
        elif c == ")":
            self.addToken(TokenType.RIGHT_PAREN)
        elif c == "{":
            self.addToken(TokenType.LEFT_BRACE)
        elif c == "{":
            self.addToken(TokenType.RIGHT_BRACE)
        elif c == ",":
            self.addToken(TokenType.COMMA)
        elif c == ".":
            self.addToken(TokenType.DOT)
        elif c == "-":
            self.addToken(TokenType.MINUS)
        elif c == "+":
            self.addToken(TokenType.PLUS)
        elif c == ";":
            self.addToken(TokenType.SEMICOLON)
        elif c == "*":
            self.addToken(TokenType.STAR)
        # Handle ambiguous (one or two char tokens)
        elif c == "!":
            self.addToken(TokenType.BANG_EQUAL if self.match("=") else TokenType.BANG)
        elif c == "=":
            self.addToken(TokenType.EQUAL_EQUAL if self.match("=") else TokenType.EQUAL)
        elif c == "<":
            self.addToken(TokenType.LESS_EQUAL if self.match("=") else TokenType.LESS)
        elif c == ">":
            self.addToken(
                TokenType.GREATER_EQUAL if self.match("=") else TokenType.GREATER
            )
        elif c == "/":
            if self.match("/"):
                while not self.isAtEnd() and self.peekNextChar() != "\n":
                    self.getNextChar()
            else:
                self.addToken(TokenType.SLASH)
        # Handle whitespace
        elif c in (" ", "\t"):
            pass
        elif c == "\n":
            self.line += 1
        # Handle string literals
        elif c == '"':
            self.handleString()
        # Handle number literals
        elif self.isDigit(c):
            self.handleNumber()
        elif self.isAlpha(c):
            self.handleIdentifier()
        else:
            raise UnrecognizedTokenError(self.line)

        # Advance start
        self.start = self.current

    def addToken(self, tokenType, literal=None):
        lexeme = self.source[self.start : self.current]
        self.tokens.append(Token(tokenType, lexeme, literal, self.line))

    def getNextChar(self):
        c = self.source[self.current]
        self.current += 1
        return c

    def peekNextChar(self):
        return self.source[self.current]

    def peek2NextChar(self):
        if self.current + 1 > len(self.source):
            return "\0"
        return self.source[self.current + 1]

    def match(self, expected):
        if self.isAtEnd():
            return False
        elif self.source[self.current] != expected:
            return False
        else:
            self.current += 1
            return True

    def isAtEnd(self):
        return self.current >= len(self.source)

    def isDigit(self, c):
        return ord(c) >= ord("0") and ord(c) <= ord("9")

    def isAlpha(self, c):
        return (
            (ord(c) >= ord("a") and ord(c) <= ord("z"))
            or (ord(c) >= ord("A") and ord(c) <= ord("Z"))
            or c == "_"
        )

    def isAlphaNumeric(self, c):
        return self.isAlpha(c) or self.isDigit(c)

    def handleString(self):
        stringStartLine = self.line

        while not self.isAtEnd() and self.peekNextChar() != '"':
            if self.peekNextChar() == "\n":
                self.line += 1
            self.getNextChar()

        if self.isAtEnd():
            raise UnterminatedStringError(stringStartLine)

        # consume closing "
        self.getNextChar()

        val = self.source[self.start + 1 : self.current - 1]
        self.addToken(TokenType.STRING, literal=val)

    # Remember numbers cannot have a leading or trailing dot
    def handleNumber(self):
        while not self.isAtEnd() and self.isDigit(self.peekNextChar()):
            self.getNextChar()

        # This is a decimal if there is a dot followed by a number
        if self.peekNextChar() == "." and self.isDigit(self.peek2NextChar()):
            self.getNextChar()
            while self.isDigit(self.peekNextChar()):
                self.getNextChar()

        val = float(self.source[self.start : self.current])
        self.addToken(TokenType.NUMBER, literal=val)

    def handleIdentifier(self):
        while not self.isAtEnd() and self.isAlphaNumeric(self.peekNextChar()):
            self.getNextChar()

        val = self.source[self.start : self.current]
        keywordMap = {
            "and": TokenType.AND,
            "class": TokenType.CLASS,
            "else": TokenType.ELSE,
            "false": TokenType.FALSE,
            "fun": TokenType.FUN,
            "for": TokenType.FOR,
            "if": TokenType.IF,
            "nil": TokenType.NIL,
            "or": TokenType.OR,
            "print": TokenType.PRINT,
            "return": TokenType.RETURN,
            "super": TokenType.SUPER,
            "this": TokenType.THIS,
            "true": TokenType.TRUE,
            "var": TokenType.VAR,
            "while": TokenType.WHILE,
        }
        try:
            self.addToken(keywordMap[val])
        except KeyError:
            self.addToken(TokenType.IDENTIFIER)


class UnrecognizedTokenError(Exception):
    def __init__(self, line):
        self.line = line
        self.message = f"Unrecognized token on line {line}"
        super().__init__(self.message)


class UnterminatedStringError(Exception):
    def __init__(self, stringStartLine):
        self.line = stringStartLine
        self.message = f"Unterminated string starting on line {stringStartLine}"
        super().__init__(self.message)
