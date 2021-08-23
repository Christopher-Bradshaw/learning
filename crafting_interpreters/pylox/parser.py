from typing import Any
from dataclasses import dataclass

from .scanner import TokenType


class Expr:
    pass


@dataclass
class Binary(Expr):
    left: Any
    operator: TokenType
    right: Any


@dataclass
class Literal(Expr):
    value: Any


@dataclass
class Grouping(Expr):
    expr: Expr


@dataclass
class Unary(Expr):
    operator: TokenType
    right: Any


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0

    def parse(self):
        return self.expression()

    def expression(self):
        return self.equality()

    def equality(self):
        expr = self.comparison()
        # etc

    def comparison(self):
        expr = self.term()
        # etc

    def term(self):
        expr = self.factor()
        # etc

    def factor(self):
        expr = self.unary()
        # etc

    def unary(self):
        expr = self.primary()
        # etc

    def primary(self):
        if self.match(TokenType.FALSE):
            Literal(False)
        elif self.match(TokenType.TRUE):
            Literal(True)
        elif self.match(TokenType.NIL):
            Literal(None)
        elif self.match((TokenType.NUMBER, TokenType.STRING)):
            Literal(self.previous().literal())
        elif self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN)
            return Grouping(expr)
        else:
            raise Exception("Don't think we should get here?")

    # Token helpers
    def peek(self):
        return self.tokens[self.current]

    def check(self, types):
        if self.isAtEnd():
            return False
        return self.peek().TokenType in set(list(types))

    def match(self, types):
        if self.check(types):
            self.current += 1
            return True
        return False

    def previous(self):
        return self.tokens[self.current - 1]

    def consume(self, types):
        if self.check(types):
            self.advance()
            return self.previous()
        raise Exception("Expected but didn't get!")

    def advance(self):
        # if not self.isAtEnd():
        self.current += 1

    def isAtEnd(self):
        if self.peek().tokenType == TokenType.EOF:
            return True
        return False
