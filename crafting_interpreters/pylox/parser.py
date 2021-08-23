from typing import Any
from dataclasses import dataclass

from scanner import TokenType


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
        tree = self.expression()
        if not self.isAtEnd():
            raise ExtraTokensInExprError(self.peek())
        return tree

    def expression(self):
        return self.equality()

    def equality(self):
        expr = self.comparison()
        while self.match(TokenType.EQUAL_EQUAL, TokenType.BANG_EQUAL):
            operator = self.previous()
            right = self.comparison()
            expr = Binary(expr, operator, right)
        return expr

    def comparison(self):
        expr = self.term()
        while self.match(
            TokenType.GREATER,
            TokenType.GREATER_EQUAL,
            TokenType.LESS,
            TokenType.LESS_EQUAL,
        ):
            operator = self.previous()
            right = self.term()
            expr = Binary(expr, operator, right)
        return expr

    def term(self):
        expr = self.factor()
        while self.match(TokenType.PLUS, TokenType.MINUS):
            operator = self.previous()
            right = self.factor()
            expr = Binary(expr, operator, right)
        return expr

    def factor(self):
        expr = self.unary()
        while self.match(TokenType.STAR, TokenType.SLASH):
            operator = self.previous()
            right = self.unary()
            expr = Binary(expr, operator, right)
        return expr

    def unary(self):
        if self.match(TokenType.MINUS, TokenType.BANG):
            operator = self.previous()
            right = self.primary()
            expr = Unary(operator, right)
        else:
            expr = self.primary()
        return expr

    def primary(self):
        if self.match(TokenType.FALSE):
            return Literal(False)
        elif self.match(TokenType.TRUE):
            return Literal(True)
        elif self.match(TokenType.NIL):
            return Literal(None)
        elif self.match(TokenType.NUMBER, TokenType.STRING):
            return Literal(self.previous().literal)
        elif self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(TokenType.RIGHT_PAREN)
            return Grouping(expr)
        elif self.isAtEnd():
            raise EndOfExpressionError()
        else:
            raise Exception("Don't think we should get here?")

    # Token helpers
    def peek(self):
        return self.tokens[self.current]

    def check(self, *types):
        if self.isAtEnd():
            return False
        return self.peek().tokenType in types

    def match(self, *types):
        if self.check(*types):
            self.current += 1
            return True
        return False

    def previous(self):
        return self.tokens[self.current - 1]

    def consume(self, *types):
        if self.check(*types):
            self.advance()
            return self.previous()
        raise UnterminatedGroupingError()

    def advance(self):
        # if not self.isAtEnd():
        self.current += 1

    def isAtEnd(self):
        if self.peek().tokenType == TokenType.EOF:
            return True
        return False


class ExtraTokensInExprError(Exception):
    def __init__(self, token):
        line = token.lineNumber
        tokenType = token.tokenType
        lexeme = token.lexeme
        self.message = f"Extra token '{lexeme}' on line {line}"
        super().__init__(self.message)


class EndOfExpressionError(Exception):
    def __init__(self):
        self.message = f"Surprise end of expression"
        super().__init__(self.message)


class UnterminatedGroupingError(Exception):
    def __init__(self):
        self.message = f"Unterminated Grouping"
        super().__init__(self.message)
