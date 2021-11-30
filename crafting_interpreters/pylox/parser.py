from typing import Any
from dataclasses import dataclass

from scanner import TokenType


class Expr:
    def accept(self, visitor):
        return getattr(visitor, f"visit{type(self).__name__}Expr")(self)


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


@dataclass
class Variable(Expr):
    name: str


class Stmt:
    def accept(self, visitor):
        return getattr(visitor, f"visit{type(self).__name__}Stmt")(self)


@dataclass
class Print(Stmt):
    expression: Expr


@dataclass
class Expression(Stmt):
    expression: Expr


@dataclass
class Var(Stmt):
    name: str
    initializer: Expr


class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current = 0
        self.statements = []

    def parse(self):
        # breakpoint()
        while not self.isAtEnd():
            self.statements.append(self.declaration())

        # tree = self.expression()

        # # We want to add a synchonization thing here once we parse full programs
        # # To allow us to catch most errors
        # # http://craftinginterpreters.com/parsing-expressions.html#panic-mode-error-recovery
        # http://craftinginterpreters.com/statements-and-state.html#parsing-variables
        # if not self.isAtEnd():
        #     raise ExtraTokensInExprError(self.peek())
        # return tree

        return self.statements

    def declaration(self):
        if self.match(TokenType.VAR):
            return self.varDeclaration()
        else:
            return self.statement()

    def varDeclaration(self):
        name = self.consume(ExpectedIdentifierError, TokenType.IDENTIFIER).lexeme
        expr = None
        if self.match(TokenType.EQUAL):
            expr = self.expression()

        self.consume(UnterminatedStatementError, TokenType.SEMICOLON)

        return Var(name, expr)

    def statement(self):
        if self.match(TokenType.PRINT):
            return self.printStatement()
        else:
            return self.expressionStatement()

    def printStatement(self):
        expr = self.expression()
        self.consume(UnterminatedStatementError, TokenType.SEMICOLON)
        return Print(expr)

    def expressionStatement(self):
        expr = self.expression()
        self.consume(UnterminatedStatementError, TokenType.SEMICOLON)
        return Expression(expr)

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
        if self.match(TokenType.IDENTIFIER):
            return Variable(self.previous().lexeme)
        elif self.match(TokenType.FALSE):
            return Literal(False)
        elif self.match(TokenType.TRUE):
            return Literal(True)
        elif self.match(TokenType.NIL):
            return Literal(None)
        elif self.match(TokenType.NUMBER, TokenType.STRING):
            return Literal(self.previous().literal)
        elif self.match(TokenType.LEFT_PAREN):
            expr = self.expression()
            self.consume(UnterminatedGroupingError, TokenType.RIGHT_PAREN)
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

    def consume(self, err, *types):
        if self.check(*types):
            self.advance()
            return self.previous()
        raise err

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


class UnterminatedStatementError(Exception):
    def __init__(self):
        self.message = f"Expected semicolon"
        super().__init__(self.message)


class ExpectedIdentifierError(Exception):
    def __init__(self):
        self.message = f"Expected identifier"
        super().__init__(self.message)


class UnterminatedGroupingError(Exception):
    def __init__(self):
        self.message = f"Unterminated Grouping"
        super().__init__(self.message)
