from scanner import TokenType
from environment import Environment

# Note that this is a visitor class (i.e. it implements that functions
# that interpret each of the parser classes)
class Interpreter:
    def __init__(self, stmts=None):
        self.stmts = stmts if stmts else []
        self.environment = Environment()

    def interpret(self):
        for stmt in self.stmts:
            self.execute(stmt)

    # Statements don't return anything
    def visitExpressionStmt(self, obj):
        self.evaluate(obj.expression)

    def visitPrintStmt(self, obj):
        print(self.evaluate(obj.expression))

    def visitVarStmt(self, obj):
        initVal = None if obj.initializer is None else self.evaluate(obj.initializer)
        self.environment.define(obj.name, initVal)

    # Expressions do return their result
    def visitVariableExpr(self, expr):
        return self.environment.get(expr.name)

    def visitLiteralExpr(self, expr):
        return expr.value

    def visitGroupingExpr(self, expr):
        return self.evaluate(expr.expr)

    def visitBinaryExpr(self, expr):
        left = self.evaluate(expr.left)
        right = self.evaluate(expr.right)
        op = expr.operator.tokenType

        if op == TokenType.PLUS:
            return left + right
        elif op == TokenType.MINUS:
            return left - right
        elif op == TokenType.STAR:
            return left * right
        elif op == TokenType.SLASH:
            return left / right
        elif op == TokenType.GREATER:
            return left > right
        elif op == TokenType.GREATER_EQUAL:
            return left >= right
        elif op == TokenType.LESS:
            return left < right
        elif op == TokenType.LESS_EQUAL:
            return left <= right
        elif op == TokenType.EQUAL_EQUAL:
            return self.isEqual(left, right)
        elif op == TokenType.BANG_EQUAL:
            return not self.isEqual(left, right)
        else:
            raise Exception("Unknown operator in binary expr")

    def visitUnaryExpr(self, expr):
        op = expr.operator.tokenType
        right = self.evaluate(expr.right)

        if op == TokenType.BANG:
            return not right
        elif op == TokenType.MINUS:
            return -right
        else:
            raise Exception("Unknown operator in unary expr")

    # Note that this takes an expression
    def evaluate(self, expr):
        return expr.accept(self)

    # While this takes a statement
    def execute(self, stmt):
        return stmt.accept(self)

    def isEqual(self, left, right):
        return left == right

    # Could define this?
    # http://craftinginterpreters.com/evaluating-expressions.html#truthiness-and-falsiness
    # def isTruthy(self, obj):
    #     pass
