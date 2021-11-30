from interpreter import Interpreter
from scanner import Scanner, TokenType
from parser import (
    Parser,
    Literal,
)


def parse(source):
    return Parser(Scanner(source).scanTokens()).parse()


def parseExp(source):
    return Parser(Scanner(source + ";").scanTokens()).parse()


class TestVariableDefinition:
    def test_variable_definitions(self):
        tests = [
            "var a = 1;",
            "var a = 0.5 + 0.5;",
        ]
        for source in tests:
            i = Interpreter(parse(source))
            i.interpret()
            assert i.environment.get("a") == 1

    def test_multiple_variable_definitions(self):
        source = "var a = 1; var b = 2; var c = a + b;"
        exp = {"a": 1, "b": 2, "c": 3}
        i = Interpreter(parse(source))
        i.interpret()
        for k in exp:
            assert i.environment.get(k) == exp[k]

    def test_unassigned_variable_definition(self):
        source = "var a;"
        i = Interpreter(parse(source))
        i.interpret()
        assert i.environment.get("a") is None


class TestEvalSimpleExpressions:
    def test_literal(self):
        tests = [
            ("1", 1),
            ('"asdf"', "asdf"),
        ]

        for (source, exp) in tests:
            res = parseExp(source)[0].expression.accept(Interpreter())
            assert res == exp

    def test_grouping(self):
        tests = [
            ("(1)", 1),
        ]
        for (source, exp) in tests:
            res = parseExp(source)[0].expression.accept(Interpreter())
            assert res == exp

    def test_binary_math(self):
        tests = [
            ("1 + 2", 3),
            ("1 - 2", -1),
            ("2 * 4", 8),
            ("4 / 2", 2),
            ('"1" + "2"', "12"),
        ]
        for (source, exp) in tests:
            res = parseExp(source)[0].expression.accept(Interpreter())
            assert res == exp

    def test_binary_comparison(self):
        tests = [
            ("1 > 2", False),
            ("1 > 0", True),
            ("2 < 2", False),
            ("2 <= 2", True),
            ("3 >= 2", True),
        ]
        for (source, exp) in tests:
            res = parseExp(source)[0].expression.accept(Interpreter())
            assert res == exp

    def test_unary(self):
        tests = [
            ("-1", -1),
            ("!1", False),
        ]
        for (source, exp) in tests:
            res = parseExp(source)[0].expression.accept(Interpreter())
            assert res == exp
