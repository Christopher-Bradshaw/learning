import pytest

from scanner import Scanner, TokenType
from parser import (
    Parser,
    Literal,
    Grouping,
    Unary,
    Binary,
    Var,
    ExtraTokensInExprError,
    EndOfExpressionError,
    UnterminatedGroupingError,
)


def scan(source):
    return Scanner(source).scanTokens()


def scanExp(source):
    return scan(source + ";")


# scan operator
def op(source):
    tokens = Scanner(source).scanTokens()
    assert len(tokens) == 2
    return tokens[0]


one, two, three, four = Literal(1), Literal(2), Literal(3), Literal(4)


@pytest.mark.skip(reason="This is still changing")
class TestErrorInvalidExpressions:
    def test_extra_tokens(self):
        tests = [
            ("1 1", ExtraTokensInExprError),
            ("1 + 1 +", EndOfExpressionError),
            ("(1 + 1", UnterminatedGroupingError),
        ]
        for (source, expError) in tests:
            with pytest.raises(expError):
                Parser(scan(source)).parse()


class TestVarStatements:
    def test_uninitialized(self):
        tests = [
            ("var x;", Var("x", None)),
        ]
        for (source, exp) in tests:
            var = Parser(scan(source)).parse()[0]
            assert var == exp

    def test_initialized(self):
        tests = [
            ("var x = 1;", Var("x", one)),
            ("var x = 1 + 2;", Var("x", Binary(one, op("+"), two))),
        ]
        for (source, exp) in tests:
            var = Parser(scan(source)).parse()[0]
            assert var == exp


class TestParseCompoundExpressions:
    def test_precedence_and_associativity(self):
        tests = [
            (
                "1 + 2 * 3 + 4",
                Binary(
                    Binary(one, op("+"), Binary(two, op("*"), three)), op("+"), four
                ),
            ),
            (
                "(1 + 2) * 3 + 4",
                Binary(
                    Binary(Grouping(Binary(one, op("+"), two)), op("*"), three),
                    op("+"),
                    four,
                ),
            ),
        ]
        for (source, exp) in tests:
            expr = Parser(scanExp(source)).parse()[0].expression
            assert expr == exp


class TestParseSimpleExpressions:
    def test_literal(self):
        tests = [
            ("1", Literal(1)),
            ('"asdf"', Literal("asdf")),
            ("true", Literal(True)),
            ("false", Literal(False)),
            ("nil", Literal(None)),
        ]
        for (source, exp) in tests:
            expr = Parser(scanExp(source)).parse()[0].expression
            assert expr == exp

    def test_grouping(self):
        tests = [
            ("(1)", Grouping(one)),
            ("(1 + 2)", Grouping(Binary(one, op("+"), two))),
        ]
        for (source, exp) in tests:
            expr = Parser(scanExp(source)).parse()[0].expression
            assert expr == exp

    def test_unary(self):
        tests = [("-1", Unary(op("-"), one)), ("!1", Unary(op("!"), one))]
        for (source, exp) in tests:
            expr = Parser(scanExp(source)).parse()[0].expression
            assert expr == exp

    def test_factor(self):
        tests = [
            ("1 * 2", Binary(one, op("*"), two)),
            ("1 / 2", Binary(one, op("/"), two)),
        ]
        for (source, exp) in tests:
            expr = Parser(scanExp(source)).parse()[0].expression
            assert expr == exp

    def test_term(self):
        tests = [
            ("1 + 2", Binary(one, op("+"), two)),
            ("1 - 2", Binary(one, op("-"), two)),
        ]
        for (source, exp) in tests:
            expr = Parser(scanExp(source)).parse()[0].expression
            assert expr == exp

    def test_comparison(self):
        tests = [
            ("1 > 2 ", Binary(one, op(">"), two)),
            ("1 >= 2", Binary(one, op(">="), two)),
            ("1 < 2 ", Binary(one, op("<"), two)),
            ("1 <= 2", Binary(one, op("<="), two)),
        ]
        for (source, exp) in tests:
            expr = Parser(scanExp(source)).parse()[0].expression
            assert expr == exp
