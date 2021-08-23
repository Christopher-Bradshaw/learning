import pytest

from scanner import Scanner, TokenType, UnterminatedStringError, UnrecognizedTokenError


def scanAndAssert(source, exp):
    tokens = Scanner(source).scanTokens()

    # Handle final EOF token
    assert tokens[-1].tokenType == TokenType.EOF
    tokens = tokens[:-1]

    assert len(tokens) == len(exp)
    for i in range(len(tokens)):
        assert tokens[i].tokenType == exp[i]
    return tokens


class TestSingleLineScanner:
    def test_basic(self):
        source = "()!=!"
        exp = [
            TokenType.LEFT_PAREN,
            TokenType.RIGHT_PAREN,
            TokenType.BANG_EQUAL,
            TokenType.BANG,
        ]
        scanAndAssert(source, exp)

    def test_handles_whitespace(self):
        source = "! +"
        exp = [
            TokenType.BANG,
            TokenType.PLUS,
        ]
        scanAndAssert(source, exp)

    def test_handles_slash(self):
        source = "!/!"
        exp = [
            TokenType.BANG,
            TokenType.SLASH,
            TokenType.BANG,
        ]
        scanAndAssert(source, exp)

    def test_handles_strings(self):
        source = '!"asdf"!'
        exp = [
            TokenType.BANG,
            TokenType.STRING,
            TokenType.BANG,
        ]
        tokens = scanAndAssert(source, exp)
        assert tokens[1].literal == "asdf"

    def test_handles_numbers(self):
        sources = ["!123!", "!12.3!", "!1.23!"]
        exp = [
            TokenType.BANG,
            TokenType.NUMBER,
            TokenType.BANG,
        ]
        values = [123, 12.3, 1.23]
        for (i, source) in enumerate(sources):
            tokens = scanAndAssert(source, exp)
            assert tokens[1].literal == values[i]

    def test_handle_keywords(self):
        source = "orchid and fun"
        exp = [
            TokenType.IDENTIFIER,
            TokenType.AND,
            TokenType.FUN,
        ]
        scanAndAssert(source, exp)


class TestMultiLineScanner:
    def test_comments(self):
        source = """
a = 1 // Assignment
!! // Garbage
"""
        exp = [
            TokenType.IDENTIFIER,
            TokenType.EQUAL,
            TokenType.NUMBER,
            TokenType.BANG,
            TokenType.BANG,
        ]

        scanAndAssert(source, exp)


class TestScannerFailures:
    def test_unterminated_string(self):
        source = '"asdf'
        with pytest.raises(UnterminatedStringError):
            Scanner(source).scanTokens()

    def test_unrecognized_token(self):
        source = ":"
        with pytest.raises(UnrecognizedTokenError):
            Scanner(source).scanTokens()
