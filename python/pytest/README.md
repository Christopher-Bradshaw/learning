# Python Testing with pytest

Pytest seems like the easiest python testing framework.

## Test Discovery

Pytest will find all tests functions by looking in:
* Files starting or ending with `test`: e.g. `test_funcs.py`, `funcs_test.py`
* Classes that start with `Test`
* Methods or functions that start with `test_`

## Exceptions

Wrap the code that will throw in a `with pytest.raises(Exception) as err:`. Then check that `err.value` contains the exception that you expect.

## Assertions

Just use assert.

## Running
* `pytest`: Run all tests that it can find

### Useful args
See `pytest --help` for a full list
* `-s`: By default pytest suppresses stdout. With -s, show stdout
* `-k "some_expression"`: Only run tests that match the expression. e.g. "Add1 and positive_number"
