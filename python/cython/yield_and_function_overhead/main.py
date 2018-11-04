import funcs
import time
""" What we learned:
    Yield can only be called from def'ed function and is therefore v slow.
    Function call overhead is negligible. Actually the func with the extra
    function call was faster? Weird...
"""


def call_and_time(f, summary=None):
    start = time.time()
    x = f(100_000_000)
    if summary: print(summary)
    print("seconds taken", time.time() - start)

call_and_time(funcs.counter, "simple")
call_and_time(funcs.counter_that_calls_a_func, "with func call")
call_and_time(funcs.counter_that_calls_a_generator, "generator")
"""
Seconds taken 0.066 - No idea why this is slower than the second one!
Seconds taken 0.032
Seconds taken 2.901 - Very slow because of the def'ed function in the loop
"""
