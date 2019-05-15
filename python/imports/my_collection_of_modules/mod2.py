from . import mod1
import my_collection_of_modules.mod1 as mod1_2
assert mod1 is mod1_2

print(__name__)
# import mod1 # would fail!
v2 = 1
