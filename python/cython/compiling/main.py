from test1 import func
import pyximport
pyximport.install()
import test2

print(func())

print(test2.__file__)
print(test2.func2())
