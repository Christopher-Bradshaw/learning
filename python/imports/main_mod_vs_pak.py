from my_collection_of_modules import mod1
import my_collection_of_modules.mod1 as mod1_2

from my_collection_of_modules import mod2
import my_collection_of_modules.deeper.mod3 as mod3

assert mod1 is mod1_2
print(mod1.v1)
print(mod2.mod1.v1)

print(mod3)
