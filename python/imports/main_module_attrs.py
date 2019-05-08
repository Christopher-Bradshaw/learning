import my_module
for k in dir(my_module):
    if k != "__builtins__": print(k, getattr(my_module, k))
