from errors import RuntimeError


class Environment:
    def __init__(self):
        self.final = {}

    def define(self, name, value):
        self.final[name] = value

    def get(self, name):
        if name in self.final:
            return self.final[name]
        else:
            raise UndefinedVariableError(name)


class UndefinedVariableError(RuntimeError):
    def __init__(self, name):
        self.message = f"Unterminated Variable: {name}"
        super().__init__(self.message)
