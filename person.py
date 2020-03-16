class Person():
    def __init__(self, name, **argum):
        self.name = name
        for key, value in argum.items():
            setattr(self, key, value)