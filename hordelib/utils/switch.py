class Switch:
    def __init__(self, initial_value=False):
        self.active = initial_value

    def activate(self):
        self.active = True

    def disable(self):
        self.active = False

    def toggle(self, value: bool):
        self.active = value
