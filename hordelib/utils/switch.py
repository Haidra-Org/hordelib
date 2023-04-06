class Switch:
    active: bool = False

    def activate(self):
        self.active = True

    def disable(self):
        self.active = False

    def toggle(self, value: bool):
        self.active = value
