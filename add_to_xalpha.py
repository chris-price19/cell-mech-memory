class noise:
    def __init__(self, mean=0., std = 1., mag = 0.1):
        self.mean = mean
        self.std = std
        self.mag = mag
        self.rg = Generator(PCG64())
    def draw(self):
        return self.mag * self.rg.normal(self.mean, self.std)