import os
import random

UNIT_LEN = 1000

class Sampler():
    def __init__(self) -> None:
        pass
    def init(self):
        pass
    def sample(self, ctx):
        pass
    def feedback(self, ctx):
        pass
    def update_weight():
        pass
    
class VOiCERandomSampler(Sampler):
    def __init__(self, keys):
        self.keys = keys.copy()
        self.random_idx = 0
        
    def init(self):
        self.random_idx = 0
        random.shuffle(self.keys)
        
    def random_sample(self):
        idx = self.random_idx
        self.random_idx = (self.random_idx + 1) % len(self.keys)
        return self.keys[idx]
    

    def sample(self, method):
        if method == "random":
            return self.random_sample()
        else:
            raise Exception(f"Invalid sample method [{method}] for VOiCERandomSampler")