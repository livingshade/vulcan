from torch.utils.data import Dataset, Sampler

class StratifiedSampler(Sampler):
    def __init__(self, data_source: Dataset):
        for data in data_source:
            print(data)
            exit(0)

    def __iter__(self):
        indices = []
        for i in range(self.length):
            for c in self.classes:
                indices.append(self.class_indices[c][i % len(self.class_indices[c])])
        return iter(indices)

    def __len__(self):
        return self.length * len(self.classes)