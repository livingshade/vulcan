from torch.utils.data import Dataset, Sampler
import torchvision.datasets as datasets
import random

class RandomSampler(Sampler):
    def __init__(self, data_source: datasets.ImageFolder):
        self.key2idx = {}
        self.idx2key = []
        for idx, sample in enumerate(data_source.samples):
            path, _ = sample
            key = path.split('/')[-1]
            self.key2idx[key] = idx
            self.idx2key.append(key)

        self.allocate_history = []

    def __iter__(self):
        for i in range(len(self.idx2key)):
            self.allocate_history.append(self.idx2key[i])
            yield i
            
    def __len__(self):
        return len(self.idx2key)

class StratifiedSampler(Sampler):
    def __init__(self, data_source: datasets.ImageFolder, cluster):
        self.cluster = cluster
        self.keys = list(cluster.keys()).copy()
        
        self.classes = data_source.classes
        self.class_keys = { c: [] for c in self.classes }
        for k, v in cluster.items():
            self.class_keys[v].append(k)
        self.class_idx = { c: 0 for c in self.classes }
        
        self.weight = [len(self.class_keys[c]) / len(self.keys) for c in self.classes]
        
        self.length = len(cluster.keys())
        self.key2idx = {}
        for idx, sample in enumerate(data_source.samples):
            path, _ = sample
            key = path.split('/')[-1]
            self.key2idx[key] = idx
        
        self.init()
        
    def init(self):
        random.shuffle(self.keys)
        for c in self.class_keys:
            random.shuffle(self.class_keys[c])
    
    def sample_class(self, w):
        for i in range(len(self.weight)):
            if w < self.weight[i]:
                return i
            w -= self.weight[i]
        return len(self.weight) - 1
    
    def __iter__(self):
        SAMPLE_UNIT = 1000
        for i in range(self.length // SAMPLE_UNIT):
            for j in range(SAMPLE_UNIT):
                cls = self.classes[self.sample_class(j / SAMPLE_UNIT)]
                
                idx = self.class_idx[cls]
                
                self.class_idx[cls] = (idx + 1) % len(self.class_keys[cls])
                
                key = self.class_keys[cls][idx]
                
                yield self.key2idx[key]

    def __len__(self):
        return len(self.keys)