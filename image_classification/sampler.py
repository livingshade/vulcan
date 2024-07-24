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

    def init(self):
        random.shuffle(self.idx2key)
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
    
        self.n_group = len(set(list(cluster.values())))
        self.groups = [[] for _ in range(self.n_group)]    
        for k, v in cluster.items():
            self.groups[v].append(k)
        self.group_idx = [0 for _ in range(self.n_group)]
        
        self.length = len(self.keys)

        self.weight = [len(self.groups[i]) / self.length for i in range(self.n_group)]
        
        self.key2idx = {}
        for idx, sample in enumerate(data_source.samples):
            path, _ = sample
            key = path.split('/')[-1]
            self.key2idx[key] = idx
        self.allocate_history = []
        self.init()
        
    def init(self):
        random.shuffle(self.keys)
        for g in self.groups:
            random.shuffle(g)
        self.group_idx = [0 for _ in range(self.n_group)]
        self.allocate_history = []
        
    def sample_group(self, w):
        for i in range(self.n_group):
            if w < self.weight[i]:
                return i
            w -= self.weight[i]
        return self.n_group - 1
    
    def __iter__(self):
        SAMPLE_UNIT = 1000
        for i in range(self.length // SAMPLE_UNIT):
            for j in range(SAMPLE_UNIT):
                g = self.sample_group(j / SAMPLE_UNIT) 
                               
                idx = self.group_idx[g]
                self.group_idx[g] = (self.group_idx[g] + 1) % len(self.groups[g])
                
                key = self.groups[g][idx]
                self.allocate_history.append(key)
                yield self.key2idx[key]

    def __len__(self):
        return len(self.keys)