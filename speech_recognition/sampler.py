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

class VOiCEBootstrapSampler(Sampler):
    def __init__(self, boostrap_th) -> None:
        self.random_data = [] 
        self.random_idx = 0
        self.stratified_data = dict()
        self.stratified_weight = dict()
        self.stratified_idx = dict()
        self.strata_next = 0
        self.feedback_idx = 0
        self.feedback_threshold = boostrap_th # 160 here
        self.feedback_data = dict()
        self.get_all_data()
        # print(f"Summary:\n\t#Samples:{len(self.random_data)}")
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        cul = 0
        for l1 in l1s:
            for l2 in l2s:
                # print(f"\t{l1}-{l2}:{len(self.stratified_data[l1][l2])} {self.stratified_weight[l1][l2] - cul}")
                cul = self.stratified_weight[l1][l2]
         
    def get_all_data(self):
        l0 = "/data/VOiCES_devkit/distant-16k/speech/test"
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        for l1 in l1s:
            self.stratified_data[l1] = dict()
            self.stratified_weight[l1] = dict()
            self.stratified_idx[l1] = dict()
            self.feedback_data[l1] = dict()
            for l2 in l2s:
                self.stratified_data[l1][l2] = []
                self.stratified_weight[l1][l2] = 0
                self.stratified_idx[l1][l2] = 0
                self.feedback_data[l1][l2] = []
                
        for l1 in l1s:
            for l2 in l2s:
                path = l0 + '/' + l1 + '/' + l2
                sps = [f"{path}/{sp}" for sp in os.listdir(path) if sp.startswith('sp')]
                for sp in sps:
                    fs = os.listdir(sp)
                    for f in fs:
                        self.random_data.append(f"{sp}/{f}")
                        self.stratified_data[l1][l2].append(f"{sp}/{f}")
                random.shuffle(self.stratified_data[l1][l2])
        culmulative_weight = 0
        for l1 in l1s:
            for l2 in l2s:
                culmulative_weight += len(self.stratified_data[l1][l2]) / len(self.random_data)
                self.stratified_weight[l1][l2] = culmulative_weight
                self.stratified_idx[l1][l2] = 0
        random.shuffle(self.random_data)

    
    def random_sample(self):
        idx = self.random_idx
        self.random_idx = (self.random_idx + 1) % len(self.random_data)
        return self.random_data[idx]

    def update_weight_based_on_feedback(self):
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        culmulative_weight = 0
        for l1 in l1s:
            for l2 in l2s:
                variance = sum([(x - sum(self.feedback_data[l1][l2]) / len(self.feedback_data[l1][l2])) ** 2 for x in self.feedback_data[l1][l2]]) / len(self.feedback_data[l1][l2])
                self.stratified_weight[l1][l2] = variance
                culmulative_weight += self.stratified_weight[l1][l2]
        for l1 in l1s:
            for l2 in l2s:
                # normalize weight
                self.stratified_weight[l1][l2] /= culmulative_weight
                
        suf = 0
        for l1 in l1s:
            for l2 in l2s:
                suf += self.stratified_weight[l1][l2]
                self.stratified_weight[l1][l2] = suf

    def sample_strata_based_on_weight(self):
        weight = random.uniform(0, 1)
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        for l1 in l1s:
            for l2 in l2s:
                if len(self.stratified_data[l1][l2]) > 0 and weight < self.stratified_weight[l1][l2]:
                    return l1, l2
        raise Exception("Weight not in range")
    
    def sample_strata_ordered(self):
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        l1 = self.strata_next % len(l1s)
        l2 = self.strata_next // len(l1s)
        self.strata_next = (self.strata_next + 1) % (len(l1s) * len(l2s))
        return l1s[l1], l2s[l2]
    
    def stratified_sample(self):  
        l1, l2 = self.sample_strata_ordered()
        # l1, l2 = self.sample_strata_based_on_weight()
        idx = self.stratified_idx[l1][l2]
        data = self.stratified_data[l1][l2]
        self.stratified_idx[l1][l2] = (idx + 1) % len(data)
        return data[idx] 

    def feedback_sample(self):
        if self.feedback_idx < self.feedback_threshold:
            return self.stratified_sample()
        else:
            raise Exception("Feedback threshold reached")
        # l1, l2 = self.sample_strata_based_on_weight()
        # idx = self.stratified_idx[l1][l2]
        # data = self.stratified_data[l1][l2]
        # self.stratified_idx[l1][l2] = (idx + 1) % len(data)
        # return data[idx]

    def weighted_sample(self):
        l1, l2 = self.sample_strata_based_on_weight()
        idx = self.stratified_idx[l1][l2]
        data = self.stratified_data[l1][l2]
        self.stratified_idx[l1][l2] = (idx + 1) % len(data)
        return data[idx]

    def sample(self, ctx) -> str:
        if ctx == "random":
            return self.random_sample()
        elif ctx == "stratified":
            return self.stratified_sample()
        elif ctx == "weighted":
            return self.weighted_sample()

    def feedback(self, ctx):
        # filename = 'Lab41-SRI-VOiCES-rm1-musi-sp5622-ch041172-sg0001-mc01-stu-clo-dg010.wav'
        def get_l1_l2(name):
            l1 = name.split('-')[3]
            l2 = name.split('-')[4]
            return l1, l2
        name, wer = ctx
        l1, l2 = get_l1_l2(name)
        self.feedback_data[l1][l2].append(wer)
        self.feedback_idx += 1
        
    def update_weight(self):
        self.update_weight_based_on_feedback()
        print(f"Update weight based on feedback, #sample = {self.feedback_idx}")
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        cul = 0
        for l1 in l1s:
            for l2 in l2s:
                print(f"\t{l1}-{l2}:{self.stratified_weight[l1][l2] - cul:4f}")
                cul = self.stratified_weight[l1][l2]
                
                
class VOiCEStratifiedSampler(Sampler):
    def __init__(self, cluster):
        self.keys = list(cluster.keys()).copy()
        self.k2g = cluster.copy()
        self.n_groups = len(set(list(cluster.values())))
        self.group_order = list(range(self.n_groups))
        self.next_group = 0
        self.group = [
            [k for k in self.keys if self.k2g[k] == i] for i in range(self.n_groups)
        ]
        self.group_idx = [0] * self.n_groups
        self.group_weight = [float(len(i)) / len(self.keys) for i in self.group]
        res = ""
        for i in self.group:
            res = res + " " + str(len(i))
        print(f"Stratified Sampler: {res}")
        
    def init(self):
        self.group_order = list(range(self.n_groups))
        random.shuffle(self.group_order)
        
        self.next_group = 0
        self.group_idx = [0] * self.n_groups
        for i in range(self.n_groups):
            random.shuffle(self.group[i])
        
    def stratified_sample(self):
        w = float(self.next_group) / UNIT_LEN
        
        self.next_group = (self.next_group + 1) % UNIT_LEN

        cul_w = 0
        key = None
        for i in range(self.n_groups):
            gid = self.group_order[i]
            cul_w += self.group_weight[gid]
            if cul_w > w:
                idx = self.group_idx[gid]
                self.group_idx[gid] = (self.group_idx[gid] + 1) % len(self.group[gid])
                key = self.group[gid][idx]
                break
        
        return key
    
    def sample(self, method):
        if method == "stratified":
            return self.stratified_sample()
        else:
            raise Exception(f"Invalid sample method [{method}] for VOiCEStratifiedSampler")
        
        
class VOiCEStratifiedWeightedSampler(Sampler):
    def __init__(self, cluster):
        self.keys = list(cluster.keys()).copy()
        self.k2g = cluster.copy()
        self.n_groups = len(set(list(cluster.values())))
        self.next_group = 0
        self.group = [
            [k for k in self.keys if self.k2g[k] == i] for i in range(self.n_groups)
        ]
        self.group_idx = [0] * self.n_groups
        res = ""
        for i in self.group:
            res = res + " " + str(len(i))
        print(f"Weighted Sampler: {res}")
        
    def init(self):        
        self.next_group = 0
        self.group_idx = [0] * self.n_groups
        for i in range(self.n_groups):
            random.shuffle(self.group[i])
            
        random.shuffle(self.keys)
       
    def sample_group(self):
        key = random.choice(self.keys)
        return self.k2g[key]
    
    def stratified_sample(self):
        g = self.sample_group()
        
        idx = self.group_idx[g]
        self.group_idx[g] = (self.group_idx[g] + 1) % len(self.group[g])
        key = self.group[g][idx]
        
        return key
    
    def sample(self, method):
        if method == "stratified":
            return self.stratified_sample()
        else:
            raise Exception(f"Invalid sample method [{method}] for VOiCEStratifiedWeightedSampler")