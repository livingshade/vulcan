from sklearn.cluster import KMeans
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
import json

def kmeans_stratify():    
    def kmeans_sklearn(X, K, num_iters=100):
        X_np = X.cpu().numpy()  # Convert tensor to numpy array
        kmeans = KMeans(n_clusters=K, max_iter=num_iters)
        kmeans.fit(X_np)
        return torch.tensor(kmeans.labels_), torch.tensor(kmeans.cluster_centers_)
    def load_data():
        with open("./cache/embedding.pkl", "rb") as f:
            data = pickle.load(f)
        return data 
        '''
        data = {
            "args" : {},
            "results": {
                filename: vec
            }
        }
        '''
        
    # Example usage
    data = load_data()
    E = data["results"]
    X = list(E.values())
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    # [6400, longest, 29]
    X_flattend = X_padded.view(X_padded.size(0), -1)
    # [6400, longest * 29]
    print(X_flattend.shape)
    K = 16
    labels, centers = kmeans_sklearn(X_flattend, K)
    labels = [int(x) for x in labels]
    dump = dict()
    for (idx, k) in enumerate(E.keys()):
        dump.update({k: labels[idx]})
    with open("./cache/cluster_kmeans.json", "w") as f:
        json.dump(dump, f)

def natural_stratify():

    def get_idx(filename):
        #Lab41-SRI-VOiCES-rm4-none-sp6518-ch066465-sg0028-mc01-stu-clo-dg130.wav
        l1, l2 = filename.split('-')[3], filename.split('-')[4]
        l1s = ["rm1", "rm2", "rm3", "rm4"]
        l2s = ['babb', 'musi', 'none', 'tele']
        return l1s.index(l1) * len(l2s) + l2s.index(l2)
    
    with open("./cache/embedding.pkl", "rb") as f:
        data = pickle.load(f)
    
    keys = list(data["results"].keys())
    dump = {}
    for k in keys:
        dump.update({k: get_idx(k)})
    with open("./cache/cluster_natural.json", "w") as f:
        json.dump(dump, f)
    
if __name__ == "__main__":
    natural_stratify()

