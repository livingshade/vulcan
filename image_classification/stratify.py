from sklearn.cluster import KMeans
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
import json

def kmeans_hidden_stratify(arch, suffix):    
    def kmeans_sklearn(X, K, num_iters=1000):
        X_np = X.cpu().numpy()  # Convert tensor to numpy array
        kmeans = KMeans(n_clusters=K, max_iter=num_iters)
        kmeans.fit(X_np)
        return torch.tensor(kmeans.labels_), torch.tensor(kmeans.cluster_centers_)
    def load_data():
        with open(f"./cache/{arch}_embedding.pkl", "rb") as f:
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
    E = data
    X = list(E.values())
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    X_flattend = X_padded.view(X_padded.size(0), -1)
    print(X_flattend.shape)
    for K in [4, 8, 12, 16, 32]:
        labels, centers = kmeans_sklearn(X_flattend, K)
        labels = [int(x) for x in labels]
        dump = dict()
        for (idx, k) in enumerate(E.keys()):
            dump.update({k: labels[idx]})
        with open(f"./cache/cluster_{suffix}_{K}_{arch}.json", "w") as f:
            json.dump(dump, f)
    
if __name__ == "__main__":
    kmeans_hidden_stratify("resnet50", "last")
    
