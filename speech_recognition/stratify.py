from sklearn.cluster import KMeans
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
import json
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
if __name__ == "__main__":
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
    with open("./cache/cluster.json", "w") as f:
        json.dump(dump, f)
    # print(labels[:10], centers)
    # X = torch.randn(100, 2)  # 100 points in 2D
    # K = 3  # Number of clusters
    # cluster_assignments, centers = kmeans_sklearn(X, K)
    # print("Cluster assignments:", cluster_assignments)
    # print("Cluster centers:", centers)
