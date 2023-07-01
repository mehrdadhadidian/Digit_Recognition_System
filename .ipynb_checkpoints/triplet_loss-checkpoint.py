import torch
import torch.nn as nn


class triplet_loss(nn.Module):
    def __init__(self, margin: float = 0.005, device = torch.device('cpu')) -> None:
        super().__init__()
        """
        triplet loss implementation, explained in section 3 of project documentation.

        Args:
            margin: margin parameter in triplet loss (equation 5 in documentation)
                default value is 0.005, you may need to set different value for margin depending on your model.
            device: pass torch.device("cuda:0") to this parameter if you are using gpu.
        """
        self.margin = torch.tensor(margin)
        self.device = device

    # Compelete this function
    def forward(self, embeddings, labels):
        """
        embeddings: [M, d] M is the batch size, d is the dimension of the embeddings (dimension of the layer before the last layer)
        embeddings are supposed to be outputs of the layer before the last layer. 

        labels: [M, ]
        """
        
        triplet_loss = ...

        return triplet_loss

    def batch_hard_triplet_loss(self, embeddings, labels):
        """
        Args: 
            embeddings -> [N, d]
            labels -> [N, 1]

        returns: 
            dp -> [N, 1] distance of furthest positive pair for each sample
            dn -> [N, 1] distance of closest negative pair for each sample
        """

        dists = self.euclidean_dist(embeddings, embeddings)
        # dists -> [N, N], square mat of all distances, 
        # dists[i, j] is distance between sample[i] and sample[j]
        
        same_identity_mask = torch.eq(labels[:, None], labels[None, :]) 
        # [N, N], same_mask[i, j] = True when sample i and j have the same label

        negative_mask = torch.logical_not(same_identity_mask)
        # [N, N], negative_mask[i, j] = True when sample i and j have different label

        positive_mask = torch.logical_xor(same_identity_mask, torch.eye(labels.shape[0], dtype=torch.bool).to(self.device))
        # [N, N], same as same_identity mask, except diagonal is zero

        dp, _ = torch.max(dists * (positive_mask.int()), dim=1)

        dn = torch.zeros_like(dp)
        for i in range(dists.shape[0]):
            dn[i] = torch.min(dists[i, :][negative_mask[i, :]])    

        return dp, dn
    
    def all_diffs(self, a, b):
        # a, b -> [N, d]
        # return -> [N, N, d]
        return a[:, None] - b[None, :]

    def euclidean_dist(self, embed1, embed2):
        # embed1, embed2 -> [N, d]
        # return [N, N] -> # get a square matrix of all diffs, diagonal is zero
        diffs = self.all_diffs(embed1, embed2) 
        t1 = torch.square(diffs)
        t2 = torch.sum(t1, dim=-1)
        return torch.sqrt(t2 + 1e-12)