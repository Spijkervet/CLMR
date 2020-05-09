import torch
import torch.nn as nn

class SupervisedLoss(nn.Module):

    def __init__(self, args, n_features, n_classes):
        super(SupervisedLoss, self).__init__()
        self.args = args
        self.n_features = n_features
        self.n_classes = n_classes

        # create linear classifier
        self.model = nn.Sequential(
            nn.Linear(n_features, n_classes)
        ).to(args.device)

        self.num_labels = 1
        self.criterion = nn.BCEWithLogitsLoss()
    
    def predict(self, x):
        x = x.permute(0, 2, 1)
        pooled = nn.functional.adaptive_avg_pool1d(x, 1) # one label
        pooled = pooled.permute(0, 2, 1).reshape(-1, self.n_features)
        output = self.model(pooled)
        predicted = output.argmax(1)
        return predicted

    def get_pooled(self, x):
        x = x.permute(0, 2, 1)
        pooled = nn.functional.adaptive_avg_pool1d(x, 1) # one label
        pooled = pooled.permute(0, 2, 1).reshape(-1, self.n_features) 
        return pooled

    def forward(self, x, y):
        x = self.get_pooled(x)
        output = self.model(x)
        loss = self.criterion(output, y)
        return loss, output

    def get(self, x, z, c, y=None):
        return self.forward(c, y) # use context vector