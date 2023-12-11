import torch.nn as nn
class PurchaseClassifier(nn.Module):
    def __init__(self, num_classes=100):
        super(PurchaseClassifier, self).__init__()

        self.features = nn.Sequential(
                nn.Linear(600, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
                nn.Tanh(),
                nn.Linear(256, 128),
                nn.Tanh(),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        hidden_out = self.features(x)
        out = self.classifier(hidden_out)

        return out
class AttackModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=64,out_classes=2):
        super(AttackModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_classes)
        )
    def forward(self, x):
        out = self.classifier(x)
        return out