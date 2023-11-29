from typing import Any
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl




class NN(pl.LightningModule):
    def __init__(self, input_size, num_classes) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    def train_step(self, batch, batch_idx):
        x, y = batch
        x.reshape(x.size(0), -1)
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss

