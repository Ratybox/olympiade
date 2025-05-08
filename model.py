import torch
import torch.nn as nn


class CoughVIDModel(nn.Module):
  def __init__(self, base_model, mfcc_in_shape):
    super().__init__()
    self.base_model = base_model
    self.mfcc_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(mfcc_in_shape[0]*mfcc_in_shape[1], 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 1000),
        nn.ReLU(),
        nn.Dropout(0.2),
    )
    self.classifier = nn.Sequential(
        nn.Linear(2000, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 2),
    )
  
  def forward(self, img, mfcc):
    out_1 = self.base_model(img)
    out_2 = self.mfcc_model(mfcc)
    out_merged = torch.cat([out_1, out_2], dim=1)
    out = self.classifier(out_merged)
    return out