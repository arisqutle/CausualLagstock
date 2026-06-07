# model_wrapper.py

import torch.nn as nn
from causal_discovery import STACD

class FastModel(nn.Module):
    def __init__(self, num_horizons: int = 5):
        super().__init__()
        self.num_horizons = int(num_horizons)
        self.num_classes = 3
        self.stacd = STACD(text_emb_dim=768)

        self.cls = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_horizons * self.num_classes)
        )

    def forward(self, text_emb, event_types, timestamps):
        event_reprs, causal_info = self.stacd(
            text_emb, event_types, timestamps
        )

        final = event_reprs[:, -1, :]
        logits = self.cls(final).view(final.shape[0], self.num_horizons, self.num_classes)

        return logits, causal_info
