import torch
import torch.nn as nn

class BillionModel(nn.Module):
    def __init__(self, vocab_size=50000, embed_dim=2048, num_heads=32, num_layers=24, block_size=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)

# Just defining it
model = BillionModel()
print(sum(p.numel() for p in model.parameters()) / 1e9, "B parameters")
 