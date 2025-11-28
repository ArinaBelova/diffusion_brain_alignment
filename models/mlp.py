import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal time embeddings for diffusion models.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    """
    A simple residual block with dense layers.
    Conditioning (time + class) is added to the hidden state.
    That's what Max meant by adding the condition to the data? 
    """
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU() # Swish activation is standard for diffusion
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, cond_emb):
        # x: (bs, hidden_dim)
        # cond_emb: (bs, hidden_dim) -> Combined time and class info
        
        h = self.norm1(x)
        h = h + cond_emb # Add conditioning info
        h = self.linear1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.linear2(h)
        
        return x + h # Residual connection

class ToyDiffusionMLP(nn.Module):
    def __init__(self, data_dim=2, hidden_dim=128, num_classes=8, num_blocks=3):
        super().__init__()
        
        # 1. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. Class Embedding (for CFG)
        # num_classes + 1 because the last index is the 'null' (unconditional) token
        self.class_emb = nn.Embedding(num_classes + 1, hidden_dim)
        self.null_class_idx = num_classes 

        # 3. Input Projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)

        # 4. Residual Backbone
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_blocks)
        ])

        # 5. Output Projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )

    def forward(self, x, t, labels):
        """
        x: (batch_size, 2)
        t: (batch_size,) 
        labels: (batch_size,) - Indices 0-7 for gaussians, 8 for null
        """
        # Embed Time
        t_emb = self.time_mlp(t)
        
        # Embed Class
        c_emb = self.class_emb(labels)
        
        # Combine embeddings (simple addition works well for MLPs)
        cond = t_emb + c_emb

        # Process Input
        h = self.input_proj(x)

        # Pass through backbone
        for block in self.blocks:
            h = block(h, cond)

        return self.output_proj(h)