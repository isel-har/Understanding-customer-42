import torch.nn as nn
import torch

class TextCNN(nn.Module):
    """
    Multi-scale TextCNN.
 
    Key improvements over the original:
      - Parallel convolution branches with kernel sizes [3, 5, 7] capture
        short, medium, and longer n-gram patterns simultaneously.
      - Batch normalisation after each convolution stabilises training and
        acts as a mild regulariser, letting you reduce dropout slightly.
      - Global max-pool + global average-pool per branch (instead of a fixed
        MaxPool1d(2)) removes the hard dependency on seq_len and generally
        yields better generalisation.
      - The final feature vector is: num_kernels × out_channels × 2
        (factor-of-2 comes from concatenating max- and avg-pool outputs).
    """
 
    def __init__(
        self,
        embed_dim: int,
        out_channels: int,
        output_size: int,
        kernel_sizes: list[int] | None = None,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]
 
        # One conv branch per kernel size
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, out_channels, k, padding="same"),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            )
            for k in kernel_sizes
        ])
 
        # After global max+avg pool we get 2 * out_channels per branch
        fc_input_dim = len(kernel_sizes) * out_channels * 2
 
        self.classifier = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, output_size),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, embed_dim, seq_len)  — already permuted in the data-prep step
        branch_outputs = []
        for branch in self.branches:
            h = branch(x)                           # (batch, out_channels, seq_len)
            h_max = h.max(dim=-1).values            # (batch, out_channels)
            h_avg = h.mean(dim=-1)                  # (batch, out_channels)
            branch_outputs.append(torch.cat([h_max, h_avg], dim=-1))
 
        out = torch.cat(branch_outputs, dim=-1)     # (batch, fc_input_dim)
        return self.classifier(out)