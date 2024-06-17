import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = 1.0 / (d_k ** 0.5)
    
    def forward(self, query, key, value):
        # query, key, value: [batch_size, seq_len, d_k]
        
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [batch_size, seq_len, seq_len]
        weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len, seq_len]
        output = torch.matmul(weights, value)  # [batch_size, seq_len, d_k]
        
        return output, weights

# Example usage
batch_size = 2
seq_len = 5
d_k = 8

query = torch.randn(batch_size, seq_len, d_k)
key = torch.randn(batch_size, seq_len, d_k)
value = torch.randn(batch_size, seq_len, d_k)

attention = ScaledDotProductAttention(d_k)
output, weights = attention(query, key, value)

print("Output Shape:", output.shape)  # [batch_size, seq_len, d_k]
print("Weights Shape:", weights.shape)  # [batch_size, seq_len, seq_len]