import torch
import math
from torch import nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attention_type, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attention_type, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len]   #if the token is padding, it will be 1, otherwise 0
        _src, _ = self.self_attention(src, src, src, src_mask)
        src     = self.self_attn_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]
        
        _src    = self.feedforward(src)
        src     = self.ff_layer_norm(src + self.dropout(_src))
        #src: [batch_size, src len, hid dim]
        
        return src
    
class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, attention_type, device, max_length = 500, **kwargs):
        super().__init__()
        self.device = device
        self.attention_type = attention_type
        self.tok_embedding = nn.Embedding(input_dim, hid_dim).to(device)
        self.pos_embedding = nn.Embedding(max_length, hid_dim).to(device)
        self.layers        = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, attention_type, device)
                                           for _ in range(n_layers)])
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len    = src.shape[1]
        
        pos        = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, src_len]
        
        src        = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        #src: [batch_size, src_len, hid_dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
        #src: [batch_size, src_len, hid_dim]
        
        return src
            

# Default Transformer attention variant
class GeneralAttention(nn.Module):
    def __init__(self, head_dim, scale=True):
        super().__init__()
        self.scale_factor = head_dim ** 0.5 if scale else 1.0  # Scale by sqrt(d) for numerical stability

    def forward(self, Q, K):
        scores = torch.einsum("bnhd,bmhd->bhnm", Q, K) / self.scale_factor  # Batch-wise matrix multiplication
        return scores
    
class MultiplicativeAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.Wq = nn.Linear(head_dim, head_dim, bias=False)  # Projection for queries
        self.Wk = nn.Linear(head_dim, head_dim, bias=False)  # Projection for keys

    def forward(self, Q, K):
        Q_proj = self.Wq(Q)  # Transform Queries
        K_proj = self.Wk(K)  # Transform Keys
        scores = torch.einsum("bnhd,bmhd->bhnm", Q_proj, K_proj)  # Efficient batch matmul
        return scores

class AdditiveAttention(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.W = nn.Linear(2 * head_dim, head_dim)  # Combined transformation
        self.V = nn.Linear(head_dim, 1, bias=False)  # Output scoring layer

    def forward(self, Q, K):
        # Expand Q and K to match dimensions
        query_len, key_len = Q.shape[2], K.shape[2]

        # Expand Q and K along the sequence dimensions for broadcasting
        Q_exp = Q.unsqueeze(3).expand(-1, -1, query_len, key_len, -1)  # [batch_size, n_heads, query_len, key_len, head_dim]
        K_exp = K.unsqueeze(2).expand(-1, -1, query_len, key_len, -1)  # [batch_size, n_heads, query_len, key_len, head_dim]

        # Concatenate along last dimension
        features = torch.tanh(self.W(torch.cat([Q_exp, K_exp], dim=-1)))  # [batch_size, n_heads, query_len, key_len, head_dim]

        scores = self.V(features).squeeze(-1)  # [batch_size, n_heads, query_len, key_len]
        return scores

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, attention_type, device):  
        super().__init__()
        assert hid_dim % n_heads == 0
        
        self.hid_dim  = hid_dim
        self.n_heads  = n_heads
        self.head_dim = hid_dim // n_heads
        self.attention_type = attention_type
        
        self.fc_q     = nn.Linear(hid_dim, hid_dim)
        self.fc_k     = nn.Linear(hid_dim, hid_dim)
        self.fc_v     = nn.Linear(hid_dim, hid_dim)
        self.fc_o     = nn.Linear(hid_dim, hid_dim)
        
        self.dropout  = nn.Dropout(dropout)
        self.scale    = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
        if attention_type == "general":
            self.attention = GeneralAttention(self.head_dim)
        elif attention_type == "multiplicative":
            self.attention = MultiplicativeAttention(self.head_dim)
        elif attention_type == "additive":
            self.attention = AdditiveAttention(self.head_dim)
        else:
            raise ValueError("Invalid attention type")
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #energy = self.attention(Q, K)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        
        return x, attention

        
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        #x = [batch size, src len, hid dim]
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attention_type, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm  = nn.LayerNorm(hid_dim)
        self.ff_layer_norm        = nn.LayerNorm(hid_dim)
        self.self_attention       = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attention_type, device)
        self.encoder_attention    = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attention_type, device)
        self.feedforward          = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout              = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg     = self.self_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg             = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        #attention = [batch_size, n heads, trg len, src len]
        
        _trg = self.feedforward(trg)
        trg  = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch_size, trg len, hid dim]
        
        return trg, attention
    
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, 
                 pf_dim, dropout, attention_type, device, max_length = 100, **kwargs):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim).to(device)
        self.pos_embedding = nn.Embedding(max_length, hid_dim).to(device)
        self.layers        = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, attention_type, device)
                                            for _ in range(n_layers)])
        self.fc_out        = nn.Linear(hid_dim, output_dim)
        self.dropout       = nn.Dropout(dropout)
        self.scale         = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = trg.shape[0]
        trg_len    = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        #pos: [batch_size, trg len]
        
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        #trg: [batch_size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
            
        #trg: [batch_size, trg len, hid dim]
        #attention: [batch_size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        #output = [batch_size, trg len, output_dim]
        
        return output, attention
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device, **kwargs):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention
    
    def generate(self, src, max_len=100):
        self.eval()
        
        with torch.no_grad():
            # Create the source mask and encode the source input
            src_mask = self.make_src_mask(src)
            enc_src = self.encoder(src, src_mask)
            
            # Initialize the target sequence with the padding index
            trg = torch.ones((src.shape[0], 1), device=self.device, dtype=torch.long) * self.trg_pad_idx
            
            # Pre-allocate the full target sequence tensor
            full_trg = trg.clone()

            for i in range(1, max_len):
                trg_mask = self.make_trg_mask(full_trg)  # Using the full target for masking
                
                # Get the output of the decoder
                output, _ = self.decoder(full_trg, enc_src, trg_mask, src_mask)
                
                # Get the predicted token (highest probability token)
                pred_token = output.argmax(2)[:, -1].unsqueeze(1)
                
                # Append the predicted token to the target sequence
                full_trg = torch.cat((full_trg, pred_token), dim=1)
                
                # Check for EOS token and stop generation if encountered
                if pred_token.item() == self.trg_pad_idx:
                    break

            # Return the generated tokens excluding the initial padding token
            return full_trg[:, 1:]
