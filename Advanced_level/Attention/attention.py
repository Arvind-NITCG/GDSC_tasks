import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Multihead_mask_attention(nn.Module):
    def __init__(self,model_dimension,numberofhead):
        super().__init__()
        assert model_dimension % numberofhead == 0

        self.model_dimension = model_dimension
        self.numberofhead = numberofhead
        self.d_k = model_dimension//numberofhead
        """ We are making a huge matrices for query,key and value later we will chop it so that context is not messed up"""
        W_q = nn.Linear(model_dimension,model_dimension)
        W_k = nn.Linear(model_dimension,model_dimension)
        W_v = nn.Linear(model_dimension,model_dimension)

        W_o = nn.linear(model_dimension,model_dimension)
    
    def forward(self,x,mask=None):
        """mask we give to create lower traingular matrix. Batch size is important as GPU nneds massive data and we pass sentences parrellely but the bascic operations stay same"""
        batch_size , seq_len , _ = x.size()
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        """We have to split it into heads as we are doing multihead attention and take transpose to assign the chunks to each head."""
        Q = Q.view(batch_size,seq_len,self.numberofhead,self.d_k).transpose(1,2)
        K = K.view(batch_size,seq_len,self.numberofhead,self.d_k).transpose(1,2)
        V = V.view(batch_size,seq_len,self.numberofhead,self.d_k).transpose(1,2)

        """using the attention formula : attention=softmax(Q.K(transpose)/sqrt(d_k)).V"""
        score = torch.matmul(Q,K.transpose(-2,-1))
        score = score/math.sqrt(self.d_k)
        """We have to apply the mask too for the decoder architecture."""
        if mask is not None:
            score = score.masked_fill(mask==0 , -1e9)
        # apply softmax now the expomemtial softmax function would turn the negative infinte values to 0 as we have applied for the mask
        attn_probs = F.softmax(score , dim=-1) # we applied dim=-1 so the probabilities sum up across the row to 1

        out = torch.matmul(attn_probs,V)
        """so here the order of the matrices in general are batch size,seq_len,numberofheads,dimension out is having a reversed order due to transpose in order to capture correct contexts..so we return it back to the original order before multipltying ny the large W_o matrix"""

        out = out.transpose(1,2).contiguous()
        #now we flatten back to original tensor as we have the real continuos matrix.
        out = out.view(batch_size,seq_len,self.model_dimension)

        out = self.W_o(out)

        return out
    
     

        
       



        
