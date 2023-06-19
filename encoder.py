import torch.nn.functional as F
import torch.nn as nn
import torch
from math import sqrt
from transformers import AutoConfig, AutoTokenizer, BertModel

text = 'hello my name is Udbhav'
model_ckpt = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = BertModel.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
inputs = tokenizer(text,return_tensors='pt',add_special_tokens=False)
inputs_embeds = token_emb(inputs.input_ids)



def scaled_dot_products(query,key,value,mask):

    dim_k = query.size(-1)
    scores = torch.bmm(query,key.transpose(1,2))/sqrt(dim_k)
    if mask is not None: # adding masking functinoality for decoder
        scores.masked_fill(mask == 0, float('-inf')) 
    weights = F.softmax(scores,dim=-1)
    return torch.bmm(weights,value)

class AttentionHead(nn.Module):

    def __init__(self,embed_dim,head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim,head_dim)
        self.k = nn.Linear(embed_dim,head_dim)
        self.v = nn.Linear(embed_dim,head_dim)

    def forward(self,hidden_state):
        attn_outputs = scaled_dot_products(self.q(hidden_state), 
                                           self.k(hidden_state), 
                                           self.v(hidden_state))
        return attn_outputs
    
class MultiAttention(nn.Module):

    def __init__(self,config):

        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim,head_dim) for _
                                    in range(num_heads)])
        self.output_linear = nn.Linear(embed_dim,embed_dim)

    def forward(self,hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads],dim=-1)
        x = self.output_linear(x)
        return x
    

multihead_attn = MultiAttention(config)
attn_output = multihead_attn(inputs_embeds)
# print(attn_output.size())
        
class FeedForward(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size,
                                  config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size,
                                  config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self,x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_output)
# print(ff_outputs.size())


#Putting it altogether - PreNorm Formulation


class TransformersEncoderLayer(nn.ModuleList):

    def __init__(self,config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self,x):
        #apply layer norm and then copy input into q,k,v
        hidden_state = self.layer_norm_1(x)
        #apply attention with skip connection
        x = x + self.attention(hidden_state)
        #apply feed forward with skip connection
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
    
encoder_layer = TransformersEncoderLayer(config)
# print(inputs_embeds.shape, encoder_layer(inputs_embeds).size())

#positional Encodings

class Embedding(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size,
                                             config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size,eps = 1e-12)
        self.dropout = nn.Dropout()

    def forward(self,input_ids):
        #create position ids for input sequence
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len,
                                    dtype=torch.long).unsqueeze(0)
        #create token and position embeddings
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #combine token and position embeddings
        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
embedding_layer = Embedding(config)
# print(embedding_layer(inputs.input_ids).size())

#Putting it altogether forming the final encoder block

class TransformerEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embeddings = Embedding(config)
        self.layers = nn.ModuleList([TransformersEncoderLayer(config)
                                     for _ in range(config.num_hidden_layers)])
        
    def forward(self,x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)

            return x
        
#sanity check
encoder = TransformerEncoder(config)
# print(encoder(inputs.input_ids).size())

# For Sequence Classification

class TransformerForSequenceClassification(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size,
                                    config.num_labels)

    def forward(self,x):
        x = self.encoder(x)[:,0,:]
        x = self.dropout(x)
        x = self.classifier(x)
        return x 
    
# before init the model we have to know num_classes 
config.num_labels = 3
encoder_classifier = TransformerForSequenceClassification(config)
print(encoder_classifier(inputs.input_ids).size())


# Coming to the Decoder