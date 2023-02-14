import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MEGA (nn.Module):
    def __init__(self, args, n_entity, n_relation,n_weight):
        super(MEGA, self).__init__()
        self._parse_args(args, n_entity, n_relation,n_weight)

        self.trans = nn.Linear(768, self.dim, bias=False)

        self.entity_emb = nn.Embedding(self.n_entity, self.dim)

        self.word_emb = nn.Embedding(self.n_entity, self.dim)

        self.relation_emb = nn.Embedding(self.n_relation, self.dim)
        self.weight_emb = nn.Embedding(self.n_weight, self.dim)
        
        self.attention = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )

        self.attention_a = nn.Sequential(
                nn.Linear(self.dim*2, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, self.dim, bias=False),
                nn.ReLU(),
                nn.Linear(self.dim, 1, bias=False),
                nn.Sigmoid(),
                )


        self.gate = nn.Sequential(
                nn.Linear(self.dim, self.dim, bias=False),
                nn.Tanh(),
                nn.Linear(self.dim, 1, bias=False),
                )
    
        
        self._init_weight()

    def build_word_emb(self,pretrain_emb):
        trans_emb = self.trans(pretrain_emb)
        trans_emb = trans_emb.sum(dim=1)
        return trans_emb
                
    def forward(
        self,
        items: torch.LongTensor,
        user_triple_set: list,
        user_triple_set_e: list,
        item_triple_set: list,
        item_triple_set_i: list,
    ):       
        user_embeddings = []
        
        # [batch_size, triple_set_size, dim]
        # user_embeddings_words = self.word_emb(user_triple_set[0][0])
        # user_embeddings.append(user_embeddings_words.mean(dim=1))

        user_emb_0 = self.entity_emb(user_triple_set[0][0]) 
        # [batch_size, dim]
        user_embeddings.append(user_emb_0.mean(dim=1))
        
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set[0][i]) 
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(user_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set[2][i]) 
            # [batch_size, dim]
            user_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_i)
        
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(user_triple_set_e[0][i]) 
            # [batch_size, triple_set_size, dim]
            r_emb = self.weight_emb(user_triple_set_e[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(user_triple_set_e[2][i]) 
            # [batch_size, dim]
            user_emb_i = self._knowledge_attention_a(h_emb, r_emb, t_emb)
            user_embeddings.append(user_emb_i)


        item_embeddings = []

        # item_embeddings_words = self.word_emb(items)
        # item_embeddings.append(item_embeddings_words)
        
        # [batch size, dim]
        item_emb_origin = self.entity_emb(items)
        # item_emb_0 = self.entity_emb(item_triple_set_i[0][0])
        #     # [batch_size, dim]
        # item_embeddings.append(item_emb_0.mean(dim=1))
        item_embeddings.append(item_emb_origin)
        
        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.relation_emb(item_triple_set[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_i)

        for i in range(self.n_layer):
            # [batch_size, triple_set_size, dim]
            h_emb = self.entity_emb(item_triple_set_i[0][i])
            # [batch_size, triple_set_size, dim]
            r_emb = self.weight_emb(item_triple_set_i[1][i])
            # [batch_size, triple_set_size, dim]
            t_emb = self.entity_emb(item_triple_set_i[2][i])
            # [batch_size, dim]
            item_emb_i = self._knowledge_attention_a(h_emb, r_emb, t_emb)
            item_embeddings.append(item_emb_i)

        
        # if self.n_layer > 0 and (self.agg == 'sum' or self.agg == 'pool'):
        #      # [batch_size, triple_set_size, dim]
        #     item_emb_0 = self.entity_emb(item_triple_set[0][0])
        #     # [batch_size, dim]
        #     item_embeddings.append(item_emb_0.mean(dim=1))
            
        scores = self.predict(user_embeddings, item_embeddings)
        return scores
    
    
    def predict(self, user_embeddings, item_embeddings):
        e_u = user_embeddings[0]
        e_v = item_embeddings[0]
    
        if self.agg == 'concat':
            if len(user_embeddings) != len(item_embeddings):
                raise Exception("Concat aggregator needs same length for user and item embedding")
            for i in range(1, len(user_embeddings)):
                e_u = torch.cat((user_embeddings[i], e_u),dim=-1)
            for i in range(1, len(item_embeddings)):
                e_v = torch.cat((item_embeddings[i], e_v),dim=-1)
        elif self.agg == 'sum':
            for i in range(1, len(user_embeddings)):
                e_u += user_embeddings[i]
            for i in range(1, len(item_embeddings)):
                e_v += item_embeddings[i]
        elif self.agg == 'pool':
            for i in range(1, len(user_embeddings)):
                e_u = torch.max(e_u, user_embeddings[i])
            for i in range(1, len(item_embeddings)):
                e_v = torch.max(e_v, item_embeddings[i])
        else:
            raise Exception("Unknown aggregator: " + self.agg)
        
        scores = (e_v * e_u).sum(dim=1)
        scores = torch.sigmoid(scores)
        return scores
    
    
    def _parse_args(self, args, n_entity, n_relation,n_weight):
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.n_weight = n_weight
        self.dim = args.dim
        self.n_layer = args.n_layer
        self.agg = args.agg
        
        
    def _init_weight(self):
        # init embedding
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.weight_emb.weight)
        # init attention
        for layer in self.attention:
            if isinstance(layer,nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    
    def _knowledge_attention(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        att_weights = self.attention(torch.cat((h_emb,r_emb),dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights,dim=-1)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [batch_size, triple_set_size]
        # att = F.softmax(self.gate(emb_i),dim=-1)
        # [batch_size, dim]
        # emb_t = (att.transpose(1,2)).matmul(emb_i).squeeze(1)
        # emb_t = emb_i.mm(att.unsqueeze(-1)).sum(1)
        # [batch_size, dim]
        emb_i = emb_i.sum(1)
        # emb_i = torch.sum(t_emb * torch.tanh(h_emb + r_emb), dim=1)
        return emb_i

    def _knowledge_attention_a(self, h_emb, r_emb, t_emb):
        # [batch_size, triple_set_size]
        f_emb = h_emb * t_emb

        att_weights = self.attention_a(torch.cat((f_emb,r_emb),dim=-1)).squeeze(-1)
        # [batch_size, triple_set_size]
        att_weights_norm = F.softmax(att_weights,dim=-1)
        # [batch_size, triple_set_size, dim]
        emb_i = torch.mul(att_weights_norm.unsqueeze(-1), t_emb)
        # [batch_size, triple_set_size]
        # att = F.softmax(self.gate(emb_i),dim=-1)
        # [batch_size, dim]
        # emb_t = (att.transpose(1,2)).matmul(emb_i).squeeze(1)
        # emb_t = emb_i.mm(att.unsqueeze(-1)).sum(1)
        # [batch_size, dim]
        emb_i = emb_i.sum(1)
        return emb_i
    
    