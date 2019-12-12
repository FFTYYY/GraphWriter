import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from .lstm import LSTM
import pdb

class StringEncoder(nn.Module):
	def __init__(self , embed_layer , d_model , dropout = 0.0):
		super().__init__()

		self.d_model = d_model

		self.emb = embed_layer
		self.emb_drop = nn.Dropout(dropout)

		self.lstm = LSTM(d_model , d_model // 2 , 1 , True , dropout = dropout)

	def forward(self , ent_names, ent_lens):
		'''
			ent_names: 	(num_ent , name_len)
			ent_lens :  (num_ent)
		'''
		bsz , d_model = len(ent_lens) , self.d_model

		y = self.emb_drop(self.emb(ent_names))
		y = self.lstm(y , lens = ent_lens)[1].view(bsz , d_model)

		return y

class EntitySelector(nn.Module):
	def __init__(self , d_model , give_me_result = False):
		super().__init__()
		self.d_model = d_model
		self.WQ = nn.Linear(d_model , d_model)
		self.WK = nn.Linear(d_model , d_model)

		self.give_me_result = give_me_result

		if give_me_result:
			self.WV = nn.Linear(d_model , d_model)
			self.WO = nn.Linear(d_model , d_model)
			self.l_norm = nn.LayerNorm(d_model)
		
	def forward(self , query , ent_emb , ent_idx_in_batch , max_entity_number):
		'''
			query: (bsz , len , d_model)
			ent_emb: (num_ent , d_model)

			ent_idx_in_batch: form graph nodes of entitys to shape (bsz , n_ent_b) , None if bsz = 1
		'''
		bsz , d_model = query.size(0) , self.d_model
		query = query.view(bsz , -1 , d_model)
		y_len = query.size(1)

		if ent_idx_in_batch is not None:
			n_ent_b = ent_idx_in_batch.size(1)
			mask = (ent_idx_in_batch != -1).float().view(bsz , n_ent_b , 1).requires_grad_(False)
			ent_emb = ent_emb[ent_idx_in_batch]
		else:
			assert bsz == 1
			n_ent_b = ent_emb.size(0)
			mask = ent_emb.new_ones(1 , n_ent_b , 1).float().requires_grad_(False)
			ent_emb = ent_emb.view(1 , n_ent_b , d_model)

		q = self.WQ(query)
		ent = self.WK(ent_emb) * mask #(bsz , n_ent_b , d_model)

		#pdb.set_trace()

		weight = (q.view(bsz,y_len,1,d_model) * ent.view(bsz,1,n_ent_b,d_model)).sum(-1) #(bsz , y_len , n_ent_b)
		weight -= (1-mask.view(bsz,1,n_ent_b)) * 100000
		weight = F.softmax(weight , dim = -1) #(bsz , y_len , n_ent_b)

		#if need result, make a weighted sum
		if self.give_me_result:
			v = self.WK(ent_emb) * mask #(bsz , n_ent_b , d_model)
			weight = weight.view(bsz , y_len , 1 , n_ent_b)
			v = v.view(bsz , 1 , n_ent_b , d_model)
			v = tc.matmul(weight * (d_model**-0.5) , v).view(bsz , y_len , d_model)
			v = self.l_norm(self.WO(v))
			return v

		#else, make weight padded and return it
		if n_ent_b < max_entity_number:
			weight = tc.cat([weight , weight.new_zeros(bsz , y_len , max_entity_number - n_ent_b)] , dim = -1)

		return weight