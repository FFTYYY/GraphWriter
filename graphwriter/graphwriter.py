import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from .lstm import LSTM
from .graph_encoder import GraphEncoder , MultiheadAttention
#from .graphwriter_old import model
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

class DecoderAttention(nn.Module):
	def __init__(self , d_model = 500, h = 4 , dropout = 0.0):
		super().__init__()

		self.h = h
		self.d_model = d_model
		self.dk = d_model // h

		self.attn = MultiheadAttention(d_model = d_model , h = h , dropout = dropout , actions = "")
		self.WQ = nn.Linear(d_model , d_model , bias = False)
		self.WK = nn.Linear(d_model , d_model , bias = False)
		self.WV = nn.Linear(d_model , d_model , bias = False)
		self.WO = nn.Linear(d_model , d_model , bias = False)

	def forward(self , query , g , glob_idx):
		'''
			query: (bsz , d_model)
			g.ndata["x"] : (n_v , d_model)
		'''
		g.ndata["k"] = self.WK(g.ndata["x"]).view(-1 , self.h , self.dk)
		g.ndata["v"] = self.WV(g.ndata["x"]).view(-1 , self.h , self.dk)

		g.ndata["q"][glob_idx] = self.WQ(query).view(len(glob_idx) , self.h , self.dk)

		g = self.attn(g)
		r = self.WO(g.ndata["x"][glob_idx].view(-1 , self.d_model))
		
		return r


class EntitySelector(nn.Module):
	def __init__(self , d_model):
		super().__init__()
		self.d_model = d_model
		self.WQ = nn.Linear(2*d_model , d_model)
		self.WK = nn.Linear(d_model , d_model)
		
	def forward(self , query , g , ent_idx_in_batch , max_entity_number):
		'''
			query: (bsz , d_model)

			ent_idx_in_batch: form graph nodes of entitys to shape (bsz , n_ent_b)
		'''
		bsz , y_len , d_model = query.size(0) , query.size(1) , self.d_model
		n_ent_b = ent_idx_in_batch.size(1)
		mask = (ent_idx_in_batch != -1).float().view(bsz , n_ent_b , 1).requires_grad_(False)
		
		q = self.WQ(query)
		ent = self.WK(g.ndata["x"][ent_idx_in_batch]) * mask #(bsz , n_ent_b , d_model)

		#pdb.set_trace()

		weight = (q.view(bsz,y_len,1,d_model) * ent.view(bsz,1,n_ent_b,d_model)).sum(-1) #(bsz , y_len , n_ent_b)
		weight -= (1-mask.view(bsz,1,n_ent_b)) * 100000
		weight = F.softmax(weight , dim = -1)

		if n_ent_b < max_entity_number:
			weight = tc.cat([weight , weight.new_zeros(bsz , y_len , max_entity_number - n_ent_b)] , dim = -1)

		return weight


class GraphWriter(nn.Module):
	def __init__(self , vocab , entity_number , sort_idx , d_model = 500 ,  use_title = False , dropout = 0.0):
		super().__init__()

		self.vocab = vocab

		self.d_model = d_model
		self.entity_number = entity_number
		self.sort_idx = sort_idx

		self.emb = nn.Embedding(len(vocab) , d_model , padding_idx = vocab["<pad>"])
		self.emb_drop = nn.Dropout(dropout)

		self.out = nn.Linear(d_model , len(vocab))

		self.ent_encoder = StringEncoder(self.emb , d_model , dropout = dropout)
		self.graph_encoder = GraphEncoder(num_layers = 6, h = 4, d_model = d_model, d_hid = 4*d_model, dropout = dropout)
		self.rel_encoder = self.emb
		#if use_title:
		#	self.title_encoder = StringEncoder(self.emb , d_model , dropout = dropout)

		self.y_embedder = self.emb
		self.decode_cell = nn.LSTMCell(2*d_model , d_model)
		self.decoder_attn = DecoderAttention(d_model = d_model , h = 4 , dropout = dropout)

		self.switch = nn.Linear(2 * d_model , 1)
		self.select_vocab  = nn.Linear(2 * d_model , len(vocab) - entity_number)
		self.select_entity = EntitySelector(d_model)
		#self.select_entity = nn.Linear(2 * d_model , entity_number)

		self.glob = nn.Parameter(tc.rand(d_model))

		self.reset_param()
		#-------------------------------
		#self._out = nn.Linear(2*d_model , vocab_size)

	def reset_param(self):
		nn.init.normal_(self.glob.data , 0 , 1)
		nn.init.normal_(self.emb.weight)

	def forward(
		self , 
		g , title , ent_names , ent_lens , rels , 
		ent_idx , rel_idx , glob_idx , decoder_inp , 
		ent_idx_b , attn_method = "naive"):
		'''
			g: 			a batched graph , node numbers aligned with index of ent_names  
			title: 		(bsz , title_len)
			ent_names: 	(num_ent , name_len)
			ent_lens :  (num_ent)
			rels: 		(num_rel)
			ent_idx:    (num_ent)
			rel_idx:    (num_rel)
			glob_idx:   (bsz)

			ent_idx_b: (bsz , num_ent_per_batch)
			self.sort_idx:  (len_vocab)

			decoder_inp:(bsz , y_len)
		'''

		#

		d_model = self.d_model
		bsz , y_len = list(decoder_inp.size())

		ent_emb = self.ent_encoder(ent_names , ent_lens)  						#(num_ent , d_model)
		rel_emb = self.emb_drop(self.rel_encoder(rels)) 						#(num_rel , d_model)
		glob_emb = self.glob.view(1,d_model).expand(bsz,d_model).contiguous() 	#(bsz , d_model)

		embs = tc.cat([ent_emb , rel_emb , glob_emb] , dim = 0)
		idxs = tc.cat([ent_idx , rel_idx , glob_idx] , dim = 0)
		_ , idxs = tc.sort(idxs)
		g.ndata["x"] = embs[idxs]

		g = self.graph_encoder(g , attn_method)

		y_emb = self.y_embedder(decoder_inp).transpose(0,1) #(y_len , bsz , d_model)
		
		h , c = g.ndata["x"][glob_idx] , g.ndata["x"][glob_idx]
		a = h.new_zeros(h.size())
		output = []
		for i in range(y_len):
			inp = tc.cat([y_emb[i] , a] , dim = -1)
			h,c = self.decode_cell(inp , (h,c))
			a = self.decoder_attn(h , g , glob_idx)
			y = tc.cat([h , a] , dim = -1)
			output.append(y)
		
		output = tc.stack(output , 1) #(bsz , y_len , 2*d_model)

		p = tc.sigmoid(self.switch(output)) #propobility for selecting entity

		select_v = tc.softmax(self.select_vocab (output) , dim = -1)
		select_e = self.select_entity(output , g , ent_idx_b , self.entity_number)
		#select_e = tc.softmax(self.select_entity(output) , dim = -1)

		out = tc.cat([select_v * (1-p) , select_e * p] , dim = -1)
		if self.sort_idx.device != out.device:
			self.sort_idx.to(out.device)
		out = out[:,:,self.sort_idx]

		out = out + out.new_ones(out.size()) * 1e-6	#avoid zeros

		return out

	def generate(self , 
		g , title , ent_names , ent_lens , rels , 
		ent_idx , rel_idx , glob_idx , 
		sos_id , eos_id ,  beam_size = 4 , max_len = 500 , vocab = None , decoded_ent_replace = {} , 
		other_args = {}):
		'''
			g: 			a batched graph , node numbers aligned with index of ent_names  
			title: 		(title_len)
			ent_names: 	(num_ent , name_len)
			ent_lens :  (num_ent)
			rels: 		(num_rel)
			ent_idx:    (num_ent)
			rel_idx:    (num_rel)
			glob_idx:   (1)
		'''

		#pdb.set_trace()
		d_model = self.d_model

		#---------------encoder--------------
		ent_emb = self.ent_encoder(ent_names , ent_lens)  			#(num_ent , d_model)
		rel_emb = self.rel_encoder(rels) 							#(num_rel , d_model)
		glob_emb = self.glob.view(1,d_model)  						#(1 , d_model)

		embs = tc.cat([ent_emb , rel_emb , glob_emb] , dim = 0)
		idxs = tc.cat([ent_idx , rel_idx , glob_idx] , dim = 0)
		_ , idxs = tc.sort(idxs)
		g.ndata["x"] = embs[idxs]

		#pdb.set_trace()
		g = self.graph_encoder(g)

		#---------------decoder--------------

		h , c = g.ndata["x"][glob_idx] , g.ndata["x"][glob_idx]
		a = h.new_zeros(h.size())
		ys = [{
			"toks":[sos_id],
			"h" : h,
			"c" : c,
			"a" : a,
			"log_prob" : 0.,
		}]

		#pdb.set_trace()
		#pdb.set_trace()

		for i in range(max_len):

			if len(ys) <= 0:
				break

			new_ys = []

			flag = False
			for y in ys:
				h,c,a,toks,log_prob = y["h"],y["c"],y["a"],y["toks"],y["log_prob"]
				if len(toks) > 0 and toks[-1] == eos_id:
					new_ys.append(y)
					continue
				flag = True


				#-----------decode--------------
				got_tok = int(toks[-1])
				if got_tok in decoded_ent_replace:
					got_tok = decoded_ent_replace[got_tok]

				y_emb = self.y_embedder(tc.LongTensor([got_tok]).cuda(ent_emb.device)).view(1 , d_model)

				inp = tc.cat([y_emb , a] , dim = -1)
				h,c = self.decode_cell(inp , (h,c))
				a = self.decoder_attn(h , g , glob_idx)
				y = tc.cat([h , a] , dim = -1) 	#(1 , 2*d_model)

				p = tc.sigmoid(self.switch(y)) #propobility for selecting entity

				select_v = tc.softmax(self.select_vocab (y) , dim = -1) #(1,len_voc)
				select_e = self.select_entity(y.unsqueeze(0) , g , ent_idx.unsqueeze(0) , self.entity_number).squeeze(0)

				out = tc.cat([select_v * (1-p) , select_e * p] , dim = -1) #(1 , len_vocab)
				if self.sort_idx.device != out.device:
					self.sort_idx = self.sort_idx.to(out.device)
				out = out[:,self.sort_idx]

				#pdb.set_trace()


				out = out + out.new_ones(out.size()) * 1e-6	#avoid zeros

				#-----------beam--------------

				out = out[0]
				top_idx = list(out.topk(beam_size)[1])
				for idx in top_idx:
					n_log_prob = log_prob + tc.log(out[idx])

					n_toks = toks + [int(idx)]
					new_ys.append({
						"toks" : n_toks,
						"h" : h,
						"c" : c,
						"a" : a,
						"log_prob" : n_log_prob,
					})

			if other_args.get("norm"):
				new_ys.sort(key = lambda x:-(x["log_prob"] / len(x["toks"])))
			else:
				new_ys.sort(key = lambda x:-(x["log_prob"]))

			ys = new_ys[:beam_size]
			if not flag:
				break

		return ys[0]["toks"]
