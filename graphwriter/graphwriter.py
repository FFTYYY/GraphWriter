import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from .lstm import LSTM
from .graph_encoder import GraphEncoder , MultiheadAttention
#from .graphwriter_old import model
import pdb
from .transformer_sublayers import Decoder
from .other_layers import StringEncoder , EntitySelector

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
		self.decoder = Decoder(
			num_layers 	= 4 , 
			d_model 	= d_model , 
			d_hid 		= 1024 , 
			h 			= 5 , 
			drop_p 		= dropout , 
			extra_layers = [ 
				[EntitySelector , [d_model , True]] , 
			] , 
		)

		self.switch = nn.Linear(d_model , 1)
		self.select_vocab  = nn.Linear(d_model , len(vocab) - entity_number)
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
		ent_emb = g.ndata["x"]

		y_emb = self.y_embedder(decoder_inp) #(bsz , y_len , d_model)
		output = self.decoder(y_emb , decoder_inp != 0 , select_params = [
			[ent_emb , ent_idx_b , self.entity_number]
		])
		
		p = tc.sigmoid(self.switch(output)) #propobility for selecting entity

		select_v = tc.softmax(self.select_vocab (output) , dim = -1)
		select_e = self.select_entity(output , ent_emb , ent_idx_b , self.entity_number)

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
				a = self.decoder_attn(h , g)
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

