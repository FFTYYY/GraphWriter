import sys
import fastNLP
import torch as tc
from torch import nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import pdb

def Attention(Q, K, V , q_mas , k_mas , att_mas = None):
	'''
		Q,K,V : (bs,h,n,dk)
		q_mas : (bs,1,n,1)
		k_mas : (bs,1,n,1)
	'''
	bs,h,n,d = Q.size()

	y = tc.matmul(Q , K.transpose(-1, -2)) / (d**0.5) #(bs,h,n,n)

	#mas = (tc.matmul(q_mas.long() , k_mas.long().transpose(-1, -2)) != 0).float()
	#mas = q_mas * k_mas.transpose(-1, -2)
	mas = 1

	if att_mas is not None:
		mas = mas * att_mas.view(bs,1,n,n)

	y = y.masked_fill((1-mas).bool() , float("-inf"))
	
	y = F.softmax(y, dim = -1) * mas

	#y = y.new_ones(y.size()) * tc.eye(n,n).to(y.device).view(1,1,n,n)

	y = y.matmul(V) * k_mas

	return y

class MultiHeadAttention(nn.Module):
	def __init__(self , h = 4 , d_model = 512 , drop_p = 0.0):
		super(MultiHeadAttention, self).__init__()

		self.WQ = nn.Linear(d_model, d_model, bias = False)
		self.WK = nn.Linear(d_model, d_model, bias = False)
		self.WV = nn.Linear(d_model, d_model, bias = False)

		self.WO = nn.Linear(d_model, d_model, bias = False)

		self.drop = nn.Dropout(drop_p)
	#	self.drop2 = nn.Dropout(dorp_p)

		#-----hyper params-----

		self.dk = d_model // h
		self.h = h
		self.d_model = d_model

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_normal_(self.WQ.weight.data , gain = 1.0)
		nn.init.xavier_normal_(self.WK.weight.data , gain = 1.0)
		nn.init.xavier_normal_(self.WV.weight.data , gain = 1.0)
		nn.init.xavier_normal_(self.WO.weight.data , gain = 1.0)
		pass

	def forward(self , Q , K , V , q_mas , k_mas = None , att_mas = None):
		'''
			Q: bs , n , d
			mas : bs , n , 1
			mas_att : bs , n , n
		'''
		#pdb.set_trace()

		#pdb.set_trace()
		bs , n , d = Q.size()
		h = self.h
		q_mas = q_mas.view(bs,n,1,1).transpose(1,2)					#(bs,h,n,1)
		if k_mas is None: 
			k_mas = q_mas

		Q = q_mas * self.WQ(Q).view(bs,n,h,self.dk).transpose(1,2)	#(bs,h,n,dk)
		K = k_mas * self.WK(K).view(bs,n,h,self.dk).transpose(1,2)	#(bs,h,n,dk)
		V = k_mas * self.WV(V).view(bs,n,h,self.dk).transpose(1,2)	#(bs,h,n,dk)

		y = Attention(Q , K , V, q_mas , k_mas , att_mas)
		#y = V

		y = y.view(bs,h,n,self.dk).transpose(1,2).contiguous().view(bs,n,d)

		y = self.WO(y) * k_mas.contiguous().view(bs,n,1)

		return y


class FFN(nn.Module):
	def __init__(self, d_model = 512 , d_hid = 512 , drop_p = 0.0):
		super(FFN, self).__init__()

		self.d_hid = d_hid
		self.L1 = nn.Linear(d_model , d_hid , bias = True)
		self.L2 = nn.Linear(d_hid , d_model , bias = True)
		self.drop = nn.Dropout(drop_p)		
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_normal_(self.L1.weight.data)
		nn.init.xavier_normal_(self.L2.weight.data)
		self.L1.bias.data.fill_(0)
		self.L2.bias.data.fill_(0)

	def forward(self , x , mas):
		x = self.drop(F.relu(self.L1(x)))
		x = self.L2(x)
		x = x * mas
		return x

class Decoder_Layer(nn.Module):
	def __init__(self , d_model = 512 , d_hid = 512 , h = 4 , drop_p = 0.2 , extra_layers = []):
		super(Decoder_Layer, self).__init__()

		self.self_att  = MultiHeadAttention(h = h , d_model = d_model , drop_p = drop_p)
		self.layernorm_1 = nn.LayerNorm([d_model])
		self.drop_1 = nn.Dropout(drop_p)

		self.extra_layers = nn.ModuleList([
			lay(*par) 
			for lay , par in extra_layers
		])
		self.layernorm_3 = nn.LayerNorm([d_model])
		self.drop_3 = nn.Dropout(drop_p)


		self.ffn = FFN(d_model = d_model , d_hid = d_hid , drop_p = drop_p)
		self.layernorm_2 = nn.LayerNorm([d_model])
		self.drop_2 = nn.Dropout(drop_p)

	def reset_parameters(self):
		self.self_att.reset_parameters()
		for x in self.exter_layer:
			x.reset_parameters()
		self.ffn.reset_parameters()

	def forward(self, x , seq_mas , att_mas = None , select_params = None):

		bsz , slen , d = x.size()

		out1 = self.self_att(x , x , x , seq_mas , att_mas = att_mas)
		x = self.layernorm_1(x + self.drop_1(out1))
		x *= seq_mas

		if len(self.extra_layers) > 0:
			out3 = 0
			for i in range(len(self.extra_layers)):
				layer = self.extra_layers[i]
				par = select_params[i]
				got_res = layer(x , *par)

				out3 = out3 + got_res

			out3 = out3 * seq_mas
			x = self.layernorm_3(x + self.drop_3(out3))

		out2 = self.ffn(x , seq_mas)
		x = self.layernorm_2(x + self.drop_2(out2))
		x *= seq_mas

		return x


class Decoder(nn.Module):
	def __init__(self , num_layers = 4 , d_model = 500 , d_hid = 1024 , h = 5 , drop_p = 0.1 , extra_layers = []):
		super(Decoder, self).__init__()

		self.pos_emb = nn.Parameter(tc.zeros(1024 , d_model))

		self.dec_layer = nn.ModuleList([
			Decoder_Layer(d_model = d_model , d_hid = d_hid , h = h , drop_p = drop_p , extra_layers = extra_layers) 
			for _ in range(num_layers)
		])


		#-----hyper params-----
		self.d_model = d_model
		self.num_layers = num_layers

	def reset_parameters(self):
		for x in self.dec_layer:
			x.reset_parameters()
		nn.init.normal_( self.pos_emb.weight.data , 0 , 0.01)

	def forward(self , x , seq_mas , att_mas = None , select_params = None):
		'''
			x : (bs , n , emb_siz)
		'''
		bs , n , d = x.size()

		seq_mas = seq_mas.view(bs , n , 1).float()
		if att_mas is not None:
			att_mas = att_mas.view(bs , n , n).float()
		if att_mas is None:
			att_mas = 1
		att_mas = att_mas * tc.tril(x.new_ones(bs,n,n))

		x = x + self.pos_emb[:n,:].view(1,n,d)

		for i in range(self.num_layers):
			x = self.dec_layer[i](x  ,seq_mas , att_mas , select_params = select_params) #(bs , len , d_model)

		return x

