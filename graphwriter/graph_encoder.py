import torch as tc
from torch import nn
from torch.nn import functional as F
import math
import pdb
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax

class MultiheadAttention(nn.Module):
	def __init__(self , h = 4 , d_model = 500 , dropout = 0.0 , actions = "qkvo"):
		super().__init__()
		
		self.d_model = d_model
		self.h = h
		self.dk = d_model // h

		if "q" in actions:
			self.WQ = nn.Linear(d_model , d_model , bias = False)
		if "k" in actions:
			self.WK = nn.Linear(d_model , d_model , bias = False)
		if "v" in actions:
			self.WV = nn.Linear(d_model , d_model , bias = False)
		if "o" in actions:
			self.WO = nn.Linear(d_model , d_model , bias = False)
		self.actions = actions

		self.attn_drop = nn.Dropout(dropout)
		self.drop = nn.Dropout(dropout)

	def propagate_attention(self, g):
		'''
			copied from gqp
		'''

		g.apply_edges(fn.u_mul_v('q', 'k', 'e'))
		e = (g.edata['e'].sum(dim = -1 , keepdim = True)) / math.sqrt(self.dk)
		
		g.edata['e'] = self.attn_drop(edge_softmax(g, e))

		g.update_all(
			fn.u_mul_e('v', 'e', 'e'),
			fn.sum('e', 'v')
		)

	def propagate_attention_old(self, g):
		'''
			copied from dgl document
		'''

		def message_func(edges):
			return {'score': ((edges.src['k'] * edges.dst['q'])
							  .sum(-1, keepdim=True)),
					'v': edges.src['v']}

		def reduce_func(nodes):
			v = nodes.mailbox['v']
			att = F.softmax(nodes.mailbox['score'] / (self.dk ** 0.5), 1)
			return {'v': (att * v).sum(1)}

		g.send_and_recv(g.edges(), message_func, reduce_func)


	def forward(self , g , attn_method = "naive"):
		if "q" in self.actions:
			g.ndata["q"] = self.WQ(g.ndata["x"]).view(-1 , self.h , self.dk)
		if "k" in self.actions:
			g.ndata["k"] = self.WK(g.ndata["x"]).view(-1 , self.h , self.dk)	
		if "v" in self.actions:
			g.ndata["v"] = self.WV(g.ndata["x"]).view(-1 , self.h , self.dk)

		if attn_method == "naive":
			self.propagate_attention_old(g)
		else:
			self.propagate_attention(g)
		
		g.ndata["v"] = self.drop(g.ndata["v"])

		if "o" in self.actions:
			g.ndata["x"] = self.WO(g.ndata["v"].view(-1 , self.d_model))

		return g

def gelu(x):
	return 0.5 * x * (1 + tc.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tc.pow(x, 3))))

class FFN(nn.Module):
	def __init__(self , d_model , d_hid , dropout):
		super().__init__()

		self.L1 = nn.Linear(d_model , d_hid)
		self.L2 = nn.Linear(d_hid , d_model)
		self.drop = nn.Dropout(dropout)

	def forward(self , x):
		x = gelu(self.L1(x))
		x = self.drop(x)
		x = self.L2(x)

		return x

class Encoder_Layer(nn.Module):
	def __init__(self , h = 4 , d_model = 500 , d_hid = 2000 , dropout = 0.0):
		super().__init__()

		self.attn = MultiheadAttention(h , d_model , dropout)
		self.ffn = FFN(d_model , d_hid , dropout)

		self.ln1 = nn.LayerNorm([d_model])
		self.ln2 = nn.LayerNorm([d_model])

	def forward(self , g , attn_method = "naive"):

		old_x = g.ndata["x"]
		g = self.attn(g , attn_method)
		g.ndata["x"] = self.ln1(g.ndata["x"] + old_x)

		g.ndata["x"] = self.ln2(self.ffn(g.ndata["x"]) + g.ndata["x"])

		return g


class GraphEncoder(nn.Module):
	def __init__(self , num_layers = 6, h = 4, d_model = 500, d_hid = 2000, dropout = 0.0):
		super(GraphEncoder, self).__init__()

		self.enc_layer = nn.ModuleList([Encoder_Layer(h, d_model, d_hid, dropout) for _ in range(num_layers)])

	def forward(self, g , attn_method = "naive"):
		'''
			"x" in g.ndata
		'''

		for layer in self.enc_layer:
			x = layer(g , attn_method)

		return x
