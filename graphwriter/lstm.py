import torch as tc
import torch.nn as nn
import torch.nn.functional as F
import pdb

class LSTM(nn.Module):
	def __init__(self , input_size , hidden_size , num_layers , bidrect , dropout = 0.0):
		super().__init__()

		if num_layers <= 1:
			dropout = 0.0
		
		self.rnn = nn.LSTM(input_size = input_size , hidden_size = hidden_size , 
			num_layers = num_layers , batch_first = True , dropout = dropout , 
			bidirectional = bidrect)


		self.number = (2 if bidrect else 1) * num_layers

	def forward(self , x , mask = None, lens = None):
		'''
			x : (bs , sl , is)
			mask : (bs , sl) 
			lens : (bs)
		'''
		assert mask is not None or lens is not None
		if lens is None:
			lens = (mask).long().sum(dim = 1)
		lens , idx_sort = tc.sort(lens , descending = True)
		_ , idx_unsort = tc.sort(idx_sort)

		x = x[idx_sort]
		
		x = nn.utils.rnn.pack_padded_sequence(x , lens , batch_first = True)
		self.rnn.flatten_parameters()
		y , (h , c) = self.rnn(x)
		y , lens = nn.utils.rnn.pad_packed_sequence(y , batch_first = True)

		h = h.transpose(0,1).contiguous() #make batch size first

		y = y[idx_unsort]		#(bs , seq_len , bid * hid_size)
		h = h[idx_unsort]		#(bs , number , hid_size)

		return y , h
