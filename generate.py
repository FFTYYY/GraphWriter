import dataloader
from graphwriter.graphwriter import GraphWriter 
from config import C
import constants as Con
import torch as tc
import torch.nn as nn
import dgl
import pdb
import time
import math
import random
import numpy as np
import os
import pickle

tc.backends.cudnn.enabled = False

def pad_string(string , pad_idx = 0):
	'''
		string: (bsz , string_len)
	'''
	if string is None:
		return None

	lenmax = max([len(x) for x in string])

	for i in range(len(string)):
		string[i] += [pad_idx] * (lenmax - len(string[i]))

	return tc.LongTensor(string)

def post_edit(t):
	t = t.strip().split(" ")

	minlen = 1
	for l in range(len(t)-1 , minlen-1 , -1):
		i = 0
		while i+2*l <= len(t):
			if t[i:i+l] == t[i+l:i+2*l]:
				t = t[:i+l] + t[i+2*l:] #remove(i+l:i+2*l)
			i += 1
	return " ".join(t)

def generate(net , data , vocab):

	net = net.eval()
	gene_fil = open(os.path.join(C.save , C.generated_file) , "w")

	for d in range(len(data)):

		x = data[d]
		
		decoded_ent_replace = {}
		for i in range(len(x["gold"])):
			if "<ent_" in vocab.idx2word[x["gold"][i]]:
				 assert vocab.idx2word[x["decoder_inp"][i+1]] in ["<task>","<method>","<metric>","<material>","<otherscientificterm>",]
				 decoded_ent_replace[x["gold"][i]] = x["decoder_inp"][i+1]

		#pdb.set_trace()

		with tc.no_grad():
			y = net.generate(
				g 			= x["g"], 
				title 		= tc.LongTensor(x["title"])							.cuda(C.gpus[0]), 
				ent_names 	= tc.LongTensor(pad_string(x["ent_names"])) 		.cuda(C.gpus[0]), 
				ent_lens 	= tc.LongTensor([len(y) for y in x["ent_names"]])	.cuda(C.gpus[0]), 
				rels 		= tc.LongTensor(x["rels"])							.cuda(C.gpus[0]), 
				ent_idx 	= tc.LongTensor(x["idx_ent"])						.cuda(C.gpus[0]), 
				rel_idx 	= tc.LongTensor(x["idx_rel"])						.cuda(C.gpus[0]), 
				glob_idx 	= tc.LongTensor([x["idx_glob"]])					.cuda(C.gpus[0]), 
				sos_id 		= vocab.word2idx["<SOS>"], 
				eos_id 		= vocab.word2idx["<EOS>"], 
				beam_size 	= 4 , 
				max_len 	= 200 , 
				decoded_ent_replace = decoded_ent_replace , 
				vocab = vocab , 
				attn_method = C.attn_method , 
				other_args = {
					"norm" : C.beam_norm , 
				}
			)

		res = [vocab.idx2word[u] for u in y]
		res = res[1:-1] #delete <sos> , <eos>
		res = " ".join(res).lower()

		for i in range(len(x["ent_names"])):
			this_name = []
			for u in x["ent_names"][i]:
				if u == 0:#<pad>
					break
				this_name.append(vocab.idx2word[u])
			this_name = " ".join(this_name).lower()

			res = res.replace("<ent_%d>" % i , this_name)
		res = res.replace("<unk> " , "")
		#res = post_edit(res)

		gene_fil.write(res + "\n")
		gene_fil.flush()

		print ("%d out of %d:" % (d+1 , len(data)))
		print (res)
		print ()
	gene_fil.close()



if __name__ == "__main__":

	if C.seed > 0:
		random.seed(C.seed)
		np.random.seed(C.seed)
		tc.manual_seed(C.seed)
		
	save_path = os.path.join(C.save , C.generate_from)

	with open(save_path , "rb") as fil:
		net , _ , _ = pickle.load(fil)

	net = net.to(C.gpus[0])
	data = dataloader.run(name = C.name , force_reprocess = C.force_reprocess)

	generate(net , data[C.test_data] , data["vocab"])





