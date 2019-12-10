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
from YTools.universe.timer import Timer
from tqdm import tqdm

if C.log_file_name:
	if os.path.exists(C.save):
		os.system("rm -rf %s" % C.save)
	os.makedirs(C.save , exist_ok = True)

	log_fil = open(C.log_file_name , "w")

start_time = time.time()
def gettime():
	return time.time() - start_time

def lprint(*args,**kwargs):
	print (*args,**kwargs)

	if C.log_file_name:
		for x in args:
			log_fil.write(str(x) + "\n")

		log_fil.flush()

lprint (C.info)

def pad_string(string , pad_idx = 0):
	'''
		string: (bsz , string_len)
	'''
	if string is None:
		return None

	lenmax = max([len(x) for x in string])

	for i in range(len(string)):
		string[i] += [pad_idx] * (lenmax - len(string[i]))

	return string

def get_a_batch(data , batch_n , device = None):
	if device is None:
		device = C.gpus[0]

	dat = data[batch_n * C.batch_size : (batch_n+1) * C.batch_size]

	g 			= dgl.batch ([x["g"				] for x in dat]) 
	title 		= pad_string([x["title"			] for x in dat])
	decoder_inp = pad_string([x["decoder_inp"	] for x in dat])
	gold 		= pad_string([x["gold"			] for x in dat])

	ent_names = []
	ent_lens  = []
	rels  	  = []
	ent_idx   = []
	rel_idx   = []
	glob_idx  = []
	ent_idx_b = []

	accumed_num = 0
	for x in dat:
		ent_names 	+= x["ent_names"]
		ent_lens  	+= [len(y) for y in x["ent_names"]]
		rels 		+= x["rels"]
		ent_idx_b.append([])
		for i in x["idx_ent"]:
			ent_idx.append(i + accumed_num)
			ent_idx_b[-1].append(i + accumed_num)
		for i in x["idx_rel"]:
			rel_idx.append(i + accumed_num)
		glob_idx.append(x["idx_glob"] + accumed_num)
		
		accumed_num += len(x["idx_ent"]) + len(x["idx_rel"]) + 1

	ent_names 	= pad_string(ent_names)
	ent_idx_b	= pad_string(ent_idx_b , pad_idx = -1)

	return (
		[g , title , ent_names , ent_lens , rels , ent_idx , rel_idx , glob_idx , decoder_inp , ent_idx_b] , 
		gold , 
	)

def valid(net):

	#net = net.eval()

	loss_func = nn.NLLLoss(ignore_index = 0)

	valid_data = data[C.dev_data]
	step = 0
	tot_loss = 0
	batch_number = (len(valid_data) // C.batch_size) + int((len(valid_data) % C.batch_size) != 0)
	pbar = tqdm(range(batch_number) , ncols = 70)
	for batch_n in pbar:
		pbar.set_description_str("(Test)")

		#-----------------get data-----------------
		inputs = []
		golds = []
		for data_device in C.gpus:
			inp , gold = get_a_batch(valid_data , batch_n , data_device)
			inputs.append(inp)
			golds.append(gold)
		assert len(inputs) == len(golds)
		#------------------repadding-----------------

		maxlen_gold = max([ max( [len(x) for x in gold] ) for gold in golds])
		for _i in range(len(inputs)):
			for _j in range(len(golds[_i])): 	#batch
					inputs[_i][-2][_j] 	+= [0] * (maxlen_gold - len(golds[_i][_j]))
					golds[_i][_j] 	 	+= [0] * (maxlen_gold - len(golds[_i][_j]))
			golds[_i] = tc.LongTensor(golds[_i]).cuda(C.gpus[_i])
			for _j in range(1,len(inputs[_i])): #first one is graph
				inputs[_i][_j] = tc.LongTensor(inputs[_i][_j]).cuda(C.gpus[_i])

		#-----------------get output-----------------
		
		if len(inputs) == 1:
			y = net(*inputs[0] , attn_method = C.attn_method)
			gold = golds[0]
		else:
			replicas = net.replicate(net.module, net.device_ids[:len(inputs)])
			outputs = net.parallel_apply(replicas, inputs, [{"attn_method" :C.attn_method}] * len(inputs))

			#pdb.set_trace()

			y = tc.cat([x.to(C.gpus[0]) for x in outputs] , dim = 0)
			gold = tc.cat([x.to(C.gpus[0]) for x in golds] , dim = 0)

		#-----------------get loss-----------------
		y = tc.log(y).view(-1 , y.size(-1))
		gold = gold.view(-1)				
		loss = loss_func(y , gold.view(-1))

		tot_loss += float(loss)

		step += 1
		
		pbar.set_postfix_str("loss: %.4f , avg_loss: %.4f" % (float(loss) , tot_loss / step))

	lprint ("valid end. valid loss = %.6f , ppl = %.6f" % (tot_loss / step , math.exp(tot_loss / step)))
	#net = net.train()

def train(net):

	def lr_schedule(epoch_number , optim):
		if C.no_lr_cyc or C.lr_cyc <= 1:
			return
		epoch_number += 4
		lr = C.high_lr - (epoch_number % C.lr_cyc) * ((C.high_lr - C.low_lr) / (C.lr_cyc-1))
		optim.param_groups[0]['lr'] = lr
 
	train_starttime = gettime()

	#optim = tc.optim.SGD(params = net.parameters() , lr = C.lr)
	if C.use_adam:
		optim = tc.optim.Adam(params = net.parameters() , lr = C.lr)
	else:
		optim = tc.optim.SGD(params = net.parameters() , lr = C.lr , momentum = 0.9)

	loss_func = nn.NLLLoss(ignore_index = 0)

	train_data = data[C.train_data]
	step = 0
	
	tot_loss = 0
	batch_number = (len(train_data) // C.batch_size) + int((len(train_data) % C.batch_size) != 0)
	#accumued_loss = None
	for epoch_n in range(C.epoch_number):

		if not C.use_adam:
			lr_schedule(epoch_n , optim)

		lprint ("epoch %d started." % (epoch_n))
		lprint ("now lr = %.3f" % (optim.param_groups[0]['lr']))

		pbar = tqdm(range(batch_number) , ncols = 70)
		for batch_n in pbar:
			pbar.set_description_str("(Train)Epoch %d" % (epoch_n+1))

			#-----------------get data-----------------
			inputs = []
			golds = []
			for data_device in C.gpus:
				inp , gold = get_a_batch(train_data , batch_n , data_device)
				inputs.append(inp)
				golds.append(gold)

			#------------------repadding-----------------

			maxlen_gold = max([ max( [len(x) for x in gold] ) for gold in golds])
			for _i in range(len(inputs)):
				for _j in range(len(golds[_i])): 	#batch
						inputs[_i][-2][_j] 	+= [0] * (maxlen_gold - len(golds[_i][_j]))
						golds[_i][_j] 	 	+= [0] * (maxlen_gold - len(golds[_i][_j]))
				golds[_i] = tc.LongTensor(golds[_i]).cuda(C.gpus[_i])
				for _j in range(1,len(inputs[_i])): #first one is graph
					inputs[_i][_j] = tc.LongTensor(inputs[_i][_j]).cuda(C.gpus[_i])

			#-----------------get output-----------------
			if len(inputs) == 1:
				y = net(*inputs[0] , attn_method = C.attn_method)
				gold = golds[0]
			else:
				replicas = net.replicate(net.module, net.device_ids[:len(inputs)])
				outputs = net.parallel_apply(replicas, inputs, [{"attn_method" :C.attn_method}] * len(inputs))

				#pdb.set_trace()

				y = tc.cat([x.to(C.gpus[0]) for x in outputs] , dim = 0)
				gold = tc.cat([x.to(C.gpus[0]) for x in golds] , dim = 0)

			#-----------------get loss-----------------
			y = tc.log(y).view(-1 , y.size(-1))
			gold = gold.view(-1)
			loss = loss_func(y , gold.view(-1))

			tot_loss += float(loss)

			#if accumued_loss is None:
			#	accumued_loss = loss
			#else:
			#	accumued_loss += loss

			step += 1
			
			#-----------------back prop-----------------
			#if step % C.update_freq == 0:
			if True:
				optim.zero_grad()
				#accumued_loss.backward()
				loss.backward()
				nn.utils.clip_grad_norm_(net.parameters(),C.clip)
				optim.step()

				#del accumued_loss
				#accumued_loss = None

			pbar.set_postfix_str("loss: %.4f , avg_loss: %.4f" % (float(loss) , tot_loss / step))

			
		lprint ("epoch %d ended." % (epoch_n))
		valid(net)

		save_path = os.path.join(C.save , "epoch_%d.pkl" % epoch_n)
		if C.save:
			
			if len(C.gpus) > 1:
				_net = net.module
			else:
				net = net.cpu()
				_net = net

			with open(save_path , "wb") as fil:
				pickle.dump( [_net , epoch_n+1 , optim] , fil )

			if len(C.gpus) == 1:
				net = net.cuda(C.gpus[0])

			os.system("cp %s %s/last.pkl" % (save_path , C.save))
			lprint ("saved...")

	lprint ("tot train time = %.2fs" % (gettime() - train_starttime))

if __name__ == "__main__":

	lprint ("--------------------args------------------")
	for x in C.__dict__:
		lprint ("%s : %s" % (x , repr(C.__dict__[x])))
	lprint ("------------------------------------------\n")

	if C.seed > 0:
		random.seed(C.seed)
		np.random.seed(C.seed)
		tc.manual_seed(C.seed)

	tc.cuda.set_device(C.gpus[0])

	data = dataloader.run(name = C.name , force_reprocess = C.force_reprocess)
	lprint ("got data.")
	lprint ("size of train/valid/test = %d / %d / %d" % (len(data["train"]) , len(data["valid"]) , len(data["test"])))

	sort_idx = data["sort_idx"].cuda(C.gpus[0])
	net = GraphWriter(
		vocab 			= data["vocab"] 			, 
		entity_number 	= Con.max_entity_per_string , 
		dropout 		= C.dropout 				, 
		sort_idx 		= sort_idx 					, 
	)

	net = net.cuda(C.gpus[0])
	if len(C.gpus) > 1:
		net = nn.DataParallel(net , C.gpus)

	lprint ("start Training")

	train(net)

	if C.log_file_name:
		log_fil.close()