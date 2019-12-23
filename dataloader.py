from fastNLP import Vocabulary
import os.path as path
import os
import pickle
import dgl
import constants as Con
import torch as tc
import pdb
import random

this_dir = path.dirname(__file__)

train_path 	= path.join(this_dir , "./data/preprocessed.train.tsv")
test_path 	= path.join(this_dir , "./data/preprocessed.test.tsv" )
valid_path 	= path.join(this_dir , "./data/preprocessed.val.tsv"  )

cache_path = path.join(this_dir , "./data_cache/")

def ent_type2num(typ):
	table = {
		"<task>" 				: 0 ,
		"<method>" 				: 1 ,
		"<metric>" 				: 2 ,
		"<material>" 			: 3 ,
		"<otherscientificterm>" : 4 ,
	}

	if table.get(typ) is None:
		raise Exception("unknown type: %s" % typ)

	return table[typ]

def decoder_inp_process(x):
	'''
		only remain the entity type
	'''
	if "<" in x and "_" in x and ">" in x:
		return x.split("_")[0] + ">"
	return x
def gold_process(x):
	'''
		only remain the entity number
	'''
	if "<" in x and "_" in x and ">" in x:
		return "<ent_" + x[1:-1].split("_")[1] + ">"
	return x

def make_graph(x):
	g = dgl.DGLGraph()

	num_entity = len(x["ent_names"])
	num_rel = 2*len(x["rels"])

	g.add_nodes(num_entity + num_rel + 1)

	idx_ent = range(num_entity)
	idx_rel = range(num_entity, num_entity + num_rel)
	idx_glob = num_entity + num_rel

	rels = []
	for u,r,v in x["rels"]:
		rels.append(r+1)
		rels.append(r+1+Con.num_relations)

		idx_fr = num_entity + len(rels)-2
		idx_bk = num_entity + len(rels)-1

		g.add_edge(u,idx_fr)
		g.add_edge(idx_fr,v)

		g.add_edge(v,idx_bk)
		g.add_edge(idx_bk,u)

	for u in range(num_entity):
		g.add_edge(u,idx_glob)
		g.add_edge(idx_glob,u)

	for u in range(num_entity + num_rel + 1):
		g.add_edge(u,u)

	x["rels"] 		= rels
	x["idx_ent"] 	= idx_ent
	x["idx_rel"] 	= idx_rel
	x["idx_glob"] 	= idx_glob
	x["g"] 			= g

	return x

vocab = Vocabulary(min_freq = 2)

def get_data(dataset_path , shuffle = True):

	dataset = []

	with open(dataset_path , "r" , encoding = "utf-8") as fil:
		for line in fil:
			line = line.lower().strip().split("\t")

			title 		= line[0]
			ent_names 	= line[1]
			ent_types 	= line[2]
			rels 		= line[3]
			gold 		= line[4]

			#pdb.set_trace()

			title 		= title.lower().strip().split(" ")
			ent_names 	= [x.strip().split(" ") for x in ent_names.strip().split(";")]
			ent_types 	= [ent_type2num(x) for x in ent_types.strip().split(" ")]
			rels 		= [[int(y) for y in x.strip().split(" ")] for x in rels.strip().split(";")]

			gold 		= gold.replace("-" , " - ").replace("/" , " / ").replace("\\" , " \\ ").strip().split(" ")
			gold 		= ["<SOS>"] + gold + ["<EOS>"]
			decoder_inp = [decoder_inp_process(x) for x in gold]
			gold 		= [gold_process(x) for x in gold]

			gold 		= gold[1:]
			decoder_inp = decoder_inp[:-1]

			dataset.append({
				"title" 		: title ,
				"ent_types" 	: ent_types ,
				"ent_names" 	: ent_names , 
				"rels" 			: rels ,
				"gold" 			: gold ,
				"decoder_inp" 	: decoder_inp ,
			})

	if shuffle:
		random.shuffle(dataset)

	for x in dataset:
		[ vocab.add(w) for w in x["title"]]
		[[vocab.add(w) for w in y] for y in x["ent_names"]]
		[ vocab.add(w) for w in x["decoder_inp"]]
		vocab.add("<EOS>")

	return dataset

def data_index(dataset):
	for x in dataset:
		x["title"] 			= [ vocab.to_index(w) for w in x["title"]]
		x["ent_names"] 		= [[vocab.to_index(w) for w in y] for y in x["ent_names"]]
		x["decoder_inp"] 	= [ vocab.to_index(w) for w in x["decoder_inp"]]
		x["gold"] 			= [ vocab.to_index(w) for w in x["gold"]]

	return dataset

def data_biuld_graph(dataset):
	for i in range(len(dataset)):
		dataset[i] = make_graph(dataset[i])
	return dataset

def run(name = "" , shuffle = True , force_reprocess = False):

	cache_loc = path.join(cache_path , name)
	
	if name and (not force_reprocess) and path.exists(cache_loc): 
		with open(cache_loc , "rb") as fil:
			ret = pickle.load(fil)
		return ret

	train_data = get_data(train_path , shuffle = shuffle)
	valid_data = get_data(valid_path , shuffle = False)
	test_data  = get_data( test_path , shuffle = False)

	'''
		if model choose to copy one entity , only need to tell the number of them
		this is for predict entity , make sure the are at the end of vocabulary
	'''
	for i in range(Con.max_entity_per_string):
		for j in range(100): #avoid low frequency
			vocab.add("<ent_%d>" % i)

	vocab.build_vocab()

	ent_idx_v = []
	oth_idx_v = []
	for i in range(Con.max_entity_per_string):
		ent_idx_v.append(vocab.word2idx["<ent_%d>" % i])
	for i in range(len(vocab)):
		if i not in ent_idx_v:
			oth_idx_v.append(i)
	ent_idx_v = tc.LongTensor(ent_idx_v)
	oth_idx_v = tc.LongTensor(oth_idx_v)
	sort_idx = tc.cat([oth_idx_v , ent_idx_v] , dim = 0)
	_ , sort_idx = tc.sort(sort_idx)

	_a = tc.zeros(len(vocab))
	for i in range(Con.max_entity_per_string):
		_a[len(vocab)-Con.max_entity_per_string+i] = i+1
	_a = _a[sort_idx]
	assert (_a[ent_idx_v] == tc.arange(1,Con.max_entity_per_string + 1)).all()

	for data in [train_data , valid_data , test_data]:
		data_index(data)
		data_biuld_graph(data)


	ret = {
		"train" : train_data,
		"valid" : valid_data,
		"test"  : test_data,
		"vocab" : vocab,
		"sort_idx" : sort_idx,
	}

	if name:
		os.makedirs(cache_path , exist_ok = True)
		with open(cache_loc , "wb") as fil:
			pickle.dump(ret , fil)
	
	return ret






