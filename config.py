import sys , os
import argparse
import pdb

parser = argparse.ArgumentParser()

#---------------------------------------------------------------------------------------------------

#main procedure
parser.add_argument("--save"			, default = "./save/test", type 	= str)
parser.add_argument("--generate_from"	, default = "./last.pkl", type 		= str)
parser.add_argument("--generated_file"	, default = "./gene.txt", type 		= str)
parser.add_argument("--log_file_name"	, default = "log.txt" 	, type 		= str)
parser.add_argument("--seed" 			, default = 2333 		, type 		= int)
parser.add_argument("--name" 			, default = "AGENDA"	, type 		= str)
parser.add_argument("--info" 			, default = ""			, type 		= str)

#device and data
parser.add_argument("--gpus" 			, default = "0" 		, type 		= str)
parser.add_argument("--train_data" 		, default = "train" 	, type 		= str)
parser.add_argument("--dev_data" 		, default = "valid" 	, type 		= str)
parser.add_argument("--test_data" 		, default = "test" 		, type 		= str)
parser.add_argument("--force_reprocess" , default = False 		, action 	= "store_true")

#training procedure control
parser.add_argument("--epoch_number"	, default = 20 			, type 		= int)
parser.add_argument("--batch_size"		, default = 8 			, type 		= int)

#gradient desent control
parser.add_argument("--clip" 			, default = 1. 			, type 		= float)

parser.add_argument("--lr" 				, default = 1e-4 		, type 		= float)

parser.add_argument("--update_freq"		, default = 1 			, type 		= int)

#generate parameters
parser.add_argument("--beam_norm" 		, default = False 		, action 	= "store_true")

#model structrue
parser.add_argument("--dropout" 		, default = 0.3 		, type 		= float)
parser.add_argument("--attn_method" 	, default = "naive" 	, type 		= str , choices = ["naive" , "edge_softmax"])

#---------------------------------------------------------------------------------------------------

C = parser.parse_args()

C.gpus = [int(x) for x in C.gpus.strip().split(",")]
if not C.save:
	C.log_file_name = ""
else: C.log_file_name = os.path.join(C.save , C.log_file_name)