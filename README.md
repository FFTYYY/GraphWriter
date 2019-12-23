A reproduction to GraphWriter (《Text Generation from Knowledge Graphs with Graph Transformers》).

### Train & Evaluate

1. train:`python train.py --save=[save dir] --gpus=[gpus]`
2. generate:`python generate.py --save=[save dir] --generate_from=[pkl file name] --generated_file=[target file]`
3. get ground truth:`python gene_ref.py`
4. eval:`python eval.py [generated file] [ground truth]`

##### Example:
```
python train.py --save=./save/test
python generate.py --save=./save/test --generate_from=epoch_10.pkl
python gene_ref.py
python eval.py ./save/test/gene.txt test_ref.txt
```
### Result

Implementation                | Bleu | METROR 
--------------                |------|--------
Original                      | 14.3 | 18.8   
My(w/o transformer)           | 12.5 | 17.5
My(w transformer)             | 12.2 | -
My(w transformer,+ glove)     | 12.3 | -

好吧我还没有完全复现出他的结果(ノДＴ)