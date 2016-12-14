# Hierarchical Recurrent Encoder-Decoder code (HRED) for Query Suggestion.

This code accompanies the paper:

"A Hierarchical Recurrent Encoder-Decoder For Generative Context-Aware Query Suggestion", by Alessandro Sordoni, Yoshua Bengio, Hossein Vahabi, Christina Lioma, Jakob G. Simonsen, Jian-Yun Nie, to appear in CIKM'15.

The pre-print of the paper is available at: http://arxiv.org/abs/1507.02221.

### Getting Started

First prepare the data just like in the `.rnk` and `.ses` files in the `data` folder. Than use the following command to create `.pkl` files.

```bash
python convert-text2dict.py data/dev/train data/dev/train --cutoff=50000 --min_freq=5
python convert-text2dict.py data/dev/valid data/dev/valid --cutoff=50000 --min_freq=5 --dict=data/dev/train.dict.pkl
```

Run the training script to train the model. You have to pass a training configuration that can be found in `state.py`. To train the model using the GPU pass the correct `THEANO_FLAGS`. The configuration also contains information on where the model will be saved.

```bash
export THEANO_FLAGS=mode\=FAST_RUN,device\=gpu,floatX\=float32,allow_gc\=True,scan.allow_gc\=False,nvcc.flags\=-use_fast_math
python train.py --prototype dev
```

After a trained model you can start sample data from the model using a set of context queries. These queries you can put in as a file, tab seperated, just like the `.ses` file.

```bash
python sample.py models/1479202412.28_test data/queries --verbose
```
