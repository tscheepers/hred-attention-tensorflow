# A Hierachical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion

An Information Retrieval 2 project.

- **Paper:** A Hierachical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion [pdf](https://arxiv.org/abs/1507.02221)
- **Data:** 
  - AOL Dataset: [Thijs' Dropbox](https://www.dropbox.com/s/thuv05pl3wyz6lq/aol-data.tar?dl=0)
  - Model input for in the `data` folder: [Thijs' Dropbox](https://www.dropbox.com/sh/d9ukeq9uptamik8/AACTfqrnP2erci0N-A3cxu0Fa?dl=0)
  - Pre-processed data: [Jörg's Dropbox](https://www.dropbox.com/sh/zm430xgouaibo5q/AABO9OuWDlkqMI5nYM9vgS80a?dl=0)

## TO-DO's 

- [x] Preprocessing of the data
- [x] Implement a decoder (prototype)
- [x] Initial implementation of a seq2seq neural network
- [x] Seperate query-level recurrent state and the session-level recurrent state, i.e. implemente session passing
- [x] Implement or incorperate "learning to rank" as reranking mechanism
- [x] Implement HRED score and include it in the ranking mechanism
- [ ] Create interface for query suggestions (both auto-complete as well as, next query suggestion), nice for a demo
- [ ] Modify the model to include a novel technique

## Links:

- [Very nice tutorial on IR and LM](http://benjaminbolte.com/blog/2016/keras-language-modeling.html#word-embeddings)
- [Original Theano implementation](https://github.com/sordonia/hred-qs)
