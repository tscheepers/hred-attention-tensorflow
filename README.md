# A Hierachical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion

An Information Retrieval 2 project.

- **Paper:** A Hierachical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion [pdf](https://arxiv.org/abs/1507.02221)
- **Data:** AOL Dataset [Thijs' Dropbox](https://www.dropbox.com/s/thuv05pl3wyz6lq/aol-data.tar?dl=0)
- **Pre-processed data:** [Jörg's Dropbox](https://www.dropbox.com/sh/zm430xgouaibo5q/AABO9OuWDlkqMI5nYM9vgS80a?dl=0)

## TO-DO's 

- [ ] Preprocessing of the data
- [ ] Implement a decoder (prototype)
- [ ] Initial implementation of a seq2seq neural network
- [ ] Seperate query-level recurrent state and the session-level recurrent state, i.e. implemente session passing
- [ ] Implement or incorperate "learning to rank" as reranking mechanism
- [ ] Implement HRED score and include it in the ranking mechanism
- [ ] Create interface for query suggestions (both auto-complete as well as, next query suggestion), nice for a demo
- [ ] Modify the model to include a novel technique

## Links:

- [Very nice tutorial on IR and LM](http://benjaminbolte.com/blog/2016/keras-language-modeling.html#word-embeddings)
- [Keras Seq2Seq](https://github.com/farizrahman4u/seq2seq)
- [Keras Recurrentshop](https://github.com/datalogai/recurrentshop) (Containers and such)
- [Keras Language Modeling](https://github.com/codekansas/keras-language-modeling)
- [Original Theano implementation](https://github.com/sordonia/hred-qs)
- [Fork of original Theano implementation](https://github.com/sweaterr/hred-qs) (with some sample data)
- [Recurrent Neural Networks: Character RNNs with Keras](http://ml4a.github.io/guides/recurrent_neural_networks/)
