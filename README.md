# A Hierachical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion

An Information Retrieval 2 project.

- **Paper:** A Hierachical Recurrent Encoder-Decoder for Generative Context-Aware Query Suggestion [pdf](https://arxiv.org/abs/1507.02221)
- **Data:** AOL Dataset [Thijs's Dropbox](https://www.dropbox.com/s/thuv05pl3wyz6lq/aol-data.tar?dl=0)

## TO-DO's 

- [ ] Preprocessing of the data (https://github.com/MBleeker & https://github.com/toologicbv)
- [ ] Initial implementation of a seq2seq neural network (https://github.com/maartjeth & https://github.com/tscheepers)
- [ ] Seperate query-level recurrent state and the session-level recurrent state, i.e. implemente session passing
- [ ] Implement or incorperate "learning to rank" as reranking mechanism
- [ ] Implement HRED score and include it in the ranking mechanism
- [ ] Create interface for query suggestions (both auto-complete as well as, next query suggestion), nice for a demo
- [ ] Modify the model to include a novel technique
