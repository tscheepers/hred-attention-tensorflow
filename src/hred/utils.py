import tensorflow as tf
import os
import numpy as np
import subprocess
import pickle
import os
import logging as logger

EOQ_SYMBOL = 1


def make_attention_mask(X, eoq_symbol=EOQ_SYMBOL):
    def make_mask(last_seen_eoq_pos, query, eoq_symbol):

        if last_seen_eoq_pos == -1:
            return np.zeros((1, len(query)))
        else:
            mask = np.ones([len(query)])
            mask[last_seen_eoq_pos:] = 0.
            mask = np.where(query == eoq_symbol, 0, mask)
            return mask.reshape(1, (len(mask)))

    # eoq_mask = np.where(X == float(EOQ_SYMBOL), float(EOQ_SYMBOL), 0.)
    first_query = True

    for i in range(X.shape[1]):  # loop over batch size --> this gives 80 queries
        query = X[:, i]  # eoq_mask[:, i] #X[:, i]
        # print("query", query)

        first = True
        last_seen_eoq_pos = -1
        for w_pos in range(len(query)):

            if query[w_pos] == float(eoq_symbol):
                last_seen_eoq_pos = w_pos

            if first:
                query_masks = make_mask(last_seen_eoq_pos, query, float(eoq_symbol))
                first = False
            else:
                new_mask = make_mask(last_seen_eoq_pos, query, float(eoq_symbol))
                query_masks = np.concatenate((query_masks, new_mask), axis=0)

        if first_query:
            batch_masks = np.expand_dims(query_masks, axis=2)
            first_query = False
        else:
            batch_masks = np.dstack((batch_masks, query_masks))

    batch_masks = np.transpose(batch_masks, (0, 2, 1))
    # print("shape batch masks", batch_masks.shape)
    # print("batch masks:", batch_masks)

    return batch_masks  # = attention masks