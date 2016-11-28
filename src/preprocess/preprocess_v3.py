import re
# import gensim
import os
import sys
import tarfile
import glob
import gzip
import argparse
from nltk import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime, timedelta
import numpy as np
from collections import Counter, defaultdict
import logging
import cPickle
import tqdm


# TODO: need to skip a query if all query words are unknown

AOL_ROOT_PATH = 'AOL-user-ct-collection'
AOL_TAR_FILE = 'aol-data.tar'
DEFAULT_OUT_PATH = "/home/jogi/git/repository/ir2_jorg/data/AOL-user-ct-collection/"
VOCAB_FILENAME = "aol_vocab.dict.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pre-process AOL data')


def safe_pickle(obj, filename):
    if os.path.isfile(filename):
        logger.info("INFO - Overwriting %s." % filename)
    else:
        logger.info("INFO - Saving to %s." % filename)
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def save_vocab(vocab, word_freq, outfile):

    safe_pickle([(word, word_id, word_freq[word_id]) \
                 for word, word_id in vocab.items()],
                outfile)


def make_a_new_vocab(vocab_file, threshold):

    assert os.path.isfile(vocab_file)
    vocab_list = [(x[0], x[1], x[2]) for x in cPickle.load(open(vocab_file, "r"))]
    print(vocab_list[:2])
    # sort ascending, last threshold items and finally reverse order
    vocab_list = sorted(vocab_list, key=lambda y: y[2])[-threshold:][::-1]
    # Add special tokens to the vocabulary
    vocab = {'<unk>': 0, '</q>': 1, '</s>': 2, '</p>': 3}
    word_freq = {0: 0, 1: 0, 2: 0, 3: 0}
    for word, idx, freq in vocab_list:
        idx = len(vocab)
        vocab[word] = idx
        word_freq[idx] = freq
    logger.info("Vocab size %d" % len(vocab))
    save_vocab(vocab, word_freq, "aol_vocab_" + str(threshold) + ".pkl")


def load_vocab_without_freq(vocab_file):
    vocab = cPickle.load(open(vocab_file, 'r'))
    assert '<unk>' in vocab
    assert '</s>' in vocab
    assert '</q>' in vocab
    assert '</p>' in vocab
    logger.info("INFO - Successfully loaded vocabulary dictionary %s." % vocab_file)
    logger.info("INFO - Vocabulary contains %d words" % len(vocab))
    return vocab


def load_vocab(vocab_file):

    assert os.path.isfile(vocab_file)
    vocab = dict([(x[0], x[1]) for x in cPickle.load(open(vocab_file, "r"))])
    # Check consistency
    assert '<unk>' in vocab
    assert '</s>' in vocab
    assert '</q>' in vocab
    assert '</p>' in vocab
    logger.info("INFO - Successfully loaded vocabulary dictionary %s." % vocab_file)
    logger.info("INFO - Vocabulary contains %d words" % len(vocab))
    return vocab


def generate_dict_and_translate(session_file, num_of_session=10000, vocab_size=2500, final_out_dir=DEFAULT_OUT_PATH):
    # Add special tokens to the vocabulary
    vocab = {'<unk>': 0, '</q>': 1, '</s>': 2, '</p>': 3}

    def word_to_seq(query):
        query_lst = []
        for w in query.strip().split():
            query_lst.append(str(vocab.get(w, 0)))
        return " ".join(query_lst)

    word_counter = Counter()
    sess_counter = 0
    for line in open(session_file, 'r'):
        s = [x for x in line.strip().split()]
        word_counter.update(s)
        sess_counter += 1
        if sess_counter >= num_of_session:
            break

    total_freq = sum(word_counter.values())
    logger.info("Total word frequency in dictionary %d " % total_freq)

    if args.cutoff != -1:
        logger.info("Cutoff %d" % vocab_size)
        vocab_count = word_counter.most_common(vocab_size)
    else:
        vocab_count = word_counter.most_common()

    for (word, count) in vocab_count:
        if count < args.min_freq:
            break
        vocab[word] = len(vocab)

    safe_pickle(vocab, os.path.join(final_out_dir, "aol_vocab_" + str(vocab_size) + ".pkl"))
    sess_counter = 0

    outfile = os.path.join(final_out_dir, "aol_sess_windices" + ".sess")
    with open(outfile, 'w') as f_out:
        for line in open(session_file, 'r'):
            queries = line.split('\t')
            session_list = []
            for query in queries:
                session_list.append(word_to_seq(query))

            f_out.write("\t".join(session_list) + "\n")
            sess_counter += 1
            if sess_counter >= num_of_session:
                break


def get_files(root_dir):
    """
        queries the "data" directory  (which should be in the root of your repository) for files to process
        Also assumes that file 'aol-data.tar' is in your "data" directory. So have you to leave it there for now.
        For sure, we can change this...who wants 500 MB extra he?
        (1) looks for "user*.txt.gz" files, meaning the raw ones

    :param root_dir:
    :return:
    """
    if os.getcwd().find('data') == -1:
        os.chdir('../data')

    if os.path.isdir(root_dir):
        # so not the first time we extracted files, AOL directory already exists
        os.chdir(root_dir)
    gz_to_process = glob.glob("user*.txt.gz")
    # we only want to re-process the gz-files we haven't done yet.
    # so if user*.txt files exists, means we already processed the gz one...
    # therefore list "files_extracted" will contain the
    print gz_to_process
    if len(gz_to_process) < 10:
        # set again to None, because we only want to process the ones we haven't extracted yet
        # we use the user*.txt files to make/extend the dictionary.
        # so you can re-process all gz files by just removing all user*.txt files from the AOL_ROOT_PATH
        # directory
        # go back to ../data directory
        os.chdir("..")

        print("INFO -- Need to extract gz files...just doing all of them")
        if not os.path.isfile(AOL_TAR_FILE):
            raise Exception('Expected %s to exist in data directory.' % AOL_TAR_FILE)
        tar = tarfile.open(AOL_TAR_FILE)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
        # go back to the directory with the extracted gz files
        os.chdir(root_dir)
        gz_to_process = glob.glob("user*.txt.gz")
        # gz_to_process = []
        # for gz_f in gz_files:
        #    if not os.path.splitext(gz_f)[0] in files_extracted:
        #        gz_to_process.append(gz_f)

    return gz_to_process


class Processor:
    def __init__(self, path, num_of_recs=None, make_dict=True, file_to_process=None, move_files=False,
                 vocab_threshold=90000, vocab_file=None):

        self.file_to_process = file_to_process
        self.file_input_path = path
        self.move_files = move_files
        self.query_vocab = []
        self.min_session_length = 2
        self.vocab = {}
        self.vocab_helper = Counter()
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.word_freq = defaultdict(lambda: 1)

        if make_dict:
            print("INFO -- Generating vocabulary during processing")

            # The queries in this dataset were sampled between 1 March, 2006 and 31 May, 2006
        # (1) BACKGROUND set
        # those submitted before 1 May, 2006 as our background data to estimate the proposed model
        # and the baselines.
        self.background_dt = datetime.strptime('2006-05-01 00:00:00', "%Y-%m-%d %H:%M:%S")
        # (2)  training data used to train rankers
        #      take 2 weeks
        self.training_dt = datetime.strptime('2006-05-15 00:00:00', "%Y-%m-%d %H:%M:%S")
        # last 17 days of May are used for a) validation and b) test set
        # need to make a split here, paper does not state how they split
        #       (a) we take 10 days for validation
        #       (b) the rest (7 days) for test
        #
        # SO WE HAVE 4 OUTPUT FILES
        self.validation_dt = datetime.strptime('2006-05-25 00:00:00', "%Y-%m-%d %H:%M:%S")
        self.make_dict = make_dict

        # get all the files we need to process
        # files list: contains user*.txt files that need to be processed
        # gz_files list: contains user*.txt.gz files that need to be processed
        self.gz_files = []
        self.num_of_recs = num_of_recs
        self.stop = set(stopwords.words('english'))

    def execute(self):

        if self.file_to_process is not None:
            print("INFO -- ONE-FILE-PROCESSING option")
            if os.getcwd().find('data') == -1:
                os.chdir('../data/' + self.file_input_path)
            self.gz_files.append(self.file_to_process)
        else:
            self.gz_files = get_files(AOL_ROOT_PATH)

        # Pre processing step 1
        # ========================
        #       - extract gz-files
        #       - take the first four columns of each row: AnonID, Query, QueryTime, ItemRank
        #       - remove non-alphanumeric characters and convert to lowercase
        #       - remove words under a minimal length, currently using 2 letters as minimum
        #       - separate into 4 files, background, training, validation and test
        #       - generate a session and a rank file
        #       - move all AOL gz files into one of the the four file types
        #       - move gz file to "processed" directory (because we query the data directory for gz-files)

        if not os.getcwd().find(self.file_input_path) == -1 and len(self.gz_files) != 0:
            print "INFO - need to process %d gz files" % len(self.gz_files)
            self.process_gz_files()
            if self.make_dict:
                self.make_vocab()
        else:
            raise Exception('Expected %s to exist in data directory.' % self.file_input_path)

    def load_bg_session(self, filename):

        with open(filename, 'r') as f:
            c = 1
            for line in f:
                line = (line.strip('\n')).replace('\t', ' ')
                self.vocab_helper.update(line.split())
                if c % 100000 == 0:
                    print("Progress %d" % c)
                c += 1
                self.query_vocab = []
        self.make_vocab()

    @staticmethod
    def tokenize_w_nltk(query):
        return word_tokenize(query)

    @staticmethod
    def remove_short_words(query_words, min_length):
        return [w for w in query_words if len(w) >= min_length]

    @staticmethod
    def remove_non_alphanumric(string):
        return re.sub(r'\W', ' ', string.lower())

    # currently not in use
    def remove_stop_words(self, string):
        words = string.split()
        return ' '.join([w for w in words if w not in self.stop])

    def update_word_count(self):
            s = [x for x in self.query_vocab]
            self.vocab_helper.update(s)

    def make_vocab(self):
        logger.info("INFO - Creating vocabulary")
        if self.vocab_threshold != -1:
            logger.info("Cutoff %d" % self.vocab_threshold)
            vocab_count = self.vocab_helper.most_common(self.vocab_threshold)
        else:
            vocab_count = self.vocab_helper.most_common()
        # Add special tokens to the vocabulary
        self.vocab = {'<unk>': 0, '</q>': 1, '</s>': 2, '</p>': 3}

        for (word, count) in vocab_count:
            if count < args.min_freq:
                break
            self.vocab[word] = len(self.vocab)
            self.word_freq[word] = count

        self.chgcwd_to_data_dir()
        logger.info("Vocab size %d" % len(self.vocab))
        save_vocab(self.vocab, self.word_freq, self.vocab_file)

    def chgcwd_to_data_dir(self):
        """
            Change the current working directory to data/AOL... directory
        """
        if os.getcwd().find('data') == -1:
            os.chdir('../data')

        if os.path.isdir(self.file_input_path):
            os.chdir(self.file_input_path)

    def check_session(self, session):
        # are the last two queries identical, then return false
        return session[-1] == session[-2]

    def process_gz_files(self):

        timeout = timedelta(minutes=30)
        file_ext_sess = ".ctx"
        file_ext_rnk  = ".rnk"
        # open the 8 output files
        # --------------------------
        # (1) We have 4 different files based on the query dates as described earlier
        #     1. background data
        #     2. training data
        #     3. validation data
        #     4. test data
        #
        # (2) for each "data type" we will generate
        #     1. a session file (.sess) where each record in the file represents a user session
        #        queries are separated by \t and words by spaces
        #     2. a rank file, indicating for each query in the session which doc in the SERP was clicked
        #        output file will have .rnk extension
        sess_outfile_bg = "bg_session" + file_ext_sess
        rnk_output_bg   = "bg_rnk" + file_ext_rnk
        sess_outfile_tr = "tr_session" + file_ext_sess
        rnk_output_tr = "tr_rnk" + file_ext_rnk
        sess_outfile_val = "val_session" + file_ext_sess
        rnk_output_val = "val_rnk" + file_ext_rnk
        sess_outfile_test = "test_session" + file_ext_sess
        rnk_output_test = "test_rnk" + file_ext_rnk

        with open(sess_outfile_bg, 'w') as bg_sess, open(sess_outfile_tr, 'w') as tr_sess, \
                open(sess_outfile_val, 'w') as val_sess, open(sess_outfile_test, 'w') as test_sess, \
                open(rnk_output_bg, 'w') as bg_rnk, open(rnk_output_tr, 'w') as tr_rnk, \
                open(rnk_output_val, 'w') as val_rnk, open(rnk_output_test, 'w') as test_rnk:
            queries_unusable = 0
            sessions_skipped = 0
            session_id = 0
            total_queries = 0
            last_2_identical = 0
            sess_unusable = False
            for filename in tqdm.tqdm(self.gz_files):
                lines_read = 1
                print "INFO - currently processing gz file: %s" % filename
                with gzip.open(filename, 'r') as src:
                    prev_ID = -1
                    session_list = []
                    rank_list = []
                    for line in src:
                        AnonID, Query, QueryTime, ItemRank = line.split('\t')[:4]

                        if AnonID == 'AnonID':
                            # skip header
                            continue
                        else:
                            if ItemRank == "":
                                ItemRank = "0"

                        query_dttm = datetime.strptime(QueryTime, "%Y-%m-%d %H:%M:%S")
                        query_words = []
                        if session_id < self.num_of_recs or self.num_of_recs is None:
                            tidy = self.remove_non_alphanumric(Query)
                            if tidy != '':
                                query_words = self.tokenize_w_nltk(tidy)
                                query_words = self.remove_short_words(query_words, min_length=2)

                            else:
                                # after pre processing nothing left of query, register that query is useless
                                # and session can't be used
                                queries_unusable += 1
                                sess_unusable = True

                        else:
                            # test purposes, just don't process the hole shit load, break after num of records
                            break
                        # determine whether or not to make a new session, see below
                        if prev_ID != -1 and (prev_ID != AnonID or \
                                                      (prev_ID == AnonID and (last_query_activity_dttm + timeout) < query_dttm)):
                            # user ID changed OR still same user but new activity more than 30 minutes ago
                            # make new session ID
                            if len(session_list) >= self.min_session_length:
                                # decide whether to write background file or "rest" file
                                assert len(session_list) == len(rank_list)
                                sess_unusable = self.check_session(session_list)
                                total_queries += len(session_list)
                                if session_list[-1] == session_list[-2]:
                                    last_2_identical += 1
                                if not sess_unusable:
                                    if self.background_dt > last_query_activity_dttm:
                                        # first add query words to vocab
                                        if self.make_dict:
                                            self.query_vocab.extend(query_words)
                                        # print(session_list)
                                        bg_sess.write("\t".join(session_list) + "\n")
                                        bg_rnk.write("\t".join(rank_list) + "\n")
                                    elif self.training_dt > last_query_activity_dttm:
                                        tr_sess.write("\t".join(session_list) + "\n")
                                        tr_rnk.write("\t".join(rank_list) + "\n")
                                    elif self.validation_dt > last_query_activity_dttm:
                                        val_sess.write("\t".join(session_list) + "\n")
                                        val_rnk.write("\t".join(rank_list) + "\n")
                                    else:
                                        test_sess.write("\t".join(session_list) + "\n")
                                        test_rnk.write("\t".join(rank_list) + "\n")
                                else:
                                    # session is unusable probably because one of the queries was empty after
                                    # clean-up or because last two queries are identical
                                    sessions_skipped += 1

                            else:
                                # unusable session because it is shorter than minimum length
                                sessions_skipped += 1
                            session_list = []
                            rank_list = []
                            session_id += 1
                            # reset our session unusable indicator
                            sess_unusable = False
                        # append query to user session (eventually new) if it contains more than one word

                        if len(query_words) and not ("www" in query_words or "com" in query_words):
                            session_list.append(" ".join(query_words))
                            rank_list.append(ItemRank)
                        else:
                            sess_unusable = True
                            queries_unusable += 1

                        # save current user ID in order to determine the session ID
                        if lines_read % 100000 == 0 and lines_read != 0:
                            print("INFO -- Progress %d" % lines_read)
                        lines_read += 1
                        prev_ID = AnonID
                        last_query_activity_dttm = query_dttm

                # end of gz-file, update vocabulary
                if self.make_dict:
                    self.update_word_count()
                    self.query_vocab = []

            # all files are processed
            print("INFO - total sessions %d, unusable after preprocessing %d" % (session_id, sessions_skipped))
            print("INFO - ==>> final # of sessions %d" % (session_id - sessions_skipped))
            print("INFO - total queries used %d, unusable after preprocessing %d" % (total_queries, queries_unusable))
            print("INFO - sessions where last 2 queries are identical %d" % last_2_identical)
            print("INFO - total sessions skipped %d" % sessions_skipped)
            # write last session to file
            # I know this is ugly, but not in the mood to make a method out of this pile of files
            if len(session_list) >= self.min_session_length:
                # decide whether to write background file or "rest" file
                assert len(session_list) == len(rank_list)
                if not sess_unusable:
                    total_queries += len(session_list)
                    if self.background_dt > last_query_activity_dttm:
                        # first add query words to vocab
                        if self.make_dict:
                            self.query_vocab.extend(query_words)
                        bg_sess.write("\t".join(session_list) + "\n")
                        bg_rnk.write("\t".join(rank_list) + "\n")
                    elif self.training_dt > last_query_activity_dttm:
                        tr_sess.write("\t".join(session_list) + "\n")
                        tr_rnk.write("\t".join(rank_list) + "\n")
                    elif self.validation_dt > last_query_activity_dttm:
                        val_sess.write("\t".join(session_list) + "\n")
                        val_rnk.write("\t".join(rank_list) + "\n")
                    else:
                        test_sess.write("\t".join(session_list) + "\n")
                        test_rnk.write("\t".join(rank_list) + "\n")
                    # last time update vocabulary counter
                if self.make_dict:
                    self.update_word_count()
                    self.query_vocab = []

            total_freq = sum(self.vocab_helper.values())
            logger.info("Total word frequency in dictionary %d " % total_freq)
        # move file to processed directory
        if self.move_files:
            if not os.path.isdir("processed"):
                os.mkdir("processed")
            for filename in self.gz_files:
                os.rename(filename, "processed/" + filename)
                print("INFO -- moved file %s" % filename)

    def word_to_seq(self, query):
        query_lst = []
        for word in query.strip().split():
            query_lst.append(str(self.vocab.get(word, 0)))
        return " ".join(query_lst)

    def translate_words_to_indices(self, final_proc_options=['bg'], final_out_dir="final_out"):
        # Pre processing step 3
        # ==========================
        # - process the *.dat files, replace all query words with vocab indices
        #   words that are not in the vocab will be replaced with default "0" = unknown
        # - because we have 4 types of files we process with options
        #       bg = background files bg*.dat
        #       tr = training
        #       val = validation
        #       test = test
        self.chgcwd_to_data_dir()
        self.vocab = load_vocab(self.vocab_file)
        print(len(self.vocab))
        t = 0
        for key, value in self.vocab.iteritems():
            print key, " / ", value
            t += 1
            if t > 6:
                break
        if final_proc_options is not None:
            if not os.path.isdir(final_out_dir):
                os.mkdir(final_out_dir)
            file_ext = ".out"
            outfile = os.path.join(final_out_dir, "aol_sess_windices" + file_ext)

            with open(outfile, 'w') as f_out:
                logger.info("Sending output to %s" % outfile)
                session_c = 0
                for opt in final_proc_options:
                    files_to_process = glob.glob(opt + "*.ctx")
                    for filename in tqdm.tqdm(files_to_process):
                        query_c = 0
                        logger.info("processing input filename %s" % filename)
                        with open(filename, 'r') as f:
                                for line in f:
                                    query_c += 1
                                    queries = line.split('\t')
                                    session_list = []
                                    for query in queries:
                                        session_list.append(self.word_to_seq(query))

                                    f_out.write("\t".join(session_list) + "\n")
                                    session_c += 1

                # notice the statistics at the end of each file
                logger.info("INFO - Converted %d sessions" % session_c)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=AOL_ROOT_PATH,
                        help="Prefix (*.ses, *.rnk) to separated session/rank file")
    parser.add_argument("--cutoff", type=int, default=90000,
                        help="Vocabulary pruning option i.e. 90k words, default is None")
    parser.add_argument("--min_freq", type=int, default=1,
                        help="Min frequency cutoff (optional)")
    parser.add_argument("--vocab_file", type=str, default=VOCAB_FILENAME,
                        help="External dictionary (pkl file)")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--make_vocab", action='store_true', help="Boolean whether or not to generate vocabulary")

    args = parser.parse_args()

    vocab_file = os.path.join(DEFAULT_OUT_PATH, 'aol_vocab_50000.pkl')
    p = Processor(args.input_dir, num_of_recs=None, vocab_file=vocab_file,
                 file_to_process=None, vocab_threshold=50000, make_dict=False)

    # generates the 4 session files (and corresponding rank file) for background, train, test, validation
    # p.execute()
    p.translate_words_to_indices(final_proc_options=['bg', 'tr', 'val', 'test'])
    # p.load_bg_session(DEFAULT_OUT_PATH + "bg_session.ctx")
    # vocab = load_vocab(os.path.join(DEFAULT_OUT_PATH, args.vocab_file))
    # vocab = load_vocab(vocab_file)
    # print("Size vocab %d" % len(vocab))
    # c = 0
    # print("Length vocab %d" % len(vocab))
    # for keys, values in vocab.iteritems():
    #     c += 1
    #     print keys
    #     print values
    #     if c > 10:
    #         break
    # make_a_new_vocab(os.path.join(DEFAULT_OUT_PATH, args.vocab_file), 50000)
    # p.translate_words_to_indices(final_proc_options=['bg', 'tr', 'val', 'test'])

    # generate_dict_and_translate(os.path.join(DEFAULT_OUT_PATH, 'bg_session.ctx'))




