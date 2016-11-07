import re
import gensim
import os
import sys
import tarfile
import glob
import gzip
import pickle
from timeit import default_timer as timer
from nltk import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime, timedelta
from keras.preprocessing.text import Tokenizer
import numpy as np

AOL_ROOT_PATH = 'AOL-user-ct-collection'
AOL_TAR_FILE = 'aol-data.tar'


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
    def __init__(self, path, num_of_recs=None, dict_file=None,
                 make_dict=False, file_to_process=None, move_files=False):

        self.file_input_path = path
        self.move_files = move_files
        self.query_vocab = []
        self.vocab_tokenizer = Tokenizer()
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

        self.dictionary = None
        self.make_dict = make_dict
        self.dict_file = dict_file
        # IMPORTANT: we only keep 90K most frequent words of background vocabulary
        # TODO: add "unkown" to dictionary
        self.max_prune_words = 90000
        # get all the files we need to process
        # files list: contains user*.txt files that need to be processed
        # gz_files list: contains user*.txt.gz files that need to be processed
        #
        self.files = []
        self.gz_files = []
        print(file_to_process)
        if file_to_process is not None:
            print("INFO -- ONE-FILE-PROCESSING option")
            if os.getcwd().find('data') == -1:
                os.chdir('../data/' + path)
            self.gz_files.append(file_to_process)
        else:
            self.gz_files = get_files(AOL_ROOT_PATH)

        self.num_of_recs = num_of_recs

        self.stop = set(stopwords.words('english'))
        # Pre processing step 1
        # ========================
        #       - extract gz-files
        #       - take the first three columns of each row: AnonID, Query, QueryTime
        #       - remove non-alphanumeric characters and convert to lowercase
        #       - save to "user*.txt" file
        #       - move gz file to "processed" directory (because we query the data directory for gz-files)

        if not os.getcwd().find(path) == -1 and len(self.gz_files) != 0:
            print "INFO - need to process %d gz files" % len(self.gz_files)
            self.process_gz_files()
            self.save_vocab()
        else:
            raise Exception('Expected %s to exist in data directory.' % path)

        # Pre processing step 2
        # ========================
        # make or extend the existing word-dictionary

        if self.make_dict:
            self.load_dict()
            if len(self.query_vocab) != 0:
                print "INFO - Make/extend dictionary"
                self.make_dictionary()
                self.save_dict()
            else:
                print("No more files to process for dictionary.")

        # Pre processing step 3
        # ==========================
        # - process the *.dat files, replace all query words with vocab indices
        #   words that are not in the vocab will be replaced with default "unknown"
        # - because we have 4 types of files we process with options
        #       bg = background files bg*.dat
        #       tr = training
        #       val = validation
        #       test = test

    def load_dict(self):
        if os.path.isfile(self.dict_file):
            self.dictionary = gensim.corpora.dictionary.Dictionary.load(self.dict_file)
        else:
            # make new dictionary, add word "unknown"
            self.dictionary = gensim.corpora.dictionary.Dictionary()

        print("INFO - Loaded vocab dictionary: %d" % len(self.dictionary))

    def save_dict(self):
        self.dictionary.save(self.dict_file)
        print "INFO -- Saved word dictionary %s. Contains %d words" % (self.dict_file, len(self.dictionary))

    def make_dictionary(self):
        """
            Loop through the user*.txt files
            extract query words (that were filtered earlier)
            and add them to the word dictionary (gensim)
        """
        if len(self.query_vocab) != 0:
            num_now = len(self.dictionary)
            self.dictionary.add_documents(self.query_vocab, self.max_prune_words)
            num_new = len(self.dictionary)
            print("INFO - Added %d words to dictionary" % (num_new - num_now))

    def process_gz_files(self):

        for file in self.gz_files:
            print "INFO - currently processing gz file: %s" % file
            self.process_one_gz_file(file)
            break

    def process_one_gz_file(self, filename, num_records=None):

        timeout = timedelta(minutes=30)
        file_ext = ".dat"

        with gzip.open(filename, 'r') as src:
            outfile_bg = "bg_" + os.path.splitext(os.path.splitext(filename)[0])[0] + file_ext
            outfile_tr = "tr_" + os.path.splitext(os.path.splitext(filename)[0])[0] + file_ext
            outfile_val = "val_" + os.path.splitext(os.path.splitext(filename)[0])[0] + file_ext
            outfile_test = "test_" + os.path.splitext(os.path.splitext(filename)[0])[0] + file_ext
            with open(outfile_bg, 'w') as bg, open(outfile_tr, 'w') as tr, open(outfile_val, 'w') as val, \
                    open(outfile_test, 'w') as test:
                num_total = 0
                num_empty = 0
                session_id = 0
                prev_ID = 0
                for line in src:
                    AnonID, Query, QueryTime = line.split('\t')[:3]

                    if AnonID == 'AnonID':
                        # skip header
                        continue
                    query_dttm = datetime.strptime(QueryTime, "%Y-%m-%d %H:%M:%S")

                    # determine whether or not to make a new session, see below
                    if prev_ID != AnonID or \
                            (prev_ID == AnonID and (last_query_activity_dttm + timeout) < query_dttm):
                        # user ID changed OR still same user but new activity more than 30 minutes ago
                        # make new session ID
                        session_id += 1

                    if num_total < self.num_of_recs or self.num_of_recs is None:
                        tidy = self.remove_non_alphanumric(Query)
                        if tidy != '':
                            query_words = self.tokenize_w_nltk(tidy)
                            if len(query_words) != 0:
                                # decide whether to write background file or "rest" file
                                if self.background_dt > query_dttm:
                                    # first add query words to vocab
                                    self.query_vocab.append(",".join(query_words))
                                    bg.write('{}\t{}\t{}\t{}\n'.format(session_id, AnonID, ",".join(query_words),
                                                                       QueryTime))
                                elif self.training_dt > query_dttm:
                                    tr.write('{}\t{}\t{}\t{}\n'.format(session_id, AnonID, " ".join(query_words),
                                                                       QueryTime))
                                elif self.validation_dt > query_dttm:
                                    val.write('{}\t{}\t{}\t{}\n'.format(session_id, AnonID, " ".join(query_words),
                                                                       QueryTime))
                                else:
                                    test.write('{}\t{}\t{}\t{}\n'.format(session_id, AnonID, " ".join(query_words),
                                                                       QueryTime))
                            else:
                                # after preprocessing nothing left of query, register that
                                num_empty += 1
                        else:
                            # after preprocessing nothing left of query, register that
                            num_empty += 1

                    else:
                        # test purposes, just don't process the hole shit load, break after num of records
                        break
                    # save current user ID in order to determine the session ID
                    num_total += 1
                    if num_total % 10000 == 0:
                        print("INFO -- Progress %d" % num_total)
                    prev_ID = AnonID
                    last_query_activity_dttm = query_dttm

            # add this file to the list that we'll use to make/extend the dictionary
            # remember only using background data to construct vocabulary
            self.files.append(outfile_bg)
            self.vocab_tokenizer.fit_on_texts(self.query_vocab)

            print("INFO - total queries %d, unusable after preprocessing %d" % (num_total, num_empty))
        # move file to processed directory
        if self.move_files:
            if not os.path.isdir("processed"):
                os.mkdir("processed")
            os.rename(filename, "processed/" + filename)
            print("INFO -- moved file %s" % filename)

    @staticmethod
    def tokenize_w_nltk(query):
        return word_tokenize(query)

    @staticmethod
    def remove_non_alphanumric(string):
        return re.sub(r'\W', ' ', string.lower())

    # currently not in use
    def remove_stop_words(self, string):
        words = string.split()
        return ' '.join([w for w in words if w not in self.stop])

    def save_vocab(self):

        print("Save objects to files in directory %s" % self.file_input_path)
        start = timer()
        with open(self.dict_file, 'wb') as f:
            pickle.dump(self.vocab_tokenizer, f)
        end = timer()
        print("Saved objects to file in %s seconds." % (end - start))

    @staticmethod
    def load_vocab(file_input_path):
        if os.getcwd().find('data') == -1:
            os.chdir('../data')

        if os.path.isdir(file_input_path):
            os.chdir(file_input_path)

        print "Load objects from directory %s" % file_input_path

        with open(file_input_path, 'rb') as f:
            return pickle.load(f)

    def translate_words_to_indices(self, final_proc_options, final_out_dir="final_out"):

        if os.getcwd().find('data') == -1:
            os.chdir('../data')

        if os.path.isdir(self.file_input_path):
            os.chdir(self.file_input_path)

        if final_proc_options is not None:
            if not os.path.isdir(final_out_dir):
                os.mkdir(final_out_dir)
            file_ext = ".out"
            for opt in final_proc_options:
                files_to_process = glob.glob(opt + "*.dat")
                for filename in files_to_process:
                    print("filename %s" % filename)
                    with open(filename, 'r') as f:
                        outfile = final_out_dir + "/" + os.path.splitext(filename)[0] + file_ext
                        with open(outfile, 'w') as f_out:
                            for line in f:
                                # TODO: unknown words need to have index e.g. 0  unknown
                                sess_id, user_id, q_words, q_time = line.split('\t')
                                # looks awkward, but, we first need to split the words into a list
                                # this seems to be necessary for kera tokenizer i.e. the translation to indices
                                q_w_list = q_words.split(",")
                                # tranforming the result of tokenizer to numpy array, getting rid of 1 dimension
                                q_w_seq = np.reshape(self.vocab_tokenizer.texts_to_sequences(q_w_list), (len(q_w_list)))
                                # then we need again convert the numpy array to a string, separated by ","
                                q_w_str = q_w_seq.astype('str')
                                # not sure whether this works, converting empty array cells to zero = unknown word
                                # we will have to deal with this situation because the vocab will be limited to 90K
                                q_w_str[q_w_str == ''] = '0'
                                q_w_str = ",".join(q_w_str)
                                f_out.write('{}\t{}\t{}\n'.format(sess_id, user_id, q_w_str))

if __name__ == '__main__':
    p = Processor(AOL_ROOT_PATH, num_of_recs=None, make_dict=False, dict_file="aol_vocab.pck",
                  file_to_process='user-ct-test-collection-09.txt.gz')
    # only translate bg = background *.dat files for the moment
    p.translate_words_to_indices(['bg'])
    print("Number of words in vocabulary %d" % len(p.vocab_tokenizer.word_counts))

    # vocab = Processor.load_vocab("/home/jogi/git/repository/ir2_jorg/data/AOL-user-ct-collection/aol_vocab.pck")
