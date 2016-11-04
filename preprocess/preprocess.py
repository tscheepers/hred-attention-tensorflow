import re
import gensim
import os
import sys
import tarfile
import glob
import gzip

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

    if not os.path.isdir(root_dir):
        if not os.path.isfile(AOL_TAR_FILE):
            raise Exception('Expected %s to exist in data directory.' % AOL_TAR_FILE)
        tar = tarfile.open(AOL_TAR_FILE)
        sys.stdout.flush()
        tar.extractall()
        tar.close()
    os.chdir(root_dir)
    files_extracted = glob.glob("user*.txt")
    print files_extracted
    if len(files_extracted) < 10:
        gz_files = glob.glob("*.txt.gz")
        gz_to_process = []
        for gz_f in gz_files:
            if not os.path.splitext(gz_f)[0] in files_extracted:
                gz_to_process.append(gz_f)

    return files_extracted, gz_to_process


class Processor:
    def __init__(self, path, num_of_recs=None, dict_file=None,
                 make_dict=False):

        self.query_texts = []
        self.dictionary = None
        self.make_dict = make_dict
        self.dict_file = dict_file
        self.max_prune_words = 90000
        # get all the files we need to process
        # files list: contains user*.txt files that need to be processed
        # gz_files list: contains user*.txt.gz files that need to be processed
        #
        self.files, self.gz_files = get_files(AOL_ROOT_PATH)
        self.num_of_recs = num_of_recs

        # self.stop = stopwords.words("english")
        # Pre processing step 1
        # ========================
        #       - extract gz-files
        #       - take the first three columns of each row: AnonID, Query, QueryTime
        #       - remove non-alphanumeric characters and convert to lowercase
        #       - save to "user*.txt" file
        #       - move gz file to "processed" directory (because we query the data directory for gz-files)

        if not os.getcwd().find(path) == -1 and len(self.gz_files) != 0:
            print "INFO - processing gz %d gz files" % len(self.gz_files)
            self.process_gz_files()
        else:
            raise Exception('Expected %s to exist in data directory.' % path)

        # Pre processing step 2
        # ========================
        # make or extend the existing word-dictionary
        # these steps can be also merged with first processing step, just started right
        #       - uses the self.files list to process all the user*.txt docs from 1st step
        #       - per file, extracts all query words (assuming these were filtered/cleaned before)
        #         and adds the words to the dictionary.
        #         gensim module automatically creates this dictionary and we also use the "prune" feature
        #         making sure the dict never grows beyond certain size, gensim prunes non-frequent words
        if self.make_dict:
            self.load_dict()
            if len(self.files) != 0:
                print "INFO - Make/extend dictionary"
                self.make_dictionary()
                self.save_dict()
            else:
                print("No more files to process for dictionary.")

    def load_dict(self):

        if os.path.isfile(self.dict_file):
            self.dictionary = gensim.corpora.dictionary.Dictionary.load(self.dict_file)
        self.dictionary = gensim.corpora.dictionary.Dictionary()

    def save_dict(self):
        self.dictionary.save(self.dict_file)
        print "INFO -- Saved word dictionary %s. Contains %d words" % (self.dict_file, len(self.dictionary))

    def make_dictionary(self):
        """
            Loop through the user*.txt files
            extract query words (that were filtered earlier)
            and add them to the word dictionary (gensim)
        """
        for filename in self.files:
            print "INFO - processing file %s" % filename
            query_words = []
            with open(filename) as f:
                for query in f:
                    query_words.append((query.split('\t')[1]).split())
            # automatically prunes words from the dictionary if size exceeds max_prune_words
            # prunes low frequency words....should do that at least
            self.dictionary.add_documents(query_words, self.max_prune_words)

    def process_gz_files(self):

        for file in self.gz_files:
            print "INFO - currently processing gz file: %s" % file
            self.process_one_gz_file(file)
            break

    def process_one_gz_file(self, filename, num_records=None):
        with gzip.open(filename, 'r') as src:
            outfile = os.path.splitext(os.path.splitext(filename)[0])[0]
            with open(outfile + ".txt", 'w') as dst:
                num_total = 0
                for line in src:
                    AnonID, Query, QueryTime = line.split('\t')[:3]

                    if AnonID == 'AnonID':
                        continue

                    if num_total < self.num_of_recs or self.num_of_recs is None:
                        tidy = self.remove_non_alphanumric(Query)
                        if tidy != '':
                            dst.write('{}\t{}\t{}\n'.format(AnonID, Query, QueryTime))
                            num_total += 1

        # move file to processed directory
        if not os.path.isdir("processed"):
            os.mkdir("processed")
        os.rename(filename, "processed/" + filename)

    def remove_non_alphanumric(self, string):
        return re.sub(r'\W', ' ', string.lower())

    # currently not in use
    def remove_stop_words(self, string):
        words = string.split()
        return ' '.join([w for w in words if w not in self.stop])

    def porter_stemming(self, string):
        result = [self.porter.stem(word, 0, len(word) - 1) for word in string.split()]
        return ' '.join(result)


if __name__ == '__main__':
    p = Processor(AOL_ROOT_PATH, num_of_recs=100, make_dict=True, dict_file="aol_wdic.dict")
