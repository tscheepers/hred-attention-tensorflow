import cPickle as pickle

path = '../../../hred-qs/data/dev_large/'
train_ses_pickle = 'train.ses.pkl'
valid_ses_pickle = 'valid.ses.pkl'



with open(path+valid_ses_pickle, "rb") as f:
    content_file = pickle.load(f)

#print len(content_file)

for line in content_file:
    print len(line), line
