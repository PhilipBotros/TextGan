from lib.preprocess.vocabulary import Vocabulary

labels = Vocabulary('data/sample-1m.jsonl', nr_words=5)
out_path = 'data/'
labels.save_vocab(out_path)
