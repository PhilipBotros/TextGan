from lib.preprocess.vocabchar import Vocabulary

labels = Vocabulary('data/sample-1M.jsonl')
out_path = 'data/'
labels.save_vocab(out_path)
