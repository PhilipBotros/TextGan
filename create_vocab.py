from lib.preprocess.vocabchar import Vocabulary

labels = Vocabulary('data/signalmedia-1m.jsonl')
out_path = 'data/'
labels.save_vocab(out_path)
