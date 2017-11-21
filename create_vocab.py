# from lib.preprocess.vocabulary import Vocabulary

# labels = Vocabulary('data/signalmedia-1m.jsonl', nr_words=5000)
# out_path = 'data/'
# labels.save_vocab(out_path)

vocab_file = 'data/vocabulary.txt'

def get_idx_2_word(vocab_file):

	idx_2_word = dict()
	with open(vocab_file, 'r') as file:
		for i, line in enumerate(file):
			idx_2_word[str(i)] = line[:-1]

	return idx_2_word
idx_2_word = get_idx_2_word(vocab_file)

print idx_2_word