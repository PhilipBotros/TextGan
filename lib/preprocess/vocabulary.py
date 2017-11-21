from helper import tokenize, convert_sentence
from json_lines import reader
from collections import Counter

class Vocabulary(object):

	def __init__(self, filename, nr_words=30000):

		self.file = filename
		self.words = set()
		self.sentences = list()
		self.cntr = Counter()
		self.create_vocab(nr_words)

	def create_vocab(self, nr_words):

		with open(self.file, 'r') as f:

		    for article in reader(f):
		        title = tokenize(article['title'])
		        self.cntr.update(title)
		        self.sentences.append(title)

		most_common = self.cntr.most_common(nr_words)
		self.words = ['<SOS>', '<EOS>', '<UNK>'] + [
		     word for word, _ in most_common 
		]

		# Convert to id directly for easier debugging wrt original implementation
		self.word_to_idx = { word:i for i,word in enumerate(self.words) }
		self.sentences = [ ''.join(convert_sentence(sentence, self.word_to_idx)) for sentence in self.sentences ]

	def save_vocab(self, out_path):

		# Can use this later directly in TF graph
		open(out_path + 'vocabulary.txt', 'w').writelines('\n'.join(self.words))

		# Save in id format direct;y for easier debugging wrt original implementation
		open(out_path + 'training_data.txt', 'w').writelines('\n'.join(self.sentences))

