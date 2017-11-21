from helper import tokenize, convert_sentence
from json_lines import reader
from collections import Counter
import re


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
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.sentences = [''.join(convert_sentence(sentence, self.word_to_idx))
                          for sentence in self.sentences]

    def save_vocab(self, out_path):

        # Can use this later directly in TF graph
        open(out_path + 'vocabulary.txt', 'w').writelines('\n'.join(self.words))

        # Save in id format direct;y for easier debugging wrt original implementation
        open(out_path + 'training_data.txt', 'w').writelines('\n'.join(self.sentences))

#---------------------------------------------------------------------------------------------------
# tokenizer


def tokenize(string):
    # Replace n't with  not.
    string = string.lower().replace('n\'t', ' not')

    # Split words on non-alphanumerical characters.
    words = re.split("[^a-z0-9]", string)

    # Remove words that are not alphanumerical.
    words = [word for word in words if word.isalnum()]

    # Remove empty words.
    words = [word for word in words if len(word) > 0]

    return words


def convert_sentence(sentence, word_to_idx):

    unknown_token = word_to_idx['<UNK>']
    sentence = ' '.join(str(word_to_idx[word]) if word in word_to_idx
                        else str(unknown_token) for word in sentence)

    return sentence
