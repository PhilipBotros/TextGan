# from helper import tokenize, convert_sentence
from json_lines import reader
from langdetect import detect
from collections import Counter
import re
import json


class Vocabulary(object):

    def __init__(self, filename, nr_words=30000, titles=False):

        self.file = filename
        self.start_token = '<SOS>'
        self.words = set()
        self.words.update(self.start_token)
        self.sentences = list()
        self.cntr = Counter()
        self.titles = titles
        self.create_vocab(nr_words)

    def create_vocab(self, nr_words):

        with open(self.file, 'r') as f:

            for i, article in enumerate(reader(f)):
                if self.titles:
                    content = tokenize(article['title'])
                else:
                    content = tokenize(article['content'])
                self.cntr.update(content)
                self.sentences.append(content)
                if i % 1000 == 0:
                    print("{} articles loaded".format(i))

        print("Content loaded...")

        print("Obama: {}".format(self.cntr["obama"]))
        print("Trump: {}".format(self.cntr["trump"]))
        most_common = self.cntr.most_common(nr_words)
        self.words = [self.start_token, '<EOS>', '<UNK>'] + [
            word for word, _ in most_common
        ]

        # Convert to id directly for easier debugging wrt original implementation
        self.word_to_idx = {word: i for i, word in enumerate(self.words)}
        self.idx_to_word = {i: word for i, word in enumerate(self.words)}
        print("Words indexed...")
        self.sentences = [''.join(self.convert_sentence(sentence, self.word_to_idx))
                          for sentence in self.sentences]
        print("Sentences converted...")

    def save_vocab(self, out_path):
        print("Saving vocab...")
        with open(out_path + 'idx_to_word.json', 'w') as f:
            json.dump(self.idx_to_word, f)

        # Save in id format direct;y for easier debugging wrt original implementation
        open(out_path + 'real_content.data', 'w').writelines('\n'.join(self.sentences))

    def convert_sentence(self, sentence, word_to_idx):

        unknown_token = word_to_idx['<UNK>']
        sentence = str(word_to_idx[self.start_token]) + ' ' + \
            ' '.join(str(word_to_idx[word]) if word in word_to_idx
                     else str(unknown_token) for word in sentence)

        return sentence

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
