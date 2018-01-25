from json_lines import reader
import re
from unidecode import unidecode
import json


class Vocabulary(object):

    def __init__(self, filename):

        self.file = filename
        self.sentences = list()
        self.start_token = '<SOS>'
        self.chars = set()
        self.chars.update(self.start_token)
        self.create_vocab()

    def create_vocab(self):

        with open(self.file, 'r') as f:
            for article in reader(f):
                title = unidecode(article['title']).strip()
                self.sentences.append(title)
                self.chars.update(title)

        # Start with SOS token for char to index mapping
        index_start_token = 0
        

        # Add rest of the characters
        self.char_to_idx = {char: i + 1 for i, char in enumerate(self.chars)}
        self.char_to_idx[self.start_token] = index_start_token
        
        # Create mapping from index to char
        self.idx_to_char = {i + 1: char for i, char in enumerate(self.chars)}
        self.idx_to_char[index_start_token] = self.start_token

        self.sentences = [''.join(self.convert_sentence(sentence, self.char_to_idx))
                          for sentence in self.sentences]

    def convert_sentence(self, sentence, char_to_idx):

        sentence = str(char_to_idx[self.start_token]) + ' ' + \
            ' '.join(str(char_to_idx[char]) for char in sentence)

        return sentence

    def save_vocab(self, out_path):

        with open('idx_to_char.json', 'w') as f:
            json.dump(self.idx_to_char, f)

        # Save in id format direct;y for easier debugging wrt original implementation
        open(out_path + 'real_char.data', 'w').writelines('\n'.join(self.sentences))



