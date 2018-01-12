import re


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


def get_idx_2_word(vocab_file):

    idx_2_word = dict()
    with open(vocab_file, 'r') as f:
        for i, line in enumerate(f):
            idx_2_word[str(i)] = line[:-1]

    return idx_2_word
