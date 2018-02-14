from lib.preprocess.vocabulary import Vocabulary
from lib.preprocess.vocabchar import VocabChar
from settings import parse_arguments

opt = parse_arguments()

if opt.mode == 'word':
    labels = Vocabulary('data/sample-1M.jsonl')
    out_path = 'data/'
    labels.save_vocab(out_path)
else if opt.mode == 'char':
    labels = VocabChar('data/sample-1M.jsonl')
    out_path = 'data/'
    labels.save_vocab(out_path)
else:
    raise Exception('Mode not recognized.')
