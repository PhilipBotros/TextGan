from lib.preprocess.vocabulary import Vocabulary
from lib.preprocess.vocabchar import VocabChar
from settings import parse_arguments

opt = parse_arguments()
path = './data/sample-1M.jsonl'

if opt.mode == 'word':
    labels = Vocabulary(path)
    out_path = './data/'
    labels.save_vocab(out_path)
elif opt.mode == 'char':
    labels = VocabChar(path)
    out_path = './data/'
    labels.save_vocab(out_path)
else:
    raise Exception('Mode not recognized.')
