from lib.scripts.vocabulary import Vocabulary
from lib.scripts.vocabchar import VocabChar
from settings import parse_arguments

path = './data/sample-1M.jsonl'

def create_vocab(opt):
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
