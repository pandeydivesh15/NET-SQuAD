import sys
import pickle
import pandas as pd

from random import shuffle

# add utils dir to path
sys.path.append('utils')

from vocab import WordVocab, POSVocab

class DataReader:
	def __init__(
			self, 
			file_path,
			vocab_path,
			named_ent_info=False,
			train_split_ratio=0.90,
			dev_split_ratio=0.05,
			test_split_ratio=0.05,
			shuffle_data=False,
			debug_mode=False, percent_debug_data=10):
		self.file_path = file_path
		self.vocab = pickle.load(open(vocab_path, 'rb'))

		self.read_data(
			named_ent_info,
			train_split_ratio, dev_split_ratio, test_split_ratio, 
			shuffle_data,
			debug_mode, percent_debug_data)

	def read_data(
			self, 
			named_ent_info,
			train_split_ratio, dev_split_ratio, test_split_ratio, 
			shuffle_data,
			debug_mode, percent_debug_data):

		assert (train_split_ratio + dev_split_ratio + test_split_ratio) == 1

		# Read CSV file
		df = pd.read_csv(self.file_path, delimiter=",")
		self.data = df.to_dict('records')

		# Temporary fix (Error in making pandas CSV)
		for x in self.data:
			x['ques'] = [int(i) for i in x['ques'].strip('[').strip(']').split(',')]
			x['context'] = [int(i) for i in x['context'].strip('[').strip(']').split(',')]
			x['pos_context'] = [int(i) for i in x['pos_context'].strip('[').strip(']').split(',')]
			x['pos_ques'] = [int(i) for i in x['pos_ques'].strip('[').strip(']').split(',')]

			if named_ent_info:
				x['CG_NE_ques'] = [int(i) for i in x['CG_NE_ques'].strip('[').strip(']').split(',')]
				x['FG_NE_ques'] = [int(i) for i in x['FG_NE_ques'].strip('[').strip(']').split(',')]
				x['CG_NE_context'] = [int(i) for i in x['CG_NE_context'].strip('[').strip(']').split(',')]
				x['FG_NE_context'] = [int(i) for i in x['FG_NE_context'].strip('[').strip(']').split(',')]

		# Now, `self.data` is a list of dict
		# shuffle(self.data)

		if debug_mode:
			self.data = self.data[:int(percent_debug_data * 0.01 * len(self.data))]

		data_size = len(self.data)
		self.train = self.data[:int(train_split_ratio*data_size)]
		self.dev = self.data[
						int(train_split_ratio * data_size):
						int((train_split_ratio + dev_split_ratio) * data_size)]
		self.test = self.data[int(- (test_split_ratio) * data_size):]
		
		if shuffle_data:
			shuffle(self.train)
			shuffle(self.dev)
			shuffle(self.test)

		self.data_dict = {
			'train':	self.train,
			'dev':		self.dev,
			'test':		self.test}

	def get_minibatch(self, batch_size):
		# Useful function to return minibatches of training data
		train_size = len(self.train)

		assert batch_size < train_size

		count = train_size // batch_size

		for i in range(count):
			yield self.train[i*batch_size : (i+1)*batch_size]

	def get_complete_batch(self, choice):
		assert (choice == 'train' or choice == 'dev' or choice == 'test')

		return self.data_dict[choice]









