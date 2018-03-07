from models import model

from utils import data_reader as DR

main_data_reader = DR.DataReader(
	file_path="./data/processed_with_NET.csv",
	vocab_path="./data/saves/vocab/word_vocab.pickle",
	named_ent_info=True,
	shuffle_data=True,
	debug_mode=False,
	percent_debug_data=1)

m = model.AttentionNetwork(main_data_reader)

m.train()