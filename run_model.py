from models import model

from utils import data_reader as DR

main_data_reader = DR.DataReader(
	file_path="./data/processed.csv",
	vocab_path="./data/saves/vocab/word_vocab.pickle",
	debug_mode=True,
	percent_debug_data=1)

x = model.AttentionNetwork(main_data_reader)