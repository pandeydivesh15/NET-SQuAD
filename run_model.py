import json

from models import model

from utils import data_reader as DR

RESULT_PRED_FILE = "./data/result_dev.json"

DEV_BATCH_SIZE = 18
PRED_DICT = {}

main_data_reader = DR.DataReader(
	file_path="./data/processed_with_NET.csv",
	vocab_path="./data/saves/vocab/word_vocab.pickle",
	named_ent_info=True,
	shuffle_data=True,
	train_split_ratio=1.0,
	dev_split_ratio=0.0, 
	test_split_ratio=0.0,
	debug_mode=False,
	percent_debug_data=1)

dev_data_reader = DR.DataReader(
	file_path="./data/processed_with_NET_dev.csv",
	vocab_path="./data/saves/vocab/word_vocab.pickle",
	named_ent_info=True,
	shuffle_data=True,
	train_split_ratio=1.0,
	dev_split_ratio=0.0, 
	test_split_ratio=0.0,
	debug_mode=False,
	percent_debug_data=1)

m = model.AttentionNetwork(main_data_reader)

# Also loads up the model if previous exists
m.train(epochs=20, test_model_after_epoch=False)

for batch in dev_data_reader.get_minibatch(DEV_BATCH_SIZE):
	predictions = m.predict(batch)

	for i, elem in enumerate(batch):
		pred = (predictions[i][0], predictions[i][1])
		text_pred = " ".join([dev_data_reader.vocab.get_word(i) for i in elem['context'][pred[0] : pred[1] + 1]])

		PRED_DICT[elem['id']] = text_pred

# Saving predictions
with open(RESULT_PRED_FILE, 'w') as f:
	json.dump(PRED_DICT, f)







