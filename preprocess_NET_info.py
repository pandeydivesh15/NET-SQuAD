import pickle
import tqdm
import pandas as pd

from utils import data_reader as DR

main_data_reader = DR.DataReader(
    file_path="./data/processed.csv",
    vocab_path="./data/saves/vocab/word_vocab.pickle",
    debug_mode=False,
    percent_debug_data=1,
    train_split_ratio=1.0,
    dev_split_ratio=0.0, test_split_ratio=0.0,
)

# Earlier processed data
complete_data = main_data_reader.get_complete_batch('train')

with open("./data/processed_NET_list", 'rb') as f:
	NET_data = pickle.load(f)

NET_data = NET_data[:len(complete_data)]

def find_NET_indices(NET_tuple, word_indices):
	txt_len = len(word_indices)

	ne_info_CG = [0 for i in range(txt_len)]
	ne_info_FG = [0 for i in range(txt_len)]

	entities = list(set(NET_tuple[0])) # Remove duplicate entries

	CG_dict = NET_tuple[1]
	FG_dict = NET_tuple[2]

	for ent in entities:
		ent_components = ent.lower().split(" ")

		ent_indices = [main_data_reader.vocab.get_index(x) for x in ent_components]

		for i in range(txt_len):
			if i + len(ent_indices) > txt_len:
			    break;

			span = word_indices[i:i + len(ent_indices)]

			
			# Temporary fix for a problem in output of SANE
			check = False
			if span != ent_indices:
				check = True
				for cnt,(ind_1, ind_2) in enumerate(zip(span, ent_indices)):
					if ind_1 != ind_2:
						word = ent_components[cnt]
						word += "."
						if word != main_data_reader.vocab.get_word(ind_1):
							check = False
							break           

			if span == ent_indices or check:
				# for NER: CG type
				if len(CG_dict[ent]) != 0 and ne_info_CG[i] == 0: 
				# Second check is to confirm if ne_info_CG is already filled(non zero) at i
			
					cg_index = main_data_reader.vocab.get_index(CG_dict[ent].lower())
					
					for j in range(i, i + len(ent_indices)):
						ne_info_CG[j] = cg_index

				# FG type
				type_ = FG_dict[ent]
				if len(type_) != 0 and type_ != 'missing' and ne_info_FG[i] == 0:

					fg_index = main_data_reader.vocab.get_index(FG_dict[ent].lower())

					for j in range(i, i + len(ent_indices)):
						ne_info_FG[j] = fg_index

	return ne_info_CG, ne_info_FG

processed_records = []

# Since many same passages will be present continuously, there is no need to process them repeatedly
# This will make code rum faster
previous_passage_data = None 
previous_passage_processed = None

for data, NE_data in tqdm.tqdm(zip(complete_data, NET_data)):
	if previous_passage_data != (NE_data[0], data['context']):
		cg_passage, fg_passage = find_NET_indices(NE_data[0], data['context'])
	else:
		cg_passage, fg_passage = (cg_passage, fg_passage)

	cg_ques, fg_ques = find_NET_indices(NE_data[1], data['ques'])

	previous_passage_data = (NE_data[0], data['context'])
	previous_passage_processed = (cg_passage, fg_passage)

	processed_records.append([
		data['context'], 
		data['ques'], 
		data['answer_start'],
		data['answer_end'],
		data['pos_context'],
		data['pos_ques'],
		cg_passage, 
		fg_passage,
		cg_ques, 
		fg_ques])

labels = [
	'context', 'ques', 
	'answer_start', 'answer_end',
	'pos_context', 'pos_ques',
	'CG_NE_context', 'FG_NE_context',
	'CG_NE_ques', 'FG_NE_ques']
df = pd.DataFrame.from_records(processed_records, columns=labels)

df.to_csv("./data/processed_with_NET.csv", sep=',', index=False)
