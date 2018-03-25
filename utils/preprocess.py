import json
import os
import subprocess
import pandas as pd
import pickle

from tqdm import tqdm

from vocab import WordVocab, POSVocab

VOCAB_SAVE_PATH = "./data/saves/vocab/"
POS_SAVE_PATH = "./data/saves/POS_embedd_train/"

WORD_VOCAB = WordVocab()
POS_VOCAB = POSVocab()

def save_vocabs():
	pickle_out = open(VOCAB_SAVE_PATH + 'word_vocab.pickle', 'wb')
	pickle.dump(WORD_VOCAB, pickle_out)
	pickle_out.close()

	pickle_out = open(VOCAB_SAVE_PATH + 'POS_vocab.pickle', 'wb')
	pickle.dump(POS_VOCAB, pickle_out)
	pickle_out.close()

def tokenize(text, POS=True):
	temp_file_path_1 = '/tmp/input1.txt'
	temp_file_path_2 = '/tmp/output1.txt'

	with open(temp_file_path_1, 'w+') as f:
		f.write(text.encode('utf8'))
	# subprocess.Popen('echo "%s" > %s' % (text, temp_file_path_1), shell=True, stdout=subprocess.PIPE, executable="/bin/bash")

	if POS:
		command =  "java edu.stanford.nlp.tagger.maxent.MaxentTagger -model " + \
				   "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger " + \
				   "-textFile %s -outputFormat tsv > %s" % (temp_file_path_1, temp_file_path_2)
	else:
		command = "java edu.stanford.nlp.process.PTBTokenizer %s > %s" % (temp_file_path_1, temp_file_path_2)
	subprocess.call(command, shell=True)

	tokens, pos_tags = [], []

	with open(temp_file_path_2, 'rb') as file:
		for l in file:
			line = l.decode('utf8').strip().split()
			if line:
				tokens.append(line[0])
				if POS:
					pos_tags.append(line[1])
	return tokens, pos_tags

def tokenize_questions(text, POS=True):
	temp_file_path_1 = '/tmp/input2.txt'
	temp_file_path_2 = '/tmp/output2.txt'

	with open(temp_file_path_1, 'w+') as f:
		f.write(text.encode('utf8'))

	if POS:
		command =  "java edu.stanford.nlp.tagger.maxent.MaxentTagger -model " + \
				   "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger " + \
				   "-textFile %s -outputFormat tsv > %s" % (temp_file_path_1, temp_file_path_2)
	else:
		command = "java edu.stanford.nlp.process.PTBTokenizer %s > %s" % (temp_file_path_1, temp_file_path_2)
	subprocess.call(command, shell=True)

	tokens_list, pos_tags_list = [], []
	tokens, pos_tags = [], []

	with open(temp_file_path_2, 'rb') as file:
		check = True
		for l in file:
			line = l.decode('utf8').strip().split()
			if line and line[0] != '<None>':
				if check:
					tokens.append(line[0])
					if POS:
						pos_tags.append(line[1])
			else:
				if line:
					tokens_list.append(tokens)
					if POS:
						pos_tags_list.append(pos_tags)
					tokens, pos_tags = [], []
					check = False
				else:
					if not check:
						check = True
	
	return tokens_list, pos_tags_list

def tokenize_boundaries(text):
	temp_file_path_1 = '/tmp/input3.txt'
	temp_file_path_2 = '/tmp/output3.txt'

	with open(temp_file_path_1, 'w+') as f:
		f.write(text.encode('utf8'))

	command = "java edu.stanford.nlp.process.PTBTokenizer %s > %s" % (temp_file_path_1, temp_file_path_2)
	
	subprocess.call(command, shell=True)

	tokens_list = []
	tokens = []

	with open(temp_file_path_2, 'rb') as file:
		for l in file:
			line = l.decode('utf8').strip().split()
			if line:
				if line[0] == "<NULL>":
					tokens_list.append(tokens)
					tokens = []
				else:
					tokens.append(line[0])
		tokens_list.append(tokens)

	return tokens_list

def find_word_tokens(words):
	tokens = []
	for word in words:
		tokens.append(WORD_VOCAB.try_inserting(word))
	return tokens

def find_pos_tokens(pos_tags):
	tokens = []
	for tag in pos_tags:
		tokens.append(POS_VOCAB.try_inserting(tag))
	return tokens

def find_boundaries(answer_dict, text):
	answer_start_char = answer_dict['answer_start']
	answer_text = answer_dict['text']

	tokens_list = tokenize_boundaries(
					text[:answer_start_char] + " <NULL> " + answer_text)
	answer_start = len(tokens_list[0])
	answer_end = answer_start + len(tokens_list[1]) - 1

	return answer_start, answer_end # 0-based indexing

def main():
	with open("./data/train-v1.1.json", 'rb') as file:
		data = json.loads(file.read())

	data = data['data']

	processed_records = []

	for data_title in tqdm(data):
		name = data_title["title"]

		pos_training_file = open(POS_SAVE_PATH + name + ".txt", 'w+')

		for passage in data_title['paragraphs']:
			context_tokens, pos_tags = tokenize(passage['context'])

			pos_training_file.write(" ".join(pos_tags) + ' \n')

			context_tokens = find_word_tokens(context_tokens)
			context_pos_tokens = find_pos_tokens(pos_tags)

			all_questions = ""
			for ques in passage['qas']:
				all_questions += ques['question']
				all_questions += " <None>. "

			all_ques_tokens, all_pos_tags = tokenize_questions(all_questions)

			for i, ques in enumerate(passage['qas']):
				ques_tokens, pos_tags = all_ques_tokens[i], all_pos_tags[i]
				ques_id = ques['id']

				pos_training_file.write(" ".join(pos_tags) + ' \n')

				ques_tokens = find_word_tokens(ques_tokens)
				ques_pos_tokens = find_pos_tokens(pos_tags)

				answer_start, answer_end = find_boundaries(ques['answers'][0], passage['context'])

				processed_records.append([
					context_tokens, ques_tokens, ques_id,
					answer_start, answer_end, 
					context_pos_tokens, ques_pos_tokens])

			pos_training_file.write("\n")

		pos_training_file.close()

	# Make a pandas dataframe and store processed data in a CSV file
	labels = ['context', 'ques', 'id', 'answer_start', 'answer_end', 'pos_context', 'pos_ques']
	df = pd.DataFrame.from_records(processed_records, columns=labels)

	save_vocabs()
	df.to_csv("./data/processed.csv", sep=',', index=False)

if __name__ == '__main__':
	main()