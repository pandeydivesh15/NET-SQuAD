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
	temp_file_path_1 = '/tmp/input.txt'
	temp_file_path_2 = '/tmp/output.txt'

	subprocess.call('echo "' + text + '" > ' + temp_file_path_1, shell=True)

	if POS:
		command = "java edu.stanford.nlp.tagger.maxent.MaxentTagger -model \
				   edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger \
				   -textFile %s -outputFormat tsv > %s" % (temp_file_path_1, temp_file_path_2)
	else:
		command = "java edu.stanford.nlp.process.PTBTokenizer %s > %s" % (temp_file_path_1, temp_file_path_2)
	subprocess.call(command, shell=True)

	tokens, pos_tags = [], []

	with open(temp_file_path_2, 'rb') as file:
		for l in file:
			line = l.strip().split()
			if line:
				tokens.append(line[0])
				if POS:
					pos_tags.append(line[1])

	return tokens, pos_tags

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

	tokens, _ = tokenize(text[:answer_start_char], POS=False)
	answer_start = len(tokens)

	tokens, _ = tokenize(answer_text, POS=False)

	answer_end = answer_start + len(tokens) - 1

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

			for ques in passage['qas']:
				ques_tokens, pos_tags = tokenize(ques['question'])

				pos_training_file.write(" ".join(pos_tags) + ' \n')

				ques_tokens = find_word_tokens(ques_tokens)
				ques_pos_tokens = find_pos_tokens(pos_tags)

				answer_start, answer_end = find_boundaries(ques['answers'][0], passage['context'])

				processed_records.append([
					context_tokens, ques_tokens, answer_start, 
					answer_end, context_pos_tokens, ques_pos_tokens])

			pos_training_file.write("\n")

		pos_training_file.close()

	# Make a pandas dataframe and store processed data in a CSV file
	labels = ['context', 'ques', 'answer_start', 'answer_end', 'pos_context', 'pos_ques']
	df = pd.DataFrame.from_records(processed_records, columns=labels)

	save_vocabs()
	df.to_csv("./data/processed.csv", sep=',', index=False)

if __name__ == '__main__':
	main()