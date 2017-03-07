from common import *
import data_preprocess
import os
import os.path
import enchant
from nltk.stem.porter import *
from math import log

from nltk.tokenize import TweetTokenizer
t = TweetTokenizer()
d = enchant.Dict("en_US")
stemmer = PorterStemmer()

def tokenize(text):
	ret = t.tokenize(text)
	return ret

def stem(text):
	ret = []
	for word in tokenize(text):
		word = word.lower()
		if not d.check(word):
			continue
		word = stemmer.stem(word)
		ret += [word]
	return ret

def calc_idf(dataset):
	idf = dict()
	for text,_,_ in dataset:
		for word in set(stem(text)):
			if word in idf:
				idf[word] += 1
			else:
				idf[word] = 1
	for k in idf:
		idf[k] = log(float(len(dataset))/idf[k])
	return idf

def total_idf(argument, candidate, idf):
	total = 0
	for word in stem(candidate):
		if word in idf:
			total += idf[word]
	return total

def average_idf(argument, candidate, idf):
	total = 0
	count = 0
	for word in stem(candidate):
		count += 1
		if word in idf:
			total += idf[word]
	if count == 0:
		return 0
	return total*1.0/count

def discourse_marker(argument, candidate, idf):
	text,_,_ = argument
	prefix = text[:text.find(candidate)].lower()
	markers = ['however','but','because','since','hence','as']
	for marker in markers:
		if prefix.rfind(marker)>len(prefix) - len(marker) - 3:
			return 1
	return 0

def numeric_token(argument, candidate, idf):
	def has_number(inputString):
		return any(char.isdigit() for char in inputString)
	ret = 0
	tokens = tokenize(candidate)
	for word in tokens:
		if has_number(word):
			ret += 1
	if len(tokens) == 0:
		return 0
	return float(ret)

def contains_quote(argument, candidate, idf):
	ret = 0
	for i in [':','"',"-",")","("]:
		if i in candidate:
			return 1
	return 0

def num_quote(argument, candidate, idf):
	ret = 0
	for i in [':','"',"-",")","("]:
		if i in candidate:
			ret += 1
	return float(ret)

def lexicon_token(argument, candidate, idf):
	ret = 0
	stems = stem(candidate)
	if len(stems) == 0:
		return 0
	for i in stems:
		if i in ['studi','show','research','indic','report','scientist','know','result','evid','instanc','exampl','case','when', 'whi', 'where', 'what']:
			ret += 1
	return float(ret)

def marker_token(argument, candidate, idf):
	ret = 0
	stems = stem(candidate)
	if len(stems) == 0:
		return 0
	for i in stems:
		if i in ['if','howev', 'becaus', 'sinc', 'henc', 'reason','as','result']:
			ret += 1
	return float(ret)

def context_position(argument, candidate, idf):
	text,_,_ = argument
	return float(text.find(candidate))

def candidate_length(argument, candidate, idf):
	return len(candidate)

def candidate_token_length(argument, candidate, idf):
	return len(tokenize(candidate))

def generate_feature(argument, candidate, idf, feature_set):
	ret = []
	for f in feature_set:
		ret += [f(argument, candidate, idf)]
	return ret

def generate_dataset(path):
	dataset = data_preprocess.generate_data(path)
	idf = calc_idf(dataset)
	feature_set = [	
					total_idf,
					average_idf,
					discourse_marker,
					numeric_token,
					contains_quote,
					num_quote,
					lexicon_token,
					marker_token,
					context_position,
					candidate_length,
					candidate_token_length
				]
	xs = []
	ys = []
	rs = []
	for argument in dataset:
		text, nr, r = argument
		for c in nr:
			xs += [generate_feature(argument, c, idf, feature_set)]
			ys += [0.0]
			rs += [c]
		for c in r:
			xs += [generate_feature(argument, c, idf, feature_set)]
			ys += [1.0]
			rs += [c]
	
	return rs,xs,ys


def main():
	print(stem('when why where what '))
	print contains_quote('','there are also reasons why abortion should be acceptable and one reason would be if it puts the mother\'s health at risk','')
	return
	path = train_path
	for i in os.listdir(path):
		if not os.path.isfile(os.path.join(path,i)):
			generate_dataset(os.path.join(path,i))
			break

if __name__ == '__main__':
	main()