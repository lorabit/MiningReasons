
import enchant
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


t = TweetTokenizer()
d = enchant.Dict("en_US")
stemmer = PorterStemmer()
stopword = set(stopwords.words('english'))

def tokenize(text):
	ret = t.tokenize(text)
	return ret

def stem(text):
	ret = []
	for word in tokenize(text):
		word = word.lower()
		if not d.check(word):
			continue
		if word in stopword:
			continue
		word = stemmer.stem(word)
		ret += [word]
	return ret

def similarity(candidate1, candidate2):
	set1 = set(stem(candidate1))
	set2 = set(stem(candidate2))
	return calc(set1,set2)

def calc(set1, set2):
	intersection = set1.intersection(set2)
	union = set1.union(set2)
	if len(union) == 0:
		return 0
	return float(len(intersection))/len(union)

def prepare(candidate):
	return set(stem(candidate))



def main():
	a = 'yes they should they are humas to and if you love some on so much you get married and live together'
	b = 'Why not? It is a choice and everybody deserves to get married to the ones they love.'
	print similarity(a,b)

if __name__ == '__main__':
	main()