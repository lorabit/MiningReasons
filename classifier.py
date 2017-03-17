from common import *
from sklearn import linear_model
from sklearn import svm
import feature_extraction
import os
import os.path
import similarity

def precision(xs,ys,t,reg):
    correct = 0
    total = 0
    for i in range(len(xs)):
    	if reg.predict(xs[i])>=t:
            total += 1
            if ys[i] == 1:
                correct += 1
    total_correct = 0
    for i in ys:
    	if i==1:
    		total_correct += 1
    if total == 0:
    	return 0,0
    return float(correct)/total,float(correct)/total_correct


def set_precision(xs,ys,reason_set):
    correct = 0
    total = 0
    for i in range(len(xs)):
    	if i in reason_set:
            total += 1
            if ys[i] == 1:
                correct += 1
    total_correct = 0
    for i in ys:
    	if i==1:
    		total_correct += 1
    if total == 0:
    	return 0,0
    return float(correct)/total,float(correct)/total_correct

def train():
	rs,xs,ys = dataset(train_path)
	reg = svm.SVR()
	# reg = linear_model.Ridge (alpha = .5)
	reg.fit(xs,ys)

	rs,xs,ys = dataset(test_path)
	print 'Threshold\tPrecision\tRecall'
	for t in range(20):
		t = 0.5+0.5/20*t
		p,total = precision(xs,ys,t,reg)
		print '%.2f\t%.2f\t%.2f' % (t,p,total)
	return reg

def gc_feature(rs,xs,ys,ci):
	maxSim = 0
	for i in range(len(rs)):
		if i!=ci:
			sim = similarity.similarity(rs[ci],rs[i])
			if sim > maxSim:
				maxSim = sim
	return [maxSim]

def train_gc():
	rs,xs,ys = dataset_gc(train_path)
	reg = svm.SVR()
	# reg = linear_model.Ridge (alpha = .5)
	reg.fit(xs,ys)

	rs,xs,ys = dataset_gc(test_path)
	print 'Threshold\tPrecision\tRecall'
	for t in range(20):
		t = 0.5+0.5/20*t
		p,total = precision(xs,ys,t,reg)
		print '%.2f\t%.2f\t%.2f' % (t,p,total)
	return reg

def test():
	ic = train()
	gc = train_gc()
	rs,bxs,ys = dataset(test_path)
	prepared = {}
	print 'Preparing...'
	for i in range(len(rs)):
		prepared[i] = similarity.prepare(rs[i])
	print 'Threshold\tPrecision\tRecall\tRounds'
	for ti in range(20):
		xs = [[j for j in i] for i in bxs]
		t = 0.5 + 0.5/20*ti
		reason_set = set()
		candidate_set = set()
		for i in range(len(rs)):
			if ic.predict(xs[i])>t:
				reason_set.add(i)
			else:
				candidate_set.add(i)
		sim = dict()
		for i in range(len(rs)):
			for j in range(i):
				sim[(i,j)] = sim[(j,i)] = similarity.calc(prepared[i],prepared[j])
		proceed = True
		for i in candidate_set:
			xs[i] += [0]
		rd = 0
		while proceed:
			rd += 1
			# print 'Round #%d' %(rd,)
			proceed = False
			new_reason = set()
			for i in candidate_set:
				xs[i][-1] = max([sim[(i,j)] for j in reason_set])
				if gc.predict(xs[i])>=0.5:
					new_reason.add(i)
			for i in new_reason:
				reason_set.add(i)
				candidate_set.remove(i)
				proceed = True
		p,total = set_precision(xs,ys,reason_set)
		print '%.2f\t%.2f\t%.2f\t%d' % (t,p,total,rd)

def dataset(path):
	rs,xs,ys = [],[],[]
	for i in os.listdir(path):
		if not os.path.isfile(os.path.join(path,i)):
			_rs,_xs,_ys = feature_extraction.generate_dataset(os.path.join(path,i))
			rs += _rs
			xs += _xs
			ys += _ys
	return rs,xs,ys

def dataset_gc(path):
	rs,xs,ys = [],[],[]
	for i in os.listdir(path):
		if not os.path.isfile(os.path.join(path,i)):
			_rs,_xs,_ys = feature_extraction.generate_dataset(os.path.join(path,i))
			print 'Adding growing features...'
			sim = {}
			tokens = {}
			for i in range(len(_rs)):
				sim[i] = 0
				tokens[i] = similarity.prepare(_rs[i])
			for i in range(len(_rs)):
				for j in range(i):
					s = similarity.calc(tokens[j],tokens[i])
					if s>sim[i] and _ys[j] == 1:
						sim[i] = s
					if s>sim[j] and _ys[i] == 1:
						sim[j] = s
			for i in range(len(_rs)):
				_xs[i] += [sim[i]]
			rs += _rs
			xs += _xs
			ys += _ys
	return rs,xs,ys



def main():
	test()
			


if __name__ == '__main__':
	main()