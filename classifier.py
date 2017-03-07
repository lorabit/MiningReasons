from common import *
from sklearn import linear_model
from sklearn import svm
import feature_extraction
import os
import os.path

def precision(xs,ys,t,reg):
    correct = 0
    total = 0
    for i in range(len(xs)):
    	if reg.predict(xs[i])>=t:
            total += 1
            if ys[i] == 1:
                correct += 1
    if total == 0:
    	return 0,0
    return float(correct)/total,total

def train():
	rs,xs,ys = dataset(train_path)
	reg = svm.SVR()
	# reg = linear_model.Ridge (alpha = .5)
	reg.fit(xs,ys)

	rs,xs,ys = dataset(test_path)
	print 'Threshold\tPrecision\tTotal'
	for t in range(20):
		t = 0.5+0.5/20*t
		p,total = precision(xs,ys,t,reg)
		print '%.2f\t%.2f\t%d' % (t,p,total)

def dataset(path):
	rs,xs,ys = [],[],[]
	for i in os.listdir(path):
		if not os.path.isfile(os.path.join(path,i)):
			_rs,_xs,_ys = feature_extraction.generate_dataset(os.path.join(path,i))
			rs += _rs
			xs += _xs
			ys += _ys
	return rs,xs,ys

def main():
	train()
			


if __name__ == '__main__':
	main()