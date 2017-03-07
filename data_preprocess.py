from common import *
import os
import os.path


def filter_reason(reasons):
	ret = []
	for i in range(len(reasons)):
		contains = False
		for j in range(len(reasons)):
			if i!=j and reasons[j].find(reasons[i])!=-1:
				contains = True
				break
		if not contains:
			ret += [reasons[i]]
	return ret

def parse_file(file):
	content = ''
	reasons = []
	with open(file,'r') as infile:
		lines = infile.readlines()
		for i in range(len(lines)):
			line = lines[i]
			# print line
			line = line.decode("utf-8",'ignore').encode("ascii", 'ignore')
			if i == 0:
				content = line[:-2]
			if line[:6] == 'Line##':
				reasons += [line[6:-2]]
	reasons = filter_reason(reasons)
	non_reasons = [content]
	for i in reasons:
		new_non_reasons = []
		for j in non_reasons:
			if j.find(i)!=-1:
				p = j.find(i)
				if p != 0 and len(j[:p-1])>0:
					new_non_reasons += [j[:p-1]]
				if p != len(j)-len(i) and len(j[p+len(i):])>0:
					new_non_reasons += [j[p+len(i):]]
			else:
				new_non_reasons += [j]
		non_reasons = new_non_reasons
	return content,non_reasons,reasons


def generate_data(path):
	dataset = []
	for i in os.listdir(path):
		if i[-3:]=='rsn' and os.path.isfile(os.path.join(path,i)):
			dataset += [parse_file(os.path.join(path,i))]
	return dataset

def main():
	path = train_path
	for i in os.listdir(path):
		if not os.path.isfile(os.path.join(path,i)):
			generate_data(os.path.join(path,i))

if __name__ == '__main__':
	main()
