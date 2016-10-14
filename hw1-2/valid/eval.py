import sys
import nltk
from nltk import distance



if __name__ == "__main__":

	ans = open(sys.argv[1], "r").read().strip().split()
	result = open(sys.argv[2], "r").read().strip().split()
	ans = [int(a) for a in ans]
	result = [int(r) for r in result]
	
	spaceId = int(sys.argv[3])

	Err = 0.0
	Wlen = 0.0
	for (w1, w2) in zip(ans, result):
		if w1 != w2 and w1 != spaceId and w2 != spaceId:
			Err += 1.0
		if w1 != spaceId:
			Wlen += 1.0

	print Err/Wlen
