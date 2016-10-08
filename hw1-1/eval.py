import sys
import nltk
from nltk import distance



if __name__ == "__main__":

	ans = open(sys.argv[1], "r").read().strip().split()
	result = open(sys.argv[2], "r").read().strip().split()[0:334]


	Err = 0.0
	Wlen = 0.0
	for (w1, w2) in zip(ans, result):
		Err += distance.edit_distance(w1, w2)
		Wlen += len(w1)

	print Err/Wlen
