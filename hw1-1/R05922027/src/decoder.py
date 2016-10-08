import numpy as np
import string
import sys
#from nltk import distance


alphabet = list(string.ascii_lowercase+string.digits+" ")

MAXLOW = -float('Inf')

def loadBigram(bigramFile):

	B = {}
	for line in open(bigramFile, "r"):
		ll = line
		w1 = ll[0:1]
		w2 = ll[2:3]
		f = np.log(float(ll[4:]))
		B[(w1, w2)] = f

	return B

def loadEncodeProb(encodeProbFile):

	EP = {}
	for line in open(encodeProbFile, "r"):
		ll = line
		w1 = ll[0:1]
		w2 = ll[2:3]
		ff = float(ll[4:])
		if ff == 0.0:
			f = MAXLOW
		else:
			f = np.log(ff)
		EP[(w1, w2)] = f

	EP[(" ", " ")] = 0.0
	return EP

def loadEncodedText(encodedTextFile):


	with open(encodedTextFile, "r") as f:
		encodedText = f.read()

	return encodedText

def viterbi(B, EP, encodedText):

	n_best = 4

	Vmatrix = np.zeros((len(encodedText), len(alphabet), n_best))
	#Trace = np.zeros((len(encodedText), len(alphabet))).astype('int')
	Trace = np.empty((len(encodedText), len(alphabet), n_best), dtype='|S'+str(len(encodedText)))
	for t in range(0, len(encodedText)):
		for i in range(0, len(alphabet)):
			if t == 0:
				w1 = " "
				w2 = encodedText[t]
				b = MAXLOW
				if (w1, alphabet[i]) in B:
					b = B[(w1, alphabet[i])]
				ep = MAXLOW
				if (alphabet[i], w2) in EP:
					ep = EP[(alphabet[i], w2)]
				for k in range(n_best):
					Vmatrix[t, i, k] = b + ep
					Trace[t, i, k] = alphabet[i]
			else:
				w2 = encodedText[t]
				can = []
				for j in range(len(alphabet)):
					w1 = alphabet[j]
					b = MAXLOW
					if (w1, alphabet[i]) in B:
						b = B[(w1, alphabet[i])]
					ep = MAXLOW
					if (alphabet[i], w2) in EP:
						ep = EP[(alphabet[i], w2)]
					can.append((j, Vmatrix[t-1, j, 0] + b + ep))
				can = sorted(can, key=lambda x: x[1], reverse=True)
				for k in range(n_best):
					Vmatrix[t, i, k] = can[k][1]
					#Trace[t, i] = np.argmax(can)
					Trace[t, i, k] = Trace[t-1, can[k][0], 0] + alphabet[i]

	# can = [(j, k, Vmatrix[len(encodedText)-1, j, k]) for j in range(len(alphabet)) for k in range(n_best)]
	# can = sorted(can, key=lambda x: x[2], reverse=True)
	# print can
	# for k in range(n_best, 0, -1):
	# 	w = Trace[len(encodedText)-1, can[k][0], can[k][1]]
	# 	print w, w.strip() in wordlist
	# 	if len(w.strip()) == len(encodedText) and w.strip() in wordlist or k == n_best:
	# 		decodedText = w

	'''decodedText = None
	
	for k in range(n_best):
		#print (k, Trace[len(encodedText)-1, 36, k], Vmatrix[len(encodedText)-1, 36, k])

		if Trace[len(encodedText)-1, 36, k].strip() in wordlist and not decodedText :
			decodedText = Trace[len(encodedText)-1, 36, k]
			
	if not decodedText:
		decodedText = Trace[len(encodedText)-1, 36, 0]
	'''

	decodedText = Trace[len(encodedText)-1, 36, 0]


	#end = np.argmax(Vmatrix[len(encodedText)-1, :, :])
	
	#iend = (end/Vmatrix.shape[2], end%Vmatrix.shape[2])
	
	#decodedText = Trace[len(encodedText)-1, iend[0], iend[1]]
	
	# decode = []
	
	# for t in range(len(encodedText)-1, -1, -1):
	# 	if t == len(encodedText)-1:
	# 		decode.insert(0, np.argmax(Vmatrix[t, :]))
	# 	else:
	# 		decode.insert(0, Trace[t+1, decode[0]])


	# decodedText = "".join([alphabet[i] for i in decode])

	return decodedText

# def splitWordlist():

# 	D = {}
# 	for w in wordlist:
# 		l = len(w)
# 		if l not in D:
# 			D[l] = [w]
# 		else:
# 			D[l].append(w)

# 	return D
				
if __name__ == "__main__":

	bigramFile = sys.argv[1]
	encodeProbFile = sys.argv[2]
	encodedTextFile = sys.argv[3]
	predictFile = sys.argv[4]

	B = loadBigram(bigramFile)
	EP = loadEncodeProb(encodeProbFile)
	encodedText = loadEncodedText(encodedTextFile)

	#wordDict = splitWordlist()


	# for each segement: do viterbi

	decoded = []

	encoded = encodedText.strip().split(" ")
	t = len(encoded)

	for i, seg in enumerate(encoded):
		print >> sys.stderr, i, "/", t
		en = " "+seg+" "
		de = viterbi(B, EP, en)
		word = de[1:len(de)-1]
		decoded.append(word)
		# try:
		# 	can = [(distance.edit_distance(word, x), x) for x in wordDict[len(word)]] 
		# except:
		# 	can = []
		# can_sorted = sorted(can, key=lambda x: x[0])
		# if len(can_sorted) == 0:
		# 	decoded.append(word)
		# else:
		# 	decoded.append(can_sorted[0][1])


	print >> open(predictFile, "w"), " ".join(decoded)



