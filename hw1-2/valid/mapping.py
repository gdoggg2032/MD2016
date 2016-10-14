import sys




def mapping(D, nums):

	A = ""
	for n in nums:
		A += D[n]

	return A


def getDict(dicFile):

	D = {}

	for line in open(dictFile, "r"):
		ll = line.split('\t')
		D[ll[1].strip()] = ll[0]

	return D




if __name__ == "__main__":

	dictFile = sys.argv[1]
	numsFile = sys.argv[2]
	dumpFile = sys.argv[3]

	D = getDict(dictFile)

	nums = open(numsFile, "r").read().split()

	A = mapping(D, nums)

	print >> open(dumpFile, "w"), A





