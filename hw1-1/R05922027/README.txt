instructions to execute codes

==================================

you have to install: numpy, python(2.7)


above packages are necessary

source: decoder.py


==================================
platform:

	OS: OS X El Capitan 10.11
	Memory: 8GB

----------------------------------
demo:

	./src/demo.sh

	this script will generate pred.txt in R05922027/


==================================


execute:

	python decoder.py [bigramFile] [encodeProbFile] [encodedTextFile] [predictFile]


	arguments:

		[bigramFile]
		bigram language model
		Each line contains a triplet <a, b, f> denoting the bigram frequency f > 0 for symbol pair (a, b)
		:f in <a, b, f> means P(b | a) = f

		[encodeProbFile]
		encode probability model
		Each line contains a triplet <x, y, p> denoting the probability of the event “symbol x is encoded as symbol y”, a.k.a., P(y | x)

		[encodedTextFile]
		encoded text
		which only contains lowercase letters, digits and whitespaces

	output:
		[predictFile]
		pred.txt
		decoding prediction, which has the same length as [encodedTextFile]



	example:

		python decoder.py ./bigram.txt ./encode.txt ./test.txt ../pred.txt


