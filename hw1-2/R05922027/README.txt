instructions to execute codes

==================================
example:



	in [R05922027]/src/: 

		make
		make training
		make testing

----------------------------------
compile:

	in [R05922027]/src/: 

		make

----------------------------------
train:

	in [R05922027]/src/: 

		make training ITER=[iteration] EN=[encodeProbFile] NUM=[encodedNumFile]

	arguments:

		[iteration] (default: 5)
		the max iteration numbers
		the EM algorithm update model iteratively util converge or run [iteration] iterations.
		bigram language model
		Each line contains a triplet <a, b, f> denoting the bigram frequency f > 0 for symbol pair (a, b)
		:f in <a, b, f> means P(b | a) = f

		[encodeProbFile] (default: ./encode.bin)
		encode probability model
		Each line contains a triplet <x, y, p> denoting the bitmap where the probability of encoding a character to another is 0 or not.

		[encodedNumFile] (default: ./test.num)
		a sequence of encoded symbols

	output:

		model.init:
		initial state probabilities

		model.trans:
		state transition probabilities

		model.emiss:
		emission probabilites


test:

	in [R05922027]/src/: 

		make testing PRED=[predicNumFile]

	output:

		[predicNumFile] (default: ./pred.num)
		decoding prediction




