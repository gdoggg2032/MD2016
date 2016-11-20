### Usage

-embedding method:
	
	You shoud change the directory of testing data in arg_parse() function in each .py file 
	before using the following instructions.

	$ python2.7 friends2vec.py 
	$ python2.7 item2vec.py
	$ python2.7 link_predict.py

	---------------------------
	you will gain three *.txt files for each instruction. They are friend_vector.txt, 
	item_vector.txt, and pred.txt respectively.

-PGM method:

	You shoud change the directory of testing data in arg_parse() function in  link_predict.py file 
	before using the following instructions.

	$ python2.7 link_predict.py

	---------------------------
	you will gain four *.txt files for each instruction. They are p_c.txt, 
	p_c_user.txt, p_c_item.txt and output.txt respectively. And output.txt is the final prediction.
