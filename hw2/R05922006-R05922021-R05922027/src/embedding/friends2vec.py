import argparse
import sys
import tensorflow as tf
import numpy as np
import math

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size of training pairs")
flags.DEFINE_float("learning_rate", 0.25, "Inital learning rate")
flags.DEFINE_integer("iterations", 100, "Number of iterations to train")
flags.DEFINE_integer("embedding_size", 100, "The embedding vector's length")
flags.DEFINE_integer("sample_num", 100, "Sampling number")
flags.DEFINE_boolean("interactive", True, "If true, enters an IPython interactive session to play with the trained")
FLAGS = flags.FLAGS

class Data(object):

	def __init__(self, train_pair):
		self.train = train_pair
		self.current = 0
		self.length = len(train_pair)
		self.progress = 0

	def next_batch(self, size):

		if self.current + size < self.length:
			train_x, train_y = self.train[self.current:self.current+size, 0], self.train[self.current:self.current+size, 1][:, None]
			self.current += size
			state = (False, self.progress)
			if int(float(self.current+1)/float(self.length)*100) >= self.progress*5:
				state = (True, self.progress)
				self.progress += 1

			return train_x, train_y, state

		else:
			train_x, train_y = self.train[self.current:, 0], self.train[self.current:, 1][:, None]
			self.current = 0
			state = (False, self.progress)
			self.progress = 0

			return train_x, train_y, (True, 20)

def progress_bar(value):
	out = '\rprogress : ['
	v = value[1]
	if not value[0]:
		v -= 1
	for i in range(v):
		out += '=='
	if v != 20:
		out += '>'
	for i in range(20-v):
		out += '  '
	out += '] '+str(v*5)+'%'

	return out

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_pair', default='./test2/relation.txt', type=str)
	parser.add_argument('--uid_file', default='./test2/user.txt', type=str)
	parser.add_argument('--vector', default='./friend_vector.txt', type=str)
	args = parser.parse_args()

	return args

def graph_network(args, uid_size, date):

	train_x = tf.placeholder(tf.int32, [None])
	train_y = tf.placeholder(tf.int32, [None, 1])

	with tf.device('/cpu:0'):
		uidv = tf.Variable(tf.random_uniform([uid_size, FLAGS.embedding_size], -.1, .1))
		emlt = tf.nn.embedding_lookup(uidv, train_x)
		w_nce = tf.Variable(tf.truncated_normal([uid_size, FLAGS.embedding_size], stddev=1.0 / math.sqrt(FLAGS.embedding_size)))
		b_nce = tf.Variable(tf.zeros([uid_size]))

	nce_loss = tf.nn.nce_loss(weights=w_nce, biases=b_nce, inputs=emlt, labels=train_y, 
				num_sampled=FLAGS.sample_num, num_classes=uid_size, name="nce_loss")

	cost = tf.reduce_mean(nce_loss, name="cost")

	optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(cost)

	with tf.Session() as sess:

		init = tf.initialize_all_variables()
		sess.run(init)

		batch_number = date.length/FLAGS.batch_size
		batch_number += 1 if date.length%FLAGS.batch_size > 0 else 0

		for ite in range(FLAGS.iterations):
			print >> sys.stderr, 'Iterations ',ite+1,':'
			avg_cost = 0.
			for b in range(batch_number):
				
				t_x, t_y, state = date.next_batch(FLAGS.batch_size)
				
				_, c = sess.run([optimizer, cost], feed_dict={train_x:t_x, train_y:t_y})
				
				avg_cost += c/batch_number

				if state[0] or b%100 == 0:
					print >> sys.stderr, progress_bar(state)+' '+str(b)+'/'+str(batch_number),
			print >> sys.stderr, '\r>>> cost : '+str(avg_cost) + '                                                   '

		return uidv.eval()

def get_max_uid(args):

	maxid = 0
	with open(args.uid_file, 'r') as f:
		for line in f.readlines():
			uid = line.strip()
			try:
				maxid = int(uid) if uid > maxid else maxid
			except:
				continue
	return maxid+1

def get_data(args):

	pair = []
	for line in open(args.train_pair, 'r'):
		p = map(int, line.strip().split())
		if len(p) > 0:
			r_p = p
			r_p.reverse()
			pair.append(p)
			pair.append(r_p)
	pair = np.array(pair)
	data = Data(pair)

	return data

def _start_shell(local_ns=None):
  # An interactive shell is useful for debugging/development.
  import IPython
  user_ns = {}
  if local_ns:
    user_ns.update(local_ns)
  user_ns.update(globals())
  IPython.start_ipython(argv=[], user_ns=user_ns)

def dump_vector(args, vectors):

	with open(args.vector, 'w') as f:
		for i in range(len(vectors)):
			out = str(i)+ ' ' + ' '.join([str(v) for v in vectors[i]])+'\n'
			f.write(out)

def main(_):

	args = arg_parse()

	maxid = get_max_uid(args)

	data = get_data(args)

	vectors = graph_network(args=args, uid_size=maxid, date=data)

	dump_vector(args, vectors)

	if FLAGS.interactive:
		_start_shell(locals())

if __name__ == '__main__':
	tf.app.run()
