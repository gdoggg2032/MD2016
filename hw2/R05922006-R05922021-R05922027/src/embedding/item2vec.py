import argparse
import sys
import tensorflow as tf
import numpy as np
import math
import progressbar as pb

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "batch size of training pairs")
flags.DEFINE_float("learning_rate", 0.25, "Inital learning rate")
flags.DEFINE_integer("iterations", 50, "Number of iterations to train")
flags.DEFINE_integer("embedding_size", 100, "The embedding vector's length")
flags.DEFINE_integer("sample_num", 100, "Sampling number")
flags.DEFINE_boolean("interactive", True, "If true, enters an IPython interactive session to play with the trained")
FLAGS = flags.FLAGS

class Data(object):

	def __init__(self, train_pair):
		self.train = train_pair
		self.current = 0
		self.length = len(train_pair)

	def next_batch(self, size):

		if self.current + size < self.length:
			train_x, train_y, train_z, train_c = self.train[self.current:self.current+size, 0], self.train[self.current:self.current+size, 1], self.train[self.current:self.current+size, 2][:, None], self.train[self.current:self.current+size, 3][:, None]
			self.current += size

			return train_x, train_y, train_z, train_c

		else:
			train_x, train_y, train_z, train_c = self.train[self.current:, 0], self.train[self.current:, 1], self.train[self.current:self.current+size, 2][:, None], self.train[self.current:self.current+size, 3][:, None]
			self.current = 0

			return train_x, train_y, train_z, train_c

def arg_parse():

	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data', default='./test2/message.txt', type=str)
	parser.add_argument('--friend_vector', default='./friend_vector.txt', type=str)
	parser.add_argument('--uid_file', default='./test2/user.txt', type=str)
	parser.add_argument('--vector', default='./item_vector.txt', type=str)
	args = parser.parse_args()

	return args

def graph_network(args, iid_size, cid_size, date, uid_v):

	uidv = tf.constant(uid_v)
	train_x = tf.placeholder(tf.int32, [None])
	train_y = tf.placeholder(tf.int32, [None])
	train_z = tf.placeholder(tf.int32, [None, 1])
	# train_c = tf.placeholder(tf.int32, [None, 1])

	with tf.device('/cpu:0'):
		iidv = tf.Variable(tf.random_uniform([iid_size, FLAGS.embedding_size], -.1, .1))
		emuid = tf.nn.embedding_lookup(uidv, train_x)
		emiid = tf.nn.embedding_lookup(iidv, train_y)
		w_nce = tf.Variable(tf.truncated_normal([cid_size, FLAGS.embedding_size], stddev=1.0 / math.sqrt(FLAGS.embedding_size)))
		b_nce = tf.Variable(tf.zeros([cid_size]))

	nce_loss = tf.nn.nce_loss(weights=w_nce, biases=b_nce, inputs=emiid, labels=train_z, 
				num_sampled=FLAGS.sample_num, num_classes=cid_size, name="nce_loss")

	u_i_loss = - tf.log(tf.sigmoid(tf.reduce_sum(tf.mul(emuid, emiid), reduction_indices=1, keep_dims=True)))

	# count_loss = tf.pow(tf.reduce_sum(tf.matmul(emiid, tf.transpose(iidv)), reduction_indices=1, keep_dims=True), 2)

	cost = tf.reduce_mean(nce_loss + u_i_loss, name="cost")

	optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(cost)

	with tf.Session() as sess:

		init = tf.initialize_all_variables()
		sess.run(init)

		batch_number = date.length/FLAGS.batch_size
		batch_number += 1 if date.length%FLAGS.batch_size > 0 else 0

		for ite in range(FLAGS.iterations):
			print >> sys.stderr, 'Iterations ',ite+1,':'
			avg_cost = 0.

			pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_number).start()
			for b in range(batch_number):
				
				t_x, t_y, t_z, t_c = date.next_batch(FLAGS.batch_size)
				
				_, c = sess.run([optimizer, cost], feed_dict={train_x:t_x, train_y:t_y, train_z:t_z})
				
				avg_cost += c/batch_number

				pbar.update(b+1)
			pbar.finish()

			print >> sys.stderr, '\r>>> cost : '+str(avg_cost) + '                                                   '

		return iidv.eval()

def get_max_id(args):

	maxiid = 0
	maxcid = 0
	with open(args.train_data, 'r') as f:
		for line in f.readlines():
			info = line.strip().split()
			try:
				maxiid = int(info[1]) if int(info[1]) > maxiid else maxiid
				maxcid = int(info[2]) if int(info[2]) > maxcid else maxcid
			except:
				continue
	return maxiid + 1, maxcid + 1

def get_data(args):

	dat = []
	for line in open(args.train_data, 'r'):
		p = map(int, line.strip().split())
		if len(p) > 0:
			dat.append(p)
	train_dat = np.array(dat)
	data = Data(train_dat)

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

def get_uid_v(args):

	uid_v = []
	with open(args.friend_vector, 'r') as f:
		for line in f.readlines():
			vector = map(float, line.strip().split()[1:])
			uid_v.append(vector)

	return np.array(uid_v).astype('float32')

def main(_):

	args = arg_parse()

	maxiid, maxcid = get_max_id(args)

	uid_v = get_uid_v(args)

	data = get_data(args)

	vectors = graph_network(args=args, iid_size=maxiid, cid_size=maxcid, date=data, uid_v=uid_v)

	dump_vector(args, vectors)

	if FLAGS.interactive:
		_start_shell(locals())

if __name__ == '__main__':
	tf.app.run()
