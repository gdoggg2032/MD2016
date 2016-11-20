import networkx as nx
# import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys
import time
import progressbar as pb



def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--user', default='./user.txt', type=str)
	parser.add_argument('--relation', default='./relation.txt', type=str)
	parser.add_argument('--message', default='./message.txt', type=str)
	parser.add_argument('--predict', default='./pred.id', type=str)
	parser.add_argument('--response', default='./response.txt', type=str)
	parser.add_argument('--mode', default=2, type=int)
	parser.add_argument('--output', default='./output.txt', type=str)
	args = parser.parse_args()

	return args

class linkpredict(object):

	def __init__(self, args):
		self.args = args
		self.load_data()

		self.get_features()


		self.path = {0:0, 1:0, 2:0}
		
		self.p = {0:[], 1:[], 2:[]}


	def load_data(self):
		nU = set()
		nI = set()
		nC = set()
		tL = {}
		i2cs = {}
		c2is = {}
		uowni = {}

		u2fs = {}

		Y_item = {}
		Y_num = 0




		print >> sys.stderr, "start loading network and preds"
		s_time = time.time()

		G = nx.Graph()

		# load user_id
		with open(self.args.user, "r") as f:
			for line in f:
				u_id = int(line)
				nU.add(u_id)
				user = ('u', u_id)
				G.add_node(user)

		# load user relationship(friends)
		with open(self.args.relation, "r") as f:
			for line in f:
				ll = line.strip().split()
				u_id1 = int(ll[0])
				u_id2 = int(ll[1])

				# bi-direction
				user1 = ('u', u_id1)
				user2 = ('u', u_id2)
				G.add_edge(user1, user2)
				G.add_edge(user2, user1)

				if user1 not in u2fs:
					u2fs[user1] = []
				u2fs[user1].append(user2)

				if user2 not in u2fs:
					u2fs[user2] = []
				u2fs[user2].append(user1)

		# load item, owner, category, link constraint
		with open(self.args.message, "r" ) as f:
			for line in f:
				ll = line.strip().split()
				owner_u_id = int(ll[0])
				item_id = int(ll[1])
				c_id = int(ll[2])
				links = int(ll[3])

				user = ('u', owner_u_id)
				item = ('i', item_id)
				c = ('c', c_id)

				if item not in i2cs:
					i2cs[item] = set()
				i2cs[item].add(c)


				if c not in c2is:
					c2is[c] = set()
				c2is[c].add(item)
				if user not in uowni:
					uowni[user] = set()
				uowni[user].add(item)

				nI.add(item_id)
				nC.add(c_id)

				G.add_node(item)
				G.add_node(c)
				tL[item] = links

				G.add_edge(user, item)
				G.add_edge(item, c)


		with open(self.args.predict, "r") as f:
			for line in f:
				ll = line.strip().split()
				u_id = int(ll[0])
				item_id = int(ll[1])
				if item_id not in Y_item:
					Y_item[item_id] = []
				Y_item[item_id].append(u_id)
				Y_num += 1

		print >> sys.stderr, "finish loading network and preds, time cost: ", time.time() - s_time

		self.nU = nU
		self.nI = nI
		self.nC = nC
		self.tL = tL
		self.G = G
		self.Y_item = Y_item
		self.Y_num = Y_num
		self.i2cs = i2cs		
		self.c2is = c2is
		self.uowni = uowni
		self.u2fs = u2fs

		print >> sys.stderr, "nI: ", len(nI)
		print >> sys.stderr, "nU: ", len(nU)
		print >> sys.stderr, "nC: ", len(nC)
		print >> sys.stderr, "tL: ", len(tL)


	def get_features(self):

		# features = num_friends
		s_time = time.time()
		print >> sys.stderr, "start computing features "

		# p(user)
		# p_user = {}
		# total = 0.0
		# for u_id in self.nU:
		# 	user = ('u', u_id)
		# 	count = sum(1 for n in self.G.neighbors(user) if n[0] == 'u') + 0.0001
		# 	p_user[u_id] = count
		# 	total += count
		# # normalize p(user)
		# for u_id in self.nU:
		# 	p_user[u_id] /= total

		# self.p_user = p_user



		# p(f, user)

		p_f_user = {}
		total = 0.0
		add_one = 0.0000
		# for u_id in self.nU:
			
		# 	user = ('u', u_id)
		# 	if user in self.u2fs:
		# 		for f in self.u2fs[user]:
		# 			f_id = f[1]
		# 			p_f_user[(u_id, f_id)] = 1.0
		# 			total += 1.0


		for user in self.u2fs:
			u_id = user[1]
			for f in self.u2fs[user]:
				f_id = f[1]
				p_f_user[(u_id, f_id)] = 1.0
				total += 1.0
		# normalize
		for v in p_f_user:
			p_f_user[v] /= total

		self.p_f_user = p_f_user


		# p(item_own, user)

		p_o_user = {}
		total = 0.0
		for user, own_items in self.uowni.iteritems():
			u_id = user[1]
			for item in own_items:
				i_id = item[1]
				p_o_user[(i_id, u_id)] = 1.0
				total += 1.0

		# normalize
		for v in p_o_user:
			p_o_user[v] /= total

		self.p_o_user = p_o_user



		# p(c , item), p(c)
		p_c_item = {}
		p_c = {}
		total = 0.0
		add_one = 0.000

		pbar = pb.ProgressBar(widgets=["p(c, item):", pb.FileTransferSpeed(unit="items"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(self.nI)).start()
		# for i, i_id in enumerate(self.nI):
		# 	pbar.update(i)
		# 	item = ('i', i_id)
		# 	for c_id in self.nC:
		# 		c = ('c', c_id)
		# 		if c in self.i2cs[item]:
		# 			p_c_item[(c_id, i_id)] = 1.0
		# 			p_c[c_id] = p_c.get(c_id, 0.0) + 1.0
		# 			total += 1.0
		# 		# else:
		# 		# 	#smooth
		# 		# 	p_c_item[(c_id, i_id)] = add_one
		# 		# 	p_c[c_id] = p_c.get(c_id, 0.0) + add_one
		# 		# 	total += add_one

		for i, item in enumerate(self.i2cs):
			pbar.update(i)
			i_id = item[1]
			for c in self.i2cs[item]:
				c_id = c[1]
				p_c_item[(c_id, i_id)] = 1.0
				p_c[c_id] = p_c.get(c_id, 0.0) + 1.0
				total += 1.0
		pbar.finish()

		

		count = {}
		for k, v in p_c.iteritems():
			count[v] = count.get(v, 0.0) + 1

		with open("./p_c.txt", "w") as f:
			for k, v in count.iteritems():
				output = "{} {}".format(int(k), int(v))
				print >> f, output

		count = {}
		for k, v in p_c_item.iteritems():
			count[v] = count.get(v, 0.0) + 1
		with open("./p_c_item.txt", "w") as f:
			for k, v in count.iteritems():
				output = "{} {}".format(int(k), int(v))
				print >> f, output

			# for c in self.i2cs[item]:
			# 	c_id = c[1]
			# 	p_c_item[(c_id, i_id)] = 1.0
			# 	p_c[c_id] = p_c.get(c_id, 0.0) + 1.0
			# 	total += 1.0
		# normalize
		for v in p_c_item:
			p_c_item[v] /= total
		for c in p_c:
			p_c[c] /= total

		self.p_c_item = p_c_item
		self.p_c = p_c

		# p(c, user)
		p_c_user = {}
		total = 0.0
		add_one = 0.000

		pbar = pb.ProgressBar(widgets=["p(c, user):", pb.FileTransferSpeed(unit="items"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(self.nU)).start()
		# for i, u_id in enumerate(self.nU):
		# 	pbar.update(i)
		# 	user = ('u', u_id)
		# 	if user in self.uowni:
		# 		for item in self.uowni[user]:
		# 			# for c_id in self.nC:
		# 			# 	c = ('c', c_id)
		# 			# 	if c in self.i2cs[item]:
		# 			# 		p_c_user[(c_id, u_id)] = p_c_user.get((c_id, u_id), 0.0) + 1.0
		# 			# 		total += 1.0
		# 			# if len(self.i2cs[item]) > 1:
		# 			# 	print self.i2cs[item]
		# 			for c in self.i2cs[item]:
		# 				c_id = c[1]
		# 				p_c_user[(c_id, u_id)] = p_c_user.get((c_id, u_id), 0.0) + 1.0
		# 				total += 1.0
		for i, user in enumerate(self.uowni):
			pbar.update(i)
			u_id = user[1]
			for item in self.uowni[user]:
				for c in self.i2cs[item]:
					c_id = c[1]
					p_c_user[(c_id, u_id)] = p_c_user.get((c_id, u_id), 0.0) + 1.0
					total += 1.0
			# for n in self.G.neighbors(user):
			# 	if n[0] == 'i':

			# 		# for c_id in self.nC:
			# 		# 	c = ('c', c_id)
			# 		# 	if c in self.i2cs[n]:
			# 		# 		p_c_user[(c_id, u_id)] = p_c_user.get((c_id, u_id), 0.0) + 1.0
			# 		# 		total += 1.0
			# 		# 	# else:
			# 		# 	# 	# smooth
			# 		# 	# 	p_c_user[(c_id, u_id)] = p_c_user.get((c_id, u_id), 0.0) + add_one
			# 		# 	# 	total += add_one
			# 		for c in self.i2cs[n]:
			# 			c_id = c[1]
			# 			p_c_user[(c_id, u_id)] = p_c_user.get((c_id, u_id), 0.0) + 1.0
			# 			total += 1.0
		pbar.finish()
		count = {}
		for k, v in p_c_user.iteritems():
			count[v] = count.get(v, 0.0) + 1
		with open("./p_c_user.txt", "w") as f:
			for k, v in count.iteritems():
				output = "{} {}".format(int(k), int(v))
				print >> f, output



		# normalize
		for v in p_c_user:
			p_c_user[v] /= total

		self.p_c_user = p_c_user


		self.center = {0:0, 1:0, 2:0}

		alpha = 1.0
		total = np.power(float(len(self.nU) + len(self.nI) + len(self.nC)), alpha)
		total_p = 0.0

		self.center[0] = total / np.power(len(self.nU), alpha)
		total_p += self.center[0]

		self.center[1] = total / np.power(len(self.nI), alpha)
		total_p += self.center[1]

		self.center[2] = total / np.power(len(self.nC), alpha)
		total_p += self.center[2]

		self.center[0] /= total_p
		self.center[1] /= total_p
		self.center[2] /= total_p



		# p(f | user)

		# p_f_user = {}
		
		# add_one = 0.0001
		# for u_id in self.nU:
		# 	total = 0.0
		# 	user = ('u', u_id)
		# 	for f_id in self.nU:
		# 		f = ('u', f_id)
		# 		if self.G.has_edge(user, f):
		# 			p_f_user[(u_id, f_id)] = 1.0
		# 			total += 1.0
		# 		else:
		# 			p_f_user[(u_id, f_id)] = add_one
		# 			total += add_one
		# 	# normalize
		# 	for f_id in self.nU:
		# 		p_f_user[(u_id, f_id)] /= total

		# self.p_f_user = p_f_user



		

		# friends = {}
		# for u_id in self.nU:
		# 	user = ('u', u_id)
		# 	# count = sum(1 for n in self.G.neighbors(user) if n[0] == 'u')
		# 	# count = sum(1 for n in self.G.neighbors(user))
		# 	# count = sum(1 for n in self.G.neighbors(user) if n[0] == 'u') * sum(1 for n in self.G.neighbors(user) if n[0] == 'i')
		# 	count = float(sum(1 for n in self.G.neighbors(user) if n[0] == 'i'))
		# 	friends[u_id] = count

		# # # normalize
		# # s = sum(friends.values())
		# # self.user_features = {k:(v/s) for k,v in friends.iteritems()}
		# self.user_features = friends

		# F = sorted(friends.iteritems(), key=lambda x:x[1], reverse=True)
		# self.sorted_user_features = {f[0]:(f[1], i) for i, f in enumerate(F)}


		# item_features = {}
		# for i_id in self.nI:
		# 	item = ('i', i_id)
		# 	c = self.i2c[item]
		# 	count = sum(1 for n in self.G.neighbors(item))
		# 	item_features[i_id] = count

		# self.item_features = item_features

		# for i_id, u_ids in self.Y_item.iteritems():
		# 	for u_id in u_ids:
		# 		user = ('u', u_id)
		# 		item = ('i', i_id)
		# 		y = (user, item)
		# 		self.G.add_node(y)
		# 		self.G.add_edge(y, user)
		# 		self.G.add_edge(y, item)


		# pr = nx.pagerank(self.G)
		# self.pr = pr
		# s = sum(pr[('u', u_id)] for u_id in self.nU)
		# self.pr = {('u', u_id):pr[('u', u_id)]/s for u_id in self.nU}

		print >> sys.stderr, "finish computing features, time cost: ", time.time() - s_time


	def get_probs(self, u_id, i_id):
		# return self.user_features[u_id]
		# s = 1.0
		# if ('i', i_id) in self.uowni[('u', u_id)]:
		# 	s = 100.0
		# return self.user_features[u_id] * s
		# return self.user_features[u_id] * self.item_features[i_id]
		# try:
		# 	s = nx.shortest_path_length(self.G, ('u', u_id), ('i', i_id))
		# except:
		# 	s = 10000000
		# s = 1.0 / (s+0.1)
		# return s * self.user_features[u_id]
		# user = ('u', u_id)
		# item = ('i', i_id)
		# y = (user, item)
		# return 1.0 / self.pr[y] * self.user_features[u_id]

		# score, rank = self.sorted_user_features[u_id]
		# p = score * (1.0 - 1.0 / (1 + np.exp(-1*(rank-self.tL[('i', i_id)]))))
		# p = score if rank <= self.tL[('i', i_id)] else 0.8 * score
		# return score

		user = ('u', u_id)
		item = ('i', i_id)

		p = 0.0
		# for c_id in self.nC:
		# 	try:
		# 		p += self.p_c_item[(c_id, i_id)] / self.p_c[c_id] * self.p_c_user[(c_id, u_id)]
		# 	except:
		# 		p += 0.0
		if item in self.i2cs:
			for c in self.i2cs[item]:
				c_id = c[1]
				try:
					p += self.p_c_item[(c_id, i_id)] / self.p_c[c_id] * self.p_c_user[(c_id, u_id)]
				except:
					p += 0.0
		p2 = p
		# p = 0.0
		# if user in self.u2fs:
		# 	for f in self.u2fs[user]:
		# 		f_id = f[1]
		# 		p += self.p_f_user[(u_id, f_id)]

		# p0 = p

		p = 0.0
		if user in self.u2fs:
			for f in self.u2fs[user]:
				f_id = f[1]
				if f in self.uowni:
					p += self.p_f_user[(u_id, f_id)] * len(self.uowni[f]) / len(self.nI)
					
		p0 = p

		p = 0.0

		if user in self.uowni:
			for item in self.uowni[user]:
				i_id = item[1]
				p += self.p_o_user[(i_id, u_id)] * 1.0 / len(self.nI)
			
		p1 = p 

		

		return self.center[0] * p0 + self.center[1] * p1 + self.center[2] * p2

		# return 1.0 * p0 + float(len(self.nU)) / len(self.nI) * p1 + float(len(self.nI)) / len(self.nC) * p2
		
		return 1 * p0 + 2 * p1 + 3 * p2
		pmax = max(p0, p1, p2)



		if p0 == pmax:
			self.path[0] += 1
			return p0 

		if p1 == pmax:
			self.path[1] += 1
			return p1 

		if p2 == pmax:
			self.path[2] += 1
			return p2 


		

		

	def predict(self):

		s_time = time.time()
		print >> sys.stderr, "start predicting, total Y_item: ", len(self.Y_item)

		preds = {}

		total = 0.0
		pairs_predict = []
		pbar = pb.ProgressBar(widgets=["get_probs:", pb.FileTransferSpeed(unit="items"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(self.Y_item)).start()
		for i, (i_id, u_ids) in enumerate(self.Y_item.iteritems()):
			pbar.update(i)
			for u_id in u_ids:
				p = self.get_probs(u_id, i_id)
				total += p
				pairs_predict.append((u_id, i_id, p))
		pbar.finish()
		print self.center
		print "sum p:", total

		sorted_pairs = sorted(pairs_predict, key=lambda (u_id, i_id, p): p, reverse=True)
		
		count = {0:0, 1:0}
		mid_p = sorted_pairs[len(sorted_pairs)/2]
		_, _, th = mid_p

		pbar = pb.ProgressBar(widgets=["predict:", pb.FileTransferSpeed(unit="items"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(sorted_pairs)).start()
		for i, (u_id, i_id, p) in enumerate(sorted_pairs):
			pbar.update(i)
			# if p > th and self.tL[('i', i_id)] > 0:
			# 	preds[(u_id, i_id)] = 1
			# 	self.tL[('i', i_id)] -= 1
			# 	count[1] += 1
			# else:
			# 	preds[(u_id, i_id)] = 0
			# 	count[0] += 1
			if p >= th:
				preds[(u_id, i_id)] = 1
				count[1] += 1
			else:
				preds[(u_id, i_id)] = 0
				count[0] += 1
		pbar.finish()
		print count


			# if i <= len(sorted_pairs)/2:
			# 	preds[p] = 1
			# else:
			# 	preds[p] = 0
		# c = [sorted_pairs[len(sorted_pairs)/2], sorted_pairs[len(sorted_pairs)/2+1], sorted_pairs[len(sorted_pairs)/2-1]]
		# for p in c:
		# 	print self.get_probs(p[0], p[1])


		# for i_id, u_ids in self.Y_item.iteritems():
		# 	if len(u_ids) / 2 > self.tL[('i', i_id)] :
		# 		print len(u_ids), self.tL[('i', i_id)]
		# 	sorted_u_ids = sorted(u_ids, key=lambda u_id: self.get_probs(u_id, i_id), reverse=True)
		# 	# can = sorted_u_ids[0:len(u_ids)/2]
		# 	can = sorted_u_ids[0:len(u_ids)/2]
		# 	for u_id in u_ids:
		# 		if u_id in can:
		# 			preds[(u_id, i_id)] = 1
		# 		else:
		# 			preds[(u_id, i_id)] = 0


		# pbar = pb.ProgressBar(widgets=["predict:", pb.FileTransferSpeed(unit="items"), pb.Percentage(), pb.Bar(), pb.Timer(), " ", pb.ETA()], maxval=len(self.Y_item)).start()
		# i = 0
		
		# for i_id, u_ids in self.Y_item.iteritems():

		# 	pbar.update(i)
			
		# 	sorted_u_ids = sorted(list(self.nU), key=lambda u_id: self.get_probs(u_id, i_id), reverse=True)
		# 	can = sorted_u_ids[0:len(self.nU)/3]
		# 	for u_id in u_ids:
		# 		if u_id in can:
		# 			preds[(u_id, i_id)] = 1
		# 		else:
		# 			preds[(u_id, i_id)] = 0
		# 	i += 1
		
		# pbar.finish()

		print >> sys.stderr, "finish predicting, time cost: ", time.time() - s_time
		return preds

	def load_answer(self):
		
		ans = {}
		for i_id, u_ids in self.Y_item.iteritems():
			for u_id in u_ids:
				ans[(u_id, i_id)] = 0

		with open(self.args.response, "r") as f:
			for line in f:
				ll = line.strip().split()
				u_id = int(ll[0])
				i_id = int(ll[1])
				ans[(u_id, i_id)] = 1

		self.ans = ans

	def eval(self, preds):

		s_time = time.time()
		print >> sys.stderr, "start evaluating"

		matches = 0.0
		

		for p in preds:
			if preds[p] == self.ans[p]:
				matches += 1.0
		acc = matches / len(preds)
		print >> sys.stderr, "finish evaluating, time cost: ", time.time() - s_time
		return acc











def main():

	args = arg_parse()

	model = linkpredict(args)

	preds = model.predict()

	if args.mode > 0:

		with open(args.output, "w") as f:
			for (u_id, i_id), p in preds.iteritems():
				output = "{} {} {}".format(u_id, i_id, int(p))
				print >> f, output


	if args.mode % 2 == 0:

		model.load_answer()
		
		acc = model.eval(preds)

		print >> sys.stderr, "acc: ", acc

	





if __name__ == "__main__":
	main()