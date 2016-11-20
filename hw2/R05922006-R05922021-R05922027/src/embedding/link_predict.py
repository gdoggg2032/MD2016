import networkx as nx
# import matplotlib.pyplot as plt
import argparse
import numpy as np
import sys
import time
import progressbar as pb


def arg_parse():

	parser = argparse.ArgumentParser()

	parser.add_argument('--user', default='./test2/user.txt', type=str)
	parser.add_argument('--relation', default='./test2/relation.txt', type=str)
	parser.add_argument('--message', default='./test2/message.txt', type=str)
	parser.add_argument('--predict', default='./test2/pred.id', type=str)
	parser.add_argument('--response', default='./test2/response.txt', type=str)
	parser.add_argument('--pred', default='./pred.txt', type=str)
	
	parser.add_argument('--probs', default='./result.txt', type=str)
	parser.add_argument('--friend_vector', default='./friend_vector.txt', type=str)
	parser.add_argument('--item_vector', default='./item_vector.txt', type=str)
	args = parser.parse_args()

	return args



class Graph(object):
	def __init__(self, nU, nI, nC, eO, eF, eB, tL, C):
		self.nU = nU
		self.nI = nI
		self.nC = nC
		self.eO = eO
		self.eF = eF
		self.eB = eB
		self.tL = tL
		self.C = C
		

# class Model(object):
# 	def __init__(self, G, Y):
# 		self.alpha = np.random.uniform(-0.01, 0.01, 3)
# 		self.beta = np.random.uniform(-0.01, 0.01, 4)
# 		self.gamma = np.random.uniform(-0.01, 0.01, 1)
# 		self.G = G
# 		self.Y = Y
# 		print >> sys.stderr, "de_f"
# 		self.de_f = de_attribute_candidate(self.G, self.Y)
# 		print >> sys.stderr, "de_g"
# 		self.de_g = de_candidate_candidate(self.G, self.Y)
# 		# self.de_h = {k:[1.0] for k in self.Y}
# 		# # self.f = attribute_candidate(G, self, Y)
# 		# # self.g = candidate_candidate(G, self, Y)
# 		# # self.h = {}
# 		# # stage 1
# 		# self.p = {}
# 		# self.f, self.za = attribute_candidate(self.G, self, self.Y)
# 		# self.g, self.zb = candidate_candidate(self.G, self, self.Y)
		

# 		# self.zc = np.sum([np.exp(self.gamma) for y in self.Y])
# 		# for y in self.Y:
# 		# 	py = 0.
# 		# 	for yi in self.Y:

# 		# 		if yi != y:
# 		# 			f = self.f[y] / self.za
# 		# 			g = self.g[y] / self.zb
# 		# 			h = 1.0 / self.zc#self.h[y]
# 		# 			py += f * g * h
# 		# 	self.p[y] = py

# 		# self.h, self.zc = candidate_count(self.G, self, self.Y,  self.p)

# 		# self.learning_rate = 0.1

	

# 	def inference(self):
# 		G = self.G
# 		M = self
# 		Y = self.Y

# 		# two stage inference
		
# 		# # stage 1
# 		# p = {}
# 		# for y in Y:
# 		# 	py = 0.
# 		# 	for yi in Y:

# 		# 		if yi != y:
# 		# 			f = attribute_candidate(G, M, Y)
# 		# 			g = candidate_candidate(G, M, Y)
# 		# 			h = 1.0
# 		# 			py += f * g * h
# 		# 	p[y] = py

# 		# stage 2
# 		fp = {}
# 		for y in Y:
# 			py = 0.
# 			for yi in Y:

# 				if yi != y:
# 					f = self.f[yi] / self.za#attribute_candidate(G, M, Y)
# 					g = self.g[yi] / self.zb#candidate_candidate(G, M, Y)
# 					h = self.h[yi] / self.zc#candidate_count(G, M, Y, yi, p)
# 					py += f * g * h
# 			fp[y] = py

# 		self.p = fp

# 	def compute_potential(self):
# 		self.de_h = de_candidate_count(self.G, self.Y, self.p)
# 		self.f, self.za = attribute_candidate(self.G, self, self.Y)
# 		self.g, self.zb = candidate_candidate(self.G, self, self.Y)
# 		self.h, self.zc = candidate_count(self.G, self, self.Y, self.p)

# 	def cost(self, item_id):

# 		p_item_id = [(k, v) for (k, v) in self.p.iteritems() if k[1] == item_id]
# 		sorted_p_item_id = sorted(p_item_id, key=lambda (k,v):v, reverse=True)
# 		top_k = self.G.tL[item_id]
# 		y_upper = [k for (k,v) in sorted_p_item_id[:top_k]]
# 		y_lower = [k for (k,v) in sorted_p_item_id[top_k:]]

# 		# print item_id, top_k, sorted_p_item_id
# 		# print y_upper, y_lower

# 		theta = np.concatenate([self.alpha, self.beta, self.gamma])



# 		S_upper = np.sum([np.exp(np.dot(theta, (self.de_f[y]+self.de_g[y]+self.de_h[y]))) for y in y_upper])
# 		S_lower = np.sum([np.exp(np.dot(theta, (self.de_f[y]+self.de_g[y]+self.de_h[y]))) for y in y_lower])

		
# 		# print "S:", S_upper, S_lower, theta

# 		log_upper = np.log(S_upper) if S_upper else 0.0
# 		log_lower = np.log(S_lower) if S_lower else 0.0

# 		# cost = np.log(S_upper) - np.log(S_lower) 
# 		cost = log_upper - log_lower
# 		# print "cost", cost

# 		y = self.Y[0]
# 		# print np.exp(np.dot(theta, (self.de_f[y]+self.de_g[y]+self.de_h[y]))).shape
# 		# print (np.array(self.de_f[y]+self.de_g[y]+self.de_h[y]) * np.exp(np.dot(theta, (self.de_f[y]+self.de_g[y]+self.de_h[y])))).shape
# 		# print len([np.array(self.de_f[y]+self.de_g[y]+self.de_h[y]) * np.exp(np.dot(theta, (self.de_f[y]+self.de_g[y]+self.de_h[y]))) for y in y_upper])
# 		g_upper = [np.array(self.de_f[y]+self.de_g[y]+self.de_h[y]) * np.exp(np.dot(theta, (self.de_f[y]+self.de_g[y]+self.de_h[y]))) for y in y_upper]
# 		g_lower = [np.array(self.de_f[y]+self.de_g[y]+self.de_h[y]) * np.exp(np.dot(theta, (self.de_f[y]+self.de_g[y]+self.de_h[y]))) for y in y_lower]
# 		if len(g_upper) == 0: 
# 			g_upper = np.zeros(theta.shape)
# 		else:
# 			g_upper = np.sum(g_upper, axis=0)
# 		if len(g_lower) == 0:
# 			g_lower = np.zeros(theta.shape)
# 		else:
# 			g_lower = np.sum(g_lower, axis=0)

# 		# print g_upper.shape
		
# 		g_theta = g_upper / S_upper if S_upper else np.zeros(g_upper.shape)
# 		g_theta = g_theta - g_lower / S_lower if S_lower else g_theta
# 		print "g", g_theta, S_upper, S_lower

# 		g_alpha, g_beta, g_gamma = g_theta[:len(self.alpha)], g_theta[len(self.alpha):len(self.alpha)+len(self.beta)], g_theta[len(self.alpha)+len(self.beta):]

# 		return cost, g_alpha, g_beta, g_gamma

# 	def train(self, item_id):

# 		cost, g_alpha, g_beta, g_gamma = self.cost(item_id)
# 		# print "train",cost, g_alpha, g_beta, g_gamma

# 		self.alpha += self.learning_rate * g_alpha
# 		self.beta += self.learning_rate * g_beta
# 		self.gamma += self.learning_rate * g_gamma

# 		return cost
# 	def eval(self, Y_truth):
# 		Y_predict = {}
# 		for item_id in self.G.nI:
# 			p_item_id = [(k, v) for (k, v) in self.p.iteritems() if k[1] == item_id]
# 			sorted_p_item_id = sorted(p_item_id, key=lambda (k,v):v, reverse=True)
# 			top_k = self.G.tL[item_id]
# 			y_upper = [k for (k,v) in sorted_p_item_id[:top_k]]
# 			y_lower = [k for (k,v) in sorted_p_item_id[top_k:]]
# 			for y in y_upper:
# 				Y_predict[y] = 1
# 			for y in y_lower:
# 				Y_predict[y] = 0

# 		acc = 0.0
# 		total = 0.0
# 		for y, predict in Y_predict.iteritems():
# 			if predict == 1 and y in Y_truth:
# 				acc += 1.0
# 			total += 1.0
# 		acc /= total
# 		return acc








def test_graph():


	# nU = []
	# nI = []
	# nC = []

	# eO = {}
	# eF = {}
	# eB = {}

	# tL = {}


	# nU = [1, 2]
	# nI = [1, 2, 3]
	# nC = [1, 2]

	# eF[2] = [1]
	# eF[1] = [2]
	# eO[1] = [1, 2]
	# eO[2] = [3]
	# eB[1] = 1
	# eB[2] = 2
	# eB[3] = 1

	# tL[1] = 2
	# tL[2] = 1
	# tL[3] = 1

	# C = {}

	# C[1] = [1,3]
	# C[2] = [2]

	# G = Graph(nU, nI, nC, eO, eF, eB, tL, C)
	# return G

	tC = {}
	tV = {}
	tE = {}

	G = nx.Graph()

	G.add_node(('i', 1))
	# tV[('i', 1)] = 'item'
	# tC[('i', 1)] = 2
	G.add_node(('i', 2))
	# tV[('i', 2)] = 'item'
	# tC[('i', 2)] = 1
	G.add_node(('i', 3))
	# tV[('i', 3)] = 'item'
	# tC[('i', 3)] = 1
	G.add_node(('u', 1))
	# tV[('u', 1)] = 'user'
	G.add_node(('u', 2))
	# tV[('u', 2)] = 'user'
	G.add_node(('c', 1))
	# tV[('c', 1)] = 'category'
	G.add_node(('c', 2))
	# tV[('c', 2)] = 'category'

	G.add_edge(('u', 2), ('u', 1))
	# tE[(('u', 2), ('u', 1))] = 'be-friend-of'

	G.add_edge(('u', 1), ('i', 1))
	# tE[(('u', 1), ('i', 1))] = 'own'
	G.add_edge(('u', 1), ('i', 2))
	# tE[(('u', 1), ('i', 2))] = 'own'
	G.add_edge(('u', 2), ('i', 3))
	# tE[(('u', 2), ('i', 3))] = 'own'

	G.add_edge(('i', 1), ('c', 1))
	# tE[(('i', 1), ('c', 1))] = 'belong-to'
	G.add_edge(('i', 2), ('c', 2))
	# tE[(('i', 2), ('c', 2))] = 'belong-to'
	G.add_edge(('i', 3), ('c', 1))
	# tE[(('i', 3), ('c', 1))] = 'belong-to'

	# return G, tC, tV, tE
	return G



def load_network(user_file, relation_file, message_file):
	

	nU = set()
	nI = set()
	nC = set()

	eO = {}
	eF = {}
	eB = {}

	tL = {}


	C = {}


	G = nx.Graph()

	with open(user_file, "r") as f:
		for line in f:
			u_id = int(line)
			nU.add(u_id)
			G.add_node(('u', u_id))
			# tV[('u', u_id)] = 'user'

	with open(relation_file, "r") as f:
		for line in f:
			ll = line.strip().split()
			u_id1 = int(ll[0])
			u_id2 = int(ll[1])
			if u_id1 not in eF:
				eF[u_id1] = [u_id2]
			else:
				eF[u_id1].append(u_id2)

			# both directions
			if u_id2 not in eF:
				eF[u_id2] = [u_id1]
			else:
				eF[u_id2].append(u_id1)
			G.add_edge(('u', u_id1), ('u', u_id2))
			G.add_edge(('u', u_id2), ('u', u_id1))
			# tE[(('u', u_id1), ('u', u_id2))] = 'be-friend-of'

	with open(message_file, "r") as f:
		for line in f:
			ll = line.strip().split()
			owner_uid = int(ll[0])
			item_id = int(ll[1])
			c_id = int(ll[2])
			link_count = int(ll[3])

			nI.add(('i',item_id))
			
			tL[('i', item_id)] = link_count

			nC.add(c_id)

			if owner_uid not in eO:
				eO[owner_uid] = [item_id]
			else:
				eO[owner_uid].append(item_id)

			eB[item_id] = c_id

			if c_id not in C:
				C[c_id] = [item_id]
			else:
				C[c_id].append(item_id)



			G.add_node(('i', item_id))
			# tV[('i', item_id)] = 'item'
			# tC[('i', item_id)] = link_count

			G.add_node(('c', c_id))
			# tV[('c', c_id)] = 'category'
			
			G.add_node(('u', owner_uid))

			G.add_edge(('u', owner_uid), ('i', item_id))
			# G.add_edge(('u', item_id), ('i', owner_uid))
			# tE[(('u', owner_uid), ('i', item_id))] = 'own'

			G.add_edge(('i', item_id), ('c', c_id))
			# G.add_edge(('c', c_id), ('i', item_id))
			# tE[(('i', item_id), ('c', c_id))] = 'belong-to'

			# return G, tC, tV, tE
	# G = Graph(nU, nI, nC, eO, eF, eB, tL, C)

	return G, nI, tL

def test_candidate():
	# Y = []
	# Y.append((1, 1))
	# Y.append((2, 1))
	# Y.append((1, 3))
	# Y.append((2, 3))
	# Y.append((1, 2))
	# Y.append((2, 2))
	# return Y
	Y = []
	Y.append((('u', 1), ('i',1)))
	Y.append((('u', 2), ('i',1)))
	Y.append((('u', 1), ('i',3)))
	Y.append((('u', 2), ('i',3)))
	Y.append((('u', 1), ('i',2)))
	Y.append((('u', 2), ('i',2)))
	return Y


def load_candidate(predict_file):

	Y = []
	Y_item = {}

	Y_u = set()

	with open(predict_file, "r") as f:
		for line in f:
			ll = line.strip().split()
			u_id = int(ll[0])
			Y_u.add(u_id)
			item_id = int(ll[1])
			Y.append((('u',u_id), ('i',item_id)))
			if ('i', item_id) not in Y_item:
				Y_item[('i', item_id)] = [('u', u_id)]
			else:
				Y_item[('i', item_id)].append(('u', u_id))


	return Y, Y_item, Y_u


def resource_allocation_index(G, Y):

	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			if G.has_edge(u, v):
				p += 0.0
			# a = G.neighbors(u)
			# b = G.neighbors(v)
			# n = list(set(a).intersection(b))
			for n in nx.common_neighbors(G, u, v):
				p += 1.0 / G.degree(n)
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def shortest_path_feature(G, Y):
	preds = {}
	for i, (u, v) in enumerate(Y):
		p = 0.0
		if G.has_node(u) and G.has_node(v):
				p += 1.0 / float(nx.shortest_path_length(G, u, v))
			
		preds[(u,v)] = p
	return preds
def jaccard_coefficient(G, Y):
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			if G.has_edge(u, v):
				p += 1.0
			a = G.neighbors(u)
			b = G.neighbors(v)
			n_i = len(list(set(a).intersection(b)))
			n_u = len(list(set(a).union(b)))
			p = n_i / float(n_u)
			# for n in nx.common_neighbors(G, u, v):
			# 	p += 1.0 / G.degree(n)
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def adamic_adar_index(G, Y):

	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			if G.has_edge(u, v):
				p += 0.0
			# a = G.neighbors(u)
			# b = G.neighbors(v)
			# n = list(set(a).intersection(b))
			for n in nx.common_neighbors(G, u, v):
				p += 1.0 / np.log(G.degree(n))
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def preferential_attachment(G, Y):

	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			if G.has_edge(u, v):
				p += 0.0
			a = G.neighbors(u)
			b = G.neighbors(v)
			p = float(len(a) * len(b))
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def cn_soundarajan_hopcroft(G, Y, label_type):
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			if G.has_edge(u, v):
				p += 0.0

			# a = G.neighbors(u)
			# b = G.neighbors(v)
			# n = list(set(a).intersection(b))
			for n in nx.common_neighbors(G, u, v):
				p += 1.0

				if n[0] == label_type:
					p += 1.0 / G.degree(n)
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def ra_index_soundarajan_hopcroft(G, Y, label_type):
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			if G.has_edge(u, v):
				p += 0.0

			# a = G.neighbors(u)
			# b = G.neighbors(v)
			# n = list(set(a).intersection(b))
			total = 0.0
			for n in nx.common_neighbors(G, u, v):
				
				total += 1.0

				if n[0] == label_type:
					p += 1.0 / G.degree(n)
			p /= total if total != 0 else 1
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def number_of_user_own(G, Y):
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			n = G.neighbors(u)
			for nn in n:
				t = nn[0]
				if t == 'i':
					p += 1.0
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def number_of_user_friend(G, Y):
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			n = G.neighbors(u)
			for nn in n:
				t = nn[0]
				if t == 'u':
					p += 1.0
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def number_of_item_belong(G, Y):
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			n = G.neighbors(v)
			for nn in n:
				t = nn[0]
				if t == 'c':
					p += 1.0
		# preds.append((u, v, p))
		preds[(u,v)] = p
	return preds

def all_one(G, Y):
	preds = {}
	for i, (u, v) in enumerate(Y):
		preds[(u,v)] = 1
	return preds

def all_zero(G, Y):
	preds = {}
	for i, (u, v) in enumerate(Y):
		preds[(u,v)] = 0
	return preds


def number_of_item_category(G, Y):
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			n = G.neighbors(v)
			for nn in n:
				t = nn[0]
				if t == 'c':
					p += G.degree(nn)
		# preds.append((u, v, p))
		# print u, v, p
		preds[(u,v)] = p
	return preds

def pagerank_user(G, Y):
	pr = nx.pagerank(G)
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			p += pr[u]
		# preds.append((u, v, p))
		# print u, v, p
		preds[(u,v)] = p
	return preds

def pagerank_item(G, Y):
	pr = nx.pagerank(G)
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			p += pr[v]
		# preds.append((u, v, p))
		# print u, v, p
		preds[(u,v)] = p
	return preds

def pagerank_user_item(G, Y):
	pr = nx.pagerank(G)
	preds = {}
	for i, (u, v) in enumerate(Y):
		# print i, len(Y)
		p = 0.0
		if G.has_node(u) and G.has_node(v):
			p += pr[v] + pr[u]
		# preds.append((u, v, p))
		# print u, v, p
		preds[(u,v)] = p
	return preds

# def hits_user(G, Y):
# 	hits = nx.hits(G)
# 	preds = {}
# 	for i, (u, v) in enumerate(Y):
# 		# print i, len(Y)
# 		p = 0.0
# 		if G.has_node(u) and G.has_node(v):
# 			p += hits[u]
# 		# preds.append((u, v, p))
# 		# print u, v, p
# 		preds[(u,v)] = p
# 	return preds

def load_answer(answer_file, Y_item):
	ans = {}
	for i_id, u_ids in Y_item.iteritems():
		for u_id in u_ids:
			ans[(u_id, i_id)] = 0

	with open(answer_file, "r") as f:
		for line in f:
			ll = line.strip().split()
			u_id = int(ll[0])
			item_id = int(ll[1])
			ans[(('u',u_id), ('i',item_id))] = 1

	return ans


def evaluate(ans, preds):
	matches = 0.0
	for p in preds:
		if preds[p] == ans[p]:
			matches += 1.0
	acc = matches / len(preds)
	return acc


def prediction(tL, Y_item, i_vector, f_vector, top):
	preds = {}
	pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(Y_item)).start()
	index = 0

	active_user = set()
	for item, users in Y_item.iteritems():
		for user in users:
			active_user.add(int(user[1]))
	active_user = list(active_user)
	f_matrix = []
	for uid in active_user:
		f_matrix.append(f_vector[uid])
	f_matrix = np.array(f_matrix)

	for item, users in Y_item.iteritems():
		tmp = {}
		iid = int(item[1])
		i_v = i_vector[iid][None,:]
		# for uid, f_v in f_vector.iteritems():
		# 	tmp[uid] = float(np.dot(f_v, i_v))
		curr_value = np.dot(f_matrix, i_v.T)

		for i, uid in enumerate(active_user):
			# f_v = f_vector[uid]
			tmp[uid] = float(curr_value[i])

		can_size = len(active_user)/top	
		cur_size = 0
		# can_list = []
		threshold = 0.
		for k, v in sorted(tmp.iteritems(), key=lambda (k,v):v, reverse=True):
			# can_list.append(k)
			cur_size += 1
			if cur_size >= can_size:
				threshold = v
				break

		user_tmp = {}
		for user in users:
			uid = user[1]
			user_tmp[uid] = tmp[uid]

		for uid, v in sorted(user_tmp.iteritems(), key=lambda (k,v):v, reverse=True):
			if v >= threshold and tL[item] > 0:
				preds[(("u", uid), item)] = 1
				tL[item] -= 1
			else:
				preds[(("u", uid), item)] = 0
		# for user in users:
		# 	uid = user[1]
		# 	if tmp[uid] >= threshold:
		# 		preds[(user, item)] = 1
		# 		# print "haha"
		# 	else:
		# 		preds[(user, item)] = 0
		index += 1
		pbar.update(index)
	pbar.finish()
		# sorted_users = sorted(users, key=lambda u: probs[(u, item)], reverse=True)
		# users_upper = sorted_users[0:tL[item]]
		# users_lower = sorted_users[tL[item]:]
		# for user in users_upper:
		# 	preds[(user, item)] = 1
		# for user in users_lower:
		# 	preds[(user, item)] = 0
	return preds

def prediction1(G, tL, Y_item, item_avg, item_stddev, i_vector, f_vector, friends):
	preds = {}
	can = friends[0:len(friends)/4]
	pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(Y_item)).start()
	cur = 0

	sum_v = 0	
	for i, i_v in i_vector.iteritems():
		sum_v += i_v
	sum_v /= float(len(i_vector))


	active_user = set()
	for item, users in Y_item.iteritems():
		for user in users:
			active_user.add(int(user[1]))
	active_user = list(active_user)

	friends = {}
	total = 0.
	for uid in active_user:
		user = ('u', uid)
		friends[uid] = sum(1 for n in G.neighbors(user) if n[0] == 'u')
		total += friends[uid]
	for k, v in friends.iteritems():
		friends[k] /= total


	pair_pred = []
	for item, users in Y_item.iteritems():
		iid = int(item[1])
		i_v = i_vector[iid]
		for user in users:
			uid = int(user[1])
			f_v = f_vector[uid]
			value = 1./(1.+np.exp(-np.dot(f_v, sum_v))) * friends[uid]
			pair_pred.append((user, item, value))
		cur += 1
		pbar.update(cur)
	pbar.finish()

	pair_pred = sorted(pair_pred, key=lambda i:i[2], reverse=True)
	_, _, th = pair_pred[len(pair_pred)/3]
	for u,i,p in pair_pred:
		if p > th and tL[('i', i[1])] > 0:
			preds[(u,i)] = 1
			tL[('i', i[1])] -= 1
		else:
			preds[(u,i)] = 0
			
	return preds

def prediction2(tL, Y_item, user_list, item_list):
	user_len = len(user_list)
	preds = {}
	# pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(Y_item)).start()
	i = 0
	pair_pred = []
	for item, users in Y_item.iteritems():
		iid = int(item[1])
		for user in users:
			uid = int(user[1])
			value = item_list[iid].vr.get(uid, 0.)
			pair_pred.append((user, item, value))

	pair_pred = sorted(pair_pred, key=lambda i:i[2], reverse=True)
	print pair_pred
	_, _, th = pair_pred[len(pair_pred)/2]
	for u,i,p in pair_pred:
		if p > th:
			preds[(u,i)] = 1
		else:
			preds[(u,i)] = 0
		# iid = item[1]
		# tmp = []
		# for k, v in sorted(item_list[iid].vr.iteritems(), key=lambda (k,v):v, reverse=True):
		# 	tmp.append(int(k))
		# can = None
		# if len(tmp) > user_len/4:
		# 	can = tmp[0:user_len/4]
		# else:
		# 	can = tmp
		# for user in users:
		# 	uid = int(user[1])
		# 	if uid in can:
		# 		preds[(user, item)] = 1
		# 	else:
		# 		preds[(user, item)] = 0
	# 	i += 1
	# 	pbar.update(i)
	# pbar.finish()
			# if user in can:
			# 	preds[(user, item)] = 1
			# 	# print "haha"
			# else:
			# 	preds[(user, item)] = 0
		# sorted_users = sorted(users, key=lambda u: probs[(u, item)], reverse=True)
		# users_upper = sorted_users[0:tL[item]]
		# users_lower = sorted_users[tL[item]:]
		# for user in users_upper:
		# 	preds[(user, item)] = 1
		# for user in users_lower:
		# 	preds[(user, item)] = 0
	return preds

def combine(probs1, probs2, alpha):
	sum_1 = sum(probs1.values())
	sum_2 = sum(probs2.values())
	probs = {}
	for y in probs1:
		p = alpha * probs1[y] /  sum_1 + (1.0 - alpha) * probs2[y] / sum_2
		probs[y] = p 
	return probs

def get_vectors(args):

	i_vector = {}
	f_vector = {}

	with open(args.friend_vector, 'r') as f:
		for line in f.readlines():
			line = line.strip().split()
			uid = int(line[0])
			vector = np.array(map(float, line[1:]))
			f_vector[uid] = vector

	with open(args.item_vector, 'r') as f:
		for line in f.readlines():
			line = line.strip().split()
			iid = int(line[0])
			vector = np.array(map(float, line[1:]))
			i_vector[iid] = vector

	return i_vector, f_vector

def get_average(args):
	item_avg = {}
	item_stddev = {}
	i_vector, f_vector = get_vectors(args)

	active_iid = set()
	with open(args.predict, "r") as f:
		for line in f:
			ll = line.strip().split()
			active_iid.add(int(ll[1]))

	sum_f_vector = 0.
	for u_id, f_v in f_vector.iteritems():
		sum_f_vector += f_v
	avg_f_vector = sum_f_vector/float(len(f_vector))

	stddev_vector = []
	for u_id, f_v in f_vector.iteritems():
		stddev_vector.append(f_v)
	stddev_vector = np.array(stddev_vector)
	stddev_vector = np.std(stddev_vector, axis=0)

	for i_id in active_iid:
		i_v = i_vector[i_id]
		item_avg[('i',i_id)] = float(np.dot(avg_f_vector, i_v))
		item_stddev[('i',i_id)] = float(np.dot(stddev_vector, i_v))

	return item_avg, item_stddev, i_vector, f_vector

def load_probs(probs_file, args):

	active_iid = set()
	with open(args.predict, "r") as f:
		for line in f:
			ll = line.strip().split()
			active_iid.add(int(ll[1]))
		
	probs = {}
	i_vector, f_vector = get_vectors(args)
	print len(active_iid), len(f_vector)
	print len(f_vector)*len(active_iid)
	"""
	print >> sys.stderr, 'start calculating'
	pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(f_vector)).start()
	i = 0
	for u_id, f_v in f_vector.iteritems():
		for i_id in active_iid:
			i_v = i_vector[i_id]
			probs[(('u',u_id),('i',i_id))] = float(np.dot(f_v, i_v))
		i += 1
		pbar.update(i)
	pbar.finish()

	print >> sys.stderr, 'start sorting'
	sorted_probs = {}
	pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(probs)).start()
	i = 0
	for k, v in sorted(probs.iteritems(), key=lambda (k,v):v, reverse=True):
		try:
			sorted_probs[k[1]].append(k[0])
		except:
			sorted_probs[k[1]] = []
			sorted_probs[k[1]].append(k[0])
		i += 1
		pbar.update(i)
	pbar.finish()

	probs = None
	# with open(probs_file, "r") as f:
	# 	for line in f:
	# 		ll = line.strip().split()
	# 		u_id = int(ll[0])
	# 		i_id = int(ll[1])
	# 		p = float(ll[2])
	# 		probs[(('u',u_id),('i',i_id))] = p
	"""
	return i_vector, f_vector #sorted_probs, len(f_vector)

class Item(object):
	def __init__(self, iid):
		self.cat = set()
		self.links = 0
		self.vr = {}
		self.id = iid
	def set_id(self, iid):
		self.id = iid
	def set_vr(self, uid, value):
		self.vr[uid] = self.vr.get(uid, 0.) + value
	def add_cat(self, c):
		self.cat.add(c)
	def set_links(self, links):
		self.links = links

class User(object):
	def __init__(self, uid):
		self.id = uid
		self.oitems = []
		self.friends = set()
		self.uv = {}
		self.own_num = 0
		self.prod_vr = 0
	def cal_prob_vr(self):
		value = 1. if self.own_num > 0 else 0.
		for i in self.oitems:
			value *= (1. - i.vr[self.id])
		self.prod_vr = value

	def add_friend(self, user):
		self.friends.add(user)
	def add_item(self, item):
		self.oitems.append(item)
		self.own_num += 1
	def set_uv(self, uid, value):
		self.uv[uid] = value

def get_max_id(args):

	maxiid = 0
	maxcid = 0
	with open(args.message, 'r') as f:
		for line in f.readlines():
			info = line.strip().split()
			try:
				maxiid = int(info[1]) if int(info[1]) > maxiid else maxiid
				maxcid = int(info[2]) if int(info[2]) > maxcid else maxcid
			except:
				continue
	return maxiid + 1, maxcid + 1

def get_uid_v(args):

	uid_v = []
	with open(args.friend_vector, 'r') as f:
		for line in f.readlines():
			vector = map(float, line.strip().split()[1:])
			uid_v.append(vector)

	return np.array(uid_v).astype('float32')

def get_AT_prob_class(args):

	num_item, _ = get_max_id(args)

	uid_v = get_uid_v(args)

	num_user = len(uid_v)

	user_list = []
	item_list = []

	for i in range(num_user):
		user_list.append(User(i))
	for i in range(num_item):
		item_list.append(Item(i))

	print 'item num: {}, user num: {}'.format(num_item, num_user)

	dat = []
	for line in open(args.message, 'r'):
		p = map(int, line.strip().split())
		if len(p) > 0:
			dat.append(p)
	train_dat = np.array(dat)

	# set up all attribute
	for info in train_dat:
		uid = info[0]
		iid = info[1]
		links = info[2]
		cat = info[3]

		user_list[uid].add_item(item_list[iid])
		item_list[iid].set_links(links)
		item_list[iid].add_cat(cat)

	for line in open(args.relation, 'r'):
		p = map(int, line.strip().split())
		if len(p) > 0:
			uid1 = p[0]
			uid2 = p[1]

			user_list[uid1].add_friend(user_list[uid2])
			user_list[uid2].add_friend(user_list[uid1])

	# calculate Pvr and Puv
	for u in user_list:
		if u.own_num > 0:
			for oitem in u.oitems:
				i_v = oitem.links
				sum_v = 0.
				for _oitem in u.oitems:
					if len(set.intersection(oitem.cat, _oitem.cat)) > 0: 
						sum_v += _oitem.links
				pvr = 0. if sum_v == 0. else float(i_v)/float(sum_v)
				oitem.set_vr(u.id, pvr)
			u.cal_prob_vr()
			puv = 1. - u.prod_vr
			for f in u.friends:
				u.set_uv(f.id, puv)
	# calculate Pur
	for u in user_list:
		for f in u.friends:
			for i in u.oitems:
				pur = u.uv[f.id] * i.vr[u.id]
				i.set_vr(f.id, pur)

	# train_prob = []
	# for i in item_list:
	# 	if len(i.vr) > 0:
	# 		for uid, prob in i.vr.iteritems():
	# 			train_prob.append([uid, i.id, prob])
	# train_prob = np.array(train_prob)

	return user_list, item_list

def user_friend(G, nU):
	friends = {}
	for u_id in nU:
		p = 0.0
		user = ('u', u_id)
		if G.has_node(user):
			n = G.neighbors(user)
			for nn in n:
				t = nn[0]
				if t == 'u':
					p += 1.0
		friends[u_id] = p
	friends_l = sorted(list(nU), key=lambda u: friends[u], reverse=True)
	return friends_l


if __name__ == "__main__":

	start = time.time()
	args = arg_parse()

	# G, tC, tV, tE = load_network(args.user, args.relation, args.message)
	print >> sys.stderr, "loading"
	s_time = time.time()
	G, nI, tL = load_network(args.user, args.relation, args.message)
	# G = test_graph()
	
	Y, Y_item, Y_u = load_candidate(args.predict)
	# Y = test_candidate()
	################################################
	# ans = load_answer(args.response, Y_item)
	print "time cost:", time.time() - s_time
	# print "resource_allocation_index"
	# preds = nx.resource_allocation_index(G, Y)
	# for u, v, p in preds:
	# 	print "{} -> {}: {}".format(u, v, p)

	# print "my_resource_allocation_index"
	# s_time = time.time()
	# probs = resource_allocation_index(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.541412570762


	# print "shortest_path_feature"
	# s_time = time.time()
	# probs = shortest_path_feature(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.608542481596

	# print "jaccard_coefficient"
	# s_time = time.time()
	# probs = jaccard_coefficient(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.541412570762

	# print "adamic_adar_index"
	# s_time = time.time()
	# probs = adamic_adar_index(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.541412570762

	# print "preferential_attachment"
	# s_time = time.time()
	# probs = preferential_attachment(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.56213707816

	# print "cn_soundarajan_hopcroft"
	# s_time = time.time()
	# probs = cn_soundarajan_hopcroft(G, Y, 'u')
	# print "time cost:", time.time() - s_time
	# 0.541412570762

	
	# print "ra_index_soundarajan_hopcroft"
	# s_time = time.time()
	# probs = ra_index_soundarajan_hopcroft(G, Y, 'u')
	# print "time cost:", time.time() - s_time
	# 0.541412570762

	# print "number_of_user_own"
	# s_time = time.time()
	# probs = number_of_user_own(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.573750577031

	# print "number_of_user_friend"
	# s_time = time.time()
	# probs = number_of_user_friend(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.555091231565

	# print "number_of_item_belong"
	# s_time = time.time()
	# probs = number_of_item_belong(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.52605748439


	# print "all_one"
	# s_time = time.time()
	# probs = all_one(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.52605748439


	# print "all_zero"
	# s_time = time.time()
	# probs = all_zero(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.52605748439

	# print "number_of_item_category"
	# s_time = time.time()
	# probs = number_of_item_category(G, Y)
	# print "time cost:", time.time() - s_time
	# 0.52605748439

	# print "pagerank_user"
	# s_time = time.time()
	# probs = pagerank_user(G, Y)
	# print "time cost", time.time() - s_time
	# 0.561991302024

	# print "pagerank_item"
	# s_time = time.time()
	# probs = pagerank_item(G, Y)
	# print "time cost", time.time() - s_time


	# print "pagerank_user_item"
	# s_time = time.time()
	# probs = pagerank_user_item(G, Y)
	# print "time cost", time.time() - s_time
	# 0.52605748439

	# print "hits_user"
	# s_time = time.time()
	# probs = hits_user(G, Y)
	# print "time cost", time.time() - s_time
	
	print "self probs"
	s_time = time.time()
	friends = user_friend(G, Y_u)
	item_avg, item_stddev, i_vector, f_vector = get_average(args)
	print "time cost", time.time() - s_time 
	preds = prediction1(G, tL, Y_item, item_avg, item_stddev, i_vector, f_vector, friends)
	# print "evaluate"
	# print evaluate(ans, preds)

	fo = open(args.pred, 'w')
	with open(args.predict, 'r') as f:
		for line in f.readlines():
			pairl = line.strip().split()
			ans = preds.get((('u',int(pairl[0])), ('i',int(pairl[1]))), 'None')
			out = str(pairl[0])+' '+str(pairl[1])+' '+str(ans)+'\n'
			fo.write(out)

	# i_vector, f_vector = load_probs(args.probs, args)
	# print "prediction"
	# preds = prediction(tL, Y_item, i_vector, f_vector, 4)
	# print "evaluate"
	# print evaluate(ans, preds)
	

	
	# user_list, item_list = get_AT_prob_class(args)
	# preds = prediction2(tL, Y_item, user_list, item_list)
	# print "evaluate"
	# print evaluate(ans, preds)
	

	# print "prediction"
	# preds = prediction(tL, Y_item, sorted_probs, total_user, 2)
	# print "evaluate"
	# print evaluate(ans, preds)



	# print "expers"
	# for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
	# 	print "combine: ", alpha
	# 	probs = combine(probs1, probs2, alpha)

	# 	print "prediction"
	# 	preds = prediction(G, nI, tL, Y, Y_item, probs)



	# 	print "evaluate"
	# 	print evaluate(ans, preds)
	# for u, v, p in preds:
	# 	if p != 0:
	# 		print "{} -> {}: {}".format(u, v, p)

	# print "shortest_path_feature"
	# preds = shortest_path_feature(G, Y)
	# for u, v, p in preds:
	# 	if p != 0:
	# 		print "{} -> {}: {}".format(u, v, p)



	# print "jaccard_coefficient"
	# preds = nx.jaccard_coefficient(G, Y)
	# for u, v, p in preds:
	# 	print "{} -> {}: {}".format(u, v, p)


	# print "adamic_adar_index"
	# preds = nx.adamic_adar_index(G, Y)
	# for u, v, p in preds:
	# 	print "{} -> {}: {}".format(u, v, p)






	# # Y_truth = load_truth(args.response)

	# print >> sys.stderr, "done loading", time.time() - start
	# print >> sys.stderr, "nI: ", len(G.nI)
	# print >> sys.stderr, "nU: ", len(G.nU)
	# print >> sys.stderr, "nC: ", len(G.nC)
	# print >> sys.stderr, "tL: ", len(G.tL)
	# print >> sys.stderr, "nY: ", len(Y)
	# # print >> sys.stderr, "nYt: ", len(Y_truth) 
	# print >> sys.stderr, "building model"
	# M = Model(G, Y)
	# print >> sys.stderr, "done building", time.time() - start

	# print >> sys.stderr, "dump features de_f, de_g, total_dim: ", 7

	# print >> sys.stderr, "in format: u_id i_id de_f1 de_f ... de_g1 de_g2 ..."

	# dump_features(args.features, Y, M.de_f, M.de_g)
	# dump_count(args.count, G)

	# epochs = 100

	# for epoch in range(epochs):

	# 	M.inference()
	# 	M.compute_potential()

	# 	cost = 0.
	# 	for i, item in enumerate(M.G.nI):
	# 		if (i+1) % 10000 == 0:
	# 			print >> sys.stderr, i, "/", len(M.G.nI)
	# 		cost += M.train(item) 
	# 	acc = M.eval(Y_truth)
	# 	print >> sys.stderr, "Epoch", epoch+1, "cost=", cost, \
	# 	"acc=", acc


	



















