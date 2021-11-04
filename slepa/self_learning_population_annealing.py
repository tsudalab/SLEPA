import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import pickle

from modlamp.sequences import Random
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor 
from modlamp.core import load_scale
from sklearn.preprocessing import StandardScaler

import physbo
import threading
import copy


class MyThread(threading.Thread):
	def __init__(self, func, args, name=''):
		threading.Thread.__init__(self)
		self.name = name
		self.func = func
		self.args = args
		self.result = self.func(*self.args)

	def get_result(self):
		try:
			return self.result
		except Exception:
			return None


class SelfLearningEPA:

	def __init__(self, plen, num_particle, MC_step=1, numiter=20, seed=1):
		self.am = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
		self.num_exp = 0
		self.num_particle = num_particle
		np.random.seed(seed)
		self.plen = plen
		self.numiter = numiter
		self.betas = np.linspace(0,10,numiter)
		self.num_discretize = 50
		self.es = np.zeros(self.num_discretize)
		self.scaler = StandardScaler()
		# fake MC_step
		self.MC_step = MC_step 
		# Initial state
		self.lib = Random(num_particle,lenmin=plen,lenmax=plen)
		self.lib.generate_sequences(proba='rand')
		# Descriptor
		self.tscale = load_scale('t_scale')[1]
		self.X_train = []
		self.X_train_scaled = []
		self.t_train = []
		self.allenergy = np.zeros([self.numiter, self.num_particle])
		self.allseq = [['']*self.num_particle for i in range(self.numiter)]
		self.E_max = 0
		self.E_min = 0

	def self_learning_population_annealing(self):
		# initial random state
		x_current = np.array(self.lib.sequences)
		E_current = self.calculating_value(list(x_current))
		allenergy_surr = np.zeros([self.numiter, self.num_particle])
		for biter in range(self.numiter):
			print("iteration =",biter)

			print("mean value of E =",np.mean(E_current),"+/-",np.std(E_current))

			print("number of training data =",len(self.t_train))

			if biter != 0:
				# resampling
				x_current, E_current_surr = self.resampling(self.betas[biter], self.betas[biter-1], E_current_surr, E_current_surr_prev, x_current)

				# MCMC
				threads = []

				for i in range(self.num_particle):
					t = MyThread(self.MCMC, (x_current[i], E_current_surr[i], biter, gp), self.MCMC.__name__)
					threads.append(t)
				for i in range(self.num_particle):
					threads[i].start()
				for i in range(self.num_particle):
					threads[i].join()
				for i in range(self.num_particle):
					x_current[i], E_current_surr[i] = threads[i].get_result()

				E_current = self.calculating_value(list(x_current))
				E_current_surr_prev = copy.deepcopy(E_current_surr)

			
			# add training data
			for i in range(len(x_current)):
				x_desc = self.trans_descriptor(x_current[i])
				self.X_train.append(x_desc)
				self.t_train.append(E_current[i])
			self.X_train_scaled = self.scaler.fit_transform(self.X_train)
			# surrogate model
			gp = self.train_model()

			# surrogate E
			X_test = []
			for i in range(len(x_current)):
				x_desc = self.trans_descriptor(x_current[i])
				X_test.append(x_desc)
			X_test_scaled = self.scaler.transform(X_test)
			E_current_surr = gp.get_post_fmean(np.array(self.X_train_scaled), np.array(X_test_scaled))

			if biter == 0: E_current_surr_prev = copy.deepcopy(E_current_surr)


			self.allenergy[biter] = E_current
			self.allseq[biter] = x_current
			allenergy_surr[biter] = E_current_surr_prev

		allseq_all = copy.deepcopy(self.allseq)
		allenergy_all = copy.deepcopy(self.allenergy)

		for biter in range(len(self.allenergy)):
			self.allseq[biter], self.allenergy[biter] = self.resampling(self.betas[biter], self.betas[biter], self.allenergy[biter], allenergy_surr[biter], self.allseq[biter])

		return allseq_all, allenergy_all

	def collect_observable(self, thresarray, f, sequences, energy):
		R = np.zeros([self.num_discretize,self.numiter])
		for i in range(self.num_discretize):
			for j in range(self.numiter):
				 R[i][j] = np.exp(-self.betas[j]*self.es[i] + f[j])
			R[i] = R[i]/np.sum(R[i])

		counter = np.zeros((len(thresarray), self.plen, len(self.am)))
		for thresid in range(len(thresarray)):
			thres = thresarray[thresid]
			# observable
			for biter in range(self.numiter):
				matches= [self.allseq[biter][i] for i,x in enumerate(self.allenergy[biter]) if x <= thres]
				matchenergy = [self.allenergy[biter][i] for i,x in enumerate(self.allenergy[biter]) if x <= thres]

				for ip in range(self.plen):
					for j in range(len(self.am)):
						mm = [matchenergy[i] for i,x in enumerate(matches) if x[ip] == self.am[j]]
						for k in mm:
							ik = int((k-self.E_min)/(self.E_max-self.E_min)*self.num_discretize)
							counter[thresid][ip][j] += R[ik][biter]*np.exp(self.betas[biter]*self.es[ik]-f[biter])
							#counter[ip][j] += 1
			edict = {}
			for i in range(self.numiter):
				for j in range(self.num_particle):
					edict[sequences[i][j]] = energy[i][j]

			c = 0
			for i in edict:
				if (edict[i] <= thres):
					c +=1

			print("********************")
			print("thres =", thres)
			print("number of experiments =", self.num_exp)

			print("UNIQsample =", len(edict))

			print("UNIQseq =",c)
			print("MIN =",np.min(energy))

			if (np.sum(counter[thresid]) > 0):
				counter[thresid] /= np.sum(counter[thresid])

		return counter


	def resampling(self, beta_current, beta_prev, E_current, E_current_prev, seq_current):
		prob = np.exp(-beta_current*E_current + beta_prev*E_current_prev)
		prob = prob/np.sum(prob)
		ids = np.random.choice(self.num_particle,self.num_particle, p=prob, replace=True)
		seq_current = np.array(seq_current)[ids]
		E_current = np.array(E_current)[ids]
		return seq_current, E_current 

	def MCMC(self, x_curr, e_curr, biter, gp):
		for _ in range(self.MC_step):
			x_proposal = self.mutation(x_curr)
			X_test_scaled = self.scaler.transform([self.trans_descriptor(x_proposal)])
			E_proposal = gp.get_post_fmean(np.array(self.X_train_scaled), np.array(X_test_scaled))[0]
			if np.exp(self.betas[biter]*(e_curr - E_proposal)) > np.random.rand():
				x_curr = x_proposal
				e_curr = E_proposal
		return x_curr, e_curr

	def mutation(self, x):
		x = list(x)
		select_dimension = np.random.randint(0, self.plen)
		x[select_dimension] = np.random.choice(self.am)
		return ''.join(x)

	def make_histogram(self):
		self.E_max = np.max(self.allenergy)+0.000001
		self.E_min = np.min(self.allenergy)-0.000001
		estdists = np.zeros([self.numiter,self.num_discretize])
		for biter in range(self.numiter):
			for e in self.allenergy[biter]:
				index_current = int((e-self.E_min)/(self.E_max-self.E_min)*self.num_discretize)
				estdists[biter][index_current] += 1
		return estdists

	def multi_histogram(self):
		estdists = self.make_histogram()
		estsum = np.sum(estdists,axis = 0)
		width = self.E_max-self.E_min

		for i in range(self.num_discretize):
			low = self.E_min + i*width/self.numiter
			high = self.E_min + (i+1)*width/self.numiter
			self.es[i] = (high+low)/2

		f = np.zeros(self.numiter)
		for fiter in range(10):
			estn = np.zeros(self.num_discretize)
			for i in range(self.num_discretize):
				res = 0
				for j in range(self.numiter):
					res = res + sum(estdists[j,:])*np.exp(-self.betas[j]*self.es[i]+f[j])
				estn[i] = estsum[i]/res

			for j in range(self.numiter):
				f[j] = -np.log(np.dot(estn, np.exp(-self.betas[j]*self.es)))

		estn = estn/np.sum(estn)
		return estn, f

	def trans_descriptor(self, x):
		desc = []
		for i in range(self.plen):
			desc += self.tscale[x[i]]
		return desc

	def train_model(self):
		cov = physbo.gp.cov.gauss(np.array(self.X_train_scaled).shape[1], ard = False)
		mean = physbo.gp.mean.const()
		lik = physbo.gp.lik.gauss()
		gp = physbo.gp.model(lik=lik,mean=mean,cov=cov)
		config = physbo.misc.set_config()
		gp.fit(np.array(self.X_train_scaled), np.array(self.t_train), config)
		gp.prepare(np.array(self.X_train_scaled), np.array(self.t_train))
		return gp

	def calculating_value(self, population, function='moment_tm'):
		score = []
		if not isinstance(population,list):
			population = [population]
		if function == 'moment_eisenberg':
			desc = PeptideDescriptor(population,'eisenberg')
			desc.calculate_moment()
			score = desc.descriptor
		elif function == 'moment_tm':
			desc = PeptideDescriptor(population,'TM_tend')
			desc.calculate_moment()
			score = desc.descriptor
		global num_exp
		self.num_exp += len(population)
		score = -score
		return [*score.flat]

	def hellinger(p,q):
		return np.linalg.norm(np.sqrt(p)-np.sqrt(q))/np.sqrt(2)

