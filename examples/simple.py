import slepa
import numpy as np

'''
Parameter
----------
plen: The peptide length
num_particle: The population size
thresarray: Top true value threshold
MC_step: Fake MC_step
numiter: The number of iteration
seed: Random seeds
'''
plen = 5
num_particle = 50 
thresarray = [-1.9480641200671531, -1.8178659419486884, -1.5769011390547363, -1.19584889576524] 
MC_step = 100 
numiter = 20 
seed = 2 

# default MC_step=1, numiter=20, seed=0
slepa = slepa.SelfLearningEPA(plen, num_particle, thresarray, MC_step=100, numiter=20, seed=0)

'''
return value
------------
allseq: a numiter*numparticle array of all peptide sequences searched
allenergy: a numiter*numparticle array of all peptide energy searched
estn: a histogram list of DoS (density of state)
descriptor: a len(thresarray)*plen*20(20 means the number of types of amino acids) array of the peptide descriptor distribution 
'''
slepa = SelfLearningEPA(plen, num_particle, MC_step=100, numiter=20, seed=seed)
allseq, allenergy = slepa.self_learning_population_annealing()
estn, free_energy = slepa.multi_histogram()
descriptor = slepa.collect_observable(thresarray, free_energy, allseq, allenergy)
print('best value', np.max(allenergy))

# thresid = 1
# slepa.hellinger(descriptor[thresid], true_distribution[thresid])