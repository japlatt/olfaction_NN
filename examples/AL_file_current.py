"""
This file goes through an example for how to use a predesigned current as an input.
Here we make use of the TimedArray, and run_regularly functionality of Brian.
"""

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import sys
import time

import matplotlib.pyplot as plt
sys.path.append('../')
if sys.version_info[0] < 3:
    import cPickle as pickle

else:
    import pickle


import neuron_models as nm
import lab_manager as lm
import experiments as ex
from sklearn.utils import shuffle as rshuffle

import time
# import skimage

from brian2 import *

plt.style.use('ggplot')

#compile into C++ standalone
#works, but asking it to compile too much will freeze computer -- uses all cores to compile
# Note: The documentations states this only works for a fixed number of run statements
# and not loops. This doesn't appear to be the case, although maybe it works with
# only a few number of loops? I've gotten 4 runs to work.
# tstart variable also doesn't work because it uses a single value for it.
comp = False

# C++ standalone code to run quicker
if comp:
    set_device('cpp_standalone', debug=True, build_on_run=False)

# Parallelize using OpenMP. Might not work....
# also only works with C++ standalone/comp = True
# prefs.devices.cpp_standalone.openmp_threads = 4


start_scope()
#path to the data folder
data_path = '../data_set/designed_currents/'
data_file = 'current.dat'
#path to folder to save data
prefix = '../total_data/'

#doesn't work if timestep > 0.05ms
defaultclock.dt = .05*ms
# total run time, too lazy to change name
time_per_image = 100 # ms

'''
Loading the data file and setting up TimedArray
'''
data = np.loadtxt(data_path + data_file)
I = TimedArray(data[:int(time_per_image/0.02)]*5.0*pA, dt = 0.02*ms )

reset_time = 30 #ms


#plot some diagnostics at the end
plot = True

# A way to use a synapse to store LFP, useful when you don't want to
# trace the voltages of all AL neurons.
lfp_syn = True
#-----------------------------------------------------------
#Tunable Parameters: the parameters to change in the network

#size of network
N_AL = 1000 #must be >= 784

"""
Amount of inhibition between AL Neurons.
Enforces WLC dynamics and needs to be scaled
with the size of the network
"""
#0.1
in_AL = 0.1

#probability inter-AL connections
#0.5
PAL = 0.5


#save the parameters to load in for testing
tunable_params = {'N_AL': N_AL,
                  'in_AL': in_AL,
                  'PAL': PAL,
                  'time_per_image': time_per_image,
                  'reset_time': reset_time}

pickle.dump( tunable_params, open(prefix+"connections/tunable_params.p", "wb" ) )

#--------------------------------------------------------

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo,
               syn_class = nm.s_FitzHughNagumo_inh,
               PAL = PAL,
               mon = [],
               lfp_syn = lfp_syn
              )

t_sim = time.time()

#create the network object
net = Network()
# monlfp if using synapse to monitor
G_AL, S_AL, trace_AL, spikes_AL, G_LFP, S_LFP, trace_LFP = lm.get_AL(al_para, net)

states = [G_AL]


#----------------------------------------------------------
'''
There are a few components to using a time dependent current.

(1) The data file must be loaded, and the TimedArray function used -- include
units. dt for the file must be specified. The file must be long enough for the
run time! If the run time exceeds the time of the file, it will use the last
current value.

(2) The run_regularly function. This exists in the namespace of the NeuronGroup
you're adding it to. It can execute code strings at regular intervals. This
object needs to be assigned a name and added to the network separately if the
neuron group has previous been added.

(3) Random input - This can be combined with get_rand_I to select the neurons
that receive this current.
'''


# dt here can be smaller than default clock. This is dt of the data provided.
G_run = G_AL.run_regularly('I_inj = I(t-tstart)')
net.add(G_run)

# troubleshooting function
@network_operation(dt=5*ms)
def f2(t):
    print(G_AL.I_inj[0])
net.add(f2)



# random input
num_classes = 2
samples_per_class = 1
n_samples = int(samples_per_class*num_classes)

p_inj = 0.3
X = np.zeros((num_classes,N_AL))
for j in range(num_classes):
    X[j,:] = ex.get_rand_I(N_AL,p_inj,1)

# troubleshooting array
#test_array = np.zeros(N_AL)
#test_array[0] = 0.5
#test_array[999] = 1.0

# set tstart!
tstart = reset_time*ms

# Run random input with gradual current
for i in range(n_samples):
    # turns off all neurons
    G_AL.active_ = 0
    net.run(reset_time*ms)

    G_AL.active_ = X[i%num_classes,:]
    #G_AL.active_ = test_array
    net.run(time_per_image*ms, report='text')

    tstart = tstart + reset_time*ms + time_per_image*ms

print('Simulation time: {0} seconds'.format(time.time()-t_sim))

# run if built in C++ standalone -- this takes a LONG time
if comp:
    print("Compiling...")
    device.build(directory=prefix+'run_dir', compile=True, run=True, debug=True)
# store function not defined for C++ standalone
else:
    net.store(name = 'trained', filename = prefix+'connections/trained')


#save some of the data
np.savez(prefix+'connections/S_AL.npz', i = S_AL.i, j = S_AL.j)

np.save(prefix+'output/spikesAL_t' ,spikes_AL.t/ms)
np.save(prefix+'output/spikesAL_i',spikes_AL.i)


np.save(prefix+'input',X)
#plot diagnostics
if plot:
    fig1 = plt.figure()
    plt.plot(spikes_AL.t/ms, spikes_AL.i, '.')
    plt.title('Spikes AL')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Number')
    fig1.savefig(prefix+'images/spikes_AL_train.png', bbox_inches = 'tight')

    if lfp_syn:
        fig9 = plt.figure()
        plt.plot(trace_LFP.t/ms, trace_LFP.V[0]/mV)
        plt.title('LFP AL')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Voltage (mV)')
        fig9.savefig(prefix+'images/lfp_AL_train.png', bbox_inches = 'tight')

    else:
        fig6 = plt.figure()
        plt.plot(trace_AL.t/ms, mean(trace_AL.V, axis = 0)/mV)
        plt.title('LFP AL')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Voltage (mV)')
        fig6.savefig(prefix+'images/lfp_AL_train.png', bbox_inches = 'tight')

    #plt.show()
