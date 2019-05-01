import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

import neuron_models as nm
import lab_manager as lm
import experiments as ex
from sklearn.utils import shuffle as rshuffle

import time
import skimage

from brian2 import *

plt.style.use('ggplot')

#compile into C++ standalone
#TODO: not working yet
comp = False

# C++ standalone code to run quicker
if comp:
    set_device('cpp_standalone', build_on_run=False)

# Parallelize using OpenMP. Might not work....
# also only works with C++ standalone/comp = True
# prefs.devices.cpp_standalone.openmp_threads = 4


start_scope()
#path to the data folder
MNIST_data_path = '/Users/Jason/Desktop/Mothnet/MNIST_data/'
# MNIST_data_path = '/home/jplatt/Mothnet/MNIST_data/'

#path to folder to save data
prefix = 'total_data/'

#doesn't work if timestep > 0.05ms
defaultclock.dt = .05*ms

#number of images to run
num_examples = int(raw_input('Number of images to train: ')) #len(training)

#plot some diagnostics at the end
plot = True

#-----------------------------------------------------------
#Tunable Parameters: the parameters to change in the network


#0-10
numbers_to_inc = frozenset([0, 1])

#size of network 
N_AL = 784 #must be >= 784
N_KC = 7840
N_BL = 2 #should be the same as the number of classes

#learning rate
eta = 0.1 #fraction of total conductance per spike

"""
Amount of inhibition between AL Neurons.
Enforces WLC dynamics and needs to be scaled
with the size of the network
"""
in_AL = 0.17

'''Excititation between AL -> KCs'''
ex_ALKC = .2

#excitation kenyon cells to beta lobes
ex_KCBL = 1.5

#Lateral inhibition beta lobe
in_BLBL = 4.0


#excitation KC->GGN
ex_KCGGN = 0.01

#inhibition GGN->KC
in_GGNKC = 0.2

#probability inter-AL connections
PAL = 0.5

#AL->KCs
PALKC = 0.01

#KCs->BLs
PKCBL = 0.3

input_intensity = 0.15 #scale input
time_per_image = 80 #ms
reset_time = 15 #ms

bin_thresh = 150 #threshold for binary

taupre = 10*ms #width of STDP
taupost = taupre

#save the parameters to load in for testing
tunable_params = {'numbers_to_inc': numbers_to_inc,
                  'N_AL': N_AL,
                  'N_KC': N_KC,
                  'N_BL': N_BL, 
                  'eta': eta,
                  'in_AL': in_AL,
                  'ex_ALKC': ex_ALKC,
                  'ex_KCBL': ex_KCBL,
                  'in_BLBL': in_BLBL,
                  'ex_KCGGN': ex_KCGGN,
                  'in_GGNKC': in_GGNKC,
                  'PAL': PAL,
                  'PALKC': PALKC,
                  'PKCBL': PKCBL,
                  'input_intensity': input_intensity,
                  'time_per_image': time_per_image,
                  'reset_time': reset_time,
                  'bin_thresh': bin_thresh,
                  'taupre': taupre,
                  'taupost': taupost}

pickle.dump( tunable_params, open(prefix+"connections/tunable_params.p", "wb" ) )

#--------------------------------------------------------

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_FitzHugh_Nagumo, 
               syn_class = nm.s_FitzHughNagumo_inh,
               PAL = PAL,
               mon = ['V']
              )

#Kenyon cell parameters
kc_para = dict( N = N_KC,
                neuron_class = nm.n_lif,
                mon = []
                )

#GGN parameters
ggn_para = dict(N = 1,
				neuron_class = nm.n_li,
				mon = ['v'])

#Beta lobe neuron parameters
bl_para = dict(N = N_BL,
			   g_syn = in_BLBL,
			   neuron_class = nm.n_lif,
			   syn_class = nm.s_lif_in,
			   mon = ['v'])

#connect all the layers together
conn_para = dict(synALKC_class = nm.s_lif_ex,
                 ex_ALKC = ex_ALKC,
                 synKCGGN_class = nm.s_lif_ex,
                 ex_KCGGN = ex_ALKC,
                 synGGNKC_class = nm.s_gapjunc_in,
                 in_GGNKC = in_GGNKC,
                 synKCBL_class = nm.s_lifSTDP_ex,
                 ex_KCBL = ex_KCBL,
                 etaSTDP = eta,
                 taupreSTDP = taupre,
                 taupostSTDP = taupost,
                 PALKC = PALKC,
                 PKCBL = PKCBL)

#create the network object
net = Network()

G_AL, S_AL, trace_AL, spikes_AL = lm.get_AL(al_para, net)

G_KC, trace_KC, spikes_KC = lm.get_KCs(kc_para, net)

G_GGN, trace_GGN = lm.get_GGN(ggn_para, net)

G_BL, S_BL, trace_BL, spikes_BL = lm.get_BL(bl_para, net)

states = [G_AL, G_KC, G_GGN, G_BL]

S_ALKC, S_KCGGN, S_GGNKC, S_KCBL = lm.connect_network(conn_para, states, net)

#----------------------------------------------------------


#load MNIST data
start = time.time()
training = ex.get_labeled_data(MNIST_data_path + 'training', MNIST_data_path)
end = time.time()
print('time needed to load training set:', end - start)

n_input = training['rows']*training['cols'] #28x28=784

num_tot_images = len(training['x'])
imgs = training['x']
labels = training['y']

imgs, labels = rshuffle(imgs, labels)

#run the network
j = 0
for i in range(num_tot_images):
    if labels[i][0] in numbers_to_inc:
        print('image: ' + str(j))
        j = j+1
        
        #reset network
        G_AL.I_inj = 0.0*np.ones(N_AL)*nA      
        net.run(reset_time*ms)
        
        #right now creating binary image
        rates = np.where(imgs[i%60000,:,:] > bin_thresh, 1, 0)*input_intensity
        downsample = rates

        linear = np.ravel(downsample)

        padding = N_AL - n_input
        I = np.pad(linear, (0,padding), 'constant', constant_values=(0,0))
        G_AL.I_inj = I*nA
        
        net.run(time_per_image*ms, report = 'text')
    if j == num_examples:
        break

# run if built in C++ standalone
if comp:
    device.build(directory=prefix+'run_dir', compile=True, run=True, debug=False)

store(name = 'trained', filename = prefix+'connections/trained')


#save some of the data
np.savez(prefix+'connections/S_AL.npz', i = S_AL.i, j = S_AL.j)
np.savez(prefix+'connections/S_KCBL.npz', i = S_KCBL.i, j = S_KCBL.j)
np.savez(prefix+'connections/S_ALKC.npz', i = S_ALKC.i, j = S_ALKC.j)

np.save(prefix+'output/spikesAL_t' ,spikes_AL.t/ms)
np.save(prefix+'output/spikesAL_i',spikes_AL.i)
# np.save(prefix+'traceAL_V', trace_AL.V)

np.save(prefix+'output/spikesKC_t' ,spikes_KC.t/ms)
np.save(prefix+'output/spikesKC_i',spikes_KC.i)
# np.save(prefix+'traceKC_V', trace_KC.v)

np.save(prefix+'output/spikesBL_t' ,spikes_BL.t/ms)
np.save(prefix+'output/spikesBL_i',spikes_BL.i)
np.save(prefix+'output/traceBL_V', trace_BL.v)
np.save(prefix+'output/trace_t', trace_BL.t)

np.save(prefix+'output/traceGGN_V', trace_GGN.v)

np.save(prefix+'output/weights', S_KCBL.w_syn)

#plot diagnostics
if plot:
    fig1 = plt.figure()
    plt.plot(spikes_AL.t/ms, spikes_AL.i, '.')
    plt.title('Spikes AL')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Number')
    fig1.savefig(prefix+'images/spikes_AL_train.png', bbox_inches = 'tight')

    fig2 = plt.figure()
    plt.plot(spikes_KC.t/ms, spikes_KC.i, '.')
    plt.title('Spikes KC')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Number')
    fig2.savefig(prefix+'images/spikes_KC_train.png', bbox_inches = 'tight')

    fig3 = plt.figure()
    plt.plot(spikes_BL.t/ms, spikes_BL.i, '.')
    plt.title('Spikes BL')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Number')
    plt.ylim(-0.5, N_BL-0.5)
    fig3.savefig(prefix+'images/spikes_BL_train.png', bbox_inches = 'tight')

    fig4 = plt.figure()
    hist(S_KCBL.w_syn / ex_KCBL, 20)
    xlabel('Weight / gmax')
    plt.title('Weights BL')
    fig4.savefig(prefix+'images/weights_BL_train.png', bbox_inches = 'tight')

    fig5 = plt.figure()
    plt.plot(trace_GGN.t/ms,trace_GGN.v[0]/mV)
    plt.title('Trace GGN')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Voltage (mV)')
    fig5.savefig(prefix+'images/trace_GGN_train.png', bbox_inches = 'tight')

    fig6 = plt.figure()
    plt.plot(trace_AL.t/ms, mean(trace_AL.V, axis = 0)/mV)
    plt.title('LFP AL')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Voltage (mV)')
    fig6.savefig(prefix+'images/lfp_AL_train.png', bbox_inches = 'tight')

    fig7 = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(trace_BL.t/ms, trace_BL.v[0]/mV)
    plt.subplot(2,1,2)
    plt.plot(trace_BL.t/ms, trace_BL.v[1], 'b')
    plt.suptitle('Trace BL train')
    fig7.savefig(prefix+'images/trace_BL_train.png', bbox_inches = 'tight')

