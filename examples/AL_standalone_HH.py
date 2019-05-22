from brian2 import *

import sys
sys.path.append("..")

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal

import matplotlib.pyplot as plt

# plt.style.use('ggplot')


defaultclock.dt = .02*ms

N_AL = 1000
in_AL = .25*msiemens/cm**2
PAL = 0.5

run_time = 150*ms

prefix = 'data/'

#Antennal Lobe parameters
al_para = dict(N = N_AL,
               g_syn = in_AL,
               neuron_class = nm.n_HH, 
               syn_class = nm.s_GABA_inh,
               PAL = PAL,
               mon = ['V']
              )

#create the network object
net = Network()

G_AL, S_AL, trace_AL, spikes_AL = lm.get_AL(al_para, net)

net.store()

for k in range(3):
    net.restore()

    I = (ex.get_rand_I(N_AL, p = 0.33, I_inp = 15)+15)*uA/cm**2
    G_AL.I_inj = I

    net.run(run_time, report = 'text')

    np.save(prefix+'spikes_t_'+str(k) ,spikes_AL.t)
    np.save(prefix+'spikes_i_'+str(k) ,spikes_AL.i)
    np.save(prefix+'I_'+str(k), I)
    np.save(prefix+'trace_V_'+str(k), trace_AL.V)
    np.save(prefix+'trace_t_'+str(k), trace_AL.t)

spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr = anal.load_wlc_data(prefix, num_runs = 3)

col = ['#f10c45','#069af3','#02590f','#ab33ff','#ff8c00','#ffd700']

fig1 = plt.figure()
plt.plot(spikes_t_arr[0]/ms, spikes_i_arr[0], '.', color = col[0])
plt.plot(spikes_t_arr[1]/ms, spikes_i_arr[1], '.', color = col[1])
plt.plot(spikes_t_arr[2]/ms, spikes_i_arr[2], '.', color = col[2])
plt.title('Hodgkin Huxley AL', fontsize = 22)
plt.xlabel('Time (ms)', fontsize = 16)
plt.ylabel('Neuron Number', fontsize = 16)
fig1.savefig('spikes_AL.pdf', bbox_inches = 'tight')

fig2 = plt.figure(figsize = (8, 6))
plt.plot(trace_t_arr[0]/ms, mean(trace_V_arr[0], axis = 0)/mV, linewidth = 2)
plt.title('LFP AL ' +str(N_AL)+' Neuron HH', fontsize = 22)
plt.xlabel('Time (ms)', fontsize = 16)
plt.ylabel('LFP (mV)', fontsize = 16)
plt.ylim(-50, -80)
fig2.savefig('lfp_AL.pdf', bbox_inches = 'tight')

PCAdata = anal.doPCA(trace_V_arr, k = 3, n = 3)
anal.plotPCA(PCAdata, N_AL, title = 'PCA HH ' + str(N_AL) + ' Neuron', el = 0, az = 0, skip = 2, start = 100)


anal.getMIM(trace_V_arr) #takes a long time
MIM = np.load('MIM.npy')
data = np.hstack(trace_V_arr).T
length = len(trace_V_arr[0][0])

InCAdata = anal.doInCA(MIM, data, length, skip = 2, k = 3)
anal.plotInCA(InCAdata, N_AL, start = 200)
