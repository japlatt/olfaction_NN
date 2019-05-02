from brian2 import *

import sys
sys.path.append("..")

import neuron_models as nm
import lab_manager as lm
import experiments as ex
import analysis as anal

import matplotlib.pyplot as plt

plt.style.use('ggplot')


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

    np.save(prefix+'spikes_t_'+str(k) ,spikes_AL.t/ms)
    np.save(prefix+'spikes_i_'+str(k) ,spikes_AL.i)
    np.save(prefix+'I_'+str(k), I)
    np.save(prefix+'trace_V_'+str(k), trace_AL.V)
    np.save(prefix+'trace_t_'+str(k), trace_AL.t)

spikes_t_arr, spikes_i_arr, I_arr, trace_V_arr, trace_t_arr = anal.load_wlc_data(prefix, num_runs = 3)

PCAdata = anal.doPCA(trace_V_arr, k = 3)

fig1 = plt.figure()
plt.plot(spikes_t_arr[0]/ms, spikes_i_arr[0], '.')
plt.plot(spikes_t_arr[1]/ms, spikes_i_arr[1], '.')
plt.plot(spikes_t_arr[2]/ms, spikes_i_arr[2], '.')
plt.title('Spikes AL')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Number')
fig1.savefig('spikes_AL.pdf', bbox_inches = 'tight')

fig2 = plt.figure()
plt.plot(trace_t_arr[0]/ms, mean(trace_V_arr[0], axis = 0)/mV)
plt.title('Trace AL')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Voltage (mV)')
fig2.savefig('lfp_AL.pdf', bbox_inches = 'tight')

anal.plotPCA(PCAdata, N_AL, el = 30, az = 30, skip = 1, start = 50)

plt.show()
