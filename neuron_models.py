from brian2 import *



#----------------------------------------------------------
#NEURONS

'''
naming convention:  all neurons start with n followed by the name
of the neuron

Each neuron class needs to have the following functions defined

def eqs(self):
    return string of equations in Brian format

def namespace(self):
    return dictionary of variables defined in eqs/threshold...etc

def threshold(self):
    return string 'V > thresh' with voltage that neuron spikes or None

def refractory(self):
    return string or number defining length of time that voltage above
    threshold does not register as spike.  Note this does not change
    behavior of membrane potential unless specified.

def reset(self):
    return voltage that neuron returns to after reaching threshold

def method(self):
    return string 'rk4' or other method of integration

def state_mon(self):
    return array of variables in string format e.g ['V'] that tell
    the simulation to record the value of that variable

def init_cond(self):
    return dict(v = self.vr) dictionary of variables and their initial
    conditions
'''
class n_FitzHugh_Nagumo:

    #global variables for all instances of the class
    nu = -1.5*mV
    a = 0.7
    b = 0.8
    t1 = 0.08*ms
    t2 = 3.1*ms

    thresh = 0*mV
    refrac = -0.5*mV

    states_to_mon = ['V']


    def __init__(self, mon):
        self.states_to_mon = mon
        return

    def eqs(self):
        # active is a variable that can be used to assign which neurons are active. For time dependent stimuli,
        # you can assign I_inj as the function ranging from [-1,1], use a network operator to update, and
        # use active to assign max amplitude and active neurons.
        eqns_AL = '''
                    I_inj : amp
                    I_syn : 1
                    dV/dt = (V-(V**3)/(3*mV**2) - w - z*(V - nu) + 0.35*mV + I_inj*active*Mohm)/t1 : volt
                    dw/dt = (V - b*w + a*mV)/ms : volt
                    dz/dt = (I_syn - z)/t2 : 1
                    active : 1
                    '''
        return eqns_AL

    def namespace(self):
        namespace = {'nu': self.nu,
                     'a' : self.a,
                     'b' : self.b,
                     't1': self.t1,
                     't2': self.t2,
                     'refrac': self.refrac,
                     'thresh': self.thresh}

        return namespace

    def threshold(self):
        return 'V > thresh'

    def refractory(self):
        return 'V >= refrac'

    def reset(self):
        return None

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon

    def init_cond(self):
        # return  dict(V = '-rand()*mV', w = 'rand()*mV', z = 'rand()')
        return  dict(V = '-1.2*mV', w = '0.6*mV', z = '0')



#leaky integrate and fire
#non-linear excitatory synapses
#gap junction inhibitory synapses
class n_lif:
    taum = 10*ms #membrane

    vt = -50*mV #threshold for spike
    vr = -74*mV #resting/reset potential

    refKC = 2*ms



    def __init__(self, mon):
        self.states_to_mon = mon
        return

    def eqs(self):
        eqns_KCs = '''
                dv/dt = ((vr - v) + (I_synE-I_synI + I_inj) / nS) / taum  : volt (unless refractory)
                I_synE                       : amp
                I_synI                       : amp
                I_inj                        : amp
                '''
        return eqns_KCs

    def namespace(self):
        ns = dict(  taum = self.taum,
                    vt = self.vt,
                    vr = self.vr,
                    refrac = self.refKC)
        return ns

    def threshold(self):
        return 'v > vt'

    def refractory(self):
        return 'refrac'

    def reset(self):
        return 'v = vr'

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon

    def init_cond(self):
        return dict(v = self.vr)


#leaky integrate (non spiking)
#Non-linear excitatory synapses
class n_li:

    #GGN

    taum = 10*ms #membrane
    vr = -74*mV #resting/reset potential

    Ee = 0*mV

    taue = 5*ms #1 ms diehl/cook

    refGGN = 0*ms

    def __init__(self, mon):
        self.states_to_mon = mon
        return

    def eqs(self):
        eqns_GGN = '''
                    dv/dt = ((vr - v) + I_synE/nS) / taum  : volt
                    I_synE : amp
                    '''
        return eqns_GGN

    def namespace(self):
        ns = dict(  taum = self.taum,
                    vr = self.vr,
                    refrac = self.refGGN)
        return ns

    def threshold(self):
        return None

    def refractory(self):
        return 'refrac'

    def reset(self):
        return None

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon

    def init_cond(self):
        return dict(v = self.vr)



class n_Projection_Neuron:

    #global variables for all instances of the class
    C_m = 142.0*pF

    #Maximum conductances
    g_Na = 7150.0*nS
    g_K = 1430.0*nS
    g_L = 21.0*nS
    g_KL = 5.72*nS
    g_A = 1430.0*nS

    #Reversal potentials
    E_Na = 50.0*mV
    E_K = -95.0*mV
    E_L = -55.0*mV
    E_KL = -95.0*mV

    #Gating Variable m parameters
    vm = -43.9*mV
    dvm = 7.4*mV
    vmt = -47.5*mV
    dvmt = 40.0*mV
    tm0 = 0.024*ms
    tm1 = 0.093*ms

    #Gating Variable h parameters
    vh = -48.3*mV
    dvh = 4.0*mV
    vht = -56.8*mV
    dvht = 16.9*mV
    th0 = 0.0*ms
    th1 = 5.6*ms

    #Gating Variable n parameters
    van = 35.1*mV
    san = 5.0*mV
    can = 0.016/(ms*mV)
    vbn = 20.0*mV
    sbn = 40.0*mV
    cbn = 0.25/ms

    #Gating Variable z parameters
    vz = -60.0*mV
    dvz = -8.5*mV
    tzv1 = -35.8*mV
    tzd1 = 19.7*mV
    tzv2 = -79.7*mV
    tzd2 = 12.7*mV

    #Gating Variable u parameters
    vu = -78.0*mV
    dvu = 6.0*mV
    tuv1 = -46.0*mV
    tud1 = 5.0*mV
    tuv2 = -238.0*mV
    tud2 = 37.5*mV
    tuv3 = -57.0*mV
    tud3 = 3.0*mV
    ctu1 = 0.27*ms
    ctu2 = 5.1*ms

    #

    shift = 70.0*mV


    # ---- double check this for PN neurons
    thresh = -5*mV
    refrac = -5*mV


    states_to_mon = ['V']


    def __init__(self, mon):
        self.states_to_mon = mon
        return

    def eqs(self):
        eqns_PN = '''
                    dV/dt = -1/C_m*(g_L*(V - E_L) + g_Na*m**3*h*(V - E_Na) \
                            + g_K*n*(V - E_K) + g_A*z**4*u*(V - E_K)  + g_KL*(V - E_K) - I_inj*active + I_syn_inh): volt
                    I_inj: amp
                    active: 1
                    I_syn_inh: amp

                    dm/dt = (xm-m)/tm : 1
                    xm = 0.5*(1 - tanh(0.5*(V - vm)/dvm)) : 1
                    tm = tm0+tm1*(1-tanh((V - vm)/dvm)**2) : second

                    dh/dt = (xh-h)/th : 1
                    xh = 0.5*(1-tanh(0.5*(V - vh)/dvh)) : 1
                    th = th0+th1*(1-tanh((V - vh)/dvh)**2): second

                    dn/dt = an*(1-n) - bn*n : 1
                    an = can*(V - van + shift)/(1-exp(-(V-van)/san)) : 1/second
                    bn = cbn*exp(-(V - vbn + shift)/sbn) : 1/second

                    dz/dt = (xz - z)/tz : 1
                    xz = 0.5*(1-tanh(0.5*(V-vz)/dvz)) : 1
                    tz = 1*ms/(exp((V-tzv1)/tzd1) + exp(-(V-tzv1)/tzd2) + 0.37) : second

                    du/dt = (xu - u)/tu : 1
                    xu = 0.5*(1-tanh(0.5*(V - vu)/dvu)) : 1
                    tu = ctu1/(exp((V-tuv1)/tud1) + exp(-(V-tuv2)/tud2)) + ctu2/2*(1+tanh((V-tuv3)/tud3)) : second

                    '''
        return eqns_PN

    def namespace(self):
        namespace = dict(C_m = self.C_m,
                         g_L = self.g_L,
                         g_Na = self.g_Na,
                         g_K = self.g_K,
                         g_KL = self.g_KL,
                         g_A = self.g_A,
                         E_L = self.E_L,
                         E_Na = self.E_Na,
                         E_K = self.E_K,
                         E_KL = self.E_KL,
                         vm = self.vm,
                         vmt = self.vmt,
                         vh = self.vh,
                         vht = self.vht,
                         dvm = self.dvm,
                         dvmt = self.dvmt,
                         dvh = self.dvh,
                         dvht = self.dvht,
                         tm0 = self.tm0,
                         tm1 = self.tm1,
                         th0 = self.th0,
                         th1 = self.th1,
                         van = self.van,
                         san = self.san,
                         can = self.can,
                         vbn = self.vbn,
                         sbn = self.sbn,
                         cbn = self.cbn,
                         vz = self.vz,
                         dvz = self.dvz,
                         tzv1 = self.tzv1,
                         tzd1 = self.tzd1,
                         tzv2 = self.tzv2,
                         tzd2 = self.tzd2,
                         vu = self.vu,
                         dvu = self.dvu,
                         tuv1 = self.tuv1,
                         tud1 = self.tud1,
                         tuv2 = self.tuv2,
                         tud2 = self.tud2,
                         tuv3 = self.tuv3,
                         tud3 = self.tud3,
                         ctu1 = self.ctu1,
                         ctu2 = self.ctu2,
                         shift = self.shift,
                         refrac = self.refrac,
                         thresh = self.thresh)

        return namespace

    def threshold(self):
        return 'V > thresh'

    def refractory(self):
        return 'V >= refrac'

    def reset(self):
        return None

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon

    def init_cond(self):
        return  dict(V = -65*mV,
                     m = 'rand()',
                     h = 'rand()',
                     n = 'rand()',
                     z = 'rand()',
                     u = 'rand()')

class n_Local_Neuron:

    #global variables for all instances of the class
    C_m = 142.0*pF

    #Maximum conductances
    g_K = 1000.0*nS
    g_L = 21.5*nS
    g_KL = 1.43*nS
    g_Ca = 290.0*nS
    g_KCa = 35.8*nS

    #Reversal potentials
    E_K = -95.0*mV
    E_L = -50.0*mV
    E_KL = -95.0*mV
    E_Ca = 140.0*mV

    #Gating Variable n parameters
    van = -35.0*mV
    san = 5.0*mV
    can = 0.02/ms
    vbn = -20.0*mV
    sbn = 40.0*mV
    cbn = 0.5/ms

    #Gating Variable s parameters
    vs = -20.0*mV
    dvs = -6.5*mV
    ts = 1.5*ms

    # Gating Variable v parameters
    vv = -25.0*mV
    dvv = 12.0*mV
    tvc1 = 0.3*ms
    tvv1 = 40*mV
    tvd1 = 13.0*mV
    tvc2 = 0.002*ms
    tvv2 = 60.0*mV
    tvd2 = 29.0*mV

    #Gating Variable u parameters
    vu = -78.0*mV
    dvu = 6.0*mV
    tuv1 = -46.0*mV
    tud1 = 5.0*mV
    tuv2 = -238.0*mV
    tud2 = 37.5*mV

    #Gating Vairable q parameters
    k = 2.0*uM
    tqc = 100*uM*ms

    #Calicum parameters
    phi = 2.86e-6*uM/(ms*pA)
    Cai = 0.24*uM
    tCa = 150*ms

    scale = 1
    #



    # ---- double check this for PN neurons
    thresh = -28*mV
    refrac = -28*mV


    states_to_mon = ['V']


    def __init__(self, mon):
        self.states_to_mon = mon
        return

    def eqs(self):
        eqns_LN = '''
                    dV/dt = -1/C_m*(g_L*(V - E_L) + g_KCa*q*(V - E_K)  \
                            + g_K*n**4*(V - E_K) + g_Ca*s**2*v*(V - E_Ca) \
                            + g_KL*(V - E_K) - I_inj*active + I_syn_ex + I_syn_inh): volt
                    I_inj: amp
                    active: 1
                    I_syn_ex: amp
                    I_syn_inh: amp

                    dCa/dt = -phi*g_Ca*s**2*v*(V-E_Ca) - (Ca - Cai)/tCa: mM

                    dn/dt = (xn - n)/tn : 1
                    xn = an/(an + bn) : 1
                    tn = 4.65/(an + bn) : second
                    an = can*(V - van)/(1-exp(-(V-van)/san)) : 1/second
                    bn = cbn*exp(-(V - vbn)/sbn) : 1/second

                    ds/dt = (xs - s)/ts : 1
                    xs = 0.5*(1-tanh(0.5*(V-vs)/dvs)) : 1

                    dv/dt = (xv - v)/tv : 1
                    xv = 0.5*(1-tanh(0.5*(V - vv)/dvv)) : 1
                    tv = tvc1*exp((V-tvv1)/tvd1) + tvc2*exp(-(V-tvv2)/tvd2) : second

                    dq/dt = (xq - q)/tq : 1
                    xq = Ca/(Ca + k) : 1
                    tq = tqc/(Ca + k) : second

                    '''
        return eqns_LN

    def namespace(self):
        namespace = dict(C_m = self.C_m,
                         g_L = self.g_L,
                         g_K = self.g_K,
                         g_KL = self.g_KL,
                         g_Ca = self.g_Ca,
                         g_KCa = self.g_KCa,
                         E_L = self.E_L,
                         E_K = self.E_K,
                         E_KL = self.E_KL,
                         E_Ca = self.E_Ca,
                         van = self.van,
                         san = self.san,
                         can = self.can,
                         vbn = self.vbn,
                         sbn = self.sbn,
                         cbn = self.cbn,
                         vs = self.vs,
                         dvs = self.dvs,
                         ts = self.ts,
                         vv = self.vv,
                         dvv = self.dvv,
                         tvv1 = self.tvv1,
                         tvd1 = self.tvd1,
                         tvc1 = self.tvc1,
                         tvv2 = self.tvv2,
                         tvd2 = self.tvd2,
                         tvc2 = self.tvc2,
                         phi = self.phi,
                         Cai = self.Cai,
                         tCa = self.tCa,
                         k = self.k,
                         tqc = self.tqc,
                         refrac = self.refrac,
                         thresh = self.thresh,
                         scale = self.scale)

        return namespace

    def threshold(self):
        return 'V > thresh'

    def refractory(self):
        return 'V >= refrac'

    def reset(self):
        return None

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon

    def init_cond(self):
        return  dict(V = -60*mV,
                     Ca = 'rand()*uM',
                     n = 'rand()',
                     s = 'rand()',
                     v = 'rand()',
                     q = 'rand()')

#classic NaKL neuron
#4 dimensional
class n_HH:

    C_m = 1.*uF/cm**2 # membrane capacitance, unit: uFcm^-2
    # Conductances
    g_L = 0.3*msiemens/cm**2 # Max. leak conductance, unit: mScm^-2
    g_Na = 120*msiemens/cm**2 # Max. Na conductance, unit: mScm^-2
    g_K = 20*msiemens/cm**2 # Max. K conductance, unit: mScm^-2

    # Nernst/reversal potentials
    E_L = -54.4*mV # Leak Nernst potential, unit: mV
    E_Na = 50*mV # Na Nernst potential, unit: mV
    E_K = -77*mV # K Nernst potential, unit: mV

    # Half potentials of gating variables
    vm = -40*mV # m half potential, unit: mV
    vh = -60*mV # h half potential, unit: mV
    vn = -55*mV # n half potential, unit: mV

    # Voltage response width (sigma)
    dvm = 15.0*mV # m voltage response width, unit: mV
    dvn = 30.0*mV
    dvh = -15.0*mV

    # time constants
    tm0 = 0.1*ms # unit ms
    tm1 = 0.4*ms
    th0 = 1.*ms
    th1 = 7.*ms
    tn0 = 1.*ms
    tn1 = 5.*ms

    thresh = -20*mV
    refrac = -20*mV

    scale = 1


    states_to_mon = ['V']


    def __init__(self, mon):
        self.states_to_mon = mon
        return

    def eqs(self):
        eqns_AL = '''
                    dV/dt = -1/C_m*(g_L*(V - E_L) + g_Na*m**3*h*(V - E_Na) \
                            + g_K*n**4*(V - E_K) - I_inj*active + I_syn): volt
                    I_inj: amp/meter**2
                    active: 1
                    I_syn: amp/meter**2
                    dm/dt = (xm-m)/tm : 1
                    xm = 0.5*(1+tanh((V - vm)/dvm)) : 1
                    tm = tm0+tm1*(1-tanh((V - vm)/dvm)**2) : second

                    dh/dt = (xh-h)/th : 1
                    xh = 0.5*(1+tanh((V - vh)/dvh)) : 1
                    th = th0+th1*(1-tanh((V - vh)/dvh)**2): second

                    dn/dt = (xn-n)/tn : 1
                    xn = 0.5*(1+tanh((V - vn)/dvn)) : 1
                    tn = tn0+tn1*(1-tanh((V - vn)/dvn)**2) : second
                    '''
        return eqns_AL

    def namespace(self):
        namespace = dict(C_m = self.C_m,
                         g_L = self.g_L,
                         g_Na = self.g_Na,
                         g_K = self.g_K,
                         E_L = self.E_L,
                         E_Na = self.E_Na,
                         E_K = self.E_K,
                         vm = self.vm,
                         vh = self.vh,
                         vn = self.vn,
                         dvm = self.dvm,
                         dvn = self.dvn,
                         dvh = self.dvh,
                         tm0 = self.tm0,
                         tm1 = self.tm1,
                         th0 = self.th0,
                         th1 = self.th1,
                         tn0 = self.tn0,
                         tn1 = self.tn1,
                         refrac = self.refrac,
                         thresh = self.thresh,
                         scale = self.scale)

        return namespace

    def threshold(self):
        return 'V > thresh'

    def refractory(self):
        return 'V >= refrac'

    def reset(self):
        return None

    def method(self):
        return 'rk4'

    def state_mon(self):
        return self.states_to_mon

    def init_cond(self):
        return  dict(V = '-70*mV*rand()',
                     m = 'rand()',
                     h = 'rand()',
                     n = 'rand()')

#-----------------------------------------------------------------
#SYNAPSES
'''
naming convention:  all synapses start with s, followed by the name of
the synapse, followed by excitation or inhibition

def eqs(self):
    return string of equations in Brian format

def onpre(self):
    return string of equations to execute in the event of a pre-synaptic spike

def onpost(self):
    return string of equations to execute in the event of a post-synaptic spike

def namespace(self):
    return dictionary of the variables and their values

def method(self):
    return method liek 'rk4'

def getDelay(self):
    return value*ms which is the delay of the synapse
    Note: only works for synapses with an onpre function that doesn't return None

def init_cond(self):
    return initial conditions of the variables
'''
class s_FitzHughNagumo_inh:

    def __init__(self, cond):
        self.g_syn = cond

    delay = 0*ms

    def __init__(self, conduct):
        self.g_syn = conduct
        return

    def eqs(self):
        syn = '''
                g_syn: 1
                I_syn_post = g_syn/(1.0+exp(-1000*V_pre/mV)): 1 (summed)
                '''
        return syn

    def onpre(self):
        return None

    def onpost(self):
        return None

    def namespace(self):
        return None

    def method(self):
        return 'rk4'

    def getDelay(self):
        return self.delay

    def init_cond(self):
        return {'g_syn': self.g_syn}


class s_lif_ex:
    taue = 5*ms
    Ee = 0*mV
    delay = 0*ms

    def __init__(self, conduct):
        # What is this even?
        self.g_syn = conduct
        return

    def eqs(self):
        eqns_slif = '''dge_syn/dt = -ge_syn/taue : 1 (clock-driven)
                    w_syn : 1
                    I_synE_post = ge_syn * nS * (Ee - v_post) : amp (summed)'''

        return eqns_slif

    def onpre(self):
        return '''ge_syn += w_syn'''

    def onpost(self):
        return None

    def namespace(self):
        ns = dict( taue = self.taue,
                    Ee = self.Ee)
        return ns

    def method(self):
        return 'rk4'

    def getDelay(self):
        return self.delay

    def init_cond(self):
        return dict(w_syn = self.g_syn)

class s_lif_in:

    delay = 0*ms
    taui = 2*ms
    Ei = -100*mV

    def __init__(self, conduct):
        self.g_syn = conduct
        return

    def eqs(self):
        return '''
                dgi_syn/dt = -gi_syn/taui : 1 (clock-driven)
                w_syn : 1
                I_synI_post = gi_syn * nS * (Ei - v) : amp (summed)'''

    def onpre(self):
        return '''gi_syn += w_syn'''

    def onpost(self):
        return None

    def namespace(self):
        ns = dict (taui = self.taui,
                    Ei = self.Ei)
        return ns

    def method(self):
        return 'rk4'

    def getDelay(self):
        return self.delay

    def init_cond(self):
        return dict(w_syn = self.g_syn)


class s_gapjunc_in:

    delay = 0*ms

    def __init__(self, conduct):
        self.g_syn = conduct
        return


    def eqs(self):
        S = '''
             w : 1
             I_synI_post = w*(v_pre - v_post)*nS : amp (summed)
             '''
        return S

    def onpre(self):
        return None

    def onpost(self):
        return None

    def namespace(self):
        return None

    def method(self):
        return 'rk4'

    def init_cond(self):
        return dict(w = self.g_syn)


#Empirical STDP
class s_lifSTDP_ex:

    delay = 0*ms
    taue = 5*ms
    Ee = 0*mV

    '''
    conduct: max conductance
    eta: learning rate
    taupre/taupost: time constant for STDP decay
    '''
    def __init__(self, conduct, eta, taupre, taupost):
        self.g_syn = conduct
        self.dApre = eta*conduct
        self.dApost = -eta*conduct * taupre / taupost * 1.05
        self.taupre = taupre
        self.taupost = taupost
        return

    def eqs(self):
        S = '''w_syn : 1
            dApre/dt = -Apre / taupre : 1 (event-driven)
            dApost/dt = -Apost / taupost : 1 (event-driven)
            dge_syn/dt = -ge_syn/taue : 1 (clock-driven)
            I_synE_post = ge_syn * nS * (Ee - v) : amp (summed)'''

        return S

    def onpre(self):
        on_pre='''ge_syn += w_syn
                Apre += dApre
                w_syn = clip(w_syn + Apost, 0, g_syn)'''
        return on_pre

    def onpost(self):
        on_post='''Apost += dApost
                w_syn = clip(w_syn + Apre, 0, g_syn)'''

        return on_post

    def namespace(self):
        return dict(g_syn = self.g_syn,
                    dApre = self.dApre,
                    dApost = self.dApost,
                    taupre = self.taupre,
                    taupost = self.taupost,
                    taue = self.taue,
                    Ee = self.Ee)

    def method(self):
        return 'rk4'

    def getDelay(self):
        return self.delay

    def init_cond(self):
        return dict(w_syn = 'rand()*g_syn')


class s_glu_ex:

    E_glu = -38.0*mV
    alphaR = 2.4/ms
    betaR = 0.56/ms
    Tm = 1.0
    Kp = 5.0*mV
    Vp = 7.0*mV

    delay = 0*ms

    def __init__(self, conduct):
        self.g_syn = conduct
        return

    def eqs(self):
        syn =   '''
                gNt: siemens/meter**2
                I_syn_post = gNt*r*(V - E_glu): amp/meter**2 (summed)
                dr/dt = (alphaR*Tm/(1+exp(-(V_pre - Vp)/Kp)))*(1-r) - betaR*r : 1 (clock-driven)
                '''
        return syn

    def onpre(self):
        return None

    def onpost(self):
        return None

    def namespace(self):
        return dict(E_glu = self.E_glu,
                    alphaR = self.alphaR,
                    betaR = self.betaR,
                    Tm = self.Tm,
                    Kp = self.Kp,
                    Vp = self.Vp)

    def method(self):
        return 'rk4'

    def getDelay(self):
        return self.delay

    def init_cond(self):
        return {'gNt': self.g_syn,
                'r': 'rand()'}
class s_GABA_inh:

    E_gaba = -80.0*mV
    alphaR = 5.0/ms
    betaR = 0.18/ms
    Tm = 1.5
    Kp = 5.0*mV
    Vp = 7.0*mV

    delay = 0*ms

    def __init__(self, conduct):
        self.g_syn = conduct
        return

    def eqs(self):
        syn =   '''
                gNt: siemens/meter**2
                I_syn_post = gNt*r*(V - E_gaba): amp/meter**2 (summed)
                dr/dt = (alphaR*Tm/(1+exp(-(V_pre - Vp)/Kp)))*(1-r) - betaR*r : 1 (clock-driven)
                '''
        return syn

    def onpre(self):
        return None

    def onpost(self):
        return None

    def namespace(self):
        return dict(E_gaba = self.E_gaba,
                    alphaR = self.alphaR,
                    betaR = self.betaR,
                    Tm = self.Tm,
                    Kp = self.Kp,
                    Vp = self.Vp)

    def method(self):
        return 'rk4'

    def getDelay(self):
        return self.delay

    def init_cond(self):
        return {'gNt': self.g_syn,
                'r': 'rand()'}

class s_PN_ex:

    E_syn = 0.0*mV
    r1 = 1.5
    tau = 1.0*ms

    #E_glu = -38.0*mV
    #alphaR = 2.4/ms
    #betaR = 0.56/ms
    #Tm = 1.0
    Kp = 1.5*mV
    #Kp = 5.0*mV # maybe this?
    #Vp = 7.0*mV # maybe should switch to this?
    Vp = 0.0*mV

    delay = 0*ms

    def __init__(self, conduct):
        self.g_syn = conduct
        return

    def eqs(self):
        syn =   '''
                gNt: siemens
                I_syn_ex_post = gNt*r*(V_post - E_syn): amp (summed)
                dr/dt = (xr - r)/(tau*(r1 - xr)) : 1 (clock-driven)
                xr = 0.5*(1-tanh(-0.5*(V_pre-Vp)/Kp)) : 1
                '''
        return syn

    def onpre(self):
        return None

    def onpost(self):
        return None

    def namespace(self):
        return dict(E_syn = self.E_syn,
                    tau = self.tau,
                    r1 = self.r1,
                    Kp = self.Kp,
                    Vp = self.Vp)

    def method(self):
        return 'rk4'

    def getDelay(self):
        return self.delay

    def init_cond(self):
        return {'gNt': self.g_syn,
                'r': 'rand()'}


class s_LN_inh:

    g_slow = 125*nS
    E_gaba = -70.0*mV
    E_K = -95.0*mV
    alphaR = 10.0/ms
    betaR = 0.16/ms
    dvra = 1.5*mV
    vra = -20.0*mV
    dvrb = 5.0*mV
    vrb = 2.0*mV

    s1 = 1.0/ms
    s2 = 0.0025/ms
    s3 = 0.1/ms
    s4 = 0.06/ms

    K = 100.0*uM**4



    delay = 0*ms

    def __init__(self, conduct):
        self.g_syn = conduct
        return

    def eqs(self):
        syn =   '''
                gNt: siemens
                I_syn_inh_post = gNt*r*(V_post - E_gaba) + g_slow*G**4/(G**4 + K)*(V_post - E_K) : amp (summed)
                dr/dt = alphaR/(1+exp(-(V_pre - vra)/dvra))*(1-r) - betaR*r : 1 (clock-driven)
                dss/dt = s1*(1 - ss)/(1+exp(-(V_pre - vrb)/dvrb)) - s2*ss : 1 (clock-driven)
                dG/dt = s3*ss - s4*G : mM (clock-driven)

                '''
        return syn

    def onpre(self):
        return None

    def onpost(self):
        return None

    def namespace(self):
        return dict(E_gaba = self.E_gaba,
                    E_K = self.E_K,
                    g_slow = self.g_slow,
                    alphaR = self.alphaR,
                    betaR = self.betaR,
                    dvra = self.dvra,
                    vra = self.vra,
                    dvrb = self.dvrb,
                    vrb = self.vrb,
                    s1 = self.s1,
                    s2 = self.s2,
                    s3 = self.s3,
                    s4 = self.s4,
                    K = self.K)

    def method(self):
        return 'rk4'

    def getDelay(self):
        return self.delay

    def init_cond(self):
        return {'gNt': self.g_syn,
                'r': 'rand()',
                'ss': 'rand()',
                'G': 'rand()*uM'}
