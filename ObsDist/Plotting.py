# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:14:56 2016

@author: dg622@cornell.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import ObsDist.Population as Population
from ObsDist.InverseTransformSampler import InverseTransformSampler
from ObsDist.eccanom import eccanom

#%% system parameters
smin = 1.0 # AU
smax = 16.0 # AU
dmag0 = 25.0

#%% planet population parameters
a_min = 0.5 # AU
a_max = 25.0 # AU
e_min = 0.0
e_max = 0.35
p_min = 0.2
p_max = 0.3
R_min = 4.0107523e-5 # 6000 km in AU
R_max = 2.0053761e-4 # 30000 km in AU

r_min = a_min*(1.0-e_max) # AU
r_max = a_max*(1.0+e_max) # AU
z_min = p_min*R_min**2 # AU**2
z_max = p_max*R_max**2 # AU**2

# get saved planet population distributions
pop = Population.Population(a_min=a_min,a_max=a_max,R_min=R_min,R_max=R_max,p_min=p_min,p_max=p_max)

#%% Monte Carlo sampling to plot
a_sampler = InverseTransformSampler(pop.f_a,a_min,a_max)
e_sampler = InverseTransformSampler(pop.f_e,e_min,e_max)
p_sampler = InverseTransformSampler(pop.f_p,p_min,p_max)
R_sampler = InverseTransformSampler(pop.f_R,R_min,R_max)
nplan = int(1e6)
bins = 100
numiter = 10 # gives 10 million samples
for i in xrange(numiter):
    # sample quantities
    aMC = a_sampler(nplan)
    eMC = e_sampler(nplan)
    pMC = p_sampler(nplan)
    RMC = R_sampler(nplan)
    IMC = np.arccos(2.0*np.random.rand(nplan)-1.0)
    OMC = 2.0*np.pi*np.random.rand(nplan)
    wMC = 2.0*np.pi*np.random.rand(nplan)
    MMC = 2.0*np.pi*np.random.rand(nplan)
    zMC = pMC*RMC**2
    # find Eccentric anomaly
    EMC = eccanom(MMC,eMC)
    r1 = aMC*(np.cos(EMC) - eMC)
    r1 = np.hstack((r1.reshape(len(r1),1), r1.reshape(len(r1),1), r1.reshape(len(r1),1)))
    r2 = (aMC*np.sin(EMC)*np.sqrt(1. -  eMC**2))
    r2 = np.hstack((r2.reshape(len(r2),1), r2.reshape(len(r2),1), r2.reshape(len(r2),1)))
                
    a1 = np.cos(OMC)*np.cos(wMC) - np.sin(OMC)*np.sin(wMC)*np.cos(IMC)
    a2 = np.sin(OMC)*np.cos(wMC) + np.cos(OMC)*np.sin(wMC)*np.cos(IMC)
    a3 = np.sin(wMC)*np.sin(IMC)
    A = np.hstack((a1.reshape(len(a1),1), a2.reshape(len(a2),1), a3.reshape(len(a3),1)))

    b1 = -np.cos(OMC)*np.sin(wMC) - np.sin(OMC)*np.cos(wMC)*np.cos(IMC)
    b2 = -np.sin(OMC)*np.sin(wMC) + np.cos(OMC)*np.cos(wMC)*np.cos(IMC)
    b3 = np.cos(wMC)*np.sin(IMC)
    B = np.hstack((b1.reshape(len(b1),1), b2.reshape(len(b2),1), b3.reshape(len(b3),1)))

    # planet position, planet-star distance, apparent separation
    r = (A*r1 + B*r2) # position vector
    d = np.sqrt(np.sum(r**2, axis=1)) # planet-star distance
    s = np.sqrt(np.sum(r[:,0:2]**2, axis=1)) # apparent separation
    betaMC = np.arccos(r[:,2]/d) # phase angle
    PhiMC = pop.Phi(betaMC)
    dMag = -2.5*np.log10(zMC/d**2*PhiMC) # difference in magnitude
    if i == 0:
        HRp, Redges = np.histogram(RMC[(s>smin) & (s<smax) & (dMag<dmag0)],bins=bins,range=[R_min,R_max],density=True)
        Hpp, pedges = np.histogram(pMC[(s>smin) & (s<smax) & (dMag<dmag0)],bins=bins,range=[p_min,p_max],density=True)
        Hap, aedges = np.histogram(aMC[(s>smin) & (s<smax) & (dMag<dmag0)],bins=bins,range=[a_min,a_max],density=True)
        Hep, eedges = np.histogram(eMC[(s>smin) & (s<smax) & (dMag<dmag0)],bins=bins,range=[e_min,e_max],density=True)
    else:
        hRp, Redges = np.histogram(RMC[(s>smin) & (s<smax) & (dMag<dmag0)],bins=bins,range=[R_min,R_max],density=True)
        HRp += hRp
        hpp, pedges = np.histogram(pMC[(s>smin) & (s<smax) & (dMag<dmag0)],bins=bins,range=[p_min,p_max],density=True)
        Hpp += hpp
        hap, aedges = np.histogram(aMC[(s>smin) & (s<smax) & (dMag<dmag0)],bins=bins,range=[a_min,a_max],density=True)
        Hap += hap
        hep, eedges = np.histogram(eMC[(s>smin) & (s<smax) & (dMag<dmag0)],bins=bins,range=[e_min,e_max],density=True)
        Hep += hep

#%% Plots

R = np.linspace(pop.Rrange[0],pop.Rrange[1],100)
fR = pop.f_R(R)
p = np.linspace(pop.prange[0],pop.prange[1],100)
fp = pop.f_p(p)

# use TeX fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend',fontsize=16.0)

# f_a'(a') plot
Hap /= numiter
aa = 0.5*(aedges[:-1]+aedges[1:])
a = np.linspace(a_min,a_max,200)
fa = pop.f_a(a)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(aa,Hap,'or',markerfacecolor='None',label='Monte Carlo')
ax1.plot(a,fa,'r--',linewidth=2.0,label='Assumed')
#ax1.plot(a,fap,'b-',linewidth=2.0,label=r'Observed')
ax1.set_ylim(ymin=0.0)
ax1.set_xlabel(r'$ a \; / \; a^\prime $ (AU)', fontsize=18)
ax1.set_ylabel(r'$ f_{\bar{a}}\left(a\right) \; / \; f_{\bar{a}^\prime}\left(a^\prime\right) $', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.legend()
fig1.show()

# f_e'(e') plot
Hep /= numiter
es = 0.5*(eedges[:-1]+eedges[1:])
e = np.linspace(e_min,e_max,200)
fe = pop.f_e(e)

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(es,Hep,'or',markerfacecolor='None',label='Monte Carlo')
ax2.plot(e,fe,'r--',linewidth=2.0,label='Assumed')
#ax2.plot(e,fep,'b-',linewidth=2.0,label=r'Observed')
ax2.set_ylim(ymin=2.5,ymax=3.0)
ax2.set_xlabel(r'$ e \; / \; e^\prime $', fontsize=18)
ax2.set_ylabel(r'$ f_{\bar{e}}\left(e\right) \; / \; f_{\bar{e}^\prime}\left(e^\prime\right) $', fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.legend(loc=4)
fig2.show()

# f_R'(R') plot
HRp /= numiter
Rs = 0.5*(Redges[:-1]+Redges[1:])
R = np.linspace(R_min,R_max,200)
fR = pop.f_R(R)

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.semilogx(Rs,HRp,'or',markerfacecolor='None',label='Monte Carlo')
ax3.semilogx(R,fR,'r--',linewidth=2.0,label='Assumed')
#ax3.plot(R,fRp,'b-',linewidth=2.0,label=r'Observed')
ax3.set_ylim(ymin=0.0)
ax3.set_xlabel(r'$ R \; / \; R^\prime $ (AU)', fontsize=18)
ax3.set_ylabel(r'$ f_{\bar{R}}\left(R\right) \; / \; f_{\bar{R}^\prime}\left(R^\prime\right) $', fontsize=18)
ax3.tick_params(axis='both', which='major', labelsize=16)
ax3.legend()
fig3.show()

# f_p'(p') plot
Hpp /= numiter
ps = 0.5*(pedges[:-1]+pedges[1:])
p = np.linspace(p_min,p_max,200)
fp = pop.f_p(p)

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
ax4.plot(ps,Hpp,'or',markerfacecolor='None',label='Monte Carlo')
ax4.plot(p,fp,'r--',linewidth=2.0,label='Assumed')
#ax4.plot(p,fpp,'b-',linewidth=2.0,label=r'Observed')
ax4.set_ylim(ymin=5.0)
ax4.set_xlabel(r'$ p \; / \; p^\prime $', fontsize=18)
ax4.set_ylabel(r'$ f_{\bar{p}}\left(p\right) \; / \; f_{\bar{p}^\prime}\left(p^\prime\right) $', fontsize=18)
ax4.tick_params(axis='both', which='major', labelsize=16)
ax4.legend(loc=4)
fig4.show()

b = np.linspace(0.0,np.pi,100)
Phi = pop.Phi(b)
fig5 = plt.figure(5)
ax5 = fig5.add_subplot(111)
ax5.plot(b,Phi,'-b')
ax5.set_ylim(ymin=0.0)
ax5.set_xlabel(r'$ \beta $', fontsize=18)
ax5.set_ylabel(r'$ \Phi\left(\beta\right) $', fontsize=18)
ax5.tick_params(axis='both', which='major', labelsize=16)
fig5.show()