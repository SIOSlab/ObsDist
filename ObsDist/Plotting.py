# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:14:56 2016

@author: Daniel
"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u

import ObsDist.Population as Population

pop = Population.Population(a_min=0.1*u.AU,a_max=10.0*u.AU,R_min=6000*u.km,R_max=50000*u.km)
a = np.linspace(pop.arange[0],pop.arange[1],100)
fa = pop.f_a(a)
e = np.linspace(pop.erange[0],pop.erange[1],100)
fe = pop.f_e(e)
R = np.linspace(pop.Rrange[0],pop.Rrange[1],100)
fR = pop.f_R(R)
p = np.linspace(pop.prange[0],pop.prange[1],100)
fp = pop.f_p(p)

# use TeX fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('legend',fontsize=16.0)

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
ax1.plot(a,fa,'-b')
ax1.set_ylim(ymin=0.0)
ax1.set_xlabel(r'$ a $', fontsize=18)
ax1.set_ylabel(r'$ f_{\bar{a}}\left(a\right) $', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=16)
fig1.show()

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
ax2.plot(e,fe,'-b')
ax2.set_ylim(ymin=0.0)
ax2.set_xlabel(r'$ e $', fontsize=18)
ax2.set_ylabel(r'$ f_{\bar{e}}\left(e\right) $', fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=16)
fig2.show()

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
ax3.plot(R,fR,'-b')
ax3.set_ylim(ymin=0.0)
ax3.set_xlabel(r'$ R $', fontsize=18)
ax3.set_ylabel(r'$ f_{\bar{R}}\left(R\right) $', fontsize=18)
ax3.tick_params(axis='both', which='major', labelsize=16)
fig3.show()

fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
ax4.plot(p,fp,'-b')
ax4.set_ylim(ymin=0.0)
ax4.set_xlabel(r'$ p $', fontsize=18)
ax4.set_ylabel(r'$ f_{\bar{p}}\left(p\right) $', fontsize=18)
ax4.tick_params(axis='both', which='major', labelsize=16)
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