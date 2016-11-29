# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:02:43 2016

@author: dg622@cornell.edu
"""
import numpy as np
import scipy.interpolate as interpolate
import scipy.integrate as integrate

import ObsDist.Population as Population

class Observed(object):
    """This class contains all the methods necessary for finding the observed
    distributions of semi-major axis, eccentricity, planetary radius, and
    geometric albedo
    
    Args:
        a_min (Quantity or float):
            minimum population semi-major axis with unit (Quantity) attached or
            in AU (float)
        a_max (Quantity or float):
            maximum population semi-major axis with unit (Quantity) attached or
            in AU (float)
        e_min (float):
            minimum population eccentricity
        e_max (float):
            maximum population eccentricity
        R_min (Quantity or float):
            minimum population planetary radius with unit (Quantity) attached 
            or in AU (float)
        R_max (Quantity or float):
            maximum population planetary radius with unit (Quantity) attached
            or in AU (float)
        p_min (float):
            minimum population geometric albedo
        p_max (float):
            maximum population geometric albedo
        
    Attributes:
        f_ap (callable):
            observed semi-major axis (in AU) probability density function 
        f_ep (callable):
            observed eccentricity probability density function
        f_Rp (callable):
            observed planetary radius (in AU) probability density function 
        f_pp (callable):
            observed geometric albedo probability density function
    
    """
    
    def __init__(self, a_min=None, a_max=None, e_min=None, e_max=None, \
                 R_min=None, R_max=None, p_min=None, p_max=None, smin = None, \
                 smax = None, dmag0 = None):
        # get population information
        self.pop = Population.Population(a_min=a_min, a_max=a_max, e_min=e_min, \
                                         e_max=e_max, R_min=R_min, R_max=R_max, \
                                         p_min=p_min, p_max=p_max)
        # check if any of these values are the same
        self.aconst = self.pop.arange[0] == self.pop.arange[1]
        self.econst = self.pop.erange[0] == self.pop.erange[1]
        self.pconst = self.pop.prange[0] == self.pop.prange[1]
        self.Rconst = self.pop.Rrange[0] == self.pop.Rrange[1]
        
        self.rmin = self.pop.arange[0]*(1.0-self.pop.erange[1])
        self.rmax = self.pop.arange[1]*(1.0+self.pop.erange[1])
        self.zmin = self.pop.prange[0]*self.pop.Rrange[0]**2
        self.zmax = self.pop.prange[1]*self.pop.Rrange[1]**2
        
        # get inverse of phase function
        beta = np.linspace(0.0, np.pi, 1000)
        Phis = self.pop.Phi(beta)
        self.Phiinv = interpolate.InterpolatedUnivariateSpline(Phis[::-1],beta[::-1],k=3,ext=1)
        
        # populate geometric and photometric detection values
        if smin == None:
            self.smin = 1.0 # AU
        else:
            self.smin = smin # AU
        if smax == None:
            self.smax = 16.0 # AU
        else:
            self.smax = smax # AU
        if dmag0 == None:
            self.dmag0 = 25.0
        else:
            self.dmag0 = dmag0
            
        # get pdf of orbital radius
        self.f_ronegrandv = np.vectorize(self.f_ronegrand)
        self.manyf_r = np.vectorize(self.onef_r)
        self.f_r = self.getf_r()
        
        # get pdf of z = p*R**2
        self.manyf_z = np.vectorize(self.onef_z)
        self.f_z = self.getf_z()
        
        # get pdf of f_ap
        self.intaezv = np.vectorize(self.intaez)
        self.intaeEv = np.vectorize(self.intaeE)
        self.intev = np.vectorize(self.inte)
        self.f_ap = self.getf_ap()
        
        # get pdf of f_ep
        self.intav = np.vectorize(self.inta)
        self.f_ep = self.getf_ep()
        
        # get pdf of f_Rp
        self.intpRrv = np.vectorize(self.intpRr)
        self.intpv = np.vectorize(self.intp)
        self.f_Rp = self.getf_Rp()
        
        # get pdf of f_pp
        self.intRprv = np.vectorize(self.intRpr)
        self.intRv = np.vectorize(self.intR)
        self.f_pp = self.getf_pp()
    
    def getf_ap(self):
        """Returns a callable probability density function for observed 
        semi-major axis"""
        a = np.linspace(self.pop.arange[0],self.pop.arange[1],201)
        grand = self.pop.f_a(a)*self.intev(a)
        ca = integrate.simps(grand,a)
        f_ap = interpolate.InterpolatedUnivariateSpline(a,grand/ca,k=3,ext=1)
    
        return f_ap
    
    def getf_ep(self):
        """Returns a callable probability density function for observed
        eccentricity"""
        e = np.linspace(self.pop.erange[0],self.pop.erange[1],51)
        grand = self.pop.f_e(e)*self.intav(e)
        ce = integrate.simps(grand,e)
        f_ep = interpolate.InterpolatedUnivariateSpline(e,grand/ce,k=3,ext=1)
        
        return f_ep
        
    def getf_Rp(self):
        """Returns a callable probability density function for observed
        planetary radius"""
        R = np.linspace(self.pop.Rrange[0],self.pop.Rrange[1],201)
        grand = self.pop.f_R(R)*self.intpv(R)
        cR = integrate.simps(grand,R)
        f_Rp = interpolate.InterpolatedUnivariateSpline(R,grand/cR,k=3,ext=1)
        
        return f_Rp
        
    def getf_pp(self):
        """Returns a callable probability density function for observed
        geometric albedo"""
        p = np.linspace(self.pop.prange[0],self.pop.prange[1],201)
        grand = self.pop.f_p(p)*self.intRv(p)
        cp = integrate.simps(grand,p)
        f_pp = interpolate.InterpolatedUnivariateSpline(p,grand/cp,k=3,ext=1)

        return f_pp
        
# f_a'(a') helper functions
    def inte(self, a):
        """Returns integral over eccentricity for probability density function
        for observed semi-major axis"""
        # a is a scalar
        grand = lambda e: self.pop.f_e(e)*self.intaeEv(a,e)
        f = integrate.fixed_quad(grand,self.pop.erange[0],self.pop.erange[1],n=5)[0]

        return f

# f_e'(e') helper functions
    def inta(self, e):
        """Returns integral over semi-major axis for probability density 
        function for observed eccentricity"""
        # e is a scalar
        grand = lambda a: self.pop.f_a(a)*self.intaeEv(a,e)
        f = integrate.fixed_quad(grand,self.pop.arange[0],self.pop.arange[1],n=100)[0]

        return f
        
    def intaeE(self, a, e):
        """Returns integral over eccentric anomaly for probability density 
        function for observed eccentricity"""
        # a is a scalar, e is a scalar
        grand = lambda E: (1.0 - e*np.cos(E))*self.intaezv(E,a,e)
        f = integrate.fixed_quad(grand,0.0,2.0*np.pi,n=5)[0]

        return f
        
    def intaez(self, E, a, e):
        """Returns integral over z = p*R**2 for probability density functions
        for observed eccentricity and semi-major axis"""
        # E, a, e are all scalars
        r = a*(1.0-e*np.cos(E))
        grand = lambda z: self.f_z(z)*self.intaeb(r,z)
        f = integrate.fixed_quad(grand, self.zmin, self.zmax, n=5)[0]
        
        return f
        
    def intaeb(self, r, z):
        """Returns integral over phase angle for probability density functions
        for observed eccentricity and semi-major axis"""
        # r is a scalar, but z can be a vector
        z = np.array(z,ndmin=1,copy=False)
        f = np.zeros(z.shape)
        # case 1
        if (r < self.smax) and (r > self.smin):
            dl = -2.5*np.log10(z/r**2)
            buval = 10.0**(-0.4*self.dmag0)*r**2/z
            # limits on beta
            b1 = np.arcsin(self.smin/r)
            b2 = np.pi - b1
            b3 = self.Phiinv(buval)
            b3[buval>1.0] = 0.0
            bl = b1
            bu = b3
            bu[b3>b2] = b2
            # value of integral
            f = 0.5*(-np.cos(bu)+np.cos(bl))
            f[dl>self.dmag0] = 0.0
        # case 2
        elif r > self.smax:
            r = np.array(r,ndmin=1,copy=False)
            dl = -2.5*np.log10(z/r**2)
            buval = 10.0**(-0.4*self.dmag0)*r**2/z
            # limits on beta
            b1 = np.arcsin(self.smin/r)
            b2 = np.arcsin(self.smax/r)
            b3 = np.pi - b2
            b4 = np.pi - b1
            b5 = self.Phiinv(buval)
            b5[buval>1.0] = 0.0
            # lower integral limits
            bl1 = b1
            bu1 = b5
            bu1[b5>b2] = b2
            # upper integral limits
            bl2 = b5
            bl2[b5<b3] = 0.0
            bu2 = b5
            bu2[b5>b4] = b4
            bu2[b5<b3] = 0.0
            # value of integral
            f = 0.5*(-np.cos(bu1)+np.cos(bl1)) + 0.5*(-np.cos(bu2)+np.cos(bl2))
            f[dl>self.dmag0] = 0.0
        else:
            f = 0.0

        return f
        
# f_R'(R') helper functions        
    def intp(self, R):
        """Returns integral over geometric albedo for probability density 
        function for observed planetary radius"""

        grand = lambda p: self.pop.f_p(p)*self.intpRrv(p,R)
        f = integrate.fixed_quad(grand,self.pop.prange[0],self.pop.prange[1],n=5)[0]
        
        return f
        
    def intpRr(self, p, R):
        """Returns integral over orbital radius for probability density 
        function for observed planetary radius"""
        # p is a scalar, R is a scalar
        grand = lambda r: self.f_r(r)*self.intRpbeta(r,R,p)
        f = integrate.fixed_quad(grand,self.smin,self.rmax,n=100)[0]
        
        return f
    
    def intRpbeta(self, r, R, p):
        """Returns innermost integral (over beta) for determining probability 
        density functions for planetary radius or geometric albedo"""
        # r is a vector, R is a scalar, p is a scalar
        r = np.array(r,ndmin=1,copy=False)
        f = np.zeros(r.shape)
        # case 1
        rb = r[r<self.smax]
        fb = f[r<self.smax]
        dl = -2.5*np.log10(self.zmax/rb**2)
        buval = 10.0**(-0.4*self.dmag0)*rb**2/(p*R**2)
        # limits on beta
        b1 = np.arcsin(self.smin/rb)
        b2 = np.pi - b1
        b3 = self.Phiinv(buval)
        b3[buval>1.0] = 0.0
        bl = b1
        bu = b2
        bu[b3<b2] = b3[b3<b2]
        # value of integral
        fb = 0.5*(-np.cos(bu)+np.cos(bl))
        fb[dl>self.dmag0] = 0.0
        # case 2
        ra = r[r>self.smax]
        fa = f[r>self.smax]
        dl = -2.5*np.log10(self.zmax/ra**2)
        buval = 10.0**(-0.4*self.dmag0)*ra**2/(p*R**2)
        # limits on beta
        b1 = np.arcsin(self.smin/ra)
        b2 = np.arcsin(self.smax/ra)
        b3 = np.pi - b2
        b4 = np.pi - b1
        b5 = self.Phiinv(buval)
        b5[buval>1.0] = 0.0
        # lower integral limits
        bl1 = b1
        bu1 = b2
        bu1[b5<b2] = b5[b5<b2]
        # upper integral limits
        bl2 = b3
        bl2[b5<b3] = 0.0
        bu2 = b4
        bu2[b5<b4] = b5
        bu2[b5<b3] = 0.0
        # value of integral
        fa = 0.5*(-np.cos(bu1)+np.cos(bl1)) + 0.5*(-np.cos(bu2)+np.cos(bl2))
        fa[dl>self.dmag0] = 0.0
        f[r<self.smax] = fb
        f[r>self.smax] = fa

        return f
        
# f_p'(p') helper functions
    def intR(self, p):
        """Returns integral over planetary radius for probability density 
        function for observed geometric albedo"""
        # p is a scalar
        grand = lambda R: self.pop.f_R(R)*self.intRprv(R,p)
        f = integrate.fixed_quad(grand,self.pop.Rrange[0],self.pop.Rrange[1],n=5)[0]
        
        return f
        
    def intRpr(self, R, p):
        """Returns integral over orbital radius for probability density 
        function for observed geometric albedo"""
        # R is a scalar, p is a scalar
        grand = lambda r: self.f_r(r)*self.intRpbeta(r,R,p)
        f = integrate.fixed_quad(grand,self.smin,self.rmax,n=100)[0]
        
        return f

# f_z(z) helper functions        
    def getf_z(self):
        """Returns a callable probability density function for z = p*R**2"""
        z = np.linspace(self.zmin,self.zmax,200)
        fz = self.manyf_z(z)
        f_z = interpolate.InterpolatedUnivariateSpline(z,fz,k=3,ext=1)
        
        return f_z
        
    def onef_z(self, z):
        """Returns probability density function value for single value of
        z = p*R**2"""
        # z is a scalar
        if (z < self.zmin) or (z > self.zmax):
            f = 0.0
        else:
            if (self.pconst & self.Rconst):
                f = 1.0
            elif self.pconst:
                f = 1.0/(2.0*np.sqrt(self.pop.prange[0]*z))*self.pop.f_R(np.sqrt(z/self.pop.prange[0]))
            elif self.Rconst:
                f = 1.0/self.pop.Rrange[0]**2*self.pop.f_p(z/self.pop.Rrange[0]**2)
            else:
                p1 = z/self.pop.Rrange[1]**2
                p2 = z/self.pop.Rrange[0]**2
                if p1 < self.pop.prange[0]:
                    p1 = self.pop.prange[0]
                if p2 > self.pop.prange[1]:
                    p2 = self.pop.prange[1]
                f = integrate.fixed_quad(self.pgrand,p1,p2,args=(z,),n=200)[0]
                
        return f
        
    def pgrand(self, p, z):
        """Returns integrand for probability density function for z = p*R**2"""
        # p is a vector, z is a scalar
        f = 1.0/(2.0*np.sqrt(z*p))*self.pop.f_R(np.sqrt(z/p))*self.pop.f_p(p)
        
        return f

# f_r(r) helper functions        
    def getf_r(self):
        """Returns a callable probability density function for orbital radius"""
        r = np.linspace(self.rmin,self.rmax,200)
        fr = self.manyf_r(r)
        f_r = interpolate.InterpolatedUnivariateSpline(r,fr,k=3,ext=1)
        
        return f_r
    
    def onef_r(self, r):
        """Returns probability density function value for single value of 
        orbital radius"""
        # r is a scalar
        if (r == self.rmin) or (r == self.rmax):
            f = 0.0
        else:
            if (self.aconst & self.econst):
                if self.pop.erange[0] == 0.0:
                    f = self.pop.f_a(r)
                else:
                    if r > self.pop.arange[0]*(1.0-self.pop.erange[0]):
                        f = (r/(np.pi*self.pop.arange[0]*np.sqrt((self.pop.arange[0]*self.pop.erange[0])**2-(self.pop.arange[0]-r)**2)))
                    else:
                        f = 0.0
            elif self.aconst:
                etest1 = 1.0 - r/self.pop.arange[0]
                etest2 = r/self.pop.arange[0] - 1.0
                if self.pop.erange[1] < etest1:
                    f = 0.0
                else:
                    if r < self.pop.arange[0]:
                        if self.pop.erange[0] > etest1:
                            low = self.pop.erange[0]
                        else:
                            low = etest1
                    else:
                        if self.pop.erange[0] > etest2:
                            low = self.pop.erange[0]
                        else:
                            low = etest2
                    f = integrate.fixed_quad(self.rgrandac, low, self.pop.erange[1], args=(self.pop.arange[0],r), n=200)[0]
            elif self.econst:
                if self.pop.erange[0] == 0.0:
                    f = self.pop.f_a(r)
                else:
                    atest1 = r/(1.0-self.pop.erange[0])
                    atest2 = r/(1.0+self.pop.erange[0])
                    if self.pop.arange[1] < atest1:
                        high = self.pop.arange[1]
                    else:
                        high = atest1
                    if self.pop.arange[0] < atest2:
                        low = atest2
                    else:
                        low = self.pop.arange[0]
                    f = integrate.fixed_quad(self.rgrandec, low, high, args=(self.pop.erange[0],r), n=200)[0]
            else:
                a1 = r/(1.0+self.pop.erange[1])
                a2 = r/(1.0-self.pop.erange[1])
                if a1 < self.pop.arange[0]:
                    a1 = self.pop.arange[0]
                if a2 > self.pop.arange[1]:
                    a2 = self.pop.arange[1]
                f = integrate.fixed_quad(self.f_ronegrandv, a1, a2, args=(r,), n=200)[0]

        return f
        
    def f_ronegrand(self, a, r):
        """Returns integral over eccentricity for probability density function 
        for orbital radius"""
        # a is a scalar, r is a scalar
        el = np.abs(1.0-r/a)
        if el < self.pop.erange[0]:
            el = self.pop.erange[0]
        if el > self.pop.erange[1]:
            f = 0.0
        else:
            f = self.pop.f_a(a)*integrate.fixed_quad(self.f_rdblgrand, el, self.pop.erange[1], args=(a,r), n=100)[0]

        return f
    
    def f_rdblgrand(self, e, a, r):
        """Returns integrand needed for probability density function for 
        orbital radius"""
        # e is a vector, a is a scalar, r is a scalar
        f = r/(np.pi*a*np.sqrt((a*e)**2 - (a-r)**2))*self.pop.f_e(e)

        return f
        
    def rgrandac(self, e, a, r):
        """Returns integrand for determining probability density of orbital 
        radius when semi-major axis is constant"""
        # e is a vector, a is a scalar, r is a scalar
        f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*self.pop.f_e(e)
        
        return f
        
    def rgrandec(self, a, e, r):
        """Returns integrand for determining probability density of orbital
        radius when eccentricity is constant"""
        # a is a vector, e is a scalar, r is a scalar
        f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*self.pop.f_a(a)
        
        return f