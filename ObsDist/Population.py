# -*- coding: utf-8 -*-
"""
v1: Created on November 28, 2016
author: Daniel Garrett (dg622@cornell.edu)
"""

import numpy as np
import astropy.units as u

class Population(object):
    """This class contains all the planetary parameters necessary for sampling
    or finding probability distribution functions
    
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
        arange (ndarray):
            1D numpy ndarray containing minimum and maximum semi-major axis in
            AU
        erange (ndarray):
            1D numpy ndarray containing minimum and maximum eccentricity
        Rrange (ndarray):
            1D numpy ndarray containing minimum and maximum planetary radius in
            AU
        prange (ndarray):
            1D numpy ndarray containing minimum and maximum geometric albedo
        Phi (callable):
            phase function
    
    """
    
    def __init__(self, a_min=None, a_max=None, e_min=None, e_max=None, \
                 R_min=None, R_max=None, p_min=None, p_max=None):
        unittest = u.quantity.Quantity
        # minimum semi-major axis (AU)
        if a_min == None:
            a_min = 0.5
        elif type(a_min) == unittest:
            a_min = a_min.to('AU').value
        # maximum semi-major axis (AU)
        if a_max == None:
            a_max = 5.0
        elif type(a_max) == unittest:
            a_max = a_max.to('AU').value
        # semi-major axis range
        self.arange = np.array([a_min, a_max])
        # minimum eccentricity
        if e_min == None:
            e_min = np.finfo(float).eps*100.0
        # maximum eccentricity
        if e_max == None:
            e_max = 0.35
        # eccentricity range
        self.erange = np.array([e_min, e_max])
        # minimum planetary radius
        if R_min == None:
            R_min = 6000*u.km
            R_min = R_min.to('AU').value
        elif type(R_min) == unittest:
            R_min = R_min.to('AU').value
        # maximum planetary radius
        if R_max == None:
            R_max = 30000*u.km
            R_max = R_max.to('AU').value
        elif type(R_max) == unittest:
            R_max = R_max.to('AU').value
        self.Rrange = np.array([R_min, R_max]) # in AU
        # minimum albedo
        if p_min == None:
            p_min = 0.2
        # maximum albedo
        if p_max == None:
            p_max = 0.3
        self.prange = np.array([p_min, p_max])
        # phase function
        self.Phi = lambda b: (1.0/np.pi)*(np.sin(b) + (np.pi-b)*np.cos(b))
            
    def f_a(self, a):
        """Probability density function for semi-major axis in AU
        
        Args:
            a (float or ndarray):
                Semi-major axis value(s) in AU
                
        Returns:
            f (ndarray):
                Probability density (units of 1/AU)
        
        """ 
        a = np.array(a, ndmin=1, copy=False)
        # uniform
#        f = ((a >= self.arange[0]) & (a <= self.arange[1])).astype(int)/(self.arange[1]-self.arange[0])
        # log-uniform
        f = ((a >= self.arange[0]) & (a <= self.arange[1])).astype(int)/(a*np.log(self.arange[1]/self.arange[0]))
        
        return f
        
    def f_e(self, e):
        """Probability density function for eccentricity
        
        Args:
            e (float or ndarray):
                eccentricity value(s)
        
        Returns:
            f (ndarray):
                probability density
        
        """
        
        e = np.array(e, ndmin=1, copy=False)
        # uniform
        f = ((e >= self.erange[0]) & (e <= self.erange[1])).astype(int)/(self.erange[1]-self.erange[0])
                
        return f
        
    def f_R(self, R):
        """Probability density function for planet radius (AU)
        
        Args:
            R (float or ndarray):
                planet radius in AU
        
        Returns:
            f (ndarray):
                probability density function value
                
        """
        
        R = np.array(R, ndmin=1, copy=False)
        # uniform
#        f = ((R >= self.Rrange[0]) & (R <= self.Rrange[1])).astype(int)/(self.Rrange[1]-self.Rrange[0])
        # log-uniform
        f = ((R >= self.Rrange[0]) & (R <= self.Rrange[1])).astype(int)/(R*np.log(self.Rrange[1]/self.Rrange[0]))
        
        return f
        
    def f_p(self, p):
        """Probability density function for geometric albedo
        
        Args:
            x (float or ndarray):
                geometric albedo
        
        Returns:
            f (ndarray):
                probability density function value
                
        """
        
        p = np.array(p, ndmin=1, copy=False)
        # uniform
        f = ((p >= self.prange[0]) & (p <= self.prange[1])).astype(int)/(self.prange[1]-self.prange[0])
        
        return f