#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:01:33 2017
Class definition for profil reconstruction
standard profile defines a class using results of script-acp.py
@author: brajard
"""

#%% import
import numpy as np
import pickle
import os
import xarray as xr
import matplotlib.pyplot as plt

OUTDIR = './output'
SVESUFFIX = ''


#%% Class definition
class profile:
    args_lat = {'latitude-regress__inv_kw_args','latitude-regress__kw_args'}

    def __init__(self,preproc,pca,depth,interp=True,lat=None):
        self._preproc = preproc
        self._pca = pca
        self._depth = depth
        self._interp = interp
        self._lat = None
        if not lat is None:
            self.lat = lat
        if not self._depth.size == self._pca.components_.shape[1]:
            raise ValueError('depth and pca not the same shape')
        #check shape of pca.components, lat
        
        
    @property
    def lat(self):
        return self._lat
    
    @lat.setter
    def lat(self,value):
        #lat has to be bi-dimensional to work with the linear regression
        if value.ndim == 1:
            value = value[:,np.newaxis]
        self._lat = value
        self.set_lat_inlr()
        
    def set_lat_inlr(self):
        for arg in self.args_lat:
            if arg not in self._preproc.get_params().keys():
                raise KeyError('latitude param names not defined')
            kw_arg = self._preproc.get_params()[arg]
            if 'lat' not in kw_arg:
                raise KeyError('lat paramater not present for transform')
            kw_arg['lat'] = self._lat
            self._preproc.set_params(**{'latitude-regress__inv_kw_args':kw_arg})
        
    def check_X(self,X):
        X = np.array(X)
        #if X is scalar
        if X.ndim == 0:
            X = X[np.newaxis]
        
        #if there is only one component, X can be unidimensional
        if X.ndim == 1 and self._pca.components_.shape[0] == 1:
            X = X[:,np.newaxis]
        
        #check if X has the correct number of components
        if not X.shape[1] == self._pca.components_.shape[0]:
            raise ValueError('control variable X has not the good number of components',X.shape)
            
        return(X)
    
    def predict(self,X=None):
        if X is None:
            X = np.zeros((self._lat.size,1))
        X = self.check_X(X)
        y = self._pca.inverse_transform(X)
        y = self._preproc.inverse_transform(y)
        return(y)

#%% Class use
fname = os.path.join(OUTDIR,'preproc'+SVESUFFIX+'.p')
with open(fname,'rb') as f:
    preproc=pickle.load(f)

fname = os.path.join(OUTDIR,'pca'+SVESUFFIX+'.p')
with open(fname,'rb') as f:
    pca = pickle.load(f)

fname = os.path.join(OUTDIR,'depth'+SVESUFFIX+'.p')
with open(fname,'rb') as f:
    depth = pickle.load(f)


prof_rec = profile(preproc=preproc,pca=pca,depth=depth)

#%% Class test

if __name__ == '__main__':
    DATADIR = './data'
    FPREFIX = 'mapped_gamma_all_sources_'
    
    year = 2006
    fname = FPREFIX+str(year)+'.nc'

    data = xr.open_dataset(os.path.join(DATADIR,fname))
    
    #Masking where profile are not complete
    count =  data['gamma'].count(dim='depth')
    mask = count==data['depth'].size

    Xumasked = data['gamma'].stack(geo=('lon','lat'))
    mask1D = mask.stack(geo=('lon','lat'))
    Xmasked = Xumasked.where(mask1D,drop=True)
    Xmasked=Xmasked.transpose('geo','depth')
    
    
    #Set the latitude to determine the mean profile
    prof_rec.lat = Xmasked.lat.values
    
    #Set the control parameter X
    X = np.zeros((Xmasked.lat.size))+4
    gamma_rec_1d = prof_rec.predict(X)
    gamma_rec_2d = xr.DataArray(gamma_rec_1d,Xmasked.coords).unstack('geo')
    
    lat = -15
    lon = 0
    gamma1d = data['gamma'].isel(lat=lat,lon=lon)
    plt.plot(gamma1d,data['gamma'].depth,label='profile')
    plt.plot(gamma_rec_2d.isel(lat=lat,lon=lon),data['gamma'].depth,\
             label='rec (x=0)')
    plt.gca().invert_yaxis()
    plt.title('lat='+str(gamma1d.lat.values)+', lon='+str(gamma1d.lon.values))
    plt.legend()
    plt.show()