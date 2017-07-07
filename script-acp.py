# -*- coding: utf-8 -*-
"""
Spyder Editor
ANalyse density profiles
Pauthenet E., Roquet F., Madec G., Nerini D., 2017. A linear decomposition of 
the Southern Ocean thermohaline structure. Journal of Physical Oceanography, 
doi: 10.1175/JPO-D-16-0083.1
This is a temporary script file.
"""

# %% Init
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import xarray as xr
import matplotlib.pyplot as plt
DATADIR = '/net/argos/data/peps/cchlod/ARGO_DATA/TempSalintyGamma'

FPREFIX = 'mapped_gamma_all_sources_'
YEARS = {2006}

# %% Load
year = YEARS.pop()
fname = FPREFIX+str(year)+'.nc'
data = xr.open_dataset(os.path.join(DATADIR,fname))

count =  data['gamma'].count(dim='depth')
mask = count==data['depth'].size

Xumasked = data['gamma'].stack(geo=('lon','lat'))
mask1D = mask.stack(geo=('lon','lat'))
Xmasked = Xumasked.where(mask1D,drop=True)
Xmasked=Xmasked.transpose('geo','depth')


#%% PCA

#Standardization
dimgeo = 0
if not Xmasked.dims[dimgeo] == 'geo':
    raise ValueError('dim not correct')
scaler = StandardScaler().fit(Xmasked)
X = scaler.transform(Xmasked)


pca = PCA(n_components=4)
pca.fit(X)

#%%plot PCA
plt.plot(scaler.mean_,data['gamma'].depth)
plt.gca().invert_yaxis()
plt.ylabel('depth[m]')
plt.xlabel('density')
plt.show()

plt.bar(range(len(pca.explained_variance_ratio_)),pca.explained_variance_ratio_)
plt.show()

plt.plot(pca.components_[0,:],data['gamma'].depth)
plt.show()

#%% Plots
gamma1d = data['gamma'].isel(lat=100,lon=30)
gamma1d.plot()
plt.show()

gamma2d = data['gamma'].isel(depth=0)
gamma2d.plot()
plt.show()