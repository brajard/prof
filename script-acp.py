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
from sklearn.linear_model import LinearRegression
import xarray as xr
import numpy as np
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
#DATADIR = '/net/argos/data/peps/cchlod/ARGO_DATA/TempSalintyGamma'
DATADIR = './data'
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


#%% basemap plot
def map_plot(XX):
    #X should be a xarray containing lon and lat
    lon_0 = 0
    boundinglat = float(XX.lat.max())
    XX=XX.transpose('lat','lon')
    m = Basemap(projection='spstere',boundinglat=boundinglat,lon_0=lon_0,resolution='l')
    xx,yy = np.meshgrid(XX.lon,XX.lat)
    x,y = m(xx,yy)
    m.contourf(x,y,XX.values)

    plt.colorbar()
 #   m.drawmapboundary(fill_color='aqua')
    m.drawcoastlines()
    m.fillcontinents(color='coral',lake_color='aqua')
# draw parallels and meridians.
    m.drawparallels(np.arange(-80.,81.,20.))
    m.drawmeridians(np.arange(-180.,181.,20.))

   

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

plt.title('mean density profile')
plt.show()


x = range(1,len(pca.explained_variance_ratio_)+1)
plt.bar(x,pca.explained_variance_ratio_)
plt.plot(x,pca.explained_variance_ratio_.cumsum(),'r.-')
plt.xticks(x)
plt.xlabel('eigen value')
plt.ylabel('explained variance')
plt.show()

plt.plot(pca.components_[0,:],data['gamma'].depth)
plt.gca().invert_yaxis()
plt.title('First PCA component')
plt.xlabel('component value')
plt.ylabel('depth[m]')

#%%geoplot
icomp = 0
proj = pca.transform(X)
proj2 = xr.DataArray(proj[:,icomp],coords=Xmasked['geo'].coords)
map_plot(proj2.unstack('geo').transpose('lat','lon'))
plt.title('PCA component number '+ str(icomp+1))


#%% reconstruct
XX=dict()
components = [1,2,3,10]

for n_components in components:

    pca1 = PCA(n_components=n_components)
    pca1.fit(X)
    proj1 = pca1.transform(X)
    Xrec = pca1.inverse_transform(proj1)
    Xrec = scaler.inverse_transform(Xrec)
    
    XX[n_components] = xr.DataArray(Xrec,Xmasked.coords).unstack('geo')



#%% Plots
n_components = 1

#North profile
lat = -35
lon = 0
gamma1d = data['gamma'].isel(lat=lat,lon=lon)
plt.plot(gamma1d,data['gamma'].depth,label='profile')
plt.plot(XX[n_components].isel(lat=lat,lon=lon),data['gamma'].depth,\
         label='rec['+str(n_components)+']')
plt.plot(scaler.mean_,data['gamma'].depth,label='mean')
plt.gca().invert_yaxis()
plt.title('lat='+str(gamma1d.lat.values)+', lon='+str(gamma1d.lon.values))
plt.legend()
plt.show()

#South profile
lat = -80
lon = 0
gamma1d = data['gamma'].isel(lat=lat,lon=lon)
plt.plot(gamma1d,data['gamma'].depth,label='profile')
plt.plot(scaler.mean_,data['gamma'].depth,label='mean')
plt.plot(XX[n_components].isel(lat=lat,lon=lon),data['gamma'].depth,label='reconstruct')
plt.gca().invert_yaxis()
plt.title('lat='+str(gamma1d.lat.values)+', lon='+str(gamma1d.lon.values))
plt.legend()
plt.show()

#%% regress on lat
lr = LinearRegression()

#predictor for linear regression
Xlin = Xmasked['lat'].values[:,np.newaxis]

#Xlin = np.concatenate([Xmasked['lat'].values[:,np.newaxis],\
#                       Xmasked['lon'].values[:,np.newaxis]],axis=1)


lr.fit(Xlin,proj[:,0].squeeze())

#%% plot results of regression   
n = 20000
I = np.random.permutation(proj.shape[0])
I = I[:n]
xy = np.vstack([Xmasked['lat'][I],proj[I,0].squeeze()])
z = gaussian_kde(xy)(xy)

#regression
yl = lr.predict(Xlin)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = Xmasked['lat'][I][idx], proj[I,0][idx], z[idx]
#plt.plot(Xmasked['lat'],proj1,'.')
plt.scatter(x,y,c=z,s=10,edgecolor='')
plt.plot(Xlin[:,0],yl,'k-')
plt.xlabel('latitude')
plt.ylabel('PCA component number 1')
R2 = lr.score(Xlin,proj[:,0].squeeze())
plt.title('R2 = ' + '{:04.3f}'.format(R2))
plt.show()

#gamma2d = data['gamma'].isel(depth=0)
#gamma2d.plot()
#plt.show()

#%% latitude component determination
