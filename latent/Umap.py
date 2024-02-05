from astropy.table import Table
import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sron_colours
sron_rainbow = sron_colours.plt_sron_colours()

from matplotlib.ticker import NullFormatter
nullfmt = NullFormatter()

#With colourbar
double_plt_cb = [1.25*5.5, 2*(5.5*2)/3.0]


import pickle


save = False


train = Table.read('../autoencoder/train_latent.fits')
ttimes = train['time'].value
print(ttimes)
train.remove_column('id')
train.remove_column('time')
tlatents = train.to_pandas().values
print(tlatents)

reducer = umap.UMAP()
trans = reducer.fit(tlatents)

if save:
    reducer = umap.UMAP()
    trans = reducer.fit(tlatents)
    with open('../models/latent/UMAP_embedding.pkl', "wb") as f:
        pickle.dump(trans, f)
        f.close()
else:
    print('loading embedding')
    with open('../models/latent/UMAP_embedding.pkl', "rb") as f:
        trans = pickle.load(f)
        f.close()
print(trans.embedding_.shape)


valid = Table.read('../autoencoder/valid_latent.fits')
vtimes = valid['time'].value
valid.remove_column('id')
valid.remove_column('time')
vlatents = valid.to_pandas().values

vrans = trans.transform(vlatents)
print(vrans.shape)



fig = plt.figure(figsize=double_plt_cb)
ax = [plt.axes([1,1,1,0.5]), plt.axes([1,0.5,1,0.5])]

#training
cbd = ax[0].hexbin(trans.embedding_[:, 0], trans.embedding_[:, 1], ttimes, gridsize=50, cmap='sron_rainbow')

xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()

ax[0].tick_params(axis='both', which='major', labelsize=14)
ax[0].tick_params(axis='x', which='both', direction='in', top=True)
ax[0].tick_params(axis='y', which='both', direction='in', right=True)
ax[0].xaxis.set_major_formatter(nullfmt)
ax[0].set_ylabel(r'y-embedding', fontsize=16)

ax[0].text(-3.2, 9.5, '(a)', fontsize=16)

cb = fig.colorbar(cbd, ax=ax[0], shrink=0.95)
cb.set_label('Average merger time', fontsize=16)
cb.ax.tick_params(labelsize=14)

#validation
cbd = ax[1].hexbin(vrans[:, 0], vrans[:, 1], vtimes, gridsize=50, cmap='sron_rainbow')
ax[1].set_xlim(xlim)
ax[1].set_ylim(ylim)

ax[1].tick_params(axis='both', which='major', labelsize=14)
ax[1].tick_params(axis='x', which='both', direction='in', top=True)
ax[1].tick_params(axis='y', which='both', direction='in', right=True)
ax[1].set_xlabel(r'x-embedding', fontsize=16)
ax[1].set_ylabel(r'y-embedding', fontsize=16)

ax[1].text(-3.2, 9.5, '(b)', fontsize=16)

cb = fig.colorbar(cbd, ax=ax[1], shrink=0.95)
cb.set_label('Average merger time', fontsize=16)
cb.ax.tick_params(labelsize=14)

#ax.set_title('LDA projection of the latent space', fontsize=24)
if save:
    plt.savefig('latent-UMAP.png', bbox_inches='tight')
plt.show()



#plane time
def mse(y_true, y_pred):
    loss = np.mean(np.square(y_true - y_pred))
    return loss

def plane(p1, p2, p3):
    #ax+by+cz=d   
    v1 = p2-p1
    v2 = p3-p1
    cp = np.cross(v1, v2)
    a, b, c = cp
    d = np.dot(cp, p3)
    return a, b, c, d
    
def plane_time(v1, v2, v3, p4):
    a, b, c, d = plane(v1, v2, v3)
    z4 = (d - (a*p4[0]) - (b*p4[1]))/c
    return z4

def separation(a, b):
    dx = np.abs(a[:,0]-b[0])
    dy = np.abs(a[:,1]-b[1])
    dxy = np.sqrt(dx**2 + dy**2)
    return dxy

def sign(p1, p2, p3):
    return (p1[0]-p3[0])*(p2[1]-p3[1])-(p2[0]-p3[0])*(p1[1]-p3[1])

def in_triangle(p1, p2, p3, p4):
    d1 = sign(p4, p1, p2)
    d2 = sign(p4, p2, p3)
    d3 = sign(p4, p3, p1)
    
    if d1 < 0 or d2 < 0 or d3 < 0:
        has_neg = True
    else:
        has_neg = False
    if d1 > 0 or d2 > 0 or d3 > 0:
        has_pos = True
    else:
        has_pos = False
        
    if has_neg and has_pos:
        return False
    else:
        return True


dx = 0.5
dy = dx
pred_time = []

minx = np.min(trans.embedding_[:,0])
maxx = np.max(trans.embedding_[:,0])
miny = np.min(trans.embedding_[:,1])
maxy = np.max(trans.embedding_[:,1])

for i in range(0, len(vrans)):    
    if vrans[i,0] < minx or vrans[i,0] > maxx or vrans[i,1] < miny or vrans[i,1] > maxy:
        pred_time.append(np.nan)
        continue
        
    #get closest 3
    dxy = separation(trans.embedding_, vrans[i])
    ixy = np.argsort(dxy)
    
    best_points = None
    best_dist = 1e10
    
    tpc = int(len(ixy)//10.)
    
    for j in range(0, len(ixy)-2):
        if dxy[ixy[j]] > best_dist:
            break
        test0 = [trans.embedding_[ixy[j]][0], trans.embedding_[ixy[j]][1], ttimes[ixy[j]]]
        for k in range(j+1, len(ixy)-1):
            if dxy[ixy[j]] + dxy[ixy[k]] > best_dist:
                break
            test1 = [trans.embedding_[ixy[k]][0], trans.embedding_[ixy[k]][1], ttimes[ixy[k]]]
            for l in range(k+1, len(ixy)):
                if dxy[ixy[j]] + dxy[ixy[k]] + dxy[ixy[l]] > best_dist:
                    break
                test2 = [trans.embedding_[ixy[l]][0], trans.embedding_[ixy[l]][1], ttimes[ixy[l]]]
                if in_triangle(test0, test1, test2, vrans[i]):
                    if dxy[ixy[j]] + dxy[ixy[k]] + dxy[ixy[l]] < best_dist:
                        best_dist = dxy[ixy[j]] + dxy[ixy[k]] + dxy[ixy[l]]
                        best_points = [np.array(test0), np.array(test1), np.array(test2)]
    
    if best_points is not None:
        pred_time.append(plane_time(best_points[0], best_points[1], best_points[2], vrans[i]))
    else:
        pred_time.append(np.nan)
    
    
print(len(vtimes))
print('mse:', mse(vtimes, pred_time))
plt.scatter(vtimes, pred_time)
plt.plot([-0.01, 1.01],[-0.01, 1.01], c='r')
plt.show()

if save:
    table = Table()
    table['time'] = vtimes
    table['prediction'] = pred_time
    table.write('valid_UMAP.fits', overwrite=True)


