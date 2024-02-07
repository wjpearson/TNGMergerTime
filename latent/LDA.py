import numpy as np
from astropy.table import Table
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt

import sron_colours
sron_rainbow = sron_colours.plt_sron_colours()

from matplotlib.ticker import NullFormatter
nullfmt = NullFormatter()

#With colourbar
double_plt_cb = [1.25*5.5, 2*(5.5*2)/3.0]


import pickle


save = False


table = Table.read('../autoencoder/train_latent.fits')
ttimes = table['time'].value
print(ttimes)
table.remove_column('id')
table.remove_column('time')
latents = table.to_pandas().values
print(latents)

if save:
    embedding = LDA(n_components=2)
    trans = embedding.fit_transform(latents, ttimes)
    with open('../models/latent/LDA_embedding.pkl', "wb") as f:
        pickle.dump(embedding, f)
        f.close()
else:
    with open('../models/latent/LDA_embedding.pkl', "rb") as f:
        embedding = pickle.load(f)
        f.close()
    trans = embedding.transform(latents)
print(trans.shape)


valid = Table.read('../autoencoder/valid_latent.fits')
vtimes = valid['time'].value
valid.remove_column('id')
valid.remove_column('time')
vlatents = valid.to_pandas().values

vrans = embedding.transform(vlatents)
print(vrans.shape)



fig = plt.figure(figsize=double_plt_cb)
ax = [plt.axes([1,1,1,0.5]), plt.axes([1,0.5,1,0.5])]

#training
cbd = ax[0].hexbin(trans[:, 0], trans[:, 1], ttimes, gridsize=50, cmap='sron_rainbow')

xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()

ax[0].tick_params(axis='both', which='major', labelsize=14)
ax[0].tick_params(axis='x', which='both', direction='in', top=True)
ax[0].tick_params(axis='y', which='both', direction='in', right=True)
ax[0].xaxis.set_major_formatter(nullfmt)
ax[0].set_ylabel(r'y-embedding', fontsize=16)

ax[0].text(-5.8, 9.1, '(a)', fontsize=16)

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

ax[1].text(-5.8, 9.1, '(b)', fontsize=16)

cb = fig.colorbar(cbd, ax=ax[1], shrink=0.95)
cb.set_label('Average merger time', fontsize=16)
cb.ax.tick_params(labelsize=14)

#ax.set_title('LDA projection of the latent space', fontsize=24)
if save:
    plt.savefig('latent-LDA.png', bbox_inches='tight')
plt.show()



def separation(a, b):
    dx = abs(a[0]-b[0])
    dy = abs(a[1]-b[1])
    dxy = np.sqrt(dx**2 + dy**2)
    return dxy

def mse(y_true, y_pred):
    loss = np.mean(np.square(y_true - y_pred))
    return loss


def poly1(a, b, x):
    return (a*x)+b

def poly2(a, b, c, x):
    return (a*x*x)+(b*x)+c

def poly3(a, b, c, d, x):
    return (a*x*x*x)+(b*x*x)+(c*x)+d

dx=0.01
x = np.arange(-5, 5+dx, dx)

#Using LDA x-axis
plt.scatter(trans[:,0], ttimes)
a1, b1 = np.polyfit(trans[:,0], ttimes, 1)
plt.plot(x, poly1(a1, b1, x), 'r')
a2, b2, c2 = np.polyfit(trans[:,0], ttimes, 2)
plt.plot(x, poly2(a2, b2, c2, x), 'c')
a3, b3, c3, d3 = np.polyfit(trans[:,0], ttimes, 3)
plt.plot(x, poly3(a3, b3, c3, d3, x), 'm')
plt.show()

if save:
    import pickle
    with open('LDA_xaxis_embedding.pkl', "wb") as f:
        pickle.dump({'a1':a1, 'b1':b1,
                     'a2':a2, 'b2':b2, 'c2':c2,
                     'a3':a3, 'b3':b3, 'c3':c3, 'd3':d3},
                    f)
        f.close()

plt.scatter(vrans[:,0], vtimes)
plt.plot(x, poly1(a1, b1, x), 'r')
plt.plot(x, poly2(a2, b2, c2, x), 'c')
plt.plot(x, poly3(a3, b3, c3, d3, x), 'm')
plt.show()


loss1_t = mse(ttimes,  poly1(a1, b1, trans[:,0]))
plt.title('train 1: '+str(loss1_t))
plt.scatter(ttimes,  poly1(a1, b1, trans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

if save:
    table = Table()
    table['time'] = ttimes
    table['prediction'] = poly1(a1, b1, trans[:,0])
    table.write('train_LDA_xaxis.fits', overwrite=True)

loss1_v = mse(vtimes,  poly1(a1, b1, vrans[:,0]))
plt.title('valid 1: '+str(loss1_v))
plt.scatter(vtimes,  poly1(a1, b1, vrans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

if save:
    table = Table()
    table['time'] = vtimes
    table['prediction'] = poly1(a1, b1, vrans[:,0])
    table.write('valid_LDA_xaxis.fits', overwrite=True)

loss2_t = mse(ttimes,  poly2(a2, b2, c2, trans[:,0]))
plt.title('train 2: '+str(loss2_t))
plt.scatter(ttimes,  poly2(a2, b2, c2, trans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

loss2_v = mse(vtimes,  poly2(a2, b2, c2, vrans[:,0]))
plt.title('valid 2: '+str(loss2_v))
plt.scatter(vtimes,  poly2(a2, b2, c2, vrans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

loss3_t = mse(ttimes,  poly3(a3, b3, c3, d3, trans[:,0]))
plt.title('train 3: '+str(loss3_t))
plt.scatter(ttimes,  poly3(a3, b3, c3, d3, trans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

loss3_v = mse(vtimes,  poly3(a3, b3, c3, d3, vrans[:,0]))
plt.title('valid 3: '+str(loss3_v))
plt.scatter(vtimes,  poly3(a3, b3, c3, d3, vrans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()


#Using LDA x-axis
x = np.arange(0, 1+dx, dx)

plt.scatter(ttimes, trans[:,0])
a1, b1 = np.polyfit(ttimes, trans[:,0], 1)
plt.plot(x, poly1(a1, b1, x), 'r')
a2, b2, c2 = np.polyfit(ttimes, trans[:,0], 2)
plt.plot(x, poly2(a2, b2, c2, x), 'c')
a3, b3, c3, d3 = np.polyfit(ttimes, trans[:,0], 3)
plt.plot(x, poly3(a3, b3, c3, d3, x), 'm')
plt.show()

plt.scatter(vtimes, vrans[:,0])
plt.plot(x, poly1(a1, b1, x), 'r')
plt.plot(x, poly2(a2, b2, c2, x), 'c')
plt.plot(x, poly3(a3, b3, c3, d3, x), 'm')
plt.show()


def invpoly1(a, b, y):
    return (y-b)/a

def invpoly2(a, b, c, y):
    return ((-1*b) + np.sqrt( (b*b) - (4*a*(c-y)) ) ) / (2*a)

def invpoly3(a, b, c, d, x):
    return (a*x*x*x)+(b*x*x)+(c*x)+d

loss1_t = mse(ttimes,  invpoly1(a1, b1, trans[:,0]))
plt.title('train 1: '+str(loss1_t))
plt.scatter(ttimes,  invpoly1(a1, b1, trans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

loss1_v = mse(vtimes,  invpoly1(a1, b1, vrans[:,0]))
plt.title('valid 1: '+str(loss1_v))
plt.scatter(vtimes,  invpoly1(a1, b1, vrans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

loss2_t = mse(ttimes,  invpoly2(a2, b2, c2, trans[:,0]))
plt.title('train 2: '+str(loss2_t))
plt.scatter(ttimes,  invpoly2(a2, b2, c2, trans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

loss2_v = mse(vtimes,  invpoly2(a2, b2, c2, vrans[:,0]))
plt.title('valid 2: '+str(loss2_v))
plt.scatter(vtimes,  invpoly2(a2, b2, c2, vrans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()


x10 = poly1(a1, b1, 0)
x20 = poly2(a2, b2, c2, 0)
x30 = poly3(a3, b3, c3, d3, 0)
x11 = poly1(a1, b1, 1)
x21 = poly2(a2, b2, c2, 1)
x31 = poly3(a3, b3, c3, d3, 1)

print(x10, x11)

def scale(x0, x1, y):
    return (y-x0)/(x1-x0)

loss1_t = mse(ttimes,  scale(x10, x11, trans[:,0]))
plt.title('train 1: '+str(loss1_t))
plt.scatter(ttimes, scale(x10, x11, trans[:,0]))
plt.plot([0,1],[0,1],'r')
plt.show()

loss1_v = mse(vtimes,  scale(x10, x11, vrans[:,0]))
plt.title('valid 1: '+str(loss1_v))
plt.scatter(vtimes, scale(x10, x11, vrans[:,0]))
plt.plot([0,1],[0,1],'r')
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

minx = np.min(trans[:,0])
maxx = np.max(trans[:,0])
miny = np.min(trans[:,1])
maxy = np.max(trans[:,1])

for i in range(0, len(vrans)):
    print(i)
    
    if vrans[i,0] < minx or vrans[i,0] > maxx or vrans[i,1] < miny or vrans[i,1] > maxy:
        pred_time.append(np.nan)
        continue
        
    #get closest 3
    dxy = separation(trans, vrans[i])
    ixy = np.argsort(dxy)
    
    best_points = None
    best_dist = 1e10
    
    for j in range(0, len(ixy)-2):
        if dxy[ixy[j]] > best_dist:
            break
        test0 = [trans[ixy[j]][0], trans[ixy[j]][1], ttimes[ixy[j]]]
        for k in range(j+1, len(ixy)-1):
            if dxy[ixy[j]] + dxy[ixy[k]] > best_dist:
                break
            test1 = [trans[ixy[k]][0], trans[ixy[k]][1], ttimes[ixy[k]]]
            for l in range(k+1, len(ixy)):
                if dxy[ixy[j]] + dxy[ixy[k]] + dxy[ixy[l]] > best_dist:
                    break
                test2 = [trans[ixy[l]][0], trans[ixy[l]][1], ttimes[ixy[l]]]
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
    table.write('valid_LDA.fits', overwrite=True)


