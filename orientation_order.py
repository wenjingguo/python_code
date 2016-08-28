__author__ = 'wenjingg'

# coding: utf-8

# In[1]:

# Purpose :  Compute in-plane oxygen-oxygen, hydrogen-hydrogen, oxygen-hydrogen radial distribtion functions
# the thickness of the slab is 1 in all cases,
# when binding the pair distances, we take into account periodic boundary condition.


# In[2]:

import sys
import pandas as pd
import numpy as np
from operator import sub
from  math import sqrt
import  matplotlib.pyplot as plt
from numpy.linalg import norm
import glob
from scipy.spatial import KDTree
from numpy import (array, dot, arccos, clip)
from scipy.integrate import cumtrapz
import re
import os
import seaborn as sns



Lx = 34.58
Ly = 29.947158
a = Lx
b = Ly

def get_number_from_filename(filename):
	return re.search(r'\d+',filename).group(0)


def xyz_parser(path):
	df = pd.read_csv(path, delim_whitespace= True, skiprows =1, names=['symbol','x','y','z'],)
	return df

#selection = df[(df['z'] > 14.53 )& (df['z'] < 15.5)]
#df = selection

#selection = twobody[(twobody['symbols'] == ''.join([A,B])) | (twobody['symbols'] == ''.join([B,A]))]

def map_x_to_y(x,y):
	mapped = np.empty((len(x), ), dtype= np.int)

	for i, index in enumerate(x):

		mapped[i] = y[index]

	return mapped

def create_unit(df,a,b):
	unitdf = df.copy()
	cell_dim = np.array([a,b])
	unitdf[['x','y']] = np.mod(unitdf[['x','y']], cell_dim)
	return unitdf

def superframe(frame,a,b):
	v = [-1,0,1]
	unit = frame[['x','y','z']].values
	n = len(unit)
	coords = np.empty((n*9,3))
	h = 0
	for i in v :
		for j in v :
			for l in range(n):
				coords[h,0] =  unit[l,0] + i * a
				coords[h,1] =  unit[l,1] + j * b
				coords[h,2] =  unit[l,2]
				h = h + 1
	return coords

#dipole orientation of 3-site water:
def dipolevector(df):
	H = df.loc[df['symbol']=='H', ['x','y','z']].values
	O = df.loc[df['symbol']=='O', ['x','y','z']].values
	if len(H) == 2 and len(O) == 1:
		vector = np.sum((np.tile(O,(2,1)) - H), axis = 0)
		norm = np.linalg.norm(vector)
		unit_vector = vector/norm
		return unit_vector
	else:
		return None

def OHvector(df):
	H = df.loc[df['symbol']=='H', ['x','y','z']].values
	O = df.loc[df['symbol']=='O', ['x','y','z']].values
	if len(H) != 0 and len(O) != 0 :
		vector = np.tile(O,(len(df)-1,1)) - H
		return vector
	else:
		return None

def angle_between(v1,v2):
	angle = np.clip(np.dot(v1,v2),-1,1)
	return np.arccos(angle)*180/np.pi, angle

def histogram(hist, dr=4, start=0, end=180):
	bins = np.arange(start,end,dr)
	bins = np.append(bins, bins[-1] + dr)
	r = (bins[1:] + bins[:-1]) / 2
	hist, bins = np.histogram(hist, bins)
	hist = hist/hist.sum()

	return pd.DataFrame.from_dict({'r':r,'hist':hist})

m = len(glob.glob(os.path.join(os.getcwd(), 'xyz*.xyz')))
s = np.empty((m))
m = 0
for file in glob.glob(os.path.join(os.getcwd(), 'xyz*.xyz')):

	df = xyz_parser(file)
	refvector = [1,0,0]
	index1 = df[(df['z'] > 2.5)&(df['symbol'] == 'O')].index.tolist()[0]
	index2 = df[(df['z'] > 3.5)&(df['symbol'] == 'O')].index.tolist()[0]
	df = df[index1:index2]

	nat = len(df)
	if nat == 0:
		pass
	else:
		unit_frame = create_unit(df,a,b)
		big_frame = superframe(unit_frame,a,b)
		kd = KDTree(big_frame)
		k = nat
		symbols=[]
		dipole_vector = np.empty((nat,3))
		dipole_angle = np.empty((nat))
		cosdipole_angle = np.empty((nat))

		OH_vector = np.empty((2*nat,3))
		OH_angle = np.empty((2*nat))
		cosOH_angle = np.empty((2*nat))

		h = 0
		k = 0
		for i in range(nat):
			index = kd.query_ball_point(unit_frame[['x', 'y', 'z']].iloc[i], r = 1.001)
			if len(index) == 1:
				pass
			else:
				xyz = big_frame[index]
				unitcell = pd.DataFrame(xyz, columns =list('xyz'))
				unitcell['symbol'] = (unit_frame['symbol'].iloc[np.mod(index,nat)]).reset_index(drop=True)
				unitcell['charge'] = np.where(unitcell['symbol'] == 'O',-0.8476,0.4238)
				unitcell['mass'] = np.where(unitcell['symbol'] == 'O',15.999,1.008)
				dipole_vector[h,:] =  dipolevector(unitcell)
				dipole_angle[h], cosdipole_angle[h] = angle_between(dipole_vector[h,:],refvector)
				if OHvector(unitcell) is None:
					pass
				elif len(OHvector(unitcell)) == 1:
					OH_vector[k,:] = OHvector(unitcell)
					OH_angle[k], cosOH_angle[k] = angle_between(OH_vector[k,:],refvector)
					k =k +1
				else:
					for j in range(len(OHvector(unitcell))):
						OH_vector[k,:] = OHvector(unitcell)[j]
						OH_angle[k], cosOH_angle[k] = angle_between(OH_vector[k,:],refvector)
						k = k +1
				h = h +1
		dipole_vector =  dipole_vector[:h]
		OH_angle =  OH_angle[:k]
		cosOH_angle =  cosOH_angle[:k]
		dipole_angle = dipole_angle[:h]
		cosdipole_angle = cosdipole_angle[:h]

	df1 = histogram(cosdipole_angle)
	df2 = histogram(cosOH_angle)

	df = pd.concat((df1,df2), axis = 1)
#	print(df)
	df.columns = ['dipole angle','r','OH angle','r1']
	del df['r1']
	df.set_index('r', inplace= True)
	df.to_csv(os.path.join('angle'+ get_number_from_filename(file) +'.csv'), sep= '\t', mode='w')


allfiles = glob.glob(os.path.join(os.getcwd(),'angle*.csv'))

dfnew = pd.concat([pd.read_csv(f, sep='\t') for f in allfiles], axis= 1, keys = [f for f in allfiles])

dfnew = dfnew.swaplevel(0, 1, axis=1).sortlevel(axis=1)
df = dfnew.groupby(level=0,axis = 1).mean()
df.to_csv(os.path.join('angle'+'.csv'), sep= '\t', mode='w')

plt.plot(df['r'], df['dipole angle'],'r')
plt.plot(df['r'],df['OH angle'],'b')
plt.xlim(-90,180)
plt.show()



















