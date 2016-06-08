"""

This is an attempt at automating data formats common among Ralph Group Instruments.
Currently supported are UtilSweep, UtilMOKE, PPMSsweep, and PPMS (QD) file formats
All methods take a file of a given type and return a pandas dataframe. Instancing 
the dataloader class is meant to remove the hassle of having to append the directory
each time.

"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from DRRA import constants



class dataloader:
	
	def __init__(self, workingdir = None):
		if workingdir is None:
			pass
		else:
			self.wkdir = workingdir



	def load_tab_delimited(self, filename,header='infer',names=None):
		
		temp = pd.read_csv(self.wkdir + filename, sep = '\t',header=header, names=None)
		
		return temp

	def load_QD(self, filename):# rdata = False):
		"""
		Uses code prototype gleaned from http://gb119.github.io/Stoner-PythonCode/_modules/Stoner/FileFormats.html for loading QD file
		formats. Empirically, Pandas has trouble with QD files so one needs to go more primitive with numpy.
		Returns a data frame with the Field, Moment, and Temperature data.  
		"""
		temp = np.genfromtxt(self.wkdir+filename, dtype = 'float', delimiter = ',', invalid_raise = False, skip_header = 23)
		field = []
		moment = []
		temperature = []

		for item in temp:
			field.append(item[3])
			moment.append(item[4])
			temperature.append(item[2])
		df = pd.DataFrame({'B':np.array(field),'M':np.array(moment),'T':np.array(temperature)})

		return df

	def load_UtilSweep(self, filename,verbose = False):
		"""
		Uses load_tab_delimited method to load data from UtilSweep. Returns pandas DataFrame
		with shortened keys for easy df.KEY access. Code should be smart enough to handle any 
		number of selected instruments. Currently only up to two lockins are supported
		"""
		temp = self.load_tab_delimited(filename)
		keys = temp.keys()
		utildict = {}
		for item in keys:
			if item == 'LockinOnex':
				utildict['l1x'] = temp[item].values
			if item == 'LockinOney':
				utildict['l1y'] = temp[item].values
			if item == 'LockinAnotherOnex':
				utildict['l2x'] = temp[item].values
			if item == 'LockinAnotherOney':
				utildict['l2y'] = temp[item].values
			if item == 'Field(nominal)':
				utildict['field'] = temp[item].values
			if item == 'Azimuthnominal':
				utildict['azimuth'] = temp[item].values
			if item == 'PolarNominal':
				utildict['polar'] = temp[item].values
		return pd.DataFrame(utildict)

	def load_PPMSsweep(self, filename):
		"""
		Uses load_tab_delimited method to load data from PPMSSweep. Returns pandas DataFrame
		with shortened keys for easy df.KEY access. Code should be smart enough to handle any 
		number of selected instruments. Currently only up to two lockins are supported
		"""
		temp = self.load_tab_delimited(filename)
		keys = temp.keys()
		ppmsdict = {}
		for item in keys:
			if item == 'Lockin1x Volts':
				utildict['l1x'] = temp[item].values
			if item == 'Lockin1y Volts':
				utildict['l1y'] = temp[item].values
			if item == 'Lockin2x Volts':
				utildict['l2x'] = temp[item].values
			if item == 'Lockin2y Volts':
				utildict['l2y'] = temp[item].values
			if item == 'Field (Oe)':
				utildict['field'] = temp[item].values
			if item == 'Temp (K)':
				utildict['temp'] = temp[item].values
			if item == 'Position (Degrees)':
				utildict['position'] = temp[item].values
		return pd.DataFrame(ppmsdict)

	def load_UtilMOKE(self, filename, channel=1,shift=0):
		"""
		Uses load_tab_delimited method to load all the standard instruments from a normal SHE MOKE
		experiment. The output is a dataframe that contains the Mirrorline values and the sum and
		difference as plus and dif. 

		Channel kwarg allows selection of which SR 7270 channel to load as data, X1 by default.
		"""
		moketemp = self.load_tab_delimited(filename)
		i=0
		templine=[]
		tempx1=[]
		tempx2=[]
		tempdc = []
		while i<len(moketemp.DC_volts):
			if i<len(moketemp.DC_volts)//2:
				templine.append(moketemp.MirrorLine.iloc[i])
				tempx1.append(moketemp.X1.iloc[i])
				tempx2.append(moketemp.X2.iloc[i])
				tempdc.append(moketemp.DC_volts.iloc[i])
			if i>=len(moketemp.DC_volts)//2:
				if not (i-shift)>=len(moketemp.DC_volts)//2:
					templine.append(moketemp.MirrorLine.iloc[i])
					tempx1.append(moketemp.X1.iloc[i])
					tempx2.append(moketemp.X2.iloc[i])
					tempdc.append(moketemp.DC_volts.iloc[i])
				else:
					templine.append(moketemp.MirrorLine.iloc[i-shift])
					tempx1.append(moketemp.X1.iloc[i-shift])
					tempx2.append(moketemp.X2.iloc[i-shift])
					tempdc.append(moketemp.DC_volts.iloc[i-shift])
			i+=1
		plt.plot(templine,moketemp.DC_volts)
		plt.show()
		
		temp = pd.DataFrame({'MirrorLine':templine,'X1':tempx1,'X2':tempx2,'Field':moketemp.Field.values,'DC':tempdc})
		if channel == 1:
			temp = pd.concat([temp.MirrorLine[:-1],temp.X1,temp.Field,temp.DC],axis=1)
		else:
			temp = pd.concat([temp.MirrorLine[:-1],temp.X2,temp.Field,temp.DC],axis=1)
		tempplus = temp[temp['Field'].isin([temp.Field[0]])]
		tempdif = temp[temp['Field'].isin([temp.Field[len(temp)-1]])]
		i=0
		dif = np.array([2]*len(tempdif),dtype='float64')
		while i<len(tempdif):
			if channel == 1:
				dif[i]=tempdif.X1.iloc[i]-tempplus.X1.iloc[i]
			else:
				dif[i]=tempdif.X2.iloc[i]-tempplus.X2.iloc[i]
			i+=1
		plus = np.array([2]*len(tempplus),dtype='float64')
		i=0
		while i<len(tempplus):
			if channel ==1:
				plus[i]=tempdif.X1.iloc[i]+tempplus.X1.iloc[i]
			else:
				plus[i]=tempdif.X2.iloc[i]+tempplus.X2.iloc[i]
			i+=1
		temp = pd.DataFrame({'MirrorLine':tempplus.MirrorLine,'plus':plus, 'dif':dif,'DC':tempplus.DC.values})
		return temp

	def load_DC_Bias_UtilSweep_neg(self,max=False,lowerbound = 100,upperbound = 100,points_from_zero = 30):
		"""
		Uses load_UtilSweep method to load data from a dc biased stfmr experiment, keeping only the negative fields
		and triming the data to a region presumably only around the resonance.
		Returns pandas DataFrame with shortened keys for easy df.KEY access. 
		"""
		i=0
		alldatadict = {}
		for item in (os.listdir(self.wkdir)):
			if item.split('_')[0] == 'azimuth':
				nparray = self.load_UtilSweep(item)[self.load_UtilSweep(item).field<0]
				curr = float(item.split('_')[9])
				freq = float(item.split('_')[7])
				if max:
					negmax = nparray[:-points_from_zero].l1x.argmax()
				if not max:
					negmax = nparray[:-points_from_zero].l1x.argmin()
				upper = negmax+upperbound
				if upper>len(nparray):
					upper = len(nparray)-1
				lower = negmax-lowerbound
				if lower<0:
					lower = 0
				nparray = nparray[lower:upper]
				nparray = pd.DataFrame({'l1x':nparray.l1x.values*1e6,'field':nparray.field.values})
				if not curr in alldatadict:
					alldatadict[curr]={}
				alldatadict[curr][freq]=nparray
				i=i+1
		return alldatadict

	def load_DC_Bias_UtilSweep_pos(self,max=False,lowerbound = 100,upperbound = 100,points_from_zero = 30):
		"""
		Uses load_UtilSweep method to load data from a dc biased stfmr experiment, keeping only the positive fields
		and triming the data to a region presumably only around the resonance.
		Returns pandas DataFrame with shortened keys for easy df.KEY access. 
		"""
		i=0
		alldatadict = {}
		for item in (os.listdir(self.wkdir)):
			if item.split('_')[0] == 'azimuth':
				nparray = self.load_UtilSweep(item)[self.load_UtilSweep(item).field>0]
				curr = float(item.split('_')[9])
				freq = float(item.split('_')[7])
				nparray = nparray.reset_index()
				if max:
					negmax = nparray[:-points_from_zero].l1x.argmax()
				if not max:
					negmax = nparray[:-points_from_zero].l1x.argmin()
				upper = negmax+upperbound
				if upper>len(nparray):
					upper = len(nparray)-1
				lower = negmax-lowerbound
				if lower<0:
					lower = 0
				nparray = nparray[lower:upper]
				nparray = pd.DataFrame({'l1x':nparray.l1x.values*1e6,'field':nparray.field.values})
				if not curr in alldatadict:
					alldatadict[curr]={}
				alldatadict[curr][freq]=nparray
				i=i+1
		return alldatadict



