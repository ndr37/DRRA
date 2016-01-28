"""

This is an attempt at automating data formats common among Ralph Group Instruments.
Currently supported are UtilSweep, UtilMOKE, PPMSsweep, and PPMS (QD) file formats
All methods take a file of a given type and return a pandas dataframe. Instancing 
the dataloader class is meant to remove the hassle of having to append the directory
each time.

"""

import numpy as np
import pandas as pd
from DRRA import constants



class dataloader:
	
	def __init__(self, workingdir = None):
		if workingdir is None:
			pass
		else:
			self.wkdir = workingdir



	def load_tab_delimited(self, filename):
		
		temp = pd.read_csv(self.wkdir + filename, sep = '\t')
		
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

	def load_UtilMOKE(self, filename):
		"""
		Uses load_tab_delimited method to load all the standard instruments from a normal SHE MOKE
		experiment. The output is a dataframe that contains the Mirrorline values and the sum and
		difference as plus and dif.
		"""
		temp = self.load_tab_delimited(filename)
		temp = pd.concat([temp.MirrorLine[:-1],temp.X1,temp.Field],axis=1)
		tempplus = temp[temp['Field'].isin([temp.Field[0]])]
		tempdif = temp[temp['Field'].isin([temp.Field[len(temp)-1]])]
		i=0
		dif = np.array([2]*len(tempdif),dtype='float64')
		while i<len(tempdif):
			dif[i]=tempdif.X1.iloc[i]-tempplus.X1.iloc[i]
			i+=1
		plus = np.array([2]*len(tempplus),dtype='float64')
		i=0
		while i<len(tempplus):
			plus[i]=tempdif.X1.iloc[i]+tempplus.X1.iloc[i]
			i+=1
		temp = pd.DataFrame({'MirrorLine':tempplus.MirrorLine,'plus':plus, 'dif':dif})
		return temp


