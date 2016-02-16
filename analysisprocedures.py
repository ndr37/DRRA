"""
Analysis software package for the automation of the various spin Hall effect 
measurements done in the Ralph group. Using this package is no excuse for any 
results derived incorrectly, there is no guarantee of quality. The author has 
made some attempt to ensure that sanity checks are enforced on the fitting and
whatnot, and generally trusts the results. YMMV.

-NDR 12/31/15
"""

import numpy as np
import pandas as pd
from scipy import integrate
from scipy import optimize
from scipy import signal
from scipy import fftpack
from scipy.optimize import leastsq
import os
from DRRA import constants
from DRRA.constants import *

#####################################################
# STFMR Methods										#
#													#
#													#
#													#
#													#
#													#
#													#
#####################################################

def s11fromr(rdc):
	"""
	Calculates the S11 parameter based on supplied impedance
	"""
	s11 = 20*np.log10((rdc-50)/(rdc+50))
	return s11

def gamma(s11):
	"""
	Calculates the reflection coefficient from supplied
	"""
	return 10**(s11/20)

def zrf(s11):
	"""
	Calculates the rf impedance based on supplied s11
	"""
	return 50*(1+gamma(s11))/(1-gamma(s11))

def sparam_from_file(dataframe,f):
	"""
	Gives s parameter from a VNA measurement
	dataframe: Must have at least as first two keys: 'F' and 's11'
			with F in GHz
	f: Must be in GHz

	returns: sparam at f, if it exists, otherwise the next smaller f
		for which there is data.
	"""
	if len(dataframe[dataframe.F==f]) is not 0:
		return dataframe[dataframe.F==f].iat[0,1]
	else:
		dif=dataframe.iat[1,0]-dataframe.iat[0,0]
		findex = int((f-dataframe.iat[0,0])/dif)
		return dataframe.iat[findex,1]

def Irf(Psource,s21,s11):
	"""Calculates rf current in the device assuming a the device terminates a lossy, 50 ohm line 
	using the s21 of the line, the power of the source, and the s11 of the device
	""" 	
	irf = np.sqrt(.001*10**((Psource+s21)/10)*(1-abs(gamma(s11))**2)/zrf(s11))
	return irf*np.sqrt(2)*np.sqrt(2)  # to get from average current to peak, to account for AM mod


def xfac_calc(r2,rtot):
	"""
	Assuming two parallel resistor model of a bilayer, gives the resistance of the resistor
	whose values contributes to rtot and is in parallel with value r2.
	"""
	return r2/(rtot/(r2-rtot)+r2)


def thetaguess(dataframe,f,alphaguess=0.01,Meffguess=1,sigmaguess=0.1):
	"""
	Creates reasonably sensical guess for the values of parameters ('theta') that may vary for the
	STFMR fit assuming your magnet is sputtered Py. 
	"""
	cxguess = dataframe.X.values.argmax()*Meffguess*alphaguess
	czguess = dataframe.X.values.argmax()*Meffguess*alphaguess
	offsetguess = dataframe.mean(0)[1]
	return np.array([alphaguess,Meffguess,cxguess,czguess,offsetguess,sigmaguess])


def STFMR_analyze(fit_data,Psource,s21,s11,fieldangle,xfac,drdth,mag_thick,bar_width,active_thick):
	
	"""
	input:(fit_data,Psource,s21,s11,fieldangle,xfac,drdth,mag_thick,bar_width,active_thick)
	fit_data:alpha,Meff,Cxeven,Cxodd,Czeven,Czodd
	output:
	return np.array([thetaxeven,thetaxodd,thetazeven,thetazodd,thetaOe,Cxeven/Czeven*thetaOe,Irf(Psource,s21,s11)])

	"""
	alpha,Meff,Cxeven,Cxodd,Czeven,Czodd = fit_data[0]
	thetaOe = echarge*mu0*Meff/mu0*mag_thick*active_thick/hbar
	scale_factor=(4*echarge*Meff/mu0*mag_thick*bar_width*active_thick)/(
		gam*hbar*np.cos(fieldangle*np.pi/180)*Irf(Psource,s21,s11)**2*xfac*drdth)*(
		1e-6) *np.sqrt(2) * 2  #uV to V, sqrt(2) to go from lockin rms to amplitude

	thetaxeven = -Cxeven*scale_factor
	thetaxodd = -Cxodd*scale_factor
	thetazeven = -Czeven*scale_factor
	thetazodd = -Czodd*scale_factor
	return np.array([thetaxeven,thetaxodd,thetazeven,thetazodd,thetaOe,Cxeven/Czeven*thetaOe,Irf(Psource,s21,s11)])

def omega0(B,Meff): #Meff in Amps/Meter
	"""
	Returns resonant frequence for in plane field with inplane Ms in Radians/s
	"""
	return gam*np.sqrt(abs(B)*(abs(B)+Meff))

def SymLor(B,f,alpha,Meff,Cx):
	w = 2*np.pi*f
	return Cx*(gam*np.square(w)*(alpha*(2*abs(B)+Meff)))/(np.square(np.square(w)-np.square(omega0(abs(B),Meff)))+np.square(alpha*gam*w)*np.square(2*abs(B)+Meff))

def ASymLor(B,f,alpha,Meff,Cz):
	w = 2*np.pi*f
	return Cz*(np.power(gam,3)*abs(B)*np.square(abs(B)+Meff)-gam*np.square(w)*(abs(B)+Meff))/(np.square(w*w-np.square(omega0(abs(B),Meff)))+np.square(alpha*gam*w)*np.square(2*abs(B)+Meff))

def TotalLor(B,f,alpha,Meff,Cx,Cz,offset):
	return SymLor(B,f,alpha,Meff,Cx)+ASymLor(B,f,alpha,Meff,Cz)+offset

def model(theta, B, f):
	"""
	A function with the appropriate form to be used in scipy.optimize style fitting computation
	"""
	alpha,Meff,Cx,Cz,offset = theta
	return TotalLor(B,f,alpha,Meff,Cx,Cz,offset)

def stfmr_residual(theta, B, f, data):
	"""
	Function that can actually be put into the Scipy.optimize
	"""
	if theta[0]<0:
		return np.inf
	if abs(theta[4])>data.mean()+100:
		return np.array([10000 for item in data])
	return data - model(theta, B, f)

def stfmr_residual_lnprob(theta, B, f, data,priors_window = 10):
	"""
	log probability of gaussian distribution of our data with our model
	"""
	theta_fraction = theta*priors_window
	i=0
	#while i <len(theta):
	#	if abs(theta[i])-abs(theta_fraction[i])<abs(theta[i])<abs(theta[i])+abs(theta_fraction[i]):
	#		return -np.inf
	#	i+=1
	return -1/2*np.sum((data - model(theta, B, f))**2)

def amr_model(theta,angles):
	amp,phase,offset = theta
	return offset+amp*np.cos(angles*np.pi/180+phase)**2

def amr_residual(theta,angles,data):
	return data - amr_model(theta,angles)

def amr_guess(dataframe):
	amp = (dataframe.R.argmax()-dataframe.R.argmin())/2
	phase = 0
	offset = dataframe.mean(0)[1]
	return np.array([amp,phase,offset])

def fitamr(dataframe,theta):
	lstsqramr = leastsq(amr_residual, theta,
						args=(dataframe.Angles,dataframe.R),maxfev=10000,
						full_output = 1)
	popt, copt, _,_,_ =lstsqramr
	return popt,copt

def fitSTFMRscan(dataframe,f,theta):
	positive=dataframe[dataframe.Field>0]
	#positive = positive[:-200]
	negative=dataframe[dataframe.Field<0]
	#negative = negative[:-200]

	lsqpos = leastsq(stfmr_residual, theta[:-1], 
					args=(positive.Field, f, positive.X), maxfev=10000,
					full_output = 1, )
	lsqneg = leastsq(stfmr_residual, theta[:-1], 
					args=(negative.Field, f, negative.X), maxfev=10000,
					full_output = 1, )
	poptpos, coptpos, _,_,_ =lsqpos
	poptneg,coptneg,_,_,_ = lsqneg
	alpha_mean = (poptpos[0]+poptneg[0])/2
	Meff_mean = (poptpos[1]+poptneg[1])/2

	cxodd = (poptpos[2]-poptneg[2])/2
	cxeven = (poptpos[2]+poptneg[2])/2
	czodd = (poptpos[3]-poptneg[3])/2
	czeven =  (poptpos[3]+poptneg[3])/2
	return (np.array([alpha_mean,Meff_mean,cxodd,cxeven,czodd,czeven]),lsqpos,lsqneg)

def directory_auto_fit_fieldwrong(_tf_dir,_sample_dir,df_s21,_xfactor,_magnetthickness,_barwidth,_activethickness):
	"""
	Fits a whole director provided you took s11 data and RF scans using Utilsweep and Daedalus in the
	standard way.
	"""

	_tfdir=_tf_dir+'/'+ _sample_dir
	_tflist=os.listdir(_tfdir)
	print(_tflist[0])
	if _tflist[0] == '.DS_Store':
		_tflist=_tflist[1:]


	dfamr=pd.read_csv(_tfdir+'/'+_tflist[0],sep='\t')
	dfamr=pd.DataFrame({'Angles':dfamr.Azimuthnominal,'R':dfamr.Resistance})
	amrval=fitamr(dfamr,amr_guess(dfamr))[0][0]
	print('AMR done')

	dftfs11=pd.read_csv(_tfdir+'/'+_tflist[len(_tflist)-1],sep='\t')
	dftfs11=pd.DataFrame({'F':dftfs11['Frequency (Hz)']*1e-9,'s11':dftfs11['Trace Value']})
	print('S11 loaded')

	sampledict={}
	i=0
	for item in _tflist[1:len(_tflist)-1]:
		angle = float(item.split('_')[1])
		power = float(item.split('_')[5])
		if abs(power)>30:
			power = 0
		freq = float(item.split('_')[7])
		dftfcurrent=pd.read_csv(_tfdir+'/'+item,sep='\t')
		dftfcurrent=pd.DataFrame({'Field':dftfcurrent['Field(nominal)'].values*2,'X':dftfcurrent.LockinOnex.values*1e6})
		fitdata=fitSTFMRscan(dftfcurrent,freq,thetaguess(dftfcurrent,freq))
		stfmr=STFMR_analyze(fitdata,power,sparam_from_file(df_s21,freq),sparam_from_file(dftfs11,freq),angle,_xfactor,amrval,_magnetthickness,_barwidth,_activethickness)
		sampledict.update({freq:(angle,power,freq,fitdata,dftfcurrent,stfmr,amrval,zrf(sparam_from_file(dftfs11,freq)))})
		print(freq, end = '\r')
		i=i+1
	print()
	return sampledict

def directory_auto_fit(_tf_dir,_sample_dir,df_s21,_xfactor,_magnetthickness,_barwidth,_activethickness,force_45=False):
	"""
	Fits a whole director provided you took s11 data and RF scans using Utilsweep and Daedalus in the
	standard way.
	"""

	_tfdir=_tf_dir+'/'+ _sample_dir
	_tflist=os.listdir(_tfdir)
	if _tflist[0] == '.DS_Store':
		_tflist=_tflist[1:]


	dfamr=pd.read_csv(_tfdir+'/'+_tflist[0],sep='\t')
	dfamr=pd.DataFrame({'Angles':dfamr.Azimuthnominal,'R':dfamr.Resistance})
	amrval=fitamr(dfamr,amr_guess(dfamr))[0][0]
	print('AMR done')

	dftfs11=pd.read_csv(_tfdir+'/'+_tflist[len(_tflist)-1],sep='\t')
	dftfs11=pd.DataFrame({'F':dftfs11['Frequency (Hz)']*1e-9,'s11':dftfs11['Trace Value']})
	print('S11 loaded')

	sampledict={}
	i=0
	for item in _tflist[1:len(_tflist)-1]:
		angle = float(item.split('_')[1])
		if force_45:
			angle = 45
		power = float(item.split('_')[5])
		if abs(power)>30:
			power = 0
		freq = float(item.split('_')[7])
		dftfcurrent=pd.read_csv(_tfdir+'/'+item,sep='\t')
		dftfcurrent=pd.DataFrame({'Field':dftfcurrent['Field(nominal)'].values,'X':dftfcurrent.LockinOnex.values*1e6})
		fitdata=fitSTFMRscan(dftfcurrent,freq,thetaguess(dftfcurrent,freq))
		stfmr=STFMR_analyze(fitdata,power,sparam_from_file(df_s21,freq),sparam_from_file(dftfs11,freq),angle,_xfactor,amrval,_magnetthickness,_barwidth,_activethickness)
		sampledict.update({freq:(angle,power,freq,fitdata,dftfcurrent,stfmr,amrval,zrf(sparam_from_file(dftfs11,freq)))})
		print(freq, end = '\r')
		i=i+1
	print()
	return sampledict

#####################################################
# Field mod DSTFMR Methods							#
#													#
#													#
#													#
#													#
#													#
#													#
#####################################################

def dfdB_model(theta,B,f):
	w=2*np.pi*f
	alpha,Meff,Cx,Cz,offset = theta
	return offset+(-2*gam**2*(2*abs(B) + Meff)*(abs(B)*gam**2*(abs(B) + Meff) + 4*(-1 + 2*alpha**2)*f**2*np.pi**2)*(abs(B)*Cz*gam**3*(abs(B) + Meff)**2 + 4*f**2*gam*(-(Cz*(abs(B) + Meff)) + alpha*Cx*(2*abs(B) + Meff))*np.pi**2) + (Cz*gam**3*(abs(B) + Meff)*(3*abs(B) + Meff) - 4*(-2*alpha*Cx + Cz)*f**2*gam*np.pi**2)*(4*alpha**2*f**2*gam**2*(2*abs(B) + Meff)**2*np.pi**2 + (abs(B)*gam**2*(abs(B) + Meff) - 4*f**2*np.pi**2)**2))/(4*alpha**2*f**2*gam**2*(2*abs(B) + Meff)**2*np.pi**2 + (abs(B)*gam**2*(abs(B) + Meff) - 4*f**2*np.pi**2)**2)**2

def dlor_model(theta,field):
	delta,A,S,B0,offset,osc = theta
	B=abs(field)
	#return (6*A*(B - B0)**2 - 2*A*delta**2 - 2*(B - B0)*((B - B0)**2 + delta**2)*S)/((B - B0)**2 + delta**2)**3 +offset+osc*1/(B)
	return (delta*(A*(-(B - B0)**2 + (delta/2)**2) + 2*(-B + B0)*(delta/2)*S))/((B - B0)**2 + (delta/2)**2)**2 +offset+ osc*1/(B)

def dlor_residual(theta, B, data):
	delta,A,S,B0,offset,osc = theta
	if delta <0 or B0<0:
		return np.ones(len(B))*1e10
	return data-dlor_model(theta,B)

def dlor_fit(theta,B,data,**kwargs):
	fitdata = leastsq(dlor_residual, theta, 
					args=(B, data), maxfev=10000,
					**kwargs)
	return fitdata

def dfdb_residual(theta, B, data):
	alpha,Meff,Cx,Cz,offset = theta
	if alpha <0 or Meff<0:
		return np.ones(len(B))*1e10
	return data-dfdB_model(theta,B,6)


def dfdb_fit(theta,B,data,**kwargs):
	fitdata = leastsq(dfdb_residual, theta, 
					args=(B, data), maxfev=10000,
					**kwargs)
	return fitdata

def fit_datadict(datadict,**kwargs):
	"""
	Fit a dictionary of trimmed data with key structure {Curr:{freq:data}}
	"""
	fitdict={};
	for curr in sorted(datadict.keys()):
		if not curr in fitdict:
			fitdict[curr]={}
		for f in sorted(datadict[curr].keys()):
			tempfit=dlor_fit((0.01,1,1,abs((datadict[curr][f].field)[datadict[curr][f].l1x.argmin()]),datadict[curr][f].l1x.mean(),1),
				datadict[curr][f].field,datadict[curr][f].l1x,**kwargs)
			fitdict[curr][f]=tempfit
		print(curr, end = '\r')
	return fitdict

def fitfull_datadict(datadict,**kwargs):
	"""
	Fit a dictionary of trimmed data with key structure {Curr:{freq:data}}
	"""
	fitdict={};
	for curr in sorted(datadict.keys()):
		if not curr in fitdict:
			fitdict[curr]={}
		for f in sorted(datadict[curr].keys()):
			tempfit=dfdb_fit((0.01,1,1,1,1),
				datadict[curr][f].field,datadict[curr][f].l1x,**kwargs)
			fitdict[curr][f]=tempfit
		print(curr, end = '\r')
	return fitdict


def current_prefactor(width, active_thick, mag_thick, xfac, Meff, B0, angle):
	return (width*active_thick/xfac)*(2*constants.echarge/constants.hbar)*(Meff/mu0*mag_thick*(B0+Meff/2))/np.sin(angle*np.pi/180)


def dfdB_with_i_model(theta_wi,B,f,i,inc_offset = False):
	w=2*np.pi*f
	alpha,Meff,Cx,Cz,hdc,k,_ = theta_wi
	if inc_offset:
		alpha,Meff,Cx,Cz,hdc,k, offset = theta_wi
		return offset+(-2*gam*(gam**3*(B + hdc*i)*(B + hdc*i + Meff)*(2*B + 2*hdc*i + Meff) + 4*f**2*(2*(-1 + 2*alpha**2)*B*gam + 4*alpha*i*k + (-1 + 2*alpha**2)*gam*(2*hdc*i + Meff))*np.pi**2)*(Cz*gam**3*(B + hdc*i)*(B + hdc*i + Meff)**2 - 4*f**2*gam*(Cz*(B + hdc*i + Meff) - alpha*Cx*(2*B + 2*hdc*i + Meff))*np.pi**2)+ (Cz*gam**3*(B + hdc*i + Meff)*(3*B + 3*hdc*i + Meff) - 4*(-2*alpha*Cx + Cz)*f**2*gam*np.pi**2)*(16*alpha*f**2*gam*i*k*(2*B + 2*hdc*i + Meff)*np.pi**2 + 4*alpha**2*f**2*gam**2*(2*B + 2*hdc*i + Meff)**2*np.pi**2 + (gam**2*(B + hdc*i)*(B + hdc*i + Meff) - 4*f**2*np.pi**2)**2))/(16*alpha*f**2*gam*i*k*(2*B + 2*hdc*i + Meff)*np.pi**2 + 4*alpha**2*f**2*gam**2*(2*B + 2*hdc*i + Meff)**2*np.pi**2 + (gam**2*(B + hdc*i)*(B + hdc*i + Meff) - 4*f**2*np.pi**2)**2)**2
	return (-2*gam*(gam**3*(B + hdc*i)*(B + hdc*i + Meff)*(2*B + 2*hdc*i + Meff) + 4*f**2*(2*(-1 + 2*alpha**2)*B*gam + 4*alpha*i*k + (-1 + 2*alpha**2)*gam*(2*hdc*i + Meff))*np.pi**2)*(Cz*gam**3*(B + hdc*i)*(B + hdc*i + Meff)**2 - 4*f**2*gam*(Cz*(B + hdc*i + Meff) - alpha*Cx*(2*B + 2*hdc*i + Meff))*np.pi**2)+ (Cz*gam**3*(B + hdc*i + Meff)*(3*B + 3*hdc*i + Meff) - 4*(-2*alpha*Cx + Cz)*f**2*gam*np.pi**2)*(16*alpha*f**2*gam*i*k*(2*B + 2*hdc*i + Meff)*np.pi**2 + 4*alpha**2*f**2*gam**2*(2*B + 2*hdc*i + Meff)**2*np.pi**2 + (gam**2*(B + hdc*i)*(B + hdc*i + Meff) - 4*f**2*np.pi**2)**2))/(16*alpha*f**2*gam*i*k*(2*B + 2*hdc*i + Meff)*np.pi**2 + 4*alpha**2*f**2*gam**2*(2*B + 2*hdc*i + Meff)**2*np.pi**2 + (gam**2*(B + hdc*i)*(B + hdc*i + Meff) - 4*f**2*np.pi**2)**2)**2

def dfdB_with_i_bare(B,i,f,alpha,Meff,hdc,k,Cx,Cz,offset):
	return offset + (-2*gam*(gam**3*(B + hdc*i)*(B + hdc*i + Meff)*(2*B + 2*hdc*i + Meff) + 4*f**2*(2*(-1 + 2*alpha**2)*B*gam + 4*alpha*i*k + (-1 + 2*alpha**2)*gam*(2*hdc*i + Meff))*np.pi**2)*(Cz*gam**3*(B + hdc*i)*(B + hdc*i + Meff)**2 - 4*f**2*gam*(Cz*(B + hdc*i + Meff) - alpha*Cx*(2*B + 2*hdc*i + Meff))*np.pi**2)+ (Cz*gam**3*(B + hdc*i + Meff)*(3*B + 3*hdc*i + Meff) - 4*(-2*alpha*Cx + Cz)*f**2*gam*np.pi**2)*(16*alpha*f**2*gam*i*k*(2*B + 2*hdc*i + Meff)*np.pi**2 + 4*alpha**2*f**2*gam**2*(2*B + 2*hdc*i + Meff)**2*np.pi**2 + (gam**2*(B + hdc*i)*(B + hdc*i + Meff) - 4*f**2*np.pi**2)**2))/(16*alpha*f**2*gam*i*k*(2*B + 2*hdc*i + Meff)*np.pi**2 + 4*alpha**2*f**2*gam**2*(2*B + 2*hdc*i + Meff)**2*np.pi**2 + (gam**2*(B + hdc*i)*(B + hdc*i + Meff) - 4*f**2*np.pi**2)**2)**2


def deriv_stfmr_residual(theta, B, f, data):
	"""
	Function that can actually be put into the Scipy.optimize
	"""
	if theta[0]<0:
		return np.inf
	return data - dfdB_model(theta, B, f)

def deriv_stfmr_with_i_residual(theta_wi, B, f,i, data):
	"""
	Function that can actually be put into the Scipy.optimize
	"""
	if theta_wi[0]<0 or theta_wi[1]<0 or theta_wi[1]>2:# or abs(theta_wi[5])>2:
		return np.inf
	return data - dfdB_with_i_model(theta_wi, B, f, i,inc_offset=True)


def fitDSTFMRscan(dataframe,f,i,theta_wi):
	positive=dataframe[dataframe.field>0.01]
	posmin= positive.l1x.argmin()
	posupper = posmin+30
	poslower = posmin-30
	positive = dataframe[poslower:posupper]
	#print(positive)
	#print('positive')
	negative=dataframe[dataframe.field<-0.01]
	negmin= negative.l1x.argmin()
	negative = dataframe[negmin-30:negmin+30]
	#print(negative)

	lsqpos = leastsq(deriv_stfmr_with_i_residual, theta_wi, 
					args=(positive.field, f,i, positive.l1x*10**6), maxfev=10000,
					full_output = 1, )
	lsqneg = leastsq(deriv_stfmr_with_i_residual, theta_wi, 
					args=(abs(negative.field), f,i, negative.l1x*10**6), maxfev=10000,
					full_output = 1, )
	poptpos, coptpos, _,_,_ =lsqpos
	poptneg,coptneg,_,_,_ = lsqneg
	alpha_pos =poptpos[0]
	alpha_neg =poptneg[0]
	Meff_neg = poptneg[1]
	Meff_pos = poptpos[1]
	hdcpos = poptpos[4]
	hdcneg = poptneg[4]
	kpos = poptpos[5]
	kneg = poptneg[5]

	cxeven = (poptpos[2]-poptneg[2])/2
	cxodd = (poptpos[2]+poptneg[2])/2
	czeven = (poptpos[3]-poptneg[3])/2
	czodd =  (poptpos[3]+poptneg[3])/2
	return (np.array([alpha_pos,alpha_neg,Meff_pos,Meff_neg,hdcpos,hdcneg,kpos,kneg,cxeven,cxodd,czeven,czodd]),lsqpos,lsqneg)   


   
#####################################################
# MOKE Methods a la Xin Fan/A. Mellnik				#
#													#
#													#
#													#
#													#
#													#
#													#
#####################################################

def vsumexact(x, w, center, csum):
    """
    Returns a numpy array the length of x which produces an ideal curve of the Oersted field induced
    out of plane rotation of the magnetization in the MOKE device for a given width, center, and amplitude
    """

    v=[]
    for element in x:
        if (center-w/2)<=element<=(center+w/2):
            v.append(csum/w*(np.log((w/2+(element-center))**2)-np.log((w/2-(element-center))**2)))
        else:
            v.append(0.)
    return np.array(v)

def vsumexact_offset(x, w, center, csum, offset):
	"""
    Returns a numpy array the length of x which produces an ideal curve of the Oersted field induced
    out of plane rotation of the magnetization in the MOKE device for a given width, center, amplitude,
    and potentially a dc offset
    """
	v=[]
	if np.shape(x)==():
		if (center-w/2)<=x<=(center+w/2):
			v = csum/w*(np.log((w/2+(x-center))**2)-np.log((w/2-(x-center))**2))+offset
		else:
			v=0
		return v
	else:
		for element in x:
			if (center-w/2)<=element<=(center+w/2):
				v.append(csum/w*(np.log((w/2+(element-center))**2)-np.log((w/2-(element-center))**2))+offset)
			else:
				v.append(0.)
		return np.array(v)

def vdifexact(x, w, center, cdif):
	"""
    Returns a numpy array the length of x which produces an ideal curve of the spin Hall induced
    out of plane rotation of the magnetization in the MOKE device for a given width, center, and amplitude
    """
	v=[]
	if np.shape(x) ==():
		if (center-w/2)<=x<=(center+w/2):
			v=cdif
		else:
			v=0
		return v
	else:
		for element in x:
			if (center-w/2)<=element<=(center+w/2):
				v.append(cdif)
			else:
				v.append(0.)
		return np.array(v)

def vdifsmooth(x, w, center, cdif, delta,spacing=0.01):
	exact=vdifexact(x,w,center,cdif)
	if np.shape(x) == ():
		gaussian = signal.gaussian(10*delta/spacing,delta/spacing)/(delta/spacing*np.sqrt(2*np.pi))
	else:
		gaussian = signal.gaussian(10*delta/(x.values[1]-x[0]),delta/((x.values[1]-x[0])))/(delta/(x.values[1]-x[0])*np.sqrt(2*np.pi))
	return signal.fftconvolve(exact,gaussian,mode='same')

def vdifsmooth_residual(theta,x,data):
	w, center, cdif, delta = theta
	temp = data - vdifsmooth(x,w,center,cdif,delta)
	return temp

def vdifsmooth_residual_scalar(theta,x,data):
	w, center, cdif, delta = theta
	return np.sum((data - vdifsmooth(x,w,center,cdif,delta))**2)

def beamprofile(x,amp,x0,delta):
	gaussian = amp/(delta*np.sqrt(2*np.pi))*np.exp(-(x-x0)**2/(2*delta**2))
	return gaussian

def kernel_nint(xp,x,w,center,csum,delta):
	convolutionkernel = vsumexact_nint(x-xp,w,center, csum)*beamprofile(xp,delta)
	return convolutionkernel

def vsumsmooth_nint(x, w, center, csum, delta):
	convolved = []
	for data in x:
		tempint = integrate.quad(kernel,-np.inf,np.inf,args=(data,w,center,csum,delta),limit=1000)[0]
		if tempint == np.inf or tempint == -np.inf:
			print(data)
		convolved.append(tempint)
	return convolved

def vsumsmooth(x, w, center, csum, delta):
	exact=vsumexact(x,w,center,csum)
	gaussian = signal.gaussian(10*delta/(x.values[1]-x[0]),delta/((x.values[1]-x[0])))/(delta/(x.values[1]-x[0])*np.sqrt(2*np.pi))
	return signal.fftconvolve(exact,gaussian,mode='same')

def vsumsmooth_offset(x, w, center, csum, delta, offset,spacing = 0.01):
	exact=vsumexact_offset(x,w,center,csum,offset)
	if np.shape(x)==():
		gaussian = signal.gaussian(10*delta/spacing,delta/spacing)/(delta/spacing*np.sqrt(2*np.pi))
	else:
		gaussian = signal.gaussian(10*delta/(x.values[1]-x[0]),delta/((x.values[1]-x[0])))/(delta/(x.values[1]-x[0])*np.sqrt(2*np.pi))
	return signal.fftconvolve(exact,gaussian,mode='same')

def vsumsmooth_residual(theta,x,data):
	w, center, csum, delta = theta
	temp = data - vsumsmooth(x,w,center,csum,delta)
	return temp

def vsumsmooth_offset_residual(theta,x,data):
	w, center, csum, delta, offset = theta
	temp = data - vsumsmooth_offset(x,w,center,csum,delta,offset)
	return temp


def vsumsmooth_residual_scalar(theta,x,data):
	w, center, csum, delta = theta
	return np.sum((data - vsumsmooth(x,w,center,csum,delta))**2)

def vsumsmooth_offset_residual_scalar(theta,x,data):
	w, center, csum, delta, offset = theta
	return np.sum((data - vsumsmooth_offset(x,w,center,csum,delta,offset))**2)

def mokesimul_offset_residual_scalar(theta,x,datasum,datadif):
	w, center, csum, cdif, delta, offset = theta
	vsum_resid = datasum - vsumsmooth_offset(x,w,center,csum,delta,offset)
	vdif_resid = datadif - vdifsmooth(x,w,center,cdif,delta)  
	return np.sum(vsum_resid**2) + np.sum(vdif_resid**2)

def fit_moke_simul_bruteforce(thetarange,x,datasum,datadif):
	"""
	Fit both plus and dif traces simultaneously using a brute force algorithm
	to ensure global minimum is probably found.
	"""
	#sometimes throws an operand broadcast error. Decreasing range on one parameter
	#seemed to fix it
	fitdata = diffit=optimize.differential_evolution(mokesimul_offset_residual_scalar,
		thetarange,
		args=(x, datasum, datadif),)
	return fitdata

def sum_area(theta,x,zeroed=True):
	"""
	Returns the area under the curve traced out by the ideal MOKE sum curve using
	simpsons rule.
	"""
	w, center, csum, delta, offset = theta
	if zeroed:
		offset = 0.0
	return integrate.simps(np.abs(vsumsmooth_offset(x,w,center,csum,delta,0)),x=x)

def dif_area(theta,x):
	"""
	Returns the area under the curve traced out by the ideal MOKE difference curve using
	simpsons rule.
	"""
	w, center, cdif, delta = theta
	return integrate.simps(np.abs(vdifsmooth(x,w,center,cdif,delta)),x=x)

def spin_Hall_prefactor(muMs, mag_thick, R_device, active_conductivity):
	return 2*constants.echarge/constants.hbar*muMs*mag_thick*np.log(4)/(2*np.pi*(R_device*60/50)*active_conductivity)

    