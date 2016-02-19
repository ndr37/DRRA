"""
Functions to help automate plotting and fitting of the DRRA routines
"""

import matplotlib.pyplot as plt
import numpy as np
import DRRA
import DRRA.analysisprocedures as ap
from DRRA.constants import *

#######
#STFMR
######

#Convenience functions
def plotscans(autofit,f):
    plt.plot(autofit[f][4][:len(autofit[f][4].Field)//2].Field,ap.model(autofit[f][3][2][0],autofit[f][4][:len(autofit[f][4].Field)//2].Field,f),
         autofit[f][4][:len(autofit[f][4].Field)//2].Field,autofit[f][4][:len(autofit[f][4].Field)//2].X)
    plt.ylabel('Vmix ($\mu$V)')
    plt.xlabel('Field (T)')
    plt.title('Negative field sweep and fit')
    plt.show()
    r=ap.model(autofit[f][3][2][0],autofit[f][4].Field,f)[:len(autofit[f][4].Field)//2]-autofit[f][4][:len(autofit[f][4].Field)//2].X
    plt.hist(r.values,bins=25,histtype='stepfilled')
    plt.title('Fit Residuals')
    plt.show()
    plt.plot(autofit[f][4][len(autofit[f][4].Field)//2:].Field,ap.model(autofit[f][3][1][0],autofit[f][4][len(autofit[f][4].Field)//2:].Field,f),
         autofit[f][4][len(autofit[f][4].Field)//2:].Field,autofit[f][4][len(autofit[f][4].Field)//2:].X)
    plt.ylabel('Vmix ($\mu$V)')
    plt.xlabel('Field (T)')
    plt.title('Positive field sweep and fit')
    plt.show()
    r=ap.model(autofit[f][3][1][0],autofit[f][4][len(autofit[f][4].Field)//2:].Field,f)-autofit[f][4][len(autofit[f][4].Field)//2:].X
    plt.hist(r.values,bins=25,histtype='stepfilled')
    plt.title('Fit Residuals')
    plt.show()
    
def plotscans_overlay(autofit):
    for f in autofit:
        plt.plot(autofit[f][4][:len(autofit[f][4].Field)//2].Field,ap.model(autofit[f][3][2][0],autofit[f][4][:len(autofit[f][4].Field)//2].Field,f),
             autofit[f][4][:len(autofit[f][4].Field)//2].Field,autofit[f][4][:len(autofit[f][4].Field)//2].X)
        
    plt.ylabel('Vmix ($\mu$V)')
    plt.xlabel('Field (T)')
    plt.title('Negative field sweep and fit')
    plt.show()
    for f in autofit:
        plt.plot(autofit[f][4][len(autofit[f][4].Field)//2:].Field,ap.model(autofit[f][3][1][0],autofit[f][4][len(autofit[f][4].Field)//2:].Field,f),
             autofit[f][4][len(autofit[f][4].Field)//2:].Field,autofit[f][4][len(autofit[f][4].Field)//2:].X)
    plt.ylabel('Vmix ($\mu$V)')
    plt.xlabel('Field (T)')
    plt.title('Positive field sweep and fit')
    plt.show()
        
    
def angle_summary(autofit, f,error=0,fdep=False, offset=0):
    print('At frequency %i GHz:' %f)
    print('ThetaxOdd = %f' % autofit[f][5][0])
    print('ThetaxEven = %f' % autofit[f][5][1])
    print('ThetazOdd = %f' % autofit[f][5][2])
    print('ThetazEven = %f' % autofit[f][5][3])
    print('ThetaSHE S/A = %f' % autofit[f][5][5])
    print('ThetaOe(nominal) = %f' % autofit[f][5][4])
    array=[]
    for f in autofit:
        array.append(autofit[f][5][5])
    print('SHE S/A mean =%f' %np.array(array).mean())
    array=[]
    freqs = []
    for f in sorted(autofit.keys()):
        array.append(autofit[f][5][0])
        freqs.append(f)
    print('SHE calc mean =%f' %np.array(array).mean())
    print('SHE calc stdev =%f' %np.std(np.array(array)))
    plt.plot(freqs,np.array(array).mean()*np.ones(len(freqs)),'g--')
    if not fdep:
        plt.errorbar(freqs, array,yerr=error*np.array(array),fmt='o')
    if fdep:
        plt.errorbar(freqs, array,yerr=(error/np.array(freqs)+offset)*np.array(array),fmt='o')
    plt.title('Calibrated SHA vs freq')
    plt.ylabel('SHA')
    plt.xlabel('Frequency (GHz)')
    plt.show()
    return(np.array([np.array(array).mean(),np.std(np.array(array))]))

#######
#DCSTFMR
#######


    #Convenience functions
def plotdict(datadict,fitdict):
    for curr in sorted(datadict.keys()):
        for freq in sorted(datadict[curr].keys()):
            print(('DC current = %f A' % curr,'Frequency = %f GHz' % freq))
            plt.plot(datadict[curr][freq].field,ap.dlor_model(fitdict[curr][freq][0],datadict[curr][6].field),datadict[curr][freq].field,datadict[curr][6].l1x)
            plt.ylabel(' dVmix/dB ($\mu$V)')
            plt.xlabel('Field (T)')
            plt.title('Negative field sweep and fit')
            plt.show()
            plt.hist(ap.dlor_model(fitdict[curr][freq][0],datadict[curr][6].field)-datadict[curr][6].l1x,bins=50,histtype='stepfilled')
            plt.title('Fit Residuals')
            plt.show()
            
def plotdict_overlay(datadict,fitdict,prescale=5000):
    for curr in sorted(datadict.keys()):
        for freq in sorted(datadict[curr].keys()):
            plt.plot(datadict[curr][freq].field,ap.dlor_model(fitdict[curr][freq][0],datadict[curr][6].field)+curr*prescale,datadict[curr][freq].field,datadict[curr][6].l1x+curr*prescale)
            plt.ylabel(' dVmix/dB ($\mu$V)')
            plt.xlabel('Field (T)')
            plt.title('Negative field sweep and fit')
    plt.show()




########
#MOKE
#######

#Convenience functions
def plotMokeTotalFit(totalfit,MOKEdata):
    difyfit = ap.vdifsmooth(MOKEdata.MirrorLine,totalfit['x'][0],totalfit['x'][1],totalfit['x'][3],totalfit['x'][4])
    plt.plot(MOKEdata.MirrorLine,MOKEdata.dif*1e6,MOKEdata.MirrorLine,difyfit, '.',lw=6)
    plt.title('MOKE field swap difference')
    plt.xlabel('Position (Volts)')
    plt.ylabel('Moke response ($\mu$V)')
    plt.show()
    plt.hist(difyfit-MOKEdata.dif*1e6, bins=50, histtype='stepfilled')
    plt.title('Fit Residuals')
    plt.show()
    sumyfit = ap.vsumsmooth_offset(MOKEdata.MirrorLine,totalfit['x'][0],totalfit['x'][1],totalfit['x'][2],totalfit['x'][4],totalfit['x'][5])
    plt.plot(MOKEdata.MirrorLine,MOKEdata.plus*1e6,MOKEdata.MirrorLine,sumyfit, '.',lw=6)
    plt.title('MOKE field swap difference')
    plt.xlabel('Position (Volts)')
    plt.ylabel('Moke response ($\mu$V)')
    plt.show()
    plt.hist(sumyfit-MOKEdata.plus*1e6, bins=50, histtype='stepfilled')
    plt.title('Fit Residuals')
    plt.show()
def area_calcs(totalfit, MOKEdata, Ms,mag_thick, deviceR,activeconductivity):
    difint=ap.dif_area((totalfit['x'][0],totalfit['x'][1],totalfit['x'][3],totalfit['x'][4]),MOKEdata.MirrorLine)
    sumint=ap.sum_area((totalfit['x'][0],totalfit['x'][1],totalfit['x'][2],totalfit['x'][4],totalfit['x'][5]),MOKEdata.MirrorLine)
    prefactor = ap.spin_Hall_prefactor(Ms,mag_thick,deviceR,activeconductivity)
    print('Dif Area = %f'% difint)
    print('Sum Area = %f'% sumint)
    print('Dif/Sum Ratio = %f'% (difint/sumint))
    print('Ratio prefactor = %f'% prefactor)
    print('SHA = %f'% (difint/sumint*prefactor))
    


