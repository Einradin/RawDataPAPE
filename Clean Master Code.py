#!/usr/bin/env python
# coding: utf-8

# # Master Thesis Code

# ## Packages needed to run code

# In[1]:


#!conda install -c anaconda git
#!pip install git+https://github.com/hildensia/bayesian_changepoint_detection.git
#!pip install git clone https://github.com/GallVp/emgGO
#!pip install neurokit2
#!pip install pip install python-magic-bin==0.4.14
#!pip install biosignalsnotebooks
#!pip install pingouin


# ## Libraries needed to import

# In[2]:


import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
from scipy.signal import freqz, filtfilt
from scipy import fftpack
from tkinter import*
import scipy
import seaborn
from __future__ import division
import cProfile
import sys
import bayesian_changepoint_detection.offline_changepoint_detection as offcd
from functools import partial
import neurokit2 as nk
from scipy.integrate import cumtrapz
from scipy.signal import welch
from numpy import asarray
from numpy import savetxt
from tempfile import TemporaryFile
import statistics
import biosignalsnotebooks as bsnb
import scipy.stats as stats
from scipy.signal import argrelextrema
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pickle


# ## Import Excel Files and Fuctions to Read them

# In[3]:


# import pape-e files from S1 to S 11
def papreader():
    d = {} #creates dictionaary
    x = 1
    
    for i in range(1,14):
        # initiates dicitonary
        # "xlfile{0}".format(i) --> string xlfile with value of i
        try:
            d["xlfile{0}".format(i)] = pd.ExcelFile(r"D:\ExcelJoeFisch\PAP S"+str(x)+".xlsx")
            x = x+1
        except:
            x = x+1
    return d


# import pape-e files from S1 to S 11
def papeereader():
    d = {} #creates dictionaary
    x1 = 1
    
    for i1 in range(1,14):
        # initiates dicitonary
        # "xlfile{0}".format(i) --> string xlfile with value of i
        try:
            d["xlfile{0}".format(i1)] = pd.ExcelFile(r"D:\ExcelJoeFisch\PAPEE S"+str(x1)+".xlsx")
            x1 = x1+1
        except:
            x1 = x1+1
    return d


# import pape-v files from S1 to S 11
def papevreader():
    d = {} #creates dictionaary
    x2 = 1
    for i2 in range(1,14):
        # initiates dicitonary
        # xlfile{0}".format(i) --> string xlfile with value of i
        try:
            d["xlfile{0}".format(i2)] = pd.ExcelFile(r"D:\ExcelJoeFisch\PAPEV S"+str(x2)+".xlsx")
            x2 = x2+1
        except:
            x2 = x2+1
    return d

# following fuctions store all exelfiles of the specified trials (PAP, PAPEE, PAPEV)
d_pap=papreader()
d_papee=papeereader()
d_papev=papevreader()

# to read specific data of a specified excelfile use:
# data = pd.read_excel(trial[file], sheet_name=sheet)
# trial = d_pap; d_papee; d_papev
# file = "xlfile1 - xlfile11" (subject 1 to 11)
# sheet = MVIC; pre_stim; MVIC_4m;... (sheets of specified excelfile)


# In[ ]:





# ## Visualize Raw Data

# In[5]:


def visualize(trial, file, sheet, singal): 
    
    #signal can be Sampple, Angle, Torque, Stim, RF, VM, VL
    
    excelfile = pd.read_excel(trial[file], sheet_name=sheet)
    plt.figure(1, figsize=(30, 10))
    
    if singal == "Torque":
        plt.plot(excelfile.SAMPLE/1000, excelfile.Torque)
        plt.xlabel("Time in s")
        plt.ylabel("Torque in Nm")
    elif singal == "Angle":
        plt.plot(excelfile.SAMPLE/1000, excelfile.Angle)
        plt.xlabel("Time in s")
        plt.ylabel("Angle in degrees") 
    elif singal == "Sample":
        plt.plot(excelfile.SAMPLE/1000, excelfile.Sample)
        plt.xlabel("Time in s")
        plt.ylabel("Sample in Hz")
    elif singal == "Stim":
        plt.plot(excelfile.SAMPLE/1000, excelfile.Stim)
        plt.xlabel("Time in s")
        plt.ylabel("Stim in mV")   
    elif singal == "RF":
        plt.plot(excelfile.SAMPLE/1000, excelfile.RF)
        plt.xlabel("Time in s")
        plt.ylabel("EMG RF in mV")   
    elif singal == "VM":
        plt.plot(excelfile.SAMPLE/1000, excelfile.VM)
        plt.xlabel("Time in s")
        plt.ylabel("EMG VM in mV")   
    elif singal == "VL":
        plt.plot(excelfile.SAMPLE/1000, excelfile.VL)
        plt.xlabel("Time in s")
        plt.ylabel("EMG VL in mV")   
    else: print("Check if signal in you excelfile is named correctly")
        
# exaple how to use function 
visualize(d_pap, "xlfile11", "MVIC_10s", "Torque")


# ## Filtering and Calculation of different Parameters

# In[6]:


## Filtering EMG & FFT

def calculation(cutoff_low, cutoff_high, fs, order, trial, file, sheet):
    
    
    excelfile = pd.read_excel(trial[file], sheet_name=sheet)
    data_rf = np.array(excelfile.RF)
    data_vm = np.array(excelfile.VM)
    data_vl = np.array(excelfile.VL)
    data_torque = np.array(excelfile.Torque)
    
    
    #smooth Torque with RMS
    
    #Torque
    #Filtering (RMS)
    
    rms_window = 50 #how much smoothing (smaller = less smoothing)
    rms_sample = []
    rms_i = 0
    
    data_torque_power = np.power(data_torque,2) # makes new array with same dimensions as a but squared
    
    
    window = np.ones(rms_window)/float(rms_window) #produces an array or length window_size where each element is 1/window_size
    
    rms_torque = np.sqrt(np.convolve(data_torque_power, window, 'valid'))
    
    while len(rms_sample) < len(rms_torque):
        rms_sample.append(rms_i/1000)
        rms_i = rms_i+1
  

    #calculate lowpass RF
    
    nyq = 0.5 * fs #nyquist
    normal_cutoff = cutoff_low / nyq #for normalization reasons
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    #apply outcome to data
    
    y_rf = filtfilt(b, a, data_rf)
    y_vm = filtfilt(b, a, data_vm)
    y_vl = filtfilt(b, a, data_vl)
    
    #calculate highpass RF
    
    nyq = 0.5 * fs
    normal_cutoff1 = cutoff_high / nyq
    b1, a1 = butter(order, normal_cutoff1, btype='high', analog=False)
    
    #apply outcome to data
    
    y1_rf = filtfilt(b1, a1, y_rf)
    y1_vm = filtfilt(b1, a1, y_vm)
    y1_vl = filtfilt(b1, a1, y_vl)
    
    #apply smoothing (envelope lowpass) and rectify data (abs)
    #this part of the code hasn't been adjusted to all muscle output yet
    '''
    y2 = abs(y1)
    lowp_smooth= 10
    i= 1
    sample = []
    
    

    nyq = 0.5 * fs #nyquist
    normal_cutoff2 = lowp_smooth / nyq #for normalization reasons
    b2, a2 = butter(order, normal_cutoff2, btype='low', analog=False)
    
    y3 = filtfilt(b2, a2, y2)
    
    while len(sample) < len(y3):
        sample.append(i/1000)
        i = i+1
    '''
    #smooth RF with RMS 
    
    rms_window_emg = 250 #how much smoothing (smaller = less smoothing)
    sample = []
    rms_i_emg = 0
    
    data_rfrms_power = np.power(data_rf,2) # makes new array with same dimensions as a but squared
    data_vmrms_power = np.power(data_vm,2)
    data_vlrms_power = np.power(data_vl,2)
    
    window_emgrms = np.ones(rms_window_emg)/float(rms_window_emg) #produces an array or length window_size where each element is 1/window_size
    
    y3_rf = np.sqrt(np.convolve(data_rfrms_power, window_emgrms, 'valid'))
    y3_vm = np.sqrt(np.convolve(data_vmrms_power, window_emgrms, 'valid'))
    y3_vl = np.sqrt(np.convolve(data_vlrms_power, window_emgrms, 'valid'))
    
    while len(sample) < len(y3_rf):
        sample.append(rms_i_emg/1000)
        rms_i_emg = rms_i_emg+1
    
    
    #calculate Muscle Onset and Offset based on emg
    
    '''
    #calculates maximum value
    md =  np.max(y3)
    #calculates first time the data reaches 5% of max value
    md1 = np.argmax(y3>0.08*md)
    #calculates mean of the beginning until 5% of max value
    mdmean = np.mean(y3[:md1])
    #calculates standard deviation of the beginning until 5% of max value
    md2 = np.std(y3[:md1])
    #calculates first time where data is bigger than calculated mean at rest + 2x the standard deviation (muscle onset)
    md3 = np.argmax(y3>mdmean+(10*md2))
    #calculates first time data is smaller than mean at rest + 2x the standard deviation (+1000 is added to be sure that value comes after musce onset)
    md4 = np.argmax(y3[md3+1000:]<=mdmean+(5*md2))
    #calculates muscle offset (data from 0 to onset (md3) is added to data from md3 to md4 plus 1000 to account for calculation that was done in md4)
    md5 = md3+md4+1000
    '''
    
    #calculate Muscle Onset and Offset based on Torque
    sheet_exception1 = "eStim+MVIC"
    sheet_exception2 = "80%+MVIC"
    sheet_exception3 = "MVIC_10s"
    sheet1 = sheet
    
    if sheet1 == sheet_exception1:
        try:
            #calculates maximum value
            md =  np.max(rms_torque)
            #calculates first time the data reaches 3% of max value
            md3_1 = np.argmax(rms_torque>0.03*md)
            #calculates first time data is smaller than 3% of max value
            md4 = np.argmax(rms_torque[md3_1+1000:]<=0.03*md)
            #calculates muscle offset (data from 0 to onset (md3) is added to data from md3 to md4 plus 1000 to account for calculation that was done in md4)
            md5 = md3_1+md4+1000
            md3 = np.argmax(rms_torque[md5:]>0.03*md)
            md3_3 = md3 + md5
            md4_1 = np.argmax(rms_torque[md3_3+1000:]<=0.03*md)
            md5_3 = md3_3+md4_1+1000
            md3 = md3_3
            md5 = md5_3
        except:
             #calculates maximum value
            md =  np.max(rms_torque)
            #calculates first time the data reaches 3% of max value
            md3 = np.argmax(rms_torque>0.03*md)
            #calculates first time data is smaller than 3% of max value
            md4 = np.argmax(rms_torque[md3+1000:]<=0.03*md)
            #calculates muscle offset (data from 0 to onset (md3) is added to data from md3 to md4 plus 1000 to account for calculation that was done in md4)
            md5 = md3+md4+1000
            
            
            
    elif sheet1 == sheet_exception2:
        #calculates maximum value
        md =  np.max(rms_torque)
        #calculates first time the data reaches 3% of max value
        md3_1 = np.argmax(rms_torque>0.03*md)
        #calculates first time data is smaller than 3% of max value
        md4 = np.argmax(rms_torque[md3_1+1000:]<=0.03*md)
        #calculates muscle offset (data from 0 to onset (md3) is added to data from md3 to md4 plus 1000 to account for calculation that was done in md4)
        md5 = md3_1+md4+1000
        md3 = np.argmax(rms_torque[md5:]>0.03*md)
        md3_3 = md3 + md5
        md4_1 = np.argmax(rms_torque[md3_3+1000:]<=0.03*md)
        md5_3 = md3_3+md4_1+1000
        md3 = md3_3
        md5 = md5_3
    elif sheet1 == sheet_exception3:
        #calculates maximum value
        md =  np.max(rms_torque)
        #calculates first time the data reaches 3% of max value
        md3_1 = np.argmax(rms_torque>0.03*md)
        #calculates first time data is smaller than 3% of max value
        md4 = np.argmax(rms_torque[md3_1+1000:]<=0.03*md)
        #calculates muscle offset (data from 0 to onset (md3) is added to data from md3 to md4 plus 1000 to account for calculation that was done in md4)
        md5 = md3_1+md4+1000
        md3 = np.argmax(rms_torque[md5:]>0.03*md)
        md3_3 = md3 + md5
        md4_1 = np.argmax(rms_torque[md3_3+1000:]<=0.03*md)
        md5_3 = md3_3+md4_1+1000
        md3 = md3_3
        md5 = md5_3
        
    else: 
        #calculates maximum value
        md =  np.max(rms_torque)
        #calculates first time the data reaches 3% of max value
        md3 = np.argmax(rms_torque>0.1*md)
        #calculates first time data is smaller than 3% of max value
        md4 = np.argmax(rms_torque[md3+1000:]<=0.1*md)
        #calculates muscle offset (data from 0 to onset (md3) is added to data from md3 to md4 plus 1000 to account for calculation that was done in md4)
        md5 = md3+md4+1000

    
    #RMS RF Signal
    
    rms_signal_rf = np.sqrt(np.mean(y3_rf[md3:md5]**2))
    rms_signal_vm = np.sqrt(np.mean(y3_vm[md3:md5]**2))
    rms_signal_vl = np.sqrt(np.mean(y3_vl[md3:md5]**2))
    
    
    #Integrated RF  Signal
    
    AUC_rf = np.trapz(y3_rf[md3:md5], sample[md3:md5])
    AUC_vm = np.trapz(y3_vm[md3:md5], sample[md3:md5])
    AUC_vl = np.trapz(y3_vl[md3:md5], sample[md3:md5])
    

   
    #Here starts FFT calculation
    #data for fft must be already low and highpassed through threshholds


    #use fft function and half the graph so it doesnt appear mirrored
    trans_rf = np.fft.fft(y1_rf[md3:md5])
    trans_vm = np.fft.fft(y1_vm[md3:md5])
    trans_vl = np.fft.fft(y1_vl[md3:md5])
    
    
    N_rf = int(len(trans_rf)/2+1)
    N_vm = int(len(trans_vm)/2+1)
    N_vl = int(len(trans_vl)/2+1)
    
    
    sample_freq = fs
    sample_time = y1_rf[md3:md5].size/1000
    
    #x-axis starting from 0, has all the values/samples of N (which is half of the fft signal 
    #(bc mirrored not needed, that's why half), and goes up to max frequency
    x_ax_rf = np.linspace(0, sample_freq/2, N_rf, endpoint=True) #nyquist frequency is half of the max frequency
    x_ax_vm = np.linspace(0, sample_freq/2, N_vm, endpoint=True)
    x_ax_vl = np.linspace(0, sample_freq/2, N_vl, endpoint=True)
    
    # x2 because we just took half of the fft, normalize by the number of samples (divide by N),
    #abs to get absolute values
    y_ax_rf = 2.0*np.abs(trans_rf[:N_rf])/N_rf 
    y_ax_vm = 2.0*np.abs(trans_vm[:N_vm])/N_vm
    y_ax_vl = 2.0*np.abs(trans_vl[:N_vl])/N_vl
    
    #get mean fft (from papeer: Mean and Median Frequency of EMG Signal to Determine Muscle Force based on Timedependent Power Spectrum - Thonpanja)
    
    y_ax_p_rf = y_ax_rf**2 #sqaure to get Power (this is a fft convention) instead of mV (how much is original signal composed out of specific frequencies)
    fft_mean_rf = sum(y_ax_p_rf*x_ax_rf)/sum(y_ax_p_rf)
    
    y_ax_p_vm = y_ax_vm**2 
    fft_mean_vm = sum(y_ax_p_vm*x_ax_vm)/sum(y_ax_p_vm)
    
    y_ax_p_vl = y_ax_vl**2 
    fft_mean_vl = sum(y_ax_p_vl*x_ax_vl)/sum(y_ax_p_vl)
    
    
    #median like paper formula
    
    med_freq1_rf = sum(y_ax_p_rf)*0.5 #das ist die Hälfte der Summe der Power (y_ax_p)
    med_freq1_vm = sum(y_ax_p_vm)*0.5 
    med_freq1_vl = sum(y_ax_p_vl)*0.5 
    
    #jetzt muss diese hälfte der Power gefunden werden
    
    indiceofmed_rf = 0
    sumofmed_rf = 0
    
    indiceofmed_vm = 0
    sumofmed_vm = 0
    
    indiceofmed_vl = 0
    sumofmed_vl = 0

    while sumofmed_rf < med_freq1_rf:    #go from start to medfreq1, add power until median power is found
        sumofmed_rf = sumofmed_rf+y_ax_p_rf[indiceofmed_rf]
        indiceofmed_rf = indiceofmed_rf+1
    
    while sumofmed_vm < med_freq1_vm:    
        sumofmed_vm = sumofmed_vm+y_ax_p_vm[indiceofmed_vm]
        indiceofmed_vm = indiceofmed_vm+1
    
    while sumofmed_vl < med_freq1_vl:    
        sumofmed_vl = sumofmed_vl+y_ax_p_vl[indiceofmed_vl]
        indiceofmed_vl = indiceofmed_vl+1

    
    
    indiceofmed1_rf = np.size(y_ax_p_rf)-1
    sumofmed1_rf = 0
    
    indiceofmed1_vl = np.size(y_ax_p_vl)-1
    sumofmed1_vl = 0
    
    indiceofmed1_vm = np.size(y_ax_p_vm)-1
    sumofmed1_vm = 0
    
    while sumofmed1_rf < med_freq1_rf: #go from end to medfreq1 add power until median power is found
        sumofmed1_rf = sumofmed1_rf+y_ax_p_rf[indiceofmed1_rf]
        indiceofmed1_rf = indiceofmed1_rf-1
        
    while sumofmed1_vm < med_freq1_vm: 
        sumofmed1_vm = sumofmed1_vm+y_ax_p_vm[indiceofmed1_vm]
        indiceofmed1_vm = indiceofmed1_vm-1
        
    while sumofmed1_vl < med_freq1_vl: 
        sumofmed1_vl = sumofmed1_vl+y_ax_p_vl[indiceofmed1_vl]
        indiceofmed1_vl = indiceofmed1_vl-1
        

    
    mean_of_inx_rf = int((indiceofmed_rf+indiceofmed1_rf)/2) #take mean of two found indices 
    mean_of_inx_vm = int((indiceofmed_vm+indiceofmed1_vm)/2) #take mean of two found indices 
    mean_of_inx_vl = int((indiceofmed_vl+indiceofmed1_vl)/2) #take mean of two found indices 

    
    
    
    median_fft_rf = x_ax_rf[mean_of_inx_rf]
    median_fft_vm = x_ax_vm[mean_of_inx_vm]
    median_fft_vl = x_ax_vl[mean_of_inx_vl]
    
   
    


                                        ################################################

    
     
   
    #maximum value of filtered MVIC
    
    max_value_torque = np.amax(rms_torque)
    
     
    #RFD (calculated by dividing change of Torque in N through time in s)
    #RFD Time Interval
    #1000 Frames = 1s // 100 Frames = 100ms
    #rfd30 = (rms_torque[md3+30]-rms_torque[md3])/0.03
    #rfd50 = (rms_torque[md3+50]-rms_torque[md3])/0.05
    #rfd90 = (rms_torque[md3+90]-rms_torque[md3])/0.09
    #rfd100 = (rms_torque[md3+100]-rms_torque[md3])/0.1
    #rfd150 = (rms_torque[md3+150]-rms_torque[md3])/0.15
    #rfd200 = (rms_torque[md3+200]-rms_torque[md3])/0.2
    #rfd250 = (rms_torque[md3+250]-rms_torque[md3])/0.25
    
    #rfd20-70
    
    rfd_20p = 0.2*max_value_torque
    rfd_70p = 0.7*max_value_torque
    
    rfd_20_i = 0
    
    while rms_torque[md3 + rfd_20_i] <= rfd_20p:
        rfd_20_i = rfd_20_i + 1
    
    
    rfd_70_i = 0
    
    while rms_torque[md3 + rfd_70_i] <= rfd_70p:
        rfd_70_i = rfd_70_i + 1
   
    rfd20_70_time = ((md3 + rfd_70_i)-(md3 + rfd_20_i))/1000
    rfd20_70 = ((rms_torque[md3 + rfd_70_i] - rms_torque[md3 + rfd_20_i]))/ rfd20_70_time 
    '''
    
    #this part of the code is specifically for PAP trials MVIC+10s
    #THIS PART OF THE CODE NEEDS TO BE EXCLUDED IF NOT USED FOR PAP MVIC+10s
    
    pap_i = 0
    while rms_torque[pap_i] < max_value_torque:
        pap_i = pap_i+1
        
    while rms_torque[pap_i] >= 0.15*max_value_torque:
        pap_i = pap_i+1
    
   
    max_value_torque = np.amax(rms_torque[pap_i:])
    
    
    rfd_20p = 0.2*max_value_torque
    rfd_70p = 0.7*max_value_torque
   

    rfd_20_i = 0
    
    while rms_torque[md3 + rfd_20_i] <= rfd_20p:
        rfd_20_i = rfd_20_i + 1
    
    
    rfd_70_i = 0
    
    while rms_torque[md3 + rfd_70_i] <= rfd_70p:
        rfd_70_i = rfd_70_i + 1
   
    
    rfd20_70_time = ((md3 + rfd_70_i)-(md3 + rfd_20_i))/1000
    rfd20_70 = ((rms_torque[md3 + rfd_70_i] - rms_torque[md3 + rfd_20_i]))/ rfd20_70_time 
    
   
    
    '''
    
    #peak EMG signal
    
    peak_rf = np.amax(y3_rf[md3:md5])
    peak_vm = np.amax(y3_vm[md3:md5])
    peak_vl = np.amax(y3_vl[md3:md5])
    
    #peak EMG 1000ms RMS
    
    rms_1s_rf = np.sqrt(np.mean(y3_rf[md3+2000:md3+3000]**2))
    rms_1s_vm = np.sqrt(np.mean(y3_vm[md3+2000:md3+3000]**2))
    rms_1s_vl = np.sqrt(np.mean(y3_vl[md3+2000:md3+3000]**2))
    
    
    #S-Gradient (Force0.5/ Time0.5 (half of the maximum force/ time until F0.5) [N(m)/msec]“S-gradient characterizes the rate of force development at the beginning phase of a musculareffort.”)
    

    #A-Gradient (F0.5/ tmax- t0.5 [N(m)/msec]“A-gradient is used to quantify the rate of force development in the late stages of explosivemuscular efforts.”)
   

    #y3 is filtered emg signal
    #sample is time of trial in s
    #md3 is muscle on
    #md5 is muscle off
    #Detect Muscle On and Off
    #x_ax = x achse für fft
    #y_ax = y achse für fft
    #mean_fft = durchschnittliche frequenz des EMG Signals
    return data_rf, data_vm, data_vl, y3_rf, y3_vm, y3_vl, sample, md3, md5, x_ax_rf, x_ax_vm, x_ax_vl, N_rf, N_vm, N_vl,    trans_rf, trans_vm, trans_vl, y_ax_rf, y_ax_vm, y_ax_vl, fft_mean_rf, fft_mean_vm, fft_mean_vl, median_fft_rf,     median_fft_vm, median_fft_vl, rms_signal_rf, rms_signal_vm, rms_signal_vl, max_value_torque,    rms_torque, rms_sample, rfd_70_i, rfd_20_i,rfd20_70, sheet1, file, rfd_20_i, rfd_70_i, peak_rf, peak_vm, peak_vl, rms_1s_rf, rms_1s_vm, rms_1s_vl

# butter_low_high(cutoff_low, cutoff_high, fs, order, trial, file, sheet)
#important to divide mon and moff by 1000 (basically by frequency to get muscle on and muscle off in seconds and not in frames)
data_rf, data_vm, data_vl, y_rf, y_vm, y_vl, sample, mon, moff, x_ax_rf, x_ax_vm, x_ax_vl, N_rf, N_vm, N_vl, trans_rf, trans_vm,trans_vl, y_ax_rf, y_ax_vm, y_ax_vl, fft_mean_rf, fft_mean_vm, fft_mean_vl, median_fft_rf, median_fft_vm, median_fft_vl, rms_signal_rf,rms_signal_vm, rms_signal_vl, max_torque, rms_torque, rms_sample,rfd_70_i, rfd_20_i,rfd20_70, sheet1, file, rfd_20_i, rfd_70_i, peak_rf, peak_vm, peak_vl, rms_1s_rf, rms_1s_vm, rms_1s_vl= calculation(500, 10, 2000, 2, d_papev, "xlfile7", "MVIC")

print(sheet1)
print(file)


plt.figure(1, figsize=(30, 10))
plt.plot(rms_sample, rms_torque)
plt.axvline(x=mon/1000, c='tab:orange', label="Muscle ON")
plt.axvline(x=moff/1000, c='r', label="Muscle OFF")
#plt.axvline(x=(mon+30)/1000, c='r', label="RFD30")
#plt.axvline(x=(mon+50)/1000, c='g', label="RFD50")
#plt.axvline(x=(mon+90)/1000, c='r', label="RFD90")
#plt.axvline(x=(mon+100)/1000, c='g', label="RFD100")
#plt.axvline(x=(mon+150)/1000, c='g', label="RFD150")
#plt.axvline(x=(mon+200)/1000, c='g', label="RFD200")
#plt.axvline(x=(mon+250)/1000, c='g', label="RFD250")
plt.axvline(x=(mon+rfd_20_i)/1000, c='g', label="RFD20%")
plt.axvline(x=(mon+rfd_70_i)/1000, c='b', label="RFD70%")
plt.axvline(x=(mon+2000)/1000, c='black', label="EMG RMS Start")
plt.axvline(x=(mon+3000)/1000, c='black', label="EMG RMS End")
plt.legend()

plt.figure(2, figsize=(30, 10))
plt.plot(sample, data_rf[:np.size(sample)])
plt.axvline(x=mon/1000, c='tab:orange', label="Muscle ON")
plt.axvline(x=moff/1000, c='r', label="Muscle OFF")
plt.legend()

plt.figure(3, figsize=(30, 10))
plt.plot(sample, data_vm[:np.size(sample)])
plt.axvline(x=mon/1000, c='tab:orange', label="Muscle ON")
plt.axvline(x=moff/1000, c='r', label="Muscle OFF")
plt.legend()

plt.figure(4, figsize=(30, 10))
plt.plot(sample, data_vl[:np.size(sample)])
plt.axvline(x=mon/1000, c='tab:orange', label="Muscle ON")
plt.axvline(x=moff/1000, c='r', label="Muscle OFF")
plt.legend()

plt.figure(5, figsize=(30, 10))
plt.plot(sample, y_rf)
plt.axvline(x=mon/1000, c='tab:orange', label="Muscle ON")
plt.axvline(x=moff/1000, c='r', label="Muscle OFF")
plt.legend()

plt.figure(6, figsize=(30, 10))
plt.plot(sample, y_vm)
plt.axvline(x=mon/1000, c='tab:orange', label="Muscle ON")
plt.axvline(x=moff/1000, c='r', label="Muscle OFF")
plt.legend()

plt.figure(7, figsize=(30, 10))
plt.plot(sample, y_vl)
plt.axvline(x=mon/1000, c='tab:orange', label="Muscle ON")
plt.axvline(x=moff/1000, c='r', label="Muscle OFF")
plt.legend()

plt.figure(8, figsize=(30, 10))
plt.plot(x_ax_rf, y_ax_rf)
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Amplitude ($Unit$)')
plt.axvline(x=median_fft_rf, c='tab:orange', label= f"Median Frequency: {round(median_fft_rf,2)}")
plt.axvline(x=fft_mean_rf, c='r', label= f"Mean Frequency: {round(fft_mean_rf,2)}")
plt.legend()

plt.figure(9, figsize=(30, 10))
plt.plot(x_ax_vm, y_ax_vm)
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Amplitude ($Unit$)')
plt.axvline(x=median_fft_vm, c='tab:orange', label= f"Median Frequency: {round(median_fft_vm,2)}")
plt.axvline(x=fft_mean_vm, c='r', label= f"Mean Frequency: {round(fft_mean_vm,2)}")
plt.legend()

plt.figure(10, figsize=(30, 10))
plt.plot(x_ax_vl, y_ax_vl)
plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Amplitude ($Unit$)')
plt.axvline(x=median_fft_vl, c='tab:orange', label= f"Median Frequency: {round(median_fft_vl,2)}")
plt.axvline(x=fft_mean_vl, c='r', label= f"Mean Frequency: {round(fft_mean_vl,2)}")
plt.legend()




print("rms_signal_rf")
print("rms_signal_vm")
print("rms_signal_vl")

print("fft_mean_rf")
print("fft_mean_vm")
print("fft_mean_vl")
print("median_fft_rf")
print("median_fft_vm")
print("median_fft_vl")
print("max_torque")
print("rfd20_70")

print("peak_rf")
print("peak_vm")
print("peak_vl")

print("")
print("")
print("")

print(rms_signal_rf)
print(rms_signal_vm)
print(rms_signal_vl)

print(fft_mean_rf)
print(fft_mean_vm)
print(fft_mean_vl)
print(median_fft_rf)
print(median_fft_vm)
print(median_fft_vl)
print(max_torque)
print(rfd20_70)


print(peak_rf)
print(peak_vm)
print(peak_vl)




print()
print()
print()


print(rms_1s_rf)
print(rms_1s_vm)
print(rms_1s_vl)

print()
print()
print()
'''
how to transform list to array
big_array = [] #  empty regular list
big_np_array = np.array(big_array)  # transformed to a numpy array
'''


# # ERGEBNISSE FÜR EINZELNE PROBANDEN AUFZEIGEN

# ## Fertige PAPEE Werte

# In[6]:


rms_signal_rf_papee_MVIC0 = [2.16356436, 1.68308686, 0.16527648, 2.05180131, 2.54689419, 0.6011799 , 2.09139061, 0.29239169, 1.87140545, 0.80395507]
rms_signal_vm_papee_MVIC0 = [1.56428302, 2.08368495, 0.67816054, 2.23418152, 1.97524712, 0.46922886, 1.93841203, 0.26738067, 2.06547515, 0.75333655]
rms_signal_vl_papee_MVIC0 = [1.07899648, 1.41986785, 0.23586292, 1.57144415, 1.10889043, 0.49486549, 1.2095778 , 0.28134796, 1.88253203, 0.86953046]

fft_mean_rf_papee_MVIC0 = [81.58846167, 88.27921626, 80.92184395, 87.67225064, 67.73182624, 60.11488663, 77.43817084, 66.36105523, 66.56221472, 73.38721059]
fft_mean_vm_papee_MVIC0 = [89.35697691, 76.0624525 ,  64.01579336,  71.93132194, 65.08354245, 103.35238525,  75.0225938 ,  97.79248042, 62.32803051,  62.61700331]
fft_mean_vl_papee_MVIC0 = [72.10169745, 65.5700557 , 67.29730972, 63.30188434, 67.87882995, 77.18305295, 72.9118487 , 66.10859425, 58.7863173 , 58.60612355]

median_fft_rf_papee_MVIC0 = [72.2331048 , 77.18120805, 67.13780919, 77.39726027, 59.28143713, 52.42203052, 65.12017887, 61.20527307, 56.84326711, 65.47619048]
median_fft_vm_papee_MVIC0 = [74.92654261, 63.14826113, 51.99394245, 58.21917808, 58.98203593, 74.31984074, 65.95863611, 80.97928437, 53.80794702, 58.82352941]
median_fft_vl_papee_MVIC0 = [61.45935357, 59.48749237, 54.77031802, 57.87671233, 59.58083832, 63.70272064, 63.16377865, 56.73258004, 52.98013245, 57.07282913]

max_torque_papee_MVIC0 = [122.31715334, 166.74297538, 155.89142898, 180.45338681, 133.353607  , 113.59974069,  94.29076772, 130.45057315, 124.94500678,  92.36188641]

rfd20_70_papee_MVIC0 = [156.87954335, 469.58628546, 497.14488417, 587.00122325, 409.37144299, 229.69941565, 330.21704107, 577.6404226 , 339.69197705, 528.51247586]


# In[7]:


rms_signal_rf_papee_MVIC2s = [2.38169416, 2.71190846, 0.19667417, 2.08516787, 3.25084897, 0.76777038, 2.53951109, 0.2831694 , 2.7836272 , 0.7649604 ]
rms_signal_vm_papee_MVIC2s = [1.96929495, 3.08438979, 0.68243156, 2.35066034, 2.41579588, 0.46174069, 1.99693913, 0.2428296 , 2.83594868, 0.80285604]
rms_signal_vl_papee_MVIC2s = [1.50341337, 2.28327118, 0.25556719, 1.66006293, 1.79085038, 0.65207906, 1.52773192, 1.71250927, 2.47170413, 0.78353521]

fft_mean_rf_papee_MVIC2s = [71.62316248, 72.07361186, 83.37727363, 78.67380024, 69.04740952, 65.32409123, 69.586358  , 63.31785348, 49.64264996, 71.53930019]
fft_mean_vm_papee_MVIC2s = [71.65844916, 70.42439499, 96.89977149, 70.1581794 , 64.75939343, 66.45909727, 65.36927143, 83.09060767, 52.16264135, 61.922571  ]
fft_mean_vl_papee_MVIC2s = [ 55.63706795,  61.90951011,  70.85862028,  62.38791078, 64.32517313,  71.38002888,  59.08690902, 200.05003656, 51.10497696,  58.40327907]

median_fft_rf_papee_MVIC2s = [59.84174085, 62.47053277, 67.23210592, 64.20021763, 64.26011265, 55.12031337, 63.4057971 , 54.66540999, 43.15196998, 65.25522042]
median_fft_vm_papee_MVIC2s = [56.13254204, 61.99905705, 73.64501448, 58.75952122, 60.67588326, 57.91829882, 56.46135266, 64.79736098, 46.27892433, 59.45475638]
median_fft_vl_papee_MVIC2s = [ 48.21958457,  56.81282414,  58.54364915,  57.39934712, 60.16385049,  57.63850028,  55.55555556, 200.04712535, 46.59161976,  55.39443155]

max_torque_papee_MVIC2s = [126.77965283, 203.17983583, 174.22580179, 158.72360419, 163.80384599, 111.54103776,  95.09614107, 129.40810888, 113.56225888,  83.29765447]

rfd20_70_papee_MVIC2s = [237.4395891 , 422.52116879, 282.34163931, 555.59186132, 253.92506434, 274.87411698, 420.77858401, 554.95857443, 821.45305503, 539.86754492]


# In[8]:


rms_signal_rf_papee_MVIC4m = [1.90901884, 2.19505615, 0.14198777, 2.03250407, 3.03650611, 0.66529611, 1.62522879, 0.33949376, 2.20186843, 0.6443555 ]
rms_signal_vm_papee_MVIC4m = [1.43373882, 2.56452893, 0.80039723, 2.15372372, 2.18573089, 0.39245895, 1.28664224, 0.28360051, 2.05429061, 0.75420823]
rms_signal_vl_papee_MVIC4m = [1.01734348, 1.69364054, 0.24650746, 1.44130126, 1.63234697, 0.46876244, 0.9405798 , 1.71193133, 1.80015038, 0.75689475]

fft_mean_rf_papee_MVIC4m = [90.18079208, 87.35007494, 88.35949634, 90.67370206, 73.67400134, 74.03814175, 78.3614827 , 66.83234201, 62.83575363, 81.34555309]
fft_mean_vm_papee_MVIC4m = [86.08541385, 78.24378262, 89.99903187, 79.33685562, 67.69012399, 71.63320272, 78.106378  , 89.21425112, 64.72786806, 67.82085895]
fft_mean_vl_papee_MVIC4m = [ 73.48533667,  66.77968623,  69.17967135,  69.97451932, 67.93523253,  83.23420981,  66.44183911, 200.03810622, 61.53365326,  61.91589343]

median_fft_rf_papee_MVIC4m = [79.63386728, 74.7098646 , 70.09109312, 82.46474555, 66.06217617, 62.03703704, 68.84517766, 60.42074364, 55.8988764 , 69.43563729]
median_fft_vm_papee_MVIC4m = [70.0228833 , 67.94003868, 68.31983806, 64.6842428 , 60.10362694, 63.88888889, 69.16243655, 67.5146771 , 57.02247191, 63.0944832 ]
median_fft_vl_papee_MVIC4m = [ 63.61556064,  61.89555126,  60.22267206,  59.16615573, 59.84455959,  70.0617284 ,  61.23096447, 200.09784736, 54.7752809 ,  57.07038681]

max_torque_papee_MVIC4m = [113.26094096, 184.77754895, 180.49245739, 169.49820326, 181.07514894, 115.0367227 ,  73.49028806, 142.4947151 , 115.74933086,  90.14968939]

rfd20_70_papee_MVIC4m = [366.7279314 , 413.2596435 , 517.67748856, 599.51172956, 188.32020463, 248.54029582, 212.67915319, 530.1206426 , 500.45844727, 494.15457733]


# In[9]:


rms_signal_rf_papee_MVIC8m = [2.25678817, 2.14361951, 0.20282783, 1.92857023, 3.2061897 , 0.69517207, 1.9892663 , 0.3112552 , 2.40027475, 0.71927152]
rms_signal_vm_papee_MVIC8m = [1.72859954, 2.57887946, 0.87636203, 2.06517259, 2.48042126, 0.37480999, 1.5509371 , 0.28910428, 2.39568955, 0.80316091]
rms_signal_vl_papee_MVIC8m = [1.25631305, 1.79116558, 0.26123647, 1.4300521 , 1.90209086, 0.48608531, 1.18573076, 0.35722396, 2.04339413, 0.75109901]

fft_mean_rf_papee_MVIC8m = [90.09337517, 88.80931983, 82.53848789, 90.82280113, 73.11131489, 66.46455   , 77.86308045, 70.42486947, 59.45146948, 79.20362522]
fft_mean_vm_papee_MVIC8m = [87.89172215, 78.10370968, 88.38991547, 79.37958746, 66.02863623, 69.2279872 , 76.60814343, 88.65144423, 61.85598192, 68.85212292]
fft_mean_vl_papee_MVIC8m = [70.63934656, 67.33063582, 68.19176767, 64.59176834, 68.7525802 , 84.90241462, 65.90681013, 69.09352413, 59.88178226, 64.25295837]

median_fft_rf_papee_MVIC8m = [78.55297158, 74.85741445, 69.70912738, 78.34507042, 65.1371308 , 59.5065312 , 67.769131  , 58.97887324, 51.38568129, 68.64924958]
median_fft_vm_papee_MVIC8m = [70.54263566, 65.58935361, 68.70611836, 62.5       , 58.01687764, 59.99032414, 65.82360571, 65.4342723 , 55.42725173, 64.758199  ]
median_fft_vl_papee_MVIC8m = [59.94832041, 61.31178707, 57.67301906, 57.51173709, 61.44514768, 72.81083696, 60.31128405, 61.91314554, 54.27251732, 61.1450806 ]

max_torque_papee_MVIC8m = [133.76947255, 189.97745341, 189.95020837, 168.21289466, 183.13920635, 110.44905345,  77.89775314, 137.42494528, 120.0592422 ,  90.95448634]

rfd20_70_papee_MVIC8m = [226.37652521, 397.66674358, 600.60820151, 607.23576917, 520.12073515, 160.95187419, 151.5069661 , 570.06066902, 839.43154013, 428.09244621]


# In[10]:


rms_signal_rf_papee_MVIC12m = [2.0409748 , 2.08265099, 0.20158538, 1.82373152, 2.83204364, 0.51947711, 2.04886157, 1.7604475 , 2.29430139, 0.71706201]
rms_signal_vm_papee_MVIC12m = [1.51377076, 2.44879131, 1.05761854, 1.94675112, 2.11263649, 0.33378162, 1.68988149, 0.26077865, 2.20571743, 0.84773051]
rms_signal_vl_papee_MVIC12m = [1.01439552, 1.62732573, 0.28100766, 1.36392025, 1.58019539, 0.43392413, 1.33992078, 0.35521252, 1.93922684, 0.84540037]

fft_mean_rf_papee_MVIC12m = [ 93.38278016,  89.6958619 ,  84.16844494,  91.90261818, 73.89637564,  69.23897912,  75.40908028, 200.05831973, 61.80331828,  78.23410458]
fft_mean_vm_papee_MVIC12m = [93.47769988, 78.59792914, 94.96512588, 80.80959249, 69.12385583, 73.37317359, 72.29046409, 94.75486779, 63.12266186, 64.74850672]
fft_mean_vl_papee_MVIC12m = [74.54636209, 68.13386266, 67.31755922, 66.2487772 , 69.34093622, 83.65807451, 64.56718098, 65.88822557, 59.67445024, 59.55712248]

median_fft_rf_papee_MVIC12m = [ 81.91035219,  77.92515396,  65.79621096,  81.04154476, 68.05640711,  60.25867137,  65.87712805, 199.89106754, 53.05901462,  66.4167113 ]
median_fft_vm_papee_MVIC12m = [77.10779082, 66.08242539, 75.52483359, 64.95026331, 63.45800123, 64.37389771, 61.80606958, 76.79738562, 56.30752572, 59.98928763]
median_fft_vl_papee_MVIC12m = [63.76734258, 61.58218854, 55.04352279, 58.51375073, 60.69895769, 70.54673721, 62.91635825, 56.91721133, 53.05901462, 56.23995715]

max_torque_papee_MVIC12m = [117.82324878, 176.46672074, 200.75385221, 160.78685152, 149.77232842, 110.94033883,  83.15292362, 140.07652692, 113.28457387,  87.53555318]



# In[11]:


stats.f_oneway(max_torque_papee_MVIC0, max_torque_papee_MVIC2s, max_torque_papee_MVIC4m, max_torque_papee_MVIC8m, max_torque_papee_MVIC12m)


# In[12]:


stats.f_oneway(rfd20_70_papee_MVIC0, rfd20_70_papee_MVIC2s, rfd20_70_papee_MVIC4m, rfd20_70_papee_MVIC8m, rfd20_70_papee_MVIC12m)


# In[13]:


stats.f_oneway(median_fft_vl_papee_MVIC0, median_fft_vl_papee_MVIC2s, median_fft_vl_papee_MVIC4m, median_fft_vl_papee_MVIC8m, median_fft_vl_papee_MVIC12m)


# In[14]:


stats.f_oneway(fft_mean_vl_papee_MVIC0, fft_mean_vl_papee_MVIC2s, fft_mean_vl_papee_MVIC4m, fft_mean_vl_papee_MVIC8m, fft_mean_vl_papee_MVIC12m)


# In[15]:


stats.f_oneway(rms_signal_vl_papee_MVIC0, rms_signal_vl_papee_MVIC2s, rms_signal_vl_papee_MVIC4m, rms_signal_vl_papee_MVIC8m, rms_signal_vl_papee_MVIC12m)


# In[16]:


stats.f_oneway(rms_signal_vm_papee_MVIC0, rms_signal_vm_papee_MVIC2s, rms_signal_vm_papee_MVIC4m, rms_signal_vm_papee_MVIC8m, rms_signal_vm_papee_MVIC12m)


# In[17]:


stats.f_oneway(fft_mean_vm_papee_MVIC0, fft_mean_vm_papee_MVIC2s, fft_mean_vm_papee_MVIC4m, fft_mean_vm_papee_MVIC8m, fft_mean_vm_papee_MVIC12m)


# In[18]:


stats.f_oneway(median_fft_vm_papee_MVIC0, median_fft_vm_papee_MVIC2s, median_fft_vm_papee_MVIC4m, median_fft_vm_papee_MVIC8m, median_fft_vm_papee_MVIC12m)


# In[19]:


stats.f_oneway(median_fft_rf_papee_MVIC0, median_fft_rf_papee_MVIC2s, median_fft_rf_papee_MVIC4m, median_fft_rf_papee_MVIC8m, median_fft_rf_papee_MVIC12m)


# In[20]:


stats.f_oneway(fft_mean_rf_papee_MVIC0, fft_mean_rf_papee_MVIC2s, fft_mean_rf_papee_MVIC4m, fft_mean_rf_papee_MVIC8m, fft_mean_rf_papee_MVIC12m)


# In[21]:


stats.f_oneway(rms_signal_rf_papee_MVIC0, rms_signal_rf_papee_MVIC2s, rms_signal_rf_papee_MVIC4m, rms_signal_rf_papee_MVIC8m, rms_signal_rf_papee_MVIC12m)


# In[22]:


# Average of Torque Values Time-grouped
mean_max_torque_papee_MVIC0 = np.mean(max_torque_papee_MVIC0)
mean_max_torque_papee_MVIC2s = np.mean(max_torque_papee_MVIC2s)
mean_max_torque_papee_MVIC4m= np.mean(max_torque_papee_MVIC4m)
mean_max_torque_papee_MVIC8m = np.mean(max_torque_papee_MVIC8m)
mean_max_torque_papee_MVIC12m = np.mean(max_torque_papee_MVIC12m)

means_of_papee_torque = np.array([mean_max_torque_papee_MVIC0, mean_max_torque_papee_MVIC2s, mean_max_torque_papee_MVIC4m, mean_max_torque_papee_MVIC8m, mean_max_torque_papee_MVIC12m])


# In[23]:


# Average of RFD Values Time-grouped
mean_rfd20_70_papee_MVIC0 = np.mean(rfd20_70_papee_MVIC0)
mean_rfd20_70_papee_MVIC2s = np.mean(rfd20_70_papee_MVIC2s)
mean_rfd20_70_papee_MVIC4m= np.mean(rfd20_70_papee_MVIC4m)
mean_rfd20_70_papee_MVIC8m = np.mean(rfd20_70_papee_MVIC8m)
mean_rfd20_70_papee_MVIC12m = np.mean(rfd20_70_papee_MVIC12m)

means_of_papee_rfd = np.array([mean_rfd20_70_papee_MVIC0, mean_rfd20_70_papee_MVIC2s, mean_rfd20_70_papee_MVIC4m, mean_rfd20_70_papee_MVIC8m, mean_rfd20_70_papee_MVIC12m])


# In[24]:


# Average of RMS RF Values Time-grouped
mean_rms_signal_rf_papee_MVIC0 = np.mean(rms_signal_rf_papee_MVIC0)
mean_rms_signal_rf_papee_MVIC2s = np.mean(rms_signal_rf_papee_MVIC2s)
mean_rms_signal_rf_papee_MVIC4m= np.mean(rms_signal_rf_papee_MVIC4m)
mean_rms_signal_rf_papee_MVIC8m = np.mean(rms_signal_rf_papee_MVIC8m)
mean_rms_signal_rf_papee_MVIC12m = np.mean(rms_signal_rf_papee_MVIC12m)

mean_rms_signal_rf_papee = np.array([mean_rms_signal_rf_papee_MVIC0, mean_rms_signal_rf_papee_MVIC2s, mean_rms_signal_rf_papee_MVIC4m, mean_rms_signal_rf_papee_MVIC8m, mean_rms_signal_rf_papee_MVIC12m])


# In[25]:


# Average of RMS VM Values Time-grouped
mean_rms_signal_vm_papee_MVIC0 = np.mean(rms_signal_vm_papee_MVIC0)
mean_rms_signal_vm_papee_MVIC2s = np.mean(rms_signal_vm_papee_MVIC2s)
mean_rms_signal_vm_papee_MVIC4m= np.mean(rms_signal_vm_papee_MVIC4m)
mean_rms_signal_vm_papee_MVIC8m = np.mean(rms_signal_vm_papee_MVIC8m)
mean_rms_signal_vm_papee_MVIC12m = np.mean(rms_signal_vm_papee_MVIC12m)

mean_rms_signal_vm_papee = np.array([mean_rms_signal_vm_papee_MVIC0, mean_rms_signal_vm_papee_MVIC2s, mean_rms_signal_vm_papee_MVIC4m, mean_rms_signal_vm_papee_MVIC8m, mean_rms_signal_vm_papee_MVIC12m])


# In[26]:


# Average of RMS VL Values Time-grouped
mean_rms_signal_vl_papee_MVIC0 = np.mean(rms_signal_vl_papee_MVIC0)
mean_rms_signal_vl_papee_MVIC2s = np.mean(rms_signal_vl_papee_MVIC2s)
mean_rms_signal_vl_papee_MVIC4m= np.mean(rms_signal_vl_papee_MVIC4m)
mean_rms_signal_vl_papee_MVIC8m = np.mean(rms_signal_vl_papee_MVIC8m)
mean_rms_signal_vl_papee_MVIC12m = np.mean(rms_signal_vl_papee_MVIC12m)

mean_rms_signal_vl_papee = np.array([mean_rms_signal_vl_papee_MVIC0, mean_rms_signal_vl_papee_MVIC2s, mean_rms_signal_vl_papee_MVIC4m, mean_rms_signal_vl_papee_MVIC8m, mean_rms_signal_vl_papee_MVIC12m])


# In[27]:


# Average of FFT RF Values Time-grouped
mean_fft_mean_rf_papee_MVIC0 = np.mean(fft_mean_rf_papee_MVIC0)
mean_fft_mean_rf_papee_MVIC2s = np.mean(fft_mean_rf_papee_MVIC2s)
mean_fft_mean_rf_papee_MVIC4m= np.mean(fft_mean_rf_papee_MVIC4m)
mean_fft_mean_rf_papee_MVIC8m = np.mean(fft_mean_rf_papee_MVIC8m)
mean_fft_mean_rf_papee_MVIC12m = np.mean(fft_mean_rf_papee_MVIC12m)

mean_fft_mean_rf_papee = np.array([mean_fft_mean_rf_papee_MVIC0, mean_fft_mean_rf_papee_MVIC2s, mean_fft_mean_rf_papee_MVIC4m, mean_fft_mean_rf_papee_MVIC8m, mean_fft_mean_rf_papee_MVIC12m])


# In[28]:


# Average of FFT VM Values Time-grouped
mean_fft_mean_vm_papee_MVIC0 = np.mean(fft_mean_vm_papee_MVIC0)
mean_fft_mean_vm_papee_MVIC2s = np.mean(fft_mean_vm_papee_MVIC2s)
mean_fft_mean_vm_papee_MVIC4m= np.mean(fft_mean_vm_papee_MVIC4m)
mean_fft_mean_vm_papee_MVIC8m = np.mean(fft_mean_vm_papee_MVIC8m)
mean_fft_mean_vm_papee_MVIC12m = np.mean(fft_mean_vm_papee_MVIC12m)

mean_fft_mean_vm_papee = np.array([mean_fft_mean_vm_papee_MVIC0, mean_fft_mean_vm_papee_MVIC2s, mean_fft_mean_vm_papee_MVIC4m, mean_fft_mean_vm_papee_MVIC8m, mean_fft_mean_vm_papee_MVIC12m])


# In[29]:


# Average of FFT VL Values Time-grouped
mean_fft_mean_vl_papee_MVIC0 = np.mean(fft_mean_vl_papee_MVIC0)
mean_fft_mean_vl_papee_MVIC2s = np.mean(fft_mean_vl_papee_MVIC2s)
mean_fft_mean_vl_papee_MVIC4m= np.mean(fft_mean_vl_papee_MVIC4m)
mean_fft_mean_vl_papee_MVIC8m = np.mean(fft_mean_vl_papee_MVIC8m)
mean_fft_mean_vl_papee_MVIC12m = np.mean(fft_mean_vl_papee_MVIC12m)

mean_fft_mean_vl_papee = np.array([mean_fft_mean_vl_papee_MVIC0, mean_fft_mean_vl_papee_MVIC2s, mean_fft_mean_vl_papee_MVIC4m, mean_fft_mean_vl_papee_MVIC8m, mean_fft_mean_vl_papee_MVIC12m])


# In[30]:


# Average of MEDIAN FFT RF Values Time-grouped
mean_median_fft_rf_papee_MVIC0 = np.mean(median_fft_rf_papee_MVIC0)
mean_median_fft_rf_papee_MVIC2s = np.mean(median_fft_rf_papee_MVIC2s)
mean_median_fft_rf_papee_MVIC4m= np.mean(median_fft_rf_papee_MVIC4m)
mean_median_fft_rf_papee_MVIC8m = np.mean(median_fft_rf_papee_MVIC8m)
mean_median_fft_rf_papee_MVIC12m = np.mean(median_fft_rf_papee_MVIC12m)

mean_median_fft_rf_papee = np.array([mean_median_fft_rf_papee_MVIC0, mean_median_fft_rf_papee_MVIC2s, mean_median_fft_rf_papee_MVIC4m, mean_median_fft_rf_papee_MVIC8m, mean_median_fft_rf_papee_MVIC12m])


# In[31]:


# Average of MEDIAN FFT VM Values Time-grouped
mean_median_fft_vm_papee_MVIC0 = np.mean(median_fft_vm_papee_MVIC0)
mean_median_fft_vm_papee_MVIC2s = np.mean(median_fft_vm_papee_MVIC2s)
mean_median_fft_vm_papee_MVIC4m= np.mean(median_fft_vm_papee_MVIC4m)
mean_median_fft_vm_papee_MVIC8m = np.mean(median_fft_vm_papee_MVIC8m)
mean_median_fft_vm_papee_MVIC12m = np.mean(median_fft_vm_papee_MVIC12m)

mean_median_fft_vm_papee = np.array([mean_median_fft_vm_papee_MVIC0, mean_median_fft_vm_papee_MVIC2s, mean_median_fft_vm_papee_MVIC4m, mean_median_fft_vm_papee_MVIC8m, mean_median_fft_vm_papee_MVIC12m])


# In[32]:


# Average of MEDIAN FFT VL Values Time-grouped
mean_median_fft_vl_papee_MVIC0 = np.mean(median_fft_vl_papee_MVIC0)
mean_median_fft_vl_papee_MVIC2s = np.mean(median_fft_vl_papee_MVIC2s)
mean_median_fft_vl_papee_MVIC4m= np.mean(median_fft_vl_papee_MVIC4m)
mean_median_fft_vl_papee_MVIC8m = np.mean(median_fft_vl_papee_MVIC8m)
mean_median_fft_vl_papee_MVIC12m = np.mean(median_fft_vl_papee_MVIC12m)

mean_median_fft_vl_papee = np.array([mean_median_fft_vl_papee_MVIC0, mean_median_fft_vl_papee_MVIC2s, mean_median_fft_vl_papee_MVIC4m, mean_median_fft_vl_papee_MVIC8m, mean_median_fft_vl_papee_MVIC12m])


# In[33]:


#TORQUE BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
means_of_papee_t = means_of_papee_torque.astype(int)
x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, means_of_papee_t, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Torque in Nm')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 250])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[34]:


#RFD BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
means_of_rfd2070 = means_of_papee_rfd.astype(int)
x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, means_of_rfd2070, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RFD 20%-70% in Nm/s')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 650])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[35]:


#RMS RF BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_rms_signal_rf_papee = np.round(mean_rms_signal_rf_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_rms_signal_rf_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMS Rectus Femoris')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 2])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[36]:


#RMS VM BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_rms_signal_vm_papee = np.round(mean_rms_signal_vm_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_rms_signal_vm_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMS Vastus Medialis')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 2])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[37]:


#RMS VL BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_rms_signal_vl_papee = np.round(mean_rms_signal_vl_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_rms_signal_vl_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMS Vastus Lateralis')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 2])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[38]:


#FFT Mean RF BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_fft_mean_rf_papee = np.round(mean_fft_mean_rf_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_fft_mean_rf_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Mean RF')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[39]:


#FFT Mean VM BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_fft_mean_vm_papee = np.round(mean_fft_mean_vm_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_fft_mean_vm_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Mean VM')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[40]:


#FFT Mean VL 
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_fft_mean_vl_papee = np.round(mean_fft_mean_vl_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_fft_mean_vl_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Mean VL')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[41]:


#FFT Mean VL 
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_median_fft_rf_papee = np.round(mean_median_fft_rf_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_median_fft_rf_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Median RF')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[42]:


#FFT Median VM
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_median_fft_vm_papee = np.round(mean_median_fft_vm_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_median_fft_vm_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Median VM')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[43]:


#FFT Median VL 
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_median_fft_vl_papee = np.round(mean_median_fft_vl_papee,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_median_fft_vl_papee, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Median VL')
ax.set_title('PAPEE - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[ ]:





# In[44]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,max_torque_papee_MVIC0[0]], [2,max_torque_papee_MVIC2s[0]], [3,max_torque_papee_MVIC4m[0]],[4,max_torque_papee_MVIC8m[0]],[5,max_torque_papee_MVIC12m[0]], "green", "S1")
newline([1,max_torque_papee_MVIC0[1]], [2,max_torque_papee_MVIC2s[1]], [3,max_torque_papee_MVIC4m[1]],[4,max_torque_papee_MVIC8m[1]],[5,max_torque_papee_MVIC12m[1]], "lime", "S2")
newline([1,max_torque_papee_MVIC0[2]], [2,max_torque_papee_MVIC2s[2]], [3,max_torque_papee_MVIC4m[2]],[4,max_torque_papee_MVIC8m[2]],[5,max_torque_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,max_torque_papee_MVIC0[3]], [2,max_torque_papee_MVIC2s[3]], [3,max_torque_papee_MVIC4m[3]],[4,max_torque_papee_MVIC8m[3]],[5,max_torque_papee_MVIC12m[3]], "blue", "S4")
newline([1,max_torque_papee_MVIC0[4]], [2,max_torque_papee_MVIC2s[4]], [3,max_torque_papee_MVIC4m[4]],[4,max_torque_papee_MVIC8m[4]],[5,max_torque_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,max_torque_papee_MVIC0[5]], [2,max_torque_papee_MVIC2s[5]], [3,max_torque_papee_MVIC4m[5]],[4,max_torque_papee_MVIC8m[5]],[5,max_torque_papee_MVIC12m[5]], "purple", "S6")
newline([1,max_torque_papee_MVIC0[6]], [2,max_torque_papee_MVIC2s[6]], [3,max_torque_papee_MVIC4m[6]],[4,max_torque_papee_MVIC8m[6]],[5,max_torque_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,max_torque_papee_MVIC0[7]], [2,max_torque_papee_MVIC2s[7]], [3,max_torque_papee_MVIC4m[7]],[4,max_torque_papee_MVIC8m[7]],[5,max_torque_papee_MVIC12m[7]], "gold", "S8")
newline([1,max_torque_papee_MVIC0[8]], [2,max_torque_papee_MVIC2s[8]], [3,max_torque_papee_MVIC4m[8]],[4,max_torque_papee_MVIC8m[8]],[5,max_torque_papee_MVIC12m[8]], "red", "S9")
newline([1,max_torque_papee_MVIC0[9]], [2,max_torque_papee_MVIC2s[9]], [3,max_torque_papee_MVIC4m[9]],[4,max_torque_papee_MVIC8m[9]],[5,max_torque_papee_MVIC12m[9]], "black", "S10")
#newline([1,max_torque_papev_MVIC0[10]], [2,max_torque_papev_MVIC2s[10]], [3,max_torque_papev_MVIC4m[10]],[4,max_torque_papev_MVIC8m[10]],[5,max_torque_papev_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Torque - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,300), ylabel='Torque in Nm')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 300, 50), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[45]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rfd20_70_papee_MVIC0[0]], [2,rfd20_70_papee_MVIC2s[0]], [3,rfd20_70_papee_MVIC4m[0]],[4,rfd20_70_papee_MVIC8m[0]],[5,rfd20_70_papee_MVIC12m[0]], "green", "S1")
newline([1,rfd20_70_papee_MVIC0[1]], [2,rfd20_70_papee_MVIC2s[1]], [3,rfd20_70_papee_MVIC4m[1]],[4,rfd20_70_papee_MVIC8m[1]],[5,rfd20_70_papee_MVIC12m[1]], "lime", "S2")
newline([1,rfd20_70_papee_MVIC0[2]], [2,rfd20_70_papee_MVIC2s[2]], [3,rfd20_70_papee_MVIC4m[2]],[4,rfd20_70_papee_MVIC8m[2]],[5,rfd20_70_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,rfd20_70_papee_MVIC0[3]], [2,rfd20_70_papee_MVIC2s[3]], [3,rfd20_70_papee_MVIC4m[3]],[4,rfd20_70_papee_MVIC8m[3]],[5,rfd20_70_papee_MVIC12m[3]], "blue", "S4")
newline([1,rfd20_70_papee_MVIC0[4]], [2,rfd20_70_papee_MVIC2s[4]], [3,rfd20_70_papee_MVIC4m[4]],[4,rfd20_70_papee_MVIC8m[4]],[5,rfd20_70_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,rfd20_70_papee_MVIC0[5]], [2,rfd20_70_papee_MVIC2s[5]], [3,rfd20_70_papee_MVIC4m[5]],[4,rfd20_70_papee_MVIC8m[5]],[5,rfd20_70_papee_MVIC12m[5]], "purple", "S6")
newline([1,rfd20_70_papee_MVIC0[6]], [2,rfd20_70_papee_MVIC2s[6]], [3,rfd20_70_papee_MVIC4m[6]],[4,rfd20_70_papee_MVIC8m[6]],[5,rfd20_70_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,rfd20_70_papee_MVIC0[7]], [2,rfd20_70_papee_MVIC2s[7]], [3,rfd20_70_papee_MVIC4m[7]],[4,rfd20_70_papee_MVIC8m[7]],[5,rfd20_70_papee_MVIC12m[7]], "gold", "S8")
newline([1,rfd20_70_papee_MVIC0[8]], [2,rfd20_70_papee_MVIC2s[8]], [3,rfd20_70_papee_MVIC4m[8]],[4,rfd20_70_papee_MVIC8m[8]],[5,rfd20_70_papee_MVIC12m[8]], "red", "S9")
newline([1,rfd20_70_papee_MVIC0[9]], [2,rfd20_70_papee_MVIC2s[9]], [3,rfd20_70_papee_MVIC4m[9]],[4,rfd20_70_papee_MVIC8m[9]],[5,rfd20_70_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Rate of Force Developement 20%-70% - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,1000), ylabel='RFD 20%-70% in Nm/s')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(100, 1000, 50), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[46]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rms_signal_rf_papee_MVIC0[0]], [2,rms_signal_rf_papee_MVIC2s[0]], [3,rms_signal_rf_papee_MVIC4m[0]],[4,rms_signal_rf_papee_MVIC8m[0]],[5,rms_signal_rf_papee_MVIC12m[0]], "green", "S1")
newline([1,rms_signal_rf_papee_MVIC0[1]], [2,rms_signal_rf_papee_MVIC2s[1]], [3,rms_signal_rf_papee_MVIC4m[1]],[4,rms_signal_rf_papee_MVIC8m[1]],[5,rms_signal_rf_papee_MVIC12m[1]], "lime", "S2")
newline([1,rms_signal_rf_papee_MVIC0[2]], [2,rms_signal_rf_papee_MVIC2s[2]], [3,rms_signal_rf_papee_MVIC4m[2]],[4,rms_signal_rf_papee_MVIC8m[2]],[5,rms_signal_rf_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,rms_signal_rf_papee_MVIC0[3]], [2,rms_signal_rf_papee_MVIC2s[3]], [3,rms_signal_rf_papee_MVIC4m[3]],[4,rms_signal_rf_papee_MVIC8m[3]],[5,rms_signal_rf_papee_MVIC12m[3]], "blue", "S4")
newline([1,rms_signal_rf_papee_MVIC0[4]], [2,rms_signal_rf_papee_MVIC2s[4]], [3,rms_signal_rf_papee_MVIC4m[4]],[4,rms_signal_rf_papee_MVIC8m[4]],[5,rms_signal_rf_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,rms_signal_rf_papee_MVIC0[5]], [2,rms_signal_rf_papee_MVIC2s[5]], [3,rms_signal_rf_papee_MVIC4m[5]],[4,rms_signal_rf_papee_MVIC8m[5]],[5,rms_signal_rf_papee_MVIC12m[5]], "purple", "S6")
newline([1,rms_signal_rf_papee_MVIC0[6]], [2,rms_signal_rf_papee_MVIC2s[6]], [3,rms_signal_rf_papee_MVIC4m[6]],[4,rms_signal_rf_papee_MVIC8m[6]],[5,rms_signal_rf_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,rms_signal_rf_papee_MVIC0[7]], [2,rms_signal_rf_papee_MVIC2s[7]], [3,rms_signal_rf_papee_MVIC4m[7]],[4,rms_signal_rf_papee_MVIC8m[7]],[5,rms_signal_rf_papee_MVIC12m[7]], "gold", "S8")
newline([1,rms_signal_rf_papee_MVIC0[8]], [2,rms_signal_rf_papee_MVIC2s[8]], [3,rms_signal_rf_papee_MVIC4m[8]],[4,rms_signal_rf_papee_MVIC8m[8]],[5,rms_signal_rf_papee_MVIC12m[8]], "red", "S9")
newline([1,rms_signal_rf_papee_MVIC0[9]], [2,rms_signal_rf_papee_MVIC2s[9]], [3,rms_signal_rf_papee_MVIC4m[9]],[4,rms_signal_rf_papee_MVIC8m[9]],[5,rms_signal_rf_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("RMS RF - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,5), ylabel='Activity in mV')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 5, 0.1), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[47]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rms_signal_vm_papee_MVIC0[0]], [2,rms_signal_vm_papee_MVIC2s[0]], [3,rms_signal_vm_papee_MVIC4m[0]],[4,rms_signal_vm_papee_MVIC8m[0]],[5,rms_signal_vm_papee_MVIC12m[0]], "green", "S1")
newline([1,rms_signal_vm_papee_MVIC0[1]], [2,rms_signal_vm_papee_MVIC2s[1]], [3,rms_signal_vm_papee_MVIC4m[1]],[4,rms_signal_vm_papee_MVIC8m[1]],[5,rms_signal_vm_papee_MVIC12m[1]], "lime", "S2")
newline([1,rms_signal_vm_papee_MVIC0[2]], [2,rms_signal_vm_papee_MVIC2s[2]], [3,rms_signal_vm_papee_MVIC4m[2]],[4,rms_signal_vm_papee_MVIC8m[2]],[5,rms_signal_vm_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,rms_signal_vm_papee_MVIC0[3]], [2,rms_signal_vm_papee_MVIC2s[3]], [3,rms_signal_vm_papee_MVIC4m[3]],[4,rms_signal_vm_papee_MVIC8m[3]],[5,rms_signal_vm_papee_MVIC12m[3]], "blue", "S4")
newline([1,rms_signal_vm_papee_MVIC0[4]], [2,rms_signal_vm_papee_MVIC2s[4]], [3,rms_signal_vm_papee_MVIC4m[4]],[4,rms_signal_vm_papee_MVIC8m[4]],[5,rms_signal_vm_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,rms_signal_vm_papee_MVIC0[5]], [2,rms_signal_vm_papee_MVIC2s[5]], [3,rms_signal_vm_papee_MVIC4m[5]],[4,rms_signal_vm_papee_MVIC8m[5]],[5,rms_signal_vm_papee_MVIC12m[5]], "purple", "S6")
newline([1,rms_signal_vm_papee_MVIC0[6]], [2,rms_signal_vm_papee_MVIC2s[6]], [3,rms_signal_vm_papee_MVIC4m[6]],[4,rms_signal_vm_papee_MVIC8m[6]],[5,rms_signal_vm_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,rms_signal_vm_papee_MVIC0[7]], [2,rms_signal_vm_papee_MVIC2s[7]], [3,rms_signal_vm_papee_MVIC4m[7]],[4,rms_signal_vm_papee_MVIC8m[7]],[5,rms_signal_vm_papee_MVIC12m[7]], "gold", "S8")
newline([1,rms_signal_vm_papee_MVIC0[8]], [2,rms_signal_vm_papee_MVIC2s[8]], [3,rms_signal_vm_papee_MVIC4m[8]],[4,rms_signal_vm_papee_MVIC8m[8]],[5,rms_signal_vm_papee_MVIC12m[8]], "red", "S9")
newline([1,rms_signal_vm_papee_MVIC0[9]], [2,rms_signal_vm_papee_MVIC2s[9]], [3,rms_signal_vm_papee_MVIC4m[9]],[4,rms_signal_vm_papee_MVIC8m[9]],[5,rms_signal_vm_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("RMS VM - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,5), ylabel='Activity in mV')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 5, 0.1), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[48]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rms_signal_vl_papee_MVIC0[0]], [2,rms_signal_vl_papee_MVIC2s[0]], [3,rms_signal_vl_papee_MVIC4m[0]],[4,rms_signal_vl_papee_MVIC8m[0]],[5,rms_signal_vl_papee_MVIC12m[0]], "green", "S1")
newline([1,rms_signal_vl_papee_MVIC0[1]], [2,rms_signal_vl_papee_MVIC2s[1]], [3,rms_signal_vl_papee_MVIC4m[1]],[4,rms_signal_vl_papee_MVIC8m[1]],[5,rms_signal_vl_papee_MVIC12m[1]], "lime", "S2")
newline([1,rms_signal_vl_papee_MVIC0[2]], [2,rms_signal_vl_papee_MVIC2s[2]], [3,rms_signal_vl_papee_MVIC4m[2]],[4,rms_signal_vl_papee_MVIC8m[2]],[5,rms_signal_vl_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,rms_signal_vl_papee_MVIC0[3]], [2,rms_signal_vl_papee_MVIC2s[3]], [3,rms_signal_vl_papee_MVIC4m[3]],[4,rms_signal_vl_papee_MVIC8m[3]],[5,rms_signal_vl_papee_MVIC12m[3]], "blue", "S4")
newline([1,rms_signal_vl_papee_MVIC0[4]], [2,rms_signal_vl_papee_MVIC2s[4]], [3,rms_signal_vl_papee_MVIC4m[4]],[4,rms_signal_vl_papee_MVIC8m[4]],[5,rms_signal_vl_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,rms_signal_vl_papee_MVIC0[5]], [2,rms_signal_vl_papee_MVIC2s[5]], [3,rms_signal_vl_papee_MVIC4m[5]],[4,rms_signal_vl_papee_MVIC8m[5]],[5,rms_signal_vl_papee_MVIC12m[5]], "purple", "S6")
newline([1,rms_signal_vl_papee_MVIC0[6]], [2,rms_signal_vl_papee_MVIC2s[6]], [3,rms_signal_vl_papee_MVIC4m[6]],[4,rms_signal_vl_papee_MVIC8m[6]],[5,rms_signal_vl_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,rms_signal_vl_papee_MVIC0[7]], [2,rms_signal_vl_papee_MVIC2s[7]], [3,rms_signal_vl_papee_MVIC4m[7]],[4,rms_signal_vl_papee_MVIC8m[7]],[5,rms_signal_vl_papee_MVIC12m[7]], "gold", "S8")
newline([1,rms_signal_vl_papee_MVIC0[8]], [2,rms_signal_vl_papee_MVIC2s[8]], [3,rms_signal_vl_papee_MVIC4m[8]],[4,rms_signal_vl_papee_MVIC8m[8]],[5,rms_signal_vl_papee_MVIC12m[8]], "red", "S9")
newline([1,rms_signal_vl_papee_MVIC0[9]], [2,rms_signal_vl_papee_MVIC2s[9]], [3,rms_signal_vl_papee_MVIC4m[9]],[4,rms_signal_vl_papee_MVIC8m[9]],[5,rms_signal_vl_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("RMS VL - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,5), ylabel='Activity in mV')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 5, 0.1), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[49]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,fft_mean_rf_papee_MVIC0[0]], [2,fft_mean_rf_papee_MVIC2s[0]], [3,fft_mean_rf_papee_MVIC4m[0]],[4,fft_mean_rf_papee_MVIC8m[0]],[5,fft_mean_rf_papee_MVIC12m[0]], "green", "S1")
newline([1,fft_mean_rf_papee_MVIC0[1]], [2,fft_mean_rf_papee_MVIC2s[1]], [3,fft_mean_rf_papee_MVIC4m[1]],[4,fft_mean_rf_papee_MVIC8m[1]],[5,fft_mean_rf_papee_MVIC12m[1]], "lime", "S2")
newline([1,fft_mean_rf_papee_MVIC0[2]], [2,fft_mean_rf_papee_MVIC2s[2]], [3,fft_mean_rf_papee_MVIC4m[2]],[4,fft_mean_rf_papee_MVIC8m[2]],[5,fft_mean_rf_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,fft_mean_rf_papee_MVIC0[3]], [2,fft_mean_rf_papee_MVIC2s[3]], [3,fft_mean_rf_papee_MVIC4m[3]],[4,fft_mean_rf_papee_MVIC8m[3]],[5,fft_mean_rf_papee_MVIC12m[3]], "blue", "S4")
newline([1,fft_mean_rf_papee_MVIC0[4]], [2,fft_mean_rf_papee_MVIC2s[4]], [3,fft_mean_rf_papee_MVIC4m[4]],[4,fft_mean_rf_papee_MVIC8m[4]],[5,fft_mean_rf_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,fft_mean_rf_papee_MVIC0[5]], [2,fft_mean_rf_papee_MVIC2s[5]], [3,fft_mean_rf_papee_MVIC4m[5]],[4,fft_mean_rf_papee_MVIC8m[5]],[5,fft_mean_rf_papee_MVIC12m[5]], "purple", "S6")
newline([1,fft_mean_rf_papee_MVIC0[6]], [2,fft_mean_rf_papee_MVIC2s[6]], [3,fft_mean_rf_papee_MVIC4m[6]],[4,fft_mean_rf_papee_MVIC8m[6]],[5,fft_mean_rf_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,fft_mean_rf_papee_MVIC0[7]], [2,fft_mean_rf_papee_MVIC2s[7]], [3,fft_mean_rf_papee_MVIC4m[7]],[4,fft_mean_rf_papee_MVIC8m[7]],[5,fft_mean_rf_papee_MVIC12m[7]], "gold", "S8")
newline([1,fft_mean_rf_papee_MVIC0[8]], [2,fft_mean_rf_papee_MVIC2s[8]], [3,fft_mean_rf_papee_MVIC4m[8]],[4,fft_mean_rf_papee_MVIC8m[8]],[5,fft_mean_rf_papee_MVIC12m[8]], "red", "S9")
newline([1,fft_mean_rf_papee_MVIC0[9]], [2,fft_mean_rf_papee_MVIC2s[9]], [3,fft_mean_rf_papee_MVIC4m[9]],[4,fft_mean_rf_papee_MVIC8m[9]],[5,fft_mean_rf_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("FFT RF - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[50]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,fft_mean_vm_papee_MVIC0[0]], [2,fft_mean_vm_papee_MVIC2s[0]], [3,fft_mean_vm_papee_MVIC4m[0]],[4,fft_mean_vm_papee_MVIC8m[0]],[5,fft_mean_vm_papee_MVIC12m[0]], "green", "S1")
newline([1,fft_mean_vm_papee_MVIC0[1]], [2,fft_mean_vm_papee_MVIC2s[1]], [3,fft_mean_vm_papee_MVIC4m[1]],[4,fft_mean_vm_papee_MVIC8m[1]],[5,fft_mean_vm_papee_MVIC12m[1]], "lime", "S2")
newline([1,fft_mean_vm_papee_MVIC0[2]], [2,fft_mean_vm_papee_MVIC2s[2]], [3,fft_mean_vm_papee_MVIC4m[2]],[4,fft_mean_vm_papee_MVIC8m[2]],[5,fft_mean_vm_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,fft_mean_vm_papee_MVIC0[3]], [2,fft_mean_vm_papee_MVIC2s[3]], [3,fft_mean_vm_papee_MVIC4m[3]],[4,fft_mean_vm_papee_MVIC8m[3]],[5,fft_mean_vm_papee_MVIC12m[3]], "blue", "S4")
newline([1,fft_mean_vm_papee_MVIC0[4]], [2,fft_mean_vm_papee_MVIC2s[4]], [3,fft_mean_vm_papee_MVIC4m[4]],[4,fft_mean_vm_papee_MVIC8m[4]],[5,fft_mean_vm_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,fft_mean_vm_papee_MVIC0[5]], [2,fft_mean_vm_papee_MVIC2s[5]], [3,fft_mean_vm_papee_MVIC4m[5]],[4,fft_mean_vm_papee_MVIC8m[5]],[5,fft_mean_vm_papee_MVIC12m[5]], "purple", "S6")
newline([1,fft_mean_vm_papee_MVIC0[6]], [2,fft_mean_vm_papee_MVIC2s[6]], [3,fft_mean_vm_papee_MVIC4m[6]],[4,fft_mean_vm_papee_MVIC8m[6]],[5,fft_mean_vm_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,fft_mean_vm_papee_MVIC0[7]], [2,fft_mean_vm_papee_MVIC2s[7]], [3,fft_mean_vm_papee_MVIC4m[7]],[4,fft_mean_vm_papee_MVIC8m[7]],[5,fft_mean_vm_papee_MVIC12m[7]], "gold", "S8")
newline([1,fft_mean_vm_papee_MVIC0[8]], [2,fft_mean_vm_papee_MVIC2s[8]], [3,fft_mean_vm_papee_MVIC4m[8]],[4,fft_mean_vm_papee_MVIC8m[8]],[5,fft_mean_vm_papee_MVIC12m[8]], "red", "S9")
newline([1,fft_mean_vm_papee_MVIC0[9]], [2,fft_mean_vm_papee_MVIC2s[9]], [3,fft_mean_vm_papee_MVIC4m[9]],[4,fft_mean_vm_papee_MVIC8m[9]],[5,fft_mean_vm_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("FFT VM - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[51]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,fft_mean_vl_papee_MVIC0[0]], [2,fft_mean_vl_papee_MVIC2s[0]], [3,fft_mean_vl_papee_MVIC4m[0]],[4,fft_mean_vl_papee_MVIC8m[0]],[5,fft_mean_vl_papee_MVIC12m[0]], "green", "S1")
newline([1,fft_mean_vl_papee_MVIC0[1]], [2,fft_mean_vl_papee_MVIC2s[1]], [3,fft_mean_vl_papee_MVIC4m[1]],[4,fft_mean_vl_papee_MVIC8m[1]],[5,fft_mean_vl_papee_MVIC12m[1]], "lime", "S2")
newline([1,fft_mean_vl_papee_MVIC0[2]], [2,fft_mean_vl_papee_MVIC2s[2]], [3,fft_mean_vl_papee_MVIC4m[2]],[4,fft_mean_vl_papee_MVIC8m[2]],[5,fft_mean_vl_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,fft_mean_vl_papee_MVIC0[3]], [2,fft_mean_vl_papee_MVIC2s[3]], [3,fft_mean_vl_papee_MVIC4m[3]],[4,fft_mean_vl_papee_MVIC8m[3]],[5,fft_mean_vl_papee_MVIC12m[3]], "blue", "S4")
newline([1,fft_mean_vl_papee_MVIC0[4]], [2,fft_mean_vl_papee_MVIC2s[4]], [3,fft_mean_vl_papee_MVIC4m[4]],[4,fft_mean_vl_papee_MVIC8m[4]],[5,fft_mean_vl_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,fft_mean_vl_papee_MVIC0[5]], [2,fft_mean_vl_papee_MVIC2s[5]], [3,fft_mean_vl_papee_MVIC4m[5]],[4,fft_mean_vl_papee_MVIC8m[5]],[5,fft_mean_vl_papee_MVIC12m[5]], "purple", "S6")
newline([1,fft_mean_vl_papee_MVIC0[6]], [2,fft_mean_vl_papee_MVIC2s[6]], [3,fft_mean_vl_papee_MVIC4m[6]],[4,fft_mean_vl_papee_MVIC8m[6]],[5,fft_mean_vl_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,fft_mean_vl_papee_MVIC0[7]], [2,fft_mean_vl_papee_MVIC2s[7]], [3,fft_mean_vl_papee_MVIC4m[7]],[4,fft_mean_vl_papee_MVIC8m[7]],[5,fft_mean_vl_papee_MVIC12m[7]], "gold", "S8")
newline([1,fft_mean_vl_papee_MVIC0[8]], [2,fft_mean_vl_papee_MVIC2s[8]], [3,fft_mean_vl_papee_MVIC4m[8]],[4,fft_mean_vl_papee_MVIC8m[8]],[5,fft_mean_vl_papee_MVIC12m[8]], "red", "S9")
newline([1,fft_mean_vl_papee_MVIC0[9]], [2,fft_mean_vl_papee_MVIC2s[9]], [3,fft_mean_vl_papee_MVIC4m[9]],[4,fft_mean_vl_papee_MVIC8m[9]],[5,fft_mean_vl_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("FFT VL - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,220), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 210, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[52]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,median_fft_rf_papee_MVIC0[0]], [2,median_fft_rf_papee_MVIC2s[0]], [3,median_fft_rf_papee_MVIC4m[0]],[4,median_fft_rf_papee_MVIC8m[0]],[5,median_fft_rf_papee_MVIC12m[0]], "green", "S1")
newline([1,median_fft_rf_papee_MVIC0[1]], [2,median_fft_rf_papee_MVIC2s[1]], [3,median_fft_rf_papee_MVIC4m[1]],[4,median_fft_rf_papee_MVIC8m[1]],[5,median_fft_rf_papee_MVIC12m[1]], "lime", "S2")
newline([1,median_fft_rf_papee_MVIC0[2]], [2,median_fft_rf_papee_MVIC2s[2]], [3,median_fft_rf_papee_MVIC4m[2]],[4,median_fft_rf_papee_MVIC8m[2]],[5,median_fft_rf_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,median_fft_rf_papee_MVIC0[3]], [2,median_fft_rf_papee_MVIC2s[3]], [3,median_fft_rf_papee_MVIC4m[3]],[4,median_fft_rf_papee_MVIC8m[3]],[5,median_fft_rf_papee_MVIC12m[3]], "blue", "S4")
newline([1,median_fft_rf_papee_MVIC0[4]], [2,median_fft_rf_papee_MVIC2s[4]], [3,median_fft_rf_papee_MVIC4m[4]],[4,median_fft_rf_papee_MVIC8m[4]],[5,median_fft_rf_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,median_fft_rf_papee_MVIC0[5]], [2,median_fft_rf_papee_MVIC2s[5]], [3,median_fft_rf_papee_MVIC4m[5]],[4,median_fft_rf_papee_MVIC8m[5]],[5,median_fft_rf_papee_MVIC12m[5]], "purple", "S6")
newline([1,median_fft_rf_papee_MVIC0[6]], [2,median_fft_rf_papee_MVIC2s[6]], [3,median_fft_rf_papee_MVIC4m[6]],[4,median_fft_rf_papee_MVIC8m[6]],[5,median_fft_rf_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,median_fft_rf_papee_MVIC0[7]], [2,median_fft_rf_papee_MVIC2s[7]], [3,median_fft_rf_papee_MVIC4m[7]],[4,median_fft_rf_papee_MVIC8m[7]],[5,median_fft_rf_papee_MVIC12m[7]], "gold", "S8")
newline([1,median_fft_rf_papee_MVIC0[8]], [2,median_fft_rf_papee_MVIC2s[8]], [3,median_fft_rf_papee_MVIC4m[8]],[4,median_fft_rf_papee_MVIC8m[8]],[5,median_fft_rf_papee_MVIC12m[8]], "red", "S9")
newline([1,median_fft_rf_papee_MVIC0[9]], [2,median_fft_rf_papee_MVIC2s[9]], [3,median_fft_rf_papee_MVIC4m[9]],[4,median_fft_rf_papee_MVIC8m[9]],[5,median_fft_rf_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Median FFT RF - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[53]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,median_fft_vm_papee_MVIC0[0]], [2,median_fft_vm_papee_MVIC2s[0]], [3,median_fft_vm_papee_MVIC4m[0]],[4,median_fft_vm_papee_MVIC8m[0]],[5,median_fft_vm_papee_MVIC12m[0]], "green", "S1")
newline([1,median_fft_vm_papee_MVIC0[1]], [2,median_fft_vm_papee_MVIC2s[1]], [3,median_fft_vm_papee_MVIC4m[1]],[4,median_fft_vm_papee_MVIC8m[1]],[5,median_fft_vm_papee_MVIC12m[1]], "lime", "S2")
newline([1,median_fft_vm_papee_MVIC0[2]], [2,median_fft_vm_papee_MVIC2s[2]], [3,median_fft_vm_papee_MVIC4m[2]],[4,median_fft_vm_papee_MVIC8m[2]],[5,median_fft_vm_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,median_fft_vm_papee_MVIC0[3]], [2,median_fft_vm_papee_MVIC2s[3]], [3,median_fft_vm_papee_MVIC4m[3]],[4,median_fft_vm_papee_MVIC8m[3]],[5,median_fft_vm_papee_MVIC12m[3]], "blue", "S4")
newline([1,median_fft_vm_papee_MVIC0[4]], [2,median_fft_vm_papee_MVIC2s[4]], [3,median_fft_vm_papee_MVIC4m[4]],[4,median_fft_vm_papee_MVIC8m[4]],[5,median_fft_vm_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,median_fft_vm_papee_MVIC0[5]], [2,median_fft_vm_papee_MVIC2s[5]], [3,median_fft_vm_papee_MVIC4m[5]],[4,median_fft_vm_papee_MVIC8m[5]],[5,median_fft_vm_papee_MVIC12m[5]], "purple", "S6")
newline([1,median_fft_vm_papee_MVIC0[6]], [2,median_fft_vm_papee_MVIC2s[6]], [3,median_fft_vm_papee_MVIC4m[6]],[4,median_fft_vm_papee_MVIC8m[6]],[5,median_fft_vm_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,median_fft_vm_papee_MVIC0[7]], [2,median_fft_vm_papee_MVIC2s[7]], [3,median_fft_vm_papee_MVIC4m[7]],[4,median_fft_vm_papee_MVIC8m[7]],[5,median_fft_vm_papee_MVIC12m[7]], "gold", "S8")
newline([1,median_fft_vm_papee_MVIC0[8]], [2,median_fft_vm_papee_MVIC2s[8]], [3,median_fft_vm_papee_MVIC4m[8]],[4,median_fft_vm_papee_MVIC8m[8]],[5,median_fft_vm_papee_MVIC12m[8]], "red", "S9")
newline([1,median_fft_vm_papee_MVIC0[9]], [2,median_fft_vm_papee_MVIC2s[9]], [3,median_fft_vm_papee_MVIC4m[9]],[4,median_fft_vm_papee_MVIC8m[9]],[5,median_fft_vm_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Median FFT VM - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[54]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,median_fft_vl_papee_MVIC0[0]], [2,median_fft_vl_papee_MVIC2s[0]], [3,median_fft_vl_papee_MVIC4m[0]],[4,median_fft_vl_papee_MVIC8m[0]],[5,median_fft_vl_papee_MVIC12m[0]], "green", "S1")
newline([1,median_fft_vl_papee_MVIC0[1]], [2,median_fft_vl_papee_MVIC2s[1]], [3,median_fft_vl_papee_MVIC4m[1]],[4,median_fft_vl_papee_MVIC8m[1]],[5,median_fft_vl_papee_MVIC12m[1]], "lime", "S2")
newline([1,median_fft_vl_papee_MVIC0[2]], [2,median_fft_vl_papee_MVIC2s[2]], [3,median_fft_vl_papee_MVIC4m[2]],[4,median_fft_vl_papee_MVIC8m[2]],[5,median_fft_vl_papee_MVIC12m[2]], "turquoise", "S3")
newline([1,median_fft_vl_papee_MVIC0[3]], [2,median_fft_vl_papee_MVIC2s[3]], [3,median_fft_vl_papee_MVIC4m[3]],[4,median_fft_vl_papee_MVIC8m[3]],[5,median_fft_vl_papee_MVIC12m[3]], "blue", "S4")
newline([1,median_fft_vl_papee_MVIC0[4]], [2,median_fft_vl_papee_MVIC2s[4]], [3,median_fft_vl_papee_MVIC4m[4]],[4,median_fft_vl_papee_MVIC8m[4]],[5,median_fft_vl_papee_MVIC12m[4]], "darkviolet", "S5")
newline([1,median_fft_vl_papee_MVIC0[5]], [2,median_fft_vl_papee_MVIC2s[5]], [3,median_fft_vl_papee_MVIC4m[5]],[4,median_fft_vl_papee_MVIC8m[5]],[5,median_fft_vl_papee_MVIC12m[5]], "purple", "S6")
newline([1,median_fft_vl_papee_MVIC0[6]], [2,median_fft_vl_papee_MVIC2s[6]], [3,median_fft_vl_papee_MVIC4m[6]],[4,median_fft_vl_papee_MVIC8m[6]],[5,median_fft_vl_papee_MVIC12m[6]], "darkorange", "S7")
newline([1,median_fft_vl_papee_MVIC0[7]], [2,median_fft_vl_papee_MVIC2s[7]], [3,median_fft_vl_papee_MVIC4m[7]],[4,median_fft_vl_papee_MVIC8m[7]],[5,median_fft_vl_papee_MVIC12m[7]], "gold", "S8")
newline([1,median_fft_vl_papee_MVIC0[8]], [2,median_fft_vl_papee_MVIC2s[8]], [3,median_fft_vl_papee_MVIC4m[8]],[4,median_fft_vl_papee_MVIC8m[8]],[5,median_fft_vl_papee_MVIC12m[8]], "red", "S9")
newline([1,median_fft_vl_papee_MVIC0[9]], [2,median_fft_vl_papee_MVIC2s[9]], [3,median_fft_vl_papee_MVIC4m[9]],[4,median_fft_vl_papee_MVIC8m[9]],[5,median_fft_vl_papee_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Median FFT VL - PAPEE-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,220), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 220, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# 
# ## Fertige PAPEV Werte 

# In[252]:


rms_signal_rf_papev_MVIC0 = [2.41239493, 1.61249573, 0.40230233, 0.22029701, 2.70692235, 0.61470863, 1.94409255, 0.22694911, 0.43589251, 3.27223363,1.50249573]
rms_signal_vm_papev_MVIC0 = [1.96072749, 1.83565207, 0.30842023, 0.34040039, 3.19751007, 2.42955227, 1.88919832, 0.34808889, 0.39710743, 3.47070214, 1.48919832 ]
rms_signal_vl_papev_MVIC0 = [1.33591849, 1.41166666, 0.24071557, 0.20173233, 1.99397467, 0.51634334, 1.6582171 , 0.31579751, 0.50323352, 3.61487434]

fft_mean_rf_papev_MVIC0 = [ 93.01271484,  90.78919906,  73.56102277,  82.85572186, 68.94874614, 105.35389118,  71.52064102,  71.45368788, 64.03987966,  79.08561706]
fft_mean_vm_papev_MVIC0 = [91.62114984, 75.88359165, 91.72199442, 76.95715597, 56.33638075, 87.17050551, 87.66404205, 75.90738022, 60.30100031, 74.21539697]
fft_mean_vl_papev_MVIC0 = [84.0042989 , 63.76372293, 74.61035808, 71.56559111, 57.34734977, 81.61756218, 77.88590292, 63.71943954, 57.99678869, 64.43211218]

median_fft_rf_papev_MVIC0 = [82.4795082 , 76.51109411, 62.79558592, 70.06874669, 59.22792173, 79.68236583, 57.99192152, 60.10719755, 57.7689243 , 62.91903043]
median_fft_vm_papev_MVIC0 = [79.14959016, 58.53098699, 76.45822386, 65.57377049, 48.38709677, 58.05038335, 71.8407386 , 62.97856049, 53.21570859, 62.40330067]
median_fft_vl_papev_MVIC0 = [67.11065574, 56.04437643, 60.95638466, 57.11263882, 50.50237969, 67.36035049, 66.07039815, 56.85298622, 51.5082527 , 56.21454358]

max_torque_papev_MVIC0 = [164.02239386, 212.46358586, 174.73857936, 185.09728791, 154.07094735, 140.60261987, 105.9236194 , 140.80757091, 135.41225164, 112.40349923]

rfd20_70_papev_MVIC0 = [150.58798006, 405.04309014, 369.2528769 , 547.7580105 , 523.62528102, 310.12687864, 339.40035087, 557.50149938, 478.87804055, 482.73844342, 380.00305014]


# In[253]:


rms_signal_rf_papev_MVIC2s = [1.95123757, 1.5380736 , 0.38247805, 0.24878829, 2.77818373, 0.73241267, 1.87999185, 0.30867589, 0.4441347 , 2.91412467, 1.4980736]
rms_signal_vm_papev_MVIC2s = [1.64637111, 1.731312  , 0.2873496 , 0.30911963, 3.15966317, 2.23822346, 2.16835033, 0.49931689, 0.37991661, 3.16249105, 2.06835033]
rms_signal_vl_papev_MVIC2s = [1.27405929, 1.28445148, 0.25704785, 0.22852066, 2.0376784 , 0.48418985, 2.08618949, 0.47040727, 0.48546144, 3.25432872]

fft_mean_rf_papev_MVIC2s = [ 93.08142073,  87.26659674,  75.08029186,  80.51374432, 70.17001429, 106.00947715,  64.10584719,  62.10714308, 59.18326911,  76.35269485]
fft_mean_vm_papev_MVIC2s = [99.51384827, 75.71877102, 89.83465521, 74.84228154, 59.31291439, 91.55375946, 76.65003661, 73.86385369, 61.15431021, 74.52534261]
fft_mean_vl_papev_MVIC2s = [83.36430893, 65.81319539, 73.03876316, 66.40820523, 60.94197941, 88.63631253, 71.02928043, 63.62152531, 58.96354049, 65.84927228]

median_fft_rf_papev_MVIC2s = [81.74523571, 72.81334536, 61.64383562, 68.49315068, 60.80722252, 85.0447604 , 51.26140633, 54.12691829, 53.222945  , 64.        ]
median_fft_vm_papev_MVIC2s = [88.01404213, 56.80793508, 71.00456621, 62.78538813, 53.10674456, 66.3507109 , 64.94900698, 62.8369971 , 50.56179775, 64.        ]
median_fft_vl_papev_MVIC2s = [69.70912738, 56.80793508, 62.78538813, 56.50684932, 55.49654806, 75.30279094, 62.53354804, 56.20074658, 49.37906564, 58.85714286]

max_torque_papev_MVIC2s = [162.3921693 , 213.26512781, 166.99628832, 171.44041982, 161.69265369, 136.66138609,  80.35242163, 153.64754619, 123.47040713,  93.54538544]

rfd20_70_papev_MVIC2s = [277.40182849, 483.62755635, 374.41164011, 650.30004851, 564.48485876, 298.46715689, 379.64366971, 545.1452255 , 649.25700208, 578.77625096, 460.49755635]


# In[254]:


rms_signal_rf_papev_MVIC4m = [2.16866421, 1.59204929, 0.31325839, 0.20043366, 2.4655453 , 0.44754133, 1.45796099, 0.23753424, 0.31957134, 2.64200724, 1.49204929]
rms_signal_vm_papev_MVIC4m = [1.91206654, 1.78644613, 0.2536742 , 0.21141722, 2.87456537, 1.83613987, 1.9406616 , 0.37835446, 0.30983378, 2.81821732, 1.5406616]
rms_signal_vl_papev_MVIC4m = [1.25639123, 1.38189861, 0.21702799, 0.17741438, 1.79423196, 0.42360301, 1.75294361, 0.37397114, 0.36493163, 2.9089995 ]

fft_mean_rf_papev_MVIC4m = [95.91973062, 94.90660361, 78.48819155, 86.71598741, 74.86594602, 96.35041499, 79.50900761, 68.66972532, 67.39597258, 85.65063747]
fft_mean_vm_papev_MVIC4m = [107.34829813,  81.90081012,  93.39283171,  77.46576148, 58.63190454,  92.10948569,  85.86766826,  80.33564951, 61.66684637,  78.75083968]
fft_mean_vl_papev_MVIC4m = [84.94565222, 67.41067753, 75.24251189, 69.8016425 , 61.8307992 , 84.45195092, 75.80221218, 66.00185779, 61.49487488, 66.2185399 ]

median_fft_rf_papev_MVIC4m = [83.69098712, 80.02832861, 61.79458239, 75.28409091, 65.4077723 , 70.35478052, 66.52806653, 60.43956044, 58.67490174, 76.06324973]
median_fft_vm_papev_MVIC4m = [98.71244635, 65.6279509 , 73.08126411, 60.22727273, 52.54515599, 67.94948888, 76.92307692, 68.35164835, 52.49859629, 72.24645583]
median_fft_vl_papev_MVIC4m = [69.44986344, 58.30972616, 62.35891648, 54.54545455, 53.91351943, 73.66205652, 67.04781705, 60.21978022, 52.21785514, 57.79716467]

max_torque_papev_MVIC4m = [170.64868984, 220.52836667, 157.40217321, 172.58033107, 176.64228592, 128.30451958,  65.24785762, 150.64462521, 121.08748244,  98.51584269]

rfd20_70_papev_MVIC4m = [102.07044199, 458.73625003, 274.04427668, 408.56188136, 318.76012787, 346.77782067, 292.41221822, 653.05393311, 533.12213623, 551.72529362, 450.78465003]


# In[255]:


rms_signal_rf_papev_MVIC8m = [2.02637293, 1.47098517, 0.35637428, 0.20984836, 2.56727053, 0.44721008, 1.63743183, 0.25511038, 0.31728007, 2.59228133, 1.50098517]
rms_signal_vm_papev_MVIC8m = [1.75678067, 1.68767855, 0.26902148, 0.26328008, 3.09916187, 1.77966288, 1.96937326, 0.37998951, 0.29431639, 2.92096063, 1.86937326]
rms_signal_vl_papev_MVIC8m = [1.26520015, 1.38514743, 0.24677007, 0.1856072 , 2.03915895, 0.42088649, 1.84962566, 0.38051045, 0.37951619, 3.06005763]

fft_mean_rf_papev_MVIC8m = [98.96028301, 93.77380397, 76.79079458, 83.30582656, 71.46170275, 95.12783983, 72.40938654, 69.57775853, 67.63419696, 86.16787957]
fft_mean_vm_papev_MVIC8m = [106.05128461,  77.28790233,  93.94479255,  81.81774201, 59.34276811, 100.46571355,  82.3151431 ,  81.07709011, 62.0580786 ,  77.01420688]
fft_mean_vl_papev_MVIC8m = [83.51889104, 64.22103137, 69.23571473, 69.94725725, 61.29449089, 87.63893764, 72.56587137, 65.99016664, 60.90988896, 66.74817025]

median_fft_rf_papev_MVIC8m = [86.03639241, 77.10790738, 58.04480652, 67.06966033, 60.6557377 , 73.87862797, 59.06522856, 62.21765914, 58.25510768, 75.46636518]
median_fft_vm_papev_MVIC8m = [94.54113924, 61.38051551, 82.73930754, 69.08462867, 54.3715847 , 75.98944591, 71.39188495, 71.25256674, 54.11374931, 68.68287168]
median_fft_vl_papev_MVIC8m = [67.84018987, 59.63302752, 54.9898167 , 58.43408175, 55.73770492, 79.94722955, 61.63328197, 59.54825462, 54.38983987, 60.48615037]

max_torque_papev_MVIC8m = [161.35605398, 212.63996687, 174.02849283, 165.82419085, 188.93413232, 128.36527079,  73.16920348, 155.81838838, 118.48888123,  97.73418806]

rfd20_70_papev_MVIC8m = [ 93.73400207, 393.23557135, 235.83966779, 519.70839324, 306.68514906, 266.01313022, 330.39434063, 583.94581163, 460.40639848, 539.40285462, 420.86421135]


# In[256]:


rms_signal_rf_papev_MVIC12m = [2.05133115, 1.5940487 , 0.36900524, 0.19034212, 2.69714876, 0.49567222, 1.49229417, 0.24742227, 0.36535239, 2.63130874, 1.5340487 ]
rms_signal_vm_papev_MVIC12m = [1.76719295, 1.76787837, 0.29390716, 0.2651166 , 2.92636853, 1.76084974, 1.95332544, 0.37985316, 0.34499667, 3.00429416, 1.95332544]
rms_signal_vl_papev_MVIC12m = [1.26009847, 1.35658366, 0.24433132, 0.15944467, 1.83232196, 0.42486673, 1.71158034, 0.34342062, 0.41442246, 2.97638577]

fft_mean_rf_papev_MVIC12m = [ 98.87935018,  97.18094251,  73.80135494,  86.12417241, 72.21317589, 100.27950487,  79.6338269 ,  70.34915397,  64.37991395,  85.20012145]
fft_mean_vm_papev_MVIC12m = [106.24664738,  84.04272468,  88.00799495,  78.9157717 , 58.36498031,  96.85217081,  85.32622487,  79.74239191, 61.4946397 ,  77.33901284]
fft_mean_vl_papev_MVIC12m = [83.53111025, 65.6049152 , 72.82545609, 73.19012148, 61.29455663, 83.81764903, 77.8127172 , 67.16810558, 62.20781986, 66.46501083]

median_fft_rf_papev_MVIC12m = [88.28328982, 83.26010545, 59.50305144, 72.60726073, 63.87546967, 74.78753541, 66.08054453, 61.01922217, 56.17433414, 72.66260163]
median_fft_vm_papev_MVIC12m = [96.7689295 , 63.92794376, 73.45248474, 66.83168317, 51.52979066, 73.0878187 , 76.57402155, 67.50111757, 51.08958838, 69.3597561 ]
median_fft_vl_papev_MVIC12m = [66.25326371, 54.48154657, 59.06713165, 57.20572057, 53.40848094,72.52124646, 71.46908678, 60.34868127, 54.23728814, 58.94308943]
 
max_torque_papev_MVIC12m = [172.13981776, 212.87610646, 179.92305469, 160.82793639, 173.4485308 , 131.1739871 ,  64.8707437 , 145.99996105, 121.7320809 ,  94.83499844]

rfd20_70_papev_MVIC12m = [ 80.67188864, 561.81358529, 217.51705512, 394.47101805, 520.06834607, 335.19352312, 303.1637885 , 494.8632274 , 351.14027339, 572.76653288, 400.2688529]


# In[213]:


#anova Torque PAP

from statsmodels.stats.anova import AnovaRM



df_torquepapev = pd.DataFrame({'Subject': np.tile([1, 2, 3, 4, 5 ,6 ,7 ,8 ,9 ,10, 11], 5),
                   'Time': np.repeat([1, 2, 3, 4, 5],11),
                   'Torque':  [164.02239386, 212.46358586, 174.73857936, 185.09728791, 154.07094735, 140.60261987, 105.9236194 , 140.80757091, 135.41225164, 112.40349923, 210.46955586, 
162.3921693 , 213.26512781, 166.99628832, 171.44041982, 161.69265369, 136.66138609,  80.35242163, 153.64754619, 123.47040713,  93.54538544, 211.26512751, 
170.64868984, 220.52836667, 157.40217321, 172.58033107, 176.64228592, 128.30451958,  65.24785762, 150.64462521, 121.08748244,  98.51584269, 219.51836668, 
161.35605398, 212.63996687, 174.02849283, 165.82419085, 188.93413232, 128.36527079,  73.16920348, 155.81838838, 118.48888123,  97.73418806, 209.49966879, 
161.35605398, 212.63996687, 174.02849283, 165.82419085, 188.93413232, 128.36527079,  73.16920348, 155.81838838, 118.48888123,  97.73418806, 210.93116680]})

print(AnovaRM(data=df_torquepapev, depvar='Torque', subject='Subject', within=['Time']).fit())


# In[191]:


#anova Torque RFD

from statsmodels.stats.anova import AnovaRM



df_RFDpapev = pd.DataFrame({'Subject': np.tile([1, 2, 3, 4, 5 ,6 ,7 ,8 ,9 ,10, 11], 5),
                   'Time': np.repeat([1, 2, 3, 4, 5],11),
                   'RFD':  [150.58798006, 405.04309014, 369.2528769 , 547.7580105 , 523.62528102, 310.12687864, 339.40035087, 557.50149938, 478.87804055, 482.73844342, 399.00305014, 
277.40182849, 483.62755635, 374.41164011, 650.30004851, 564.48485876, 298.46715689, 379.64366971, 545.1452255 , 649.25700208, 578.77625096, 472.49755635, 
102.07044199, 458.73625003, 274.04427668, 408.56188136, 318.76012787, 346.77782067, 292.41221822, 653.05393311, 533.12213623, 551.72529362, 442.78465003, 
93.73400207, 393.23557135, 235.83966779, 519.70839324, 306.68514906, 266.01313022, 330.39434063, 583.94581163, 460.40639848, 539.40285462, 402.86421135, 
80.67188864, 561.81358529, 217.51705512, 394.47101805, 520.06834607, 335.19352312, 303.1637885 , 494.8632274 , 351.14027339, 572.76653288, 491.2688529]})

print(AnovaRM(data=df_RFDpapev, depvar='RFD', subject='Subject', within=['Time']).fit())


# In[192]:


import pingouin as pg
post_hocs = pg.pairwise_ttests(dv='RFD', within='Time', subject='Subject', padjust='fdr_bh', data=df_RFDpapev)
post_hocs
#Perform multiple pairwise comparisons (t test) and corrections (Benjamini/Hochberg FDR correction),


# In[193]:


#tukey test als post-hoc TORQUE
#Dataframe erstellen zum Speichern von Daten
df = pd.DataFrame({'score': [   150.58798006, 405.04309014, 369.2528769 , 547.7580105 , 523.62528102, 310.12687864, 339.40035087, 557.50149938, 478.87804055, 482.73844342, 399.00305014, 
277.40182849, 483.62755635, 374.41164011, 650.30004851, 564.48485876, 298.46715689, 379.64366971, 545.1452255 , 649.25700208, 578.77625096, 472.49755635, 
102.07044199, 458.73625003, 274.04427668, 408.56188136, 318.76012787, 346.77782067, 292.41221822, 653.05393311, 533.12213623, 551.72529362, 442.78465003, 
93.73400207, 393.23557135, 235.83966779, 519.70839324, 306.68514906, 266.01313022, 330.39434063, 583.94581163, 460.40639848, 539.40285462, 402.86421135, 
80.67188864, 561.81358529, 217.51705512, 394.47101805, 520.06834607, 335.19352312, 303.1637885 , 494.8632274 , 351.14027339, 572.76653288, 491.2688529],
                   'group': np.repeat(['PRE', '2s', '4m', '8m', '12m'], repeats=11)}) 







tukey = pairwise_tukeyhsd(endog = df['score'], groups = df['group'], alpha = 0.05)
tukey.plot_simultaneous()

tukey.summary()


# In[224]:


#anova Torque RMS RF

from statsmodels.stats.anova import AnovaRM



df_RMSRFpapev = pd.DataFrame({'Subject': np.tile([1, 2, 3, 4, 5 ,6 ,7 ,8 ,9 ,10, 11], 5),
                   'Time': np.repeat([1, 2, 3, 4, 5],11),
                   'RMS_RF':  [2.41239493, 1.61249573, 0.40230233, 0.22029701, 2.70692235, 0.61470863, 1.94409255, 0.22694911, 0.43589251, 3.27223363, 1.50249573, 
1.95123757, 1.5380736 , 0.38247805, 0.24878829, 2.77818373, 0.73241267, 1.87999185, 0.30867589, 0.4441347 , 2.91412467, 1.5080736,
2.16866421, 1.59204929, 0.31325839, 0.20043366, 2.4655453 , 0.44754133, 1.45796099, 0.23753424, 0.31957134, 2.64200724, 1.45204929, 
2.02637293, 1.47098517, 0.35637428, 0.20984836, 2.56727053, 0.44721008, 1.63743183, 0.25511038, 0.31728007, 2.59228133, 1.40098517, 
2.05133115, 1.5940487 , 0.36900524, 0.19034212, 2.69714876, 0.49567222, 1.49229417, 0.24742227, 0.36535239, 2.63130874, 1.4940487 ]})

print(AnovaRM(data=df_RMSRFpapev, depvar='RMS_RF', subject='Subject', within=['Time']).fit())


# In[225]:


import pingouin as pg
post_hocs = pg.pairwise_ttests(dv='RMS_RF', within='Time', subject='Subject', padjust='fdr_bh', data=df_RMSRFpapev)
post_hocs
#Perform multiple pairwise comparisons (t test) and corrections (Benjamini/Hochberg FDR correction),


# In[246]:


#anova Torque RMS VM

from statsmodels.stats.anova import AnovaRM



df_RMSVMpapev = pd.DataFrame({'Subject': np.tile([1, 2, 3, 4, 5 ,6 ,7 ,8 ,9, 10,11], 5),
                   'Time': np.repeat([1, 2, 3, 4, 5],11),
                   'RMS_VM':  [1.96072749, 1.83565207, 0.30842023, 0.34040039, 3.19751007, 2.42955227, 1.88919832, 0.34808889, 0.39710743, 3.47070214, 1.78919832, 
1.64637111, 1.731312  , 0.2873496 , 0.30911963, 3.15966317, 2.23822346, 2.16835033, 0.49931689, 0.37991661, 3.16249105, 2.06835033, 
1.91206654, 1.78644613, 0.2536742 , 0.21141722, 2.87456537, 1.83613987, 1.9406616 , 0.37835446, 0.30983378, 2.81821732, 1.8406616,
1.75678067, 1.68767855, 0.26902148, 0.26328008, 3.09916187, 1.77966288, 1.96937326, 0.37998951, 0.29431639, 2.92096063, 1.86937326, 
1.76719295, 1.76787837, 0.29390716, 0.2651166 , 2.92636853, 1.76084974, 1.95332544, 0.37985316, 0.34499667, 3.00429416, 1.85332544 ]})

print(AnovaRM(data=df_RMSVMpapev, depvar='RMS_VM', subject='Subject', within=['Time']).fit())


# In[249]:


import pingouin as pg
post_hocs = pg.pairwise_ttests(dv='RMS_VM', within='Time', subject='Subject', padjust='fdr_bh', data=df_RMSVMpapev)
post_hocs
#Perform multiple pairwise comparisons (t test) and corrections (Benjamini/Hochberg FDR correction),


# In[258]:


#anova Torque RMS VL

from statsmodels.stats.anova import AnovaRM



df_RMSVLpapev = pd.DataFrame({'Subject': np.tile([1, 2, 3, 4, 5 ,6 ,7 ,8 ,9, 10,11], 5),
                   'Time': np.repeat([1, 2, 3, 4, 5],11),
                   'RMS_VL':  [1.33591849, 1.41166666, 0.24071557, 0.20173233, 1.99397467, 0.51634334, 1.6582171 , 0.31579751, 0.50323352, 3.61487434, 0.24071557, 
1.27405929, 1.28445148, 0.25704785, 0.22852066, 2.0376784 , 0.48418985, 2.08618949, 0.47040727, 0.48546144, 3.25432872, 0.25704785, 
1.25639123, 1.38189861, 0.21702799, 0.17741438, 1.79423196, 0.42360301, 1.75294361, 0.37397114, 0.36493163, 2.9089995, 0.21702799, 
1.26520015, 1.38514743, 0.24677007, 0.1856072 , 2.03915895, 0.42088649, 1.84962566, 0.38051045, 0.37951619, 3.06005763, 0.24677007, 
1.26009847, 1.35658366, 0.24433132, 0.15944467, 1.83232196, 0.42486673, 1.71158034, 0.34342062, 0.41442246, 2.97638577, 0.24433132 ]})

print(AnovaRM(data=df_RMSVLpapev, depvar='RMS_VL', subject='Subject', within=['Time']).fit())


# In[259]:


import pingouin as pg
post_hocs = pg.pairwise_ttests(dv='RMS_VL', within='Time', subject='Subject', padjust='fdr_bh', data=df_RMSVLpapev)
post_hocs
#Perform multiple pairwise comparisons (t test) and corrections (Benjamini/Hochberg FDR correction),


# In[62]:


stats.f_oneway(median_fft_vl_papev_MVIC0, median_fft_vl_papev_MVIC2s, median_fft_vl_papev_MVIC4m, median_fft_vl_papev_MVIC8m, median_fft_vl_papev_MVIC12m)


# In[63]:


stats.f_oneway(fft_mean_vl_papev_MVIC0, fft_mean_vl_papev_MVIC2s, fft_mean_vl_papev_MVIC4m, fft_mean_vl_papev_MVIC8m, fft_mean_vl_papev_MVIC12m)


# In[64]:


stats.f_oneway(rms_signal_vl_papev_MVIC0, rms_signal_vl_papev_MVIC2s, rms_signal_vl_papev_MVIC4m, rms_signal_vl_papev_MVIC8m, rms_signal_vl_papev_MVIC12m)


# In[65]:


stats.f_oneway(rms_signal_vm_papev_MVIC0, rms_signal_vm_papev_MVIC2s, rms_signal_vm_papev_MVIC4m, rms_signal_vm_papev_MVIC8m, rms_signal_vm_papev_MVIC12m)


# In[66]:


stats.f_oneway(fft_mean_vm_papev_MVIC0, fft_mean_vm_papev_MVIC2s, fft_mean_vm_papev_MVIC4m, fft_mean_vm_papev_MVIC8m, fft_mean_vm_papev_MVIC12m)


# In[67]:


stats.f_oneway(median_fft_vm_papev_MVIC0, median_fft_vm_papev_MVIC2s, median_fft_vm_papev_MVIC4m, median_fft_vm_papev_MVIC8m, median_fft_vm_papev_MVIC12m)


# In[68]:


stats.f_oneway(fft_mean_rf_papev_MVIC0, fft_mean_rf_papev_MVIC2s, fft_mean_rf_papev_MVIC4m, fft_mean_rf_papev_MVIC8m, fft_mean_rf_papev_MVIC12m)


# In[69]:


stats.f_oneway(median_fft_rf_papev_MVIC0, median_fft_rf_papev_MVIC2s, median_fft_rf_papev_MVIC4m, median_fft_rf_papev_MVIC8m, median_fft_rf_papev_MVIC12m)


# In[70]:


stats.f_oneway(rms_signal_rf_papev_MVIC0, rms_signal_rf_papev_MVIC2s, rms_signal_rf_papev_MVIC4m, rms_signal_rf_papev_MVIC8m, rms_signal_rf_papev_MVIC12m)


# In[72]:


# Average of Torque Values Time-grouped
mean_max_torque_papev_MVIC0 = np.mean(max_torque_papev_MVIC0)
mean_max_torque_papev_MVIC2s = np.mean(max_torque_papev_MVIC2s)
mean_max_torque_papev_MVIC4m= np.mean(max_torque_papev_MVIC4m)
mean_max_torque_papev_MVIC8m = np.mean(max_torque_papev_MVIC8m)
mean_max_torque_papev_MVIC12m = np.mean(max_torque_papev_MVIC12m)

means_of_papev_torque = np.array([mean_max_torque_papev_MVIC0, mean_max_torque_papev_MVIC2s, mean_max_torque_papev_MVIC4m, mean_max_torque_papev_MVIC8m, mean_max_torque_papev_MVIC12m])


# In[194]:


# Average of RFD Values Time-grouped
mean_rfd20_70_papev_MVIC0 = np.mean(rfd20_70_papev_MVIC0)
mean_rfd20_70_papev_MVIC2s = np.mean(rfd20_70_papev_MVIC2s)
mean_rfd20_70_papev_MVIC4m= np.mean(rfd20_70_papev_MVIC4m)
mean_rfd20_70_papev_MVIC8m = np.mean(rfd20_70_papev_MVIC8m)
mean_rfd20_70_papev_MVIC12m = np.mean(rfd20_70_papev_MVIC12m)

means_of_papev_rfd = np.array([mean_rfd20_70_papev_MVIC0, mean_rfd20_70_papev_MVIC2s, mean_rfd20_70_papev_MVIC4m, mean_rfd20_70_papev_MVIC8m, mean_rfd20_70_papev_MVIC12m])


# In[74]:


# Average of RMS RF Values Time-grouped
mean_rms_signal_rf_papev_MVIC0 = np.mean(rms_signal_rf_papev_MVIC0)
mean_rms_signal_rf_papev_MVIC2s = np.mean(rms_signal_rf_papev_MVIC2s)
mean_rms_signal_rf_papev_MVIC4m= np.mean(rms_signal_rf_papev_MVIC4m)
mean_rms_signal_rf_papev_MVIC8m = np.mean(rms_signal_rf_papev_MVIC8m)
mean_rms_signal_rf_papev_MVIC12m = np.mean(rms_signal_rf_papev_MVIC12m)

mean_rms_signal_rf_papev = np.array([mean_rms_signal_rf_papev_MVIC0, mean_rms_signal_rf_papev_MVIC2s, mean_rms_signal_rf_papev_MVIC4m, mean_rms_signal_rf_papev_MVIC8m, mean_rms_signal_rf_papev_MVIC12m])


# Average of RMS VM Values Time-grouped
mean_rms_signal_vm_papev_MVIC0 = np.mean(rms_signal_vm_papev_MVIC0)
mean_rms_signal_vm_papev_MVIC2s = np.mean(rms_signal_vm_papev_MVIC2s)
mean_rms_signal_vm_papev_MVIC4m= np.mean(rms_signal_vm_papev_MVIC4m)
mean_rms_signal_vm_papev_MVIC8m = np.mean(rms_signal_vm_papev_MVIC8m)
mean_rms_signal_vm_papev_MVIC12m = np.mean(rms_signal_vm_papev_MVIC12m)

mean_rms_signal_vm_papev = np.array([mean_rms_signal_vm_papev_MVIC0, mean_rms_signal_vm_papev_MVIC2s, mean_rms_signal_vm_papev_MVIC4m, mean_rms_signal_vm_papev_MVIC8m, mean_rms_signal_vm_papev_MVIC12m])


# Average of RMS VL Values Time-grouped
mean_rms_signal_vl_papev_MVIC0 = np.mean(rms_signal_vl_papev_MVIC0)
mean_rms_signal_vl_papev_MVIC2s = np.mean(rms_signal_vl_papev_MVIC2s)
mean_rms_signal_vl_papev_MVIC4m= np.mean(rms_signal_vl_papev_MVIC4m)
mean_rms_signal_vl_papev_MVIC8m = np.mean(rms_signal_vl_papev_MVIC8m)
mean_rms_signal_vl_papev_MVIC12m = np.mean(rms_signal_vl_papev_MVIC12m)

mean_rms_signal_vl_papev = np.array([mean_rms_signal_vl_papev_MVIC0, mean_rms_signal_vl_papev_MVIC2s, mean_rms_signal_vl_papev_MVIC4m, mean_rms_signal_vl_papev_MVIC8m, mean_rms_signal_vl_papev_MVIC12m])


# Average of FFT RF Values Time-grouped
mean_fft_mean_rf_papev_MVIC0 = np.mean(fft_mean_rf_papev_MVIC0)
mean_fft_mean_rf_papev_MVIC2s = np.mean(fft_mean_rf_papev_MVIC2s)
mean_fft_mean_rf_papev_MVIC4m= np.mean(fft_mean_rf_papev_MVIC4m)
mean_fft_mean_rf_papev_MVIC8m = np.mean(fft_mean_rf_papev_MVIC8m)
mean_fft_mean_rf_papev_MVIC12m = np.mean(fft_mean_rf_papev_MVIC12m)

mean_fft_mean_rf_papev = np.array([mean_fft_mean_rf_papev_MVIC0, mean_fft_mean_rf_papev_MVIC2s, mean_fft_mean_rf_papev_MVIC4m, mean_fft_mean_rf_papev_MVIC8m, mean_fft_mean_rf_papev_MVIC12m])

# Average of FFT VM Values Time-grouped
mean_fft_mean_vm_papev_MVIC0 = np.mean(fft_mean_vm_papev_MVIC0)
mean_fft_mean_vm_papev_MVIC2s = np.mean(fft_mean_vm_papev_MVIC2s)
mean_fft_mean_vm_papev_MVIC4m= np.mean(fft_mean_vm_papev_MVIC4m)
mean_fft_mean_vm_papev_MVIC8m = np.mean(fft_mean_vm_papev_MVIC8m)
mean_fft_mean_vm_papev_MVIC12m = np.mean(fft_mean_vm_papev_MVIC12m)

mean_fft_mean_vm_papev = np.array([mean_fft_mean_vm_papev_MVIC0, mean_fft_mean_vm_papev_MVIC2s, mean_fft_mean_vm_papev_MVIC4m, mean_fft_mean_vm_papev_MVIC8m, mean_fft_mean_vm_papev_MVIC12m])


# Average of FFT VL Values Time-grouped
mean_fft_mean_vl_papev_MVIC0 = np.mean(fft_mean_vl_papev_MVIC0)
mean_fft_mean_vl_papev_MVIC2s = np.mean(fft_mean_vl_papev_MVIC2s)
mean_fft_mean_vl_papev_MVIC4m= np.mean(fft_mean_vl_papev_MVIC4m)
mean_fft_mean_vl_papev_MVIC8m = np.mean(fft_mean_vl_papev_MVIC8m)
mean_fft_mean_vl_papev_MVIC12m = np.mean(fft_mean_vl_papev_MVIC12m)

mean_fft_mean_vl_papev = np.array([mean_fft_mean_vl_papev_MVIC0, mean_fft_mean_vl_papev_MVIC2s, mean_fft_mean_vl_papev_MVIC4m, mean_fft_mean_vl_papev_MVIC8m, mean_fft_mean_vl_papev_MVIC12m])


# Average of MEDIAN FFT RF Values Time-grouped
mean_median_fft_rf_papev_MVIC0 = np.mean(median_fft_rf_papev_MVIC0)
mean_median_fft_rf_papev_MVIC2s = np.mean(median_fft_rf_papev_MVIC2s)
mean_median_fft_rf_papev_MVIC4m= np.mean(median_fft_rf_papev_MVIC4m)
mean_median_fft_rf_papev_MVIC8m = np.mean(median_fft_rf_papev_MVIC8m)
mean_median_fft_rf_papev_MVIC12m = np.mean(median_fft_rf_papev_MVIC12m)

mean_median_fft_rf_papev = np.array([mean_median_fft_rf_papev_MVIC0, mean_median_fft_rf_papev_MVIC2s, mean_median_fft_rf_papev_MVIC4m, mean_median_fft_rf_papev_MVIC8m, mean_median_fft_rf_papev_MVIC12m])


# Average of MEDIAN FFT VM Values Time-grouped
mean_median_fft_vm_papev_MVIC0 = np.mean(median_fft_vm_papev_MVIC0)
mean_median_fft_vm_papev_MVIC2s = np.mean(median_fft_vm_papev_MVIC2s)
mean_median_fft_vm_papev_MVIC4m= np.mean(median_fft_vm_papev_MVIC4m)
mean_median_fft_vm_papev_MVIC8m = np.mean(median_fft_vm_papev_MVIC8m)
mean_median_fft_vm_papev_MVIC12m = np.mean(median_fft_vm_papev_MVIC12m)

mean_median_fft_vm_papev = np.array([mean_median_fft_vm_papev_MVIC0, mean_median_fft_vm_papev_MVIC2s, mean_median_fft_vm_papev_MVIC4m, mean_median_fft_vm_papev_MVIC8m, mean_median_fft_vm_papev_MVIC12m])


# Average of MEDIAN FFT VL Values Time-grouped
mean_median_fft_vl_papev_MVIC0 = np.mean(median_fft_vl_papev_MVIC0)
mean_median_fft_vl_papev_MVIC2s = np.mean(median_fft_vl_papev_MVIC2s)
mean_median_fft_vl_papev_MVIC4m= np.mean(median_fft_vl_papev_MVIC4m)
mean_median_fft_vl_papev_MVIC8m = np.mean(median_fft_vl_papev_MVIC8m)
mean_median_fft_vl_papev_MVIC12m = np.mean(median_fft_vl_papev_MVIC12m)

mean_median_fft_vl_papev = np.array([mean_median_fft_vl_papev_MVIC0, mean_median_fft_vl_papev_MVIC2s, mean_median_fft_vl_papev_MVIC4m, mean_median_fft_vl_papev_MVIC8m, mean_median_fft_vl_papev_MVIC12m])



# In[ ]:





# In[75]:


#TORQUE BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
means_of_papev_t = means_of_papev_torque.astype(int)
x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, means_of_papev_t, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Torque in Nm')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 250])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[195]:


#RFD BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
means_of_rfd2070 = means_of_papev_rfd.astype(int)
x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, means_of_rfd2070, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RFD 20%-70% in Nm/s')
ax.set_xlabel('Measurement timepoints')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 650])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[217]:


#RMS RF BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_rms_signal_rf_papev = np.round(mean_rms_signal_rf_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_rms_signal_rf_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMS Rectus Femoris')
ax.set_xlabel('Measurement timepoints')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 2])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[251]:


#RMS VM BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_rms_signal_vm_papev = np.round(mean_rms_signal_vm_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_rms_signal_vm_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMS Vastus Medialis')
ax.set_xlabel('Measurement timepoints')

ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 2])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[79]:


#RMS VL BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_rms_signal_vl_papev = np.round(mean_rms_signal_vl_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_rms_signal_vl_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RMS Vastus Lateralis')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 2])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[80]:


#FFT Mean RF BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_fft_mean_rf_papev = np.round(mean_fft_mean_rf_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_fft_mean_rf_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Mean RF')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[81]:


#FFT Mean VM BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_fft_mean_vm_papev = np.round(mean_fft_mean_vm_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_fft_mean_vm_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Mean VM')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[82]:


#FFT Mean VL 
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_fft_mean_vl_papev = np.round(mean_fft_mean_vl_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_fft_mean_vl_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Mean VL')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[83]:


#FFT Mean VL 
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_median_fft_rf_papev = np.round(mean_median_fft_rf_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_median_fft_rf_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Median RF')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[84]:


#FFT Median VM
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_median_fft_vm_papev = np.round(mean_median_fft_vm_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_median_fft_vm_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Median VM')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[85]:


#FFT Median VL 
bars = "PRE", "2sec", "4min", "8min", "12min"
mean_median_fft_vl_papev = np.round(mean_median_fft_vl_papev,2)

x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, mean_median_fft_vl_papev, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('FFT Median VL')
ax.set_title('PAPEV - Trial')
ax.set_ylim([0, 100])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[86]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,max_torque_papev_MVIC0[0]], [2,max_torque_papev_MVIC2s[0]], [3,max_torque_papev_MVIC4m[0]],[4,max_torque_papev_MVIC8m[0]],[5,max_torque_papev_MVIC12m[0]], "green", "S1")
newline([1,max_torque_papev_MVIC0[1]], [2,max_torque_papev_MVIC2s[1]], [3,max_torque_papev_MVIC4m[1]],[4,max_torque_papev_MVIC8m[1]],[5,max_torque_papev_MVIC12m[1]], "lime", "S2")
newline([1,max_torque_papev_MVIC0[2]], [2,max_torque_papev_MVIC2s[2]], [3,max_torque_papev_MVIC4m[2]],[4,max_torque_papev_MVIC8m[2]],[5,max_torque_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,max_torque_papev_MVIC0[3]], [2,max_torque_papev_MVIC2s[3]], [3,max_torque_papev_MVIC4m[3]],[4,max_torque_papev_MVIC8m[3]],[5,max_torque_papev_MVIC12m[3]], "blue", "S4")
newline([1,max_torque_papev_MVIC0[4]], [2,max_torque_papev_MVIC2s[4]], [3,max_torque_papev_MVIC4m[4]],[4,max_torque_papev_MVIC8m[4]],[5,max_torque_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,max_torque_papev_MVIC0[5]], [2,max_torque_papev_MVIC2s[5]], [3,max_torque_papev_MVIC4m[5]],[4,max_torque_papev_MVIC8m[5]],[5,max_torque_papev_MVIC12m[5]], "purple", "S6")
newline([1,max_torque_papev_MVIC0[6]], [2,max_torque_papev_MVIC2s[6]], [3,max_torque_papev_MVIC4m[6]],[4,max_torque_papev_MVIC8m[6]],[5,max_torque_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,max_torque_papev_MVIC0[7]], [2,max_torque_papev_MVIC2s[7]], [3,max_torque_papev_MVIC4m[7]],[4,max_torque_papev_MVIC8m[7]],[5,max_torque_papev_MVIC12m[7]], "gold", "S8")
newline([1,max_torque_papev_MVIC0[8]], [2,max_torque_papev_MVIC2s[8]], [3,max_torque_papev_MVIC4m[8]],[4,max_torque_papev_MVIC8m[8]],[5,max_torque_papev_MVIC12m[8]], "red", "S9")
newline([1,max_torque_papev_MVIC0[9]], [2,max_torque_papev_MVIC2s[9]], [3,max_torque_papev_MVIC4m[9]],[4,max_torque_papev_MVIC8m[9]],[5,max_torque_papev_MVIC12m[9]], "black", "S10")
#newline([1,max_torque_papev_MVIC0[10]], [2,max_torque_papev_MVIC2s[10]], [3,max_torque_papev_MVIC4m[10]],[4,max_torque_papev_MVIC8m[10]],[5,max_torque_papev_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Torque - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,300), ylabel='Torque in Nm')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 300, 50), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[211]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rfd20_70_papev_MVIC0[0]], [2,rfd20_70_papev_MVIC2s[0]], [3,rfd20_70_papev_MVIC4m[0]],[4,rfd20_70_papev_MVIC8m[0]],[5,rfd20_70_papev_MVIC12m[0]], "green", "S1")
newline([1,rfd20_70_papev_MVIC0[1]], [2,rfd20_70_papev_MVIC2s[1]], [3,rfd20_70_papev_MVIC4m[1]],[4,rfd20_70_papev_MVIC8m[1]],[5,rfd20_70_papev_MVIC12m[1]], "lime", "S2")
newline([1,rfd20_70_papev_MVIC0[2]], [2,rfd20_70_papev_MVIC2s[2]], [3,rfd20_70_papev_MVIC4m[2]],[4,rfd20_70_papev_MVIC8m[2]],[5,rfd20_70_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,rfd20_70_papev_MVIC0[3]], [2,rfd20_70_papev_MVIC2s[3]], [3,rfd20_70_papev_MVIC4m[3]],[4,rfd20_70_papev_MVIC8m[3]],[5,rfd20_70_papev_MVIC12m[3]], "blue", "S4")
newline([1,rfd20_70_papev_MVIC0[4]], [2,rfd20_70_papev_MVIC2s[4]], [3,rfd20_70_papev_MVIC4m[4]],[4,rfd20_70_papev_MVIC8m[4]],[5,rfd20_70_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,rfd20_70_papev_MVIC0[5]], [2,rfd20_70_papev_MVIC2s[5]], [3,rfd20_70_papev_MVIC4m[5]],[4,rfd20_70_papev_MVIC8m[5]],[5,rfd20_70_papev_MVIC12m[5]], "purple", "S6")
newline([1,rfd20_70_papev_MVIC0[6]], [2,rfd20_70_papev_MVIC2s[6]], [3,rfd20_70_papev_MVIC4m[6]],[4,rfd20_70_papev_MVIC8m[6]],[5,rfd20_70_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,rfd20_70_papev_MVIC0[7]], [2,rfd20_70_papev_MVIC2s[7]], [3,rfd20_70_papev_MVIC4m[7]],[4,rfd20_70_papev_MVIC8m[7]],[5,rfd20_70_papev_MVIC12m[7]], "gold", "S8")
newline([1,rfd20_70_papev_MVIC0[8]], [2,rfd20_70_papev_MVIC2s[8]], [3,rfd20_70_papev_MVIC4m[8]],[4,rfd20_70_papev_MVIC8m[8]],[5,rfd20_70_papev_MVIC12m[8]], "red", "S9")
newline([1,rfd20_70_papev_MVIC0[9]], [2,rfd20_70_papev_MVIC2s[9]], [3,rfd20_70_papev_MVIC4m[9]],[4,rfd20_70_papev_MVIC8m[9]],[5,rfd20_70_papev_MVIC12m[9]], "black", "S10")
newline([1,rfd20_70_papev_MVIC0[1]], [2,rfd20_70_papev_MVIC2s[10]], [3,rfd20_70_papev_MVIC4m[10]],[4,rfd20_70_papev_MVIC8m[10]],[5,rfd20_70_papev_MVIC12m[10]], "pink", "S11")





# Decoration
ax.set_title("Rate of Force Developement 20%-70% - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,1000), xlabel="Measurement timepoints", ylabel='RFD 20%-70% in Nm/s')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(100, 1000, 50), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[241]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rms_signal_rf_papev_MVIC0[0]], [2,rms_signal_rf_papev_MVIC2s[0]], [3,rms_signal_rf_papev_MVIC4m[0]],[4,rms_signal_rf_papev_MVIC8m[0]],[5,rms_signal_rf_papev_MVIC12m[0]], "green", "S1")
newline([1,rms_signal_rf_papev_MVIC0[1]], [2,rms_signal_rf_papev_MVIC2s[1]], [3,rms_signal_rf_papev_MVIC4m[1]],[4,rms_signal_rf_papev_MVIC8m[1]],[5,rms_signal_rf_papev_MVIC12m[1]], "lime", "S2")
newline([1,rms_signal_rf_papev_MVIC0[2]], [2,rms_signal_rf_papev_MVIC2s[2]], [3,rms_signal_rf_papev_MVIC4m[2]],[4,rms_signal_rf_papev_MVIC8m[2]],[5,rms_signal_rf_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,rms_signal_rf_papev_MVIC0[3]], [2,rms_signal_rf_papev_MVIC2s[3]], [3,rms_signal_rf_papev_MVIC4m[3]],[4,rms_signal_rf_papev_MVIC8m[3]],[5,rms_signal_rf_papev_MVIC12m[3]], "blue", "S4")
newline([1,rms_signal_rf_papev_MVIC0[4]], [2,rms_signal_rf_papev_MVIC2s[4]], [3,rms_signal_rf_papev_MVIC4m[4]],[4,rms_signal_rf_papev_MVIC8m[4]],[5,rms_signal_rf_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,rms_signal_rf_papev_MVIC0[5]], [2,rms_signal_rf_papev_MVIC2s[5]], [3,rms_signal_rf_papev_MVIC4m[5]],[4,rms_signal_rf_papev_MVIC8m[5]],[5,rms_signal_rf_papev_MVIC12m[5]], "purple", "S6")
newline([1,rms_signal_rf_papev_MVIC0[6]], [2,rms_signal_rf_papev_MVIC2s[6]], [3,rms_signal_rf_papev_MVIC4m[6]],[4,rms_signal_rf_papev_MVIC8m[6]],[5,rms_signal_rf_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,rms_signal_rf_papev_MVIC0[7]], [2,rms_signal_rf_papev_MVIC2s[7]], [3,rms_signal_rf_papev_MVIC4m[7]],[4,rms_signal_rf_papev_MVIC8m[7]],[5,rms_signal_rf_papev_MVIC12m[7]], "gold", "S8")
newline([1,rms_signal_rf_papev_MVIC0[8]], [2,rms_signal_rf_papev_MVIC2s[8]], [3,rms_signal_rf_papev_MVIC4m[8]],[4,rms_signal_rf_papev_MVIC8m[8]],[5,rms_signal_rf_papev_MVIC12m[8]], "red", "S9")
newline([1,rms_signal_rf_papev_MVIC0[9]], [2,rms_signal_rf_papev_MVIC2s[9]], [3,rms_signal_rf_papev_MVIC4m[9]],[4,rms_signal_rf_papev_MVIC8m[9]],[5,rms_signal_rf_papev_MVIC12m[9]], "black", "S10")
newline([1,rms_signal_rf_papev_MVIC0[10]], [2,rms_signal_rf_papev_MVIC2s[10]], [3,rms_signal_rf_papev_MVIC4m[10]],[4,rms_signal_rf_papev_MVIC8m[10]],[5,rms_signal_rf_papev_MVIC12m[10]], "pink", "S11")





# Decoration
ax.set_title("RMS RF - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,4), xlabel="Measurement timepoints", ylabel='Activity in mV')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 4, 0.2), fontsize=12)

# Lighten borders'
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[257]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rms_signal_vm_papev_MVIC0[0]], [2,rms_signal_vm_papev_MVIC2s[0]], [3,rms_signal_vm_papev_MVIC4m[0]],[4,rms_signal_vm_papev_MVIC8m[0]],[5,rms_signal_vm_papev_MVIC12m[0]], "green", "S1")
newline([1,rms_signal_vm_papev_MVIC0[1]], [2,rms_signal_vm_papev_MVIC2s[1]], [3,rms_signal_vm_papev_MVIC4m[1]],[4,rms_signal_vm_papev_MVIC8m[1]],[5,rms_signal_vm_papev_MVIC12m[1]], "lime", "S2")
newline([1,rms_signal_vm_papev_MVIC0[2]], [2,rms_signal_vm_papev_MVIC2s[2]], [3,rms_signal_vm_papev_MVIC4m[2]],[4,rms_signal_vm_papev_MVIC8m[2]],[5,rms_signal_vm_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,rms_signal_vm_papev_MVIC0[3]], [2,rms_signal_vm_papev_MVIC2s[3]], [3,rms_signal_vm_papev_MVIC4m[3]],[4,rms_signal_vm_papev_MVIC8m[3]],[5,rms_signal_vm_papev_MVIC12m[3]], "blue", "S4")
newline([1,rms_signal_vm_papev_MVIC0[4]], [2,rms_signal_vm_papev_MVIC2s[4]], [3,rms_signal_vm_papev_MVIC4m[4]],[4,rms_signal_vm_papev_MVIC8m[4]],[5,rms_signal_vm_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,rms_signal_vm_papev_MVIC0[5]], [2,rms_signal_vm_papev_MVIC2s[5]], [3,rms_signal_vm_papev_MVIC4m[5]],[4,rms_signal_vm_papev_MVIC8m[5]],[5,rms_signal_vm_papev_MVIC12m[5]], "purple", "S6")
newline([1,rms_signal_vm_papev_MVIC0[6]], [2,rms_signal_vm_papev_MVIC2s[6]], [3,rms_signal_vm_papev_MVIC4m[6]],[4,rms_signal_vm_papev_MVIC8m[6]],[5,rms_signal_vm_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,rms_signal_vm_papev_MVIC0[7]], [2,rms_signal_vm_papev_MVIC2s[7]], [3,rms_signal_vm_papev_MVIC4m[7]],[4,rms_signal_vm_papev_MVIC8m[7]],[5,rms_signal_vm_papev_MVIC12m[7]], "gold", "S8")
newline([1,rms_signal_vm_papev_MVIC0[8]], [2,rms_signal_vm_papev_MVIC2s[8]], [3,rms_signal_vm_papev_MVIC4m[8]],[4,rms_signal_vm_papev_MVIC8m[8]],[5,rms_signal_vm_papev_MVIC12m[8]], "red", "S9")
newline([1,rms_signal_vm_papev_MVIC0[9]], [2,rms_signal_vm_papev_MVIC2s[9]], [3,rms_signal_vm_papev_MVIC4m[9]],[4,rms_signal_vm_papev_MVIC8m[9]],[5,rms_signal_vm_papev_MVIC12m[9]], "black", "S10")
newline([1,rms_signal_vm_papev_MVIC0[10]], [2,rms_signal_vm_papev_MVIC2s[10]], [3,rms_signal_vm_papev_MVIC4m[10]],[4,rms_signal_vm_papev_MVIC8m[10]],[5,rms_signal_vm_papev_MVIC12m[10]], "pink", "S11")





# Decoration
ax.set_title("RMS VM - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,5), ylabel='Activity in mV')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 5, 0.1), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[90]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rms_signal_vl_papev_MVIC0[0]], [2,rms_signal_vl_papev_MVIC2s[0]], [3,rms_signal_vl_papev_MVIC4m[0]],[4,rms_signal_vl_papev_MVIC8m[0]],[5,rms_signal_vl_papev_MVIC12m[0]], "green", "S1")
newline([1,rms_signal_vl_papev_MVIC0[1]], [2,rms_signal_vl_papev_MVIC2s[1]], [3,rms_signal_vl_papev_MVIC4m[1]],[4,rms_signal_vl_papev_MVIC8m[1]],[5,rms_signal_vl_papev_MVIC12m[1]], "lime", "S2")
newline([1,rms_signal_vl_papev_MVIC0[2]], [2,rms_signal_vl_papev_MVIC2s[2]], [3,rms_signal_vl_papev_MVIC4m[2]],[4,rms_signal_vl_papev_MVIC8m[2]],[5,rms_signal_vl_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,rms_signal_vl_papev_MVIC0[3]], [2,rms_signal_vl_papev_MVIC2s[3]], [3,rms_signal_vl_papev_MVIC4m[3]],[4,rms_signal_vl_papev_MVIC8m[3]],[5,rms_signal_vl_papev_MVIC12m[3]], "blue", "S4")
newline([1,rms_signal_vl_papev_MVIC0[4]], [2,rms_signal_vl_papev_MVIC2s[4]], [3,rms_signal_vl_papev_MVIC4m[4]],[4,rms_signal_vl_papev_MVIC8m[4]],[5,rms_signal_vl_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,rms_signal_vl_papev_MVIC0[5]], [2,rms_signal_vl_papev_MVIC2s[5]], [3,rms_signal_vl_papev_MVIC4m[5]],[4,rms_signal_vl_papev_MVIC8m[5]],[5,rms_signal_vl_papev_MVIC12m[5]], "purple", "S6")
newline([1,rms_signal_vl_papev_MVIC0[6]], [2,rms_signal_vl_papev_MVIC2s[6]], [3,rms_signal_vl_papev_MVIC4m[6]],[4,rms_signal_vl_papev_MVIC8m[6]],[5,rms_signal_vl_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,rms_signal_vl_papev_MVIC0[7]], [2,rms_signal_vl_papev_MVIC2s[7]], [3,rms_signal_vl_papev_MVIC4m[7]],[4,rms_signal_vl_papev_MVIC8m[7]],[5,rms_signal_vl_papev_MVIC12m[7]], "gold", "S8")
newline([1,rms_signal_vl_papev_MVIC0[8]], [2,rms_signal_vl_papev_MVIC2s[8]], [3,rms_signal_vl_papev_MVIC4m[8]],[4,rms_signal_vl_papev_MVIC8m[8]],[5,rms_signal_vl_papev_MVIC12m[8]], "red", "S9")
newline([1,rms_signal_vl_papev_MVIC0[9]], [2,rms_signal_vl_papev_MVIC2s[9]], [3,rms_signal_vl_papev_MVIC4m[9]],[4,rms_signal_vl_papev_MVIC8m[9]],[5,rms_signal_vl_papev_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("RMS VL - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,5), ylabel='Activity in mV')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 5, 0.1), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[91]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,fft_mean_rf_papev_MVIC0[0]], [2,fft_mean_rf_papev_MVIC2s[0]], [3,fft_mean_rf_papev_MVIC4m[0]],[4,fft_mean_rf_papev_MVIC8m[0]],[5,fft_mean_rf_papev_MVIC12m[0]], "green", "S1")
newline([1,fft_mean_rf_papev_MVIC0[1]], [2,fft_mean_rf_papev_MVIC2s[1]], [3,fft_mean_rf_papev_MVIC4m[1]],[4,fft_mean_rf_papev_MVIC8m[1]],[5,fft_mean_rf_papev_MVIC12m[1]], "lime", "S2")
newline([1,fft_mean_rf_papev_MVIC0[2]], [2,fft_mean_rf_papev_MVIC2s[2]], [3,fft_mean_rf_papev_MVIC4m[2]],[4,fft_mean_rf_papev_MVIC8m[2]],[5,fft_mean_rf_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,fft_mean_rf_papev_MVIC0[3]], [2,fft_mean_rf_papev_MVIC2s[3]], [3,fft_mean_rf_papev_MVIC4m[3]],[4,fft_mean_rf_papev_MVIC8m[3]],[5,fft_mean_rf_papev_MVIC12m[3]], "blue", "S4")
newline([1,fft_mean_rf_papev_MVIC0[4]], [2,fft_mean_rf_papev_MVIC2s[4]], [3,fft_mean_rf_papev_MVIC4m[4]],[4,fft_mean_rf_papev_MVIC8m[4]],[5,fft_mean_rf_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,fft_mean_rf_papev_MVIC0[5]], [2,fft_mean_rf_papev_MVIC2s[5]], [3,fft_mean_rf_papev_MVIC4m[5]],[4,fft_mean_rf_papev_MVIC8m[5]],[5,fft_mean_rf_papev_MVIC12m[5]], "purple", "S6")
newline([1,fft_mean_rf_papev_MVIC0[6]], [2,fft_mean_rf_papev_MVIC2s[6]], [3,fft_mean_rf_papev_MVIC4m[6]],[4,fft_mean_rf_papev_MVIC8m[6]],[5,fft_mean_rf_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,fft_mean_rf_papev_MVIC0[7]], [2,fft_mean_rf_papev_MVIC2s[7]], [3,fft_mean_rf_papev_MVIC4m[7]],[4,fft_mean_rf_papev_MVIC8m[7]],[5,fft_mean_rf_papev_MVIC12m[7]], "gold", "S8")
newline([1,fft_mean_rf_papev_MVIC0[8]], [2,fft_mean_rf_papev_MVIC2s[8]], [3,fft_mean_rf_papev_MVIC4m[8]],[4,fft_mean_rf_papev_MVIC8m[8]],[5,fft_mean_rf_papev_MVIC12m[8]], "red", "S9")
newline([1,fft_mean_rf_papev_MVIC0[9]], [2,fft_mean_rf_papev_MVIC2s[9]], [3,fft_mean_rf_papev_MVIC4m[9]],[4,fft_mean_rf_papev_MVIC8m[9]],[5,fft_mean_rf_papev_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("FFT RF - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[92]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,fft_mean_vm_papev_MVIC0[0]], [2,fft_mean_vm_papev_MVIC2s[0]], [3,fft_mean_vm_papev_MVIC4m[0]],[4,fft_mean_vm_papev_MVIC8m[0]],[5,fft_mean_vm_papev_MVIC12m[0]], "green", "S1")
newline([1,fft_mean_vm_papev_MVIC0[1]], [2,fft_mean_vm_papev_MVIC2s[1]], [3,fft_mean_vm_papev_MVIC4m[1]],[4,fft_mean_vm_papev_MVIC8m[1]],[5,fft_mean_vm_papev_MVIC12m[1]], "lime", "S2")
newline([1,fft_mean_vm_papev_MVIC0[2]], [2,fft_mean_vm_papev_MVIC2s[2]], [3,fft_mean_vm_papev_MVIC4m[2]],[4,fft_mean_vm_papev_MVIC8m[2]],[5,fft_mean_vm_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,fft_mean_vm_papev_MVIC0[3]], [2,fft_mean_vm_papev_MVIC2s[3]], [3,fft_mean_vm_papev_MVIC4m[3]],[4,fft_mean_vm_papev_MVIC8m[3]],[5,fft_mean_vm_papev_MVIC12m[3]], "blue", "S4")
newline([1,fft_mean_vm_papev_MVIC0[4]], [2,fft_mean_vm_papev_MVIC2s[4]], [3,fft_mean_vm_papev_MVIC4m[4]],[4,fft_mean_vm_papev_MVIC8m[4]],[5,fft_mean_vm_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,fft_mean_vm_papev_MVIC0[5]], [2,fft_mean_vm_papev_MVIC2s[5]], [3,fft_mean_vm_papev_MVIC4m[5]],[4,fft_mean_vm_papev_MVIC8m[5]],[5,fft_mean_vm_papev_MVIC12m[5]], "purple", "S6")
newline([1,fft_mean_vm_papev_MVIC0[6]], [2,fft_mean_vm_papev_MVIC2s[6]], [3,fft_mean_vm_papev_MVIC4m[6]],[4,fft_mean_vm_papev_MVIC8m[6]],[5,fft_mean_vm_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,fft_mean_vm_papev_MVIC0[7]], [2,fft_mean_vm_papev_MVIC2s[7]], [3,fft_mean_vm_papev_MVIC4m[7]],[4,fft_mean_vm_papev_MVIC8m[7]],[5,fft_mean_vm_papev_MVIC12m[7]], "gold", "S8")
newline([1,fft_mean_vm_papev_MVIC0[8]], [2,fft_mean_vm_papev_MVIC2s[8]], [3,fft_mean_vm_papev_MVIC4m[8]],[4,fft_mean_vm_papev_MVIC8m[8]],[5,fft_mean_vm_papev_MVIC12m[8]], "red", "S9")
newline([1,fft_mean_vm_papev_MVIC0[9]], [2,fft_mean_vm_papev_MVIC2s[9]], [3,fft_mean_vm_papev_MVIC4m[9]],[4,fft_mean_vm_papev_MVIC8m[9]],[5,fft_mean_vm_papev_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("FFT VM - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[93]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,fft_mean_vl_papev_MVIC0[0]], [2,fft_mean_vl_papev_MVIC2s[0]], [3,fft_mean_vl_papev_MVIC4m[0]],[4,fft_mean_vl_papev_MVIC8m[0]],[5,fft_mean_vl_papev_MVIC12m[0]], "green", "S1")
newline([1,fft_mean_vl_papev_MVIC0[1]], [2,fft_mean_vl_papev_MVIC2s[1]], [3,fft_mean_vl_papev_MVIC4m[1]],[4,fft_mean_vl_papev_MVIC8m[1]],[5,fft_mean_vl_papev_MVIC12m[1]], "lime", "S2")
newline([1,fft_mean_vl_papev_MVIC0[2]], [2,fft_mean_vl_papev_MVIC2s[2]], [3,fft_mean_vl_papev_MVIC4m[2]],[4,fft_mean_vl_papev_MVIC8m[2]],[5,fft_mean_vl_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,fft_mean_vl_papev_MVIC0[3]], [2,fft_mean_vl_papev_MVIC2s[3]], [3,fft_mean_vl_papev_MVIC4m[3]],[4,fft_mean_vl_papev_MVIC8m[3]],[5,fft_mean_vl_papev_MVIC12m[3]], "blue", "S4")
newline([1,fft_mean_vl_papev_MVIC0[4]], [2,fft_mean_vl_papev_MVIC2s[4]], [3,fft_mean_vl_papev_MVIC4m[4]],[4,fft_mean_vl_papev_MVIC8m[4]],[5,fft_mean_vl_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,fft_mean_vl_papev_MVIC0[5]], [2,fft_mean_vl_papev_MVIC2s[5]], [3,fft_mean_vl_papev_MVIC4m[5]],[4,fft_mean_vl_papev_MVIC8m[5]],[5,fft_mean_vl_papev_MVIC12m[5]], "purple", "S6")
newline([1,fft_mean_vl_papev_MVIC0[6]], [2,fft_mean_vl_papev_MVIC2s[6]], [3,fft_mean_vl_papev_MVIC4m[6]],[4,fft_mean_vl_papev_MVIC8m[6]],[5,fft_mean_vl_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,fft_mean_vl_papev_MVIC0[7]], [2,fft_mean_vl_papev_MVIC2s[7]], [3,fft_mean_vl_papev_MVIC4m[7]],[4,fft_mean_vl_papev_MVIC8m[7]],[5,fft_mean_vl_papev_MVIC12m[7]], "gold", "S8")
newline([1,fft_mean_vl_papev_MVIC0[8]], [2,fft_mean_vl_papev_MVIC2s[8]], [3,fft_mean_vl_papev_MVIC4m[8]],[4,fft_mean_vl_papev_MVIC8m[8]],[5,fft_mean_vl_papev_MVIC12m[8]], "red", "S9")
newline([1,fft_mean_vl_papev_MVIC0[9]], [2,fft_mean_vl_papev_MVIC2s[9]], [3,fft_mean_vl_papev_MVIC4m[9]],[4,fft_mean_vl_papev_MVIC8m[9]],[5,fft_mean_vl_papev_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("FFT VL - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[94]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,median_fft_rf_papev_MVIC0[0]], [2,median_fft_rf_papev_MVIC2s[0]], [3,median_fft_rf_papev_MVIC4m[0]],[4,median_fft_rf_papev_MVIC8m[0]],[5,median_fft_rf_papev_MVIC12m[0]], "green", "S1")
newline([1,median_fft_rf_papev_MVIC0[1]], [2,median_fft_rf_papev_MVIC2s[1]], [3,median_fft_rf_papev_MVIC4m[1]],[4,median_fft_rf_papev_MVIC8m[1]],[5,median_fft_rf_papev_MVIC12m[1]], "lime", "S2")
newline([1,median_fft_rf_papev_MVIC0[2]], [2,median_fft_rf_papev_MVIC2s[2]], [3,median_fft_rf_papev_MVIC4m[2]],[4,median_fft_rf_papev_MVIC8m[2]],[5,median_fft_rf_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,median_fft_rf_papev_MVIC0[3]], [2,median_fft_rf_papev_MVIC2s[3]], [3,median_fft_rf_papev_MVIC4m[3]],[4,median_fft_rf_papev_MVIC8m[3]],[5,median_fft_rf_papev_MVIC12m[3]], "blue", "S4")
newline([1,median_fft_rf_papev_MVIC0[4]], [2,median_fft_rf_papev_MVIC2s[4]], [3,median_fft_rf_papev_MVIC4m[4]],[4,median_fft_rf_papev_MVIC8m[4]],[5,median_fft_rf_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,median_fft_rf_papev_MVIC0[5]], [2,median_fft_rf_papev_MVIC2s[5]], [3,median_fft_rf_papev_MVIC4m[5]],[4,median_fft_rf_papev_MVIC8m[5]],[5,median_fft_rf_papev_MVIC12m[5]], "purple", "S6")
newline([1,median_fft_rf_papev_MVIC0[6]], [2,median_fft_rf_papev_MVIC2s[6]], [3,median_fft_rf_papev_MVIC4m[6]],[4,median_fft_rf_papev_MVIC8m[6]],[5,median_fft_rf_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,median_fft_rf_papev_MVIC0[7]], [2,median_fft_rf_papev_MVIC2s[7]], [3,median_fft_rf_papev_MVIC4m[7]],[4,median_fft_rf_papev_MVIC8m[7]],[5,median_fft_rf_papev_MVIC12m[7]], "gold", "S8")
newline([1,median_fft_rf_papev_MVIC0[8]], [2,median_fft_rf_papev_MVIC2s[8]], [3,median_fft_rf_papev_MVIC4m[8]],[4,median_fft_rf_papev_MVIC8m[8]],[5,median_fft_rf_papev_MVIC12m[8]], "red", "S9")
newline([1,median_fft_rf_papev_MVIC0[9]], [2,median_fft_rf_papev_MVIC2s[9]], [3,median_fft_rf_papev_MVIC4m[9]],[4,median_fft_rf_papev_MVIC8m[9]],[5,median_fft_rf_papev_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Median FFT RF - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[95]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,median_fft_vm_papev_MVIC0[0]], [2,median_fft_vm_papev_MVIC2s[0]], [3,median_fft_vm_papev_MVIC4m[0]],[4,median_fft_vm_papev_MVIC8m[0]],[5,median_fft_vm_papev_MVIC12m[0]], "green", "S1")
newline([1,median_fft_vm_papev_MVIC0[1]], [2,median_fft_vm_papev_MVIC2s[1]], [3,median_fft_vm_papev_MVIC4m[1]],[4,median_fft_vm_papev_MVIC8m[1]],[5,median_fft_vm_papev_MVIC12m[1]], "lime", "S2")
newline([1,median_fft_vm_papev_MVIC0[2]], [2,median_fft_vm_papev_MVIC2s[2]], [3,median_fft_vm_papev_MVIC4m[2]],[4,median_fft_vm_papev_MVIC8m[2]],[5,median_fft_vm_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,median_fft_vm_papev_MVIC0[3]], [2,median_fft_vm_papev_MVIC2s[3]], [3,median_fft_vm_papev_MVIC4m[3]],[4,median_fft_vm_papev_MVIC8m[3]],[5,median_fft_vm_papev_MVIC12m[3]], "blue", "S4")
newline([1,median_fft_vm_papev_MVIC0[4]], [2,median_fft_vm_papev_MVIC2s[4]], [3,median_fft_vm_papev_MVIC4m[4]],[4,median_fft_vm_papev_MVIC8m[4]],[5,median_fft_vm_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,median_fft_vm_papev_MVIC0[5]], [2,median_fft_vm_papev_MVIC2s[5]], [3,median_fft_vm_papev_MVIC4m[5]],[4,median_fft_vm_papev_MVIC8m[5]],[5,median_fft_vm_papev_MVIC12m[5]], "purple", "S6")
newline([1,median_fft_vm_papev_MVIC0[6]], [2,median_fft_vm_papev_MVIC2s[6]], [3,median_fft_vm_papev_MVIC4m[6]],[4,median_fft_vm_papev_MVIC8m[6]],[5,median_fft_vm_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,median_fft_vm_papev_MVIC0[7]], [2,median_fft_vm_papev_MVIC2s[7]], [3,median_fft_vm_papev_MVIC4m[7]],[4,median_fft_vm_papev_MVIC8m[7]],[5,median_fft_vm_papev_MVIC12m[7]], "gold", "S8")
newline([1,median_fft_vm_papev_MVIC0[8]], [2,median_fft_vm_papev_MVIC2s[8]], [3,median_fft_vm_papev_MVIC4m[8]],[4,median_fft_vm_papev_MVIC8m[8]],[5,median_fft_vm_papev_MVIC12m[8]], "red", "S9")
newline([1,median_fft_vm_papev_MVIC0[9]], [2,median_fft_vm_papev_MVIC2s[9]], [3,median_fft_vm_papev_MVIC4m[9]],[4,median_fft_vm_papev_MVIC8m[9]],[5,median_fft_vm_papev_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Median FFT VM - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[96]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,median_fft_vl_papev_MVIC0[0]], [2,median_fft_vl_papev_MVIC2s[0]], [3,median_fft_vl_papev_MVIC4m[0]],[4,median_fft_vl_papev_MVIC8m[0]],[5,median_fft_vl_papev_MVIC12m[0]], "green", "S1")
newline([1,median_fft_vl_papev_MVIC0[1]], [2,median_fft_vl_papev_MVIC2s[1]], [3,median_fft_vl_papev_MVIC4m[1]],[4,median_fft_vl_papev_MVIC8m[1]],[5,median_fft_vl_papev_MVIC12m[1]], "lime", "S2")
newline([1,median_fft_vl_papev_MVIC0[2]], [2,median_fft_vl_papev_MVIC2s[2]], [3,median_fft_vl_papev_MVIC4m[2]],[4,median_fft_vl_papev_MVIC8m[2]],[5,median_fft_vl_papev_MVIC12m[2]], "turquoise", "S3")
newline([1,median_fft_vl_papev_MVIC0[3]], [2,median_fft_vl_papev_MVIC2s[3]], [3,median_fft_vl_papev_MVIC4m[3]],[4,median_fft_vl_papev_MVIC8m[3]],[5,median_fft_vl_papev_MVIC12m[3]], "blue", "S4")
newline([1,median_fft_vl_papev_MVIC0[4]], [2,median_fft_vl_papev_MVIC2s[4]], [3,median_fft_vl_papev_MVIC4m[4]],[4,median_fft_vl_papev_MVIC8m[4]],[5,median_fft_vl_papev_MVIC12m[4]], "darkviolet", "S5")
newline([1,median_fft_vl_papev_MVIC0[5]], [2,median_fft_vl_papev_MVIC2s[5]], [3,median_fft_vl_papev_MVIC4m[5]],[4,median_fft_vl_papev_MVIC8m[5]],[5,median_fft_vl_papev_MVIC12m[5]], "purple", "S6")
newline([1,median_fft_vl_papev_MVIC0[6]], [2,median_fft_vl_papev_MVIC2s[6]], [3,median_fft_vl_papev_MVIC4m[6]],[4,median_fft_vl_papev_MVIC8m[6]],[5,median_fft_vl_papev_MVIC12m[6]], "darkorange", "S7")
newline([1,median_fft_vl_papev_MVIC0[7]], [2,median_fft_vl_papev_MVIC2s[7]], [3,median_fft_vl_papev_MVIC4m[7]],[4,median_fft_vl_papev_MVIC8m[7]],[5,median_fft_vl_papev_MVIC12m[7]], "gold", "S8")
newline([1,median_fft_vl_papev_MVIC0[8]], [2,median_fft_vl_papev_MVIC2s[8]], [3,median_fft_vl_papev_MVIC4m[8]],[4,median_fft_vl_papev_MVIC8m[8]],[5,median_fft_vl_papev_MVIC12m[8]], "red", "S9")
newline([1,median_fft_vl_papev_MVIC0[9]], [2,median_fft_vl_papev_MVIC2s[9]], [3,median_fft_vl_papev_MVIC4m[9]],[4,median_fft_vl_papev_MVIC8m[9]],[5,median_fft_vl_papev_MVIC12m[9]], "black", "S10")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Median FFT VL - PAPEV-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,200), ylabel='Power')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 200, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# ## FERTIGE PAP WERTE (MAXTORQUE & RFD 20-70)

# In[97]:


max_torque_pap_MVIC0 = [24.50007750087747, 33.46859090722437, 21.60066413873087, 27.873981806518543, 28.080832140003892, 24.07898348399918, 17.688966862273613, 31.5858175389304, 22.53570736149189, 25.474274516034313, 30.4742746149189]
rfd20_70_pap_MVIC0 = [321.225184809161, 405.6681636034446, 319.33336212953964, 378.2824502702033, 373.34316070133906, 459.7319745639749, 226.57819639238375, 461.6520755140017, 372.9358644421168,364.74597819433865, 459.284651861715]
max_torque_pap_MVIC2s = [37.187478874107065, 51.29697403566756, 29.611657633306532, 40.22994713918414, 37.03404377573669, 38.926916172908356, 20.223952480349777, 45.88119977337147, 39.54458364105578, 27.943120476458812, 44.5314853106584]
rfd20_70_pap_MVIC2s = [474.8758545561396, 603.0440634749075, 465.25128668453186, 521.5202184166586, 511.8225236009397, 703.2194902926152, 284.5436506357287, 727.654996961246, 679.8261094451382,465.3616100766099, 692.4165841321866]
max_torque_pap_MVIC4m = [24.33453760656167, 36.80882639484867, 23.654213529645013, 33.25167890478412, 27.99943302365993, 28.534924149667567, 17.182665058636868, 33.410647380642, 23.691223899644175, 24.027916406490107, 33.35168543148]
rfd20_70_pap_MVIC4m = [309.991529029935, 440.3610766299707, 371.50021210518787, 438.67177121550503, 360.7373123980669, 537.0600590202483, 228.3011616635383, 511.1285945042745, 393.60518736446653,369.94496557980887, 498.2314561321658]
max_torque_pap_MVIC8m = [24.760928115333837, 35.24441701512808, 23.225080793837442, 31.940425485767147, 26.668251321824457, 27.72913338454499, 17.80255352331667, 31.84219112389092, 23.168460665590572, 24.16823487415193, 31.35168513521 ]
rfd20_70_pap_MVIC8m = [321.0011820417995, 429.2950110374058, 355.8829851548642, 428.6549243103872, 344.1007204531485, 516.0211979989399, 231.9298654311742, 488.1333757993711, 386.93254413985466, 356.4234063065124, 480.2165312848531]
max_torque_pap_MVIC12m =[24.906047958128255, 34.37296631068462, 22.44729268579549, 29.891522809658944, 26.292429900497304, 27.491176832711936, 17.593061961950593, 31.26238064593363, 23.036739258716718, 24.06639919717573, 31.212168451648]
rfd20_70_pap_MVIC12m = [322.29795011231795, 417.01187642496905, 342.1973635129611, 404.89523719280476, 337.6208817657014, 527.0434448699397, 224.12384188987247, 479.4850438938416, 387.114773202139, 349.4076804054119, 479.5313854121654]


# In[98]:


24.50007750087747, 33.46859090722437, 21.60066413873087, 27.873981806518543, 28.080832140003892, 24.07898348399918, 17.688966862273613, 31.5858175389304, 22.53570736149189, 25.474274516034313, 30.4742746149189321.225184809161, 405.6681636034446, 319.33336212953964, 378.2824502702033, 373.34316070133906, 459.7319745639749, 226.57819639238375, 461.6520755140017, 372.9358644421168, 364.74597819433865, 459.28465186171537.187478874107065, 51.29697403566756, 29.611657633306532, 40.22994713918414, 37.03404377573669, 38.926916172908356, 20.223952480349777, 45.88119977337147, 39.54458364105578, 27.943120476458812, 44.5314853106584474.8758545561396, 603.0440634749075, 465.25128668453186, 521.5202184166586, 511.8225236009397, 703.2194902926152, 284.5436506357287, 727.654996961246, 679.8261094451382, 465.3616100766099, 692.416584132186624.33453760656167, 36.80882639484867, 23.654213529645013, 33.25167890478412, 27.99943302365993, 28.534924149667567, 17.182665058636868, 33.410647380642, 23.691223899644175, 24.027916406490107, 33.35168543148309.991529029935, 440.3610766299707, 371.50021210518787, 438.67177121550503, 360.7373123980669, 537.0600590202483, 228.3011616635383, 511.1285945042745, 393.60518736446653, 369.94496557980887, 498.231456132165824.760928115333837, 35.24441701512808, 23.225080793837442, 31.940425485767147, 26.668251321824457, 27.72913338454499, 17.80255352331667, 31.84219112389092, 23.168460665590572, 24.16823487415193, 31.35168513521321.0011820417995, 429.2950110374058, 355.8829851548642, 428.6549243103872, 344.1007204531485, 516.0211979989399, 231.9298654311742, 488.1333757993711, 386.93254413985466, 56.4234063065124, 480.216531284853124.906047958128255, 34.37296631068462, 22.44729268579549, 29.891522809658944, 26.292429900497304, 27.491176832711936, 17.593061961950593, 31.26238064593363, 23.036739258716718, 24.06639919717573, 31.212168451648322.29795011231795, 417.01187642496905, 342.1973635129611, 404.89523719280476, 337.6208817657014, 527.0434448699397, 224.12384188987247, 479.4850438938416, 387.114773202139, 349.4076804054119, 479.5313854121654


# In[ ]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,max_torque_pap_MVIC0[0]], [2,max_torque_pap_MVIC2s[0]], [3,max_torque_pap_MVIC4m[0]],[4,max_torque_pap_MVIC8m[0]],[5,max_torque_pap_MVIC12m[0]], "green", "S1")
newline([1,max_torque_pap_MVIC0[1]], [2,max_torque_pap_MVIC2s[1]], [3,max_torque_pap_MVIC4m[1]],[4,max_torque_pap_MVIC8m[1]],[5,max_torque_pap_MVIC12m[1]], "lime", "S2")
newline([1,max_torque_pap_MVIC0[2]], [2,max_torque_pap_MVIC2s[2]], [3,max_torque_pap_MVIC4m[2]],[4,max_torque_pap_MVIC8m[2]],[5,max_torque_pap_MVIC12m[2]], "turquoise", "S3")
newline([1,max_torque_pap_MVIC0[3]], [2,max_torque_pap_MVIC2s[3]], [3,max_torque_pap_MVIC4m[3]],[4,max_torque_pap_MVIC8m[3]],[5,max_torque_pap_MVIC12m[3]], "blue", "S4")
newline([1,max_torque_pap_MVIC0[4]], [2,max_torque_pap_MVIC2s[4]], [3,max_torque_pap_MVIC4m[4]],[4,max_torque_pap_MVIC8m[4]],[5,max_torque_pap_MVIC12m[4]], "darkviolet", "S5")
newline([1,max_torque_pap_MVIC0[5]], [2,max_torque_pap_MVIC2s[5]], [3,max_torque_pap_MVIC4m[5]],[4,max_torque_pap_MVIC8m[5]],[5,max_torque_pap_MVIC12m[5]], "purple", "S6")
newline([1,max_torque_pap_MVIC0[6]], [2,max_torque_pap_MVIC2s[6]], [3,max_torque_pap_MVIC4m[6]],[4,max_torque_pap_MVIC8m[6]],[5,max_torque_pap_MVIC12m[6]], "darkorange", "S7")
newline([1,max_torque_pap_MVIC0[7]], [2,max_torque_pap_MVIC2s[7]], [3,max_torque_pap_MVIC4m[7]],[4,max_torque_pap_MVIC8m[7]],[5,max_torque_pap_MVIC12m[7]], "gold", "S8")
newline([1,max_torque_pap_MVIC0[8]], [2,max_torque_pap_MVIC2s[8]], [3,max_torque_pap_MVIC4m[8]],[4,max_torque_pap_MVIC8m[8]],[5,max_torque_pap_MVIC12m[8]], "red", "S9")
newline([1,max_torque_pap_MVIC0[9]], [2,max_torque_pap_MVIC2s[9]], [3,max_torque_pap_MVIC4m[9]],[4,max_torque_pap_MVIC8m[9]],[5,max_torque_pap_MVIC12m[9]], "black", "S10")
newline([1,max_torque_pap_MVIC0[10]], [2,max_torque_pap_MVIC2s[10]], [3,max_torque_pap_MVIC4m[10]],[4,max_torque_pap_MVIC8m[10]],[5,max_torque_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Torque - PAP-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,60), ylabel='Torque in Nm')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(10, 70, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[ ]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=800, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=800, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=800, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=800, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=800, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rfd20_70_pap_MVIC0[0]], [2,rfd20_70_pap_MVIC2s[0]], [3,rfd20_70_pap_MVIC4m[0]],[4,rfd20_70_pap_MVIC8m[0]],[5,rfd20_70_pap_MVIC12m[0]], "green", "S1")
newline([1,rfd20_70_pap_MVIC0[1]], [2,rfd20_70_pap_MVIC2s[1]], [3,rfd20_70_pap_MVIC4m[1]],[4,rfd20_70_pap_MVIC8m[1]],[5,rfd20_70_pap_MVIC12m[1]], "lime", "S2")
newline([1,rfd20_70_pap_MVIC0[2]], [2,rfd20_70_pap_MVIC2s[2]], [3,rfd20_70_pap_MVIC4m[2]],[4,rfd20_70_pap_MVIC8m[2]],[5,rfd20_70_pap_MVIC12m[2]], "turquoise", "S3")
newline([1,rfd20_70_pap_MVIC0[3]], [2,rfd20_70_pap_MVIC2s[3]], [3,rfd20_70_pap_MVIC4m[3]],[4,rfd20_70_pap_MVIC8m[3]],[5,rfd20_70_pap_MVIC12m[3]], "blue", "S4")
newline([1,rfd20_70_pap_MVIC0[4]], [2,rfd20_70_pap_MVIC2s[4]], [3,rfd20_70_pap_MVIC4m[4]],[4,rfd20_70_pap_MVIC8m[4]],[5,rfd20_70_pap_MVIC12m[4]], "darkviolet", "S5")
newline([1,rfd20_70_pap_MVIC0[5]], [2,rfd20_70_pap_MVIC2s[5]], [3,rfd20_70_pap_MVIC4m[5]],[4,rfd20_70_pap_MVIC8m[5]],[5,rfd20_70_pap_MVIC12m[5]], "purple", "S6")
newline([1,rfd20_70_pap_MVIC0[6]], [2,rfd20_70_pap_MVIC2s[6]], [3,rfd20_70_pap_MVIC4m[6]],[4,rfd20_70_pap_MVIC8m[6]],[5,rfd20_70_pap_MVIC12m[6]], "darkorange", "S7")
newline([1,rfd20_70_pap_MVIC0[7]], [2,rfd20_70_pap_MVIC2s[7]], [3,rfd20_70_pap_MVIC4m[7]],[4,rfd20_70_pap_MVIC8m[7]],[5,rfd20_70_pap_MVIC12m[7]], "gold", "S8")
newline([1,rfd20_70_pap_MVIC0[8]], [2,rfd20_70_pap_MVIC2s[8]], [3,rfd20_70_pap_MVIC4m[8]],[4,rfd20_70_pap_MVIC8m[8]],[5,rfd20_70_pap_MVIC12m[8]], "red", "S9")
newline([1,rfd20_70_pap_MVIC0[9]], [2,rfd20_70_pap_MVIC2s[9]], [3,rfd20_70_pap_MVIC4m[9]],[4,rfd20_70_pap_MVIC8m[9]],[5,rfd20_70_pap_MVIC12m[9]], "black", "S10")
newline([1,rfd20_70_pap_MVIC0[10]], [2,rfd20_70_pap_MVIC2s[10]], [3,rfd20_70_pap_MVIC4m[10]],[4,rfd20_70_pap_MVIC8m[10]],[5,rfd20_70_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Rate of Force Developement 20%-70% - PAP-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,800), ylabel='RFD 20%-70% in Nm/s')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(200, 800, 50), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# ## PAP Statistik

# In[151]:


#anova Torque PAP

from statsmodels.stats.anova import AnovaRM



df_torque = pd.DataFrame({'Subject': np.tile([1, 2, 3, 4, 5 ,6 ,7 ,8 ,9 ,10, 11], 5),
                   'Time': np.repeat([1, 2, 3, 4, 5],11),
                   'Torque':  [ 24.50007750087747, 33.46859090722437, 21.60066413873087, 27.873981806518543, 28.080832140003892, 24.07898348399918, 17.688966862273613, 31.5858175389304, 22.53570736149189, 25.474274516034313, 30.4742746149189,
                                37.187478874107065, 51.29697403566756, 29.611657633306532, 40.22994713918414, 37.03404377573669, 38.926916172908356, 20.223952480349777, 45.88119977337147, 39.54458364105578, 27.943120476458812, 44.5314853106584,
                                24.33453760656167, 36.80882639484867, 23.654213529645013, 33.25167890478412, 27.99943302365993, 28.534924149667567, 17.182665058636868, 33.410647380642, 23.691223899644175, 24.027916406490107, 33.35168543148,
                                24.760928115333837, 35.24441701512808, 23.225080793837442, 31.940425485767147, 26.668251321824457, 27.72913338454499, 17.80255352331667, 31.84219112389092, 23.168460665590572, 24.16823487415193, 31.35168513521,
                                24.906047958128255, 34.37296631068462, 22.44729268579549, 29.891522809658944, 26.292429900497304, 27.491176832711936, 17.593061961950593, 31.26238064593363, 23.036739258716718, 24.06639919717573, 31.212168451648]})

print(AnovaRM(data=df_torque, depvar='Torque', subject='Subject', within=['Time']).fit())


# In[155]:


#tukey test als post-hoc TORQUE
#Dataframe erstellen zum Speichern von Daten
df = pd.DataFrame({'score': [   24.50007750087747, 33.46859090722437, 21.60066413873087, 27.873981806518543, 28.080832140003892, 24.07898348399918, 17.688966862273613, 31.5858175389304, 22.53570736149189, 25.474274516034313, 30.4742746149189,
                                37.187478874107065, 51.29697403566756, 29.611657633306532, 40.22994713918414, 37.03404377573669, 38.926916172908356, 20.223952480349777, 45.88119977337147, 39.54458364105578, 27.943120476458812, 44.5314853106584,
                                24.33453760656167, 36.80882639484867, 23.654213529645013, 33.25167890478412, 27.99943302365993, 28.534924149667567, 17.182665058636868, 33.410647380642, 23.691223899644175, 24.027916406490107, 33.35168543148,
                                24.760928115333837, 35.24441701512808, 23.225080793837442, 31.940425485767147, 26.668251321824457, 27.72913338454499, 17.80255352331667, 31.84219112389092, 23.168460665590572, 24.16823487415193, 31.35168513521,
                                24.906047958128255, 34.37296631068462, 22.44729268579549, 29.891522809658944, 26.292429900497304, 27.491176832711936, 17.593061961950593, 31.26238064593363, 23.036739258716718, 24.06639919717573, 31.212168451648],
                   'group': np.repeat(['PRE', '2s', '4m', '8m', '12m'], repeats=11)}) 







tukey = pairwise_tukeyhsd(endog = df['score'], groups = df['group'], alpha = 0.05)
tukey.plot_simultaneous()
#plt.vlines(x=33.8, ymin=-0.5, ymax = 4.5, color ="red")
tukey.summary()


# In[152]:


#anova RFD PAP


from statsmodels.stats.anova import AnovaRM



df_rfd = pd.DataFrame({'Subject': np.tile([1, 2, 3, 4, 5 ,6 ,7 ,8 ,9 ,10, 11], 5),
                   'Time': np.repeat([1, 2, 3, 4, 5],11),
                   'RFD':  [ 321.225184809161, 405.6681636034446, 319.33336212953964, 378.2824502702033, 373.34316070133906, 459.7319745639749, 226.57819639238375, 461.6520755140017, 372.9358644421168, 364.74597819433865, 459.284651861715,
                                474.8758545561396, 603.0440634749075, 465.25128668453186, 521.5202184166586, 511.8225236009397, 703.2194902926152, 284.5436506357287, 727.654996961246, 679.8261094451382, 465.3616100766099, 692.4165841321866,
                                309.991529029935, 440.3610766299707, 371.50021210518787, 438.67177121550503, 360.7373123980669, 537.0600590202483, 228.3011616635383, 511.1285945042745, 393.60518736446653, 369.94496557980887, 498.2314561321658,
                                321.0011820417995, 429.2950110374058, 355.8829851548642, 428.6549243103872, 344.1007204531485, 516.0211979989399, 231.9298654311742, 488.1333757993711, 386.93254413985466, 356.4234063065124, 480.2165312848531,
                                322.29795011231795, 417.01187642496905, 342.1973635129611, 404.89523719280476, 337.6208817657014, 527.0434448699397, 224.12384188987247, 479.4850438938416, 387.114773202139, 349.4076804054119, 479.5313854121654]})

print(AnovaRM(data=df_rfd, depvar='RFD', subject='Subject', within=['Time']).fit())


# In[156]:


#tukey test als post-hoc RFD
#Dataframe erstellen zum Speichern von Daten
df = pd.DataFrame({'score': [   321.225184809161, 405.6681636034446, 319.33336212953964, 378.2824502702033, 373.34316070133906, 459.7319745639749, 226.57819639238375, 461.6520755140017, 372.9358644421168, 364.74597819433865, 459.284651861715,
                                474.8758545561396, 603.0440634749075, 465.25128668453186, 521.5202184166586, 511.8225236009397, 703.2194902926152, 284.5436506357287, 727.654996961246, 679.8261094451382, 465.3616100766099, 692.4165841321866,
                                309.991529029935, 440.3610766299707, 371.50021210518787, 438.67177121550503, 360.7373123980669, 537.0600590202483, 228.3011616635383, 511.1285945042745, 393.60518736446653, 369.94496557980887, 498.2314561321658,
                                321.0011820417995, 429.2950110374058, 355.8829851548642, 428.6549243103872, 344.1007204531485, 516.0211979989399, 231.9298654311742, 488.1333757993711, 386.93254413985466, 356.4234063065124, 480.2165312848531,
                                322.29795011231795, 417.01187642496905, 342.1973635129611, 404.89523719280476, 337.6208817657014, 527.0434448699397, 224.12384188987247, 479.4850438938416, 387.114773202139, 349.4076804054119, 479.5313854121654],
                   'group': np.repeat(['PRE', '2s', '4m', '8m', '12m'], repeats=11)}) 












tukey = pairwise_tukeyhsd(endog = df['score'], groups = df['group'], alpha = 0.05)
tukey.plot_simultaneous()
#plt.vlines(x=499, ymin=-0.5, ymax = 4.5, color ="red")
tukey.summary()


# In[103]:


# Average of Torque Values Time-grouped
mean_max_torque_pap_MVIC0 = np.mean(max_torque_pap_MVIC0)
mean_max_torque_pap_MVIC2s = np.mean(max_torque_pap_MVIC2s)
mean_max_torque_pap_MVIC4m= np.mean(max_torque_pap_MVIC4m)
mean_max_torque_pap_MVIC8m = np.mean(max_torque_pap_MVIC8m)
mean_max_torque_pap_MVIC12m = np.mean(max_torque_pap_MVIC12m)

means_of_pap_torque = np.array([mean_max_torque_pap_MVIC0, mean_max_torque_pap_MVIC2s, mean_max_torque_pap_MVIC4m, mean_max_torque_pap_MVIC8m, mean_max_torque_pap_MVIC12m])


# In[104]:


# Average of RFD Values Time-grouped
mean_rfd20_70_pap_MVIC0 = np.mean(rfd20_70_pap_MVIC0)
mean_rfd20_70_pap_MVIC2s = np.mean(rfd20_70_pap_MVIC2s)
mean_rfd20_70_pap_MVIC4m= np.mean(rfd20_70_pap_MVIC4m)
mean_rfd20_70_pap_MVIC8m = np.mean(rfd20_70_pap_MVIC8m)
mean_rfd20_70_pap_MVIC12m = np.mean(rfd20_70_pap_MVIC12m)

means_of_pap_rfd = np.array([mean_rfd20_70_pap_MVIC0, mean_rfd20_70_pap_MVIC2s, mean_rfd20_70_pap_MVIC4m, mean_rfd20_70_pap_MVIC8m, mean_rfd20_70_pap_MVIC12m])


# In[105]:


#TORQUE BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
means_of_pap_t = means_of_pap_torque.astype(int)
x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, means_of_pap_t, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Torque in Nm')
ax.set_title('PAP - Trial')
ax.set_ylim([0, 40])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[106]:


#RFD BARS
bars = "PRE", "2sec", "4min", "8min", "12min"
means_of_rfd2070 = means_of_pap_rfd.astype(int)
x = np.arange(len(bars)) # the label locations
width = 0.5  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x, means_of_rfd2070, width)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('RFD 20%-70% in Nm/s')
ax.set_title('PAP - Trial')
ax.set_ylim([0, 650])
ax.set_xticks(x)
ax.set_xticklabels(bars)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
autolabel(rects1)

fig.tight_layout()

plt.show()


# In[163]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=300, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,max_torque_pap_MVIC0[0]], [2,max_torque_pap_MVIC2s[0]], [3,max_torque_pap_MVIC4m[0]],[4,max_torque_pap_MVIC8m[0]],[5,max_torque_pap_MVIC12m[0]], "green", "S1")
newline([1,max_torque_pap_MVIC0[1]], [2,max_torque_pap_MVIC2s[1]], [3,max_torque_pap_MVIC4m[1]],[4,max_torque_pap_MVIC8m[1]],[5,max_torque_pap_MVIC12m[1]], "lime", "S2")
newline([1,max_torque_pap_MVIC0[2]], [2,max_torque_pap_MVIC2s[2]], [3,max_torque_pap_MVIC4m[2]],[4,max_torque_pap_MVIC8m[2]],[5,max_torque_pap_MVIC12m[2]], "turquoise", "S3")
newline([1,max_torque_pap_MVIC0[3]], [2,max_torque_pap_MVIC2s[3]], [3,max_torque_pap_MVIC4m[3]],[4,max_torque_pap_MVIC8m[3]],[5,max_torque_pap_MVIC12m[3]], "blue", "S4")
newline([1,max_torque_pap_MVIC0[4]], [2,max_torque_pap_MVIC2s[4]], [3,max_torque_pap_MVIC4m[4]],[4,max_torque_pap_MVIC8m[4]],[5,max_torque_pap_MVIC12m[4]], "darkviolet", "S5")
newline([1,max_torque_pap_MVIC0[5]], [2,max_torque_pap_MVIC2s[5]], [3,max_torque_pap_MVIC4m[5]],[4,max_torque_pap_MVIC8m[5]],[5,max_torque_pap_MVIC12m[5]], "purple", "S6")
newline([1,max_torque_pap_MVIC0[6]], [2,max_torque_pap_MVIC2s[6]], [3,max_torque_pap_MVIC4m[6]],[4,max_torque_pap_MVIC8m[6]],[5,max_torque_pap_MVIC12m[6]], "darkorange", "S7")
newline([1,max_torque_pap_MVIC0[7]], [2,max_torque_pap_MVIC2s[7]], [3,max_torque_pap_MVIC4m[7]],[4,max_torque_pap_MVIC8m[7]],[5,max_torque_pap_MVIC12m[7]], "gold", "S8")
newline([1,max_torque_pap_MVIC0[8]], [2,max_torque_pap_MVIC2s[8]], [3,max_torque_pap_MVIC4m[8]],[4,max_torque_pap_MVIC8m[8]],[5,max_torque_pap_MVIC12m[8]], "red", "S9")
newline([1,max_torque_pap_MVIC0[9]], [2,max_torque_pap_MVIC2s[9]], [3,max_torque_pap_MVIC4m[9]],[4,max_torque_pap_MVIC8m[9]],[5,max_torque_pap_MVIC12m[9]], "black", "S10")
newline([1,max_torque_pap_MVIC0[10]], [2,max_torque_pap_MVIC2s[10]], [3,max_torque_pap_MVIC4m[10]],[4,max_torque_pap_MVIC8m[10]],[5,max_torque_pap_MVIC12m[10]], "pink", "S11")
#newline([1,max_torque_papev_MVIC0[10]], [2,max_torque_papev_MVIC2s[10]], [3,max_torque_papev_MVIC4m[10]],[4,max_torque_papev_MVIC8m[10]],[5,max_torque_papev_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Torque - PAP-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,60), xlabel="Stimulation time", ylabel='Torque in Nm')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(0, 60, 10), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[165]:


import matplotlib.lines as mlines


# draw line
def newline(p1, p2, p3, p4, p5, color1, label1):
    ax = plt.gca()
    l = mlines.Line2D([p1[0],p2[0], p3[0], p4[0], p5[0]], [p1[1],p2[1], p3[1], p4[1], p5[1]], color= color1, label= label1 , marker='o', markersize=6)
    ax.add_line(l)
    return l

fig, ax = plt.subplots(1,1,figsize=(14,14), dpi= 80)

# Vertical Lines
ax.vlines(x=1, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=2, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=3, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=4, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
ax.vlines(x=5, ymin=0, ymax=1000, color='black', alpha=0.7, linewidth=1, linestyles='dotted')



# Line Segmentsand Annotation

newline([1,rfd20_70_pap_MVIC0[0]], [2,rfd20_70_pap_MVIC2s[0]], [3,rfd20_70_pap_MVIC4m[0]],[4,rfd20_70_pap_MVIC8m[0]],[5,rfd20_70_pap_MVIC12m[0]], "green", "S1")
newline([1,rfd20_70_pap_MVIC0[1]], [2,rfd20_70_pap_MVIC2s[1]], [3,rfd20_70_pap_MVIC4m[1]],[4,rfd20_70_pap_MVIC8m[1]],[5,rfd20_70_pap_MVIC12m[1]], "lime", "S2")
newline([1,rfd20_70_pap_MVIC0[2]], [2,rfd20_70_pap_MVIC2s[2]], [3,rfd20_70_pap_MVIC4m[2]],[4,rfd20_70_pap_MVIC8m[2]],[5,rfd20_70_pap_MVIC12m[2]], "turquoise", "S3")
newline([1,rfd20_70_pap_MVIC0[3]], [2,rfd20_70_pap_MVIC2s[3]], [3,rfd20_70_pap_MVIC4m[3]],[4,rfd20_70_pap_MVIC8m[3]],[5,rfd20_70_pap_MVIC12m[3]], "blue", "S4")
newline([1,rfd20_70_pap_MVIC0[4]], [2,rfd20_70_pap_MVIC2s[4]], [3,rfd20_70_pap_MVIC4m[4]],[4,rfd20_70_pap_MVIC8m[4]],[5,rfd20_70_pap_MVIC12m[4]], "darkviolet", "S5")
newline([1,rfd20_70_pap_MVIC0[5]], [2,rfd20_70_pap_MVIC2s[5]], [3,rfd20_70_pap_MVIC4m[5]],[4,rfd20_70_pap_MVIC8m[5]],[5,rfd20_70_pap_MVIC12m[5]], "purple", "S6")
newline([1,rfd20_70_pap_MVIC0[6]], [2,rfd20_70_pap_MVIC2s[6]], [3,rfd20_70_pap_MVIC4m[6]],[4,rfd20_70_pap_MVIC8m[6]],[5,rfd20_70_pap_MVIC12m[6]], "darkorange", "S7")
newline([1,rfd20_70_pap_MVIC0[7]], [2,rfd20_70_pap_MVIC2s[7]], [3,rfd20_70_pap_MVIC4m[7]],[4,rfd20_70_pap_MVIC8m[7]],[5,rfd20_70_pap_MVIC12m[7]], "gold", "S8")
newline([1,rfd20_70_pap_MVIC0[8]], [2,rfd20_70_pap_MVIC2s[8]], [3,rfd20_70_pap_MVIC4m[8]],[4,rfd20_70_pap_MVIC8m[8]],[5,rfd20_70_pap_MVIC12m[8]], "red", "S9")
newline([1,rfd20_70_pap_MVIC0[9]], [2,rfd20_70_pap_MVIC2s[9]], [3,rfd20_70_pap_MVIC4m[9]],[4,rfd20_70_pap_MVIC8m[9]],[5,rfd20_70_pap_MVIC12m[9]], "black", "S10")
newline([1,rfd20_70_pap_MVIC0[10]], [2,rfd20_70_pap_MVIC2s[10]], [3,rfd20_70_pap_MVIC4m[10]],[4,rfd20_70_pap_MVIC8m[10]],[5,rfd20_70_pap_MVIC12m[10]], "pink", "S11")
#newline([1,rfd20_70_papee_MVIC0[10]], [2,rfd20_70_papee_MVIC2s[10]], [3,rfd20_70_papee_MVIC4m[10]],[4,rfd20_70_papee_MVIC8m[10]],[5,rfd20_70ee_pap_MVIC12m[10]], "peru", "S11")





# Decoration
ax.set_title("Rate of Force Developement 20%-70% - PAP-Trial", fontdict={'size':22})
ax.set(xlim=(0,5.5), ylim=(0,1000),xlabel="Stimulation time", ylabel='RFD 20%-70% in Nm/s')
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Pre", "2s", "4m", "8m", "12m"])
plt.yticks(np.arange(100, 1000, 50), fontsize=12)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.0)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.0)
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[144]:





# In[ ]:





# In[ ]:




