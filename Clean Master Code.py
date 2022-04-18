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

# In[4]:


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

# In[7]:


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
data_rf, data_vm, data_vl, y_rf, y_vm, y_vl, sample, mon, moff, x_ax_rf, x_ax_vm, x_ax_vl, N_rf, N_vm, N_vl, trans_rf, trans_vm,trans_vl, y_ax_rf, y_ax_vm, y_ax_vl, fft_mean_rf, fft_mean_vm, fft_mean_vl, median_fft_rf, median_fft_vm, median_fft_vl, rms_signal_rf,rms_signal_vm, rms_signal_vl, max_torque, rms_torque, rms_sample,rfd_70_i, rfd_20_i,rfd20_70, sheet1, file, rfd_20_i, rfd_70_i, peak_rf, peak_vm, peak_vl, rms_1s_rf, rms_1s_vm, rms_1s_vl= calculation(500, 10, 2000, 2, d_pap, "xlfile5", "pre_stim")

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
print("")
print("")
print("")
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

