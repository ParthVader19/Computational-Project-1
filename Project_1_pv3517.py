# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 07:30:35 2019

@author: p_vag
"""

import numpy as np
import matplotlib.pyplot as plt
import time
#from matplotlib import cm
#from mpl_toolkits import mplot3d

#values given in the project
theta=np.pi/4
D_m=2.4e-3
L=295

num=200

E=np.linspace(0,10,num=200,endpoint=True)+0.025#energy in the region of interest.

D_m_test=np.linspace(0,4e-3,num=num,endpoint=False)+1e-30#a range for the oscillation parameters 
theta_test=np.linspace(0,2*np.pi/4,num=num,endpoint=False)+1e-30
sig_rate_test=np.linspace(0,2,num=num,endpoint=False)+1e-30
#the ranges have a small offset so the expected rate is never 0. This is to avoid np.log(infinity) in the NLL calculation

#observed number of muon neutrino events from 0–10 GeV in energy
data=[0	,0	,1	,0	,8	,13	,11	,10	,5	,5	,0	,0	,8	,10	,9	,20	,12	,14	,15	,8	,5	,6	,3	,7	,6	,3	,3	,5	,1	,6	,2	,4	,3	,3	,2	,0	,2	,3	,6	,6	,4	,5	,5	,2	,0	,3	,1	,5	,2	,4	,1	,2	,0	,7	,3	,2	,1	,0	,5	,2	,4	,1	,1	,0	,3	,0	,3	,6	,3	,9	,2	,3	,4	,4	,3	,4	,2	,8	,4	,9	,10	,5	,2	,3	,5	,4	,4	,4	,3	,1	,3	,3	,6	,6	,4	,5	,5	,7	,4	,3	,4	,3	,8	,2	,2	,3	,6	,2	,1	,1	,6	,4	,2	,1	,5	,2	,1	,1	,2	,2	,3	,1	,3	,0	,0	,3	,2	,1	,4	,2	,1	,0	,3	,2	,0	,2	,3	,1	,0	,0	,0	,2	,3	,1	,1	,2	,0	,1	,2	,0	,2	,0	,0	,1	,2	,1	,0	,2	,0	,0	,1	,1	,1	,0	,0	,0	,1	,0	,0	,1	,1	,0	,0	,0	,0	,0	,0	,0	,0	,1	,0	,0	,1	,0	,0	,0	,0	,0	,0	,0	,0	,1	,0	,0	,0	,0	,0	,0	,0	,1]
#simulated event rate prediction assuming the muon neutrinos do not oscillate with energy from 0–10 GeV
unOssFlux=[1.21438190	,5.62230639	,11.77215522	,18.36411225	,27.21794480	,38.10249543	,51.19882744	,64.91880229	,80.03030054	,97.50418953	,120.84615030	,145.78791260	,152.75765800	,131.59242800	,100.08952310	,73.63209608	,54.60231744	,39.48028161	,26.47861379	,17.63748553	,12.19992341	,9.28609970	,7.64249628	,6.21910485	,5.24044992	,4.65509306	,4.07045634	,3.51118618	,3.25569148	,2.80925349	,2.61069663	,2.39112544	,2.17705579	,1.99959170	,1.92357213	,1.75687067	,1.64994092	,1.56213126	,1.52333637	,1.41260526	,1.38441627	,1.27925977	,1.20698624	,1.16664927	,1.15796934	,1.12386360	,1.07388241	,1.00892838	,1.00460651	,0.98270005	,1.00503534	,0.93658968	,0.93019993	,0.94318783	,0.93619016	,0.89221666	,0.91207888	,0.91133428	,0.89153404	,0.91460231	,0.90674946	,0.89750422	,0.90380133	,0.89190251	,0.87688821	,0.87117490	,0.88346685	,0.87092222	,0.87619063	,0.89471357	,0.85915394	,0.89143436	,0.88726983	,0.89438375	,0.90156309	,0.89889511	,0.90664908	,0.91678198	,0.91735551	,0.92122476	,0.91037453	,0.91495478	,0.92157440	,0.91190644	,0.88883750	,0.89049827	,0.89026236	,0.85621603	,0.84290360	,0.82111156	,0.81243927	,0.79481040	,0.78264333	,0.75665926	,0.74517757	,0.70948241	,0.70109569	,0.67637266	,0.66340054	,0.63552800	,0.61294571	,0.59219409	,0.55411851	,0.53539725	,0.51340540	,0.47555007	,0.47816945	,0.45024840	,0.42395841	,0.39611960	,0.38262378	,0.35831866	,0.35320362	,0.34146820	,0.31617893	,0.30419352	,0.28289478	,0.29198448	,0.25936696	,0.25987293	,0.24309008	,0.23736998	,0.22430119	,0.22206898	,0.21028855	,0.20396381	,0.19452954	,0.18692220	,0.18346423	,0.17720325	,0.16601350	,0.17040633	,0.15991563	,0.15169332	,0.14868690	,0.14962555	,0.13757585	,0.13089846	,0.13002453	,0.12403647	,0.12442329	,0.11985344	,0.11272974	,0.10665933	,0.10935548	,0.10548309	,0.10238063	,0.09527973	,0.09327825	,0.09051113	,0.08820124	,0.08702189	,0.08249344	,0.08421637	,0.07895646	,0.07525440	,0.06863699	,0.07162078	,0.06418893	,0.06745232	,0.06564193	,0.06297574	,0.06379705	,0.06113915	,0.05391862	,0.05683304	,0.05303102	,0.05072655	,0.05001585	,0.05327111	,0.04463483	,0.04730818	,0.04339604	,0.04422675	,0.04255934	,0.04203425	,0.04107220	,0.03729259	,0.03287503	,0.03320166	,0.03638846	,0.03233257	,0.03283617	,0.02868774	,0.02943291	,0.02913883	,0.02856542	,0.02825330	,0.02511534	,0.02472052	,0.02411842	,0.02670751	,0.02173993	,0.02349169	,0.02007420	,0.02392712	,0.02255197	,0.01960980	,0.01891021	,0.02015019
]

# Functions

def surv_prob_corr(theta,m,sig_rate=1/E,E=E,L=L):#survival probability. It also include the cross-section rate to reduce the number of functions needed. 
    return (1-(np.sin(2*theta)**2)*(np.sin(1.267*(m)*L/E))**2)*sig_rate*E

# NLL as a function of theta, m, and sigma (the cross-section rate) all in one.
def NLL(theta,m,sigma=1/E,O=data,unOssFlux=unOssFlux,surv_prob_corr=surv_prob_corr,variable="theta",single=False): #finds the NLL for a given a range of mixing angle.
    
    if single==False:
        NLL_rate=[]#loop used to get an array of expected oscillating rate 
        if variable=="theta":
            for i in range(len(theta)):
                NLL_rate.append(unOssFlux*surv_prob_corr(theta[i],m,sigma))
        elif variable=="m":
            for i in range(len(m)):
                NLL_rate.append(unOssFlux*surv_prob_corr(theta,m[i],sigma))
        else:
            for i in range(len(sigma)):
                NLL_rate.append(unOssFlux*surv_prob_corr(theta,m,sigma[i]))
    else:
        NLL_rate=unOssFlux*surv_prob_corr(theta,m,sigma)
        
    if single==False:
        NLL_y=[]#use the expected oscillating rate and the actual data to perform NLL using the given formula
        for j in range(len(NLL_rate)):
            sum0=0
            for i in range(len(NLL_rate[j])):
                if O[i]==0:#computer cannot handle 0*np.log(0) hence condition used to avoid 'nan' answer
                    sum0+=NLL_rate[j][i]-O[i]
                else:
                    sum0+=NLL_rate[j][i]-O[i]+O[i]*np.log(O[i]/NLL_rate[j][i])
            NLL_y.append(sum0)
    else:
        #use the expected oscillating rate and the actual data to perform NLL using the given formula
        NLL_y=0
        for i in range(len(NLL_rate)):
            if O[i]==0:#computer cannot handle 0*np.log(0) hence condition used to avoid 'nan' answer
                NLL_y+=NLL_rate[i]-O[i]
            else:
                NLL_y+=NLL_rate[i]-O[i]+O[i]*np.log(O[i]/NLL_rate[i])
            
    return NLL_y

def parabolic_min(x,y,plot=False,i1=40,i2=51,i3=60,choose_start=False):# 1 parameter minimasation using the parabolic method
    if choose_start==True:
        x_test=[x[i1],x[i2],x[i3]]
        y_test=[y[i1],y[i2],y[i3]]
    elif choose_start==False:
        x_test=[x[np.argmin(y)-10],x[np.argmin(y)],x[np.argmin(y)+10]]
        y_test=[y[np.argmin(y)-10],y[np.argmin(y)],y[np.argmin(y)+10]]

    current_min=1e10#used to compare to the minimum for the while loop. 
    
    colour=["black","blue","green","purple","orange","grey"]#colour and name to show the iteration in the parabolic estimation
    name=[1,2,3,4,5,6,7]
    c_index=0


    while abs(min(y_test)-current_min)>1e-10:#loop stops when the change in the minimum value of the function is very small
        current_min=min(y_test)#setting the new minimum value 
        x_test1=x_test#setting the new points for the parabolic plotting if it is enabled.
        
        if plot!=False: #plotting the parabolic iterations if it is enabled
            plt.plot(x_test1,y_test,'o', color=colour[c_index],label="Iteration "+str(name[c_index]))
            z = np.polyfit(x_test, y_test, 2)
            poly_y=np.poly1d(z)
            plt.plot(x,poly_y(x),color=colour[c_index])
        c_index+=1        
        
        #finds the minimum of the parabola constructed using the 3 points.
        x3=0.5*((x_test[2]**2 -x_test[1]**2)*y_test[0] +(x_test[0]**2 -x_test[2]**2)*y_test[1] +(x_test[1]**2 -x_test[0]**2)*y_test[2])/((x_test[2]-x_test[1])*y_test[0] + (x_test[0]-x_test[2])*y_test[1]+ (x_test[1]-x_test[0])*y_test[2])
        
        #finds the corresponding value of y for x3
        y3=(x3-x_test[1])*(x3-x_test[2])*y_test[0]/((x_test[0]-x_test[1])*(x_test[0]-x_test[2]))+ (x3-x_test[0])*(x3-x_test[2])*y_test[1]/((x_test[1]-x_test[0])*(x_test[1]-x_test[2]))+ (x3-x_test[0])*(x3-x_test[1])*y_test[2]/((x_test[2]-x_test[0])*(x_test[2]-x_test[1]))
        y_test.append(y3)
        x_test.append(x3)
        
        #removes the maximum value of y, and the coresponding value of x, leaving the 3 points for the next iteration 
        x_test.remove(x_test[y_test.index(max(y_test))])
        y_test.remove(max(y_test))
    #finding the curvature of the parabola to estimate the error in the parabolic fit.
    C=2*(y_test[0]/(x_test[0]-x_test[1])*(x_test[0]-x_test[2]) + y_test[1]/(x_test[1]-x_test[0])*(x_test[1]-x_test[2]) +y_test[2]/(x_test[2]-x_test[0])*(x_test[2]-x_test[1]))
    para_err=[x_test[y_test.index(min(y_test))]-1/np.sqrt(C), x_test[y_test.index(min(y_test))]+1/np.sqrt(C)]
    #Occassionally, the last iteration will have values being very close together, hence when the difference is taken, the computer will round to zero. 
    #this causes error warning but doesnt effect the code.

    return x_test[y_test.index(min(y_test))],x_test1,para_err

#the following function is only used for the heat map and contour plot.
def NLL_ScalPlot(theta_1,D_m_1,E=E,L=L,O=data,unOssFlux=unOssFlux):#This functions determines the NLL for a range of mixing angle and mass squared as an array correctly shaped for a contour and heat maps
    prob_b=[]
    for i in range(len(theta_1)):#loop determines the survival probability array in the correct shape
        prob=[]
        for j in range(len(D_m_1[0])):
           prob.append(1-(np.sin(2*theta_1[i][j])**2)*(np.sin(1.267*(D_m_1[i][j])*L/E))**2) 
        prob_b.append(prob)
    
    rate_b=[]
    for i in range(len(prob_b)):#loop determines the expected rate array in the correct shape
        rate=[]
        for j in range(len(prob_b[0])):
            rate.append(unOssFlux*prob_b[i][j])
        rate_b.append(rate)
        
    NLL_y_b=[]
    for i in range(len(rate_b)):#loop determines the NLL array in the correct shape
        NLL_y=[]#use the expected oscillating rate and the actual data to perform NLL using the formula given
        for j in range(len(rate_b[i])):
            sum0=0
            for k in range(len(rate_b[i][j])):
                if O[k]==0:#computer cannot handle 0*np.log(0) hence condition used to avoid 'nan' answer
                    sum0+=rate_b[i][j][k]-O[k]
                else:
                    sum0+=rate_b[i][j][k]-O[k]+O[k]*np.log(O[k]/rate_b[i][j][k])
            NLL_y.append(sum0)
        NLL_y_b.append(NLL_y)
    return NLL_y_b

def univariate_corr(theta,delta_m,sig_rate,sigmaON,j1=45,j2=60,j3=50,NLL=NLL):
    theta_min=theta[j1]#inital parameters from the beginning part.
    delta_m_min=delta_m[j2]
    
    if sigmaON==True:
        sig_rate_min=sig_rate[j3]
    else:
        sig_rate_min=0#for sigmaON=False, i.e. effects of the cross section is not taken into account

    #print(sig_rate_min)
    old_theta_min=1e10
    old_delta_m_min=1e10#used to compare determine the change in the values of the parameters for the while loop. 
    old_sig_rate_min=1e10
    
    theta_1=[theta_min]#arrays of the parameters used for plotting later
    delta_m_1=[delta_m_min]
    sig_rate_1=[sig_rate_min]
        
    #loop stops when the change in the parameter values giving the minimum value of NLL is very small
    while abs(old_theta_min-theta_min)>1e-7 or abs(old_delta_m_min-delta_m_min)>1e-7 or abs(old_sig_rate_min-sig_rate_min)>1e-7:
        
        old_theta_min=theta_min#setting the current parameter values
        old_delta_m_min=delta_m_min
        old_sig_rate_min=sig_rate_min
        
        
        if sigmaON==True:#Take into account the effect of cross section if enabled
            NLL_y=NLL(theta,delta_m_min,sig_rate_min,variable="theta")#calculating NLL as a function of mixing angle for the current value of mass squared
            
            a,theta_test3,theta_min_curvature=parabolic_min(theta_test,NLL_y,choose_start=False,plot=False)#performing parabolic minimisation to find the new value of mixing angleand cross section rate
            theta_min=a
            theta_1.append(theta_min)#storing the values for plotting later
            delta_m_1.append(delta_m_min)
            sig_rate_1.append(sig_rate_min)
            
            NLL_y_1=NLL(theta_min,delta_m,sig_rate_min,variable="m")#calculating NLL as a function of mass sqaured for the current value of mixing angle and cross section rate
            
            b,delta_m_test3,delta_m_min_curvature=parabolic_min(delta_m,NLL_y_1,choose_start=False,plot=False)#performing parabolic minimisation to find the new value of mixing angle
            delta_m_min=b
            theta_1.append(theta_min)
            delta_m_1.append(delta_m_min)#storing the values for plotting later
            sig_rate_1.append(sig_rate_min)
            
            
            NLL_y_2=NLL(theta_min,delta_m_min,sig_rate,variable="sigma")#calculating NLL as a function of cross section rate for the current value of mass squared and mixing angle
            
            c,sigma_test3,sigma_min_curvature=parabolic_min(sig_rate,NLL_y_2,choose_start=False,plot=False)#performing parabolic minimisation to find the new value of mixing angle
            sig_rate_min=c
            theta_1.append(theta_min)
            delta_m_1.append(delta_m_min)#storing the values for plotting later
            sig_rate_1.append(sig_rate_min)
                
        elif sigmaON==False:
            NLL_y_0=NLL(theta,delta_m_min,variable="theta")#calculating NLL as a function of mixing angle for the current value of mass squared
            
            theta_min,theta_test3,theta_min_curvature=parabolic_min(theta,NLL_y_0,choose_start=False,plot=False)#performing parabolic minimisation to find the new value of mixing angle

            theta_1.append(theta_min)
            delta_m_1.append(delta_m_min)#storing the values for plotting later
           
            
            NLL_y_1=NLL(theta_min,delta_m,variable="m")#calculating NLL as a function of mass squared for the current value of mixing angle
            
            b,delta_m_test3,delta_m_min_curvature=parabolic_min(delta_m,NLL_y_1,choose_start=False,plot=False)#performing parabolic minimisation to find the new value of mixing angle
            delta_m_min=b
            theta_1.append(theta_min)
            delta_m_1.append(delta_m_min)#storing the values for plotting later

        
    return theta_1,delta_m_1,sig_rate_1

def gradMethod_corr(x,y,j,sigmaON,a1=0.00001,a2=0.000000001,a3=0.000001,j1=45,j2=60,j3=50,delta_theta=0.01,delta_m=0.0001,delta_sig_rate=0.0001,NLL=NLL):
    x_min=x[j1]#starting values of the parameter that minimise NLL
    y_min=y[j2]
    if sigmaON==True:
        j_min=j[j3]
    else:
        j_min=0

    x_min_old=0#used to compare determine the change in the values of the parameters for the while loop. 
    y_min_old=0
    j_min_old=0
    
    x_array=[x_min]#arrays of the parameters used to see the change in the parameter. It will not be used for plotting
    y_array=[y_min]
    j_array=[j_min]
    
    while abs(y_min-y_min_old)>1e-7 or abs(x_min-x_min_old)>1e-7 or abs(j_min-j_min_old)>1e-7 :
        
        x_min_old=x_min#setting the current parameter values
        y_min_old=y_min
        j_min_old=j_min
        
        if sigmaON==True:#for minimising with 3 parameters
            z_0=NLL(x_min,y_min,j_min,single=True)#finds the NLL at the current value of the parameters
                
            z_x1=NLL(x_min+delta_theta,y_min,j_min,single=True)#finds the NLL at the current value of y and j, and an updated value of x of the parameters
            diff_x=(z_x1-z_0)/delta_theta#partial differentiation with repect to x 
            
            z_y1=NLL(x_min,y_min+delta_m,j_min,single=True)#finds the NLL at the current value of x and j and an updated value of y of the parameters
            diff_y=(z_y1-z_0)/delta_m#partial differentiation with repect to y
            
            z_y2=NLL(x_min,y_min,j_min+delta_sig_rate,single=True)#finds the NLL at the current value of x and y and an updated value of j of the parameters
            diff_j=(z_y2-z_0)/delta_sig_rate#partial differentiation with repect to j
            
            x_min=x_min_old-a1*diff_x#taking a small step in x, y and j in the negative direction to the gradient change   
            y_min=y_min_old-a2*diff_y
            j_min=j_min_old-a3*diff_j
            
            x_array.append(x_min)
            y_array.append(y_min)
            j_array.append(j_min)
            
        elif sigmaON==False :#for miniminsing with 2 parameters
            z_0=NLL(x_min,y_min,single=True)#finds the NLL at the current value of the parameters
                
            z_x1=NLL(x_min+delta_theta,y_min,single=True)#finds the NLL at the current value of y and an updated value of x of the parameters
            diff_x=(z_x1-z_0)/delta_theta#partial differentiation with repect to x 
            
            z_y1=NLL(x_min,y_min+delta_m,single=True)#finds the NLL at the current value of x and an updated value of y of the parameters
            diff_y=(z_y1-z_0)/delta_m#partial differentiation with repect to y
            
            x_min=x_min_old-a1*diff_x#taking a small step in x and y in the negative direction to the gradient change   
            y_min=y_min_old-a2*diff_y
            
            x_array.append(x_min)#storing the iterated values of the parameter for plotting later 
            y_array.append(y_min)
            
    return x_array,y_array,j_array

def STerr(x,y):# shifting method to find the standard deviation
    y_min=y[np.argmin(y)]
    x_n=[]
    for i in range(len(x)):
        if (y[i]-y_min)<0.5:#checks when the difference in y from the minimum is less than 0.5
            if y[i]-y_min>=0:
                x_n.append(x[i])
    return [x[np.argmin(y)]-x_n[0],-x[np.argmin(y)]+x_n[-1]]

#The following is a function to determine the error for each variable analytically, however it is not talked about in the final report 
def err_analytic(x_min,y_min,j_min=1/E,data=data,E=E,L=L,O=data,unOssFlux=unOssFlux,variable="theta"):
    if variable=="theta":
        d_1=unOssFlux*(-np.sin(1.267*y_min*L/E)**2)*np.sin(4*x_min)*2*j_min*E
        d_2=unOssFlux*(-np.sin(1.267*y_min*L/E)**2)*np.cos(4*x_min)*8*j_min*E
        rate=unOssFlux*surv_prob_corr(x_min,y_min,j_min)
        a=x_min#second order differential w.r.t theta
    elif variable=="m":  
        d_1=unOssFlux*(-np.sin(2*1.267*y_min*L/E)*np.sin(2*x_min)**2)*(1.267*L/E)*j_min*E
        d_2=unOssFlux*(-np.cos(2*1.267*y_min*L/E)*np.sin(2*x_min)**2)*2*((1.267*L/E)**2)*j_min*E
        rate=unOssFlux*surv_prob_corr(x_min,y_min,j_min)
        a=y_min#second order differential w.r.t m
        
    elif variable=="sigma":
        d_1=unOssFlux*(1-(np.sin(2*x_min)**2)*(np.sin(1.267*y_min*L/E)**2))*E
        d_2=0*E
        rate=unOssFlux*surv_prob_corr(x_min,y_min,j_min)
        a=j_min #second order differential w.r.t sigma

    f_2=0
    for j in range(len(rate)):
        f_2+=(1-data[j]/rate[j])*(d_2[j])+(data[j]/rate[j]**2)*(d_1[j])**2
        
    return [a-1/np.sqrt(f_2),a+1/np.sqrt(f_2)]

def para_err(x,y):#parabolic approximation method to find the error about the minimum. Uses the same function as what's found in the
    #parabolic minimisation method
    x_test=[x[np.argmin(y)-3],x[np.argmin(y)],x[np.argmin(y)+3]]
    y_test=[y[np.argmin(y)-3],y[np.argmin(y)],y[np.argmin(y)+3]]

    C=2*(y_test[0]/(x_test[0]-x_test[1])*(x_test[0]-x_test[2]) + y_test[1]/(x_test[1]-x_test[0])*(x_test[1]-x_test[2]) +y_test[2]/(x_test[2]-x_test[0])*(x_test[2]-x_test[1]))
    return [x_test[y_test.index(min(y_test))]-1/np.sqrt(C), x_test[y_test.index(min(y_test))]+1/np.sqrt(C)]

def normal(x):#function to normal the data for plotting
    return (x-np.amin(x))/(np.amax(x)-np.amin(x))

def TEST1D(x,a=0):#testing function for prabolic minimisation method
    return -1000*np.sin(x) - 1000*np.sin(a)+2000

def TEST2D(x,y,z=0,variable="noNeed",single="noNeed"):#testing function for multi-variable minimisation method
    return -1000*np.sin(x)-1000*np.sin(y)+2000

# TESTING

x=np.linspace(0,np.pi,100)#variables for testing
y=np.linspace(0,np.pi,100)

y_1d=TEST1D(x)

X,Y=np.meshgrid(x,y)
Z=TEST2D(X,Y)


plt.figure()
min_val,min_array,min_para_err=parabolic_min(x,y_1d,plot=True,i1=30,i2=40,i3=60,choose_start=True)#parabolic minismisation testing
plt.plot(x,y_1d,color="red",label="f(x)=-sin(x)")
plt.grid()
plt.legend()
plt.ylabel(r'f(x)')
plt.xlabel(r'x')
plt.title("Parabolic minimisation with f(x)=-1000sin(x)+2000")
plt.show()

f_x_min=TEST1D(min_val)
stdev_min_val=STerr(x,y_1d)#finding the error at the minimum

print('-------------')
print('-------------')
print("TESTING FUNCTIONS f(x)")
print('--Single Variable- Parabolic Minimisation Method: --')
print('x_min:', min_val,'+/-',min_val-min_para_err[0])

x_min,y_min,j_min=univariate_corr(x,y,0,sigmaON=False,j1=10,j2=10,j3=10,NLL=TEST2D)#unicariate minimisation of x and y
x_min_1,y_min_1,j_min_1=gradMethod_corr(x,y,0,a1=0.001,a2=0.001,a3=0.001,j1=10,j2=10,j3=10,delta_theta=0.001,delta_m=0.001,delta_sig_rate=0.001,sigmaON=False,NLL=TEST2D) #graident minimisation of x and y

x_para_err_uni=para_err(x,TEST2D(x,y_min[-1]))#Error found using standard deviation for univariate
y_para_err_uni=para_err(y,TEST2D(x_min[-1],y))

x_para_err_grad=para_err(x,TEST2D(x,y_min_1[-1]))#Error found using standard deviation for gradient
y_para_err_grad=para_err(y,TEST2D(x_min_1[-1],y))

print('-------------')
print('-------------')
print("TESTING FUNCTIONS f(x,y)")
print('--Multi-Variable Methods: --')
print('Univariate: (', x_min[-1],'+/-',x_min[-1]-x_para_err_uni[0],',',y_min[-1],'+/-',y_min[-1]-y_para_err_uni[0],')')
print('Gradient: (', x_min_1[-1],'+/-',x_min_1[-1]-x_para_err_grad[0],',',y_min_1[-1],'+/-',y_min_1[-1]-y_para_err_grad[0],')')

fig1=plt.figure()
surf1=plt.pcolormesh(X,Y, Z, cmap='inferno',) #heat map to see how NLL changes for given values of the parameters
fig1.colorbar(surf1, shrink=1, aspect=10)
cs_1=plt.contour(X,Y, Z,levels=np.linspace(0,2000,num=10,endpoint=False))#contour plot
for i in range(len(x_min)):
    if i==0:
        plt.plot(x_min[i:i+2], y_min[i:i+2], '.-',color="green",label="Univariate Method ")
    else:
        plt.plot(x_min[i:i+2], y_min[i:i+2], '.-',color="green")

for i in range(len(x_min_1)):
    if i==0 :
        plt.plot(x_min_1[i:i+2], y_min_1[i:i+2], '.-',color="grey",label="Gradient Method")
    elif i==len(x_min_1)-1:
        plt.plot(x_min_1[i:i+2], y_min_1[i:i+2], '.-',color="grey")
    else:
        plt.plot(x_min_1[i:i+2], y_min_1[i:i+2], '-',color="grey")
plt.title("Multi-variable minimisation methods with f(x,y)=-1000(sin(x)+sin(y))+2000")
plt.ylabel('x')
plt.xlabel(r'y')
plt.legend()
plt.show()

#%%


    
inital_fit=unOssFlux*surv_prob_corr(theta_test[100],D_m_test[120])#normalising expected count  
inital_fit_norm=normal(inital_fit)
data_norm=normal(data)#normalising data

plt.figure()#plot of normalised oscillated flux x survial probalitity as a function of mixing angle and mass squared, compared to the actual data
plt.bar(x=E,height=data_norm,width=0.05,color='blue',label="Data")
plt.bar(x=E,height=inital_fit_norm,width=0.05,alpha=0.7,color='red',label="Expected number of neutrinos")
plt.legend()
plt.xlabel("Energy (GeV)")
plt.ylabel('Normalised number of neutrinos')
plt.title("Expected count based on intial guess")
plt.grid()
plt.show()

NLL_y=NLL(theta_test,D_m_test[120])#NLL (as a function of mixing angle) for the estimated mass squared value previously found

plt.figure()
plt.plot(theta_test,NLL_y,'-r')#plotting NLL as a funtion of theta at a fixed m.
plt.xlabel(r'$\theta_{23}$')
plt.ylabel(r'NLL($\theta_{23}$)')
plt.title(r'NLL($\theta_{23},m^2_{23}=0.0024$)')
plt.grid()
plt.show()


plt.figure()
theta_min,theta_test3,para_err_0=parabolic_min(theta_test,NLL_y,plot=True,i1=20,i2=80,i3=120,choose_start=True)#minimising using the parabolic minisation method
NLL_y_min=NLL(theta_min,D_m_test[120],single=True)


stdev_theta=STerr(theta_test,NLL_y)#determining the error of minimum 
stdev_theta_analytic=err_analytic(theta_min,D_m_test[120],variable="theta")
print('-------------')
print('-------------')
print("TESTING FUNCTIONS NLL(theta,m_fixed)")
print("Minimising Mixing Angle for Change in mass squared=",D_m_test[120],"without considering the Cross-Section:",theta_min)
print("->Standard Dev error for Mixing Angle:",stdev_theta[0])#shifting err
print("->Parabolic approx error for Mixing Angle:",theta_min-para_err_0[0])#parabolic approx err
print("->Analytic error for Mixing Angle:",theta_min-stdev_theta_analytic[0])#analytic error -> not mentioned in the report


plt.plot(theta_test,NLL_y,'-r',label=r'NLL($\theta_{23}$)')
plt.xlabel(r'$\theta_{23}$')#re-plotting graph
plt.ylabel(r'NLL($\theta_{23}$)')
plt.title(r'NLL($\theta_{23},m^2_{23}=0.0024$)')
plt.legend()
plt.grid()
plt.show()

#%%
theta_min_1,delta_m_min_1,sigma_1=univariate_corr(theta_test,D_m_test,sig_rate_test,sigmaON=False)#unicariate minimisation of theta and m
theta_min_g,delta_m_min_g,sigma_2=gradMethod_corr(theta_test,D_m_test,sig_rate_test,sigmaON=False)#graident minimisation of theta and m

stdev_theta_min=STerr(theta_test,NLL(theta_test,delta_m_min_1[-1],variable="theta"))#Error found using standard deviation for univariate
stdev_m_min=STerr(D_m_test,NLL(theta_min_1[-1],D_m_test,variable="m"))

stdev_theta_min_g=STerr(theta_test,NLL(theta_test,delta_m_min_g[-1],variable="theta"))#Error found using standard deviation for gradient
stdev_m_min_g=STerr(D_m_test,NLL(theta_min_g[-1],D_m_test,variable="m"))
print('-------------')
print('-------------')
print("TESTING FUNCTIONS NLL(theta,m)")
print('Considering only the Mixing Angle and Change in Mass Squared:' )
print('--Univariate Method:--')
print( '->Mixing Angle=',theta_min_1[-1],'+/-',stdev_theta_min[0])
print('->Change in Mass Squared=',delta_m_min_1[-1],'+/-',stdev_m_min[0])
print('-------------')
print('--Gradient Method:--' )
print('->Mixing Angle=',theta_min_g[-1],'+/-',stdev_theta_min_g[0])
print('->Change in Mass Squared=',delta_m_min_g[-1],'+/-',stdev_m_min_g[0])
#%%

theta_1,D_m_1=np.meshgrid(theta_test,D_m_test)#meshgrid function generates arrays of the parameters in the correct shape
NLL_1=NLL_ScalPlot(theta_1,D_m_1)

fig = plt.figure()
surf=plt.pcolormesh(theta_1,D_m_1, NLL_1, cmap='inferno',) #heat map to see how NLL changes for given values of the parameters
fig.colorbar(surf, shrink=1, aspect=10)
cs=plt.contour(theta_1,D_m_1, NLL_1,levels=np.linspace(400,1200,num=10,endpoint=False))#contour plot

for i in range(len(theta_min_g)):
    if i==0:
        plt.plot(theta_min_g[i:i+2], delta_m_min_g[i:i+2], '.-',color="green",label="Gradient Method")
    else:
        plt.plot(theta_min_g[i:i+2], delta_m_min_g[i:i+2], '.-',color="green")

#plots the values of the parameters that NLL parabola at the end of each iteration. This is done by using the univariate minimization function. 
for i in range(len(theta_min_1)):
    if i==0:
        plt.plot(theta_min_1[i:i+2], delta_m_min_1[i:i+2], '.-',color="grey",label="Univariate Method")
    else:
        plt.plot(theta_min_1[i:i+2], delta_m_min_1[i:i+2], '.-',color="grey")
plt.xlabel(r'$\theta_{23}$')
plt.ylabel(r'$\Delta m_{23}^{2}$')
plt.title(r'NLL($\theta_{23},m^2_{23}$)')
plt.legend()
plt.show()


theta_min_4,delta_m_min_4,sig_rate_min_4=univariate_corr(theta_test,D_m_test,sig_rate_test,sigmaON=True)#unicariate minimisation of theta, sigma and m
theta_min_j,delta_m_min_j,sig_rate_min_j=gradMethod_corr(theta_test,D_m_test,sig_rate_test,sigmaON=True)#gradient minimisation of theta, sigma and m


stdev_theta_min_4=STerr(theta_test,NLL(theta_test,delta_m_min_4[-1],sig_rate_min_4[-1],variable="theta"))#Error found using standard deviation for univariate
stdev_m_min_4=STerr(D_m_test,NLL(theta_min_4[-1],D_m_test,sig_rate_min_4[-1],variable="m"))
stdev_sigma_min_4=STerr(sig_rate_test,NLL(theta_min_4[-1],delta_m_min_4[-1],sig_rate_test,variable="sigma"))

stdev_theta_min_j=STerr(theta_test,NLL(theta_test,delta_m_min_j[-1],sig_rate_min_j[-1],variable="theta"))#Error found using standard deviation for gradient
stdev_m_min_j=STerr(D_m_test,NLL(theta_min_j[-1],D_m_test,sig_rate_min_j[-1],variable="m"))
stdev_sigma_min_j=STerr(sig_rate_test,NLL(theta_min_j[-1],delta_m_min_j[-1],sig_rate_test,variable="sigma"))

print('-------------')
print('-------------')
print("TESTING FUNCTIONS NLL(theta,m_fixed,sigma)")
print('Considering the Mixing Angle, Change in Mass Squared and the Rate of Increase of the Cross-Section: ')
print('--Univariate Method:--')
print('->Mixing Angle=',theta_min_4[-1],'+/-',stdev_theta_min_4[0])
print('->Change in Mass Squared=',delta_m_min_4[-1],'+/-',stdev_m_min_4[0])
print('->Rate of Increase of Cross-Section=',sig_rate_min_4[-1],'+/-',stdev_sigma_min_4[0])
print('-------------')
print('--Gradient Method:--')
print('->Mixing Angle=',theta_min_j[-1],'+/-',stdev_theta_min_j[0])
print('->Change in Mass Squared=',delta_m_min_j[-1],'+/-',stdev_m_min_j[0])
print('->Rate of Increase of Cross-Section=',sig_rate_min_j[-1],'+/-',stdev_sigma_min_j[0])

plt.figure()
plt.bar(x=E,height=normal(data),width=0.05,color='blue',label="Data")
plt.bar(x=E,height=normal(unOssFlux*surv_prob_corr(theta_min_4[-1],delta_m_min_4[-1],sig_rate_min_4[-1])),width=0.05,alpha=0.7,color='red',label="Expected number of neutrinos")
plt.legend()
plt.xlabel("Energy (GeV)")
plt.ylabel('Normalised number of neutrinos')
plt.title("Expected count based on final minimised parameters")

plt.show()