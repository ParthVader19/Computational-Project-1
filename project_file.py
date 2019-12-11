
import numpy as np
import matplotlib.pyplot as plt
import time
#from matplotlib import cm
#from mpl_toolkits import mplot3d



#survival probablity assuming that cross-section is constant with energy
def survival_prob(theta,D_m,E=E,L=L):
    return 1-(np.sin(2*theta)**2)*(np.sin(1.267*(D_m)*L/E))**2

def surv_prob_corr(theta,m,sig_rate=1/E,E=E,L=L):
    return (1-(np.sin(2*theta)**2)*(np.sin(1.267*(m)*L/E))**2)*sig_rate*E

# NLL as a function of theta

def NLL(theta,m,sigma=1/E,O=data,unOssFlux=unOssFlux,variable="theta",single=False): #finds the NLL for a given a range of mixing angle.
    
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


def parabolic_min(x,y,plot=False):# 1 parameter minimasation using the parabolic method

    i1=40 #set indexes that give the best result. 
    i2=51
    i3=60
    x_test=[x[i1],x[i2],x[i3]]
    y_test=[y[i1],y[i2],y[i3]]

    current_min=1000#used to compare to the minimum for the while loop. 
    
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
    """
    NEED TO FIX THIS:
        -curvature
        -stdev
    """
    d=(x_test[1]-x_test[0])*(x_test[2]-x_test[0])*(x_test[2]-x_test[1])
    C=(x_test[2]-x_test[1])*y_test[0]/d + (x_test[0]-x_test[2])*y_test[1]/d +(x_test[1]-x_test[0])*y_test[2]/d

    return x_test[y_test.index(min(y_test))],x_test1,C



#the following is the univariate minimization function 
def univariate(theta,delta_m):# 2 parameter minimasation using the parabolic method to minimise the paramaters sequential.
    
    theta_min=theta[45]#inital parameters from the beginning part.
    delta_m_min=delta_m[60]
    
    old_theta_min=0#used to compare determine the change in the values of the parameters for the while loop. 
    old_delta_m_min=0
    
    theta_1=[theta_min]#arrays of the parameters used for plotting later
    delta_m_1=[delta_m_min]
    
    while abs(old_theta_min-theta_min)>1e-7 and abs(old_delta_m_min-delta_m_min)>1e-7:#loop stops when the change in the minimum value of the function is very small
        old_theta_min=theta_min#setting the current parameter values
        old_delta_m_min=delta_m_min
        
        NLL_y=NLL(theta,delta_m_min)#calculating NLL as a function of mixing angle for the current value of mass squared
        
        theta_min,theta_test3,theta_min_curvature=parabolic_min(theta_test,NLL_y,plot=False)#performing parabolic minimisation to find the new value of mixing angle
        theta_1.append(theta_min)
        delta_m_1.append(delta_m_min)#storing the values for plotting later
        
        NLL_y_1=NLL(theta_min,D_m_test,variable="m")#calculating NLL as a function of mass squared for the current value of mixing angle
        
        delta_m_min,delta_m_test3,delta_m_curvature=parabolic_min(D_m_test,NLL_y_1,plot=False)#performing parabolic minimisation to find the new value of mass squared
        theta_1.append(theta_min)#storing the values for plotting later
        delta_m_1.append(delta_m_min)
        
    return theta_1,delta_m_1#final values of the arrays are the values of the parameters that minimise NLL

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


#The following is the gradient method for minimizing NLL
def gradMethod(x,y,a1=0.00001,a2=0.000000001,delta_theta=0.001,delta_m=0.00001):
    x_min=x[45]#starting values of the parameter that minimise NLL
    y_min=y[60]

    x_min_old=0#used to compare determine the change in the values of the parameters for the while loop. 
    y_min_old=0
    
    x_array=[x_min]#arrays of the parameters used for plotting later
    y_array=[y_min]

    
    while abs(y_min-y_min_old)>1e-9 and abs(x_min-x_min_old)>1e-9:#loop stops when the change in the minimum value of the function is very small
        x_min_old=x_min#setting the current parameter values
        y_min_old=y_min
        
        z_0=NLL(x_min,y_min,single=True)#finds the NLL at the current value of the parameters
            
        z_x1=NLL(x_min+delta_theta,y_min,single=True)#finds the NLL at the current value of y and an updated value of x of the parameters
        diff_x=(z_x1-z_0)/delta_theta#partial differentiation with repect to x 
        
        z_y1=NLL(x_min,y_min+delta_m,single=True)#finds the NLL at the current value of x and an updated value of y of the parameters
        diff_y=(z_y1-z_0)/delta_m#partial differentiation with repect to y
        
        x_min=x_min_old-a1*diff_x#taking a small step in x and y in the negative direction to the gradient change   
        y_min=y_min_old-a2*diff_y
        
        x_array.append(x_min)#storing the iterated values of the parameter for plotting later 
        y_array.append(y_min)
    
    return x_array,y_array

def univariate_corr(theta,delta_m,sig_rate,sigmaON,NLL=NLL,):
    theta_min=theta[45]#inital parameters from the beginning part.
    delta_m_min=delta_m[60]
    
    if sigmaON==True:
        sig_rate_min=sig_rate[50]
    else:
        sig_rate_min=0#for sigmaON=False, i.e. effects of the cross section is not taken into account

    #print(sig_rate_min)
    old_theta_min=0
    old_delta_m_min=0#used to compare determine the change in the values of the parameters for the while loop. 
    old_sig_rate_min=0
    
    theta_1=[theta_min]#arrays of the parameters used for plotting later
    delta_m_1=[delta_m_min]
    sig_rate_1=[sig_rate_min]
        
    #loop stops when the change in the parameter values giving the minimum value of NLL is very small
    while abs(old_theta_min-theta_min)>1e-7 or abs(old_delta_m_min-delta_m_min)>1e-7 or abs(old_sig_rate_min-sig_rate_min)>1e-4:
        
        old_theta_min=theta_min#setting the current parameter values
        old_delta_m_min=delta_m_min
        old_sig_rate_min=sig_rate_min
        
        if sigmaON==True:#Take into account the effect of cross section if enabled
            NLL_y=NLL(theta,delta_m_min,sig_rate_min)#calculating NLL as a function of mixing angle for the current value of mass squared
            
            theta_min,theta_test3,theta_min_curvature=parabolic_min(theta_test,NLL_y,plot=False)#performing parabolic minimisation to find the new value of mixing angleand cross section rate
            theta_1.append(theta_min)#storing the values for plotting later
            delta_m_1.append(delta_m_min)
            sig_rate_1.append(sig_rate_min)
            
            NLL_y_1=NLL(theta_min,delta_m,sig_rate_min,variable="m")#calculating NLL as a function of mass sqaured for the current value of mixing angle and cross section rate
            
            delta_m_min,delta_m_test3,delta_m_min_curvature=parabolic_min(delta_m,NLL_y_1,plot=False)#performing parabolic minimisation to find the new value of mixing angle
            theta_1.append(theta_min)
            delta_m_1.append(delta_m_min)#storing the values for plotting later
            sig_rate_1.append(sig_rate_min)
            
            
            NLL_y_2=NLL(theta_min,delta_m_min,sig_rate,variable="sigma")#calculating NLL as a function of cross section rate for the current value of mass squared and mixing angle
            
            sig_rate_min,sigma_test3,sigma_min_curvature=parabolic_min(sig_rate,NLL_y_2,plot=False)#performing parabolic minimisation to find the new value of mixing angle
            theta_1.append(theta_min)
            delta_m_1.append(delta_m_min)#storing the values for plotting later
            sig_rate_1.append(sig_rate_min)
            
        elif sigmaON==False:
            NLL_y=NLL(theta,delta_m_min)#calculating NLL as a function of mixing angle for the current value of mass squared
            
            theta_min,theta_test3,theta_min_curvature=parabolic_min(theta_test,NLL_y,plot=False)#performing parabolic minimisation to find the new value of mixing angle
            theta_1.append(theta_min)
            delta_m_1.append(delta_m_min)#storing the values for plotting later
           
            
            NLL_y_1=NLL(theta_min,delta_m,variable="m")#calculating NLL as a function of mass squared for the current value of mixing angle
            
            delta_m_min,delta_m_test3,delta_m_min_curvature=parabolic_min(delta_m,NLL_y_1,plot=False)#performing parabolic minimisation to find the new value of mixing angle
            theta_1.append(theta_min)
            delta_m_1.append(delta_m_min)#storing the values for plotting later
        
    return theta_1,delta_m_1,sig_rate_1


#%%
def gradMethod_corr(x,y,j,sigmaON,a1=0.00001,a2=0.000000001,a3=0.000001,delta_theta=0.01,delta_m=0.0001,delta_sig_rate=0.0001,NLL=NLL):
    x_min=x[45]
    y_min=y[60]
    if sigmaON==True:
        j_min=j[50]
    else:
        j_min=0

    x_min_old=0
    y_min_old=0
    j_min_old=0
    
    x_array=[x_min]
    y_array=[y_min]
    j_array=[j_min]
    
    while abs(y_min-y_min_old)>1e-7 or abs(x_min-x_min_old)>1e-7 or abs(j_min-j_min_old)>1e-4 :
#    for k in range(1000):
        
        x_min_old=x_min#setting the current parameter values
        y_min_old=y_min
        j_min_old=j_min
        
        if sigmaON==True:
            z_0=NLL(x_min,y_min,j_min,single=True)#finds the NLL at the current value of the parameters
                
            z_x1=NLL(x_min+delta_theta,y_min,j_min,single=True)#finds the NLL at the current value of y and an updated value of x of the parameters
            diff_x=(z_x1-z_0)/delta_theta#partial differentiation with repect to x 
            
            z_y1=NLL(x_min,y_min+delta_m,j_min,single=True)#finds the NLL at the current value of x and an updated value of y of the parameters
            diff_y=(z_y1-z_0)/delta_m#partial differentiation with repect to y
            
            z_y2=NLL(x_min,y_min,j_min+delta_sig_rate,single=True)#finds the NLL at the current value of x and an updated value of y of the parameters
            diff_j=(z_y2-z_0)/delta_sig_rate#partial differentiation with repect to y
            
            x_min=x_min_old-a1*diff_x#taking a small step in x and y in the negative direction to the gradient change   
            y_min=y_min_old-a2*diff_y
            j_min=j_min_old-a3*diff_j
            
            x_array.append(x_min)#storing the iterated values of the parameter for plotting later 
            y_array.append(y_min)
            j_array.append(j_min)
            
        elif sigmaON==False :
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

#
#for i in range(len(theta_min_g)):
#    plt.plot(theta_min_g[i:i+2], delta_m_min_g[i:i+2], '.-',color="green")
#plt.show()






#
#%% 
#values given in the project
theta=np.pi/4
D_m=2.4e-3
L=295

num=100

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


#TESTING:
time_s=time.time()
plt.figure()#plot of Unoscillated flux x survial probalitity as a function of mixing angle and mass squared, compared to the actual data
plt.bar(x=E,height=data,width=0.05,color='blue',label="Observed events")
plt.bar(x=E,height=unOssFlux*surv_prob_corr(theta_test[45],D_m_test[60]),width=0.05,alpha=0.7,color='red',label="Expected event rate")
plt.legend()
plt.xlabel("Energy (GeV)")
plt.ylabel("No. of occurrences")
plt.grid()
plt.show()


NLL_y=NLL(theta_test,D_m_test[60])#NLL (as a function of mixing angle) for the estimated mass squared value previously found

plt.figure()
plt.plot(theta_test,NLL_y,'-r')
plt.xlabel("Theta")
plt.ylabel("NLL")
plt.grid()
plt.show()
#%%
#NLL_y_1=NLL(theta_test[45],np.linspace(0,1000e-3,num=1000,endpoint=False)+1e-30,variable="m")#NLL (as a function of mixing angle) for the estimated mass squared value previously found
#
#plt.figure()
#plt.plot(np.linspace(0,1000e-3,num=1000,endpoint=False)+1e-30,NLL_y_1,'-r')
#plt.xlabel("Theta")
#plt.ylabel("NLL")
#plt.grid()
#plt.show()
#%%
plt.figure()
theta_min,theta_test3,curvature=parabolic_min(theta_test,NLL_y,plot=True)
plt.plot(theta_test,NLL_y,'-r',label="NLL(Theta)")
plt.xlabel("Theta")
plt.ylabel("NLL")
plt.legend()
plt.grid()
plt.show()


theta_min,delta_m_min,sigma_1=univariate_corr(theta_test,D_m_test,sig_rate_test,sigmaON=False)
theta_min_g,delta_m_min_g,sigma_2=gradMethod_corr(theta_test,D_m_test,sig_rate_test,sigmaON=False)


print(theta_min[-1],delta_m_min[-1])
print(theta_min_g[-1],delta_m_min_g[-1])

theta_1,D_m_1=np.meshgrid(theta_test,D_m_test)#meshgrid function generates arrays of the parameters in the correct shape
NLL_1=NLL_ScalPlot(theta_1,D_m_1)

fig = plt.figure()
surf=plt.pcolormesh(theta_1,D_m_1, NLL_1, cmap='inferno',) #heat map to see how NLL changes for given values of the parameters
fig.colorbar(surf, shrink=1, aspect=10)
cs=plt.contour(theta_1,D_m_1, NLL_1,levels=[500,600,700, 900, 1100])#contour plot


for i in range(len(theta_min_g)):
    if i==0:
        plt.plot(theta_min_g[i:i+2], delta_m_min_g[i:i+2], '.-',color="green",label="Gradient Method")
    else:
        plt.plot(theta_min_g[i:i+2], delta_m_min_g[i:i+2], '.-',color="green")

#plots the values of the parameters that NLL parabola at the end of each iteration. This is done by using the univariate minimization function. 
for i in range(len(theta_min)):
    if i==0:
        plt.plot(theta_min[i:i+2], delta_m_min[i:i+2], '.-',color="grey",label="Univariate Method")
    else:
        plt.plot(theta_min[i:i+2], delta_m_min[i:i+2], '.-',color="grey")
plt.xlabel("Mixing angle")
plt.ylabel("Change in mass sqaured")
plt.legend()
plt.show()


print("------------")
theta_min_4,delta_m_min_4,sig_rate_min_4=univariate_corr(theta_test,D_m_test,sig_rate_test,sigmaON=True)
theta_min_j,delta_m_min_j,sig_rate_min_j=gradMethod_corr(theta_test,D_m_test,sig_rate_test,sigmaON=True)
print(theta_min_4[-1],delta_m_min_4[-1],sig_rate_min_4[-1])
print(theta_min_j[-1],delta_m_min_j[-1],sig_rate_min_j[-1])

plt.figure()
plt.bar(x=E,height=data,width=0.05,color='blue',label="Observed events")
plt.bar(x=E,height=unOssFlux*surv_prob_corr(theta_min_4[-1],delta_m_min_4[-1],sig_rate_min_4[-1]),width=0.05,alpha=0.7,color='red',label="Expected event rate")
plt.legend()
plt.show()
time_e=time.time()
print(time_e-time_s)

