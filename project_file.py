
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import cm
#from mpl_toolkits import mplot3d


#values given in the project
theta=np.pi/4
D_m=2.4e-3
L=295

E=np.linspace(0,10,num=200,endpoint=True)+0.025#energy in the region of interest.
D_m_test=np.linspace(0,4e-3,num=100,endpoint=False)#a range for the oscillation parameters 
theta_test=np.linspace(0,2*np.pi/4,num=100,endpoint=False)

#observed number of muon neutrino events from 0–10 GeV in energy
data=[0	,0	,1	,0	,8	,13	,11	,10	,5	,5	,0	,0	,8	,10	,9	,20	,12	,14	,15	,8	,5	,6	,3	,7	,6	,3	,3	,5	,1	,6	,2	,4	,3	,3	,2	,0	,2	,3	,6	,6	,4	,5	,5	,2	,0	,3	,1	,5	,2	,4	,1	,2	,0	,7	,3	,2	,1	,0	,5	,2	,4	,1	,1	,0	,3	,0	,3	,6	,3	,9	,2	,3	,4	,4	,3	,4	,2	,8	,4	,9	,10	,5	,2	,3	,5	,4	,4	,4	,3	,1	,3	,3	,6	,6	,4	,5	,5	,7	,4	,3	,4	,3	,8	,2	,2	,3	,6	,2	,1	,1	,6	,4	,2	,1	,5	,2	,1	,1	,2	,2	,3	,1	,3	,0	,0	,3	,2	,1	,4	,2	,1	,0	,3	,2	,0	,2	,3	,1	,0	,0	,0	,2	,3	,1	,1	,2	,0	,1	,2	,0	,2	,0	,0	,1	,2	,1	,0	,2	,0	,0	,1	,1	,1	,0	,0	,0	,1	,0	,0	,1	,1	,0	,0	,0	,0	,0	,0	,0	,0	,1	,0	,0	,1	,0	,0	,0	,0	,0	,0	,0	,0	,1	,0	,0	,0	,0	,0	,0	,0	,1]
#simulated event rate prediction assuming the muon neutrinos do not oscillate with energy from 0–10 GeV
unOssFlux=[1.21438190	,5.62230639	,11.77215522	,18.36411225	,27.21794480	,38.10249543	,51.19882744	,64.91880229	,80.03030054	,97.50418953	,120.84615030	,145.78791260	,152.75765800	,131.59242800	,100.08952310	,73.63209608	,54.60231744	,39.48028161	,26.47861379	,17.63748553	,12.19992341	,9.28609970	,7.64249628	,6.21910485	,5.24044992	,4.65509306	,4.07045634	,3.51118618	,3.25569148	,2.80925349	,2.61069663	,2.39112544	,2.17705579	,1.99959170	,1.92357213	,1.75687067	,1.64994092	,1.56213126	,1.52333637	,1.41260526	,1.38441627	,1.27925977	,1.20698624	,1.16664927	,1.15796934	,1.12386360	,1.07388241	,1.00892838	,1.00460651	,0.98270005	,1.00503534	,0.93658968	,0.93019993	,0.94318783	,0.93619016	,0.89221666	,0.91207888	,0.91133428	,0.89153404	,0.91460231	,0.90674946	,0.89750422	,0.90380133	,0.89190251	,0.87688821	,0.87117490	,0.88346685	,0.87092222	,0.87619063	,0.89471357	,0.85915394	,0.89143436	,0.88726983	,0.89438375	,0.90156309	,0.89889511	,0.90664908	,0.91678198	,0.91735551	,0.92122476	,0.91037453	,0.91495478	,0.92157440	,0.91190644	,0.88883750	,0.89049827	,0.89026236	,0.85621603	,0.84290360	,0.82111156	,0.81243927	,0.79481040	,0.78264333	,0.75665926	,0.74517757	,0.70948241	,0.70109569	,0.67637266	,0.66340054	,0.63552800	,0.61294571	,0.59219409	,0.55411851	,0.53539725	,0.51340540	,0.47555007	,0.47816945	,0.45024840	,0.42395841	,0.39611960	,0.38262378	,0.35831866	,0.35320362	,0.34146820	,0.31617893	,0.30419352	,0.28289478	,0.29198448	,0.25936696	,0.25987293	,0.24309008	,0.23736998	,0.22430119	,0.22206898	,0.21028855	,0.20396381	,0.19452954	,0.18692220	,0.18346423	,0.17720325	,0.16601350	,0.17040633	,0.15991563	,0.15169332	,0.14868690	,0.14962555	,0.13757585	,0.13089846	,0.13002453	,0.12403647	,0.12442329	,0.11985344	,0.11272974	,0.10665933	,0.10935548	,0.10548309	,0.10238063	,0.09527973	,0.09327825	,0.09051113	,0.08820124	,0.08702189	,0.08249344	,0.08421637	,0.07895646	,0.07525440	,0.06863699	,0.07162078	,0.06418893	,0.06745232	,0.06564193	,0.06297574	,0.06379705	,0.06113915	,0.05391862	,0.05683304	,0.05303102	,0.05072655	,0.05001585	,0.05327111	,0.04463483	,0.04730818	,0.04339604	,0.04422675	,0.04255934	,0.04203425	,0.04107220	,0.03729259	,0.03287503	,0.03320166	,0.03638846	,0.03233257	,0.03283617	,0.02868774	,0.02943291	,0.02913883	,0.02856542	,0.02825330	,0.02511534	,0.02472052	,0.02411842	,0.02670751	,0.02173993	,0.02349169	,0.02007420	,0.02392712	,0.02255197	,0.01960980	,0.01891021	,0.02015019
]

#survival probablity assuming that cross-section is constant with energy
def survival_prob(theta,D_m,E=E,L=295):
    return 1-(np.sin(2*theta)**2)*(np.sin(1.267*(D_m)*L/E))**2

plt.figure()#plot of Unoscillated flux x survial probalitity as a function of mixing angle and mass squared, compared to the actual data
plt.bar(x=E,height=data,width=0.05,color='blue',label="Observed events")
plt.bar(x=E,height=unOssFlux*survival_prob(theta_test[45],D_m_test[60]),width=0.05,alpha=0.7,color='red',label="Expected event rate")
plt.legend()
plt.xlabel("Energy (GeV)")
plt.ylabel("No. of occurrences")
plt.grid()
plt.show()

# NLL as a function of theta
  
def NLL(theta,m,O=data,variable="theta"): #finds the NLL for a given a range of mixing angle.
    
    
    NLL_rate=[]#loop used to get an array of expected oscillating rate 
    if variable=="theta":
        for i in range(len(theta)):
            NLL_rate.append(unOssFlux*survival_prob(theta[i],m))
    else:
        for i in range(len(m)):
            NLL_rate.append(unOssFlux*survival_prob(theta,m[i]))
    
    NLL_y=[]#use the expected oscillating rate and the actual data to perform NLL using the formula given
    for j in range(len(NLL_rate)):
        sum0=0
        for i in range(len(NLL_rate[j])):
            if O[i]==0:#computer cannot handle 0*np.log(0) hence condition used to avoid 'nan' answer
                sum0+=NLL_rate[j][i]-O[i]
            else:
                sum0+=NLL_rate[j][i]-O[i]+O[i]*np.log(O[i]/NLL_rate[j][i])
        NLL_y.append(sum0)
    
    return NLL_y


NLL_y=NLL(theta_test,D_m_test[60])#NLL (as a function of mixing angle) for the estimated mass squared value previously found

plt.figure()
plt.plot(theta_test,NLL_y,'-r')
plt.xlabel("Theta")
plt.ylabel("NLL")
plt.grid()
plt.show()

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

plt.figure()
theta_min,theta_test3,curvature=parabolic_min(theta_test,NLL_y,plot=True)
plt.plot(theta_test,NLL_y,'-r',label="NLL(Theta)")
plt.xlabel("Theta")
plt.ylabel("NLL")
plt.legend()
plt.grid()
plt.show()

print("min:",theta_min)

print("curvature:",curvature)

#%%
"""
------------------------------
continue commenting and editting!!!
"""

def univariate(theta,delta_m):# 2 parameter minimasation using the parabolic method to minimise the paramaters sequential.
    
    theta_min=theta[45]#inital parameters from the beginning part.
    delta_m_min=delta_m[60]
    
    old_theta_min=0#used to compare determine the change in the values of the parameters for the while loop. 
    old_delta_m_min=0
    
    theta_1=[theta_min]#arrays of the parameters used for plotting later
    delta_m_1=[delta_m_min]
    
    while abs(old_theta_min-theta_min)>1e-7 and abs(old_delta_m_min-delta_m_min)>1e-7:#loop stops when the change in the minimum value of the function is very small
        old_theta_min=theta_min
        old_delta_m_min=delta_m_min
        
        NLL_y=NLL(theta,delta_m_min)
        
        theta_min,theta_test3,theta_min_curvature=parabolic_min(theta_test,NLL_y,plot=False)
        theta_1.append(theta_min)
        delta_m_1.append(delta_m_min)
        
        NLL_y_1=NLL(theta_min,D_m_test,variable="m")
        
        delta_m_min,delta_m_test3,delta_m_curvature=parabolic_min(D_m_test,NLL_y_1,plot=False)
        theta_1.append(theta_min)
        delta_m_1.append(delta_m_min)
        
    return theta_1,delta_m_1
        
theta_min,delta_m_min=univariate(theta_test,D_m_test)
print(theta_min[-1],delta_m_min[-1])

#%%
theta_1,D_m_1=np.meshgrid(theta_test,D_m_test)

def survival_prob_new(E,theta_1,D_m_1,L):
    prob_b=[]
    for i in range(len(theta_1)):
        prob=[]
        for j in range(len(D_m_1[0])):
           prob.append(1-(np.sin(2*theta_1[i][j])**2)*(np.sin(1.267*(D_m_1[i][j])*L/E))**2) 
        prob_b.append(prob)
    return prob_b

sur_prob=survival_prob_new(E,theta_1,D_m_1,L)

def NLL_new(sur_prob):
    rate_b=[]
    for i in range(len(sur_prob)):
        rate=[]
        for j in range(len(sur_prob[0])):
            rate.append(unOssFlux*sur_prob[i][j])
        rate_b.append(rate)
        
    NLL_y_b=[]
    for i in range(len(rate_b)):
        NLL_y=[]
        for j in range(len(rate_b[0])):
            NLL_y.append(NLL(rate_b[i][j],data))
        NLL_y_b.append(NLL_y)
        
    return NLL_y_b

Z=NLL_new(sur_prob)
fig = plt.figure()
surf=plt.pcolormesh(theta_1,D_m_1, Z, cmap='inferno') 
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.contour(theta_1,D_m_1, Z)
for i in range(len(theta_min)):
    plt.plot(theta_min[i:i+2], delta_m_min[i:i+2], '.-',color="black")
print(theta_min[-1],delta_m_min[-1])

#%%
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#
#ax.plot_surface(theta_1,D_m_1, np.array(Z),cmap='viridis', edgecolor='none')
#ax.set_title('Surface plot')
#plt.show()

#%%
def gradMethod(x,y,a1=0.00001,a2=0.000000001,delta_theta=0.01,delta_m=0.0001):
    x_min=x[20]
    y_min=y[50]
#    x_min=0.6777809204207589
#    y_min=0.0026619195151839697
    x_min_old=0
    y_min_old=0
    
    x_array=[x_min]
    y_array=[y_min]
    zx=[]
    
    while abs(y_min-y_min_old)>0.0000000001 and abs(x_min-x_min_old)>0.000001 :
    
        x_min_old=x_min
        y_min_old=y_min
        
        sur_prob_0=unOssFlux*survival_prob(E,x_min,y_min,L)
        z_0=NLL(sur_prob_0,data)
        zx.append(z_0)
            
        sur_prob_x1=unOssFlux*survival_prob(E,x_min+delta_theta,y_min,L)
        z_x1=NLL(sur_prob_x1,data)
        diff_x=(z_x1-z_0)/delta_theta
        
        sur_prob_y1=unOssFlux*survival_prob(E,x_min,y_min+delta_m,L)
        z_y1=NLL(sur_prob_y1,data)
        diff_y=(z_y1-z_0)/delta_m
        
        x_min=x_min_old-a1*diff_x
        y_min=y_min_old-a2*diff_y
        
        x_array.append(x_min)
        y_array.append(y_min)
    
    return x_array,y_array,zx

theta_min_g,delta_m_min_g,zx=gradMethod(theta_test,D_m_test)
for i in range(len(theta_min_g)):
    plt.plot(theta_min_g[i:i+2], delta_m_min_g[i:i+2], '.-',color="green")
plt.show()
print(theta_min_g[-1],delta_m_min_g[-1])
print(theta_min[-1],delta_m_min[-1])
#plt.plot(delta_m_min_g,zx,'.-')
##plt.plot(theta_min_g,zx,'.-')
##plt.plot(theta_min_g[-1],zx[-1],'or')
##plt.plot(theta_min_g[0],zx[0],'o',color="black")
#plt.plot(delta_m_min_g[-1],zx[-1],'or')
#plt.plot(delta_m_min_g[0],zx[0],'o',color="black")
#%%
sig_rate_test=np.linspace(0,2,num=100,endpoint=False)

def surv_prob_corr(sig_rate,E,theta,D_m,L):
    return survival_prob(E,theta,D_m,L)*sig_rate*E


def univariate_corr(theta,delta_m,sig_rate):
    theta_min=theta[90]
    delta_m_min=delta_m[90]
    sig_rate_min=sig_rate[90]
    
    old_theta_min=0
    old_delta_m_min=0
    old_sig_rate_min=0
    
    theta_1=[theta_min]
    delta_m_1=[delta_m_min]
    sig_rate_1=[sig_rate_min]
    
    while abs(old_theta_min-theta_min)>0.00001 and abs(old_delta_m_min-delta_m_min)>0.00001 and abs(old_sig_rate_min-sig_rate_min)>0.0001:

        old_theta_min=theta_min
        old_delta_m_min=delta_m_min
        old_sig_rate_min=sig_rate_min
        
        NLL_rate=[]
        for i in range(len(theta_test)):
            NLL_rate.append(unOssFlux*surv_prob_corr(sig_rate_min,E,theta_test[i],delta_m_min,L))
            
        NLL_y=[]
        for i in range(len(NLL_rate)):
            NLL_y.append(NLL(NLL_rate[i],data))
        
        theta_min,theta_test3,theta_min_curvature=parabolic_min(theta_test,NLL_y,plot=False)
        theta_1.append(theta_min)
        delta_m_1.append(delta_m_min)
        sig_rate_1.append(sig_rate_min)
        
        NLL_rate_1=[]
        for i in range(len(D_m_test)):
            NLL_rate_1.append(unOssFlux*surv_prob_corr(sig_rate_min,E,theta_min,D_m_test[i],L))
            
        NLL_y_1=[]
        for i in range(len(NLL_rate_1)):
            NLL_y_1.append(NLL(NLL_rate_1[i],data))
        
        delta_m_min,delta_m_test3,delta_m_curvature=parabolic_min(D_m_test,NLL_y_1,plot=False)
        theta_1.append(theta_min)
        delta_m_1.append(delta_m_min)
        sig_rate_1.append(sig_rate_min)
        
        NLL_rate_2=[]
        for i in range(len(sig_rate_test)):
            NLL_rate_2.append(unOssFlux*surv_prob_corr(sig_rate_test[i],E,theta_min,delta_m_min,L))
            
        NLL_y_2=[]
        for i in range(len(NLL_rate_2)):
            NLL_y_2.append(NLL(NLL_rate_2[i],data))
        
        sig_rate_min,sig_rate_test3,sig_rate_curvature=parabolic_min(sig_rate_test,NLL_y_2,plot=False)
        theta_1.append(theta_min)
        delta_m_1.append(delta_m_min)
        sig_rate_1.append(sig_rate_min)
        
    return theta_1,delta_m_1,sig_rate_1


theta_min_4,delta_m_min_4,sig_rate_min_4=univariate_corr(theta_test,D_m_test,sig_rate_test)
print(theta_min_4[-1],delta_m_min_4[-1],sig_rate_min_4[-1])
#%%
#plt.figure()
#plt.bar(x=E,height=unOssFlux*surv_prob_corr(1.2846535223642612,E,0.7853981633974484,0.00233318519016147,L),color='red',width=0.05)
#plt.bar(x=E,height=data,width=0.05,color='blue',alpha=0.7,label="Data")
#plt.show()
#
#plt.figure()
#plt.bar(x=E,height=unOssFlux*survival_prob(E,0.7853981633974484,0.00233318519016147,L),color='red',width=0.05)
#plt.bar(x=E,height=data,width=0.05,color='blue',alpha=0.7,label="Data")
#plt.show()
#%%
def gradMethod_corr(x,y,j,a1=0.00001,a2=0.000000001,a3=0.000001,delta_theta=0.01,delta_m=0.0001,delta_sig_rate=0.0001):
    x_min=x[90]
    y_min=y[90]
    j_min=j[90]
#    x_min=0.6777809204207589
#    y_min=0.0026619195151839697
    x_min_old=0
    y_min_old=0
    j_min_old=0
    
    x_array=[x_min]
    y_array=[y_min]
    j_array=[j_min]
    
    while abs(y_min-y_min_old)>0.0000000001 and abs(x_min-x_min_old)>0.000001 and abs(j_min-j_min_old)>0.000001 :
#    for k in range(1000):
    
        x_min_old=x_min
        y_min_old=y_min
        j_min_old=j_min
        
        sur_prob_0=unOssFlux*surv_prob_corr(j_min,E,x_min,y_min,L)
        z_0=NLL(sur_prob_0,data)
            
        sur_prob_x1=unOssFlux*surv_prob_corr(j_min,E,x_min+delta_theta,y_min,L)
        z_x1=NLL(sur_prob_x1,data)
        diff_x=(z_x1-z_0)/delta_theta
        
        sur_prob_y1=unOssFlux*surv_prob_corr(j_min,E,x_min,y_min+delta_m,L)
        z_y1=NLL(sur_prob_y1,data)
        diff_y=(z_y1-z_0)/delta_m
        
        sur_prob_j1=unOssFlux*surv_prob_corr(j_min+delta_sig_rate,E,x_min,y_min,L)
        z_j1=NLL(sur_prob_j1,data)
        diff_j=(z_j1-z_0)/delta_m
        
        x_min=x_min_old-a1*diff_x
        y_min=y_min_old-a2*diff_y
        j_min=j_min_old-a3*diff_j
        
        x_array.append(x_min)
        y_array.append(y_min)
        j_array.append(j_min)
    
    return x_array,y_array,j_array

theta_min_j,delta_m_min_j,sig_rate_min_j=gradMethod_corr(theta_test,D_m_test,sig_rate_test)
#for i in range(len(theta_min_g)):
#    plt.plot(theta_min_g[i:i+2], delta_m_min_g[i:i+2], '.-',color="green")
#plt.show()
print(theta_min_g[-1],delta_m_min_g[-1],sig_rate_min_j[-1])
print(theta_min_4[-1],delta_m_min_4[-1],sig_rate_min_4[-1])
#%%

