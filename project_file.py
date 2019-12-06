"""
Things to do:
    -create an animation of the bar graph with different parameters of theata and delta m
    to figure out a good approx. of the parameters.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d


#inital values
theta=np.pi/4
D_m=2.4e-3
L=295

data,unOssFlux=np.loadtxt("data.csv",delimiter=',',unpack=True)#inporting the data

def survival_prob(E,theta,D_m,L):#survival probability for muon neutrino 
    return 1-(np.sin(2*theta)**2)*(np.sin(1.267*(D_m)*L/E))**2

E=np.linspace(0,10,num=200,endpoint=True)+0.025#energy in the region of interest.
#E_data=np.linspace(0,10,num=200,endpoint=True)+0.025
#%%
D_m_test=np.linspace(0,4e-3,num=100,endpoint=False)#a range for the oscillation parameters 
theta_test=np.linspace(0,2*np.pi/4,num=100,endpoint=False)


#%%
"""
Note that the data for a histogram (data is binned), therefore the line graph is plotted with
energy for each point representing the middle of the bin. E.g. first point E=0.025
"""
"""
plt.figure(3)#plot of Unoscillated flux
plt.bar(x=E_data,height=unOssFlux,width=0.05,color='blue',label="Unoscillated flux")
plt.bar(x=E_data,height=unOssFlux*survival_prob(E_data,theta,D_m,L),width=0.05,color='red',label="Unoscillated flux with Survival Prob applied")
plt.legend()
plt.xlabel("energy (GeV)")
plt.ylabel("No. of occurrences")
plt.grid()
plt.show()
"""

#%% NLL as a function of theta
"""

"""    
def NLL(R,O): #finds the NLL
    sum0=0
    for i in range(len(R)):
        if O[i]==0:
            sum0+=R[i]-O[i]
        else:
            sum0+=R[i]-O[i]+O[i]*np.log(O[i]/R[i])
    return sum0
"""
Need to put the next 2 for loops into the NLL function. 
"""
NLL_rate=[]
for i in range(len(theta_test)):
    NLL_rate.append(unOssFlux*survival_prob(E,theta_test[i],D_m,L))
    
NLL_y=[]
for i in range(len(NLL_rate)):
    NLL_y.append(NLL(NLL_rate[i],data))
    
plt.figure(10)
plt.plot(theta_test,NLL_y,'-r')
plt.xlabel("Theta")
plt.ylabel("NLL")
plt.grid()
plt.show()

#%% minimasation
def parabolic_min(x,y,plot=False):
    i1=40
    i2=51
    i3=60
    x_test=[x[i1],x[i2],x[i3]]
    y_test=[y[i1],y[i2],y[i3]]

    current_min=1000
    colour=["black","red","blue","green","purple","orange","grey"]
    c_index=0
    while abs(min(y_test)-current_min)>1e-10:
    #for k in range(4):
        print(len(x_test))
        current_min=min(y_test)
        x_test1=x_test
        if plot!=False:
            plt.plot(x_test1,y_test,'o', color=colour[c_index])
            z = np.polyfit(x_test, y_test, 2)
            poly_y=np.poly1d(z)
            plt.plot(theta_test,poly_y(theta_test),color=colour[c_index])
            plt.show()
        c_index+=1        
        
        x3=0.5*((x_test[2]**2 -x_test[1]**2)*y_test[0] +(x_test[0]**2 -x_test[2]**2)*y_test[1] +(x_test[1]**2 -x_test[0]**2)*y_test[2])/((x_test[2]-x_test[1])*y_test[0] + (x_test[0]-x_test[2])*y_test[1]+ (x_test[1]-x_test[0])*y_test[2])
        
        y3=(x3-x_test[1])*(x3-x_test[2])*y_test[0]/((x_test[0]-x_test[1])*(x_test[0]-x_test[2]))+ (x3-x_test[0])*(x3-x_test[2])*y_test[1]/((x_test[1]-x_test[0])*(x_test[1]-x_test[2]))+ (x3-x_test[0])*(x3-x_test[1])*y_test[2]/((x_test[2]-x_test[0])*(x_test[2]-x_test[1]))
        y_test.append(y3)
        x_test.append(x3)

        x_test.remove(x_test[y_test.index(max(y_test))])
        y_test.remove(max(y_test))
    
    d=(x_test[1]-x_test[0])*(x_test[2]-x_test[0])*(x_test[2]-x_test[1])
    C=(x_test[2]-x_test[1])*y_test[0]/d + (x_test[0]-x_test[2])*y_test[1]/d +(x_test[1]-x_test[0])*y_test[2]/d
    
    #error by standard deviation
  
    return x_test[y_test.index(min(y_test))],x_test1,C

#%%
    
"""
Need to fix the error finding. Use the stdev method discussed.
"""
theta_min,theta_test3,curvature=parabolic_min(theta_test,NLL_y,plot=True)
print("min:",theta_min)

print("curvature:",curvature)

#%%

def univariate(theta,delta_m):
    theta_min=theta[20]
    delta_m_min=delta_m[50]
    old_theta_min=0
    old_delta_m_min=0
    theta_1=[theta_min]
    delta_m_1=[delta_m_min]
    while abs(old_theta_min-theta_min)>0.0000001 and abs(old_delta_m_min-delta_m_min)>0.0000001:
    #for k in range(10):
        old_theta_min=theta_min
        old_delta_m_min=delta_m_min
        NLL_rate=[]
        for i in range(len(theta_test)):
            NLL_rate.append(unOssFlux*survival_prob(E,theta_test[i],delta_m_min,L))
            
        NLL_y=[]
        for i in range(len(NLL_rate)):
            NLL_y.append(NLL(NLL_rate[i],data))
        
        theta_min,theta_test3,theta_min_curvature=parabolic_min(theta_test,NLL_y,plot=False)
        theta_1.append(theta_min)
        delta_m_1.append(delta_m_min)
        
        NLL_rate_1=[]
        for i in range(len(D_m_test)):
            NLL_rate_1.append(unOssFlux*survival_prob(E,theta_min,D_m_test[i],L))
            
        NLL_y_1=[]
        for i in range(len(NLL_rate)):
            NLL_y_1.append(NLL(NLL_rate_1[i],data))
        
        delta_m_min,delta_m_test3,delta_m_curvature=parabolic_min(D_m_test,NLL_y_1,plot=False)
        theta_1.append(theta_min)
        delta_m_1.append(delta_m_min)
        
        
    return theta_1,delta_m_1


        
theta_min,delta_m_min=univariate(theta_test,D_m_test)
print(theta_min[-1],delta_m_min[-1])

#%%
theta_1,D_m_1=np.meshgrid(theta_test,D_m_test)

def survival_prob_new(E,theta_1,D_m_1,L):
    #E array of 200
    #theta,D_m array of 200
    #function should return [200*200,200] shape
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
surf=plt.pcolormesh(theta_1,D_m_1, Z, cmap=cm.coolwarm) 
fig.colorbar(surf, shrink=0.5, aspect=5)
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

