"""
Things to do:
    -create an animation of the bar graph with different parameters of theata and delta m
    to figure out a good approx. of the parameters.
"""
import numpy as np
import matplotlib.pyplot as plt

#inital values
theta=np.pi/4
D_m=2.4e-3
L=295

data,unOssFlux=np.loadtxt("data_mika.csv",delimiter=',',unpack=True)

def survival_prob(E,theta,D_m,L):#survival probability
    return 1-(np.sin(2*theta)**2)*(np.sin(1.267*(D_m)*L/E))**2

E=np.linspace(0,10,num=200,endpoint=True)+0.025#in the region of interest.
E_data=np.linspace(0,10,num=200,endpoint=True)+0.025
#%%
D_m_test=np.linspace(2.4e-3 -0.5*2.4e-3,2.4e-3 +0.5*2.4e-3,num=5,endpoint=False)
theta_test=np.linspace(np.pi/4-np.pi/4,np.pi/4,num=500,endpoint=False)

#lineType=['-','--','-.',':']
names_theta=[]
for i in range(len(theta_test)):
    names_theta.append(str(theta_test[i]))
names_D_m=[]
for i in range(len(D_m_test)):
    names_D_m.append(str(D_m_test[i]))
#%%
"""
plt.figure(1)#survival probability plot
for i in range(len(D_m_test)-1):
    plt.plot(E,survival_prob(E,theta,D_m_test[i+1],L),'-',label=names_D_m[i])
plt.legend()
plt.grid()
plt.show()

plt.figure(2)#survival probability plot
for i in range(len(theta_test)-1):
    plt.plot(E,survival_prob(E,theta_test[i+1],D_m,L),'-',label=names_theta[i])
plt.legend()
plt.grid()
plt.show()
"""
#%%
"""
Note that the data for a histogram (data is binned), therefore the line graph is plotted with
energy for each point representing the middle of the bin. E.g. first point E=0.025
"""
plt.figure(3)#plot of Unoscillated flux
plt.bar(x=E_data,height=unOssFlux,width=0.05,color='blue',label="Unoscillated flux")
plt.bar(x=E_data,height=unOssFlux*survival_prob(E_data,theta,D_m,L),width=0.05,color='red',label="Unoscillated flux with Survival Prob applied")
plt.legend()
plt.xlabel("energy (GeV)")
plt.ylabel("No. of occurrences")
plt.grid()
plt.show()

#%% c
"""
#plt.figure(4)#plot of data
#plt.bar(x=E_data,height=unOssFlux*survival_prob(E_data,theta,D_m,L),width=0.05,color='red',label="Unoscillated flux with Survival Prob applied")
#plt.bar(x=E_data,height=unOssFlux*survival_prob(E_data,theta,D_m,L),width=0.05,color='red',label="Unoscillated flux with Survival Prob applied")
for i in range(len(theta_test)-1):
    plt.figure(i+5)
    plt.bar(x=E_data,height=unOssFlux*survival_prob(E,theta_test[i+1],D_m,L),color='red',width=0.05,label=names_theta[i])
    plt.bar(x=E_data,height=data,width=0.05,color='blue',alpha=0.7,label="Data")
    plt.xlabel("energy (GeV)")
    plt.ylabel("No. of occurrences")
    plt.legend()
    plt.grid()
    plt.show()
"""
#%% To test effect of D_m changing
"""
for i in range(len(D_m_test)-1):
    plt.figure(i+5)
    plt.bar(x=E_data,height=unOssFlux*survival_prob(E,theta,D_m_test[i],L),color='red',width=0.05,label=names_D_m[i])
    plt.bar(x=E_data,height=data,width=0.05,color='blue',alpha=0.7,label="Data")
    plt.xlabel("energy (GeV)")
    plt.ylabel("No. of occurrences")
    plt.legend()
    plt.grid()
    plt.show()
"""
#%% NLL as a function of theta
    
def NLL(R,O):
    sum0=0
    for i in range(len(R)):
        if O[i]==0:
            sum0+=R[i]-O[i]
        else:
            sum0+=R[i]-O[i]+O[i]*np.log(O[i]/R[i])
        """
        0.05*R[i]=number of events
        R[i]=event rate
        """
    return sum0

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
    i1=300+np.random.randint(-10,10)
    i2=400+np.random.randint(-10,10)
    i3=490+np.random.randint(-10,10)
    x_test=[x[i1],x[i2],x[i3]]
    y_test=[y[i1],y[i2],y[i3]]

    current_min=1000

    while min(y_test)!=current_min:
        x_test1=x_test
        if plot!=False:
            plt.plot(x_test[y_test.index(min(y_test))],min(y_test),'ob')
            plt.show()
        current_min=min(y_test)
        x3=0.5*((x_test[2]**2 -x_test[1]**2)*y_test[0] +(x_test[0]**2 -x_test[2]**2)*y_test[1] +(x_test[1]**2 -x_test[0]**2)*y_test[2])/((x_test[2]-x_test[1])*y_test[0] + (x_test[0]-x_test[2])*y_test[1]+ (x_test[1]-x_test[0])*y_test[2])
        
        y3=(x3-x_test[1])*(x3-x_test[2])*y_test[0]/((x_test[0]-x_test[1])*(x_test[0]-x_test[2]))+ (x3-x_test[0])*(x3-x_test[2])*y_test[1]/((x_test[1]-x_test[0])*(x_test[1]-x_test[2]))+ (x3-x_test[0])*(x3-x_test[1])*y_test[2]/((x_test[2]-x_test[0])*(x_test[2]-x_test[1]))
        y_test.append(y3)
        x_test.append(x3)

        x_test.remove(x_test[y_test.index(max(y_test))])
        y_test.remove(max(y_test))
        
  
    return x_test[y_test.index(min(y_test))],x_test1

#%%
theta_min,theta_test3=parabolic_min(theta_test,NLL_y)
print("min:",theta_min)
theta_minp=0.5+theta_min
theta_minn=theta_min-0.5
print("min+:",theta_minp)
print("min-:",theta_minn)

#def parabolic_min_err(x_min,x_array):
    
        

