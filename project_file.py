import numpy as np
import matplotlib.pyplot as plt

#inital values
theta=np.pi/4
D_m=2.4e-3
L=295

data,unOssFlux=np.loadtxt("data.csv",delimiter=',',unpack=True)

def survival_prob(E,theta,D_m,L):#survival probability
    return 1-(np.sin(2*theta)**2)*(np.sin(1.267*(D_m)*L/E))**2

E=np.linspace(0,10,num=200,endpoint=True)+0.025#in the region of interest.
E_data=np.linspace(0,10,num=200,endpoint=True)+0.025
#%%
plt.figure(1)#survival probability plot
plt.plot(E,survival_prob(E,theta,D_m,L),'-r')
plt.grid()
plt.show()
#%%
"""
Note that the data for a histogram (data is binned), therefore the line graph is plotted with
energy for each point representing the middle of the bin. E.g. first point E=0.025
"""
plt.figure(2)#plot of Unoscillated flux
plt.bar(x=E_data,height=unOssFlux,width=0.05,color='blue',label="Unoscillated flux")
plt.bar(x=E_data,height=unOssFlux*survival_prob(E_data,theta,D_m,L),width=0.05,color='red',label="Unoscillated flux with Survival Prob applied")
plt.legend()
plt.xlabel("energy (GeV)")
plt.ylabel("No. of occurrences")
plt.grid()
plt.show()

#%%
plt.figure(3)#plot of data
plt.bar(x=E_data,height=unOssFlux*survival_prob(E_data,theta,D_m,L),width=0.05,color='red',label="Unoscillated flux with Survival Prob applied")
plt.bar(x=E_data,height=data,width=0.05,color='blue',alpha=0.7,label="Data")
plt.xlabel("energy (GeV)")
plt.ylabel("No. of occurrences")
plt.legend()
plt.grid()
plt.show()

#%%




