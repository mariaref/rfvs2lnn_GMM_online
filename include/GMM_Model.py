import csv
import math
import scipy as sp
import scipy.special as spe
import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import relu as relu
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as tdist
import numpy as np
device = torch.device("cpu")



def get_means(N=4,NUM_GAUSSIANS=2,length=1.):
    ## mus (2 NUM_GAUSSIANS N)
    mus=torch.zeros((2,NUM_GAUSSIANS,N))
    #For + label
    #angles between the means
    alphas=2.*math.pi/NUM_GAUSSIANS*(torch.arange(NUM_GAUSSIANS).float())
    coord=torch.stack((torch.cos(alphas),torch.sin(alphas)),dim=-1 )
    shape = np.shape(coord)
    mus[0][:shape[0],:shape[1]] = coord
    #For - label
    alphas=2.*math.pi/NUM_GAUSSIANS*(torch.arange(NUM_GAUSSIANS).float()+0.5*torch.ones(NUM_GAUSSIANS).float())
    coord=torch.stack( (torch.cos(alphas),torch.sin(alphas)),dim=-1 )
    shape = coord.shape

    mus[1][:shape[0],:shape[1]] = coord
    
    #FOR NORMaLISATION
    mus*=length
    mus*=math.sqrt(N)
    return mus


######SUPER CLASS MODEL##################
class Model():
    def __init__(self,N=500,NUM_GAUSSIANS=2,rho=0.5,mus=None,length=1.,rand_MEANS=False):
        print("SuperClass initialisation")
        self.N=N
        self.rho=rho
        self.NUM_GAUSSIANS=NUM_GAUSSIANS       
        if mus is None:
            if rand_MEANS:
                self.mus=torch.zeros(2,NUM_GAUSSIANS,N)
                self.mus[0]=torch.randn(NUM_GAUSSIANS,N)
                self.mus[1]=-self.mus[0]
            else:
                self.mus=get_means(N=N,NUM_GAUSSIANS=NUM_GAUSSIANS,length=length)
                print(torch.norm(self.mus.view(2*NUM_GAUSSIANS,N),dim=1))
        else:
            print("Taking input means")
            self.mus=mus
        if list(self.mus.shape)==[2,NUM_GAUSSIANS,N]:
            print("The shape of the model means is [2,NUM_GAUSSIANS,N]=[%d,%d,%d]"%(self.mus.shape[0],self.mus.shape[1],self.mus.shape[2]))
            print(torch.norm(self.mus.reshape(2*NUM_GAUSSIANS,N),dim=1))
        else: print("PROBLEMS! THE MEANS ARE THE WRONG DIMENSIONS")
            
    def plot_means(self,axs=None):
        if axs is None:
            fig,axs=plt.subplots(1,1,figsize=(5,5))
        widths = np.linspace(0, 2, 2*self.NUM_GAUSSIANS)
        for mm in range(self.NUM_GAUSSIANS):
            axs.plot([0,self.mus[0][mm][0]/float(math.sqrt(self.N))],[0,self.mus[0][mm][1]/float(math.sqrt(self.N))],color='r')
            axs.plot([0,self.mus[1][mm][0]/float(math.sqrt(self.N))],[0,self.mus[1][mm][1]/float(math.sqrt(self.N))],color='b')
        axs.grid()    

class Model_iid(Model):
    """simplified scenario in which the gaussians have identity covariance
    """
    def __init__(self,N=10,var=0.1,rho=0.5,NUM_GAUSSIANS=2,mus=None,length=1.,rand_MEANS=False):
        print("Creating a model with N: %d, var: %g , rho: %g, NUM_GAUSSIANS: %d"%(N,var,rho,NUM_GAUSSIANS) )
        self.var=var
        super().__init__(N=N,NUM_GAUSSIANS=NUM_GAUSSIANS,rho=rho,mus=mus,length=length,rand_MEANS=rand_MEANS)
        
    
    def get_X_from_label(self,NUM_SAMPLES,l,ran=None):
        ind=l.view(NUM_SAMPLES)
        ind=(ind+1)*0.5 ## sets l to either 0 or 1
        if ran is None: ran=torch.randint(self.NUM_GAUSSIANS,size=[NUM_SAMPLES]) ## which Gaussian clusters to select
        X=torch.zeros((NUM_SAMPLES,self.N))
        for i in range(NUM_SAMPLES):
            a=int(ind[i])
            b=int(ran[i])
            X[i]=self.mus[a][b]/math.sqrt(self.N)
        X+=math.sqrt(self.var)*torch.randn(NUM_SAMPLES,self.N)
        return X
        
    def get_test_set(self,NUM_SAMPLES=1, l=None,ran=None):
        if l is None:
            l=torch.randint(2,size=[NUM_SAMPLES,1]) ##note cannot require gradient on integer valued tensor
            l=2*l-1
        if not list(l.shape)==[NUM_SAMPLES,1]: print("There is a problem, the shape of the labels is not NUM_SAMPLES=%d"%NUM_SAMPLES)
        X=self.get_X_from_label(NUM_SAMPLES=NUM_SAMPLES,l=l,ran=ran)
        # X is dim [NUM_SAMPLES, N]
        # l is dim [NUM_SAMPLES]
        return l,X
    
class Model_AC(Model):
    """Gaussian mixture with arbitrary covariance within each cluster"""
    def __init__(self,N=784,rho=0.5,NUM_GAUSSIANS=5,mus=None,length=1.,rand_MEANS=False,Fs=None,rotate=False):
        print("Creating a model with N: %d, rho: %g, NUM_GAUSSIANS: %d"%(N,rho,NUM_GAUSSIANS) )
        if Fs is None:
            self.Fs=torch.zeros(2,NUM_GAUSSIANS,N,N)
            print("Have been given no initial covariance")
            #for b in range(2):
            #    for a in range(NUM_GAUSSIANS):
            #        Fs[a][b]=0.1*torch.eye(N)
            Fs=0.1*torch.eye(N)
        else: 
            print("Taking input Fs")
            print("Max value of the Fs is %g"%torch.max(Fs))
        self.Fs=Fs
        Covs=self.Fs.t()@self.Fs+1e-5*torch.eye(N)
        Masses,Psis=torch.symeig(Covs, eigenvectors=True)
        Psis*=math.sqrt(N)
        self.Covs=Covs
        self.Masses=Masses
        ## A rotated vector into the eigenbasis is written vector@eigenvectors
        self.Psis=Psis
        
        if self.Covs.shape==(N,N):
            print("The shape of the model covariance is [N,N]=[%d,%d]"%(self.Fs.shape[0],self.Fs.shape[1]))
        else: print("PROBLEMS! THE Covariances ARE THE WRONG DIMENSIONS")  
        super().__init__(N=N,NUM_GAUSSIANS=NUM_GAUSSIANS,mus=mus,rho=rho,length=length,rand_MEANS=rand_MEANS)
    
    def get_X_from_label(self,NUM_SAMPLES,l,ran=None):
        ind=l.view(NUM_SAMPLES)
        #print(ind.shape)
        #ind= ## sets l to either 0 or 1
        if ran is None: ran=torch.randint(self.NUM_GAUSSIANS,size=[NUM_SAMPLES]) ## which Gaussian clusters to select
        X=torch.zeros((NUM_SAMPLES,self.N))
        for i in range(NUM_SAMPLES):
            a=int((ind[i]+1.)/2.)
            b=int(ran[i])
            X[i]=self.mus[a][b]/math.sqrt(self.N)
        X+=torch.randn(NUM_SAMPLES,self.N)@self.Fs
        return X
        
    def get_test_set(self,NUM_SAMPLES=1, l=None,ran=None):
        if l is None:
            l=torch.randint(2,size=[NUM_SAMPLES,1]) ##note cannot require gradient on integer valued tensor
            l=2*l-1
        if not list(l.shape)==[NUM_SAMPLES,1]: print("There is a problem, the shape of the labels is not NUM_SAMPLES=%d"%NUM_SAMPLES)
        X=self.get_X_from_label(NUM_SAMPLES=NUM_SAMPLES,l=l,ran=ran)
        # X is dim [NUM_SAMPLES, N]
        # l is dim [NUM_SAMPLES]
        return l,X
    
class Model_AC_MULTIPLE(Model):
    def __init__(self,N=784,rho=0.5,NUM_GAUSSIANS=5,mus=None,length=1.,rand_MEANS=False,Fs=None,rotate=False):
        print("Creating a model with N: %d, rho: %g, NUM_GAUSSIANS: %d"%(N,rho,NUM_GAUSSIANS) )
        if Fs is None:
            self.Fs=torch.zeros(2,NUM_GAUSSIANS,N,N)
            print("Have been given no initial covariance")
            for b in range(2):
                for a in range(NUM_GAUSSIANS):
                    Fs[a][b]=0.1*torch.eye(N)
        else: 
            print("Taking input Fs")
            print("Max value of the Fs is %g"%torch.max(Fs))
        self.Fs=Fs
        
        Covs=torch.zeros(2,NUM_GAUSSIANS,N,N)
        Psis=torch.zeros(2,NUM_GAUSSIANS,N,N)
        Masses=torch.zeros(2,NUM_GAUSSIANS,N)
        for b in range(2):
            for a in range(NUM_GAUSSIANS):
                Covs[b][a]=self.Fs[b][a].t()@self.Fs[b][a]+1e-5*torch.eye(N)
                Masses[b][a],Psis[b][a]=torch.symeig(Covs[b][a], eigenvectors=True)
                Psis[b][a]*=math.sqrt(N)
        self.Covs=Covs
        self.Masses=Masses
        ## A rotated vector into the eigenbasis is written vector@eigenvectors
        self.Psis=Psis
        
        if self.Covs.shape==(2,NUM_GAUSSIANS,N,N):
            print("The shape of the model covariance is [2,NG,N,N]=[%d,%d,%d,%d]"%(self.Fs.shape[0],self.Fs.shape[1],self.Fs.shape[2],self.Fs.shape[3]))
        else: print("PROBLEMS! THE Covariances ARE THE WRONG DIMENSIONS")  
        super().__init__(N=N,NUM_GAUSSIANS=NUM_GAUSSIANS,mus=mus,rho=rho,length=length,rand_MEANS=rand_MEANS)
    
    def get_X_from_label(self,NUM_SAMPLES,l,ran=None):
        ind=l.view(NUM_SAMPLES)
        #print(ind.shape)
        #ind= ## sets l to either 0 or 1
        if ran is None: ran=torch.randint(self.NUM_GAUSSIANS,size=[NUM_SAMPLES]) ## which Gaussian clusters to select
        X=torch.zeros((NUM_SAMPLES,self.N))
        for i in range(NUM_SAMPLES):
            a=int((ind[i]+1.)/2.)
            b=int(ran[i])
            X[i]=self.mus[a][b]/math.sqrt(self.N)+torch.randn(1,self.N)@self.Fs[a][b]
        return X
        
    def get_test_set(self,NUM_SAMPLES=1, l=None,ran=None):
        if l is None:
            l=torch.randint(2,size=[NUM_SAMPLES,1]) ##note cannot require gradient on integer valued tensor
            l=2*l-1
        if not list(l.shape)==[NUM_SAMPLES,1]: print("There is a problem, the shape of the labels is not NUM_SAMPLES=%d"%NUM_SAMPLES)
        X=self.get_X_from_label(NUM_SAMPLES=NUM_SAMPLES,l=l,ran=ran)
        # X is dim [NUM_SAMPLES, N]
        # l is dim [NUM_SAMPLES]
        return l,X
    
####Student Creations ########

def erf(x):
    return torch.erf(x/math.sqrt(2))
def linear(x):
    return x
def centered_relu(x,var):
    a=math.sqrt(var)/math.sqrt(2*math.pi)
    return torch.relu(x)-a

def centered_erf(x,var):
    return torch.erf(x/math.sqrt(2))
def centered_linear(x,var):
    return x

##For an Arbitrary Covariance Model######
def get_RQT1T0M(model,student,quiet=True):
    """Returns the order parameters for a given model:
    - R: (2*NUM_GAUSSIANS,K)
    - Q: (K,K)"""
    K=student.K
    N=model.N
    w=student.fc1.weight.data
    if not quiet:
        print("Computing the order parameters: Student of %d nodes, %d Gaussians"%(K,2*len(model.mus.numpy()[0])))
    
    wtau=w@model.Psis*1./math.sqrt(N) ## same for all distributions  K x N
    mustau=model.mus.reshape((2*model.NUM_GAUSSIANS,N))@model.Psis*1./math.sqrt(N)
    
    r=np.asarray([[(wtau[k]*mustau[alpha]).numpy() for k in range(K)] for alpha in range(2*model.NUM_GAUSSIANS)])
    
    q=np.asarray([[(wtau[k]*wtau[l]).numpy() for k in range(K)] for l in range(K)])
    
    t=np.asarray([[(mustau[beta]*mustau[alpha]).numpy() for alpha in range(2*model.NUM_GAUSSIANS)] for beta in range(2*model.NUM_GAUSSIANS)])
    
    Q=1/float(N)*(wtau*model.Masses)@wtau.t()
    R=1/float(N)*(mustau*model.Masses)@wtau.t()
    T=1/float(N)*(mustau*model.Masses)@mustau.t()
    M=1/float(N)*(mustau)@wtau.t()
    MT=1/float(N)*(mustau)@mustau.t()
    Rn=R.numpy()
    Qn=Q.numpy()
    Tn=T.numpy()
    MTn=MT.numpy()
    Mn=M.numpy()
    return Rn,Qn,Tn,MTn,Mn,r,q,t


####For an iid. model ########
def get_RQT(model,student,quiet=True):
    """Returns the order parameters for a given model:
    - R: (2*NUM_GAUSSIANS,K)
    - Q: (K,K)"""
    K=student.K
    N=model.N
    w=student.fc1.weight.data
    if not quiet:
        print("Computing the order parameters: Student of %d nodes, %d Gaussians"%(K,2*len(model.mus.numpy()[0])))
    Q=1/float(N)*w@w.t()
    mus=model.mus.reshape(2*model.NUM_GAUSSIANS,N).detach().clone()
    R=1/float(N)*mus@w.t()
    T=1/float(N)*mus@mus.t()
    Rn=R.numpy()
    Qn=Q.numpy()
    Tn=T.numpy()
    return Rn,Qn,Tn


#Create a Student Network of K Nodes
class Student(nn.Module):
    def __init__(self,K,N,act_function="erf",bias=False,w=None,v=None):
        """ initialisation of a student with:
        - K hidden nodes
        - N input dimensions
        -activation function act_function
        -initial weight: can be given or not """
        print("Creating a Student with N: %d, K: %d"%(N,K) )
        super(Student, self).__init__()
        
        self.N=N
        self.bias=bias
        self.gname=act_function
        if act_function=="erf":
            self.g=erf
        if act_function=="relu":
            self.g=relu
        if act_function=="lin":
            self.g=linear
        self.K=K
        #self.loss=HalfMSELoss(reduction='mean')
        self.loss=nn.MSELoss(reduction='mean')
        #change bias to true
        self.fc1 = nn.Linear(N, K, bias=bias)
        
        self.fc2 = nn.Linear(K, 1, bias=False)
        
        
        #normalise to small initial weights
        #in c++ did not initialise small
        nn.init.normal_(self.fc1.weight)
        nn.init.normal_(self.fc2.weight)
        #self.w=self.fc1.weight.data
        if w is None:
            self.w=self.fc1.weight.data
        else :
            self.w=w
            self.fc1.weight.data=w
            
        if v is None:
            self.v=self.fc2.weight.data
        else :
            self.v=v
            self.fc2.weight.data=v
            
    def get_lambda(self,X):
        """Returns lambda for a test set of size (K,NUM_SAMPLES)"""
        return 1./math.sqrt(self.N)*X@self.w.T
    
    def get_output(self,x):
        """Returns the output of the student for a given sample x:
        -x : sample (N)"""
        x = self.g(self.fc1(x) / math.sqrt(self.N))
        x = self.fc2(x)
            
            
        return x
    
    def forward(self, x):
        # input to hidden
        x = self.g(self.fc1(x) / math.sqrt(self.N))
        x = self.fc2(x)
        return x
    
    def get_err(self,label,x):
        """ For a given label computes the output of the student.
        - label: the true label = +-1 (#test, )
        - x: the test sample (#test, N)"""
        out=self.get_output(x)
        err=0.5*self.loss(label,out)
        return err
