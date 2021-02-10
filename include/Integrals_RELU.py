import csv
import pandas as pd
import math
import scipy as sp
import scipy.special as spe
from scipy.linalg import sqrtm as sqrtm
import scipy.integrate as integrate
import numpy as np

import torch



########## Integrator that allows to compute using Montecarlo techniques the high dimensional integrals appearing in the ODEs

def Generate_MC_Set(model,K,NUM_GAUSSIANS,R,Q,T,num):
    Nsamples=num.Nsamples
    X=np.zeros((2*NUM_GAUSSIANS,Nsamples,K+2*NUM_GAUSSIANS))
    C=np.zeros((2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS))
    mu=np.zeros((2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS))
    for alpha in range(2*NUM_GAUSSIANS):
        C[alpha][:K,:K]=Q
        C[alpha][K:,:K]=R
        C[alpha][:K,K:]=R.T
        C[alpha][K:,K:]=T
        mu[alpha][:K]=R[alpha,:]
        mu[alpha][K:]=T[alpha,:]
        C[alpha]*=model.var
        X[alpha]=num.generate_samples(C=C[alpha],mu=mu[alpha])
    return X , C , mu


def Update_MC_Set_IID(model,K,NUM_GAUSSIANS,R,Q,T,num,X,C,mu):
    Nsamples=num.Nsamples
    for alpha in range(2*NUM_GAUSSIANS):
        C[alpha][:K,:K]=Q
        C[alpha][K:,:K]=R
        C[alpha][:K,K:]=R.T
        C[alpha][K:,K:]=T
        mu[alpha][:K]=R[alpha,:]
        mu[alpha][K:]=T[alpha,:]
        C[alpha]*=model.var
        X[alpha]=num.generate_samples(C=C[alpha],mu=mu[alpha])
        
def Update_MC_Set(K,NUM_GAUSSIANS,R,M,Q,T,MT,num,X,C,mu):
    ## allows not too copy the variables and hence to make faster
    Nsamples=num.Nsamples
    for alpha in range(2*NUM_GAUSSIANS):
        # for all distributions alpha the covariances are the same
        C[alpha][:K,:K]=Q
        C[alpha][K:,:K]=R 
        C[alpha][:K,K:]=R.T
        C[alpha][K:,K:]=T
        # the means change depending on the distribution
        mu[alpha][:K]=M[alpha,:]
        mu[alpha][K:]=MT[alpha,:]
        X[alpha]=num.generate_samples(C=C[alpha],mu=mu[alpha])
                        

def Update_MC_Set_Linear(K,NUM_GAUSSIANS,R,M,Q,T,MT,num,X,C,mu):
    ## allows not too copy the variables and hence to make faster
    Nsamples=num.Nsamples
    for alpha in range(2*NUM_GAUSSIANS):
        # for all distributions alpha the covariances are the same
        C[alpha][:K,:K]=Q
        C[alpha][K:,:K]=R 
        C[alpha][:K,K:]=R.T
        C[alpha][K:,K:]=T
        # the means change depending on the distribution
        mu[alpha][:K]=M[alpha,:]
        mu[alpha][K:]=MT[alpha,:]
        X[alpha]={"C":C[alpha],"mu":mu[alpha]}
        
#######without the analytical integrals######
def erf_np(x):
    return spe.erf(x/math.sqrt(2))
def relu_np(x):
    return x * (x > 0)
def lin_np(x):
    return x

def Drelu(x):
    return np.int_(x>0)
def Dlin(x):
    return np.ones(x.shape)
def Derf(x):
    return math.sqrt(2./math.pi)*np.exp(-x**2/2.)


class Integrals():
    def __init__(self,gname="erf"):
        self.gname=gname
        if gname=="erf":
            self.g=erf_np
            self.Dg=Derf
        if gname=="relu":
            self.g=relu_np
            self.Dg=Drelu
        if gname=="lin":
            self.g=lin_np
            self.Dg=Dlin
        print("creating an integrator with g %s"%gname)    
    def I2(self,X,i1,i2):
        print("Super Class--> no object initialised")
    
    def I3(self,X,i1,i2,i3):
        print("Super Class--> no object initialised")
    
    def I4(self,X,i1,i2,i3,i4):
        print("Super Class--> no object initialised")

        
############## 
class Integrals_LINEAR(Integrals):
    """Gaussian Mixture Integrals for the case in which the activation function is linear
       Inputs: - X : [C,mu] for analytical and set for MC for Numerical where C is the covariance and mu the mean
               - (i1,...) the variables one is integrating over 
       Outputs: Value of the integral given by the paper . """
    def __init__(self,Nsamples=1000,gname="lin"):
        self.Nsamples=Nsamples
        print("Creating a LINEAR INTEGRATOR")
        super().__init__(gname="lin")
    def generate_samples(self,C,mu):
        """Generates Nsamples from distribution of covariance C and mean mu"""
        print("THIS IS LINEAR ACTIVATION FUNCTION! YOU SHOULD NOT HAVE TO GENERATE SAMPLES!")
        dim=len(C)
        #X=np.zeros((self.Nsamples,dim))
        m = torch.distributions.multivariate_normal.MultivariateNormal(mu, C)
        X = m.sample(self.Nsamples)
        #X[:]=mu
        #vec=np.random.normal(size=(self.Nsamples,dim))
        #F=np.real(sqrtm(C))
        ## [Nsamples Dim]
        #X+=vec@F
        return X
    def I1(self,X,i1):
        """<g(i1)>"""
        mmu=X["mu"]
        return mmu[i1]
    def I1_Ana(self,X,i1):
        """<g(i1)>"""
        return self.I1(self,X,i1)
    def I3(self,X,i1,i2,i3):
        """<g'(i1) i2 g(i3)>"""
        CC=X["C"]
        mmu=X["mu"]
        C=CC[np.array([i1,i2,i3])]
        C=(C.T[np.array([i1,i2,i3])].T)
        mu=mmu[np.array([i1,i2,i3])]
        return C[1,2]+mu[2]*mu[1]
    
    def I3_2(self,X,i1,i2):
        """<g'(i1) i2>"""
        mmu=X["mu"]
        return mmu[i2]
    def I2(self,X,i1,i2):
        res=0
        """0.5*<g(i1) g(i2)>"""
        CC=X["C"]
        mmu=X["mu"]
        C=CC[np.array([i1,i2])]
        C=(C.T[np.array([i1,i2])].T)
        mu=mmu[np.array([i1,i2])]
        return 0.5*(C[0,1]+mu[0]*mu[1])
    def I4(self,X,i1,i2,i3,i4):
        """<g'(i1) g'(i2) g(i3) g(i4)>"""
        CC=X["C"]
        mmu=X["mu"]
        C=CC[np.array([i1,i2,i3,i4])]
        C=(C.T[np.array([i1,i2,i3,i4])].T)
        mu=mmu[np.array([i1,i2,i3,i4])]
        return C[2,3]+mu[2]*mu[3]

    def I22(self,X,i1,i2):
        """<g'(i1) g(i2)>"""
        mmu=X["mu"]
        return mmu[i2]
    def I3_1(self,X,i1):
        """<g'(i1)>"""
        return 1
    def I4_2(self,X,i1,i2):
        """<g'(i1)g'(i2)>"""
        return 1
    def I4_3(self,X,i1,i2,i3):
        """<g'(i1)g'(i2)g(i3)>"""
        mmu=X["mu"]
        return mmu[i3]


############## 
class Integrals_GM(Integrals):
    """Gaussian Mixture Integrals
       Inputs: - X : [C,mu] for analytical and set for MC for Numerical where C is the covariance and mu the mean
               - (i1,...) the variables one is integrating over 
       Outputs: Value of the integral given by the paper . """
    def __init__(self,Nsamples=10000,gname="erf",dim=800):
        self.Nsamples=Nsamples
        self.dim=dim
        self.vec = np.random.normal(size=(self.Nsamples,dim))
        super().__init__(gname=gname)
        
    def generate_samples(self,C,mu):
        """Generates Nsamples from distribution of covariance C and mean mu"""
        dim=len(C)
        X=np.zeros((self.Nsamples,dim))
        X[:]=mu
        F=np.real(sqrtm(C))
        # [Nsamples Dim]
        X+=self.vec@F
        return X
    
    def debug(self,X,i1,i2):
        """In order to Debug, this integral should give C[i1,i2]: <i1 i2>+<i1><i2>"""
        # X=[num samples , N] has colums such that x_i x_j =C_ij
        transformed=np.asarray([X.T[i1],X.T[i2]])
        func=transformed[0]*transformed[1]
        return np.mean(func)
    
    def I22(self,X,i1,i2):
        """<g'(i1) g(i2)>"""
        transformed=np.asarray([X.T[i1],X.T[i2]])
        transformed[0]=Derf(transformed[0])
        transformed[1]=erf_np(transformed[1])
        func=transformed[0]*transformed[1]
        return np.mean(func)
    def I3_1(self,X,i1):
        """<g'(i1)>"""
        transformed=X.T[i1]
        transformed=Derf(transformed)
        func=transformed
        return np.mean(func)
    
    #######################################
    def I1(self,X,i1):
        """<g(i1)>"""
        # X=[num samples , N] has colums such that x_i x_j =C_ij
        transformed=np.asarray([X.T[i1]])
        transformed[0]=self.g(transformed[0])
        func=transformed[0]
        return np.mean(func)
    def I2(self,X,i1,i2):
        """0.5*<g(i1) g(i2)>"""
        # X=[num samples , N] has colums such that x_i x_j =C_ij
        transformed=np.asarray([X.T[i1],X.T[i2]])
        transformed[0]=self.g(transformed[0])
        transformed[1]=self.g(transformed[1])
        func=transformed[0]*transformed[1]
        return 0.5*np.mean(func)
    def I3(self,X,i1,i2,i3):
        """<g'(i1) i2 g(i3)>"""
        transformed=np.asarray([X.T[i1],X.T[i2],X.T[i3]])
        transformed[0]=self.Dg(transformed[0])
        transformed[2]=self.g(transformed[2])
        func=transformed[0]*transformed[1]*transformed[2]
        return np.mean(func)
    def I4(self,X,i1,i2,i3,i4):
        """<g'(i1) g'(i2) g(i3) g(i4)>"""
        transformed=np.asarray([X.T[i1],X.T[i2],X.T[i3],X.T[i4]])
        transformed[0]=self.Dg(transformed[0])
        transformed[1]=self.Dg(transformed[1])
        transformed[2]=self.g(transformed[2])
        transformed[3]=self.g(transformed[3])
        func=transformed[0]*transformed[1]*transformed[2]*transformed[3]
        return np.mean(func)
    def I3_2(self,X,i1,i2):
        """<g'(i1) i2>"""
        transformed=np.asarray([X.T[i1],X.T[i2]])
        transformed[0]=self.Dg(transformed[0])
        func=transformed[0]*transformed[1]
        return np.mean(func)
    def I4_2(self,X,i1,i2):
        """<g'(i1)g'(i2)>"""
        transformed=np.asarray([X.T[i1],X.T[i2]])
        transformed[0]=self.Dg(transformed[0])
        transformed[1]=self.Dg(transformed[1])
        func=transformed[0]*transformed[1]
        return np.mean(func)
    def I4_3(self,X,i1,i2,i3):
        """<g'(i1)g'(i2)g(i3)>"""
        transformed=np.asarray([X.T[i1],X.T[i2],X.T[i3]])
        transformed[0]=self.Dg(transformed[0])
        transformed[1]=self.Dg(transformed[1])
        transformed[2]=self.g(transformed[2])
        func=transformed[0]*transformed[1]*transformed[2]
        return np.mean(func)
    ##########################
    def I1_Ana(self,X,i1):
        """<g(i1)>"""
        CC=X[0]
        mmu=X[1]
        C=CC[np.array([i1])]
        C=(C.T[np.array([i1])].T)
        mu=mmu[np.array([i1])]
        n=len(mu)
        if self.gname=="erf":
            res=math.sqrt(1+C[0,0])
            res=mu[0]/res
            res=erf_np(res)
            return res
        if self.gname=="relu":
            res=1+erf_np(mu/math.sqrt(C[0,0]))
            res*=mu[0]*0.5
            res+=math.sqrt(C[0,0]/(2*math.pi))*np.exp(-0.5/C[0,0]*mu[0]**2)
            res=res[0]
            return res
        if self.gname=="lin":
            return mu[0]
    def I3_2_Ana(self,X,i1,i2):
        """<g'(i1) i2>"""
        res=0
        if i1==i2 and self.gname=="erf":
            return 0
        CC=X[0]
        mmu=X[1]
        C=CC[np.array([i1,i2])]
        C=(C.T[np.array([i1,i2])].T)
        #print("C within I3_2_Ana")
        #print(C)
        mu=mmu[np.array([i1,i2])]
        if i1==i2 and self.gname=="relu":
            res=mu[0]*0.25*(1+erf_np(mu[0]/math.sqrt(C[0,0])))
            res+=0.25*math.sqrt(2*C[0,0]/math.pi)
            return res
        n=len(mu)
        if self.gname=="erf":
            G=np.linalg.inv(C)+np.asarray([[1,0],[0,0]])
            exp1=-1./2.*mu.T@np.linalg.inv(C)@mu
            exp2=1./2*mu.T@np.linalg.inv(C)@np.linalg.inv(G)@np.linalg.inv(C)@mu
            prevec=np.asarray([0,1])@np.linalg.inv(G)@np.linalg.inv(C)@mu
            pre=1./math.sqrt(np.linalg.det(G)*np.linalg.det(C))
            pre*=math.sqrt(2./math.pi)
            res=pre*prevec*np.exp(exp1+exp2)
        if self.gname=="relu":
            res=C[0,1]/math.sqrt(2*math.pi*C[0,0])
        if self.gname=="lin":
            return mu[1]
        return res
    


def err_ana(R,Q,v,lr,num,label,X,C,mu,quiet=True):
    """Computes the analytical pmse given by the order parameters
    """
    NUM_GAUSSIANS=int(len(R)/2)
    K=len(v)
    err=0.5
    I2=[[ [num.I2(X[alpha],k,l) for l in range(K)] for k in range(K)] for alpha in range(2*NUM_GAUSSIANS)]
    I2=np.asarray(I2)
    derr=np.prod(label.T[[1,2]].T,axis=1).T@(v.T@I2@v);
    err+=derr
    I1=np.asarray([[num.I1(X[alpha],k) for k in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
    derr=-np.prod(label,axis=1).T@(I1@v);
    err+=derr
    return err


def err_ana_LINEAR(R,Q,v,lr,num,label,C,mu,quiet=True):
    """Computes the analytical pmse given by the order parameters in the simplified case in which the activation function is linear
    """
    NUM_GAUSSIANS=int(len(R)/2)
    K=len(v)
    err=0.5
    I2=[[ [num.I2_Ana([C[alpha],mu[alpha]],k,l) for l in range(K)] for k in range(K)] for alpha in range(2*NUM_GAUSSIANS)]
    I2=np.asarray(I2)
    derr=np.prod(label.T[[1,2]].T,axis=1).T@(v.T@I2@v);
    err+=derr
    I1=np.asarray([[num.I1_Ana([C[alpha],mu[alpha]],k) for k in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
    derr=-np.prod(label,axis=1).T@(I1@v);
    err+=derr
    return err

def classification_analytical(X,v,label):
    """computes the classification error as E[ Theta( y* y)  ] where the expectation is performed using MC samples X that have mean [M T0] amd cpvariance [[ Q, R],[R, T]]"""
    K = v.shape[0]
    g=np.maximum(X[:,:,:K],0 )## selects only the lambdas K
    out = g@v
    for l,o in zip(label,out): o*=l[0] ## multiplies by y* an 1/2
    out = np.float_(out<0)
    for l,o in zip(label,out): o*=l[1]*l[2] 
    out = out.mean(axis=1)
    out = np.sum(out )
    return out