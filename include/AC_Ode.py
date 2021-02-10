############
## implementation of the equation for the densities r and m defined in the paper 

import sys
sys.path.insert(1, 'include')
from GMM_Model import *
from Integrals_RELU import *
from ToolBox import *
from time import time
from math import sqrt as sqrt
import argparse
import io
from os import path
eps=1e-6


def update_r(Masses,v,r,t,q,Q,R,M,T1,T0,lr,num,label,X,NUM_GAUSSIANS,reg,N):
    def dr_comp(k,beta,m,v,r,t,q,Q,R,M,T1,T0,lr,num,label,X,NUM_GAUSSIANS,reg,N):
        K=len(v)
        I31k=np.asarray( [num.I3_1(X[alpha],k) for alpha in range(2*NUM_GAUSSIANS)])
        I32kk=np.asarray([num.I3_2(X[alpha],k,k) for alpha in range(2*NUM_GAUSSIANS)])
        I3kkk=np.asarray( [num.I3(X[alpha],k,k,k) for alpha in range(2*NUM_GAUSSIANS)])
        I22kk=np.asarray([num.I22(X[alpha],k,k) for alpha in range(2*NUM_GAUSSIANS)])
        
        line1= np.asarray([I31k[alpha]*v[k]*lr*t[alpha,beta] for alpha in range(2*NUM_GAUSSIANS)])
        line1+= np.asarray([v[k]*lr*m/Q[k,k]*(I32kk[alpha]-I31k[alpha]*M[alpha,k])*r[beta,k] for alpha in range(2*NUM_GAUSSIANS)])
        
        line2=np.asarray([(-lr*v[k]*v[k])*t[alpha,beta]*I22kk[alpha] for alpha in range(2*NUM_GAUSSIANS)])
        line2+=np.asarray([(-lr*v[k]*v[k])*m/Q[k,k]*(I3kkk[alpha]-M[alpha,k]*I22kk[alpha])*r[beta,k] for alpha in range(2*NUM_GAUSSIANS)])
        for j in range(K):
            if j!=k:
                det=Q[k,k]*Q[j,j]-Q[k,j]**2
                if det<eps:
                    print("det is too small %g"%det)
                    det=eps
                I3kkj=np.asarray([num.I3(X[alpha],k,k,j) for alpha in range(2*NUM_GAUSSIANS)])
                I22kj=np.asarray([num.I22(X[alpha],k,j) for alpha in range(2*NUM_GAUSSIANS)])
                I3kjj=np.asarray([num.I3(X[alpha],k,j,j) for alpha in range(2*NUM_GAUSSIANS)])
                
                line2+=np.asarray([(-lr*v[k]*v[j])*I22kj[alpha]*t[alpha,beta] for alpha in range(2*NUM_GAUSSIANS)])
                line2+=np.asarray([ (-lr*v[k]*v[j])*m/det*(I3kkj[alpha]-I22kj[alpha]*M[alpha,k])*(Q[j,j]*r[beta,k]-Q[k,j]*r[beta,j]) for alpha in range(2*NUM_GAUSSIANS)])
                line2+=np.asarray([ (-lr*v[k]*v[j])*m/det*(I3kjj[alpha]-I22kj[alpha]*M[alpha,j])*(Q[k,k]*r[beta,j]-Q[k,j]*r[beta,k]) for alpha in range(2*NUM_GAUSSIANS)])
        res=line1.T@np.prod(label,axis=1)
        res+=line2.T@np.prod(label.T[[1,2]].T,axis=1)
        return res 
    
    K=len(v)
    dd=np.zeros(r.shape)
    for k in range(K):
        for beta in range(2*NUM_GAUSSIANS):
            dd[beta,k]+=dr_comp(k,beta,Masses,v,r,t,q,Q,R,M,T1,T0,lr,num,label,X,NUM_GAUSSIANS,reg,N)-reg*lr*r[beta,k]
    return dd

def update_q(Masses,v,r,t,q,Q,R,M,T1,T0,lr,num,label,X,NUM_GAUSSIANS,reg,N):
    #print("Updating q")
    def dq_comp(k,l,m,v,r,t,q,Q,R,M,T1,T0,lr,num,label,X,NUM_GAUSSIANS,reg,N):
        K=len(v)
        #print("Evolving component %d %d"%(k,l))
        
        ########## delta w^k w^l ######
        I31k=np.asarray( [num.I3_1(X[alpha],k) for alpha in range(2*NUM_GAUSSIANS)])
        I32kk=np.asarray([num.I3_2(X[alpha],k,k) for alpha in range(2*NUM_GAUSSIANS)])
        I3kkk=np.asarray( [num.I3(X[alpha],k,k,k) for alpha in range(2*NUM_GAUSSIANS)])
        I22kk=np.asarray([num.I22(X[alpha],k,k) for alpha in range(2*NUM_GAUSSIANS)])
        #C
        C = np.asarray([ I31k[alpha]*v[k]*lr*r[alpha,l]+lr*v[k]*m*q[k,l]/Q[k,k]*(I32kk[alpha]-I31k[alpha]*M[alpha,k]) for alpha in range(2*NUM_GAUSSIANS)])
        
        #B
        line2=-1*np.asarray([lr*v[k]*v[k]*r[alpha,l]*I22kk[alpha] for alpha in range(2*NUM_GAUSSIANS)])
        line2-=np.asarray([(lr*v[k]*v[k])*m/Q[k,k]*(I3kkk[alpha]-M[alpha,k]*I22kk[alpha])*q[k,l] for alpha in range(2*NUM_GAUSSIANS)])
        
        #A
        for j in range(K):
            if j!=k:
                det=Q[k,k]*Q[j,j]-Q[k,j]**2
                if det<eps:
                    print("det is too small %g"%det)
                    det=eps
                I3kkj=np.asarray([num.I3(X[alpha],k,k,j) for alpha in range(2*NUM_GAUSSIANS)])
                I22kj=np.asarray([num.I22(X[alpha],k,j) for alpha in range(2*NUM_GAUSSIANS)])
                I3kjj=np.asarray([num.I3(X[alpha],k,j,j) for alpha in range(2*NUM_GAUSSIANS)])
                
                line2-=np.asarray([lr*v[k]*v[j]*I22kj[alpha]*r[alpha,l] for alpha in range(2*NUM_GAUSSIANS)])
                line2-=np.asarray([lr*v[k]*v[j]*m/det*(I3kkj[alpha]-I22kj[alpha]*M[alpha,k])*(Q[j,j]*q[k,l]-Q[k,j]*q[j,l]) for alpha in range(2*NUM_GAUSSIANS)])
                line2-=np.asarray([lr*v[k]*v[j]*m/det*(I3kjj[alpha]-I22kj[alpha]*M[alpha,j])*(Q[k,k]*q[j,l]-Q[k,j]*q[k,l]) for alpha in range(2*NUM_GAUSSIANS)])
        
        res=C.T@np.prod(label,axis=1)
        res+=line2.T@np.prod(label.T[[1,2]].T,axis=1)
        ########## w^k delta w^l ######
        I31k=np.asarray( [num.I3_1(X[alpha],l) for alpha in range(2*NUM_GAUSSIANS)])
        I32kk=np.asarray([num.I3_2(X[alpha],l,l) for alpha in range(2*NUM_GAUSSIANS)])
        I3kkk=np.asarray( [num.I3(X[alpha],l,l,l) for alpha in range(2*NUM_GAUSSIANS)])
        I22kk=np.asarray([num.I22(X[alpha],l,l) for alpha in range(2*NUM_GAUSSIANS)])
        
        #C
        C = np.asarray([ I31k[alpha]*v[l]*lr*r[alpha,k]+lr*v[l]*m*q[k,l]/Q[l,l]*(I32kk[alpha]-I31k[alpha]*M[alpha,l]) for alpha in range(2*NUM_GAUSSIANS)])
        
        #B
        line2=-1*np.asarray([lr*v[l]*v[l]*r[alpha,k]*I22kk[alpha] for alpha in range(2*NUM_GAUSSIANS)])
        line2-=np.asarray([(lr*v[l]*v[l])*m/Q[l,l]*(I3kkk[alpha]-M[alpha,l]*I22kk[alpha])*q[k,l] for alpha in range(2*NUM_GAUSSIANS)])
        
        #A
        for j in range(K):
            if j!=l:
                det=Q[l,l]*Q[j,j]-Q[l,j]**2
                if det<eps:
                    print("det is too small %g"%det)
                    det=eps
                I3kkj=np.asarray([num.I3(X[alpha],l,l,j) for alpha in range(2*NUM_GAUSSIANS)])
                I22kj=np.asarray([num.I22(X[alpha],l,j) for alpha in range(2*NUM_GAUSSIANS)])
                I3kjj=np.asarray([num.I3(X[alpha],l,j,j) for alpha in range(2*NUM_GAUSSIANS)])
                
                line2-=np.asarray([lr*v[l]*v[j]*I22kj[alpha]*r[alpha,k] for alpha in range(2*NUM_GAUSSIANS)])
                line2-=np.asarray([lr*v[l]*v[j]*m/det*(I3kkj[alpha]-I22kj[alpha]*M[alpha,l])*(Q[j,j]*q[k,l]-Q[l,j]*q[j,k]) for alpha in range(2*NUM_GAUSSIANS)])
                line2-=np.asarray([lr*v[l]*v[j]*m/det*(I3kjj[alpha]-I22kj[alpha]*M[alpha,j])*(Q[l,l]*q[j,k]-Q[l,j]*q[k,l]) for alpha in range(2*NUM_GAUSSIANS)])
        res+=C.T@np.prod(label,axis=1)
        res+=line2.T@np.prod(label.T[[1,2]].T,axis=1)
        
        #quadratic term
        resquad=0.
        prefactor_quad=v[k]*v[l]*lr**2*m
        
        I42=np.asarray([num.I4_2(X[alpha],k,l) for alpha in range(2*NUM_GAUSSIANS)])
        resquad+=I42.T@ np.prod(label.T[[1,2]].T,axis=1)
        
        I43=np.asarray([ [num.I4_3(X[alpha],k,l,j) for j in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
        resquad-=2*(I43@v).T@np.prod(label,axis=1)
        
        I4=np.asarray([[[ num.I4(X[alpha],k,l,j,i) for j in range(K)] for i in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
        resquad+=(np.asarray([v.T@I4[alpha]@v for alpha in range(2*NUM_GAUSSIANS)])).T@ np.prod(label.T[[1,2]].T,axis=1)
        
        res+=resquad*prefactor_quad
        
        return res 
    
    K=len(v)
    dd=np.zeros(q.shape)
    for k in range(K):
        for l in range(k+1):
            dd[k,l]+=dq_comp(k,l,Masses,v,r,t,q,Q,R,M,T1,T0,lr,num,label,X,NUM_GAUSSIANS,reg,N)-2*lr*reg*q[k,l]
            if l!=k:
                dd[l,k]=dd[k,l]
    return dd
                
def update_v(v,lr,num,label,X,NUM_GAUSSIANS,reg,N):
    #print("Updating v")
    K=len(v)
    def dv_comp(k,v,lr,num,label,X,NUM_GAUSSIANS):
        K=len(v)
   
        res=0.
        
        ##First term
        I1=np.asarray([num.I1(X[alpha],k) for alpha in range(2*NUM_GAUSSIANS)])
        res+=I1.T@np.prod(label,axis=1)
        ##Second Term-->Added the time 2 because defined with a 1/2
        I2=2.0*np.asarray([ [num.I2(X[alpha],k,j) for j in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
        res-=(I2@v).T@np.prod(label.T[[1,2]].T,axis=1)
        res*=lr
        #np.prod(label,axis=1) -> vector that has 2*NUM_GAUSSIANS elements containing proba*sign*number_gaussians in cluster
        return res
    ##ATTENTION! I2=0.5*erf*erf
    dd=np.zeros(v.shape)
    for k in range(K):
        dd[k]+=dv_comp(k,v,lr,num,label,X,NUM_GAUSSIANS)-reg*lr*v[k]
    return dd

