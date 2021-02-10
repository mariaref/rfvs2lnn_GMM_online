import numpy as np
import numpy.random as rnd
from math import sqrt as sqrt
from math import pi as pi

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import torch
from torch import erf as erf
import argparse

import sys
sys.path.insert(1,'include')
from Integrals_RELU import erf_np

def get_projection(projection): ### normalising the rows of the projection matrix
    D,N=projection.shape
    projection/=projection.norm(dim=0).repeat(D,1)
    projection*=sqrt(D) ## puts the rows on the sphere od radius sqrt(D)
    return projection
def mu_XOR(D):
    mus=torch.zeros(2,2,D)
    mus[0,0,0]= sqrt(D)
    mus[0,1,0]=-sqrt(D)
    mus[1,0,1]= sqrt(D)
    mus[1,1,1]=-sqrt(D)
    return mus
def mu_intermediate(D):
    mus=torch.zeros(2,2,D)
    mus[0,0,0]= sqrt(D)
    mus[0,0,1]= sqrt(1)
    mus[0,1,0]=-sqrt(D)
    mus[0,1,1]=-sqrt(1)
    mus[1,0,0]= sqrt(D)
    mus[1,0,1]=-sqrt(1)
    mus[1,1,0]=-sqrt(D)
    mus[1,1,1]= sqrt(1) 
    mus=sqrt(D)*mus
    return mus 
def mu_three_1(D):
    mus=torch.zeros(2,2,D)
    mus[0,0,0]=math.sqrt(D)
    mus[0,1,0]=-math.sqrt(D)
    return mus
def mu_three_2(D):
    mus=mu_three_1(D)*math.sqrt(D)
    return mus
def mu_three_3(D):
    mus=torch.zeros(2,2,D)
    mus[0,0,0]=D
    mus[0,1,0]=-math.sqrt(D)
    return mus*sqrt(D)

def get_statistics(bs,D,N,mus,gamma,sigma,projection):
    """return the means of the features as a tensor [2,2,N]"""
    xs=torch.zeros(2,2,bs,D)
    ys=torch.zeros(2,2,bs,1)
    zs=torch.zeros(2,2,bs,N)
    muZ=torch.zeros(2,2,N)
    covZ=torch.zeros(2,2,N,N)
    for sign in range(2):
        for clusterID in range(2):
            ys[sign,clusterID] = (2*sign-1)*torch.ones(ys[sign,clusterID].shape)
            xs[sign,clusterID]=torch.randn(bs,D)*sigma+mus[sign,clusterID].repeat(bs,1)*gamma/sqrt(D)
            zs[sign,clusterID]=get_rf(xs[sign,clusterID],projection)
            muZ[sign,clusterID]=zs[sign,clusterID].mean(dim=0)
            covZ[sign,clusterID]=(zs[sign,clusterID]-muZ[sign,clusterID]).t()@(zs[sign,clusterID]-muZ[sign,clusterID])
            covZ[sign,clusterID]/=float(zs[sign,clusterID].shape[0])
    return ys, xs, zs, muZ, covZ
def get_rf(x,projection,Psi=torch.relu):
    D,N=projection.shape
    z=Psi(x@projection/sqrt(D))
    return z
def getAnalyticalMoments(proj,mu,sigma,gamma):
    D, N = proj.shape
    rho = mu @ proj/(D*sigma)*gamma ## rho is defined as E[u]/std(u) 
    def rerf(x): return erf(x/sqrt(2))
    exp = torch.exp(-rho**2/2.)
    # analytical mean
    mu_ana = sigma/2.*( rho*(1 + rerf(rho))+exp*sqrt(2./pi) )
    # analytical variance
    diag_ana =  sigma**2/2*( rho * exp*sqrt(2./pi) + (rho*rho + 1) *(1+rerf(rho)) )
    diag_ana-=mu_ana*mu_ana
    # analytical covariance
    M=proj*(1+rerf(rho))
    off_diag_ana=M.t()@M/float(D)*sigma**2/4
    off_diag_ana-=torch.diag(off_diag_ana.diag())
    cov_ana = off_diag_ana + torch.diag(diag_ana)
    return mu_ana, cov_ana
def getAnalyticalTotalMoments(muV,covV, NG, N ):
    ## computes total mean
    muZ  = muV.mean(dim=0)
    ## computes total covariance 
    covZ = torch.zeros(N,N)
    for alpha, mu1 in enumerate(muV):
        covZ += ( covV[alpha] + torch.einsum('i,j->ij',mu1,mu1) )
        for beta, mu2 in enumerate(muV):
            covZ-= 1./(2.*NG)*torch.einsum('i,j->ij',mu1,mu2)
    covZ*=1./(2.*NG)
    return muZ , covZ
def get_analytical_estimator(covZ, muZ,NG,N):
    ## compute analytical estimator
    ## computes input-label covariance
    Phi = 1./(2.*NG)*(-torch.sum(muZ[:int( muZ.shape[0]/2)],dim=0)+torch.sum(muZ[int( muZ.shape[0]/2):],dim=0))
    print("computing evecs...")
    evals, evecs = torch.symeig(covZ+1e-6,eigenvectors=True)
    print("done computing evecs")
    Phi_Tau = (Phi @ evecs).squeeze()
    w_Tau = torch.zeros(N)
    w_Tau = torch.einsum('i,i->i',Phi_Tau,(evals**-1) )*sqrt(N)
    w = (evecs @ w_Tau)
    return w

def get_RF_mse_Analytical(proj, mus, sigma, gamma):
    NG = mus.shape[1]
    D, N = proj.shape
    muAnaV=torch.zeros(2,NG,N)
    covAnaV=torch.zeros(2,NG,N,N)
    for sign in range(2):
        for clusterID in range(NG):
            muAnaV[sign,clusterID],  covAnaV[sign,clusterID] = getAnalyticalMoments(proj,mus[sign,clusterID],sigma,gamma) 
    muAnaV = muAnaV.reshape(2*NG,N)
    covAnaV = covAnaV.reshape(2*NG,N,N)
    ## computes total moments
    muZ , covZ = getAnalyticalTotalMoments(muAnaV,covAnaV, NG, N )
    
    w = get_analytical_estimator(covZ, muAnaV,NG,N)
    
    ## computes input-label covariance
    Phi = 1./(2.*NG)*(-torch.sum(muAnaV[:int( muAnaV.shape[0]/2)],dim=0)+torch.sum(muAnaV[int( muAnaV.shape[0]/2):],dim=0))
    
    print("computing evecs...")
    evals, evecs = torch.symeig(covZ+1e-6,eigenvectors=True)
    print("done computing evecs")
    Phi_Tau = (Phi @ evecs).squeeze()
    w_Tau = torch.zeros(N)
    w_Tau = torch.einsum('i,i->i',Phi_Tau,(evals**-1) )*sqrt(N)
    ##computes the error from the formula
    error = ( Phi_Tau*Phi_Tau*(evals**-1) ).sum()
    error = 0.5*(1-error)
    print("error from formula is %g"%error)
    return error

def get_RF_mse_Numerical(proj, mus, sigma, gamma,bs=10000):
    D,N = proj.shape
    ys, xs, zs, muVarious, covVarious = get_statistics(bs,D,N,mus,gamma,sigma,proj)
    zs = zs.reshape(4*bs, N)
    ys = ys.reshape(4*bs).float()
    muZ = zs.mean(dim=0)
    zs -= muZ
    Phi = ys.t()@zs/float(zs.shape[0])
    covZ = zs.t()@zs/float(zs.shape[0])
    covZ = (covZ+covZ.t())*0.5
    print("computing evecs...")
    evals, evecs = torch.symeig(covZ,eigenvectors=True)
    print("done computing evecs")
    Phi_Tau = (Phi @ evecs).squeeze()
    w_Tau = torch.zeros(N)
    #print(evals>0)
    w_Tau = torch.einsum('i,i->i',Phi_Tau,(evals**-1) )*sqrt(N)
    #w_Tau = torch.einsum('i,i->i',Phi_Tau,(evals**-1) )*sqrt(N)
    ## computes the error from the estimator
    w = (evecs @ w_Tau)
    preds = zs@w.t()/sqrt(N)
    error_est = torch.mean( 0.5*( ys - preds )**2 )
    print("error from estimator is %g"%error_est)
    ##computes the error from the formula
    error = ( Phi_Tau*Phi_Tau*(evals**-1) ).sum()
    error = 0.5*(1-error)
    print("error from formula is %g"%error)
    return error_est, error

def get_RF_classification_analytical(proj, mus, sigma,gamma):
    var = sigma**2; NG = mus.shape[1]
    D, N = proj.shape;
    muAnaV=torch.zeros(2,NG,N)
    covAnaV=torch.zeros(2,NG,N,N)
    for sign in range(2):
        for clusterID in range(NG):
            muAnaV[sign,clusterID],  covAnaV[sign,clusterID] = getAnalyticalMoments(proj,mus[sign,clusterID],sigma,gamma) 
    muAnaV = muAnaV.reshape(2*NG,N)
    covAnaV = covAnaV.reshape(2*NG,N,N)
    muZ , covZ = getAnalyticalTotalMoments(muAnaV,covAnaV, NG, N )
    w = get_analytical_estimator(covZ, muAnaV,NG,N)
    #For each cluster compute the mean and the standard deviation of the local fieds
    rho=0.5
    Q =  np.zeros((2*NG)); M =  np.zeros((2*NG))
    for cluster in range(2*NG):
        Q[cluster]  = 1./N*w@covAnaV[cluster]@w.T
        M[cluster]  = 1./sqrt(N)*w@(muAnaV[cluster]-muZ).T
    ## compute the error
    error = 1./2.
    error+= 1/8*( erf_np(M[0]/sqrt(Q[0])) +erf_np(M[1]/sqrt(Q[1])) -erf_np(M[2]/sqrt(Q[2])) -erf_np(M[3]/sqrt(Q[3]))  )
    return error

def main(args):
    print("You have just entered this amazing script to compute the performances of RF model.")
    print("please wait while I do the computations... Go have a cofee, its good for you!")
    N=args.D*args.alpha
    if args.three == 0:
        if   args.regime==1: mus = mu_XOR(args.D)
        elif args.regime==2: mus = mu_Long(args.D)
        elif args.regime==3: mus = mu_intermediate(args.D)  
    elif args.three == 1:
        if   args.regime==1: mus = mu_three_1(args.D)
        elif args.regime==2: mus = mu_three_2(args.D)
        elif args.regime==3: mus = mu_three_3(args.D)
            
    loc = args.loc
    if args.loc !="": loc = loc+"/" 
    if args.prefix is None:
        logfile=loc+"RF%s%s%s_regime%d_gamma%d_D%d_N%d_sigma%g.dat"%(args.comment,"_"+args.error,("_three" if args.three==1 else ""), args.regime,args.gamma, args.D, N,args.sigma)
        print("writing in a new file : %s"%logfile)
        logfile = open(logfile, "w", buffering=1)
    elif args.prefix is not None:
        print("writing in an existing file : %s"%args.prefix)
        logfile = open(logfile, "w", buffering=1)
    
    proj=get_projection(torch.randn(args.D,N))
    if args.error == "mse": error = get_RF_mse_Analytical(proj, mus, args.sigma, args.gamma)
    elif args.error == "class": error = get_RF_classification_analytical(proj, mus, args.sigma, args.gamma)
    logfile.write("%g,%g"%(args.sigma, error )+"\n")
    print("Done! Have a nice day and please come again!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-regime", "--regime",type=int, default=1, help="regime in which we opearte (options 1, 2 or 3).")
    parser.add_argument("-three", "--three",type=int, default=0, help="whether or not to use the 3 clusters model.")
    parser.add_argument('-D', '--D', type=int, default=300, help='input dimension. Default 300')
    parser.add_argument('-alpha', '--alpha', type=int, default=2, help='ratio number of features / input dimensions. Default 2') 
    parser.add_argument('-error', '--error', type=str, default="class", help='use MSE or classification error')
    parser.add_argument('-sigma', '--sigma', type=float, default=0.01, help='std of the direct space clusters. Default 0.01')
    parser.add_argument('-gamma', '--gamma', type=float, default=1, help='scaling constant gamma. Default 1.')
    parser.add_argument("-comment", "--comment", type=str, default="",help="add a nice comment to your logfile")
    parser.add_argument("-loc", "--loc", type=str, default="",help="where to save the data. Default="" ")
    parser.add_argument("-prefix", "--prefix", type=str, default=None,help="prefix to keep writing. Default="" ")
    args = parser.parse_args()
    main(args)
    
    
