import torch
import math

def get_MNIST_model_2CLUSTERS(dataset,dataset_name="mnist"):
    mu={};xs={};ys={};cov={};F={};std={};evals={};evecs={};diag={};

    P,N,N=dataset.data.shape
    xs["tot"]=dataset.data.clone().detach()
    ys["tot"]=dataset.targets.clone().detach()
    xs["tot"]=xs["tot"].view(-1,N*N).float()
    N*=N

    xs["tot"]-=xs["tot"].mean(dim=0)
    ##Important.. 
    xs["tot"]/=xs["tot"].std()
   
    cov["mean"]=torch.zeros(10,N,N)

    for sign in range(2):
        xs[sign]=xs["tot"][(ys["tot"]+1)%2==sign].clone().detach().float()
        Pint , _ =xs[sign].shape
        ys[sign]=ys["tot"][(ys["tot"]+1)%2==sign].clone().detach().float() 
        mu[sign]=xs[sign].mean(dim=0) ## mus[0] contains the odd numbers
        std[sign]=xs[sign].std()
        cov[sign]=( (xs[sign]-mu[sign]).t()@(xs[sign]-mu[sign]))
        cov[sign]/=float(Pint) 
        cov[sign]+= 10**-4* torch.diag(torch.ones(N))
        cov["mean"][sign]=cov[sign]
        diag[sign]=torch.diag(cov[sign])
        
        evals[sign],evecs[sign]=torch.symeig(cov[sign], eigenvectors=True)
        
        if torch.all(evals[sign]>0):
            print("%d covariance matrix is positive defite"%sign)
        if torch.max(cov[sign]-cov[sign].t())==0.:
            print("%d covariance matrix is symmetric"%sign)   
        F[sign]=evecs[sign]@torch.diag(torch.sqrt(evals[sign]))@evecs[sign].T
    
    cov["mean"]=cov["mean"].mean(dim=0)
    evals["mean"],evecs["mean"]=torch.symeig(cov["mean"], eigenvectors=True)
    
    NUM_GAUSSIANS=1;
    mus_Model=torch.zeros(2,NUM_GAUSSIANS,N)
    if dataset_name=="mnist":
        for sign in range(2):
            if (sign%2==0): a=1 #even number are in a==1 and odd numbers in a==0
            else: a=0
            mus_Model[a][0]=mu[sign]
    elif dataset_name=="fmnist":
        mus_Model[0][0]=torch.mean(mu[:5],dim=0)
        mus_Model[1][0]=torch.mean(mu[5:],dim=0)
    print("Building F as mean of all sign")
    F["mean"]=evecs["mean"]@torch.diag(torch.sqrt(evals["mean"]))@evecs["mean"].T
    
    return F["mean"], mus_Model*math.sqrt(N)

def get_MNIST_model(dataset,dataset_name="mnist"):
    mu={};xs={};ys={};cov={};Fs={};std={};evals={};evecs={};diag={};

    P,N,N=dataset.data.shape
    xs["tot"]=dataset.data.clone().detach()
    ys["tot"]=dataset.targets.clone().detach()
    xs["tot"]=xs["tot"].view(-1,N*N).float()
    N*=N
        
    xs["tot"]-=xs["tot"].mean(dim=0)
    xs["tot"]/=xs["tot"].std()
    cov["mean"]=torch.zeros(10,N,N)

    for dig in range(10):
        xs[dig]=xs["tot"][ys["tot"]==dig].clone().detach().float()
        Pint , _ =xs[dig].shape
        ys[dig]=ys["tot"][ys["tot"]==dig].clone().detach().float()
        mu[dig]=xs[dig].mean(dim=0)
        std[dig]=xs[dig].std()
        cov[dig]=( (xs[dig]-mu[dig]).t()@(xs[dig]-mu[dig]))
        cov[dig]/=float(Pint) 
        cov["mean"][dig]=cov[dig]
        
         
    cov["mean"]=cov["mean"].mean(dim=0)
    evals["mean"],evecs["mean"]=torch.symeig(cov["mean"]+1e-5*torch.eye(N), eigenvectors=True)
    if torch.all(evals["mean"]>0):
        print("mean covariance matrix is positive defite")
    if torch.max(cov["mean"]-cov["mean"].t())==0.:
        print("mean covariance matrix is symmetric")  
    NUM_GAUSSIANS=5;
    mus_Model=torch.zeros(2,NUM_GAUSSIANS,N)
    if dataset_name=="mnist":
        for dig in range(10):
            if (dig%2==0): a=1
            else: a=0
            b=dig//2
            mus_Model[a][b]=mu[dig]
    elif dataset_name=="fmnist":
        for n in range(5):
            mus_Model[0][n]=mu[n]
            mus_Model[1][n]=mu[5+n]
    print("Building F as mean of all digits")
    Fs["mean"]=evecs["mean"]@torch.diag(torch.sqrt(evals["mean"]))@evecs["mean"].T
    return Fs["mean"], mus_Model*math.sqrt(N)