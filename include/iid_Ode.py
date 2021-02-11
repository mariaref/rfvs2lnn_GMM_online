import sys
sys.path.insert(1, '../include')
from GMM_Model import *
from Integrals_RELU import *
from ToolBox import *
from time import time
import argparse
import pandas as pd
from time import time
from Train_iid_Model import *
## implements the ODEs for the order parameters in the simplified case in which the clusters have identity covariance

def dR_comp(k,beta,v,lr,num,label,X,NUM_GAUSSIANS):
    K=len(v)
    prefactor=v[k]*lr
    res=0.
    I32=np.asarray([num.I3_2(X[alpha],k,K+beta) for alpha in range(2*NUM_GAUSSIANS)])
    ## I3 is of shape (2*NUM_GAUSSIANS, K ) and contains all the I3(k,beta,j) integrals
    I3=np.asarray([ [num.I3(X[alpha],k,K+beta,j) for j in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
    res+=I32.T@np.prod(label,axis=1)
    res-=(I3@v).T @ np.prod(label.T[[1,2]].T,axis=1)
    res*=prefactor
    #np.prod(label,axis=1) -> vector that has 2*NUM_GAUSSIANS elements containing proba*sign*number_gaussians in cluster
    return res
def dR(v,R,lr,num,label,X,NUM_GAUSSIANS,reg,N):
    K=len(v)
    dR=np.zeros((2*NUM_GAUSSIANS,K))
            
    for k in range(K):
        for beta in range(0,2*NUM_GAUSSIANS,2):
            dR[beta,k]=dR_comp(k,beta,v,lr,num,label,X,NUM_GAUSSIANS)-reg*lr*R[beta,k]
            dR[beta+1,k]=-dR[beta,k]
    #dR=np.asarray([[dR_comp(k,beta,v,lr,num,label,X,C,mu) for k in range(K)]for beta in range(2*NUM_GAUSSIANS)])
    return dR

def dQ_comp(var,k,l,v,lr,num,label,X,NUM_GAUSSIANS):
    K=len(v)
    ###linear term Delta_w^l
    resl=0.
    I32l=np.asarray([num.I3_2(X[alpha],l,k) for alpha in range(2*NUM_GAUSSIANS)])
    I3l=np.asarray([ [num.I3(X[alpha],l,k,j) for j in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
    resl+=I32l.T@np.prod(label,axis=1)
    resl-=(I3l@v).T @ np.prod(label.T[[1,2]].T,axis=1)
    resl*=v[l]*lr
    ###linear term Delta_w^k
    resk=0.
    I32k=np.asarray([num.I3_2(X[alpha],k,l) for alpha in range(2*NUM_GAUSSIANS)])
    I3k=np.asarray([ [num.I3(X[alpha],k,l,j) for j in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
    resk+=I32k.T@np.prod(label,axis=1)
    resk-=(I3k@v).T @ np.prod(label.T[[1,2]].T,axis=1)
    resk*=v[k]*lr
    ##sum of linear terms
    res=resk+resl
    
    #quadratic term
    resquad=0.
    prefactor_quad=v[k]*v[l]*lr**2*var
    
    I42=np.asarray([num.I4_2(X[alpha],k,l) for alpha in range(2*NUM_GAUSSIANS)])
    resquad+=I42.T@ np.prod(label.T[[1,2]].T,axis=1)
    
    I43=np.asarray([ [num.I4_3(X[alpha],k,l,j) for j in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
    resquad-=2*(I43@v).T@np.prod(label,axis=1)
    
    I4=np.asarray([[[ num.I4(X[alpha],k,l,j,i) for j in range(K)] for i in range(K)] for alpha in range(2*NUM_GAUSSIANS)])
    resquad+=(np.asarray([v.T@I4[alpha]@v for alpha in range(2*NUM_GAUSSIANS)])).T@ np.prod(label.T[[1,2]].T,axis=1)
    resquad*=prefactor_quad
    
    res+=resquad
    #np.prod(label,axis=1) -> vector that has 2*NUM_GAUSSIANS elements containing proba*sign*number_gaussians in cluster
    if k==0 and l==1 : print(res)
    return res
    
def dQ(var,v,Q,lr,num,label,X,NUM_GAUSSIANS,reg,N):
    K=len(v)
    dQ=np.zeros((K,K))
    for k in range(K):
        ## Diagonal terms
        #dQ[k,k]=dQ_comp
        for l in range(k+1):
           # print("%d%d"%(k,l))
            dQ[k,l]=dQ_comp(var,k,l,v,lr,num,label,X,NUM_GAUSSIANS)-2*lr*reg*Q[k,l]#+lr**2/float(N)*reg**2*Q[k,l]
            if l!=k:
                dQ[l,k]=dQ[k,l] 
    return dQ

def dv(v,lr,num,label,X,NUM_GAUSSIANS,reg,N):
    K=len(v)
    ##ATTENTION! I2=0.5*erf*erf
    dv=np.zeros(K)
    for k in range(K):
        dv[k]=dv_comp(k,v,lr,num,label,X,NUM_GAUSSIANS)-reg*lr*v[k]
    return dv

def dv_comp(k,v,lr,num,label,X,NUM_GAUSSIANS):
    K=len(v)
    #print("evolving component %d"%(k))
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


def run(model,student,lr=0.5,verbose=False, Nsamples=int(1e6),location="data/",prefix=None,seed=0,reg=0.0,steps=1000,comment="",init_time=0,dstep = None):
    N=model.N
    # output file + welcome message
    log_fname = (location+"GaussianMixtures%s_%s_NUMGAUS%d_rho%g_var%g_N%d_K%d_lr%g_reg%g%s_seed%d_ode.dat" %
                 (comment,student.gname, model.NUM_GAUSSIANS , model.rho , model.var, model.N,student.K, lr, reg,("_dstep%g"%dstep if dstep is not None else ""),seed))
    print("saving in %s"%log_fname)
    
    logfile = open(log_fname, "w", buffering=1)
    
    start=time()
    NUM_GAUSSIANS=model.NUM_GAUSSIANS
    rho=model.rho
    var=model.var
    lplus=np.stack(  ( np.ones(NUM_GAUSSIANS),rho*np.ones(NUM_GAUSSIANS),1./len(model.mus[0])*np.ones(NUM_GAUSSIANS))   ,axis=-1)
    lminus=np.stack(  (-1*np.ones(NUM_GAUSSIANS),(1-rho)*np.ones(NUM_GAUSSIANS),1./len(model.mus[1])*np.ones(NUM_GAUSSIANS) )  ,axis=-1)
    label=np.concatenate((lminus,lplus), axis=0) 
    num=Integrals_GM(Nsamples=Nsamples,gname=student.gname,dim= student.K + model.NUM_GAUSSIANS*2)
    
    #loads initial conditions
    vec=np.loadtxt(prefix,delimiter=',')[init_time]
    q,r,vv=reshape_inv(vec[3:],student.K,model.NUM_GAUSSIANS)
    v=[vv]
    R=[r]
    Q=[q]
    
    mus=model.mus.reshape(2*model.NUM_GAUSSIANS,model.N)
    T=1/float(model.N)*mus@mus.t()
    Tn=T.numpy()
    
    eg_ana=[]
    max_iter=int(steps*N)
    if dstep is None: dstep=1./model.N
    
    ##########
    ## MC set initially 0 
    X=np.zeros((2*model.NUM_GAUSSIANS,Nsamples,student.K+2*model.NUM_GAUSSIANS))
    C=np.zeros((2*model.NUM_GAUSSIANS,student.K+2*model.NUM_GAUSSIANS,student.K+2*model.NUM_GAUSSIANS))
    mu=np.zeros((2*model.NUM_GAUSSIANS,student.K+2*model.NUM_GAUSSIANS))
    
    # when to print?
    end = torch.log10(torch.tensor([1. * steps])).item()
    steps_to_print = list(torch.logspace(-2, end, steps=50))
    print("I am going to print for %d steps"%len(steps_to_print))
    print("I am going to run for %d iterations"%max_iter)
    t_start=time()
    t_it=0
    for t in range(max_iter):
        t_it=time()
        step=t*dstep
        print(t)
        tic = time()
        Update_MC_Set_IID(model,student.K,model.NUM_GAUSSIANS,R[-1],Q[-1],T,num,X,C,mu)
        if step >= steps_to_print[0].item() or step == 0:
            eg_ana.append( err_ana(R=R[-1],Q=Q[-1],v=v[-1],lr=lr,num=num,label=label,X=X,C=C,mu=mu))
            eg_c = classification_analytical(X,v[-1],label)
            msg = ("%g,%g, %g" % (step,eg_c,  eg_ana[-1]))
            msg_op = write_order_param({"Q": Q[-1],"R": R[-1],"v": v[-1]})
            msg+=","+msg_op
            print(msg)
            logfile.write(msg + "\n")
            steps_to_print.pop(0)
        varQ=dQ(model.var,v[-1],Q[-1],lr,num,label,X,model.NUM_GAUSSIANS,reg,N)*dstep
        varR=dR(v[-1],R[-1],lr,num,label,X,model.NUM_GAUSSIANS,reg,N)*dstep
        varv=dv(v[-1],lr,num,label,X,model.NUM_GAUSSIANS,reg,N)*dstep
        Q.append(Q[-1]+varQ)
        R.append(R[-1]+varR)
        v.append(v[-1]+varv)
        
    logfile.write("# took %g seconds to converge "%(time()-t_start) + "\n")
    print("Bye-bye")
    return R,Q,v

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    (N,var,rho,NUM_GAUSSIANS, K,lr) = (args.N,args.var,args.rho,args.NG, args.K, args.lr)
    (steps,NUM_TESTS,Nsamples)=(args.steps,args.NT,args.NS)
    comment=args.comment
    
    comment+="regime%d_"%args.regime
    if args.Three==1:
        comment+="3CLUST_"
        if args.regime==1:
            NUM_GAUSSIANS=2
            mus=torch.zeros(2,NUM_GAUSSIANS,args.N)
            mus[0,0,0]=math.sqrt(args.N)
            mus[0,1,0]=-math.sqrt(args.N)
        elif args.regime==2:
            NUM_GAUSSIANS=2
            mus=torch.zeros(2,NUM_GAUSSIANS,args.N)
            mus[0,0,0]=math.sqrt(args.N)
            mus[0,1,0]=-math.sqrt(args.N)
            mus*=math.sqrt(N)
        elif args.regime==3:
            NUM_GAUSSIANS=2
            mus=torch.zeros(2,NUM_GAUSSIANS,args.N)
            mus[0,0,0]=math.sqrt(args.N)
            mus[0,1,0]=-1
            mus*=math.sqrt(N)
    elif args.Three==0:
        if args.regime==1:
            mus=get_means(N=N,NUM_GAUSSIANS=NUM_GAUSSIANS,length=1.)
        elif args.regime==2 :
            mus=get_means(N=N,NUM_GAUSSIANS=NUM_GAUSSIANS,length=1.)
            mus*=math.sqrt(N)
        elif args.regime==3:
            mus=torch.zeros((2,NUM_GAUSSIANS,N))
            mus[0,0,0]=math.sqrt(args.N)
            mus[0,0,1]=math.sqrt(1)
            mus[0,1,0]=-math.sqrt(args.N)
            mus[0,1,1]=-math.sqrt(1)
            mus[1,0,0]=math.sqrt(args.N)
            mus[1,0,1]=-math.sqrt(1)
            mus[1,1,0]=-math.sqrt(args.N)
            mus[1,1,1]=math.sqrt(1) 
            mus*=math.sqrt(N)
            
    model=Model_iid(N=N,var=var,rho=rho,NUM_GAUSSIANS=NUM_GAUSSIANS,rand_MEANS=False,mus=mus)
    student=Student(K,N,act_function=args.g,w=None,v=None)
    prefix=args.prefix
      
    if prefix is None:
        prefix=train(model,student,location=args.location,NUM_TESTS=NUM_TESTS,bs=1,lr=lr,steps=args.steps,quiet=args.quiet,ana=False,seed=args.seed,reg=args.reg)
    
    R,Q,v=run(model,student,
               lr=lr,verbose=args.quiet, Nsamples=args.NS,location=args.location,prefix=prefix,seed=args.seed,reg=args.reg,steps=args.steps,comment=comment,init_time=args.t0, dstep = args.dstep)
    
    print("Bye-bye")

if __name__ == '__main__':
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--g", default="erf",
                        help="activation function for the student;"
                             "'erf'")
    parser.add_argument('-N', '--N', metavar='N', type=int, default=500,
                        help='number of inputs')
    parser.add_argument('-NG', '--NG', metavar='NG', type=int, default=1,
                        help="Number of Gaussians per cluster")
    parser.add_argument('-K', '--K', metavar='K', type=int, default=2,
                        help="size of the student's intermediate layer")
    parser.add_argument("--lr", type=float, default=.5,
                        help="learning constant")
    parser.add_argument("--steps", type=float, default=int(1e8),
                        help="training steps")
    parser.add_argument('-q', '--quiet', help="be quiet",
                        action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random number generator seed. Default=0")
    parser.add_argument("-NT", "--NT", type=int, default=int(1e4),
                        help="Number of tests to compute numerical error. Default=1e4")
    parser.add_argument("-NS", "--NS", type=int, default=int(10**4),
                    help="Number of samples to compute MontecarloIntegrals error. Default=1e6")
    parser.add_argument("-rho", "--rho", type=float, default=0.5,
                        help="Relative size of the cluaters. Default=0.5")
    parser.add_argument("-var", "--var", type=float, default=0.01,
                        help="Variance within each cluster. Default=0.01")
    parser.add_argument("-location", "--location", type=str, default="..",
                        help="Where to save the data. Default=..")
    parser.add_argument("-reg", "--reg", type=float, default=0.0,
                        help="Regularisation parameter. Default=0.0.")
    parser.add_argument("-prefix", "--prefix", type=str, default=None,
                        help="Initial Conditions. Default=""")
    parser.add_argument('-comment', '--comment', type=str, default="",
                      help="initialisation of the weights")
    parser.add_argument('-Three', '--Three', type=int , default=0,
                        help="Use 3 clusters model. Default 0")
    parser.add_argument('-regime', '--regime', type=int , default=0,
                        help="regime in which to run training")
    parser.add_argument('-t0', '--t0', type=int , default=0,
                        help="index to initiale the ODEs")
    parser.add_argument('-dstep', '--dstep', type=float , default=None,
                        help="time step of the ODEs")
      
    args = parser.parse_args()
    main(args)
