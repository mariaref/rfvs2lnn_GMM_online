import sys
from iid_Ode import *
from time import time
import argparse
import pandas as pd
from time import time
## runs the state evolution in order to study the asymptotic properties of the solution found by the 2LNN when trained on a GMM with diagonal covariance matrix
def update_se_reduced(R,Q,v,var,lr,num,label,X,C,mu,reg,N):
    '''
    One step update of state evolution for a reduced number of degrees of freedom.
    We impose the constrains of the Ansatz on the OP.
    '''
    
    NUM_GAUSSIANS=int(len(R)/2)
    K=len(v)
    print(K)
    Rnew=np.zeros(R.shape)
    Qnew=np.zeros(Q.shape)
    vnew=np.zeros(v.shape)
    
    for beta in np.arange(0,2*NUM_GAUSSIANS,2):
        Rnew[beta,0]=R[beta,0]-reg*lr*R[beta,0]+dR_comp(0,beta,v,lr,num,label,X,NUM_GAUSSIANS) ## 1 parameter
        Rnew[beta,1]=R[beta,1]-reg*lr*R[beta,1]+dR_comp(1,beta,v,lr,num,label,X,NUM_GAUSSIANS) ## 1 parameter
    
    Rnew[0,2]=-Rnew[2,1]
    Rnew[2,2]=-Rnew[0,1]
    Rnew[0,3]=-Rnew[2,0]
    Rnew[2,3]=-Rnew[0,0]
    
    vnew[0]=-R[0,0]
    vnew[1]=R[2,1]
    vnew[2]=R[0,2]
    vnew[3]=-R[2,3]
    
    for k in range(K):
        Rnew[1,k]=-Rnew[0,k]
        Rnew[3,k]=-Rnew[2,k]
    for k in range(K):
        for l in range(K):
            Qnew[k,l]=Rnew[0,k]*Rnew[0,l]+Rnew[2,k]*Rnew[2,l]
    
    #vnew=np.sign(v)*np.sqrt(np.abs(np.diag(Q)))
    
    return Rnew,Qnew,vnew
             
def update_se(R,Q,v,var,lr,num,label,X,C,mu,reg,N):
    '''
    One step update of state evolution 
    '''
    
    NUM_GAUSSIANS=int(len(R)/2)
    
    Rnew=R+dR(v,R,lr,num,label,X,NUM_GAUSSIANS,reg,N)
    Qnew=Q+dQ(var,v,Q,lr,num,label,X,NUM_GAUSSIANS,reg,N)
    vnew=v+dv(v,lr,num,label,X,NUM_GAUSSIANS,reg,N)
    return Rnew,Qnew,vnew


def damping(q_new, q_old, coef_damping=0.4):
    return (1 - coef_damping) * q_new + coef_damping * q_old




def iterate_se(model,K,max_iter=int(1e10), time_max=None,eps=1e-10, 
               lr=0.1,verbose=False, damp=0.5,Nsamples=int(10**4),location="",initial_conditions=None,seed=0,reg=0.0,gname="relu",comment="",reduced = False):
    """ 
    Update state evolution equations. 
    
    Parameters:
    * eps = threshold to reach convergence.
    * max_iter = maximum number of steps if convergence not reached.
    * reduced = iterate only 4 degrees of freedom.
    * Nsamples = number of samples for MC integration.
    * time_max = maximal time to integral the equations for 
    """
    
    N=model.N
    NUM_GAUSSIANS=model.NUM_GAUSSIANS
    rho=model.rho
    var=model.var
    if location != "": location += "/"
    log_fname = (location+"GMM_SE%s%s_%s_NUMGAUS%d_rho%g_var%g_D%d_K%d_lr%g_seed%d_damp%g_reg%g.dat" %
                 (comment,("_red" if reduced else ""),gname, model.NUM_GAUSSIANS , model.rho , model.var, model.N,K, lr, seed,damp,reg))
    logfile = open(log_fname, "w", buffering=1)
    
    print("saving in %s"%log_fname)
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# State Evolution for Online learning with the teacher-student framework\n"
    welcome += ("# NUM_GAUSSIANS=%d, K=%d, lr=%g, seed=%d\n" % (model.NUM_GAUSSIANS, K, lr, seed))
    welcome += ("# damp%g, NSample %d" % (damp,Nsamples))
    print(welcome)
    logfile.write(welcome+"\n")
    
    [R0,Q0,T0,v0]=[initial_conditions["R"],initial_conditions["Q"],initial_conditions["T"],initial_conditions["v"]]
    lplus=np.stack(  ( np.ones(NUM_GAUSSIANS),rho*np.ones(NUM_GAUSSIANS),1./len(model.mus[0])*np.ones(NUM_GAUSSIANS))   ,axis=-1)
    lminus=np.stack(  (-1*np.ones(NUM_GAUSSIANS),(1-rho)*np.ones(NUM_GAUSSIANS),1./len(model.mus[1])*np.ones(NUM_GAUSSIANS) )  ,axis=-1)
    label=np.concatenate((lminus,lplus), axis=0) 

    num=Integrals_GM(Nsamples=Nsamples,gname=gname,dim=model.N)
    ## MC set initially 0 
    X=np.zeros((2*NUM_GAUSSIANS,Nsamples,K+2*NUM_GAUSSIANS))
    C=np.zeros((2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS))
    mu=np.zeros((2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS))
    Update_MC_Set_IID(model,K,model.NUM_GAUSSIANS,R0,Q0,T0,num,X,C,mu)

    R=R0
    Q=Q0
    v=v0
    msg=str(0)+","
 
    eg_ana=err_ana(R=R,Q=Q,v=v,lr=lr,num=num,label=label,X=X,C=C,mu=mu)
    eg_class   = classification_analytical(X,v,label)
    
    msg += ("%g,%g" % (eg_ana,eg_class))
    msg_op = write_order_param({"Q": Q,"R": R,"v": v} )
    msg+=","+msg_op
    print(msg)  
    logfile.write(msg + "\n")
 
    print_every=1
    start=time()
    tfin=0
    titer=0
    for t in range(max_iter):
        tfin=t
        if not reduced:
            Rtmp,Qtmp,vtmp = update_se(R,Q,v,model.var,lr,num,label,X,C,mu,reg,N)
        else: 
            Rtmp,Qtmp,vtmp = update_se_reduced(R,Q,v,model.var,lr,num,label,X,C,mu,reg,N)
 
        R = damping(Rtmp, R, coef_damping=damp)
        Q = damping(Qtmp, Q, coef_damping=damp)
        v = damping(vtmp, v, coef_damping=damp)

        Update_MC_Set_IID(model,K,model.NUM_GAUSSIANS,R,Q,T0,num,X , C, mu)
 
        eg_ana_new  =err_ana(R,Q,v,lr=lr,num=num,label=label,X=X,C=C,mu=mu)
        eg_class   = classification_analytical(X,v,label)
        diff = np.abs(eg_ana_new - eg_ana)
        eg_ana = eg_ana_new
 
        if (t+1)%print_every==0:
            msg=("%g,%g,%g,%g"%(t+1,eg_class,eg_ana,diff))
            msg_op = write_order_param({"Q": Q,"R": R,"v": v} )
            msg+=","+msg_op
            print(msg)
            logfile.write(msg+"\n")
 
        if diff < eps:
            logfile.write("#Bye Bye -- Converged"+"\n")
            return;
        if time_max is not None and (time()-start)>time_max:
            logfile.write("#Bye Bye -- Did not converge -- max time reached %g "%(time()-start)+"\n")
            return;
    logfile.write("#Bye Bye -- Did not converge -- iter reached"+"\n")

def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    (N,var,rho,NUM_GAUSSIANS, K,lr) = (args.N,args.var,args.rho,args.NG, args.K, args.lr)
    (damp,Nsamples,max_iter)=(args.damp,args.NS,args.MITER)
    (reg,damp,location,seed)=(args.reg,args.damp,args.location,args.seed)
    comment=args.comment
    mus=None
    
    model=Model_iid(N=N,var=var,rho=rho,NUM_GAUSSIANS=NUM_GAUSSIANS,rand_MEANS=False,mus=mus)
    if args.prefix is not None:
        print("Loading initial weights from %s"%args.prefix)
        data=np.loadtxt(args.prefix+".dat",delimiter=',')
        print(data[-1,1])
        Q,R,v=reshape_inv(data[-1,3:],K,NUM_GAUSSIANS)
        initial_conditions={}
        initial_conditions["Q"]=Q
        initial_conditions["R"]=R
        initial_conditions["v"]=v
    
    if args.K==4 and args.prefix is None:
        K=args.K
        norm=1.1*np.ones(K)
        norm=np.asarray([0.98196293,1.21409575, 1.21608174,0.97920306])
        Q=np.eye(K)
        R=np.zeros(Q.shape)
        v=np.zeros(K)
        cos=0.9
        sin=math.sqrt(1-cos**2)
        
        angles=np.asarray([cos,-sin,-cos,sin])
        R[0,:]=norm*angles
        R[1,:]=-R[0,:]
        angles=np.asarray([sin,cos,-sin,-cos])
        R[2,:]=norm*angles
        R[3,:]=-R[2,:]
        
        v=norm*np.asarray([-cos,cos,-cos,cos])
        
        for k in range(K):
            for l in range(K):
                Q[k,l]=R[0,k]*R[0,l]+R[2,k]*R[2,l]
        initial_conditions={"Q":Q,"R":R,"v":v}
        for e,el in initial_conditions.items():
            el+=np.ones(el.shape)*1e-2
    
    

    if args.K>4 and args.prefix is None:
        K=args.K
        Q=np.eye(K)
        R=np.zeros((4,K))
        v=np.zeros(K)
        ##
        R[0,0]=1.07
        R[0,2]=-0.98
        R[2,0]=0.22
        R[2,2]=0.2
        R[0,4]=0.39
        R[2,4]=-0.08
        
        if K>6:
            for k in range(6,K,2):
                R[0,k]=0.1
                R[2,k]=-0.01
        for k in np.arange(0,K,2):
            if k+1<K:
                R[2,k+1]=R[0,k]
                R[0,k+1]=R[2,k]
        for k in range(K):
            R[1,:]=-R[0,:]
            R[3,:]=-R[2,:]
        for k in np.arange(0,K,2):
            v[k]=-np.abs(R[0,k])
            if k+1<K:
                v[k+1]=np.abs(R[2,k+1])
        for k in range(K):
            for l in range(K):
                Q[k,l]=R[0,k]*R[0,l]+R[2,k]*R[2,l]
                
    print(initial_conditions)
    initial_conditions["T"]=1./float(N)*model.mus.reshape((2*NUM_GAUSSIANS,N))@model.mus.reshape((2*NUM_GAUSSIANS,N)).T
    
    ######Iterates State Evolution 
    R,Q,v,t,diff=iterate_se(model,K,max_iter=args.MITER, initial_conditions=initial_conditions,eps=args.eps,time_max=args.max_time,lr=args.lr,verbose=args.quiet,Nsamples=args.NS,location=args.location,seed=args.seed,damp=args.damp,reg=args.reg,gname=args.g,comment=comment,reduced = (args.reduced==1))    
    
    print("Bye-bye")

if __name__ == '__main__':
    
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--g", default="relu",
                        help="activation function for the student;"
                             "'relu'")
    parser.add_argument('-N', '--N', metavar='N', type=int, default=500,
                        help='number of inputs')
    parser.add_argument('-NG', '--NG', metavar='NG', type=int, default=1,
                        help="Number of Gaussians per label")
    parser.add_argument('-K', '--K', metavar='K', type=int, default=2,
                        help="size of the student's intermediate layer")
    parser.add_argument("--lr", type=float, default=.5,
                        help="learning constant")
    parser.add_argument('-q', '--quiet', help="be quiet",
                        action="store_true")
    parser.add_argument('-Three', '--Three',default=False, help="use 3 clusters",
                        action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random number generator seed. Default=0")
    parser.add_argument("-NS", "--NS", type=int, default=int(10**4),
                    help="Number of samples to compute MontecarloIntegrals error. Default=1e4")
    parser.add_argument("-MITER", "--MITER", type=int, default=int(1e5),
                    help="Maximum number of iterations for SE. Default=1e5")
    parser.add_argument("-rho", "--rho", type=float, default=0.5,
                        help="Relative size of the cluaters. Default=0.5")
    parser.add_argument("-var", "--var", type=float, default=0.05,
                        help="Variance within each cluster. Default=0.05")
    parser.add_argument("-damp", "--damp", type=float, default=0.0,
                        help="Damping constant for SE. Default=0.0")
    parser.add_argument("-max_time", "--max_time", type=float, default=None,
                        help="Max time to run SE. Default = None.")
    parser.add_argument("-eps", "--eps", type=float, default=1e-10,
                        help="Convergence criterion. Default=1e-10.")
    parser.add_argument("-location", "--location", type=str, default=None,
                        help="Where to save the data. Default=None")
    parser.add_argument("-comment", "--comment", type=str, default="",
                        help="Nice comment to add to your file. Default=""")
    parser.add_argument("-reg", "--reg", type=float, default=0.0,
                        help="Regularisation of weights. Default 0.0")
    parser.add_argument("-prefix", "--prefix", type=str, default=None,
                        help="File containing the initial SE weights. Default=None")
    parser.add_argument("-reduced", "--reduced", type=int, default=0,
                        help="Run only for the reduced asatz that is true for XOR. Default=0")
    args = parser.parse_args()
    main(args)