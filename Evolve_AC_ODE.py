import sys
sys.path.insert(1,'include')
from AC_Ode import *
import torchvision
import torchvision.datasets as datasets
from DataSet_Generation import *
from AC_Train_Model import *


def get_msg(Q,R,M,v,NUM_GAUSSIANS,K):
    msg=""
    for k in range(K):
        for l in range(k+1):
            msg+="%g,"%Q[k,l]
    for alpha in range(2*NUM_GAUSSIANS):
        for l in range(K):
            msg+="%g,"%R[alpha,l]
    for k in range(K):
        msg+="%g,"%v[k]
    for alpha in range(2*NUM_GAUSSIANS):
        for l in range(K):
            msg+="%g,"%M[alpha,l]
    msg=msg[:-1]
    return msg

def evolve(model,student,steps,lr,OP_SIMU,OP=None,reg=0,Nsamples=int(10**4),location="",comment="",dstep = None):
    NUM_GAUSSIANS=model.NUM_GAUSSIANS
    rho=model.rho
    N=model.N
    K=student.K
    Masses=model.Masses.numpy()
    if location !="": location+="/"
    log_fname = (location+"GMM_2lnn_%s%s_NUMGAUS%d_rho%g_N%d_K%d_lr%g_reg%g%s_steps%g" %
                 (student.gname,comment, model.NUM_GAUSSIANS , model.rho , model.N,student.K, lr, reg,("_dstep%g"%dstep if dstep is not None else ""),steps))
    log_fname+="_ode.dat"
    print("saving in %s"%log_fname)
    
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Online learning with the teacher-student framework\n"
    welcome += ("# NUM_GAUSSIANS=%d, K=%d, lr=%g, reg=%g,\n" % (model.NUM_GAUSSIANS, student.K, lr,reg))
    print(welcome)
    logfile.write(welcome + '\n')
    
    r=OP_SIMU["r"]
    q=OP_SIMU["q"]
    v=OP_SIMU["v"]
    t=OP_SIMU["t"]
    
    if OP is None:
        OP={"R":[],"M":[],"Q":[],"v":[],"step":[]}
    num=Integrals_GM(Nsamples=Nsamples,gname=student.gname, dim= K + NUM_GAUSSIANS*2)
    lplus=np.stack(  ( np.ones(NUM_GAUSSIANS),rho*np.ones(NUM_GAUSSIANS),1./len(model.mus[0])*np.ones(NUM_GAUSSIANS))   ,axis=-1)
    lminus=np.stack(  (-1*np.ones(NUM_GAUSSIANS),(1-rho)*np.ones(NUM_GAUSSIANS),1./len(model.mus[1])*np.ones(NUM_GAUSSIANS) )  ,axis=-1)
    label=np.concatenate((lminus,lplus), axis=0) 
    mus=model.mus.reshape(2*model.NUM_GAUSSIANS,N).detach().clone()
    Fs=model.Fs.detach().clone()
    Psis=model.Psis.detach().clone()
    mustau=(mus@Psis*1./math.sqrt(N)).numpy()

    # when to print?
    end = torch.log10(torch.tensor([1. * steps])).item()
    steps_to_print = list(torch.logspace(-2, end, steps=100))
    print("I am going to print for %d steps"%len(steps_to_print))
    step = 0.
    if dstep is None: dstep = 1. / float(N)
    i=0
    
    ## MC set initially 0 
    X=np.zeros((2*NUM_GAUSSIANS,Nsamples,K+2*NUM_GAUSSIANS))
    C=np.zeros((2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS))
    mu=np.zeros((2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS))
    
    while step<=steps:
        OP["step"].append(step)
        R=np.mean((r*Masses),axis=2)
        Q=np.mean((q*Masses),axis=2)
        T=np.mean((t*Masses),axis=2)
        M=np.mean(r,axis=2)
        MT=np.mean(t,axis=2)
        if step==0:
            print("R=")
            print(R)
            print("Q=")
            print(Q)
            print("v=")
            print(v)
        OP["v"].append(v)
        OP["R"].append(R)
        OP["Q"].append(Q)
        OP["M"].append(M)
        OP["step"].append(step)
        Update_MC_Set(model,K,NUM_GAUSSIANS,R,M,Q,T,MT,num,X , C, mu)
        
        #eg_ana=err_ana_ANA(R,Q,v,lr,num,label,X,C,mu)
        eg_ana=err_ana(R=R,Q=Q,v=v,lr=lr,num=num,label=label,X=X,C=C,mu=mu)
        eg_c = classification_analytical(X,v,label)
        msg = ("%g, %g,%g,")% (step,eg_ana, eg_c)
        msg+=get_msg(Q,R,M,v,NUM_GAUSSIANS,K)
        if step >= steps_to_print[0].item() or step == 0:
            print(msg)
            logfile.write(msg + "\n")
            steps_to_print.pop(0)
        
        ##befor updating stores them#####
        rnew=r+update_r(Masses,v,r,t,q,Q,R,M,T,MT,lr,num,label,X,model.NUM_GAUSSIANS,reg,model.N)*dstep
        qnew=q+update_q(Masses,v,r,t,q,Q,R,M,T,MT,lr,num,label,X,model.NUM_GAUSSIANS,reg,model.N)*dstep
                   #update_q(Masses,v,r,t,q,Q,R,M,T1,T0,lr,num,label,X,NUM_GAUSSIANS,reg,N)
        vnew=v+update_v(v,lr,num,label,X,model.NUM_GAUSSIANS,reg,model.N)*dstep
        v , q , r = vnew, qnew ,rnew
        step+=dstep
        i+=1
        
    return OP

def main():
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--g", type=str,default="erf",
                        help="activation function for the student;"
                             "'erf'")
    parser.add_argument('-N', '--N', metavar='N', type=int, default=500,
                        help='number of inputs')
    parser.add_argument('-NG', '--NG', metavar='NG', type=int, default=1,
                        help="Number of Gaussians per cluster")
    parser.add_argument('-K', '--K', metavar='K', type=int, default=2,
                        help="size of the student's intermediate layer")
    parser.add_argument("--lr", type=float, default=.2,
                        help="learning constant")
    parser.add_argument("--dstep", type=float, default=None,
                        help="integration time step")
    parser.add_argument('-q', '--quiet', help="be quiet",
                        action="store_true")
    parser.add_argument('-both', '--both', help="Train both layers",
                        action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random number generator seed. Default=0")
    parser.add_argument("-NT", "--NT", type=int, default=int(1e4),
                        help="Number of tests to compute numerical error. Default=1e4")
    parser.add_argument("-NS", "--NS", type=int, default=int(1e6),
                    help="Number of samples to compute MontecarloIntegrals error. Default=1e6")
    parser.add_argument("-rho", "--rho", type=float, default=0.5,
                        help="Relative size of the cluaters. Default=0.5")
    parser.add_argument("-location", "--location", type=str, default="",
                        help="Where to save the data. Default="".")
    parser.add_argument("-comment", "--comment", type=str, default="",
                        help="Comment string to be added in name. Default=""")
    parser.add_argument('-steps', '--steps', type=float, default=int(1e5),
                        help="steps of simulations. Default 1e5")
    parser.add_argument('-reg', '--reg', type=float, default=0.0,
                        help="regularisation for weight decay")
    parser.add_argument('-oddeven','--oddeven', default=False, action='store_true',help="Tain GMM with only one even and 1 odd cluster")
    parser.add_argument('-norm', '--norm', help="normalise second layer weights",
                        action="store_true")
    parser.add_argument('-save', '--save', type=int , default=0,
                        help="Save initial conditions. Default 0")
    parser.add_argument('-F', '--F', type=int , default=3,
                        help="Which F to use : 0 F=sqrt(N)*I ,1: F=sqrt(N)*0.5*I,2: F=sqrt(N)*torch.randn(N)*I ")
    parser.add_argument('-prefix', '--prefix', type=str, default=None,
                        help="Initial conditions for the student weights. Default None")
    
    args = parser.parse_args()
    prefix=args.prefix
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    (N,rho,NUM_GAUSSIANS, K,lr) = (args.N,args.rho,args.NG, args.K, args.lr)
    (NUM_TESTS,steps,Nsamples)=(args.NT,args.steps,args.NS)
    
    log_fname = (args.location+"/GaussianMixtures_%s%s_NUMGAUS%d_rho%g_N%d_K%d_lr%g_reg%g_steps%g" %
                 (args.g,args.comment, args.NG , args.rho , args.N,args.K, args.lr, args.reg,args.steps))
    log_fname+="_seed%d"%args.seed
        ## Pay attention because I have 
    comment=args.comment
    mus = torch.zeros(2,NUM_GAUSSIANS,N)
    if args.prefix is None:
        Fs=torch.randn((N,N))/math.sqrt(N)
        mus=get_means(N=N,NUM_GAUSSIANS=NUM_GAUSSIANS)
    if args.save:
        torch.save(Fs, log_fname+'_Model_F.pt')
    if args.prefix is not None:
        fname=args.prefix+'_Model_F.pt'
        with open(fname, 'rb') as f:
            buffer = io.BytesIO(f.read())
        Fs=torch.load(buffer)
        fname=args.prefix+'_Model_mus.pt'
        with open(fname, 'rb') as f:
            buffer = io.BytesIO(f.read())
        mus=torch.load(buffer)
        comment=comment+"_Loaded"
    model=Model_AC(N=N,rho=rho,NUM_GAUSSIANS=NUM_GAUSSIANS,mus=mus,Fs=Fs,rotate=False)
    print(Fs[:3,:3])
    Masses=model.Masses.numpy()
    
    
    both=args.both;quiet=args.quiet;
    
    student=Student(K,N,act_function=args.g,w=None,v=None,bias=False)
    
    ###########Trains the model   
    steps=args.steps
    Nsamples=int(10**3)
    if args.prefix is None:
        OP_SIMU=train(model,student,Nsamples=Nsamples,lr=args.lr,steps=args.steps,ana=False,location=args.location,seed=args.seed,comment=args.comment)
        OP_SIMU = {k: [dic[k] for dic in OP_SIMU] for k in OP_SIMU[0]}
        OP_SIMU = [dict(zip(OP_SIMU,t)) for t in zip(*OP_SIMU.values())][0]
        print("Done Training")
    if args.prefix is not None:
        if path.exists(args.prefix+"_weights.pt"):
            print("Loading initial conditions from : "+args.prefix+"_weights.pt")
            student.load_state_dict(torch.load(args.prefix+"_weights.pt"))
        else: print("Did not find prefix file. Was looking for "+args.prefix+"_weights.pt"+"\n Am going to run with random initial conditions")
    
        R,Q,T,MT,M,r,q,t=get_RQT1T0M(model,student)
        v=student.fc2.weight.data[0].numpy() 
        R_SIMU=np.mean((r*Masses),axis=2)
        Q_SIMU=np.mean((q*Masses),axis=2)
        print("R_SIMU=")
        print(R_SIMU)
        print("QSIMU=")
        print(Q_SIMU)
        print("vsimu=")
        print(v)
        print("Tsimu=")
        print(T)
        OP_SIMU={"r":r,"q":q,"t":t,"v":v}
    
    
    
    OP_ODE={"r":[],"q":[],"t":[],"R":[],"M":[],"Q":[],"v":[],"step":[]}
    OP_ODE=evolve(model,student,steps=steps,lr=lr,OP_SIMU=OP_SIMU,OP=OP_ODE,Nsamples=Nsamples,comment=comment,location=args.location,reg=args.reg)
    
    print("Bye-bye")

if __name__ == '__main__':
    main()
