import sys
sys.path.insert(1,'include')
from NTK_Train_Model import *
import argparse
import io
import os.path
from os import path
### FIle written in order to run just the training for varey long ##

####### with different covariance matrices #########
def get_GMM_Model(dict_,NSamples=int(10**5.5)):
    print("GENERATING MODEL From Input file")
    mus=dict_["mus"]
    covs=dict_["covs"]
    NG=int(len(mus)/2)
    N=len(mus[0])
    mus=mus.reshape(2,NG,N)
    Fs=torch.zeros(2*NG,N,N)
    for n in range(len(covs)):
      evals,evecs=torch.symeig(covs[n], eigenvectors=True)
      evals+=1.1*torch.abs(torch.min(evals))
      Fs[n]=evecs@torch.diag(torch.sqrt(evals))@evecs.T
    Fs=Fs.reshape(2,NG,N,N)
    model=Model_AC_MULTIPLE(N=N,NUM_GAUSSIANS=NG,mus=mus*math.sqrt(N),Fs=Fs)
    return model
    
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)    
    fname=args.dict
    dict_=pickle.load( open(fname, "rb" ) )
    D=len(dict_["mus"][0])
    (alpha,lr,K) = (args.alpha, args.lr,args.K)
    (NUM_TESTS,steps,Nsamples)=(args.NT,args.steps,args.NS)
    N=alpha*D
    comment=args.comment
    if args.bias==0: bias=False
    else: bias=True
    if args.both==0: both=False
    else: both=True
    
    model=get_GMM_Model(dict_)
    P=torch.randn(D,N)
    print("P is ")
    print(P[:5,:5])
    ###################################################
    Psi_name=args.Psi
    dstep=1./float(D)
    comment=comment+"_Prandn"
    K=K;both=args.both;quiet=args.quiet;
    
    student=Student(K,N,act_function=args.g,w=None,v=None,bias=args.bias)
    
    comment+="_init%g"%args.init
    student.fc1.weight.data*=args.init
    student.fc2.weight.data=torch.ones((1,1))
    log_fname = ("RF%s_%s_%s_N%d_K%d_lr%g_reg%g_seed%d" %
                     (comment,Psi_name,student.gname,N,student.K, args.lr, args.reg,args.seed))
    if args.save:
        weights_file=args.location+"/"+log_fname+"_weights.pt"
        print("saving initial conditions in : \n "+weights_file)
        torch.save(student.state_dict(),weights_file)
        
   ###########Trains the model   
    train(model,student,P=P,Psi_name=Psi_name,Nsamples=Nsamples,NUM_TESTS=NUM_TESTS,bs=1,lr=lr,steps=steps,quiet=quiet,both=False,location=args.location,seed=args.seed,reg=args.reg,comment=comment,dstep=dstep,model_iid=False)
    print("Bye-bye")

########## IN order to use submitit #######
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument("-g", "--g", type=str,default="lin",
                  help="activation function for the student")
  parser.add_argument("-Psi", "--Psi", type=str,default="relu",
                  help="activation function for the student")
  
  parser.add_argument('-alpha', '--alpha', metavar='alpha', type=int, default=1,
                  help='number of inputs')
  
  parser.add_argument('-K', '--K', metavar='K', type=int, default=1,
                      help="size of the student's intermediate layer")
  parser.add_argument("--lr", type=float, default=.5,
                      help="learning constant")
  parser.add_argument('-q', '--quiet', help="be quiet",
                      action="store_true")
  parser.add_argument('-both', '--both',  type=int , default=0,help="Train both layers")
  parser.add_argument("-s", "--seed", type=int, default=0,
                      help="random number generator seed. Default=0")
  parser.add_argument("-NT", "--NT", type=int, default=int(1e4),
                      help="Number of tests to compute numerical error. Default=1e4")
  parser.add_argument("-NS", "--NS", type=int, default=int(1e6),
                  help="Number of samples to compute MontecarloIntegrals error. Default=1e6")
  parser.add_argument("-location", "--location", type=str, default="",
                      help="Where to save the data. Default=data.")
  parser.add_argument('-steps', '--steps', type=int, default=int(1e5),
                      help="steps of simulations. Default 1e5")
  parser.add_argument('-reg', '--reg', type=float, default=0.01,
                      help="regularisation for weight decay")
  parser.add_argument('-init', '--init', type=float, default=1.,
                      help="initialisation of the weights")
  parser.add_argument('-comment', '--comment', type=str, default="",
                      help="initialisation of the weights")
  parser.add_argument('-dict', '--dict', type=str, default="",
                    help="dictionary where mus and covs are stores")
  parser.add_argument('-norm', '--norm',type=int, default=0, help="normalise second layer weights")
  parser.add_argument('-save', '--save', type=int , default=0,
                      help="Save initial conditions. Default 0")
  parser.add_argument('-bias','--bias', type=int , default=0)
  
  args = parser.parse_args()
  main(args)
