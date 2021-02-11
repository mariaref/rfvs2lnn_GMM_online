import sys
sys.path.insert(1,'include')
from RF_Train_Model import *
import argparse
import io
import os.path
from os import path
### FIle written in order to run just the training for varey long ##


#######get the means #########
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

def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)    
    K=1; both = False;
    (alpha,var,rho,NUM_GAUSSIANS,lr) = (args.alpha,args.var,args.rho,args.NG, args.lr)
    (NUM_TESTS,steps,Nsamples)=(args.NT,args.steps,args.NS)
    D=args.D
    N=alpha*D
    comment=args.comment
    comment+="_var%g"%args.var
    comment+="_regime%d"%args.regime
    if args.Three==1:
        comment+="3CLUST_"
        if args.regime==1:
            NUM_GAUSSIANS=2
            mus=torch.zeros(2,NUM_GAUSSIANS,D)
            mus[0,0,0]=math.sqrt(D)
            mus[0,1,0]=-math.sqrt(D)
        elif args.regime==2:
            NUM_GAUSSIANS=2
            mus=torch.zeros(2,NUM_GAUSSIANS,D)
            mus[0,0,0]=math.sqrt(D)
            mus[0,1,0]=-math.sqrt(D)
            mus*=math.sqrt(D)
        elif args.regime==3:
            NUM_GAUSSIANS=2
            mus=torch.zeros(2,NUM_GAUSSIANS,D)
            mus[0,0,0]=math.sqrt(D)
            mus[0,1,0]=-1
            mus*=math.sqrt(D)
        else: mus = None
    elif args.Three==0:
        if args.regime==1:
            mus=get_means(N=D,NUM_GAUSSIANS=NUM_GAUSSIANS,length=1.)
        elif args.regime==2 :
            mus=get_means(N=D,NUM_GAUSSIANS=NUM_GAUSSIANS,length=1.)
            mus*=math.sqrt(D)
        elif args.regime==3:
            mus=torch.zeros((2,NUM_GAUSSIANS,D))
            mus[0,0,0]=math.sqrt(D)
            mus[0,0,1]=math.sqrt(1)
            mus[0,1,0]=-math.sqrt(D)
            mus[0,1,1]=-math.sqrt(1)
            mus[1,0,0]=math.sqrt(D)
            mus[1,0,1]=-math.sqrt(1)
            mus[1,1,0]=-math.sqrt(D)
            mus[1,1,1]=math.sqrt(1) 
            mus*=math.sqrt(D)
        else: mus = None
        
    model=Model_iid(N=D,var=var,rho=rho,NUM_GAUSSIANS=NUM_GAUSSIANS,rand_MEANS=False,mus=mus)
   
    P=torch.randn(D,N)
    
    print("P is ")
    print(P[:5,:5])
    
    ###################################################
    Psi_name=args.Psi
    dstep=1./float(args.D) ## very important --> the ODES need to be run with 1/sqrt(D)!!
    
    comment=comment+"_Prandn"
    quiet=args.quiet;
    
    student=Student(K,D*args.alpha,act_function=args.g,w=None,v=None,bias=(args.bias==1))
    comment+="_init%g"%args.init
    student.fc1.weight.data*=args.init
    student.fc2.weight.data=torch.ones((1,1))
    student.fc2.weight.requires_grad=False
    
    train_desc="RF"#("NTK" if args.NTK==1 else "RF" )
    log_fname = ("%s%s%s_%s_%s_N%d_K%d_lr%g_reg%g_seed%d" %
                     (train_desc,("_bias" if args.bias==1 else ""),comment,Psi_name,student.gname,N,student.K, args.lr, args.reg,args.seed))
    if args.save:
        weights_file=args.location+"/"+log_fname+"_weights.pt"
        print("saving initial conditions in : \n "+weights_file)
        torch.save(student.state_dict(),weights_file)
        
   ###########Trains the model   
    train(model,student,P=P,Psi_name=Psi_name,Nsamples=Nsamples,NUM_TESTS=NUM_TESTS,bs=1,lr=lr,steps=steps,quiet=quiet,both=False,location=args.location,seed=args.seed,reg=args.reg,comment=comment,dstep=dstep)
    
    
    print("Bye-bye")

########## IN order to use submitit #######
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument("-g", "--g", type=str,default="lin",
                  help="activation function for the student")
  parser.add_argument("-Psi", "--Psi", type=str,default="relu",
                  help="activation function for the student")
  parser.add_argument('-D', '--D', metavar='D', type=int, default=300,
                  help='number of inputs')
  parser.add_argument('-alpha', '--alpha', metavar='alpha', type=int, default=1,
                  help='number of inputs')
  parser.add_argument('-NG', '--NG', metavar='NG', type=int, default=1,
                  help="Number of Gaussians per cluster")
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
  parser.add_argument("-rho", "--rho", type=float, default=0.5,
                      help="Relative size of the cluaters. Default=0.5")
  parser.add_argument("-var", "--var", type=float, default=0.01,
                      help="Variance within each cluster. Default=0.01")
  parser.add_argument("-location", "--location", type=str, default="data",
                      help="Where to save the data. Default=data.")
  parser.add_argument('-steps', '--steps', type=int, default=int(1e5),
                      help="steps of simulations. Default 1e5")
  parser.add_argument('-reg', '--reg', type=float, default=0.01,
                      help="regularisation for weight decay")
  parser.add_argument('-init', '--init', type=float, default=1.,
                      help="initialisation of the weights")
  parser.add_argument('-comment', '--comment', type=str, default="",
                      help="initialisation of the weights")
  parser.add_argument('-norm', '--norm',type=int, default=0, help="normalise second layer weights")
  parser.add_argument('-save', '--save', type=int , default=0,
                      help="Save initial conditions. Default 0")
  parser.add_argument('-Three', '--Three', type=int , default=0,
                      help="Use 3 clusters model. Default 0")
  parser.add_argument('-regime', '--regime', type=int , default=0,
                          help="snr regime in which to run training: (1) low snr (2) high snr (3) mixed snr")
  parser.add_argument('-bias','--bias', type=int , default=0)

  args = parser.parse_args()
  main(args)
