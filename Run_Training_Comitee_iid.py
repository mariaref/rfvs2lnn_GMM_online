import sys
import argparse
sys.path.insert(1, 'include')
from Train_iid_Model import *

####### Run training for the iid gaussian mixture #########
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

### FIle written in order to run just the training for varey long ##
def main(args):
    print(args)
    # read command line arguments
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    (N,var,rho,NUM_GAUSSIANS, K,lr) = (args.N,args.var,args.rho,args.NG, args.K, args.lr)
    (NUM_TESTS,steps,Nsamples)=(args.NT,args.steps,args.NS)
     
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
            mus=get_means(N=args.N,NUM_GAUSSIANS=NUM_GAUSSIANS,length=1.)
        elif args.regime==2 :
            mus=get_means(N=args.N,NUM_GAUSSIANS=NUM_GAUSSIANS,length=1.)
            mus*=math.sqrt(N)
        elif args.regime==3:
            mus=torch.zeros((2,NUM_GAUSSIANS,args.N))
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
    print(model.mus)
    print(1./N*model.mus.reshape(2*NUM_GAUSSIANS,N)@model.mus.reshape(2*NUM_GAUSSIANS,N).T)
    K=K;quiet=args.quiet;
    
    student=Student(K,N,act_function=args.g,w=None,v=None,bias=(args.bias==1))
    
    comment+="init%g_"%args.init
    student.fc1.weight.data*=args.init
    student.fc2.weight.data*=args.init
    log_fname = ("GMM%s%s_%s_init%g_N%d_K%d_lr%g_reg%g_seed%d" %
                     (comment, ("_bias" if args.bias==1 else ""),student.gname,args.init,N,student.K, args.lr, args.reg,args.seed))
    if args.informed==1:
        if args.K==4 and args.Three!=1:
            comment+="informed"
            mus=mus.reshape(2*args.NG,N)
            for i in range(student.K):
                student.fc1.weight.data[i]=mus[i]/mus[i].norm()*student.fc1.weight.data[i].norm()
        else: raise NotImplementedError
    
    if args.norm==1:
        print("setting second layer weights to big values")
        student.fc2.weight.data= torch.ones(student.fc2.weight.data.shape)
        print(student.fc2.weight.data)
    if args.save==1:
        print("Saving initial weights in %s"%log_fname)
        np.savetxt(args.location+"/"+log_fname+"_w.dat" ,student.fc1.weight.data.numpy(),delimiter=',')
        np.savetxt(args.location+"/"+log_fname+"_v.dat" ,student.fc2.weight.data.numpy(),delimiter=',')
    print(args.both)
    
    ###########Trains the model   
    train(model,student,Nsamples=Nsamples,NUM_TESTS=NUM_TESTS,bs=1,lr=lr,steps=steps,quiet=quiet,both=(args.both==1),ana=False,location=args.location,seed=args.seed,reg=args.reg,comment=comment,print_OP_fields=(args.print_OP==1))
    
    
    print("Bye-bye")

########## IN order to use submitit #######
if __name__ == '__main__':
  
  parser = argparse.ArgumentParser()
  
  parser.add_argument("-g", "--g", type=str,default="erf",
                  help="activation function for the student")
  parser.add_argument('-N', '--N', metavar='N', type=int, default=500,
                  help='number of inputs')
  parser.add_argument('-NG', '--NG', metavar='NG', type=int, default=1,
                  help="Number of Gaussians per cluster")
  parser.add_argument('-K', '--K', metavar='K', type=int, default=2,
                      help="size of the student's intermediate layer")
  parser.add_argument("--lr", type=float, default=.5,
                      help="learning constant")
  parser.add_argument('-q', '--quiet', help="be quiet",
                      action="store_true")
  parser.add_argument('-both', '--both',  type=int , default=1,help="Train both layers")
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
  parser.add_argument('-informed', '--informed', type=int , default=0,
                      help="Informed initialisation in the XOR model. Default 0")
  parser.add_argument('-Three', '--Three', type=int , default=0,
                      help="Use 3 clusters model. Default 0")
  parser.add_argument('-regime', '--regime', type=int , default=1,
                      help="snr regime in which to run training: (1) low snr (2) high snr (3) mixed snr")
  parser.add_argument('-bias','--bias', type=int , help="train with a bias", default=0)
  parser.add_argument('-print_OP','--print_OP', type=int , default=0)

  
  args = parser.parse_args()
  main(args)
