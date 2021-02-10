import sys
sys.path.insert(1,'include')
from AC_Ode import *
from DataSet_Generation import *
import argparse
import torch
import torchvision
import torchvision.datasets as datasets
from AC_Train_Model import *

### FIle written in order to run just the training for varey long ##
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    (N,var,rho,NUM_GAUSSIANS, K,lr) = (args.N,args.var,args.rho,args.NG, args.K, args.lr)
    (NUM_TESTS,steps,Nsamples)=(args.NT,args.steps,args.NS)
    reg=args.reg
    
    comment=args.comment
    if args.dataset=="mnist":
        N=784;
        NUM_GAUSSIANS=5
        comment=args.comment+"_MNIST"
        #####loads the dataset
        mnist_trainset= datasets.MNIST(root='~/datasets/'+'mnist', train=True, download=True, transform=None)
        if args.oddeven:
            NUM_GAUSSIANS=1;
            Fs,mus =get_MNIST_model_2CLUSTERS(mnist_trainset,dataset_name="mnist")
            comment=comment+"_oddeven"
        else:
            NUM_GAUSSIANS=5;
            ## Pay attention because I have 
            Fs,mus= get_MNIST_model(mnist_trainset,dataset_name="mnist")    
    elif args.dataset=="fmnist":
        N=784;
        NUM_GAUSSIANS=5
        comment=args.comment+"_FMNIST"
        #####loads the dataset
        fmnist_trainset = datasets.FashionMNIST(root='~/datasets/'+'fmnist', train=True, download=True, transform=None)
        if args.oddeven:
            NUM_GAUSSIANS=1;
            Fs,mus =get_MNIST_model_2CLUSTERS(fmnist_trainset,dataset_name="fmnist")
            comment=comment+"_oddeven"
        else:
            NUM_GAUSSIANS=5;
            ## Pay attention because I have 
            Fs,mus= get_MNIST_model(fmnist_trainset,dataset_name="fmnist")
    else:
        Fs=torch.randn((N,N))/math.sqrt(N)
        comment=comment+"_Fran"
        if args.randMus==1: 
            print('setting random means')
            mus = torch.randn(2,NUM_GAUSSIANS,N)
            comment+="_Muran"
        else: mus = None
    
    model=Model_AC(N=N,rho=rho,NUM_GAUSSIANS=NUM_GAUSSIANS,mus=mus,Fs=Fs,rotate=False)
    
    if args.location!="": location =args.location+"/"
    else: location= args.location
    log_fname = (location+"GMM_2lnn_%s%s_NUMGAUS%d_rho%g_N%d_K%d_lr%g_reg%g_steps%g_seed%d.dat" %
                     (args.g,comment, NUM_GAUSSIANS , args.rho ,N,args.K, args.lr, reg,args.steps,args.seed))
    if args.save:
        torch.save(Fs, log_fname[:-4]+'_Model_F.pt')
        torch.save(model.mus, log_fname[:-4]+'_Model_mus.pt')
        
    
    print(Fs[:3,:3])
    K=K;both=args.both;quiet=args.quiet;
    
    student=Student(K,N,act_function=args.g,w=None,v=None,bias=False)
    
    if args.norm:
        print("setting second layer weights to big values")
        student.fc2.weight.data=(0.5 - 1.0) * torch.rand(student.fc2.weight.data.shape).data + 1.0
        print(student.fc2.weight.data)
    if args.save==1:
        weights_file=log_fname[:-4]+"_weights.pt"
        print("saving initial conditions in : \n "+weights_file)
        torch.save(student.state_dict(),weights_file)
        
    if args.prefix is not None:
        if path.exists(args.prefix+"_weights.pt"):
            print("Loading initial conditions from : "+args.prefix+"_weights.pt")
            student.load_state_dict(torch.load(args.prefix+"_weights.pt"))
        else: print("Did not find prefix file. Was looking for "+args.prefix+"_weights.pt"+"\n Am going to run with random initial conditions")
        
   
    ###########Trains the model   
    OP=train(model,student,Nsamples=Nsamples,NUM_TESTS=args.NT,lr=args.lr,steps=args.steps,ana=False,location=args.location,seed=args.seed,comment=comment,reg=args.reg,log_fname = log_fname)
    
    
    print("Bye-bye")

if __name__ == '__main__':
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
    parser.add_argument("--lr", type=float, default=.5,
                        help="learning constant")
    parser.add_argument("--reg", type=float, default=0.,
                        help="weight decay")
    parser.add_argument('-q', '--quiet', help="be quiet",
                        action="store_true")
    parser.add_argument('-both', '--both', help="Train both layers",
                        action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random number generator seed. Default=0")
    parser.add_argument("-NT", "--NT", type=int, default=int(1e5),
                        help="Number of tests to compute numerical error. Default=1e4")
    parser.add_argument("-NS", "--NS", type=int, default=int(1e6),
                    help="Number of samples to compute MontecarloIntegrals error. Default=1e6")
    parser.add_argument("-rho", "--rho", type=float, default=0.5,
                        help="Relative size of the cluaters. Default=0.5")
    parser.add_argument("-var", "--var", type=float, default=0.01,
                        help="Variance within each cluster. Default=0.01")
    parser.add_argument("-location", "--location", type=str, default="",
                        help="Where to save the data. Default="".")
    parser.add_argument('-steps', '--steps', type=float, default=int(1e3),
                        help="steps of simulations. Default 1e3")
    parser.add_argument("-comment", "--comment", type=str, default="",
                        help="Comment string to be added in name. Default=""")
    parser.add_argument('-norm', '--norm', type=int , default=0)
    
    parser.add_argument('-save', '--save', type=int , default=0,
                        help="Save initial conditions. Default 0")
    parser.add_argument('-dataset', '--dataset', type=str , default="mnist",
                        help="MNIST")
    parser.add_argument('-randMus', '--randMus', type=int , default=0,
                        help="randMus")
    parser.add_argument('-oddeven','--oddeven', default=False, action='store_true',help="Tain GMM with only one even and 1 odd cluster")
    parser.add_argument('-bias','--bias', default=False, action='store_true')
    parser.add_argument('-prefix', '--prefix', type=str, default=None,
                        help="Initial conditions for the student weights. Default None")
    args = parser.parse_args()
    main(args)