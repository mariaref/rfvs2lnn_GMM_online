from GMM_Model import *
from Integrals_RELU import *
from ToolBox import *
## implements training with random features online:
## at each time step, sample an input from a Gaussian mixture -> apply the RF transformation -> perform linear regression on the features 

def get_rf_test_set(model,P,Psi,NUM_SAMPLES=1, l=None,ran=None):
    """dataset sampled from a gaussian mixture model and applied random features transformation:
         - P : projection matrix of the random features
         - NUM_SAMPLES: number of samples in the dataset
         - Psi : activation function of the random features
         - l : the label corresponding to the inputs
               if None, the label will be selected randomly for each sample in the dataset
         - ran : from which Gaussian cluster to draw the inputs X
                 if None, the cluster is selected randomly, given the label, for each input X"""
    l,C=model.get_test_set(NUM_SAMPLES=NUM_SAMPLES)
    D,N=P.shape
    X=Psi(C@P/math.sqrt(D),0)
    return l,X


########### TRAINING ############
def train(model,student,P,Psi_name,reg=0.0,Nsamples=int(10**4),NUM_TESTS=int(1e4),bs=1,lr=0.1,steps=int(1e5),quiet=False,both=True,location="",seed=0,comment="",dstep=None):
    
    
    NUM_GAUSSIANS=model.NUM_GAUSSIANS
    rho=model.rho
    N=model.N
    K=student.K
    gname=student.gname
    if Psi_name=="erf":
        Psi=erf
    if Psi_name =="relu":
        Psi=centered_relu
    if Psi_name=="lin":
        Psi=linear
    
    if not both:
        student.fc2.weight.requires_grad = False
    device='cpu'
    
    student.to(device)
    
    D,N=P.shape
    print("Training two layer NN with orijection with %s  %s D %d N %d, K %d , NUM_TESTS %d, both: %d"%(student.gname,Psi_name,D,N,student.K,NUM_TESTS,both))
    
    test_ys,test_xs=get_rf_test_set(model,P,Psi,NUM_SAMPLES=NUM_TESTS)
    print(" the mean norm is %g"%torch.mean(torch.norm(test_xs,dim=1)))

    # collect the parameters that are going to be optimised by SGD
    params = []
    params += [{'params': student.fc1.parameters()}]
    # If we train the last layer, ensure its learning rate scales correctly
    params += [{'params': student.fc2.parameters(),
                'lr': lr / N,
                'weight_decay':reg}]
    
    ##weight_decay=reg adds the L2 regularisation
    optimizer = optim.SGD(params, lr=lr, weight_decay=reg/float(N))
    criterion = student.loss
    
    print("STUDENT: ")
    print(student)
    for param in student.parameters():
        print(param)
        
    # when to print?
    end = torch.log10(torch.tensor([1. * steps])).item()
    steps_to_print = list(torch.logspace(-2, end, steps=100))
    print("I am going to print for %d steps"%len(steps_to_print))    
    if location!="": location+="/"
    # output file + welcome message
    log_fname = (location+"RF%s%s_%s_%s_NUMGAUS%d_D%d_N%d_K%d_lr%g_reg%g" %
    (comment, ("_bias" if student.fc1.bias is not None else "" ),Psi_name,student.gname, model.NUM_GAUSSIANS, D,N,student.K, lr, reg))
    
    log_fname+="_seed%d.dat"%seed
    print("saving in %s"%log_fname)
    
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Online learning with the teacher-student framework\n"
    welcome += ("# NUM_GAUSSIANS=%d, K=%d, lr=%g, reg=%g, seed=%d\n" % (model.NUM_GAUSSIANS, student.K, lr,reg, seed))
    print(welcome)
    logfile.write(welcome + '\n')
    
    step = 0
    if dstep is None:
        dstep = 1. / float(N)
    
    while step<=steps:
        if step >= steps_to_print[0].item() or step == 0:
            student.eval()
            with torch.no_grad():
                # compute the generalisation error w.r.t. the noiseless teacher
                preds = student(test_xs)
                ## implementing with halfMSEloss so do not need the factor 1/2
                eg =  HalfMSE(preds, test_ys)
                eg_class = 1-torch.relu(torch.sign(preds).squeeze()*test_ys.squeeze())
                eg_class = eg_class.sum()/float(preds.shape[0])
                v=student.fc2.weight.data[0].numpy()
                msg = ("%g, %g,%g" % (step, eg,eg_class))
                for k in range(K):
                    msg+=",%g"%v[k]
                print(msg)
                logfile.write(msg + "\n")
                steps_to_print.pop(0)
        
        # TRAINING
        student.train()
        targets,inputs=get_rf_test_set(model,P,Psi,NUM_SAMPLES=bs)
        preds = student(inputs)
        loss = HalfMSE(preds, targets)
        student.zero_grad()
        loss.backward()
        ############
        optimizer.step()
        #############
        step += dstep
    print("Bye-bye")
    return log_fname
