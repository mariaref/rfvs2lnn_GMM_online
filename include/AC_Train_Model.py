from GMM_Model import *
from Integrals_RELU import *
from ToolBox import *
import pickle

#######
def HalfMSE(output, target):
    loss = 0.5*torch.mean((output - target)**2)
    return loss

########### trains a 2LNN on a GMM with arbitrary covariance ############
def train(model,student,reg=0.0,Nsamples=int(10**4),NUM_TESTS=int(1e5),bs=1,lr=0.5,steps=int(1e5),quiet=False,both=True,ana=False,location="",seed=0,comment="",save_all_steps=False,OP=None,save_final_weights=False,save_final_OP=False, log_fname = None):
    if OP is None:
        OP=[]
    if ana:
      R,Q,T,MT,M,r,q,t=get_RQT1T0M(model,student,quiet=quiet)

    NUM_GAUSSIANS=model.NUM_GAUSSIANS
    rho=model.rho
    N=model.N
    K=student.K
    
    if not both:
        student.fc2.weight.requires_grad = False
    if not quiet and ana:
        print("Initial order parameters:\n R")
        print(R)
        print("Q")
        print(Q)
    device='cpu'
    
    student.to(device)
    
    print("Training two layer NN with N %d, K %d , NUM_TESTS %d, both: %d"%(N,student.K,NUM_TESTS,both))
    
    test_ys,test_xs=model.get_test_set(NUM_SAMPLES=NUM_TESTS)       
    
    ###############
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
    if log_fname is None:
        log_fname = (location+"GaussianMixtures_%s%s_NUMGAUS%d_rho%g_N%d_K%d_lr%g_reg%g_steps%g" %
                     (student.gname,comment, model.NUM_GAUSSIANS , model.rho , model.N,student.K, lr, reg,steps))
        log_fname+="_seed%d.dat"%seed
    print("saving in %s"%log_fname)
    
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Online learning with the teacher-student framework\n"
    welcome += ("# NUM_GAUSSIANS=%d, K=%d, lr=%g, reg=%g, seed=%d\n" % (model.NUM_GAUSSIANS, student.K, lr,reg, seed))
    print(welcome)
    logfile.write(welcome + '\n')
    
     ## MC set initially 0 
    X=np.zeros((2*NUM_GAUSSIANS,Nsamples,K+2*NUM_GAUSSIANS))
    C=np.zeros((2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS))
    mu=np.zeros((2*NUM_GAUSSIANS,K+2*NUM_GAUSSIANS))
    
    step = 0
    dstep = 1. / float(N)
    
    while step<=steps:
        student.eval()
        with torch.no_grad():
            if (save_final_OP and step>steps-dstep):
                R,Q,T,MT,M,r,q,t=get_RQT1T0M(model,student,quiet=True)
                v=student.fc2.weight.data[0].numpy().copy()
                OP.append({"q": q.copy(),"r": r.copy(),"t": t.copy(),"v": v.copy(),"step": step})
                
                
            if step >= steps_to_print[0].item() or step == 0 or save_all_steps:
                # compute the generalisation error w.r.t. the noiseless teacher
                preds = student(test_xs)
                ## implementing with halfMSEloss so do not need the factor 1/2
                eg =  HalfMSE(preds, test_ys)
                eg_class = 1-torch.relu(torch.sign(preds).squeeze()*test_ys.squeeze())
                eg_class = eg_class.sum()/float(preds.shape[0])
                if ana:
                    ## checks computing the mse analyticaly
                    R,Q,T,MT,M,r,q,t=get_RQT1T0M(model,student,quiet=True)
                    v=student.fc2.weight.data[0].numpy().copy()
                    OP.append({"q": q.copy(),"r": r.copy(),"t": t.copy(),"v": v.copy(),"step": step})
                    lplus=np.stack(  ( np.ones(NUM_GAUSSIANS),rho*np.ones(NUM_GAUSSIANS),1./len(model.mus[0])*np.ones(NUM_GAUSSIANS))   ,axis=-1)
                    lminus=np.stack(  (-1*np.ones(NUM_GAUSSIANS),(1-rho)*np.ones(NUM_GAUSSIANS),1./len(model.mus[1])*np.ones(NUM_GAUSSIANS) )  ,axis=-1)
                    label=np.concatenate((lminus,lplus), axis=0)
                    num=Integrals_GM(Nsamples=Nsamples,gname=student.gname,  dim= student.K + model.NUM_GAUSSIANS*2)
                    #X , C, mu =Generate_MC_Set(model,K,NUM_GAUSSIANS,R,M,Q,T,MT,num)
                    Update_MC_Set(model,K,NUM_GAUSSIANS,R,M,Q,T,MT,num,X , C, mu)
                    v=student.fc2.weight.data[0].numpy()
                    eg_ana=err_ana_ANA(R,Q,v,lr,num,label,X,C,mu)
                    msg = ("%g, %g,%g," % (step, eg,eg_ana))
                else:
                    msg = ("%g, %g,%g" % (step, eg,eg_class))
                    R,Q,T,MT,M,r,q,t=get_RQT1T0M(model,student,quiet=True)
                    v=student.fc2.weight.data[0].numpy().copy()
                    msg+=get_msg(Q,R,M,v,model.NUM_GAUSSIANS,student.K)
                
                logfile.write(msg + "\n")
                print(msg)
                if not save_all_steps:
                    steps_to_print.pop(0)
        
        # TRAINING
        student.train()
        targets,inputs=model.get_test_set(NUM_SAMPLES=bs)
        preds = student(inputs)
        loss = HalfMSE(preds, targets)
        student.zero_grad()
        loss.backward()
        optimizer.step()
        step += dstep
    ##SAVES FINAL WEIGHTS TO A FILE########
    if save_final_weights:
        weights_file=log_fname[:-4]+"_weights.pt"
        print("saving final weights in : \n "+weights_file)
        torch.save(student.state_dict(),weights_file)
    print("Bye-bye")
    
    if save_final_OP:
        OP_file=log_fname[:-4]+"_OP.p"
        print("saving final Order parameters in : \n "+OP_file)
        pickle.dump( OP[-1], open(OP_file, "wb" ) )
    return OP
