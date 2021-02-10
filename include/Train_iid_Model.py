from GMM_Model import *
from Integrals_RELU import *
from ToolBox import *

def HalfMSE(output, target):
    loss = 0.5*torch.mean((output - target)**2)
    return loss

########### Implements training of a 2LNN on a Gaussian mixture via online learning ############
def train(model,student,reg=0.0,Nsamples=int(4),NUM_TESTS=int(1e4),bs=1,lr=0.2,steps=int(1e5),quiet=False,both=True,ana=False,location="",seed=0,debeug_SGD=False,eps=1e-10,debeug_label=False,comment="",print_OP_fields=False):
    R,Q,T=get_RQT(model,student)
    NUM_GAUSSIANS=model.NUM_GAUSSIANS
    rho=model.rho
    var=model.var
    N=model.N
    K=student.K
    gname=student.gname
    
    if not both:
        student.fc2.weight.requires_grad = False
    device='cpu'
    
    student.to(device)
    
    N=model.N
    print("Training two layer NN with %s N %d, K %d , NUM_TESTS %d, both: %d"%(student.gname,N,student.K,NUM_TESTS,both))
    
    test_ys,test_xs=model.get_test_set(NUM_SAMPLES=NUM_TESTS)  
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
    # output file + welcome message
    if location!="": location+="/"
    log_fname = (location+"GMM_2lnn%s%s_%s_%s_NUMGAUS%d_rho%g_var%g_N%d_K%d_lr%g_reg%g" %
                 (comment,("_bias" if student.fc1.bias is not None else "" ),student.gname,comment, model.NUM_GAUSSIANS , model.rho , model.var, model.N,student.K, lr, reg))
    log_fname+="_seed%d.dat"%seed
    print("saving in %s"%log_fname)
    
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Online learning with the teacher-student framework\n"
    welcome += ("# NUM_GAUSSIANS=%d, K=%d, lr=%g, reg=%g, seed=%d\n" % (model.NUM_GAUSSIANS, student.K, lr,reg, seed))
    print(welcome)
    logfile.write(welcome + '\n')
    
    
    step = 0
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
                R,Q,T=get_RQT(model,student,quiet=quiet)
                v=student.fc2.weight.data[0].numpy()
                msg = ("%g, %g,%g" % (step, eg,eg_class))
                if not print_OP_fields:
                    msg_op = write_order_param({"Q": Q.copy(),"R": R.copy(),"v": v.copy()})
                elif print_OP_fields:
                    msg_op = order_param_msg(student,model)
                msg+=","+msg_op
                print(msg)
                logfile.write(msg + "\n")
                steps_to_print.pop(0)
        
        # TRAINING
        student.train()
        targets,inputs=model.get_test_set(NUM_SAMPLES=bs)
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
