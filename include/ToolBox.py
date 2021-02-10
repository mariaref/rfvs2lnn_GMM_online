from GMM_Model import *
## ToolBox package that implements various usefull functions for our computations

################################## USEFULL FOR AC ##########
def get_msg(Q,R,M,v,NUM_GAUSSIANS,K):
    """writes the order parameters in a .dat file"""
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
###################
def HalfMSE(output, target):
    """Hald mse loss used in the optimisation step"""
    loss = 0.5*torch.mean((output - target)**2)
    return loss
#######
def Derf_torch(x):
    """derivative of the erf function"""
    return math.sqrt(2./math.pi)*torch.exp(-x**2/2.)

def reshape_inv(vec,K,NG):
    """from an array computes the order parameters, very usefull to obtain them from .dat file"""
    Q=np.zeros((K,K))
    R=np.zeros((2*NG,K))
    v=np.zeros(K)
    i=0
    for k in range(K):
        for l in range(k+1):
            Q[k,l]=vec[i]
            if l!=k:
                Q[l,k]=Q[k,l]
            i+=1
    for beta in range(2*NG):
        for k in range(K):
            R[beta,k]=vec[i]
            i+=1
    for k in range(K):
        v[k]=vec[i]
        i+=1
    return Q,R,v

#######AUXILIARY TO WRITE for IID ######
def write_order_param(order_param):
    """from a dictionary of order parameters returns a vector containing their values, is usefull for printing and initial conditions"""
    K=len(order_param["v"])
    NUM_GAUSSIANS=int(len(order_param["R"])/2.)
    msg=""
    for k in range(K):
        for l in range(k+1):
            msg+=str(order_param["Q"][k,l])+","
    for alpha in range(2*NUM_GAUSSIANS):
        for k in range(K):
            msg+=str(order_param["R"][alpha,k])+","
    for k in range(K):
        msg+=str(order_param["v"][k])+","
    msg=msg[:-1]
    #msg+="\n"
    return msg

##############
def order_param_msg(student,model,all_same=False):
    """prints the order parameters from the local fields:
       - all_same : shortens the message in the case where all the covariances are equal """
    v , M , T0 , Q , R , T1 = get_OP_fields(student,model)
    _ , NG2, K = R.shape
    if all_same:
        print('printing the average values of the second moments')
    msg=''
    for k in range(K):
        msg+='%g,'%v[k]
    for cluster in range(NG2):
        ### prints M
        for m in M[cluster]:
            msg+='%g,'%m
        ### prints T0
        for m in T0[cluster]:
            msg+='%g,'%m
        ### prints Q
        for row in Q[cluster]:
            for q in row:
                msg+='%g,'%q
        ### prints R
        for row in R[cluster]:
            for q in row:
                msg+='%g,'%q
        ### prints T1
        for row in T1[cluster]:
            for q in row:
                msg+='%g,'%q
    msg=msg[:-1]
    return msg
                
    
def get_OP_fields(student,model):
    """computes the order parameters from the local fields"""
    N=model.N
    K=student.K
    NG=model.NUM_GAUSSIANS
    bs = int(1e4)
    ##
    w=student.fc1.weight.data.clone().detach()
    v=student.fc2.weight.data.clone().detach()[0]
    mus=model.mus.view(2*NG,N)
    
    ###
    xs  = torch.zeros(2 , NG , bs, N)
    las = torch.zeros(2 , NG , bs, K)
    nus = torch.zeros(2 , NG , bs, 2*NG)
    
    ###
    M  = torch.zeros(2 ,NG ,K   )
    T0 = torch.zeros(2 ,NG ,2*NG)
    Q  = torch.zeros(2 ,NG ,K    ,K   )
    R  = torch.zeros(2 ,NG ,2*NG ,K   )
    T1 = torch.zeros(2 ,NG ,2*NG ,2*NG)
    
    ## Attention: the averages ar always defined per cluster! 
    for sign in range(2):
        l=torch.ones(bs,1)*(2*sign-1)
        for cluster in range(NG):
            ran=torch.ones(bs)*cluster
            xs[sign,cluster] = model.get_X_from_label(bs,l,ran=ran)
            las[sign,cluster] = ( xs[sign,cluster] @ w.t() )/math.sqrt(N) ## bs x K
            nus[sign,cluster] = ( xs[sign,cluster] @ mus.t())/math.sqrt(N) ## bs x 2 NG
            M[sign,cluster]  = las[sign,cluster].mean(dim=0)
            T0[sign,cluster] = nus[sign,cluster].mean(dim=0)
            Q[sign,cluster]  = (las[sign,cluster] - M[sign,cluster]).t() @(las[sign,cluster] - M[sign,cluster]) / float(bs) ## K x K
            R[sign,cluster]  = (nus[sign,cluster] - T0[sign,cluster]).t()@(las[sign,cluster] - M[sign,cluster]) / float(bs) ## 2 NG x K
            T1[sign,cluster] = (nus[sign,cluster] - T0[sign,cluster]).t()@(nus[sign,cluster] - T0[sign,cluster]) / float(bs) ## 2 NG x 2 NG
    return v , M.view(2*NG,K) , T0.view(2*NG,2*NG) , Q.view(2*NG,K,K) , R.view(2*NG,2*NG,K) , T1.view(2*NG,2*NG,2*NG)
    





