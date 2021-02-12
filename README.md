# rfvs2lnn_GMM_online : Training two-layer neural networks (2LNN) and Random Features (RF) with online SGD on inputs sampled from a Mixture of Gaussian distribution

This package provides utilities to simulate learning in fully connected
two-layer neural networks (2LNN) and random features (RF) trained in the limit of one-pass/online SGD on inputs sampled from a mixture of Gaussian.
It further provides the code to obtain the analytical curves and assymptotic performances of both networks.

## Stucture

The data distribution is implemented using a class ```Model``` (see  ```include/GMM_Model.py ```).
The Model class contains the information on the means and covariances of the Gaussians in the mixture.
It has two subclasses, for Gaussian mixtures with identity covaraince or with random covariance which can be taken equal or different for each cluster in the mixture.
It has build-in functions allowing to sample data from the gaussian mixture.

## Training Two-layer Neural Networks (2LNN)

2LNN are implemented as a class ```Student``` (see  ```include/GMM_Model.py ```). 

***Simulations***

Simulating the training is performed using the packages python scripts ```Run_Training_Comitee_iid.py``` and ```Run_Training_AC.py```
The first is a simplified version of the second for which the Gaussians in the mixture have diagonal covariance matrix.
By specifying the ```--dataset``` argument (FMNIST or MNIST) in ```Run_Training_AC.py``` one has the obtion to use the Gaussian mixture generated by censoring these dataset.
An example of command for both is: 

```
python3 Run_Training_AC.py --location data_folder --dataset MNIST --K 4
```

```
python3 Run_Training_Comitee_iid.py --seed 0 --g relu --N 1000 --NG 2 --K 8 --lr 0.1 --both 1 --rho 0.5 --var 0.05 --location data_folder --steps 500000000 --reg 0.01 --init 0.1
```

***Analytical implementation***

Implementation of the ordinary differential equations (ODEs) that track the dynamical evolution of a 2LNN trained on a mixture of Gaussian. 

Many of the integrals appearing in the equations cannot be performed analytically, we perform them using Montecarlo techniques.

The class responsible for these integrations is the ```Integrator.py``` (see ```include/Integrals_RELU.py ```).
It must be initialised via specification of:

* ```act_function``` :  defining the activation function of the 2LNN\\
* ```Nsamples```     :  batchsize to perform integration (10000 samples is sufficient for precise evaluation)\\
* ```dim```          :  dimensions of the MC samples (=K + number Gaussian clusters in the case of the 2LNNs) 

Each time an ```Integrator``` is defined, it generates ```integrator.Nsamples``` i.i.d. random variables in  integrator.dim dimensions and stores them in ```integrator.vec```.
At each iteration of the equations, the montecarlo (MC) samples are then obtained by multiplying ```integrator.vec``` by the proper covariance matrix and adding it the proper mean.

An example of command to integrate the equations in the simplified i.i.d. case:

```python3 include/iid_Ode.py --N 1000 --K 4 --lr 0.1 --reg 0.01 --seed 0 --regime 1 --K 4 --g relu --var 0.05 --NS 10000 --prefix file_simulations.dat --NG 2 --comment _loremipsum --dstep 0.1```

An example for the case with arbitrary covariance:

```python3 Evolve_AC_ODE.py --NG 5 --location data_folder --g erf --N 784 --NS 10000 --comment _loremipsum --steps 1000000 --dstep 0.1 --K 3 --lr 0.1 --reg 0```

In both cases, one can integrate the ODEs from a given initial condition found via simulations by specyfying the ```--prefix``` option for example:

```python3 Evolve_AC_ODE.py --prefix simulations_file```

**Note**: in this case, one must take care to run the ODEs with the same parameters as those used in the simulations.

The fix point solver that allows to derive the asymptotic solution found by a 2LNN trained on the XOR via state evolution is implemented in ```include/iid_SE.py```. 
A usage example is: 

```python3 include/iid_SE.py --seed 0 --g relu --N 1000 --NG 2 --K 4 --lr 0.1 --NS 10000000 --rho 0.5 --var 1.0 --location data_folder --reg 1e-05 --comment _loremipsum --damp 0.6 --eps 1e-10```

All options are detailed in the help of the package.
Importantly, one can run only the reduced system of equations, imposing the constains on the fix point solution via the ```--reduced 1``` option.


## Training RF (2LNN)

***Simulations***

The implementation for training RF is similar to the one for 2LNN. The inputs in direct space are sampled from a ```Model```.
The function ```get_RF_test_set``` then applies the RF transformation on those inputs. The RF are implement as a K=1 ```Student``` with linear activation function and unit second layer weights.

Training can be done either using ```Run_Training_RF.py``` for identity covariance Gaussian clusters or via ```Run_Training_RF_fromDict.py``` for an arbitrary covariance model. 
The later requires specification of a dictionary where the means and covariances are stored. An example of usages is:

```python3  Run_Training_RF.py  --Psi relu --NG 2 --lr 0.1 --rho 0.5 --var 0.05 --location data_folder --steps 100000000```

***Analytical implementation***

```RF_analytical_error.py``` implements the computations for the assymnptotic performances of RF trained on the XOR-like mixture of Gaussians.
It contains functions both for the asymnptotic $\pmse$ and classification error. 
It also contains helper functions allowing to visualise the feature distribution after transformation. Please see additional help in the file.

Example:

```
python3 RF_analytical_error.py --D 800 --alpha 20 --regime 1 --sigma 10.0 --error class --three 0 --loc data_folder --comment _loremipsum
```

## Requirements

* python 3
* pytorch and other common python packages

## Plot Example

![](data_folder/weights_evolution.gif){width = "2px"}
