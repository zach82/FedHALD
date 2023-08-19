# FedHALD: Personalized Models Meet Global Knowledge: Accommodating Heterogeneity in Federated Learning


## Requirments
This code requires the following:
* Python 3.6 or greater
* PyTorch 1.6 or greater
* Torchvision
* Numpy 1.18.5

## Data Preparation
* Download train and test datasets manually from the given links, or they will use the defalt links in torchvision:

[MNIST](http://yann.lecun.com/exdb/mnist/)

[CIFAR10, CIFAR100](http://www.cs.toronto.edu/∼kriz/cifar.html)
* Experiments are run on MNIST, CIFAR-10 and CIFAR-100.

## Quick Run Instructions:
* 1. Navigate to the 'FedHALD/scripts' directory using the terminal.
* 2. Locate the 'run.sh' script in that directory.
* 3. Copy the 'run.sh' file to the 'FedHALD/exps/' directory.
* 4. Set the executable permissions for the 'run.sh' script.
* 5. Run the 'run.sh' script using sudo.
*
* Commands to execute:
```
cd ./FedHALD/scripts             # Navigate to the scripts directory
cp run.sh ../exps/               # Copy the script to the exps directory
chmod +x ../exps/run.sh          # Set executable permissions
sudo ../exps/run.sh              # Run the script with sudo
```


## Running the experiments

* To train the FedHALD on MNIST with n=5, k=224 under both data and model heterogeneous setting:
```
python fedhald_main.py --dataset mnist --num_classes 10 --num_users 20 --ways 5 --shots 224 --optimizer sgd --local_bs 32 --stdev 2 --rounds 100 --ldc 0.9 --ld 0.9 --tau 0.07 --k 20
```

* To train the FedHALD on CIFAR10 with n=5, k=500 under both data and model heterogeneous setting:
```
python fedhald_main.py --dataset cifar10 --num_classes 10 --num_users 20 --ways 5 --shots 500 --optimizer sgd --local_bs 32 --stdev 2 --rounds 100 --ldc 0.9 --ld 0.9 --tau 0.07 --k 20
```

* To train the FedHALD on CIFAR100 with n=50, k=352 under both data and model heterogeneous setting:
```
python fedhald_main.py --dataset cifar100 --num_classes 100 --num_users 20 --ways 50 --shots 352 --optimizer sgd --local_bs 32 --stdev 20 --rounds 100 --ldc 0.9 --ld 0.9 --tau 0.07 --k 20
```

You can change the default values of other parameters to simulate different conditions. Refer to the options section.

## Options
The default values for various paramters parsed to the experiment are given in ```options.py```. Details are given some of those parameters:

* ```--dataset:```  Default: 'mnist'. Options: 'mnist', 'cifar10', 'cifar100'
* ```--num_classes:``` Number of classes included in the dataset. Default: 10. 
* ```--seed:```     Random Seed. Default set to 1234.
* ```--lr:```       Learning rate set to 0.01 by default.
* ```--local_bs:```  Local batch size set to 4 by default.
* ```--verbose:```  Detailed log outputs. Activated by default, set to 0 to deactivate.
* ```--num_users:```Number of users. Default is 20.
* ```--ways:```      Average number of local classes. Default is 3.
* ```--shots:```      Average number of samples for each local class. Default is 224.
* ```--test_shots:```      Average number of test samples for each local class. Default is 15.
* ```--stdev:```     Standard deviation. Default is 1.
* ```--train_ep:``` Number of local training epochs in each user. Default is 1.
* ```--tau:``` Temperature hyperparameter for contrastive learning.
* ```--ld:```  Regulation Parameter in Distillation.
* ```--ldc:``` Contrastive Learning Parameter.
* ```--k:``` Number of Samples in the Public Dataset.
* ```--optimizer:``` Type of optimizer.
* ```--rounds:``` Number of the global communication rounds.


## Acknowledgements
We built our project based on the foundational work found in [FedProto](https://github.com/yuetan031/FedProto). Our version includes modifications and enhancements tailored to our specific needs. We extend our gratitude to the original creators and maintainers of FedProto for their contribution. For anyone interested in the source, please explore the original [FedProto](https://github.com/yuetan031/FedProto).

