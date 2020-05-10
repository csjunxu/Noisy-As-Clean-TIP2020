# Noisy-As-Clean


# Install

- python = 3.6
- [pytorch](http://pytorch.org/) = 0.4
- numpy
- scipy
- matplotlib
- scikit-image

# Training
 To reproduce the results in the NAC paper:
 

 - Training and test for Set12 at the same time
```
 python nac_resnet_on_set12.py
```

# Testing

Pre-trained models for AWGN denoising on Set12 are provided in the results folder

 - Testing for Set12
 
 Modify the specific settings that need to test in test_nac_resnet_on_set12.py
 
```
 run test_nac_resnet_on_set12.py
```

#Further comments:
The code is mainly written by Yuan Huang.

Part of the code borrow from [[Deep Image Prior]](https://github.com/DmitryUlyanov/deep-image-prior)





