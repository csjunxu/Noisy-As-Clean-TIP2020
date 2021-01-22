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

The test is performed after the models are trained. 

If you have pre-trained models, for AWGN denoising on Set12, you can modify the specific settings that need to test in test_nac_resnet_on_set12.py
 
```
 run test_nac_resnet_on_set12.py
```

# Draw Figrues
The code for drawing the figures in ablation study is provided in the figures folder

```
 run plot_for_large_sigma.py
```


#Further comments:
The code is mainly written by Yuan Huang.

Part of the code borrow from [[Deep Image Prior]](https://github.com/DmitryUlyanov/deep-image-prior)





