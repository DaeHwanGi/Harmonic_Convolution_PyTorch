# Harmonic Convolution

[Deep Audio Priors Emerge from Harmonic Convolutional Networks](http://dap.csail.mit.edu/) PyTorch Motivation Example (paper section 2.3) implementation.

**I failed to reproduce the motivation example results**

## Tested Environment

* Windows10 + WSL2(Ubuntu 20.04 LTS)
* Docker Image(pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel)
* Pytorch 1.6.0
* cuda 10.1
* DCNv2 (https://github.com/CharlesShang/DCNv2)


## Build
```console
$ cd ./DCNv2
$ python3 setup.py install
```

## To Do
- [ ] add argparser
- [ ] do audio restoration
- [ ] improve harmoniv covolution inference time

