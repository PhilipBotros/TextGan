# SeqGAN-PyTorch
A implementation of SeqGAN in PyTorch, with extensions.


## Requirements:
* **PyTorch v0.1.12**
* Python 3.5-3.6
* CUDA 8.0+ (For GPU)

## Origin
The idea is from paper [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/pdf/1609.05473.pdf)

The code is rewrited in PyTorch with the structure largely from [Tensorflow Implementation](https://github.com/LantaoYu/SeqGAN)

## Pretrain
```
$ python pretrain.py
```
After running this file, pretrained models will be saved.

## Running
```
$ python main.py
```
After runing this file, the results will be printed on terminal. You can change the parameters in the ```main.py```.
