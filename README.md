# GAN'S APPLIED TO MNIS DATASET

## Summary

Educational application of a fully connected GAN destinated to produce hand-written digits.

## Dependencies (Tested on python 3.8.2)

```bash
cycler==0.10.0
future==0.18.2
idx2numpy==1.2.2
kiwisolver==1.2.0
matplotlib==3.2.1
numpy==1.18.4
Pillow==7.1.2
pyparsing==2.4.7
python-dateutil==2.8.1
six==1.14.0
torch==1.5.0+cpu
torchvision==0.6.0+cpu
```

## Instruction

### Training

Running

```bash
python main.py
```

Will prompt inputs to specify the training parameters

```
Batch size:
Epochs:
Total imgs:
```

If no values are inserted, the default will afterwards

```
1 120 None
```

Finalizing the training, `outputs` folder will be created, cotaining both the Generator and Discriminator nets, besides the showcase data evaluated during the training (`data.p`), and the error data (`error.p`), both pickle serialized.

### Showing

Running

```
python Show.py
```

will ask for the path of the folder where the `data.p` is located. During execution, `.png` will be exported to an inner folder `imgs/` where each frame can be later used in an animation. (Using external tool as `convert`)

### Examples

#### Trained using 15k images

![15k data training process](https://github.com/luislve17/GAN-MNIST/blob/master/imgs/animation_15k_out.gif?raw=true)
