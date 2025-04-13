# FibroSoft Python Version

> (c) 2022-2025 by Prof. Dr. Markus Graf (Contact [markus.graf@hs-heilbronn.de](mailto:markus.graf@hs-heilbronn.de))
> 
> free for private und non-commercial use
> 

This software is provided **as-is**; without any warranty or liability issues.   

***WARNING:***  
It is not suitable/applicable for diagnostics or therapeutic use.   
Use at own risk.

For more information see license.md

For scientific use & citation please come back at a later point - resp. contact me in the meantime
markus.graf@hs-heilbronn.de

```
Modification history  
version   author  changes  
- 0.9.8     gm      added dynamic kmeans on all slices  
- 0.9.9     gm      added UNet convolutional neural network ml approach  
                    included helper.py with helper classes in order to prepare 
                    labels and files to be trained  
- 0.10.0    gm      improved uNet learning phases, auto-store model
- 0.11.0    gm      added multithreading for the training process
- 0.11.1    gm      utility functions (tiles, create 256x256 for faster calcs)
- 0.12.0    gm      added various optimizers to choose from, learning rate patience set to 20
- 0.12.1    gm      added 1by1 convolution UNet as UNetBW 
- 0.13.0    gm      added UNetGray + UNetGrayProcessing and renaming the BW-UNet to UNet1by1Processing       
- 0.14.0    gm      remodelled the threading by the use of Queue for result value transferral
                    added command line interface UNet, UNet1by1, and UNetGray (--network option)
                    UI improvements: enable/disable parameter boxes accordingly to the selected method
- 0.14.1    gm      several bugfixes, modified postprocessing, fixed 0..1 (instead of 0..1/255) ranges
                    as torchvision.transforms.ToTensor automatically normalizes it
                    and Ftrans.toPilImage denormalizes it automatically
                    Also: postprocessing best_class now uses argmax
                          and linear uses threshold. rgb just stays as predicted
                    Removed multithreading issue (passing via queue didn't solve it)
- 0.16.2    gm      Early stopping included, last 50 values are counted (and checked if we are above the 
                    mean/average for more than xx times (default 50)
- 0.17.0    gm      UserInterface allows setting hyperparameters for dnns while being intialized
                    additionally, you can specify an early.stop file in the root folder manually in order to
                    stop the process
                    >> INFO: 
                    >> In order to stop manually you can place a file called
                    >>    early.stop    <<
                    >> into the root directory of the software. Next epoch it stops
- 0.18.0    gm      added "experiment" handling; an experiment name will be used for logging and saving related data.
- 0.18.1    gm      some gui tweaks/workflow: renamed buttons, enable and disable of controls
- 0.18.2    gm      accuracy calculations done for U-Net based MLs in batch execution mode                    
- 0.18.3    gm      accuracy calculations + iou in fibrounet_run client console command      
- 0.19.0    gm      Loading of resource file (config) adapted for macOS specific one-file .app        
- 1.0.0     gm      first Release (0.19 with some minor adjustments)      
```

# Command line tool for UNet
Command line starting the UNet training with a folder structure

```
+-images
  +- train      # containing training images
  +- labels     # containing corresponding labels to the images above
```

Run default UNet first time using *Adam* optimizer, a learning rate of *4.5e-5* and train for *200* epochs
```
python fibrounet_cli.py /home/user/data/folder/train --lr 4.5e-5 --optimizer Adam --epochs 200 
```

Transfer-learning using previously obtained model *models/model_1.phy*, applying *Adam* optimizer, a learning rate of *4.5e-5* and train for *200* epochs
```
python fibrounet_cli.py /Users/monk/Documents/Research/ScientificData/sirius_red_h2_small/train --lr 4.5e-5 --optimizer Adam --epochs 20 --model models/model_1.phy
```

Execute grayscaling before training (UNetGray Network) applying *Adam* optimizer, a learning rate of *4.5e-5* and train for *200* epochs
```
python fibrounet_cli.py /Users/monk/Documents/Research/ScientificData/sirius_red_h2_small/train --network UNetGray --lr 4.5e-5 --optimizer Adam --epochs 20 --model models/model_1.phy
```

#### Parameters are as follows: ####
> path to folder with training images  
> *Important: must have a corresponding labels-folder, see folder structure shown above*

```
--lr <Learning rate> :      
   a float value describing the learning rate   
--optimizer <name> :       
   optimizer to be used (SGD, SGDMomentum, Adam, AdaGrad, Adamax, RMSProp)  
--epochs <number> :        
   integer value defining number of epochs  
--model <model filename> :   
   a network-corresponding .phy model that should be loaded in the beginning  
--network <neural network type> :   
   choose from UNet (default), UNet1by1, UNetGray  
--visualize_linear <yes|no> :  
   no (default) does real segmentation
   yes is for debugging of the output (returns probability map instead)
--experiment <name> :    
   name of the experiment; will be used in filenames for documentation, logs, ...
```

> ***Network model descriptions***  
> *UNet*:   
> Standard U-Net implementation   
>
> *UNet1by1*:   
> The data is put through a 1by1 Convolution filter in order to reduce dimensionality from 3 to 1 channels
> while still keeping the potential feature set    
>
> *UNetGray*:   
> Adding a gray scaling transformation beforehand, therefore, reducing dimensionality with also expecting
> some loss in features (as color information will just be discarded)
