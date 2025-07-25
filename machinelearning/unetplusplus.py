
import os
import time

import numpy as np

from machinelearning import mlprocessing

from machinelearning.unet import (EarlyStopper, EARLY_STOP_PATIENCE, UNetProcessing, NUM_EPOCHS, INIT_LR, BATCH_SIZE,
                  SyntheticTrainingDataset, UnsupervisedMaskedTrainingDataset, SegmentationDataset)

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms.functional as Ftrans
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Dropout2d(0.5),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.depth = len(features)
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        for feature in features:
            self.down.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # Decoder (nested)
        self.up = nn.ModuleDict()
        for i in range(1, self.depth):
            for j in range(self.depth - i):
                key = f"X{j}_{i}"
                in_ch = features[j] * i + features[j+1]
                self.up[key] = ConvBlock(in_ch, features[j])

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        enc = []
        for down in self.down:
            x = down(x)
            enc.append(x)
            x = self.pool(x)

        # Nested decoder
        X = dict()
        for i in range(self.depth):
            X[f"X{i}_0"] = enc[i]

        for i in range(1, self.depth):
            for j in range(self.depth - i):
                upsampled = F.interpolate(X[f"X{j+1}_{i-1}"], scale_factor=2, mode='bilinear', align_corners=True)
                concat = torch.cat([X[f"X{j}_{k}"] for k in range(i)] + [upsampled], dim=1)
                X[f"X{j}_{i}"] = self.up[f"X{j}_{i}"](concat)       #

        return self.final(X["X0_{}".format(self.depth - 1)])


class UNetPlusPlusProcessing(UNetProcessing):
    def __init__(self, classes=3, epochs=NUM_EPOCHS, learning_rate=INIT_LR, optimizer='SGD', train_on_gpu=True,
                 visualize_linear=False, unsupervised=False, synthetic_data=False):
        super().__init__(classes, epochs, learning_rate, optimizer, train_on_gpu, visualize_linear, unsupervised, synthetic_data)
        self.model = UNetPlusPlus(classes, classes, [32, 64, 128, 256, 512])
        self.model.to(self.device)


    def train(self):
        print("Training")
        self._init_training()
        self._init_optimizer()

        early_stopper = EarlyStopper(EARLY_STOP_PATIENCE)

        # calculate steps per epoch for training and test set
        trainSteps = len(self.training_dataset)
        testSteps = len(self.test_dataset)
        # initialize a dictionary to store training history
        self.Hist = {"train_loss": [], "test_loss": []}

        # loop over epochs
        print("[INFO] training the network...")
        self.log("[INFO] training the network...")
        self.log("[INFO] Learning rate is set to {}\n".format(self.learning_rate))
        startTime = time.time()

        for e in range(self.epochs):     #tqdm( ... for displaying)
            # set the model in training mode
            self.model.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalTestLoss = 0
            # loop over the training set
            for (i, (x, y)) in enumerate(self.training_loader):
                # send the input to the device
                (x, y) = (x.to(self.device), y.to(self.device))
                # perform a forward pass and calculate the training loss
                pred = self.model(x)
                loss = self.loss_function[0](pred, y)+self.loss_function[1](pred, y)
                # first, zero out any previously accumulated gradients, then
                # perform backpropagation, and then update model parameters
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # add the loss to the total training loss so far
                totalTrainLoss += loss
            # switch off autograd
            with torch.no_grad():
                # set the model in evaluation mode
                self.model.eval()
                # loop over the validation set
                for (x, y) in self.test_loader:
                    # print(f"Shapes of input and output: {x.shape} | {y.shape}")
                    # send the input to the device
                    (x, y) = (x.to(self.device), y.to(self.device))
                    # make the predictions and calculate the validation loss
                    pred = self.model(x)
                    testloss = self.loss_function[0](pred, y)+self.loss_function[1](pred, y)
                    totalTestLoss += testloss

                    if (e % 100) == 0 or e == self.epochs-1 or self.epochs < 50:  # random.randint(0, 100) > 90:
                        res = pred[0]
                        # res = torch.sigmoid(res) # run through sigmoid function
                        res = torch.nn.functional.softmax(res, dim=0)  # or run through softmax (non different.)
                        res = res.cpu().detach().numpy()  # squeeze(0). already done
                        res = np.transpose(res, (1, 2, 0))  # H, W, C
                        mask_image = Ftrans.to_pil_image(res)
                        mask_image = self.postprocessing(mask_image)
                        my = y[0].cpu().detach().numpy()
                        my = np.transpose(my, (1,2,0))
                        orig_image = Ftrans.to_pil_image(my)
                        orig_image = self.postprocessing(orig_image)

                        orig_image.save("log/{}/{}_UNet++_temp_{}_orig.png".format(
                            self.experiment_name,
                            time.strftime("%Y-%d-%b %H:%M:%S", time.gmtime()), e
                        ))
                        mask_image.save("log/{}/{}_UNet++_temp_{}_mask.png".format(
                            self.experiment_name,
                            time.strftime("%Y-%d-%b %H:%M:%S", time.gmtime()), e
                        ))
            # adapt learning rate
            last_lr = self.learning_rate
            try:
                last_lr = self.scheduler.get_last_lr()    # ubuntu torch has a bug :(
            except Exception as lr_exception:
                print("[ERROR] torch problem, learning rate: {}".format(lr_exception))
            self.scheduler.step(totalTestLoss)

            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgTestLoss = totalTestLoss / testSteps
            # update our training history
            #self.Hist["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            #self.Hist["test_loss"].append(avgTestLoss.cpu().detach().numpy())
            self.Hist["train_loss"].append(avgTrainLoss.item())
            self.Hist["test_loss"].append(avgTestLoss.item())
            # print(self.Hist)  # displays all up to now
              # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(e + 1, self.epochs))
            print("Train loss: {:.8f}, Test loss: {:.6f}".format(avgTrainLoss, avgTestLoss))
            self.log("[INFO] EPOCH: {}/{}".format(e + 1, self.epochs))
            self.log("Train loss: {:.8f}, Test loss: {:.6f}".format(avgTrainLoss, avgTestLoss))
            try:
                if last_lr != self.scheduler.get_last_lr():
                   print("[INFO] learn.rate changes from {} to {}".format(last_lr, self.scheduler.get_last_lr()))
            except Exception as lr_exception2:
                print("[ERROR] learning rate scheduling: {}".format(lr_exception2))

            if early_stopper(totalTrainLoss):
                print(f"[INFO]: early stopping at epoch {e} with loss of {totalTrainLoss}")
                self.log(f"[INFO] Early stopping after {e} epochs with {totalTrainLoss}.")
                break

        # display the total time needed to perform the training
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))

        # perform model on any training data item to check its validity
        with torch.no_grad():
            self.model.eval()
            img, msk = self.training_dataset[0]
            img = img.unsqueeze(0).to(self.device)
            res = self.model(img)
            res = res.squeeze(0)
            # res = torch.sigmoid(res) # run through sidmoid function
            res = torch.nn.functional.softmax(res, dim=0)

        res = res.cpu().detach().numpy()        # squeeze(0). already done
        res = np.transpose(res, (1, 2, 0)) # H, W, C
        mask_image = Ftrans.to_pil_image(res)
        mask_image = self.postprocessing(mask_image)
        self.log(str(self.Hist))
        return mask_image, self.get_statistics_result(mask_image)

    def _init_training(self):
        # retrieve paths (label files should be inside a folder called labels and the file is prefixed with labels)
        labels = [os.path.join(os.path.split(x)[0], "../labels/labels_"+os.path.split(x)[1]) for x in self.images]

        split = train_test_split(self.images, labels, test_size=0.2, random_state=42)
        (trainImages, testImages) = split[:2]
        (trainMasks, testMasks) = split[2:]

        self.log("Starting GMT {}\n".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())))
        self.log("\n".join(testImages))
        self.log(f"\nUsing {self.model.__class__.__name__}\n")

        trans = self._get_transforms()
        if not self.unsupervised:
            self.training_dataset = SegmentationDataset(image_paths=trainImages, mask_paths=trainMasks, transforms=trans)
            self.test_dataset = SegmentationDataset(image_paths=testImages, mask_paths=testMasks, transforms=trans)
        else:
            self.training_dataset = UnsupervisedMaskedTrainingDataset(image_paths=trainImages, transforms=trans, neighborhood=1)
            self.test_dataset = UnsupervisedMaskedTrainingDataset(image_paths=testImages, transforms=trans, neighborhood=1)

        if self.synthetic_data:
            self.training_dataset = SyntheticTrainingDataset(amount=400, transforms=trans, bleeding_is_always_on=True)
            self.test_dataset = SyntheticTrainingDataset(amount=60, transforms=trans, bleeding_is_always_on=True)

        if self.device == 'mpsold':
            self.training_loader = DataLoader(self.training_dataset, shuffle=True, batch_size=BATCH_SIZE,
                                              pin_memory=True,
                                              num_workers=os.cpu_count(), multiprocessing_context="forkserver",
                                              persistent_workers=True)
            self.test_loader = DataLoader(self.test_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True,
                                          num_workers=os.cpu_count(), multiprocessing_context="forkserver",
                                          persistent_workers=True)
        else:
            self.training_loader = DataLoader(self.training_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True,
                                              num_workers=os.cpu_count())
            self.test_loader = DataLoader(self.test_dataset, shuffle=True, batch_size=BATCH_SIZE, pin_memory=True,
                                              num_workers=os.cpu_count())

    def execute(self, imagefile):
        # image = Image.open(imagefile) - when opening, we use now
        # if we wouldn't use the transformations/the data loaders, we'd have W x H x C, then we would need to prepare for
        #    B x C x H x W
        # image = np.transpose(image, (2, 0, 1))
        # image = np.expand_dims(image, 0)
        # but we transform it like this
        #imagetensor = trans(image)
        #imagetensor.unsqueeze(0)        # adding a batch index/dimension   (now it is done by dataloader)
        loader = self._creating_execution_dataloader(imagefile)
        #    needs to be B x C x H x W (dataloader takes care of it)
        for (i, (image)) in loader:
            imagetensor = image.to(self.device)
            self.model.eval()
            with torch.no_grad():
                result = self.model(imagetensor)
                result = result.squeeze(0)
                # result = torch.sigmoid(result) #we are using softmax, as we create multi-segmentation masks (exclusive)
                result = torch.nn.functional.softmax(result, dim=0)
            res = result.cpu().detach().numpy()
            res = np.transpose(res,(1,2,0))
        mask_image = self.postprocessing(Ftrans.to_pil_image(res),True)
        return mask_image, self.get_statistics_result(mask_image)

    def get_statistics_result(self, mask_image=None):
        class_counter = [1] * len(self.class_labels)
        if mask_image is not None:
            # the 1 is for the laplacian correction, in order to avoid division by 0 errors
            width, height = mask_image.size
            imgdata = mask_image.getdata()
            for y in range(0, height):
                for x in range(0, width):
                    rgb = imgdata[y*width+x]
                    for check_idx in range(len(self.class_labels)):
                        if rgb == self.parameters[self.class_labels[check_idx] + '_color']:
                            class_counter[check_idx] += 1

        return mlprocessing.StatisticsResult(self.class_labels, class_counter)
