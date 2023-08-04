# S12

# CIFAR10 Image Classification with PyTorch Lightning

This project implements an image classifier trained on the CIFAR10 dataset using PyTorch Lightning. The project aims to showcase the use of ResNet architecture, data augmentation, custom dataset classes, and learning rate schedulers.

## Project Structure

The project is structured as follows:

1. Data loading and preprocessing
2. Dataset statistics calculation
3. Data Augmentation
3. Model creation
4. Training and evaluation

### Data Loading and Preprocessing

The data for this project is the CIFAR10 dataset, which is loaded using PyTorch's built-in datasets. To ensure that our model generalizes well, we apply several data augmentations to our training set including normalization, padding, random cropping, and horizontal flipping.

### Dataset Statistics Calculation

Before we start training our model, we calculate per-channel mean and standard deviation for our dataset. These statistics are used to normalize our data, which helps make our training process more stable.

```
Dataset Mean - [0.49139968 0.48215841 0.44653091]
Dataset Std - [0.24703223 0.24348513 0.26158784] 
```

### Dataset Augmentation
```python
def get_transforms(means, stds):
  train_transforms = A.Compose(
      [
          A.Normalize(mean=means, std=stds, always_apply=True),
          A.RandomCrop(height=32, width=32, pad=4, always_apply=True),
          A.HorizontalFlip(),
          A.Cutout (fill_value=means),
          ToTensorV2(),
      ]
  )

  test_transforms = A.Compose(
      [
          A.Normalize(mean=means, std=stds, always_apply=True),
          ToTensorV2(),
      ]
  )

  return(train_transforms, test_transforms)
```
![image](https://github.com/Delve-ERAV1/S10/assets/11761529/a0098b5b-e9d4-448b-a6c1-4b24ea9bdd98)


### Model Creation

The model we use for this project is a Custom ResNet, a type of convolutional neural network known for its high performance on image classification tasks.

```
  | Name        | Type               | Params
---------------------------------------------------
0 | criterion   | CrossEntropyLoss   | 0     
1 | accuracy    | MulticlassAccuracy | 0     
2 | prep_layer  | Sequential         | 1.9 K 
3 | layer_one   | Sequential         | 74.0 K
4 | res_block1  | ResBlock           | 295 K 
5 | layer_two   | Sequential         | 295 K 
6 | layer_three | Sequential         | 1.2 M 
7 | res_block2  | ResBlock           | 4.7 M 
8 | max_pool    | MaxPool2d          | 0     
9 | fc          | Linear             | 5.1 K 
---------------------------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.292    Total estimated model params size (MB)
```

#### ResNet Architecture and Residual Blocks

The defining feature of the ResNet architecture is its use of residual blocks and skip connections. Each residual block consists of a series of convolutional layers followed by a skip connection that adds the input of the block to its output. These connections allow the model to learn identity functions, making it easier for the network to learn complex patterns. This characteristic is particularly beneficial in deeper networks, as it helps to alleviate the problem of vanishing gradients.

### Training and Evaluation

To train our model, we use the Adam optimizer with a OneCycle learning rate scheduler. 

```
Epoch 23: 100%
196/196 [00:27<00:00, 7.11it/s, v_num=0, val_loss=0.639, val_acc=0.776, train_loss=0.686, train_acc=0.762]

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│         test_acc          │    0.8758000135421753     │
│         test_loss         │    0.39947837591171265    │

```

#### OneCycle Learning Rate Scheduler

The OneCycle learning rate scheduler varies the learning rate between a minimum and maximum value according to a certain policy. This dynamic learning rate can help improve the performance of our model. We train our model for a total of 24 epochs.

### Learning Rate Finder

```python
def LR_Finder(model, criterion, optimizer, trainloader):

  lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
  lr_finder.range_test(trainloader, end_lr=10, num_iter=200, step_mode='exp')
  max_lr = lr_finder.plot(suggest_lr=True, skip_start=0, skip_end=0)
  lr_finder.reset()
  
  return(max_lr[1])
```

![image](https://github.com/Delve-ERAV1/S12/assets/11761529/7f86fde6-532a-4c58-be91-5252216e125b)


## Dependencies

This project requires the following dependencies:

- torch
- torchvision
- numpy
- albumentations
- matplotlib
- torchsummary


## Usage

To run this project, you can clone the repository and run the main script:

```bash
git clone https://github.com/Delve-ERAV1/S12.git
cd S12
gradio app.py
```

### Upload New Image
![cam](https://github.com/Delve-ERAV1/S12/assets/11761529/465538f3-884e-4446-8e9b-ed0824dd5670)

### View Misclassified Images

![upload](https://github.com/Delve-ERAV1/S12/assets/11761529/cbb2cb46-21ee-420b-af93-e08f5a1f4505)

## Results

![image](https://github.com/Delve-ERAV1/S12/assets/11761529/9f8843f5-9465-445c-9068-50b3197ea371)

## References

Deep Residual Learning for Image Recognition Kaiming He et al
Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates Leslie N. Smith
