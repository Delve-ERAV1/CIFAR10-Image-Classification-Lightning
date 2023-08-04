from model.network import *
from utils.utils import *
from augment.augment import *
from dataset.dataset import *
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from pprint import pprint
from torch_lr_finder import LRFinder
import torch
import matplotlib.pyplot as plt
import seaborn as sn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn

config = process_config("utils/config.yml")
pprint(config)

classes = config["data_loader"]["classes"]
batch_size = config["data_loader"]['args']["batch_size"]
num_workers = config["data_loader"]['args']["num_workers"]
dropout = config["model_params"]["dropout"]
seed = config["model_params"]["seed"]
epochs = config["training_params"]["epochs"]


#####################################

SEED = 42
cuda = torch.cuda.is_available()
print("CUDA Available?", cuda)

# For reproducibility
torch.manual_seed(SEED)

if cuda:
  torch.cuda.manual_seed(SEED)

# dataloader arguments
dataloader_args = dict(shuffle=True, batch_size=batch_size, num_workers=2, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)


train = CIFAR10Dataset(transform=None)
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
mu, std = get_stats(train_loader)
train_transforms, test_transforms = get_transforms(mu, std)

# train dataloader
train = CIFAR10Dataset(transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)


# test dataloader
test = CIFAR10Dataset(transform = test_transforms, train=False)
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)


# get some random training images
images, labels = next(iter(train_loader))
classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# show images
imshow(torchvision.utils.make_grid(images[:4]))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


###########################

criterion = nn.CrossEntropyLoss()

device = get_device()

model = ResNet18(len(train_loader), criterion)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
max_lr = LR_Finder(model, criterion, optimizer, train_loader)
print(f"Max LR @ {max_lr}")


model = ResNet18(len(train_loader), criterion, max_lr=max_lr)

# training
trainer = Trainer(
  log_every_n_steps=1, 
  enable_model_summary=True,
  max_epochs=epochs, 
  precision=16,
  accelerator='auto',
  devices=1 if torch.cuda.is_available() else None,
  logger=CSVLogger(save_dir="logs/"),
  callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
)




def main():
   
  trainer.fit(model, train_loader, test_loader)
  trainer.test(model, test_loader)

  metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
  del metrics["step"]
  metrics.set_index("epoch", inplace=True)
  print(metrics.dropna(axis=1, how="all").head())
  sn.relplot(data=metrics, kind="line")
  plt.show()


if __name__ == "__main__":
    main()
