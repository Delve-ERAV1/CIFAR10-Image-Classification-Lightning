import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy


class ResBlock(nn.Module):

  def __init__(self, in_channel, out_channel, stride=1):
    super(ResBlock, self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),

        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )

  def forward(self, x):
    return(self.conv(x))



class ResNet18(pl.LightningModule):
  def __init__(self, train_loader_len, criterion, num_classes=10, lr=0.001):
    super().__init__()
    self.save_hyperparameters()

    self.criterion = criterion
    self.train_loader_len = train_loader_len
    self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    self.prep_layer = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    self.layer_one = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )

    self.res_block1 = ResBlock(128, 128)

    self.layer_two = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )

    self.layer_three = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
        nn.MaxPool2d(2,2),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )

    self.res_block2 = ResBlock(512, 512)

    self.max_pool = nn.MaxPool2d(4,4)
    self.fc = nn.Linear(512, num_classes, bias=False)

  def forward(self, x):
    x = self.prep_layer(x)
    x = self.layer_one(x)
    R1 = self.res_block1(x)
    x = x + R1

    x = self.layer_two(x)

    x = self.layer_three(x)
    R2 = self.res_block2(x)
    x = x + R2

    x = self.max_pool(x)

    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return(x)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
                            optimizer, 
                            max_lr=1.45E-03,
                            epochs=self.trainer.max_epochs,
                            steps_per_epoch=self.train_loader_len,
                            pct_start=5/self.trainer.max_epochs,
                            div_factor=100,
                            three_phase=False,
                        )
    return([optimizer], [scheduler])

  def training_step(self, train_batch, batch_idx):
    data, target = train_batch
    y_pred = self(data)
    loss = self.criterion(y_pred, target)

    pred = torch.argmax(y_pred.squeeze(), dim=1)
    acc = accuracy(pred, target, task="multiclass", num_classes=self.hparams.num_classes)

    self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
    self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

    return(loss)

  def validation_step(self, batch, batch_idx):
    return(self.evaluate(batch, 'val'))

  def test_step(self, batch, batch_idx):
    return(self.evaluate(batch, 'test'))

  def evaluate(self, batch, stage=None):
      data, target = batch
      y_pred = self(data)

      loss = self.criterion(y_pred, target).item()
      pred = torch.argmax(y_pred.squeeze(), dim=1)
      acc = accuracy(pred, target, task="multiclass", num_classes=self.hparams.num_classes)

      if stage:
          self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
          self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True)

      return pred, target
