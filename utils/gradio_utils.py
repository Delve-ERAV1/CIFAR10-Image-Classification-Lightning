import torch
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import PIL
import io
from PIL import Image
import numpy as np
import random

transform = transforms.ToTensor()
targets = None
device = torch.device("cpu")


mu = [0.49139968, 0.48215841, 0.44653091]
std = [0.24703223, 0.24348513, 0.26158784] 


inv_normalize = transforms.Normalize(
    mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
    std=[1/0.23, 1/0.23, 1/0.23]
)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.ToTensor()


def get_examples():
  example_images = [f'{c}.jpg' for c in classes]
  example_top = [random.randint(0, 9) for r in range(10)]
  example_transparency = [random.choice([0.6, 0.7, 0.8]) for r in range(10)]
  examples = [[example_images[i], example_top[i], example_transparency[i]] for i in range(len(example_images))]
  return(examples)


def image_to_array(input_img, model, layer_val, transparency=0.6):
  input_tensor = input_img[0]
  print(input_tensor.shape)

  cam = GradCAM(model=model, target_layers=[model.res_block2.conv[-layer_val]])
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
  grayscale_cam = grayscale_cam[0, :]

  img = input_tensor.squeeze(0)
  img = inv_normalize(img)
  rgb_img = np.transpose(img, (1, 2, 0))
  rgb_img = rgb_img.numpy()

  visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True,
                                    image_weight=transparency)

  plt.imshow(visualization)
  plt.title(r"Correct: " + classes[input_img[1].item()] + '\n' + 'Output: ' + classes[input_img[2].item()])

  with io.BytesIO() as buffer:
      plt.savefig(buffer, format = "png")
      buffer.seek(0)
      image = Image.open(buffer)
      ar = np.asarray(image)

  return(ar)
