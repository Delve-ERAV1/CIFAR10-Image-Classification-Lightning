import torch, torchvision
from torchvision import transforms
import numpy as np
import gradio as gr
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from model.network import ResNet18
import matplotlib.pyplot as plt
import PIL
import io
from PIL import Image

from model.network import *
from utils.gradio_utils import *
from augment.augment import *
from dataset.dataset import *



model = ResNet18(20, None)
model = model.load_from_checkpoint("/content/drive/MyDrive/ERAv1/S12/resnet18.ckpt", map_location=torch.device("cpu"))
model.eval()

dataloader_args = dict(shuffle=True, batch_size=64)
_, test_transforms = get_transforms(mu, std)

test = CIFAR10Dataset(transform=test_transforms, train=False)
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

target_layers = [model.res_block2.conv[-1]]
targets = None
device = torch.device("cpu")

def upload_image_inference(input_img, n_top_classes, transparency):

  org_img = input_img.copy()

  input_img = transform(input_img)
  input_img = input_img.unsqueeze(0)

  outputs = model(input_img)

  softmax = torch.nn.Softmax(dim=0)
  o = softmax(outputs.flatten())
  confidences = {classes[i]: float(o[i]) for i in range(n_top_classes)}
  _, prediction = torch.max(outputs, 1)

  cam = GradCAM(model=model, target_layers=target_layers)

  grayscale_cam = cam(input_tensor=input_img, targets=None)
  grayscale_cam = grayscale_cam[0, :]
  img = input_img.squeeze(0)
  img = inv_normalize(img)

  rgb_img = np.transpose(img.cpu(), (1, 2, 0))
  rgb_img = rgb_img.numpy()
  visualization = show_cam_on_image(org_img/255, grayscale_cam, use_rgb=True, image_weight=transparency)

  return([confidences, [org_img, grayscale_cam, visualization]])


def misclass_gr(num_images, layer_val, transparency):
  images_list = misclassified_data[:num_images]

  images_list = [image_to_array(img, layer_val, transparency) for img in images_list]
  return(images_list)


def class_gr(num_images, layer_val, transparency):
  images_list = classified_data[:num_images]

  images_list = [image_to_array(img, layer_val, transparency) for img in images_list]
  return(images_list)


def image_to_array(input_img, layer_val, transparency=0.6):
  input_tensor = input_img[0]

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


def get_misclassified_data(model, device, test_loader):
  """
  Function to run the model on test set and return misclassified images
  :param model: Network Architecture
  :param device: CPU/GPU
  :param test_loader: DataLoader for test set
  """
  mis_count = 0
  correct_count = 0

  # Prepare the model for evaluation i.e. drop the dropout layer
  model.eval()
  # List to store misclassified Images
  misclassified_data, classified_data = [], []
  # Reset the gradients
  with torch.no_grad():
    # Extract images, labels in a batch
    for data, target in test_loader:
      # Migrate the data to the device
      data, target = data.to(device), target.to(device)
      # Extract single image, label from the batch
      for image, label in zip(data, target):
        # Add batch dimension to the image
        image = image.unsqueeze(0)
        # Get the model prediction on the image
        output = model(image)
        # Convert the output from one-hot encoding to a value
        pred = output.argmax(dim=1, keepdim=True)
        # If prediction is incorrect, append the data
        if pred != label:
          misclassified_data.append((image, label, pred))
          mis_count += 1
        else:
          classified_data.append((image, label, pred))
          correct_count += 1

        if ((mis_count>=20) and (correct_count>=20)):
          return ((classified_data, misclassified_data))


title = "CIFAR10 trained on ResNet18 (Pytorch Lightning) Model with GradCAM"
description = "A simple Gradio interface to infer on ResNet model, get GradCAM results for existing & new Images"

examples = [["image1.jpg", 4, 0.9], 
            ["image2.jpg", 5, 0.7], 
            ["image3.jpg", 6, 0.6], 
            ["image4.jpg", 7, 0.6], 
            ["image5.jpg", 3, 0.7], 
            ["image6.jpg", 5, 0.2], 
            ["image7.jpg", 2, 0.4], 
            ["image8.jpg", 3, 0.5], 
            ["image9.jpg", 1, 0.5], 
            ["image0.jpg", 6, 0.1]]




with gr.Blocks() as gradcam:
    classified_data, misclassified_data = get_misclassified_data(model, device, test_loader)

    gr.Markdown("Make Grad-Cam of uploaded image, or existing images.")
    with gr.Tab("Upload New Image"):
        upload_input = [gr.Image(shape=(32, 32)),
                        gr.Number(minimum=0, maximum=10, label='n Top Classes', value=3, precision=0),
                        gr.Slider(0, 1, label='Transparency', value=0.6)]

        upload_output = [gr.Label(),
                         gr.Gallery(label="Image | CAM | Image+CAM",
                                    show_label=True, elem_id="gallery1").style(columns=[3],
                                                                              rows=[1],
                                                                              object_fit="contain",
                                                                              height="auto")]
        button1 = gr.Button("Perform Inference")


    with gr.Tab("View Class Activate Maps"):
      with gr.Row():
          with gr.Column():
            cam_input21 = [gr.Number(minimum=1, maximum=20, precision=0, value=3, label='View Correctly Classified CAM | Num Images'),
                          gr.Number(minimum=1, maximum=3, precision=0, value=1, label='(-) Target Layer'),
                          gr.Slider(0, 1, value=0.6, label='Transparency')]

            image_output21 = gr.Gallery(label="Images - Grad-CAM (correct)",
                                      show_label=True, elem_id="gallery21")
            button21 = gr.Button("View Images")

          with gr.Column():
            cam_input22 = [gr.Number(minimum=1, maximum=20, precision=0, value=3, label='View Misclassified CAM | Num Images'),
                          gr.Number(minimum=1, maximum=3,  precision=0, value=1, label='(-) Target Layer'),
                          gr.Slider(0, 1, value=0.6, label='Transparency')]

            image_output22 = gr.Gallery(label="Images - Grad-CAM (Misclassified)",
                                        show_label=True, elem_id="gallery22")
            button22 = gr.Button("View Images")

    button1.click(upload_image_inference, inputs=upload_input, outputs=upload_output)
    button21.click(class_gr, inputs=cam_input21, outputs=image_output21)
    button22.click(misclass_gr, inputs=cam_input22, outputs=image_output22)
    gr.Examples(
          examples=examples,
          inputs=upload_input,
          outputs=upload_output,
          fn=upload_image_inference,
          cache_examples=True,
    )
    

gradcam.launch(debug=True)
