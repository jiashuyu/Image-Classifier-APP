# Imports packages
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image


data_transforms = transforms.Compose([transforms.Resize(256),
                         transforms.CenterCrop(224),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], 
                                       [0.229, 0.224, 0.225])])


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./flowers/valid/1/image_06739.jpg',
                  help='path to the image that you want to predict')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint.pth',
                  help='path of the checkpoint')
    parser.add_argument('--top_k', type=int, default='5',
                  help='top k categories predicted')
    parser.add_argument('--category_names', type=str, default='./cat_to_name.json',
                  help='path of the category names')
    parser.add_argument('--gpu', type=bool, default=True,
                  help='whether to use gpu for training')
    return parser.parse_args()    

# Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath):
    cp = torch.load(filepath)
    
    # Load my models
    densenet = models.densenet121(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg = models.vgg11(pretrained=True)
    
    model_dict = {'densenet121': densenet, 'alexnet': alexnet, 'vgg11': vgg}
    model = model_dict[cp['model_name']]
    
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(cp['classifier_input_size'], cp['classifier_hidden_layers'])),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(cp['classifier_hidden_layers'], cp['output_size'])),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    model.classifier = classifier
    model.load_state_dict(cp['state_dict'])
    model.class_to_idx = cp['class_to_idx']
    return model

def process_image(image):
    im = Image.open(image)
    image = data_transforms(im)
    return image.numpy()

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    # predict the top 5 classes from an image file
    np_image = process_image(image_path)
    image = torch.from_numpy(np_image)
    
    # image.to('cuda')
    # model.to('cuda')
    
    image.unsqueeze_(0)
    model.eval()
    output = model(image)
    x = torch.topk(output, topk)
    list_of_class = {}
    np_log_probs = x[0][0].detach().numpy()
    tags = x[1][0].detach().numpy()
    for i in range(topk):
        for classes, idx in model.class_to_idx.items():
            if idx == tags[i]:
                list_of_class[classes] = np.exp(np_log_probs[i])
    return list_of_class

def show_names(cat_to_name, dictionary):
    name_of_class = {}
    for classes, prob in dictionary.items():
        name_of_class[cat_to_name[classes]] = prob
    return name_of_class
    
def main():
    in_arg = get_input_args()
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    m = load_checkpoint(in_arg.checkpoint)
       
    # Display the top 5 classes along with their probabilities
    list_of_class = predict(in_arg.input, m, in_arg.top_k)
    print(show_names(cat_to_name, list_of_class))
    
    
if __name__ == "__main__":
	main()