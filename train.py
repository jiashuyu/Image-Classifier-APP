# Import packages
import argparse
from time import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Define transform the training, validation, and testing sets, note that the validation and testing sets share the same transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                          transforms.RandomResizedCrop(224),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], 
                                        [0.229, 0.224, 0.225])])

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./flowers',
                  help='path to folder of images')
    parser.add_argument('--save_dir', type=str, default='./',
                  help='path to the checkpoint')
    parser.add_argument('--arch', type=str, default='alexnet',
                  help='chosen model from alexnet, vgg11, densenet121')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                  help='learning rate')
    parser.add_argument('--hidden_units', type=int, default=1000,
                  help='number of neurons in the hidden layer')
    parser.add_argument('--epochs', type=int, default=1,
                  help='number of epochs used for training')
    parser.add_argument('--gpu', type=bool, default=True,
                  help='whether to use gpu for training')
    return parser.parse_args()

def main():
    in_arg = get_input_args()
    
    train_dir = in_arg.data_dir + '/train'
    valid_dir = in_arg.data_dir + '/valid'
    test_dir = in_arg.data_dir + '/test'
    
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    
    # Load my models
    densenet = models.densenet121(pretrained=True)
    alexnet = models.alexnet(pretrained=True)
    vgg = models.vgg11(pretrained=True)
    
    model_dict = {'densenet121': densenet, 'alexnet': alexnet, 'vgg11': vgg}
    model = model_dict[in_arg.arch]
    classifier_inputs = {'densenet121': 1024, 'alexnet': 9216, 'vgg11': 25088}
    
    # Freeze parameters so we don't backprop through them
    for p in model.parameters():
        p.requires_grad = False
    
    # change the classifier of the model so that we can have 102 outputs
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(classifier_inputs[in_arg.arch], in_arg.hidden_units)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(in_arg.hidden_units, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifier
    
    # Train a model with a pre-trained network
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)
    
    epochs = in_arg.epochs
    print_every = 40
    steps = 0

    # change to cuda if gpu is available and requested
    if (in_arg.gpu == True) & (torch.cuda.is_available()):
        model.to('cuda')
    
    start_time = time()
    for e in range(epochs):
        model.train()
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
            
            if (in_arg.gpu == True) & (torch.cuda.is_available()):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                correct = 0
                total = 0
                test_loss = 0
                with torch.no_grad():
                    for data in valid_loader:
                        images, labels = data
                        if (in_arg.gpu == True) & (torch.cuda.is_available()):
                            images = images.to('cuda')
                            labels = labels.to('cuda')
                        outputs = model.forward(images)
                        test_loss += criterion(outputs, labels).item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                print("Epoch: {}/{}... ".format(e+1, epochs),
                    "Training Loss: {:.4f}".format(running_loss/print_every),
                    "   Test Loss: {:.4f}".format(test_loss/len(valid_loader)),
                    "   Accuracy: {:.4f}".format(correct / total))
                
                running_loss = 0
                model.train()
    end_time = time()
    print('total time used for training:', end_time-start_time, 'seconds')
    
    
    # Save the checkpoint 
    checkpoint = {'class_to_idx': train_dataset.class_to_idx,
             'model_name': in_arg.arch,
             'classifier_input_size': classifier_inputs[in_arg.arch],
             'output_size': 102,
             'classifier_hidden_layers': in_arg.hidden_units,
             'state_dict': model.state_dict()}

    torch.save(checkpoint, in_arg.save_dir + 'checkpoint.pth')

    

if __name__ == "__main__":
	main()