import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Sampler, SubsetRandomSampler, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

NUM_EPOCH = 8
device=""
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
           
validation_loss_history = []
validation_model_history = []
validation_optimizer_history = []

def _select_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(' \n\n\n****************************************************************\n')
        print('The code will run on GPU using CUDA')
    else:
        device = torch.device("cpu")
        print('The code will run on CPU using CUDA\n')
    return device
def sampler(dataset,shuffle):
    divide_data=0.7
    validation_data_size= 0.05
    test_data_size= 0.1
    random_seed = 0
    num_train = int(len(dataset)* divide_data)
    indices = list(range(num_train))
    v_split = int(np.floor(validation_data_size * num_train))
    t_split=int(np.floor(test_data_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    valid_indices,test_indices,train_indices =  indices[:v_split],indices[v_split:v_split+t_split],indices[v_split+t_split:]
    return  train_indices,valid_indices,test_indices

def _make_dataloaders():
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    batch_size=4
    data_set = datasets.CIFAR10('./data', download=True, transform=transform)
    train_indices, valid_indices, test_indices=sampler(data_set, shuffle=True)
    print ('****************************************************************')
    print(' Number of training data: '+str(len(train_indices)))
    print(' Number of Validation data: '+str(len(valid_indices)))
    print(' Number of test data: '+str(len(test_indices)))
    print ('****************************************************************')
 
    train_loader = torch.utils.data.DataLoader(data_set, pin_memory=True, batch_size=batch_size,
                                               sampler=SubsetRandomSampler(train_indices))
    test_loader = torch.utils.data.DataLoader(data_set, pin_memory=True, batch_size=batch_size,
                                                sampler=SubsetRandomSampler(test_indices))
    valid_loader = torch.utils.data.DataLoader(data_set, pin_memory=True, batch_size=batch_size,
                                                   sampler=SubsetRandomSampler(valid_indices))

    return train_loader, valid_loader, test_loader 

class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)


def train(model, criterion,optimizer ,train_loader,valid_loader):
    ## Do the training
 
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, train_data in enumerate(train_loader, 0):
            # get the inputs
            train_inputs, train_labels = train_data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(train_inputs.to(device))
            loss = criterion(outputs, train_labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
        Validate(epoch,model,optimizer,valid_loader)

    print('Finished Training')


def test(model ,test_loader):
    ## Do the training
    print("\n*******************************************************************")
    print('Testing Process:')
    model.cuda()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Test is end')
    print("\n*******************************************************************")

    return   100 * correct / total


    

def Validate(epoch_no,model,optimizer,valid_loader):
    print("\n****************************************************************")
    print('Validating Process:\n on epoch no'+ str(epoch_no))

    valid_loss = 0.0
    for j, valid_data in enumerate(valid_loader, 0):
        # get the inputs        
        valid_inputs, valid_labels = valid_data
        valid_outputs = model(valid_inputs.to(device))
        loss = criterion(valid_outputs, valid_labels.to(device))
        valid_loss += loss.item()
        if j % 20 == 19:    # print every 20 mini-batches
            print('VALIDATION STATUS - Running epoch %d, batch %d' %(epoch_no + 1, j + 21))

    print("\n")
    valid_loss = valid_loss/j+1
    print('Loss until this epoch: %d',valid_loss)
    validation_loss_history.append( valid_loss)
    validation_model_history.append( model.state_dict())
    validation_optimizer_history.append( optimizer.state_dict())
    print('Validation on this epoch is end')
    print("****************************************************************\n")


if __name__ == '__main__':
    #select device
    device=_select_device()

    ## Define the training, validation, and test dataloader
    train_loader,valid_loader,test_loader=_make_dataloaders()

    ## Create model, objective function and optimizer
    model = ResNet50_CIFAR()
    model=model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                           lr=0.001, momentum=0.9)
    train(model,criterion,optimizer, train_loader,valid_loader)
    print(validation_loss_history)
    best_loss_indx=np.argmin(validation_loss_history)
    
    plt.figure(figsize=(8,8))
    print(range(len(validation_loss_history)))
    print('\n')
    print(validation_loss_history)
    plt.plot(range(len(validation_loss_history)), validation_loss_history)
    plt.ylabel('Cross Entropy Loss')
    plt.title('Loss on validation set for every 2 epochs')
    plt.xlabel('Epochs')
    plt.show()
    
    print('Model trained and find best result in %d epochs with validation error of %.3f has been saved' % (best_loss_indx + 1, validation_loss_history[best_loss_indx]))
    accuracy= test(model,test_loader)
    torch.save({'tested_accuracy':accuracy, 'best_model': validation_model_history[best_loss_indx],'best_optimizer': validation_optimizer_history[best_loss_indx],'best_epoch': best_loss_indx,'best_loss': validation_loss_history[best_loss_indx]}, 'bestmodel.pth')
    print ('\nThe model accuracy on test data is'+ str(accuracy))


