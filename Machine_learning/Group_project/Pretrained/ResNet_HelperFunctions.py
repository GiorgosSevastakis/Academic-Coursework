from torch.utils.data import Dataset as DS

def apply_val(model, dataloader, loss_fn, device='cpu'):
    from torch import no_grad
    from tqdm import tqdm
    from numpy import array

    model.eval()
    with no_grad():
        losses = []
        weights = []
        n_correct = 0
        N = 0

        for (X, y) in tqdm(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            losses.append(loss_fn(y_pred, y))
            weights.append(y.size()[0])
            n_correct += (y == y_pred.argmax(axis=1)).sum().item()
            N += y.size()[0]

        losses, weights = array([l.cpu().item() for l in losses]), array(weights)
        return (losses * weights).sum() / weights.sum(), n_correct / N
        
class ImageDataset(DS):
    def __init__(self, df, image_type='nucleus'):
        from numpy.random import permutation

        if image_type == 'nucleus': self.fnames = df['nucleus_fname']
        else: self.fnames = df['cell_fname']

        self.labels = df['label']

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        from torchvision.io import read_image
        return read_image(self.fnames[idx]).float(), self.labels[idx]

class LoadImageDataset(DS):
    def __init__(self, parent_dir):
        from os import listdir

        control_dir = parent_dir + 'control/'
        drug_dir = parent_dir + 'drug/'

        self.fnames = [control_dir + fname for fname in listdir(control_dir)] + [drug_dir + fname for fname in listdir(drug_dir)]
        self.labels = [0 for _ in listdir(control_dir)] + [1 for _ in listdir(drug_dir)]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        from torchvision.io import read_image
        return read_image(self.fnames[idx]).float(), self.labels[idx]
    
### Pretrained Models
    
from torchvision.models import resnet34
from torch.nn import Module, Sequential, Flatten, Linear, ReLU, Softmax

class Resnet34L(Module):
    def __init__(self, n_layers:int=3, n_nodes:int=512):
        super().__init__()

        rn34 = resnet34(pretrained=True)

        model = Sequential(*list(rn34.children())[:7])
        for p in model.parameters():
            p.requires_grad = False

        model.append(Flatten())

        for i in range(n_layers):
            if i == 0:
                model.append(Linear(50_176, n_nodes))
                model.append(ReLU())
            elif i == n_layers-1:
                model.append(Linear(n_nodes, 2))
            else:
                model.append(Linear(n_nodes, n_nodes))
                model.append(ReLU())

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))
    
class Resnet34M(Module):
    def __init__(self, n_layers:int=3, n_nodes:int=512):
        super().__init__()

        rn34 = resnet34(pretrained=True)

        model = Sequential(*list(rn34.children())[:8])
        for p in model.parameters():
            p.requires_grad = False

        model.append(Flatten())

        for i in range(n_layers):
            if i == 0:
                model.append(Linear(25_088, n_nodes))
                model.append(ReLU())
            elif i == n_layers-1:
                model.append(Linear(n_nodes, 2))
            else:
                model.append(Linear(n_nodes, n_nodes))
                model.append(ReLU())

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))
    
class Resnet34S(Module):
    def __init__(self, n_layers:int=3, n_nodes:int=512):
        super().__init__()

        rn34 = resnet34(pretrained=True)

        model = Sequential(*list(rn34.children())[:9])
        for p in model.parameters():
            p.requires_grad = False

        model.append(Flatten())

        for i in range(n_layers):
            if i == 0:
                model.append(Linear(512, n_nodes))
                model.append(ReLU())
            elif i == n_layers-1:
                model.append(Linear(n_nodes, 2))
            else:
                model.append(Linear(n_nodes, n_nodes))
                model.append(ReLU())

        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))
    
### New generation of models
    
class Resnet18(Module):
    def __init__(self, train_layers:int=10):
        super().__init__()
        from torchvision.models import resnet18

        rn = resnet18(pretrained=True)
        self.model = Sequential(
            rn,
            ReLU(),
            Linear(1000, 2),
            Softmax()
        )

        for count, p in enumerate(list(self.model.parameters())):
            if count < len(list(self.model.parameters())) - 1 - train_layers:
                p.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))

class Resnet34(Module):
    def __init__(self, train_layers:int=10):
        super().__init__()
        from torchvision.models import resnet34

        rn = resnet34(pretrained=True)
        self.model = Sequential(
            rn,
            ReLU(),
            Linear(1000, 2),
            Softmax()
        )

        for count, p in enumerate(list(self.model.parameters())):
            if count < len(list(self.model.parameters())) - 1 - train_layers:
                p.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))
    
class Resnet152(Module):
    def __init__(self, train_layers:int=10):
        super().__init__()
        from torchvision.models import resnet152

        rn = resnet152(pretrained=True)
        self.model = Sequential(
            rn,
            ReLU(),
            Linear(1000, 2),
            Softmax()
        )

        for count, p in enumerate(list(self.model.parameters())):
            if count < len(list(self.model.parameters())) - 1 - train_layers:
                p.requires_grad = False
    
    def forward(self, x):
        return self.model(x)
    
    def getSummary(self):
        from torchsummary import summary
        return summary(self, (3, 224, 224))