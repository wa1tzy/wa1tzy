import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from nets import Net
import os
from torch.optim import lr_scheduler


class Trainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lossfn_cls = nn.NLLLoss(reduction="sum")
    def train(self):
        save_path = "models/net_arcloss2.pth"
        epoch = 0
        train_data = torchvision.datasets.MNIST(root="./MNIST",download=True, train=True,
                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5,],std=[0.5,])]))
        train_loader = data.DataLoader(train_data,shuffle=True, batch_size=100)
        net = Net().to(self.device)
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path))
        else:
            print("NO Param")
        optimizer = torch.optim.SGD(net.parameters(),lr=1e-3, momentum=0.9, weight_decay=0.0005)
        # optimizer = torch.optim.Adam(net.parameters())
        scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.8)
        while True:
            feat_loader = []
            label_loader = []
            for i, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                feature, output = net.forward(x)
                loss = self.lossfn_cls(output, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                feat_loader.append(feature)
                label_loader.append(y)

                if i % 600 == 0:
                    print("epoch:", epoch, "i:", i, "arcsoftmax_loss:", loss.item())
            feat = torch.cat(feat_loader, 0)
            labels = torch.cat(label_loader, 0)
            net.visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)
            epoch += 1
            torch.save(net.state_dict(), save_path)
            scheduler.step()
            if epoch==150:
                break
t =Trainer()
t.train()
