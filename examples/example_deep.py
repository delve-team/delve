import torch
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models.vgg import vgg11
from torchvision.transforms import Compose, ToTensor

# setup compute device
from tqdm import tqdm

from delve import CheckLayerSat

if __name__ == "__main__":

    device = "cuda:0" if is_available() else "cpu"

    # Get some data
    train_data = CIFAR10(root="./tmp",
                         train=True,
                         download=True,
                         transform=Compose([ToTensor()]))
    test_data = CIFAR10(root="./tmp",
                        train=False,
                        download=True,
                        transform=Compose([ToTensor()]))

    train_loader = DataLoader(train_data,
                              batch_size=1024,
                              shuffle=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=1024,
                             shuffle=False,
                             pin_memory=True)

    # instantiate model
    model = vgg11(num_classes=10).to(device)

    # instantiate optimizer and loss
    optimizer = Adam(params=model.parameters())
    criterion = CrossEntropyLoss().to(device)

    # initialize delve
    tracker = CheckLayerSat("experiment",
                            save_to="plotcsv",
                            stats=["lsat"],
                            modules=model,
                            device=device)

    # begin training
    for epoch in range(10):
        # only record saturation for uneven epochs
        if epoch % 2 == 1:
            tracker.resume()
        else:
            tracker.stop()
        model.train()
        for (images, labels) in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            prediction = model(images)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total = 0
        test_loss = 0
        correct = 0
        model.eval()
        for (images, labels) in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += torch.sum((predicted == labels)).item()
            test_loss += loss.item()

        # add some additional metrics we want to keep track of
        tracker.add_scalar("accuracy", correct / total)
        tracker.add_scalar("loss", test_loss / total)

        # add saturation to the mix
        tracker.add_saturations()
        tracker.save()

    # close the tracker to finish training
    tracker.close()
