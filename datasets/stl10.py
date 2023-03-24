from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

def load_STL10_dataset(batch_size = 64):
    training_data = datasets.STL10(
        root="data",
        split="train",
        download=True,
        transform= transforms.Compose([
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]) 
    )

    test_data = datasets.STL10(
        root="data",
        split="test",
        download=True,
        transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ])
    )

    classes = [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck']


    return DataLoader(training_data, batch_size=batch_size, shuffle=True), DataLoader(test_data, batch_size=batch_size), len(training_data.classes)
