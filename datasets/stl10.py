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
                        (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
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
                        (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
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
