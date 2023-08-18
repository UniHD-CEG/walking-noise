from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST as torch_FashionMNIST


class FashionMNIST():
    name = "FashionMNIST"
    dims = (1, 28, 28)
    has_test_dataset = False

    def __init__(self, batch_size=128, num_workers=2, data_root="/tmp/datasets/"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def prepare_data(self) -> None:
        torch_FashionMNIST(root=self.data_root, train=True, download=True)
        torch_FashionMNIST(root=self.data_root, train=False, download=True)

    def setup(self) -> None:
        self.train_set = torch_FashionMNIST(
            root=self.data_root, train=True,
            download=False, transform=self.transform_train)
        self.val_set = torch_FashionMNIST(
            root=self.data_root, train=False,
            download=False, transform=self.transform_val)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)
        return val_loader

