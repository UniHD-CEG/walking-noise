from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST as troch_MNIST


class MNIST():
    name = "MNIST"
    dims = (1, 28, 28)
    has_test_dataset = False

    def __init__(self, batch_size=128, num_workers=2, data_root="/tmp/datasets/", normalize=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        self.normalize = normalize

        self.transform_train = self.default_transforms()

        self.transform_val = self.default_transforms()

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def prepare_data(self) -> None:
        troch_MNIST(root=self.data_root, train=True, download=True)
        troch_MNIST(root=self.data_root, train=False, download=True)

    def setup(self) -> None:
        self.train_set = troch_MNIST(
            root=self.data_root, train=True,
            download=False, transform=self.transform_train)
        self.val_set = troch_MNIST(
            root=self.data_root, train=False,
            download=False, transform=self.transform_val)

    def default_transforms(self):
        if self.normalize:
            mnist_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
            )
        else:
            mnist_transforms = transforms.Compose([transforms.ToTensor()])

        return mnist_transforms

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

