from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import SVHN as torch_SVHN


class SVHN():
    name = "SVHN"
    dims = (3, 32, 32)
    has_test_dataset = False

    def __init__(self, batch_size=128, num_workers=2, data_root="/tmp/datasets/"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def prepare_data(self) -> None:
        torch_SVHN(root=self.data_root, split='train', download=True)
        torch_SVHN(root=self.data_root, split='test', download=True)

    def setup(self) -> None:
        self.train_set = torch_SVHN(
            root=self.data_root, split='train',
            download=False, transform=self.transform_train)
        self.val_set = torch_SVHN(
            root=self.data_root, split='test',
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

