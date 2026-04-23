import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(filename):
    return os.path.splitext(filename.lower())[1] in IMG_EXTENSIONS


def list_class_folders(root):
    classes = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            classes.append(name)
    return sorted(classes)


def build_class_splits(data_root):
    base_train_root = os.path.join(data_root, "base_train")
    base_test_root = os.path.join(data_root, "base_test")
    new_test_root = os.path.join(data_root, "new_test")

    if not os.path.isdir(base_train_root):
        raise FileNotFoundError(f"base_train not found: {base_train_root}")
    if not os.path.isdir(base_test_root):
        raise FileNotFoundError(f"base_test not found: {base_test_root}")
    if not os.path.isdir(new_test_root):
        raise FileNotFoundError(f"new_test not found: {new_test_root}")

    seen_train_classes = list_class_folders(base_train_root)
    seen_test_classes = list_class_folders(base_test_root)
    unseen_classes = list_class_folders(new_test_root)

    seen_classes = sorted(list(set(seen_train_classes) | set(seen_test_classes)))

    if seen_train_classes != seen_test_classes:
        print("[Warning] base_train and base_test class folders are not exactly identical.")
        print(f"base_train classes: {len(seen_train_classes)}")
        print(f"base_test  classes: {len(seen_test_classes)}")
        extra_in_train = sorted(list(set(seen_train_classes) - set(seen_test_classes)))
        extra_in_test = sorted(list(set(seen_test_classes) - set(seen_train_classes)))
        if extra_in_train:
            print(f"Extra classes in base_train: {extra_in_train}")
        if extra_in_test:
            print(f"Extra classes in base_test: {extra_in_test}")

    overlap = set(seen_classes) & set(unseen_classes)
    if len(overlap) > 0:
        raise ValueError(f"Seen and unseen classes overlap: {sorted(list(overlap))}")

    all_classes = seen_classes + unseen_classes
    return seen_classes, unseen_classes, all_classes


class _PlantWildBaseDataset(Dataset):
    def __init__(self, root, all_classes):
        self.root = root
        self.all_classes = all_classes
        self.class_to_idx = {name: i for i, name in enumerate(all_classes)}
        self.samples = []
        self._build_samples()

    def _build_samples(self):
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        for class_name in sorted(os.listdir(self.root)):
            class_dir = os.path.join(self.root, class_name)
            if not os.path.isdir(class_dir):
                continue

            if class_name not in self.class_to_idx:
                raise ValueError(f"Class '{class_name}' not found in all_classes.")

            label = self.class_to_idx[class_name]

            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath) and is_image_file(fname):
                    self.samples.append((fpath, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No image samples found under: {self.root}")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def load_image(image_path):
        return Image.open(image_path).convert("RGB")


class PlantWildDataset(_PlantWildBaseDataset):
    def __init__(self, root, all_classes, transform=None):
        super().__init__(root=root, all_classes=all_classes)
        self.transform = transform

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = self.load_image(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
