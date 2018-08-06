import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.gluon.data import dataset
from mxnet.gluon.data import dataloader
import collections
import os


class ImageWithMaskDataset(dataset.Dataset):
    """
    A dataset for loading images (with masks) stored as `xyz.jpg` and `xyz_mask.png`.

    Parameters
    ----------
    root : str
        Path to root directory.
    num_classes : int
        The number of classes in your data set.
    transform : callable, default None
        A function that takes data and label and transforms them:
    ::
        transform = lambda data, label: (data.astype(np.float32)/255, label)
    """

    def __init__(self, root, num_classes, transform=None):
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)
        self.num_classes = num_classes

    def _list_images(self, root):
        images = collections.defaultdict(dict)
        for filename in sorted(os.listdir(root)):
            name, ext = os.path.splitext(filename)
            mask_flag = name.endswith("_mask")
            if ext.lower() not in self._exts:
                continue
            if not mask_flag:
                images[name]["base"] = filename
            else:
                name = name[:-5]  # to remove '_mask'
                images[name]["mask"] = filename
        self._image_list = list(images.values())

    def one_hot(self, Y):
        one_hot_mask = nd.zeros(
            (Y.shape[0],) + (self.num_classes,) + Y.shape[1:])
        for c in range(self.num_classes):
            one_hot_mask[:, c, :, :] = (Y == c)
        return one_hot_mask

    def preprocess(self, data, label):
        gray_data = nd.sum_axis(nd.array([[[[0.3]], [[0.59]], [[0.11]]]]) * data, 1, keepdims=True)
        gray_label = nd.sum_axis(nd.array([[[[1]], [[1]], [[1]]]]) * label, 1)
        one_hot_label = self.one_hot(gray_label)
        return gray_data, one_hot_label

    def __getitem__(self, idx):
        assert 'base' in self._image_list[idx], "Couldn't find base image for: " + \
            self._image_list[idx]["mask"]
        base_filepath = os.path.join(self._root, self._image_list[idx]["base"])
        base = mx.image.imread(base_filepath, 0).transpose((2, 0, 1)).astype(np.float32)
        assert 'mask' in self._image_list[idx], "Couldn't find mask image for: " + \
            self._image_list[idx]["base"]
        mask_filepath = os.path.join(self._root, self._image_list[idx]["mask"])
        mask = mx.image.imread(mask_filepath, 0).transpose((2, 0, 1)).astype(np.float32)
        mask = mask.astype(np.float32)
        one_hot_mask = nd.zeros((self.num_classes,) + mask.shape[1:], dtype=np.float32)
        for c in range(self.num_classes):
            one_hot_mask[c, :, :] = (mask == c)[0]
        if self._transform is not None:
            return self._transform(base, one_hot_mask)
        else:
            return base, one_hot_mask

    def __len__(self):
        return len(self._image_list)


def DataLoaderGenerator(data_loader):
    """
    A generator wrapper for loading images (with masks) from a 'ImageWithMaskDataset' dataset.

    Parameters
    ----------
    data_loader : 'Dataset' instance
        Instance of Gluon 'Dataset' object from which image / mask pairs are yielded.
    """
    for data, label in data_loader:
        data_desc = mx.io.DataDesc(name='data', shape=data.shape, dtype=np.float32)
        label_desc = mx.io.DataDesc(name='label', shape=label.shape, dtype=np.float32)
        batch = mx.io.DataBatch(
            data=[data],
            label=[label],
            provide_data=[data_desc],
            provide_label=[label_desc])
        yield batch


class DataLoaderIter(mx.io.DataIter):
    """
    An iterator wrapper for loading images (with masks) from an 'ImageWithMaskDataset' dataset.
    Allows for MXNet Module API to train using Gluon data loaders.

    Parameters
    ----------
    root : str
        Root directory containg image / mask pairs stored as `xyz.jpg` and `xyz_mask.png`.
    num_classes : int
        Number of classes in data set.
    batch_size : int
        Size of batch.
    shuffle : Bool
        Whether or not to shuffle data.
    num_workers : int
        Number of sub-processes to spawn for loading data. Default 0 means none.
    """
    def __init__(self, root, num_classes, batch_size, shuffle=False, num_workers=0):

        self.batch_size = batch_size
        self.dataset = ImageWithMaskDataset(root=root, num_classes=num_classes)
        if mx.__version__ == "0.11.0":
            self.dataloader = mx.gluon.data.DataLoader(
                self.dataset, batch_size=batch_size, shuffle=shuffle, last_batch='rollover')
        else:
            self.dataloader = mx.gluon.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                last_batch='rollover')
        self.dataloader_generator = DataLoaderGenerator(self.dataloader)

    def __iter__(self):
        return self

    def reset(self):
        self.dataloader_generator = DataLoaderGenerator(self.dataloader)

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return [
            mx.io.DataDesc(name='data', shape=(self.batch_size,) + self.dataset[0][0].shape, dtype=np.float32)
        ]

    @property
    def provide_label(self):
        return [
            mx.io.DataDesc(name='label', shape=(self.batch_size,) + self.dataset[0][1].shape, dtype=np.float32)
        ]

    def next(self):
        return next(self.dataloader_generator)
