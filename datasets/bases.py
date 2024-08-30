import torch
from PIL import Image, ImageFile

from torch.utils.data import Dataset
import os.path as osp

ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt


def read_image(img_list):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    if type(img_list) == type("This is a str"):
        img_path = img_list
        got_img = False
        if not osp.exists(img_path):
            raise IOError("{} does not exist".format(img_path))
        while not got_img:
            try:
                img = Image.open(img_path).convert('RGB')
                RGB = img.crop((0, 0, 256, 128))
                NI = img.crop((256, 0, 512, 128))
                TI = img.crop((512, 0, 768, 128))
                img3 = [RGB, NI, TI]
                got_img = True
            except IOError:
                print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                pass
    else:
        img3 = []
        for i in img_list:
            img_path = i
            got_img = False
            if not osp.exists(img_path):
                raise IOError("{} does not exist".format(img_path))
            while not got_img:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img3.append(img)
                    got_img = True
                except IOError:
                    print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
                    pass
    return img3


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids, cams, tracks = [], [], []

        for _, pid, camid, trackid in data:
            pids += [pid]
            cams += [camid]
            tracks += [trackid]
        pids = set(pids)
        cams = set(cams)
        tracks = set(tracks)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        num_views = len(tracks)
        return num_pids, num_imgs, num_cams, num_views

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams, num_train_views = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams, num_train_views = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


class ImageDatasetSet(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img3 = read_image(img_path)
        # /home/Newdisk/luowenlong/Datasets/msvr310/bounding_box_train/0268/vis/0268_s000_v0_000.jpg
        # /home/Newdisk/luowenlong/Datasets/RGBNT100/rgbir/bounding_box_train/0539_c0001_009.jpg
        # /home/Newdisk/luowenlong/Datasets/RGBNT201/train_171/RGB/000205_cam2_0_02.jpg
        if self.transform is not None:
            self.transform.randomize_parameters()
            img = [self.transform(img) for img in img3]

        # img = torch.stack(img)
        # print(img.shape)
        # torch.save(img, '/home/Newdisk/luowenlong/Projects/Multi-Modal-ReID/output-individual-backbone/msvr310/fusion/spatial/img_' + img_path[0][70:-4] + '.pth' )
        return img, pid, camid, trackid, img_path[0].split('/')[-1]


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img3 = read_image(img_path)

        if self.transform is not None:
            img = [self.transform(img) for img in img3]
        return img, pid, camid, trackid, img_path[0].split('/')[-1]


class ImageDatasetSet_v2(Dataset):
    def __init__(self, dataset, transform1, transform_m, transform2):
        self.dataset = dataset
        self.transform_1 = transform1
        self.transform_m = transform_m
        self.transform_2 = transform2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.dataset[index]
        img3 = read_image(img_path)
        img2 = [self.transform_1(img) for img in img3]
        self.transform_m.randomize_parameters()
        img1 = [self.transform_m(img) for img in img2]
        img0 = [self.transform_2(img) for img in img1]

        # img = torch.stack(img)
        # print(img.shape)
        # torch.save(img, '/home/Newdisk/luowenlong/Projects/Multi-Modal-ReID/output-individual-backbone/msvr310/fusion/spatial/img_' + img_path[0][70:-4] + '.pth' )
        return img0, pid, camid, trackid, img_path[0].split('/')[-1]