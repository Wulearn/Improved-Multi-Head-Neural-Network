import os
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import ImageFile, Image
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ParametersDataset(Dataset):
    def __init__(
        self,
        csv_file,
        root_dir,
        image_dim=(320, 320),
        pre_crop_transform=None,
        post_crop_transform=None,
        per_img_normalisation=False,
        pumpback=False,
        pumpback_speed=False,
        speed=False,
        hotend=False,
    ):
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.pre_crop_transform = pre_crop_transform
        self.post_crop_transform = post_crop_transform
        self.image_dim = image_dim
        self.per_img_normalisation = per_img_normalisation
        self.targets = []
        self.use_pumpback = pumpback
        self.use_pumback_speed = pumpback_speed
        self.use_speed = speed
        self.use_hotend = hotend

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        self.targets = []
        self.parmeters=[]
        if torch.is_tensor(idx):
            idx = idx.tolist()
        self.use_pumpback = self.dataframe.pumpback[idx]#回抽
        self.use_pumpback_speed = self.dataframe.pumpback_speed[idx]#回抽速度
        self.use_speed= self.dataframe.speed[idx]#速度
        self.use_hotend = self.dataframe.hotend[idx]#温度
        self.use_max_distancce = self.dataframe.distance[idx]#空驶距离
        prefix = 'E:\\'
        result =self.dataframe.img_path[idx].replace(prefix, '', 1)
        result = result.replace('\\', os.sep)
        img_name = os.path.join(self.root_dir, result)
        dim = self.image_dim[0] / 2
        left = self.dataframe.nozzle_tip_x[idx] -dim
        top = self.dataframe.nozzle_tip_y[idx]-20
        right = self.dataframe.nozzle_tip_x[idx] + dim
        bottom = self.dataframe.nozzle_tip_y[idx]+300

        image = Image.open(img_name)
        if self.pre_crop_transform:
            image = self.pre_crop_transform(image)
        image = image.crop((left, top, right, bottom))
        if self.per_img_normalisation:
            tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
            image = tfms(image)
            mean = torch.mean(image, dim=[1, 2])
            std = torch.std(image, dim=[1, 2])
            image = transforms.Normalize(mean, std)(image)
        else:
            if self.post_crop_transform:
                image = self.post_crop_transform(image)

        parameters = torch.tensor(
            [self.dataframe.pumpback[idx], self.dataframe.pumpback_speed[idx], self.dataframe.hotend[idx],
             self.dataframe.speed[idx],self.dataframe.distance[idx]], dtype=torch.float)
        # 计算均值和标准差
        mean_params = parameters.mean()
        std_params = parameters.std()
        parameters = (parameters - mean_params) / std_params
        if self.use_pumpback:
            pumpback_class = int(self.dataframe.pumpback_label[idx])
            self.targets.append(pumpback_class)

        if self.use_pumback_speed:
            pumback_speed_class = int(self.dataframe.pumback_speed_label[idx])
            self.targets.append(pumback_speed_class)

        if self.use_speed:
            speed_class = int(self.dataframe.speed_label[idx])
            self.targets.append(speed_class)

        if self.use_hotend:
            hotend_class = int(self.dataframe.hotend_label[idx])
            self.targets.append(hotend_class)
        y = torch.tensor(self.targets, dtype=torch.long)
        return image, parameters, y
        # return image, y