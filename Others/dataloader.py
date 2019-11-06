from torch.utils.data.dataset import Dataset
import os.path as osp
import os

class VesselsDataset_fromfile(Dataset):
    def __init__(self, img_dir, gt_dir, transforms=None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_names = sorted(os.listdir(self.img_dir))
        self.gt_names = sorted(os.listdir(self.gt_dir))
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(osp.join(self.img_dir, self.img_names[index]))
        gt = Image.open(osp.join(self.gt_dir, self.gt_names[index]))
        if self.transforms is not None:
            img = self.transforms(img)
            gt = self.transforms(gt)

        return img, gt

    def __len__(self):
        return len(self.img_names)

if __name__ == "__main__":
    print(2+2)
