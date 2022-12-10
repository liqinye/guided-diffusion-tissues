import xarray as xr
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from os import listdir
from os.path import join
import numpy as np
import torch


day1 = '03'
day2 = '02'
# h0_path = f"data/h0/SPCAM_NN_v2*.h0*2003-08-{day}*.nc"
p1 = f"data/h3/SPCAM_NN_v2*.h3*2003-08-{day1}*.nc"
p2 = f"data/h3/SPCAM_NN_v2*.h3*2003-08-{day2}*.nc"

# def is_nc_file(file: str):
#     return not file.startswith(".") \
#         and any(file.endswith(extension) for extension in [".nc"])

'''
filter data by a lattitue range
'''
def _preprocess(x, latt_bound=(-30.00, -5.00)):
    return x.sel(lat=slice(*latt_bound))

class TrainDataset(Dataset):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset = xr.open_mfdataset(dataset_dir, preprocess=_preprocess).CRM_QV

    def __getitem__(self, index):
        x = torch.unsqueeze(torch.squeeze(torch.tensor(self.dataset.isel(time=index).values)),0)
        x = torch.reshape(x, (x.size()[0], x.size()[1], x.size()[2], -1))
        shape = x.size()
        return torch.reshape(x, (shape[0], shape[-1], shape[-3], shape[-2]))

    def __len__(self):
        return len(self.dataset.time.values)

if __name__ == "__main__":
    latt_bound = (-30.00, -5.00)
    partial_func = partial(_preprocess, latt_bound=latt_bound)

    train_dataset = TrainDataset("data/h3/*.nc")
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    # print(train_dataset[0].size())
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    train_bar = tqdm(train_loader)
    for data in train_bar:
        print(data.size())
        # d = torch.reshape(data, (data.size()[0], data.size()[1], data.size()[-1], data.size()[-3], data.size()[-2]))
        # print(torch.reshape(d, (data.size()[0], data.size()[1], data.size()[2], data.size()[3], data.size()[4])) == data)
        # print(torch.reshape(data[0][0][0], (-1,)))
        # print(torch.reshape(data, (data.size()[0], data.size()[1], data.size()[2], data.size()[3], -1)).size())
        break

    # h3_dataset = xr.open_mfdataset(h3_path, preprocess=partial_func)

    # CRM_QV = h3_dataset.CRM_QV

    # # h3_dataset = h3_dataset.where(h3_dataset.lat<-30.00)
    # # print(h3_dataset.sel(time=slice('2003-08-01 00:00:00')).CRM_QV.values[0].shape)
    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # h3_dataset.CRM_QC.isel(time=3, lon=100, lat=5).plot(ax=axes[0][0])
    # h3_dataset.CRM_QV.isel(time=2, lon=100, lat=5).plot(ax=axes[0][1])
    # h3_dataset.CRM_QV.isel(time=3, lon=20, lat=5).plot(ax=axes[1][0])
    # h3_dataset.CRM_QV.isel(time=3, lon=20, lat=1).plot(ax=axes[1][1])
    # plt.show()
    # d1 = xr.open_mfdataset(p1, preprocess=_preprocess)
    # d2 = xr.open_mfdataset(p2, preprocess=_preprocess)

    # d3 = xr.merge([d1.CRM_QV, d2.CRM_QV])

    # print(d1.CRM_QV)
    # print('\n')
    # print(d2.CRM_QV)
    # print('\n')
    # print(d3)


