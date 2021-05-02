import torch
import sys
from fastmri.data.subsample import create_mask_for_mask_type
sys.path.insert(0, "/srv/share4/ksarangmath3/mri/fastMRI/fastmri/models/")
from fastmri.pl_modules import FastMriDataModule, UnetModule
from fastmri.data.transforms import UnetDataTransform
from fastmri.data import CombinedSliceDataset, SliceDataset
from fastmri.evaluate import ssim
from PIL import Image
import numpy as np

#LOAD MODEL HERE, THIS SHOULD BE ALL YOU NEED TO CHANGE
model = UnetModule.load_from_checkpoint(checkpoint_path='/srv/share4/ksarangmath3/mri/fastMRI/fastmri_examples/unet/unet/unet_demo/checkpoints/epoch=40-step=87944.ckpt')
model.eval()

challenge = 'singlecoil'
data_partition = 'val'
data_path =  f"/srv/share4/ksarangmath3/mri/data/{challenge}_{data_partition}"
mask = create_mask_for_mask_type(
        'random', [0.08], [4]
    )
val_transform = UnetDataTransform(challenge, mask_func=mask)
dataset = SliceDataset(
                root=data_path,
                transform=val_transform,
                # sample_rate=sample_rate,
                # volume_sample_rate=volume_sample_rate,
                challenge=challenge,
                use_dataset_cache=True
            )
# sampler = torch.utils.data.DistributedSampler(dataset)
dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            num_workers=0,
            # worker_init_fn=worker_init_fn,
            # sampler=sampler,
        )
dataset_len = len(dataset)
ssim_list = []

#baseline SSIM means no reconstruction is done on downsampled image when computing SSIM to target
ssim_baseline_list = []
for i,data in enumerate(dataloader):
    with torch.no_grad():
        if i % 100 == 0:
            print(f"done with {i} steps out of {dataset_len} steps")
            print(f"model SSIM is {0 if i == 0 else np.mean(ssim_list)}")
            print(f"baseline SSIM is {0 if i == 0 else np.mean(ssim_baseline_list)}")
        ssim_list.append(ssim(model(data[0]).numpy(), data[1].numpy()))
        ssim_baseline_list.append(ssim(data[0].numpy(), data[1].numpy()))
        # print(ssim_list[-1])

print(np.mean(ssim_list), '- MODEL SSIM')
print(np.mean(ssim_baseline_list), '- BASELINE SSIM')
    
# import pathlib
# from fastmri.data import subsample
# from fastmri.data import transforms, mri_data
# import torch
# import sys
# from fastmri.data.subsample import create_mask_for_mask_type
# sys.path.insert(0, "/srv/share4/ksarangmath3/mri/fastMRI/fastmri/models/")
# from fastmri.pl_modules import UnetModule
# from fastmri.data.transforms import UnetDataTransform
# from fastmri.data import CombinedSliceDataset, SliceDataset
# from fastmri.evaluate import ssim
# from PIL import Image
# import numpy as np

# challenge = 'singlecoil'
# data_partition = 'val'
# data_path =  f"/srv/share4/ksarangmath3/mri/data/{challenge}_{data_partition}"

# # model = UnetModule(
# #         in_chans=1,  # number of input channels to U-Net
# #         out_chans=1,  # number of output chanenls to U-Net
# #         chans=32,  # number of top-level U-Net channels
# #         num_pool_layers=4,  # number of U-Net pooling layers
# #         drop_prob=0.0,  # dropout probability
# #         lr=0.001,  # RMSProp learning rate
# #         lr_step_size=40,  # epoch at which to decrease learning rate
# #         lr_gamma=0.1,  # extent to which to decrease learning rate
# #         weight_decay=0.0,  # weight decay regularization strength
# #     )
# model = UnetModule.load_from_checkpoint(checkpoint_path='/srv/share4/ksarangmath3/mri/unet_demo_run1/checkpoints/epoch=40-step=87944.ckpt')
# model.eval()

# # def data_transform(kspace, mask, target, data_attributes, filename, slice_num):
# #     # Transform the data into appropriate format
# #     # Here we simply mask the k-space and return the result
# #     kspace = transforms.to_tensor(kspace)
# #     masked_kspace, _ = transforms.apply_mask(kspace, mask_func)
# #     return masked_kspace, target

# mask = create_mask_for_mask_type(
#         'random', [0.08], [4]
#     )
# val_transform = UnetDataTransform(challenge, mask_func=mask)

# dataset = SliceDataset(
#     root=pathlib.Path(
#       data_path
#     ),
#     transform=val_transform,
#     challenge='singlecoil'
# )

# for i,data in enumerate(dataset):
#     out = model(data[0].unsqueeze(0))
#     print(ssim(np.expand_dims(data[0].numpy(), 0), np.expand_dims(data[1].numpy(), 0)))
#     print(ssim(out.detach().numpy(), np.expand_dims(data[1].numpy(), 0)))
#     if i > 50:
#         break