import os

# dataroot_hdr = "transfer_to_ldr/hdr_images_small_wrap"
dataroot_hdr_2 = "transfer_to_ldr/hdr_images_small_wrap/hdr_images"
# dataroot_ldr = "transfer_to_ldr/ldr_images_small_wrap"
dataroot_output = "transfer_to_ldr/processed_data"
test_data_hdr_tonemap = "transfer_to_ldr/hdr_images_small_wrap"
test_data_hdr = "transfer_to_ldr/processed_data"
# dataroot_output = "transfer_to_ldr/processed_data"

# dataroot_npy = "data/npy_data"
dataroot_npy = "data/hdr_data"
dataroot_ldr = "data/ldr_data2"
# test_dataroot_npy = "data/test_npy_data"
test_dataroot_npy = "data/hdr_data"
test_dataroot_ldr = "data/test_ldr_data"
test_dataroot_red_wind_ldr = "data/red_hdr_wind_data"
apply_windows_loss = False

s_dataroot_npy = "data/npy_data"
s_dataroot_ldr = "data/ldr_data"
s_test_dataroot_npy = "data/test_npy_data"
s_test_dataroot_ldr = "data/test_ldr_data"
s_test_dataroot_red_wind_ldr = "data/red_hdr_wind_data"




image_key = "hdr_image"
window_image_key = "binary_windows"

models_save_path = os.path.join("models", "net.pth")
loss_path = os.path.join("loss_plot")
results_path = os.path.join("result_images")

n_downsample = 2
n_downsamples_d = 4
# n_downsamples_d = 3
input_dim = 3
dim = 6
# Number of workers for dataloader
workers = 8

# Batch size during training
batch_size = 28

# Spatial size of training images. All images will be resized to this
# size using a transformer.
image_size = 256

# Number of training epochs
num_epochs = 4

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# # Number of GPUs available. Use 0 for CPU mode.
# ngpu = 1

manualSeed = 999

g_d_loss_factor = 0.2
