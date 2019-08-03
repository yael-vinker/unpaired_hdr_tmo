import os

train_dataroot_hdr = "data/check_data/train_fake"
train_dataroot_ldr = "data/check_data/train_real"
test_dataroot_hdr = "data/check_data/test_fake"
test_dataroot_ldr = "data/check_data/test_real"
test_dataroot_red_wind_ldr = "data/check_data/test_real"
apply_windows_loss = False

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
num_epochs = 10

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# # Number of GPUs available. Use 0 for CPU mode.
# ngpu = 1

manualSeed = 999

g_d_loss_factor = 0.2
