import os

# train_dataroot_hdr = "data/check_data/triangle_train_fake"
train_dataroot_hdr = "/Users/yaelvinker/PycharmProjects/lab/data/hdr_log_data"
train_dataroot_ldr = "/Users/yaelvinker/PycharmProjects/lab/data/ldr_npy"

test_dataroot_hdr = "data/hdr_log_data"
test_dataroot_ldr = "data/ldr_npy"
test_dataroot_original_hdr = "data/hdr_data/hdr_data"
apply_windows_loss = False

image_key = "hdr_image"
window_image_key = "binary_windows"
gray_input_image_key = "input_im"
color_image_key = "color_im"

models_save_path = os.path.join("models")
loss_path = os.path.join("loss_plot")
best_acc_images_path = os.path.join("best_acc_images")
results_path = os.path.join("result_images")
gradient_flow_path = os.path.join("gradient_flow")

input_size = 256
n_downsample = 1
n_downsamples_d = 3
dim = 6
# Number of workers for dataloader
workers = 1

# Batch size during training
batch_size = 28

epsilon = 1e-10

# Spatial size of training images. All images will be resized to this
# size using a transformer.

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

unet_network = "unet"
torus_network = "torus"

layer_factor_2_operators = ["original_unet"]
layer_factor_3_operators = ["square", "square_root"]
layer_factor_4_operators = ["square_and_square_root"]

# con operators
original_unet = "original_unet"
square = "square"
square_root = "square_root"
square_and_square_root = "square_and_square_root"

filters = 64


