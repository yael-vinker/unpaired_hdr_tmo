import torch
from utils import params


def load_data_set(data_root, batch_size_, shuffle, addFrame, hdrMode, normalization, use_c3):
    from utils import ProcessedDatasetFolder
    npy_dataset = ProcessedDatasetFolder.ProcessedDatasetFolder(root=data_root, addFrame=addFrame, hdrMode=hdrMode,
                                                                normalization=normalization, use_c3=use_c3)
    dataloader = torch.utils.data.DataLoader(npy_dataset, batch_size=batch_size_,
                                             shuffle=shuffle, num_workers=params.workers)
    return dataloader


def load_data(train_root_npy, train_root_ldr, batch_size, addFrame, title, normalization, use_c3):
    from utils import printer
    """
    :param isHdr: True if images in "dir_root" are in .hdr format, False otherwise.
    :param dir_root: path to wanted directory
    :param b_size: batch size
    :return: DataLoader object of images in "dir_root"
    """
    train_hdr_dataloader = load_data_set(train_root_npy, batch_size, shuffle=True, addFrame=addFrame, hdrMode=True,
                                         normalization=normalization, use_c3=use_c3)
    train_ldr_dataloader = load_data_set(train_root_ldr, batch_size, shuffle=True, addFrame=addFrame, hdrMode=False,
                                         normalization=normalization, use_c3=use_c3)

    printer.print_dataset_details([train_hdr_dataloader, train_ldr_dataloader],
                                  [train_root_npy, train_root_ldr],
                                  [title + "_hdr_dataloader", title + "_ldr_dataloader"],
                                  [True, False],
                                  [True, True])

    printer.load_data_dict_mode(train_hdr_dataloader, train_ldr_dataloader, title, images_number=2)
    return train_hdr_dataloader, train_ldr_dataloader


def crop_input_hdr_batch(input_hdr_batch):
    b, c, h, w = input_hdr_batch.shape
    th, tw = h - 2 * params.shape_addition, w - 2 * params.shape_addition
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    i, j, h, w = i, j, th, tw
    input_hdr_batch = input_hdr_batch[:, :, i:i + h, j:j + w]
    return input_hdr_batch


def add_frame_to_im(input_im):
    input_im = torch.squeeze(input_im)
    first_row = input_im[0].repeat(params.shape_addition, 1)
    im = torch.cat((first_row, input_im), 0)
    last_row = input_im[-1].repeat(params.shape_addition, 1)
    im = torch.cat((im, last_row), 0)
    left_col = torch.t(im[:, 0].repeat(params.shape_addition, 1))
    im = torch.cat((left_col, im), 1)
    right_col = torch.t(im[:, -1].repeat(params.shape_addition, 1))
    im = torch.cat((im, right_col), 1)
    im = torch.unsqueeze(im, dim=0)
    return im

def add_frame_to_im_batch(images_batch):
    b_size = images_batch.shape[0]
    output = []
    for i in range(b_size):
        im_hdr = images_batch[i].detach()
        im_hdr_frame = add_frame_to_im(im_hdr)
        output.append(im_hdr_frame)
    return torch.stack(output)
