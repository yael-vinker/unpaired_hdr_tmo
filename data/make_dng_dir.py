import os
import shutil

if __name__ == '__main__':
    root = os.path.join("/cs/labs/raananf/yael_vinker/new_hdr_dset/20171106_subset/results_20171023")
    dst = os.path.join("/cs/labs/raananf/yael_vinker/new_hdr_dset/dng_collection")
    for dir_ in os.listdir(root):
        print(dir_)
        sub_root = os.path.join(root, dir_)
        for img_name in os.listdir(sub_root):
            base, extension = os.path.splitext(img_name)
            new_name = base + "_" + dir_ + extension

            im_path = os.path.join(sub_root, img_name)

            if extension == ".dng":
                cur_dst = os.path.join(dst)
                shutil.copy(im_path, cur_dst)
                dst_file = os.path.join(dst, img_name)
                new_dst_file_name = os.path.join(dst, new_name)
                os.rename(dst_file, new_dst_file_name)