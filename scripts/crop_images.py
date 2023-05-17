import os
from PIL import Image
from tqdm import tqdm


def crop_image(load_data_path, save_data_path):
    for image in tqdm(os.listdir(load_data_path)):
        img = Image.open(os.path.join(load_data_path, image))
        img_croped = img.crop(box=(0, 250, 953, 768))
        img_croped.save(os.path.join(save_data_path, image))


if __name__ == "__main__":
    original_image_path = "croped_data\\original_data\\train\\0"
    croped_image_path_to_save = "croped_data\\croped\\train\\0"

crop_image(original_image_path, croped_image_path_to_save)
