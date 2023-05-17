import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def augment_images(path_to_save, num_of_augmentations, generator):
    for i in range(0, num_of_augmentations):
        print(f'processing_{i}_batch_of_images')
        for batch in generator:
            for image_index in tqdm(range(0, batch[0].shape[0])):
                image = batch[0][image_index, :, :, :]
                image_name = generator.filenames[image_index].split('\\')[1].split('.')[0]
                new_image_name = f'{image_name}_augmentation_{i}.jpg'
                path_to_save_image = os.path.join(path_to_save, new_image_name)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(path_to_save_image, image)
            break


if __name__ == '__main__':
    path_to_images = 'scraped_data\\augmentacijoms\\test\\'
    path_to_save_images = 'scraped_data\\augmentation\\1\\'
    os.makedirs(path_to_save_images, exist_ok=True)
    batch_size = len(os.listdir(os.path.join(path_to_images, '1')))

    # instantiate the ImageDataGenerator class
    datagen = ImageDataGenerator(brightness_range=[0.5, 1])

    augm_generator = datagen.flow_from_directory(path_to_images,
                                                 target_size=(768, 1024),
                                                 batch_size=batch_size,
                                                 class_mode='binary',
                                                 shuffle=False)

    augment_images(path_to_save_images, 2, augm_generator)
