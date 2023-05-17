import datetime
import requests
import os

source = requests.get('https://eismoinfo.lt/eismoinfo-backend/camera-info-table')
path_to_saved_images = 'scraped_data\\'
os.makedirs(path_to_saved_images, exist_ok=True)
camera_name = 'Elektrėnų EV 2, A1 45,86 km'
number_of_images_to_save = 3000


def get_images_source(src, cam_name, img_number):
    cam_list = src.json()
    for cam in cam_list:
        if cam['name'] == cam_name:
            cam_id = cam['id']
            cam_source = requests.get(f'https://eismoinfo.lt/eismoinfo-backend/camera-info-table/{cam_id}'
                                      f'?pageNumber=0&pageSize={img_number}')
            return cam_source.json()


def save_images(img_path):
    cam_images_info = get_images_source(source, camera_name, number_of_images_to_save)
    for image in cam_images_info['info']:
        img_file = requests.get(image['image']).content
        ts = datetime.datetime.fromtimestamp(image['date'] / 1000).strftime('%Y_%m_%d_%H_%M')
        filename = f'{image["id"]}_{ts}.jpg'
        with open(os.path.join(img_path, filename), 'wb') as f:
            f.write(img_file)


save_images(path_to_saved_images)
