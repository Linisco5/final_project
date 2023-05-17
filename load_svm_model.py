import os
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import pickle


def preprocess_image(img):
    img = cv2.resize(img, (50, 50))
    image = np.array(img).flatten()
    return image


def perform_ensemble(test_data_path, models_path):
    rows = []
    models_names = os.listdir(models_path)
    for model_name in models_names:
        print(f'Predicts: {model_name}')
        with open(os.path.join(models_path, model_name), 'rb') as file:
            model = pickle.load(file)
        # iterate through classes
        for class_label in os.listdir(test_data_path):
            # iterate through images names
            for image_name in tqdm(os.listdir(os.path.join(test_data_path, class_label))):
                full_image_path = os.path.join(test_data_path, class_label, image_name)
                image = cv2.imread(full_image_path)
                processed_image = preprocess_image(image)
                processed_image = processed_image.reshape(1, -1)
                prediction = model.predict(processed_image)
                rows.append({'full_image_path': full_image_path, 'model_name': model_name.split('.')[0],
                             'score': prediction})
                if prediction > 0.5 and class_label == '0':
                    fullpath = f'models_bad_classification\\{model_name}\\no_car_car\\{image_name}.jpg'
                    cv2.imwrite(fullpath, image)
                elif prediction <= 0.5 and class_label == '1':
                    fullpath = f'models_bad_classification\\{model_name}\\car_no_car\\{image_name}.jpg'
                    cv2.imwrite(fullpath, image)
                elif prediction <= 0.5 and class_label == '0':
                    fullpath = f'models_bad_classification\\{model_name}\\no_car\\{image_name}.jpg'
                    cv2.imwrite(fullpath, image)
                else:
                    fullpath = f'models_bad_classification\\{model_name}\\car\\{image_name}.jpg'
                    cv2.imwrite(fullpath, image)
    df = pd.DataFrame.from_dict(rows)
    df_pivot = df.pivot_table(index='full_image_path', values='score', columns='model_name').reset_index()
    return df_pivot


if __name__ == '__main__':
    test_datas_path = 'scraped_data\\test\\'
    model_path = 'SVM\\model\\'

    # get predictions
    d_frame = perform_ensemble(test_datas_path, model_path)
    d_frame.to_csv('SVM_results_20230517_3.csv', index=False)
