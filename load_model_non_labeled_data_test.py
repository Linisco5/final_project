import cv2
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm


def preprocess_image(img, shape):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    im_pil = im_pil.resize(shape, Image.NEAREST)
    im_pil = np.asarray(im_pil)
    im_pil = im_pil / 255
    return np.expand_dims(im_pil, axis=0)


def perform_ensemble(test_data_path, models_path):
    rows = []
    models_names = os.listdir(models_path)
    for model_name in models_names:
        print(f'Predicts: {model_name}')
        model = load_model(os.path.join(models_path, model_name), compile=False)
        image_shape = model.inputs[0].shape[1:3]
        # iterate through classes
        for class_label in os.listdir(test_data_path):
            # iterate through images names
            for image_name in tqdm(os.listdir(os.path.join(test_data_path, class_label))):
                full_image_path = os.path.join(test_data_path, class_label, image_name)
                image = cv2.imread(full_image_path)
                processed_image = preprocess_image(image, image_shape)
                prediction = model.predict(processed_image, verbose=0)[0][0]
                rows.append({'full_image_path': full_image_path,
                             'model_name': model_name.split('.')[0],
                             'score': round(prediction, 3)})
                if prediction > 0.5 and class_label == '0':
                    fullpath = f'models_bad_classification\\{model_name}\\no_car_car\\{image_name}'
                    cv2.imwrite(fullpath, image)
                elif prediction <= 0.5 and class_label == '1':
                    fullpath = f'models_bad_classification\\{model_name}\\car_no_car\\{image_name}'
                    cv2.imwrite(fullpath, image)
                elif prediction <= 0.5 and class_label == '0':
                    fullpath = f'models_bad_classification\\{model_name}\\no_car\\{image_name}'
                    cv2.imwrite(fullpath, image)
                else:
                    fullpath = f'models_bad_classification\\{model_name}\\car\\{image_name}'
                    cv2.imwrite(fullpath, image)

    df = pd.DataFrame.from_dict(rows)
    df_pivot = df.pivot_table(index='full_image_path', values='score', columns='model_name').reset_index()
    return df_pivot


if __name__ == '__main__':
    tests_data_path = 'klasifikavimas\\new_data\\'
    model_path = 'final_models'

    # get predictions
    d_frame = perform_ensemble(tests_data_path, model_path)
    d_frame.to_csv('final_models_learn_results_20230517_21.csv', index=False)
