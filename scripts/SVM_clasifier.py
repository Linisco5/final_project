import os
import numpy as np
import cv2
import random
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pickle

path_to_save_model = "SVM\\model"
os.makedirs(path_to_save_model, exist_ok=True)
path_to_save_plots = "SVM\\plots"
os.makedirs(path_to_save_plots, exist_ok=True)
model_name = f"binary_SVM_v1.0.0"
train_path = "scraped_data\\train\\"
test_path = "scraped_data\\test\\"
categories = ["0", "1"]


def data_prepare(data_path, data_categories):
    data = []
    for category in data_categories:
        img_path = os.path.join(data_path, category)
        img_label = categories.index(category)
        for img in os.listdir(img_path):
            full_img_path = os.path.join(img_path, img)
            image = cv2.imread(full_img_path)
            image_resized = cv2.resize(image, (50, 50))
            image_prepare = np.array(image_resized).flatten()
            data.append([image_prepare, img_label])
    return data


train_data = data_prepare(train_path, categories)
test_data = data_prepare(test_path, categories)


def data_split_x_y(data_set):
    random.shuffle(data_set)
    features = []
    labels = []
    for feature, label in data_set:
        features.append(feature)
        labels.append(label)
    return features, labels


x_train, y_train = data_split_x_y(train_data)
x_test, y_test = data_split_x_y(test_data)

model = SVC(C=1, kernel="poly", gamma="auto")
model.probability = True
model.fit(x_train, y_train)

with open(f"{path_to_save_model}\\{model_name}", "wb") as file:
    pickle.dump(model, file)

prediction = model.predict(x_test)
accuracy = model.score(x_test, y_test)

print("Acc", accuracy)

y_train_pred = cross_val_predict(model, x_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)

print("Precision: ", precision_score(y_train, y_train_pred))
print("Recall: ", recall_score(y_train, y_train_pred))
print("F1 score: ", f1_score(y_train, y_train_pred))

cm = confusion_matrix(y_test, prediction)
print(cm)

confusion_matrix = metrics.confusion_matrix(y_test, prediction)
cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, display_labels=["no transport", "transport"]
)
cm_display.plot()
plt.savefig(os.path.join(path_to_save_plots, f"confussion_mtx_{model_name}.png"))
plt.show()
