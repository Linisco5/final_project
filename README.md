# Final project: Images clasifiers
Final project Image clasification using ML

Clasification of images made in electric transport charging station (there is any transport in charging zone or not).<br/>
Used CNN custom architecture, transfer learning with ResNet50, MobileNet, VGG16, Xception and SVM clasifier.

## Packages
All needed packages are in requirement.txt file.

## Data
For this porject was used public data from https://eismoinfo.lt/. <br/>
Images from camera: Elektrėnų EV 2, A1 45,86 km: A1 (Vilnius–Kaunas–Klaipėda) - 45.86 km.

## File to data augmentation
data_augmentation.py - create more data by changing brightness range.

## File to crope images:
crop_images.py - test how it works if images contains mostly targeted area of charging station.

## File to get data:
data_scraper.py

## Files to get models:
CNN_train_and_test_binary_model.py - to create custom CNN model;<br/>
Transfer_learning_binary_model.py - helps to create 4 models (I test data with models based on ResNet50, MobileNet, VGG16 and Xception archtecture)<br/>
SVM_clasifier.py - Support vector machine clasifier model.

## Files to monitor model failures:
load_model_non_labled_data_test.py - for CNN models <br/>
load_svm_model.py - fro SVM clasifier.
