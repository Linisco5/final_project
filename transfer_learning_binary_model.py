import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from sklearn.metrics import classification_report
from sklearn import metrics

if __name__ == "__main__":
    training_dataset_path = "scraped_data\\train\\"
    test_dataset_path = "scraped_data\\test\\"
    path_to_save_model = "transfer_learning\\model\\"
    model_name_to_save = f"MobileNet_3_2_0_best_base_20230517_2130_freezed_3.hdf5"
    model_name_to_load = f"MobileNet_3_2_0_best_base_20230517_2130_freezed_3.hdf5"
    os.makedirs(path_to_save_model, exist_ok=True)
    image_size = (128, 128)
    batch_size = 32
    epochs = 10
    lr = 0.00001
    threshold = 0.5

    path_to_save_plots = f"transfer_learning\\plots\\"
    os.makedirs(path_to_save_plots, exist_ok=True)

    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(
        training_dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
    )

    test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False,
    )

    steps_per_epoch = int(np.floor(test_generator.samples / batch_size))

    # Base model
    base_model = MobileNet(
        weights="imagenet",
        include_top=False,
        input_tensor=layers.Input(shape=image_size + (3,)),
    )
    # base_model.summary()

    base_model.trainable = False

    x = layers.Conv2D(32, (3, 3), activation="relu", input_shape=image_size + (3,))(
        base_model.output
    )
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dense(units=1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=x)

    # load weights
    model.load_weights(os.path.join(path_to_save_model, model_name_to_load))

    for layer in model.layers:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True

    model.summary()

    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(path_to_save_model, model_name_to_save),
        monitor="val_loss",
        save_best_only=True,
        mode="min",
    )

    # Compiling the CNN
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    # Fitting the CNN
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        steps_per_epoch=steps_per_epoch,
        callbacks=[checkpoint],
    )

    # plot loss during training
    plt.subplot(211)
    plt.title("Model Loss")
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title("Model Accuracy")
    plt.plot(history.history["binary_accuracy"], label="train")
    plt.plot(history.history["val_binary_accuracy"], label="test")
    plt.legend()
    plt.savefig(os.path.join(path_to_save_plots, f"acc_loss_{model_name_to_save}.png"))
    plt.show()

    tf.keras.backend.clear_session()

    model = keras.models.load_model(
        os.path.join(path_to_save_model, model_name_to_save)
    )
    predicted_classes = model.predict(test_generator, verbose=1)

    predicted_classes = [1 if pred[0] > threshold else 0 for pred in predicted_classes]
    gt_classes = test_generator.classes

    print(classification_report(gt_classes, predicted_classes))
    confusion_matrix = metrics.confusion_matrix(gt_classes, predicted_classes)
    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=["no transport", "transport"]
    )
    cm_display.plot()
    plt.savefig(
        os.path.join(path_to_save_plots, f"confussion_mtx_{model_name_to_save}.png")
    )
    plt.show()
