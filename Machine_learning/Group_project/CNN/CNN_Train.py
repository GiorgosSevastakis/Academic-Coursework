import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from kerastuner.tuners import BayesianOptimization

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import seaborn as sns
sns.set_style("darkgrid")

from CNN_HelperFunctions import *
from CNN_Model import CNNHyperModel

if __name__ == "__main__":

    ####################################PREPROCESSING####################################
    path_ctrl = r"control"
    path_pntr = r"drug"

    ctrl_im = load_images_from_folder(path_ctrl)
    pntr_im = load_images_from_folder(path_pntr)

    ##Loading of the segmnentation masks computed by Cellpose##
    ctrl_mask_list,pntr_mask_list = get_masks("segments")

    ##Export the cells from the raw images##
    ctrl_cell_images_bef  = get_cells_re(ctrl_im,ctrl_mask_list)
    pntr_cell_images_bef  = get_cells_re(pntr_im,pntr_mask_list)

    ctrl_dis,pntr_dis = get_dis()

    print("Number of control cells:",len(ctrl_cell_images_bef))
    print("Number of drug cells:",len(pntr_cell_images_bef))

    ##Optional manual discarding from .txt files##
    ctrl_cell_images = discard(ctrl_cell_images_bef,ctrl_dis)
    pntr_cell_images = discard(pntr_cell_images_bef,pntr_dis)

    print("Number of control cells after filtering:",len(ctrl_cell_images))
    print("Number of drug cells after filtering",len(pntr_cell_images))

    ##Get median image to pad and resize to one common size##
    median_image = get_median_image(ctrl_mask_list,pntr_mask_list)
    print("Size of median image: ",median_image)

    ##Data augmentation 
    pntr_new = process_and_add_images(ctrl_cell_images)
    ctrl_new = process_and_add_images(pntr_cell_images)

    ctrl_cell_images.extend(pntr_new)
    pntr_cell_images.extend(ctrl_new)

    num_ctrl_cell_images = len(ctrl_cell_images)
    num_pntr_cell_images = len(pntr_cell_images)

    print("Total unique control cell images:", len(ctrl_cell_images))
    print("Total unique pointer cell images:", len(pntr_cell_images))

    all_images = ctrl_cell_images + pntr_cell_images

    resized_images_list = []
    scaling_factors_list = []

    for i in range(len(all_images)):
        resized_image, scale_factor = pad_and_resize_to_square(all_images[i], median_image)
        resized_images_list.append(resized_image)
        scaling_factors_list.append(scale_factor)
        
    ##All images resized and storing their scaling factors##
    all_images_resized = np.array(resized_images_list).astype(int)
    scaling_factors = np.array(scaling_factors_list)

    ctrl_labels = np.zeros(num_ctrl_cell_images)
    pntr_labels = np.ones(num_pntr_cell_images)
    all_labels = np.concatenate((ctrl_labels, pntr_labels), axis=0)

    ## Shuffle and split the dataset into training, validation, and testing sets and theire respective scaling factors
    X_train,fact_train,y_train,X_val,fact_val,y_val,X_test,fact_test,y_test = shuffle_and_split(all_images_resized,scaling_factors,all_labels)
    print("Training images shape:", X_train.shape)
    print("Training scaling factors shape:", fact_train.shape)
    print("Training labels shape:", y_train.shape)
    print("Validation images shape:", X_val.shape)
    print("Validation scaling factors shape:", fact_val.shape)
    print("Validation labels shape:", y_val.shape)
    print("Testing images shape:", X_test.shape)
    print("Testing scaling factors shape:", fact_test.shape)
    print("Testing labels shape:", y_test.shape)

    ##Normalization
    X_training = X_train/255.
    X_validation = X_val/255.
    X_testing = X_test/255.

    ########################################TRAINING###############################################
    input_shape = (median_image, median_image, 3)
    scales_shape = (1,)

    hypermodel = CNNHyperModel(input_shape=input_shape, scales_shape=scales_shape)
    
    #Hyperparameter tuning using Bayesian Optimization
    tuner = BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='dir',
        project_name='hyperparam_tuning'
    )

    tuner.search(
        [X_training, fact_train], y_train,
        epochs=10,
        validation_data=([X_validation, fact_val], y_val),
        verbose=1
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
    The hyperparameter search is complete. 
    The optimal number of convolutional layers is {best_hps.get('num_conv_layers')}.
    The optimal number of dense layers is {best_hps.get('num_dense_layers')}.
    The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
    """)

    fixed_params = {
        'num_conv_layers': best_hps.get('num_conv_layers'),
        'num_dense_layers': best_hps.get('num_dense_layers'),
        'learning_rate': best_hps.get('learning_rate'),
    }

    for i in range(fixed_params['num_conv_layers']):
        fixed_params[f'conv_{i+1}_filters'] = best_hps.get(f'conv_{i+1}_filters')

    for i in range(fixed_params['num_dense_layers']):
        fixed_params[f'dense_{i+1}_units'] = best_hps.get(f'dense_{i+1}_units')
        fixed_params[f'dropout_{i+1}_rate'] = best_hps.get(f'dropout_{i+1}_rate')

    ##Rebuilding the final model with the hyperparameters found
    final_model_builder = CNNHyperModel(input_shape=input_shape, scales_shape=scales_shape, fixed_params=fixed_params)
    model = final_model_builder.build()

    ##Training...
    history = model.fit([X_training, fact_train], y_train,
                            epochs=10,
                            validation_data=([X_validation, fact_val], y_val),
                            verbose=1)


    history_dict = history.history

    ##########################################VISUALIZATION##############################################
    ##Visualization of the training history##
    plt.figure(figsize=(12, 6))

    epochs = range(1, len(history_dict['loss']) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict['loss'], 'bo-', label='Training loss')
    plt.plot(epochs, history_dict['val_loss'], 'ro-', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict['accuracy'], 'bo-', label='Training accuracy')
    plt.plot(epochs, history_dict['val_accuracy'], 'ro-', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    ######################################EVALUATION ON THE TEST SET##########################################
    predictions = model.predict([X_testing,fact_test])
    print(accuracy_score(y_test,predictions.round()))

    conf_matrix = tf.math.confusion_matrix(labels=y_test, predictions=predictions.round(), num_classes=2)
    print('Confusion Matrix: ', conf_matrix)

    conf_matrix_np = conf_matrix.numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_np, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    ###########################################K-FOLD CROSS-VALIDATION##############################################
    #Data
    X_data = all_images_resized/255.
    fact_data = scaling_factors
    y_data = all_labels

    k = 8
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = []
    for train_index, val_index in kf.split(X_data):
        X_train, X_val = X_data[train_index], X_data[val_index]
        fact_train, fact_val = fact_data[train_index], fact_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        model = CNNHyperModel(input_shape=input_shape, scales_shape=scales_shape, fixed_params=fixed_params).build()

        ##Training...
        history = model.fit([X_train, fact_train], y_train,
                            epochs=15,
                            validation_data=([X_val, fact_val], y_val),
                            verbose=1)
        ##Evaluation
        score = model.evaluate([X_val, fact_val], y_val, verbose=0)
        scores.append(score[1])


    ##Average Accuracy for the presentation
    average_accuracy = np.mean(scores)
    print(f'Average accuracy: {average_accuracy}')
