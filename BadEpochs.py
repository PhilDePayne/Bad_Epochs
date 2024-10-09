"""
0. No cleaning
1. AR (no interpolation)
2. AR (no interpolation) + ICA (EOG)
3. AR (no interpolation) + ICA (EOG + EMG)
4. AR (interpolation)
5. AR (interpolation) + ICA (EOG)
6. AR (interpolation) + ICA (EOG + EMG)
7. RANSAC
8. RANSAC + ICA (EOG)
9. RANSAC + ICA (EOG + EMG)
10. FASTER (no interpolation)
11. FASTER (no interpolation) + ICA (EOG)
12. FASTER (no interpolation) + ICA (EOG + EMG)
13. FASTER (interpolation)
14. FASTER (interpolation) + ICA (EOG)
15. FASTER (interpolation) + ICA (EOG + EMG)
"""

methods_quantity = 16
method = 13

EOG_proxy = 'EEG 001'
repeat_count = 10
log_info = False

nb_classes = 4
classes_grouping = 1
# 0 - audio/visual, 1 - left/right

#====== IMPORTS ======#

import numpy as np
import statistics
import math
from collections import Counter

# mne imports
import mne
from mne import io
from mne.datasets import sample
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

# EEGNet-specific imports
from EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# autoreject imports
import autoreject
from autoreject import Ransac
from autoreject.utils import interpolate_bads

# FASTER imports
from mne_faster import (find_bad_channels, find_bad_epochs,
                        find_bad_components, find_bad_channels_in_epochs)

import os

#====== FUNC ======#
def find_AR_interpolated(reject_log):
    
    count = 0
    for column in reject_log.labels.T:
        if 2 in column:
            count += 1
            
    return count         

def clean_data_AR(epochs):
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], 
                               random_state=11,
                               thresh_func='bayesian_optimization',
                               n_jobs=2, verbose=log_info)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    if log_info:
        reject_log.plot('horizontal')
        print("Interpolated: " + str(find_AR_interpolated(reject_log)))
    
    if(interpolate == False):
        epochs_ar = epochs[~reject_log.bad_epochs]
        
    if(ICA_mode != 0):
        return perform_ICA(epochs_ar), reject_log.bad_epochs
    else:
        return epochs_ar, reject_log.bad_epochs

def clean_data_RANSAC(epochs):
    ransac = Ransac(verbose=log_info, n_jobs=1)
    epochs_ransac = ransac.fit_transform(epochs)
    
    if(ICA_mode != 0):
        return perform_ICA(epochs_ransac)
    else:
        return epochs_ransac

def clean_data_FASTER(epochs):
    epochs_FASTER = epochs.copy()
    if(interpolate):
        epochs_FASTER.info['bads'] = find_bad_channels(epochs_FASTER, eeg_ref_corr=True)
        if len(epochs_FASTER.info['bads']) > 0:
            epochs_FASTER.interpolate_bads()

    # Step 2: mark bad epochs
    bad_epochs = find_bad_epochs(epochs_FASTER)
    if len(bad_epochs) > 0:
        epochs_FASTER.drop(bad_epochs)

    # Step 3: mark bad ICA components (using the build-in MNE functionality for this)
    if(ICA_mode != 0):
        ica = mne.preprocessing.ICA(0.99).fit(epochs_FASTER)
        if(ICA_mode == 2):
            muscle_indices, muscle_scores = ica.find_bads_muscle(epochs,
                                                         threshold = 0.7)
            ica.exclude += muscle_indices
        ica.exclude += find_bad_components(ica, epochs_FASTER, proxy_name=EOG_proxy)
        
        ica.apply(epochs_FASTER)
        # Need to re-baseline data after ICA transformation
        epochs_FASTER.apply_baseline(epochs_FASTER.baseline)

    # Step 4: mark bad channels for each epoch and interpolate them.
    if(interpolate):
        ret = 0
        bad_channels_per_epoch = find_bad_channels_in_epochs(epochs_FASTER, eeg_ref_corr=True)
        for i, b in enumerate(bad_channels_per_epoch):
            if len(b) > 0:
                ret += 1
                ep = epochs_FASTER[i]
                ep.info['bads'] = b
                ep.interpolate_bads() 
                epochs_FASTER._data[i, :, :] = ep._data[0, :, :]
    if log_info:
        print("Interpolated " + str(ret) + " epochs") 
            
    return epochs_FASTER, bad_epochs

def perform_ICA(epochs):
    ica = ICA(n_components=20, max_iter="auto", random_state=97)
    ica.fit(epochs)
    
    eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name=EOG_proxy,
                                                measure = 'correlation', threshold = 0.7)
    muscle_indices, muscle_scores = ica.find_bads_muscle(epochs,
                                                         threshold = 0.7)
    ica.exclude = eog_indices
    
    if(ICA_mode == 2):
        ica.exclude += muscle_indices
    
    print(ica.exclude)
    ica.apply(epochs)
    epochs.apply_baseline(epochs.baseline)
    
    return epochs

def set_params(mode):
    global cleaning_method
    global ICA_mode
    global interpolate
    
    if(mode < 7):
        cleaning_method = 'AR'
        if(mode < 4):
            interpolate = False
        else:
            interpolate = True
    elif(mode < 10):
        cleaning_method = 'RANSAC'
    else:
        cleaning_method = 'FASTER'
        if(mode < 13):
            interpolate = False
        else:
            interpolate = True

    if(mode % 3 == 0):
        ICA_mode = 2
    elif(mode % 3 == 1):
        ICA_mode = 0
    else:
        ICA_mode = 1   

def balance(epochs, labels):
    # Count appearances of each class
    c = np.bincount(labels - 1)
    n = c.min()
    # Accumulated counts for each class shifted one position
    cs = np.roll(np.cumsum(c), 1)
    cs[0] = 0
    # Compute appearance index for each class
    i = np.arange(len(labels)) - cs[labels - 1]
    # Mask excessive appearances
    m = i < n
    # Return corresponding tokens
    return epochs[m], labels[m]

def plot(history, name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(name)
    plt.close()

def mean_square_error(raw, clean):
    squared_errors = (raw - clean) ** 2
    
    mse = np.mean(squared_errors)
    
    return mse

def root_mean_square_error(raw, clean):
    return math.sqrt(mean_square_error(raw, clean))

def signal_noise_ratio(raw, clean):
    squared_raw = raw ** 2
    
    squared_errors = (raw - clean) ** 2
    
    snr = 10 * np.emath.logn(10, squared_raw/squared_errors)
    
    return np.mean(snr)

def percentage_root_mean_square_difference(raw, clean):
    abs_errors = np.absolute(raw - clean)
    
    abs_raw = np.absolute(raw)
    
    prd = np.mean(abs_errors / abs_raw) * 100
    
    return prd
    
def run_EEGNet(epochs, clean_epochs):
    
    labels = clean_epochs.events[:, -1]
    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    X = clean_epochs.get_data()*1000 
    y = labels
    
    X, y = balance(X, y)

    # format is in (trials, channels, samples)
    kernels, chans, samples = 1, 60, 151

    size = y.shape[0]
    halfSize = int(size * 0.5)
    tQuarterSize = int(halfSize + (halfSize * 0.5))

    
    # take 50/25/25 percent of the data to train/validate/test
    X_train      = X[0:halfSize,]
    Y_train      = y[0:halfSize]
    X_validate   = X[halfSize:tQuarterSize,]
    Y_validate   = y[halfSize:tQuarterSize]
    X_test       = (epochs.get_data()*1000)[tQuarterSize:,]
    Y_test       = (epochs.events[:,-1])[tQuarterSize:]

    ############################# EEGNet portion ##################################

    # convert labels to one-hot encodings.
    Y_train      = np_utils.to_categorical(Y_train-1)
    Y_validate   = np_utils.to_categorical(Y_validate-1)
    Y_test       = np_utils.to_categorical(Y_test-1)

    # convert data to NHWC (trials, channels, samples, kernels) format. Data 
    # contains 60 channels and 151 time-points. Set the number of kernels to 1.
    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

    if log_info:
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes = nb_classes, Chans = chans, Samples = samples, 
                   dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
                   dropoutType = 'Dropout')

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])

    # count number of parameters in the model
    numParams    = model.count_params()    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.keras', verbose=int(log_info == True),
                                   save_best_only=True)

    ###############################################################################
    # if the classification task was imbalanced (significantly more trials in one
    # class versus the others) you can assign a weight to each class during 
    # optimization to balance it out. This data is approximately balanced so we 
    # don't need to do this, but is shown here for illustration/completeness. 
    ###############################################################################

    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    #
    if nb_classes == 4:
        class_weights = {0:1, 1:1, 2:1, 3:1}
    elif nb_classes == 2:
        class_weights = {0:1, 1:1}

    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
    # Riemannian geometry classification (below)
    ################################################################################
    fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 300, 
                            verbose = int(log_info == True), validation_data=(X_validate, Y_validate),
                            callbacks=[checkpointer], class_weight = class_weights)

    # load optimal weights
    model.load_weights('/tmp/checkpoint.keras')
    
    dot_img_file = '/tmp/model_1.png'
    np_utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

    ###############################################################################
    # can alternatively used the weights provided in the repo. If so it should get
    # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
    # system.
    ###############################################################################

    WEIGHTS_PATH = './Weights/Vendor.h5'
    #model.load_weights(WEIGHTS_PATH)

    ###############################################################################
    # make prediction on test set.
    ###############################################################################

    probs       = model.predict(X_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == Y_test.argmax(axis=-1))
    #print("Classification accuracy: %f " % (acc))
    plt.clf()
    plot(fittedModel, str(method) + '.png')
    
    return acc

def calculate_ERP(epochs, clean_epochs, class_nr):
    raw = epochs[list(filter(lambda x: epochs.event_id[x] == class_nr, epochs.event_id))[0]].average()
    fig1 = raw.plot(show=False)
    fig1.savefig(str(method) + "_class" + str(class_nr) + "_raw_ERP_" + ".png")
    
    clean = clean_epochs[list(filter(lambda x: clean_epochs.event_id[x] == class_nr, clean_epochs.event_id))[0]].average()
    fig2 = clean.plot(show=False)
    fig2.savefig(str(method) + "_class" + str(class_nr) + "_clean_ERP_" + ".png")
    
    raw_data = raw.get_data()
    clean_data = clean.get_data()
    
    mse = (mean_square_error(raw_data, clean_data))
    rmse = (root_mean_square_error(raw_data, clean_data))
    snr = (signal_noise_ratio(raw_data, clean_data))
    prd = (percentage_root_mean_square_difference(raw_data, clean_data))
    
    return mse, rmse, snr, prd
#====== PARAMS ======#

cleaning_method = 'AR' # AR, RANSAC, FASTER
ICA_mode = 0 # 0-no ICA, 1-EOG, 2-EOG+EMG (2. not available for FASTER)
interpolate = False # not available for RANSAC

#====== MAIN ======#
def main_process():
    set_params(method)

    # while the default tensorflow ordering is 'channels_last' we set it here
    # to be explicit in case if the user has changed the default ordering
    K.set_image_data_format('channels_last')

    # Set parameters and read data
    raw_fname = os.path.join('./Data',  'sample_audvis_filt-0-40_raw.fif')
    event_fname = os.path.join('./Data', 'sample_audvis_filt-0-40_raw-eve.fif')
    tmin, tmax = -0., 1
    event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

    # Setup for reading the raw data
    raw = io.Raw(raw_fname, preload=True, verbose=log_info)
    raw.filter(2, None, method='iir')  # replace baselining with high-pass
    events = mne.read_events(event_fname)
    
    #print(events)

    raw.info['bads'] = ['MEG 2443']  # set bad channels

    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                           exclude='bads')

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                        picks=picks, baseline=None, preload=True, verbose=log_info)
    
    if nb_classes == 2:
        if classes_grouping == 0:
            epochs = mne.epochs.combine_event_ids(epochs, ['aud_l', 'aud_r'], {'aud': 6})
            epochs = mne.epochs.combine_event_ids(epochs, ['vis_l', 'vis_r'], {'vis': 7})
            epochs = mne.epochs.combine_event_ids(epochs, ['aud'], {'audio': 1})
            epochs = mne.epochs.combine_event_ids(epochs, ['vis'], {'visual': 2})
        elif classes_grouping == 1:
            epochs = mne.epochs.combine_event_ids(epochs, ['aud_l', 'vis_l'], {'l': 6})
            epochs = mne.epochs.combine_event_ids(epochs, ['aud_r', 'vis_r'], {'r': 7})
            epochs = mne.epochs.combine_event_ids(epochs, ['l'], {'left': 1})
            epochs = mne.epochs.combine_event_ids(epochs, ['r'], {'right': 2})

    clean_epochs = None

    if(method == 0):
        clean_epochs = epochs
    elif(cleaning_method == 'AR'):
        clean_epochs, bad_epochs = clean_data_AR(epochs)    
    elif(cleaning_method == 'RANSAC'):
        clean_epochs = clean_data_RANSAC(epochs)
    elif(cleaning_method == 'FASTER'):
        clean_epochs, bad_epochs = clean_data_FASTER(epochs)
        
    mse_first, rmse_first, snr_first, prd_first = calculate_ERP(epochs, clean_epochs, 1)
    
    mse_second, rmse_second, snr_second, prd_second = calculate_ERP(epochs, clean_epochs, 2)
    
    mse = (mse_first + mse_second) / 2
    rmse = (rmse_first + rmse_second) / 2
    snr = (snr_first + snr_second) / 2
    prd = (prd_first + prd_second) / 2
    
    return epochs, clean_epochs, mse, rmse, snr, prd

#====== TEST ======#
file_path = "./res.txt"
#print(main_process())


with open(file_path, 'w') as file:  
    for x in range(10, methods_quantity):
        method = x
        file.write(str(x))
        file.write(".\n")
        acc_sum = []
        raw, clean, mse, rmse, snr, prd = main_process()
        file.write(str(mse) + " " + str(rmse) + " " + str(snr) + " " + str(prd) + "\n")
        
        for y in range(repeat_count):
            acc_sum.append(run_EEGNet(raw, clean))
            print(y + 1, "/", repeat_count)
        file.write(str(acc_sum) + '\n')   
        file.write(str(statistics.mean(acc_sum)) + '\n') 
