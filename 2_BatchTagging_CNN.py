import os

import numpy as np
import pandas as pd
### Avoid certificat error (source: https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)
import requests

requests.packages.urllib3.disable_warnings()

from keras.applications import inception_resnet_v2

from keras.preprocessing import image

img_width, img_height = 331, 331
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

import fnmatch

from shutil import copyfile



# only mac
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

# from tensorflow.python.keras import backend as k


home_path = '/Users/seo-b/'
home_path = '/home/alan/'
default_path = home_path + 'Dropbox/KIT/FlickrEU/deepGreen'



#!export HIP_VISIBLE_DEVICES=0,1 #  For 2 GPU training
# os.environ['HIP_VISIBLE_DEVICES'] = '0,1'
os.environ['HIP_VISIBLE_DEVICES'] = '1'

# Seattle Middle Fork
model_json = "Model/InceptionResnetV2_retrain_Seattle_architecture_dropout0.3.json"

dataname = "MiddleFork"
photo_path_base = home_path + 'Dropbox/KIT/FlickrEU/Seattle/Seattle_TaggedData_BigSize/All images/MiddleFork_AllPhotos/' # Middle Fork

#dataname = "MountainLoop"
#photo_path_base = home_path + 'Dropbox/KIT/FlickrEU/Seattle/Seattle_TaggedData_BigSize/All images/MountainLoop_AllPhotos/' # Mountain Loop

out_path_base = "/DATA10TB/FlickrSeattle_Tagging_Feb2021/"
# out_path_base = "/SSDSATA1TB/FlickrTagging/"
#out_path_base = home_path + "Downloads/FlickrTagging/"


# Class #0 = backpacking
# Class #1 = birdwatching
# Class #2 = boating
# Class #3 = camping
# Class #4 = fishing
# Class #5 = flooding
# Class #6 = hiking
# Class #7 = horseriding
# Class #8 = mtn_biking
# Class #9 = noactivity
# Class #10 = otheractivities
# Class #11 = pplnoactivity
# Class #12 = rock climbing
# Class #13 = swimming
# Class #14 = trailrunning

classes = ["backpacking", "birdwatching", "boating", "camping", "fishing", "flooding", "hiking", "horseriding",
           "mtn_biking", "noactivity", "otheractivities", "pplnoactivity", "rock climbing", "swimming",
           "trailrunning"]


modelname = "InceptionResnetV2_dropout30_noweighting"
trainedweights_name = "../Seattle/Seattle_TaggedData_BigSize/ModelAndTrained weights/InceptionResnetV2_Seattle_retrain_instagram_15classes_Okt2019_val_acc0.88.h5"

#modelname = "InceptionResnetV2_dropout30_weighting"
#trainedweights_name = "../Seattle/Seattle_TaggedData_BigSize/ModelAndTrained weights/InceptionResnetV2_Seattle_retrain_instagram_15classes_Weighted_Dec2020_val_acc0.87_redone_for_traininghistory.h5"

os.chdir(default_path)

out_path = out_path_base + modelname + "/" + dataname + "/"
# out_path = out_path_base +  "/" + dataname + "/"

prediction_batch_size = 512  # to increase the speed of tagging .
# num+ber of images for one batch prediction

top = 10  # print top-n classes




classes_arr = np.array(classes)
# # Imagenet class labels
# imagenet_labels_filename = "Data/imagenet_class_index.json"
# with open(imagenet_labels_filename) as f:
#     CLASS_INDEX = json.load(f)
#
# classlabel = []
# for i in range(CLASS_INDEX.__len__()):
#     classlabel.append(CLASS_INDEX[str(i)][1])
# classes = np.array(classlabel)

num_classes = len(classes)


##### Predict

# Load the retrained CNN model

# Model reconstruction from JSON file
# with open(model_json, 'r') as f:
#    model_trained = model_from_json(f.read())



model_trained = inception_resnet_v2.InceptionResNetV2(include_top=False, weights='imagenet',input_tensor=None, input_shape=(img_width, img_height, 3))
x = model_trained.output
x = GlobalAveragePooling2D()(x) # before dense layer
x = Dense(1024, activation='relu')(x)
predictions_new = Dense(num_classes, activation='softmax', name='softmax')(x)
model_trained = Model(inputs=model_trained.input, outputs=predictions_new)

# Load weights into the new model
model_trained.load_weights(trainedweights_name)

# model_final = multi_gpu_model(model_final, gpus=2, cpu_merge=True, cpu_relocation=False)


def onlyfolders(path):
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            yield file


def onlyfiles(path):
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            yield file



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



foldernames = os.listdir(photo_path_base)

# f_idx = 0

for f_idx in range(0, len(foldernames)):

    foldername = foldernames[f_idx]
    print(f_idx)
    print(foldername)
    photo_path_aoi = photo_path_base + "/" + foldername

    ### Read filenames
    # filenames = os.listdir(photo_path_aoi)
    filenames = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(photo_path_aoi)) for f in fn]

    if len(filenames) == 0:
        continue  # skip the folder

    filenames1 = fnmatch.filter(filenames, "*.jpg")
    filenames2 = fnmatch.filter(filenames, "*.JPG")

    filenames = filenames1 + filenames2

    filenames = filenames1 + filenames2

    base_filenames = list(map(os.path.basename, filenames))

    n_files = len(filenames)

    prediction_steps_per_epoch = int(np.ceil(n_files / prediction_batch_size))

    # load all images into a list
    batch_size_folder = min(n_files, prediction_batch_size)  # n_files can be smaller than the batch size

    for step_start_idx in range(0, n_files, batch_size_folder):

        end_idx = min(step_start_idx + batch_size_folder, n_files)

        print(step_start_idx)
        print(end_idx)

        if step_start_idx == end_idx:

            filenames_batch = [filenames[step_start_idx]]
        else:

            filenames_batch = filenames[step_start_idx:end_idx]

        bsize_tmp = min(batch_size_folder, len(filenames_batch))  # for the last batch

        images = []

        for img_name in filenames_batch:
            print(img_name)
            img_name = os.path.join(photo_path_aoi, img_name)

            # load an image in PIL format
            img = image.load_img(img_name, target_size=(img_width, img_height))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)

            # prepare the image (normalisation for channels)
            img_preprocessed = inception_resnet_v2.preprocess_input(img.copy())
            images.append(img_preprocessed)

        images_vstack = np.vstack(images)

        # stack up images list to pass for prediction
        predictions = model_trained.predict(images_vstack, batch_size=bsize_tmp)

        # predictions.shape

        ## top selected classes
        top_classes_idx_arr = np.argsort(predictions)[:, ::-1][:, :top]

        top_classes_arr = classes_arr[top_classes_idx_arr]
        print(top_classes_arr)

        # create an empty array
        top_classes_probs_arr = np.empty([bsize_tmp, top])
        top_classes_probs_arr[:] = 0

        for i in range(0, bsize_tmp):
            top_classes_probs_arr[i,] = predictions[i, [top_classes_idx_arr[i,]]]

        # np.argsort(predictions)[:, ::-1][:,:top][0, :]

        # chainlink_fence', 'worm_fence', 'lakeside', 'seashore', 'stone_wall', 'cliff', 'breakwater']
        # Out[61]: array([489, 912, 975, 978, 825, 972, 460])
        top_classes_arr[0, :]
        top_classes_probs_arr[0, :]

        predicted_class_v = top_classes_arr[:, 0] # top1
        predicted_class_top2_v = top_classes_arr[:, 1] # top2

        #print('Predicted:', predicted_class_v)


        # 2nd-level
        # kind of equivalent to `sapply()' in R
        def foo_get_predicted_filename(x, x2):
            return (out_path + "/" + "" + foldername + "/" + x)
            #return (out_path + "Result/" + "ClassifiedPhotos/" + "/" + x + "/2ndClass_" +x2 )


        predicted_filenames = list(map(foo_get_predicted_filename, predicted_class_v, predicted_class_top2_v))
        save_folder_names = list(map(os.path.basename, predicted_filenames))

        # create necessary folders
        # for i in range(0, n_files):
        #     if not (os.path.exists(save_folder_names[i])):
        #         os.makedirs(save_folder_names[i], exist_ok=False)

        for i in range(0, bsize_tmp):

            save_folder = predicted_filenames[i]
            print(save_folder)

            if not (os.path.exists(save_folder)):
                os.makedirs(save_folder, exist_ok=False)

            copyfile(filenames_batch[i], predicted_filenames[i] + '/' + os.path.basename(filenames_batch[i]) )

        arr_tmp = pd.DataFrame(np.concatenate((top_classes_arr, top_classes_probs_arr), axis=1))

        if step_start_idx == 0:
            arr_aoi = arr_tmp
        else:
            arr_aoi = np.concatenate((arr_aoi, arr_tmp), axis=0)

    # Write csv files

    name_csv = out_path + "/" + "/CSV/" + foldername + ".csv"
    if not (os.path.exists(os.path.dirname(name_csv))):
        os.makedirs(os.path.dirname(name_csv), exist_ok=False)

    # Write a Pandas data frame
    df_aoi = pd.concat([pd.DataFrame(base_filenames), pd.DataFrame(arr_aoi)], axis=1)
    header = np.concatenate(
        (["Filename"], ["Top1", "Top2", "Top3", "Top4", "Top5", "Top6", "Top7", "Top8", "Top9", "Top10"],
         ["Prob1", "Prob2", "Prob3", "Prob4", "Prob5", "Prob6", "Prob7", "Prob8", "Prob9", "Prob10"]))

    df_aoi.columns = header
    df_aoi.to_csv(name_csv, index=False, columns=header)
