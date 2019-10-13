# import the necessary packages
from imutils import face_utils
from keras.models import load_model
from fr_utils import *
from inception_blocks_v2 import *
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import glob
import os
import shutil

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def initialize():
    #load_weights_from_FaceNet(FRmodel)
    #we are loading model from keras hence we won't use the above method
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("target/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        database[identity] = []
        for image in glob.glob(file + "/*"):
            database[identity].append(fr_utils.img_path_to_encoding(image, FRmodel))
    return database

def recognize_face(face_descriptor):
    encoding = img_to_encoding(face_descriptor, FRmodel)
    min_dist = 100
    identity = None

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc_list) in database.items():
        for db_enc in db_enc_list:
            # Compute L2 distance between the target "encoding" and the current "emb" from the database.
            dist = np.linalg.norm(db_enc - encoding)

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
            if dist < min_dist:
                min_dist = dist
                identity = name
    
    print(identity + "  " + "{:.4f}".format(min_dist))
    return identity, min_dist

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--identification", default="face-rec_Google.h5",
        help="path to Google pre-trained model")
ap.add_argument("-v", "--video", default="./video/koren4.mov",
        help="the name of the video you want to analyze")
ap.add_argument("-p", "--prototxt", default="deploy.prototxt.txt",
        help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", default="res10_300x300_ssd_iter_140000.caffemodel",
        help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model(Face detection)...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] loading model(Face identification)...")
FRmodel = load_model('face-rec_Google.h5')

database = initialize()
create_folder("./found/")

# loop over the frames from the video stream
while True:
    # load all the images of individuals to recognize into the database
    for file in glob.glob("./face/*"):
        fr_utils.img_path_to_encoding(file, FRmodel)
        
        min_dist = 100
        identity = None

        # Loop over the database dictionary's names and encodings.
        for (name, db_enc_list) in database.items():
            for db_enc in db_enc_list:
                # Compute L2 distance between the target "encoding" and the current "emb" from the database.
                dist = np.linalg.norm(db_enc - encoding)

                # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
                if dist < min_dist:
                    min_dist = dist
                    identity = name
    
        print(identity + "  " + "{:.4f}".format(min_dist))
        
        if min_dist < 0.06:
            create_folder("./found/" + identity)
            shutil.move("./face/" + file, "./found/" + identity "/" + file)
        else:
            os.remove("./face/" + file)
