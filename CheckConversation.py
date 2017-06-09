"""
To run this demo, you only need a protobuf containing the model graph + weights.

You can find the face detector model here :
s3://affdex-classifier-train/experiments/deeplearning/sci-293-faster-rcnn/models/zf4_tiny_3900000.pb

Then you can run the script as follows :
python cam_demo_pb.py --pb models/zf4_tiny_3900000.pb
"""
import numpy as np
import cv2, argparse, cv
import tensorflow as tf
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool
import vadcollector

def checkVideo(path):
    

if __name__ == "__main__":
    graph = load_model_pb("./zf4_tiny_3900000.pb")
    sess = tf.Session(graph=graph)
    print(checkForFace("notMoving.mp4"))
