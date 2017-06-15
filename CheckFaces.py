"""
To run this demo, you only need a protobuf containing the model graph + weights.

You can find the face detector model here :
s3://affdex-classifier-train/experiments/deeplearning/sci-293-faster-rcnn/models/zf4_tiny_3900000.pb

Then you can run the script as follows :
python cam_demo_pb.py --pb models/zf4_tiny_3900000.pb
"""
import numpy as np
import cv2, argparse
import tensorflow as tf
import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool



def load_model_pb(pb_path):
    with tf.gfile.GFile(pb_path,'r') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, input_map=None,
            return_elements=None, name="")
    return graph

def test_image(sess, image):
    prob_tensor_name = "postprocess/Gather_1:0"
    bbox_tensor_name = "postprocess/Gather:0"
    feed_dict = {"image:0": image,
                 "dst_shape:0": [400, 666],
                 "prob_threshold:0": 0.5}
    prob, bbox = sess.run([prob_tensor_name, bbox_tensor_name], feed_dict=feed_dict)
    box_score = np.hstack((bbox, prob[:, np.newaxis])).astype(np.float32, copy=False)
    return box_score

def getFrame(cap):
    ret = False
    counter = 0
    while ret == False:
        ret, frame = cap.read()
        if counter > 100:
            return (False, frame)
        counter += 1
    return (True, frame)

def checkForFace(path, graph, sess, skipFrames=None):
    df_columns = ['time_msec', 'faceid', 'x1', 'y1', 'x2', 'y2', 'score']
    try:
        cap = cv2.VideoCapture(path)
    except:
        print("Couldn't open: ", path)
        return
    if skipFrames == None:
        skipFrames = int(cap.get(cv2.PROP_FRAME_COUNT)/100)
    if skipFrames == 0:
        skipFrames = 200
    if not cap.isOpened():
        raise RuntimeError ("Failed to open video file")

    counter = 0
    threshold = 5
    while cap.grab():
        for i in range(skipFrames-1):
            ret, curframe = cap.read()
        ret, curframe = getFrame(cap)
        if not ret:
            break
        _, frame = cap.retrieve()
        time_msec = cap.get( cv2.PROP_POS_MSEC )   #optional
        box_score = test_image(sess, frame)
        for indx, abox in enumerate(box_score):
            ser = pd.Series(data=[time_msec, indx]+abox.tolist(), index=df_columns)
            if ser["score"] > .85:
                counter += 1
            if counter > threshold:
                print("Enough frames of faces for keeping")
                cap.release()
                return True
    cap.release()
    return False

if __name__ == "__main__":
    # args = parse_args()
    graph = load_model_pb("./zf4_tiny_3900000.pb")
    sess = tf.Session(graph=graph)
    print(checkForFace("notMoving.mp4", graph, sess))
    # pool = Pool(processes=int(5))
    # for file in [f for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]:
    #     pool.apply_async(checkForFace, args=(PATH+file,))
    # pool.close()
    # pool.join()
