#!/usr/bin/python
# Example usage: python2 youtube_scraper.py -q "carpool karaoke" -n 3 -v

# For documentation and requirements, refer to Youtube Scraper.pdf

import os
from apiclient.discovery import build_from_document, build
from apiclient.errors import HttpError
from oauth2client.client import flow_from_clientsecrets, AccessTokenCredentials
from oauth2client.file import Storage
from oauth2client.tools import argparser, run_flow
import argparse
from retrying import retry
import pafy
from multiprocessing import Pool
import logging
import csv
import cv2
import httplib2
import shutil
import sys
from youtube2srt import cli
import pandas as pd
import numpy as np
import HTMLParser
import time
import ffmpy
import boto3
import pdb
import re
from tqdm import tqdm
import tensorflow as tf
from CheckFaces import load_model_pb, checkForFace
import VadCollector
import traceback
# import MovementDetect
from datetime import datetime
import timeout_decorator

# Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
# tab of
#   https://cloud.google.com/console
# Please ensure that you have enabled the YouTube Data API for your project.
# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account.
YOUTUBE_READ_WRITE_SCOPE = "https://www.googleapis.com/auth/youtube"
DEVELOPER_KEY = "AIzaSyBjhRlaxeYd0_b27J0JmosTuf1H5DsN3O4"
YOUTUBE_READ_WRITE_SSL_SCOPE = "https://www.googleapis.com/auth/youtube.force-ssl"
MISSING_CLIENT_SECRETS_MESSAGE = ""
YOUTUBE_API_SERVICE_NAME = "youtube"
DATA_BUCKET_NAME = "youtube-video-data"
YOUTUBE_API_VERSION = "v3"
YOUTUBE_MAX_SEARCH_REASULTS = 50
SAVES_PER_SYNC = 20
BACKUP_EVERY_N_VIDEOS = 5  # Backup to the CSV every this number of videos
FACE_DETECTION_MODEL = "./zf4_tiny_3900000.pb"  # Face tracking model
NUM_VIDS = 10  # Number of videos that should be downloaded on default
SPEECH_TRHESHHOLD = .5  # percentage of video that contains speech
# The tolerance of the speech model. Options are 1, 2, or 3.
CONVERSATION_AGGRESSIVENESS = 2
CSV_PATH = None  # worker_id.csv should be in out/
QUERIES = []  # Queries given as command line arguments split up.
OPEN_ON_DOWNLOAD = False # Should the program open the videos once downloaded?
WORKER_UUID = open("Worker_Key.key").readlines()[0].strip()
MASTER_PROCESS = open("Worker_Key.key").readlines()[1].strip() == "True" # Should this process take it upon itself to join cloud csv's
bucket, graph, sess = None, None, None  # Initializing variables globally
parser = HTMLParser.HTMLParser()
s3 = boto3.resource('s3')

# Create dataframe
columns = ["Url", "UUID", "Date Updated", "Query", "Format", "File Path", "Dimensions",
            "Title", "Description", "Duration",
           "Captions", "Size(bytes)", "Keywords", "Viewcount", "Faces", "Conversation", "Author", "Uploaded", "Worker"]

columnTypes = [str, str, str, str, str, str, str, str, str, float, str, float, str, float, bool, bool, str, bool, str]
information_csv = pd.DataFrame(columns=columns)
backup_counter = 0
sync_counter = 0

def parse_args():
    """
    Creates command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Perform a video search and sorts the results into the proper.')
    parser.add_argument('-q,--query', action="store", dest="query",
                        help='Search term to use, separated by comma', type=str)
    parser.add_argument("-t, --num_threads", action="store",
                        dest="num_threads", help="Number of concurrent Threads", default=3)
    parser.add_argument("-n, --num_videos", action="store", dest="num_vids",
                        help="Number of videos for each keyword that will be downloaded", default=10)
    parser.add_argument("-v, --verbose", help="Verbose output",
                        dest="verbose", action="store_true", default=False)
    parser.add_argument("-r, --rebuild", help="Rebuild the search cache?",
                        action="store_true", dest="rebuild", default=False)
    parser.add_argument("-b, --backup_every", help="Backup to CSV every N number of videos",
                        action="store", dest="backup_every", default=5)
    parser.add_argument("--open", help="Open every new video on download",
                        action="store_true", dest="openOnDownload", default=False)
    parser.add_argument("--categorize", help="Categorize into Faces, Conversation, multimodal, and trash.",
                        action="store_true", dest="categorize", default=False)
    parser.add_argument("--convert", help="Convert videos.",
                        action="store_true", dest="convert", default=False)
    parser.add_argument("--clean", help="Cleans the downloads directory",
                        action="store_true", dest="clean", default=False)
    parser.add_argument("--upload", help="Uploads to S3",
                        action="store_true", dest="upload", default=False)

    args = parser.parse_args()
    return args

def saveCSVToBoto3():
    global bucket, s3, information_csv
    print_and_log("Syncing to s3...")
    fileName = WORKER_UUID+'.csv'
    if bucket == None:
        bucket = s3.Bucket(DATA_BUCKET_NAME)
    bucket.put_object(
        Bucket=DATA_BUCKET_NAME,
        Body=open("out/"+fileName, 'rb'),
        Key='Workers/'+fileName
        )
    if MASTER_PROCESS:
        try:
            print_and_log("I AM MASTER -> Combining online CSVs...")
            csvs = []
            client = boto3.client('s3')
            master_df = pd.DataFrame(columns=columns)
            for item in client.list_objects(Bucket=DATA_BUCKET_NAME, Prefix='Workers')["Contents"]:
                if 'csv' in item["Key"] and "master" not in item["Key"] and "Archive" not in item["Key"]:
                    name = item["Key"]
                    name = name[name.rfind('/')+1:]
                    csvs.append(name)
                    s3.meta.client.download_file(DATA_BUCKET_NAME, item["Key"], "out/tmp/"+name)
            master_df = pd.concat([pd.read_csv("out/tmp/"+name) for name in csvs])
            master_df = master_df.sort_values(['Query', "File Path", "Uploaded", "Size(bytes)"])
            master_df.drop_duplicates(subset="UUID", inplace=True)
            master_df.to_csv("out/tmp/master.csv", index=False, encoding='utf-8')
            bucket.upload_file("out/tmp/master.csv", "Workers/Archive/master"+str(time.time())+".csv")
            bucket.upload_file("out/tmp/master.csv", "Workers/master.csv")
            moveFromTo("out/tmp/master.csv", "Workers/master.csv")
            information_csv = master_df
        except Exception, e:
            pdb.set_trace()
    else:
        s3.meta.client.download_file(DATA_BUCKET_NAME, "Workers/master.csv", CSV_PATH)
        master_df = master_df.sort_values(['Query'])
        master_df.drop_duplicates(subset="UUID", inplace=True)
        information_csv = master_df
    

def saveCSV(path):
    global sync_counter
    """
    Saves information_csv, as a csv, to the path in the argument provided
    """
    print_and_log("Saving the CSV...")
    try:
        with open(path, 'wb') as fout:
            information_csv.to_csv(path, index=False, encoding='utf-8')
        saveCSVToBoto3()
        print_and_log("Saved")
    except Exception, e:
        logging.error("Failed to save csv:"+str(e)+"\n"+traceback.format_exc())
    if sync_counter >= SAVES_PER_SYNC:
        saveCSVToBoto3()
        sync_counter = 0
    else:
        sync_counter += 1

def convertDataTypes():
    """
    Converts all the columns in information_csv to their proper
    datatype. Otherwise, they all become `object`
    """
    global information_csv
    assert len(columns) == len(columnTypes)
    for i in range(len(columns)):
        column = columns[i]
        information_csv[[column]] = information_csv[
            [column]].astype(columnTypes[i])

def is_empty_or_false(column):
    if information_csv[column].dtype == bool:
        return (information_csv[column].isnull() | (information_csv[column] == False))
    return (information_csv[column] == "") | (information_csv["File Path"].str.contains("nan"))

def recover_or_get_youtube_id_dictionary(args):
    """
    Used during the startup. It will either create a new csv or load the old one.
    If it loads an old csv, it will check the folder and update the csv based on
    which files are downloaded.
    """
    global information_csv, CSV_PATH
    # Create JSON file if not there
    CSV_PATH = os.path.join("out/", WORKER_UUID+'.csv')
    if not os.path.exists(CSV_PATH):
        information_csv = pd.DataFrame()
        information_csv.columns = columns
    else:
        information_csv = pd.read_csv(CSV_PATH)
    convertDataTypes()
    if set(information_csv.keys()) != set(columns): # Check CSV
        print_and_log("CSV and columns disagree")
        sys.exit()
    for key in QUERIES: # iterate through the queries
        try:
            # The number of queries in csv that haven't
            # been downloaded yet.
            num_ids_to_get = NUM_VIDS - len(information_csv[(is_empty_or_false("File Path")) &\
                        (information_csv['Query'].str.contains(key))]["Query"].tolist())
            if num_ids_to_get <= 0\
                        and not args.rebuild:
                logging.info("Found enough non-downloaded results with query:" + key +
                             " Using solely cached results.")
            else:
                if args.query != None:
                    logging.info("Didn't find enough non-downloaded results with query:" + key +
                                 " scraping now...")
                    scrape_id(key, num_ids_to_get)
        except:
            pdb.set_trace()

def convert_caption_to_str(trackList):
    """
    Converts a list of `Track` objects to a string
    """
    retStr = ""
    if trackList == None:
        return retStr
    for track in trackList:
        retStr += "Starts at " + str(track.start) + "s and lasts " + str(
            track.duration) + "s: " + parser.unescape(track.text) + "\n"
    # Make the characters solely ascii
    return retStr.encode('ascii', 'ignore').decode('ascii')

# @retry(wait_fixed=600000, stop_max_attempt_number=5)
def download_caption(video_id):
    """
    Downloads the captions if available and returns an infodict
    """
    global backup_counter
    if backup_counter < BACKUP_EVERY_N_VIDEOS:
        backup_counter += 1
    else:
        backup_counter = 0
        saveCSV(CSV_PATH)
    logging.info("Downloading caption at url: %s" %
                 ("youtube.com/watch?v="+video_id))
    capStr = convert_caption_to_str(cli.get_track(video_id, ['en', "en-GB"]))
    logging.info("Finished downloading caption at url: %s" %
                 ("youtube.com/watch?v="+video_id))
    return {"UUID": video_id, "Captions": capStr}

def findFile(uuid):
    mp4File = uuid+".mp4"
    webmFile = uuid+".webm"
    for root, subdirs, files in os.walk("out/"):
        for file in files:
            if mp4File == file:
                return str(root+"/"+mp4File)
            elif webmFile == file:
                return str(root+"/"+webmFile)
    return ""

def exists_in_boto3(path, search=False):
    '''
    Checks if file exists in boto3. If search=True, it will search for it
    doesn't find it at first.
    '''
    global bucket
    print(path)
    if path == "":
        return False
    if bucket == None:
        bucket = s3.Bucket(DATA_BUCKET_NAME)
    objs = list(bucket.objects.filter(Prefix=path))
    at_path = len(objs) > 0 and objs[0].key == path
    if not at_path and search:
        uid = path[path.rfind('.')+1:]
        client = boto3.client('s3')
        for item in client.list_objects(Bucket=DATA_BUCKET_NAME)["Contents"]:
            if uid in item["Key"]:
                return True
    return at_path

# TO-DO: Review this function
def get_attributes(uuid, requested_columns, HardReset=False, categorize=False):
    """
    Gets a list of attributes with all the possible failsafes I can think of.
    Assumptions:
    There are one or zero rows with the given uuid.
    There is internet.
    """
    global information_csv
    if len(requested_columns) == 0: # Empty query
        return []
    infoDict = {"UUID": uuid}
    row = None
    if uuid in information_csv["UUID"]: # Check if uuid exists in dataframe
        row = information_csv[information_csv["UUID"] == row][0]
        infoDict = row.to_dict()

    if "File Path" not in infoDict.keys() or infoDict["File Path"] == "" or os.path.exists(infoDict["File Path"]): # Confirm location
        infoDict["File Path"] = findFile(uuid)

    retArr = []
    
    path = infoDict["File Path"]
    vidInfo = None
    stream = None
    cap = None
    for column in requested_columns:
        if column in infoDict.keys() and not HardReset:
            retArr.append(infoDict[column])
        else:
            if column == "File Path": # Already confirmed earlier
                retArr.append(infoDict["File Path"])
            elif column == "Worker":
                retArr.append(WORKER_UUID)
            elif column == "Url":
                url = uid_to_url(uuid)
                infoDict["Url"] = uuid
                retArr.append(url)
            elif column == "UUID":
                retArr.append(uuid)
            elif column == "Date Updated":
                retArr.append(datetime.now().strftime('%Y/%m/%d %H:%M:%S'))
            elif column == "Format":
                format_ = infoDict["File Path"]
                format_ = format_[format_.rfind(".")+1:]
                retArr.append(format_)
            elif column == "File Path":
                retArr.append(infoDict["File Path"])
            elif column == "Dimensions":
                dim = ""
                if HardReset:
                    if vidInfo == None:
                        vidInfo = pafy.new(uid_to_url(uuid))
                    if stream == None:
                        stream = vidInfo.getbest(preftype='mp4')
                    dim = stream.resolution
                retArr.append(dim)
            elif column == "Query":
                if "Query" in infoDict.keys():
                    retArr.append(infoDict["Query"])
                else:
                    retArr.append("")
            elif column == "Duration":
                if HardReset:
                    if cap == None:
                        cap = cv2.VideoCapture(path)
                    retArr.append(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)/cap.get(5)) # 5=cv2.CAP_PROP_FPS
                elif "Duration" in infoDict.keys():
                    retArr.append(infoDict["Duration"])
                else:
                    retArr.append("")
            elif column == "Captions":
                if HardReset:
                    captions = download_caption(uuid)
                    retArr.append(captions)
                else:
                    retArr.append("")
            elif column == "Size(bytes)":
                size = -1
                if infoDict["File Path"] != "":
                    size = os.path.getsize(infoDict["File Path"])
                retArr.append(size)
                infoDict["Size(bytes)"] = size
            elif column == "Title":
                title = ""
                if HardReset:
                    if vidInfo == None:
                        vidInfo = pafy.new(uid_to_url(uuid))
                    title = vidInfo.title
                retArr.append(title)
                infoDict["Title"] = title
            elif column == "Description":
                desc = ""
                if HardReset:
                    if vidInfo == None:
                        vidInfo = pafy.new(uid_to_url(uuid))
                    try:
                        desc = parser.unescape(vidInfo.description).encode('ascii', 'ignore').decode('ascii')
                    except:
                        continue
                retArr.append(desc)
                infoDict["Description"] = desc
            elif column == "Keywords":
                keys = ""
                if HardReset:
                    if vidInfo == None:
                        vidInfo = pafy.new(uid_to_url(uuid))
                    keys = str(vidInfo.keywords)
                retArr.append(keys)
                infoDict["Keywords"] = keys
            elif column == "Viewcount":
                vc = -1
                if HardReset:
                    if vidInfo == None:
                        vidInfo = pafy.new(uid_to_url(uuid))
                    vc = vidInfo.viewcount
                retArr.append(keys) 
                infoDict["Viewcount"] = keys
            elif column == "Author":
                author = ""
                if HardReset:
                    if vidInfo == None:
                        vidInfo = pafy.new(uid_to_url(uuid))
                    author = vidInfo.author
                retArr.append(author) 
                infoDict["Author"] = author
            elif column == "Faces":
                if infoDict["File Path"] != "" and categorize:
                    doesHaveMovement = isMoving(infoDict["File Path"])
                    doesHaveFaces = False
                    if doesHaveMovement:
                        print_and_log("Checking faces in " + uuid + "...")
                        doesHaveFaces = hasFaces(path)
                        print_and_log(uuid + " Faces? " + str(doesHaveFaces))
                    infoDict["Faces"] = doesHaveFaces
                else:
                    infoDict["Faces"] = ""
                retArr.append(infoDict["Faces"])
            elif column == "Conversation":
                if infoDict["File Path"] != "" and categorize:
                    doesHaveConversation = hasConversation(uuid)
                    retArr.append(doesHaveConversation)
                    infoDict["Conversation"] = doesHaveConversation
                else:
                    infoDict["Conversation"] = ""
                    retArr.append("")
            elif column == "Uploaded":
                if HardReset:
                    retArr.append(exists_in_boto3(infoDict["File Path"].replace("out/", ""), search=True))
                else:
                    retArr.append(False)
            else:
                print_and_log("Invalid requested frame. Column: " + column + " ID: " + uuid, error=True)
                traceback.print_stack()
                retArr.append("")
    if cap is not None:
        cap.release()
    return tuple(retArr)

def create_or_update_entry(infoDict, shouldSave=True, reset=False):
    """
    Creates or updates the entry in information_csv, and, by proxy, the csv
    Also backs up every N videos to the CSV.

    infoDict Complete keys: UUID, Keywords,
    ViewCount, Title, Author, Dimensions, Format, Description
    Duration
    """
    global information_csv, backup_counter
    if infoDict is None:
        print_and_log("Invalid entry blocked. Infodict is none. " + "\n", error=True)
        traceback.print_stack()
        return
    if "UUID" not in infoDict.keys() or len(infoDict["UUID"]) != 11: # all infoDicts need a UUID entry for row identification
                                    # and all UUIDs are 11 characters long.
        print_and_log("Invalid entry blocked: " + str(infoDict.keys()) + " UUID:"+infoDict["UUID"] + "\n", error=True)
        traceback.print_stack()
        return
    try: # make sure whole process doesn't stop based on one error.
        if backup_counter >= BACKUP_EVERY_N_VIDEOS and shouldSave:
            backup_counter = 0
            saveCSV(CSV_PATH)
        else:
            backup_counter += 1
        uid = str(infoDict["UUID"])
        url = uid_to_url(uid)
        date = time.strftime("%d/%m/%Y %H:%M:%S")
        columns_except_url_and_uid = columns[4:]
        row_in_csv = information_csv[information_csv["UUID"] == uid] # get row
        if len(row_in_csv) == 1:  # If it is already in the CSV
            try:
                for column in infoDict.keys():
                    information_csv.loc[information_csv[
                        "UUID"] == uid, column] = infoDict[column]
            except Exception, e:
                print_and_log("Error on item! "+str(e)+"\n"+traceback.format_exc())
                pdb.set_trace()
        elif len(row_in_csv) > 1: # If there are multiple entries
            rows = row_in_csv
            newRow = []
            rowsToDrop = []
            for column in columns:
                added = False
                for index, row in rows.iterrows():
                    rowsToDrop.append(index)
                    if row[column] == "" or pd.isnull(row[column]):
                        newRow.append(row[column])
                        added = True
                        break
                if not added:
                    newRow.append("")
            information_csv.drop(information_csv.index[rowsToDrop], inplace=True)
            information_csv.loc[len(information_csv)] = pd.Series(
                newRow, index=columns)
        elif len(row_in_csv) == 0:  # If it isn't already in CSV
            newRow = [url, uid, date, infoDict["Query"] if "Query" in infoDict.keys() else ""]
            newRow += get_attributes(uid, columns_except_url_and_uid)
            assert len(information_csv.keys()) == len(newRow)
            information_csv.loc[len(information_csv)] = pd.Series(
                newRow, index=columns)
        else:
            print_and_log("Error in updating an entry", error=True)
    except Exception, e:
        print_and_log("ERROR ON THREAD: FAILED TO ADD OBJECT!!!!!" +
                    str(e)+"\n"+traceback.format_exc(), error=True)
        pdb.set_trace()
    try:
        information_csv = information_csv[pd.notnull(information_csv['UUID'])] # Remove all null UUID entries from csv, they are useless
    except:
        print("HERE?????????????")

# @retry(wait_fixed=600000, stop_max_attempt_number=5)
def scrape_id(query, num_to_download=NUM_VIDS):
    """
    Scrapes youtube and creates or updates entries.
    """
    global BACKUP_EVERY_N_VIDEOS
    temp = BACKUP_EVERY_N_VIDEOS
    BACKUP_EVERY_N_VIDEOS = BACKUP_EVERY_N_VIDEOS*20
    youtube_api = build(YOUTUBE_API_SERVICE_NAME,
                        YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    # Call the search.list method to retrieve results matching the specified
    # query term.
    counter = 0
    search = youtube_api.search().list(
        q=query,
        type="video",
        part="id",
        videoDuration="medium",
        maxResults=YOUTUBE_MAX_SEARCH_REASULTS
    )
    allResultsRead = False
    while not allResultsRead:
        searchResponse = search.execute()
        for search_result in searchResponse.get("items", []):
            try:
                uid = str(search_result["id"]["videoId"])
                print_and_log("Adding " + uid + " to CSV. From query: " + str(query))
                create_or_update_entry({"UUID": uid, "Query": str(query)})
                if uid not in information_csv["UUID"]:
                    counter += 1
            except Exception, e:
                print("Error on item: ", str(e)+"\n"+traceback.format_exc())
            if counter >= num_to_download:
                counter = 0
                allResultsRead = True
                break
        try:
            search = youtube_api.search().list(
                q=query,
                type="video",
                part="id",
                videoDuration="medium",
                maxResults=YOUTUBE_MAX_SEARCH_REASULTS,
                pageToken=searchResponse["nextPageToken"]
            )
        except KeyError:
            allResultsRead = True
    BACKUP_EVERY_N_VIDEOS = temp

@retry(wait_fixed=60*15, stop_max_attempt_number=5)
def download_video(uid):
    """
    Downloads a video of a specific uid
    """
    video_url = uid_to_url(uid)
    logging.info("Downloading video at url: %s" %
                 ("youtube.com/watch?v="+video_url))
    video_object = pafy.new(video_url)
    stream = video_object.getbest()
    filename = uid+"."+stream.extension
    filepath = "out/toCheck/"+filename if "mp4" in filename else "out/toConvert/"+filename
    # starts download in the same directory of the script
    filename = stream.download(filepath=filepath)
    logging.info("Finished downloading video at url: %s" % (video_url))
    captions = str(download_caption(uid))
    infoDict = {}
    if OPEN_ON_DOWNLOAD:
        os.system("open "+filepath)
    try:
        infoDict = {"UUID": uid, "Keywords": str(video_object.keywords),
                "Viewcount": video_object.viewcount, "Title": video_object.title, "Author": video_object.author,
                "Dimensions": stream.resolution, "Format": stream.extension,
                "Size(bytes)": stream.get_filesize(), "File Path": filepath,
                "Duration": parser.unescape(video_object.duration).encode('ascii', 'ignore').decode('ascii'),
                "Description": parser.unescape(video_object.description).encode('ascii', 'ignore').decode('ascii'),
                "Captions": captions, "Date Updated": datetime.now().strftime('%Y/%m/%d %H:%M:%S'), "Worker": WORKER_UUID}
    except KeyError, e:
        print_and_log("Pafy backend failure on "+video_id)
    return infoDict

def uid_to_url(uid):
    """
    Converts a uid to a url
    """
    return "youtube.com/watch?v="+uid

def start_logger(args):
    """
    Starts the logger
    """
    logging.basicConfig(level=logging.INFO)
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    logger = logging.getLogger()
    fileHandler = logging.FileHandler("out/output.log")
    fileHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(fileHandler)
    logging.getLogger().addHandler(logging.StreamHandler())
    if args.verbose:  # default off for the sake of clean output
        logging.getLogger().setLevel("INFO")
    else:
        logging.getLogger().setLevel("ERROR")
    logging.getLogger('googleapicliet.discovery_cache').setLevel(
        logging.ERROR)  # Removes annoying OAuth error

def convertVideo(video_id):
    """
    Converts a video from webm to mp4
    """
    path = get_attributes(video_id, ["File Path"])[0]
    newPath = "out/toCheck/"+video_id+".mp4"
    ff = ffmpy.FFmpeg(
        inputs={path: "-y"},
        outputs={newPath: "-strict -2"}
    )

    ff.run()
    delFile(path)
    information_csv[information_csv["UUID"] == video_id]["Format"] = ".mp4"
    information_csv[information_csv["UUID"] == video_id]["File Path"] = newPath

def stripAudio(video_id):
    """
    Strips wav from mp4
    """
    path = get_attributes(video_id, ["File Path"])[0]
    newPath = "out/tmp/"+video_id+".wav"
    if not os.path.exists(newPath):
        ff = ffmpy.FFmpeg(
            inputs={path: None},
            outputs={newPath: "-y -codec:v copy -af pan=\"mono: c0=FL\" -ar 32000"}
        )
        ff.run(stdout=open("/dev/null", 'wb'), stderr=open("/dev/null", 'wb'))
    return newPath

def createDir(path):
    """
    Create a directory if not already there
    """
    if not os.path.exists(path):
        os.makedirs(path)

def isMoving(path):
    """
    Check if there is movement in a video at a path
    """
    moving = MovementDetect.checkFile(path)
    if moving == -1:
        print_and_log("isMoving got passed invalid file: " + path, error=True)
        return
    return moving == 1

def hasConversation(id_):
    """
    Check if a video has conversation in it
    """
    framesize = 20  # in msec
    padding_width = 300  # in msec
    path = stripAudio(id_)
    vc = VadCollector.VadCollector(
        path, CONVERSATION_AGGRESSIVENESS, framesize, padding_width, thresh=0.9)
    percentage = vc.get_percentage()
    print_and_log(id_ + " is " + str(percentage) + " talk.")
    os.remove(path)
    return percentage > SPEECH_TRHESHHOLD

@timeout_decorator.timeout(60*5, timeout_exception=StopIteration)
def hasFaces(path):
    """
    Check if a video
    """
    global graph, sess
    if not os.path.exists(path) and path != "":
        print_and_log("hasFaces got passed invalid path: " + path, error=True)
        return
    if graph == None or sess == None:
        print_and_log("graph and sess are none, reinitializing...")
        graph = load_model_pb(FACE_DETECTION_MODEL)
        sess = tf.Session(graph=graph)
    return checkForFace(path, graph, sess)

def moveFromTo(from_, to_):
    """
    Moves a file from path to another path
    """
    if not os.path.exists(from_) and from_ != "":
        print("Move got passed an invalid path: "+from_)
    os.rename(from_, to_)

def uploadToS3(args, video_id):
    global bucket
    """
    Upload to S3 and update information_csv
    """
    global information_csv
    path, type_ = get_attributes(video_id, ["File Path", "Format"])
    infoDict = {"UUID": video_id, "File Path": path, "Format": type_}
    if path == "":
        return infoDict
    # get second to last occurence
    s3path = path[path.rfind("/", 0, path.rfind("/"))+1:]
    print_and_log("Uploading " + path + " to " + s3path)
    bucket.upload_file(path, s3path)
    infoDict["Uploaded"] = True
    infoDict["Worker"] = "On Master"
    return infoDict

def categorize_video(video_id):
    """
    Categorize a video, move to correct folder, and return new infoDict
    """
    print_and_log("Categorizing "+video_id)
    doesHaveConversation, doesHaveFaces, filepath = get_attributes(video_id, ["Conversation", "Faces", "File Path"], HardReset=True, categorize=True)
    infoDict = {"UUID": video_id, "Faces": doesHaveFaces, "Conversation": doesHaveConversation}
    if filepath == "":
        return infoDict
    fileName = filepath[filepath.rfind("/")+1:]
    print("Face, "+str(doesHaveFaces)+" | Speech, "+str(doesHaveConversation))
    newPath = ""
    if doesHaveConversation and doesHaveFaces:
        newPath = "out/Multimodal/"+fileName
    elif doesHaveConversation:
        newPath = "out/Conversation/"+fileName
    elif doesHaveFaces:
        newPath = "out/Faces/"+fileName
    else:
        newPath = "out/Trash/"+fileName
    moveFromTo(filepath, newPath)
    infoDict["Faces"] = doesHaveFaces
    infoDict["Conversation"] = doesHaveConversation
    infoDict["File Path"] = newPath
    return infoDict

def createOutputDirs():
    """
    Creates the output directory
    """
    folders = ["out/", "out/toCheck/", "out/tmp/", "out/toConvert/", "out/Conversation/",
                "out/Multimodal/", "out/Faces/", "out/Trash/"]
    for folder in folders:
        createDir(folder)

def createBotoDir(folder):
    global bucket
    if bucket == None:
        bucket = s3.Bucket(DATA_BUCKET_NAME)
    bucket.put_object(
        Bucket=DATA_BUCKET_NAME,
        Body='',
        Key=folder
        )

def createBotoDirs():
    folders = ["Conversation/", "Multimodal/", "Faces/", "Trash/", "Workers/"]
    for folder in folders:
        createBotoDir(folder)

def print_and_log(str_, error=False):
    """
    Both print and log given string. If error is true,
    the error will appear in red.
    """
    if error:
        str_ = "\033[1;31m"+str_+"\033[0m\n"
        logging.error(str_)
    else:
        logging.info(str_)
    print(str_)

def delFile(path, message=None):
    os.remove(path)
    if message != None:
        logging.info(message)

def clean_downloads():
    """
    Cleans the downloads and syncs the information_csv.
    If this process is interupted, please restart it
    and wait until completion.
    """
    global information_csv, BACKUP_EVERY_N_VIDEOS, bucket
    print_and_log("Cleaning time!!!.....")
    createOutputDirs()
    print_and_log("Deleting duplicates...")
    information_csv.drop_duplicates(subset="UUID", inplace=True)
    # Remove Duplicates from toConvert and toCheck
    for root, subdirs, files in os.walk("out/toConvert"):
        for file in files:
            if '.' in file:
                uid = file[:file.rfind('.')]
                if os.path.exists("out/toCheck/"+uid+".mp4"):
                    delFile(root+"/"+file, message=uid+" has already been converted. Deleting duplicate.")
            else:
                path = root+"/"+file
                delFile(path, message="Deleting corrupt file: "+path)
    
    # Go through the folders and update the csv to reflect file structure.
    information_csv["File Path"] = ""
    information_csv["Conversation"] = False
    information_csv["Faces"] = False
    for root, subdirs, files in os.walk("out/"):
        if len(files) > 0:
            for file in tqdm(files):
                if '.' not in file:
                    continue
                format_ = file[file.find('.')+1:]
                uid = file[:-len(format_)-1]
                path = os.path.join(root, file)
                if "temp" in file or "tmp" in root:  # Temp file
                        path = os.path.join(root, file)
                        delFile(path, "Deleting temporary file "+path)
                elif ".webm" in file:
                    if "toConvert/" not in path:
                        moveFromTo(path, "out/toConvert/"+uid+".webm")
                    create_or_update_entry({"UUID": uid, "File Path": "out/toConvert/"+uid+".webm", "Format": "webm", "Worker": WORKER_UUID}, shouldSave=False, reset=True)
                elif ".mp4" in file:
                    if "Multimodal" in root:
                        create_or_update_entry({"UUID": uid, "Conversation": True, "Faces": True, "File Path": path, "Format": "mp4", "Worker": WORKER_UUID}, shouldSave=False, reset=True)
                    elif "Conversation" in root:
                        create_or_update_entry({"UUID": uid, "Conversation": True, "File Path": path, "Format": "mp4", "Worker": WORKER_UUID}, shouldSave=False, reset=True)
                    elif "Faces" in root:
                        create_or_update_entry({"UUID": uid, "Faces": True, "File Path": path, "Format": "mp4", "Worker": WORKER_UUID}, shouldSave=False, reset=True)
                    elif "Trash" in root:
                        create_or_update_entry({"UUID": uid, "Faces": False, "Conversation":False, "File Path": path, "Format": "mp4", "Worker": WORKER_UUID}, shouldSave=False, reset=True)
                    elif "toCheck" in root:
                        create_or_update_entry({"UUID": uid, "File Path": path, "Format": "mp4", "Worker": WORKER_UUID}, shouldSave=False, reset=True)
    information_csv = information_csv[information_csv['UUID'].map(len) == 11]
    print_and_log("Fixing and updating CSV...")
    print_and_log("Checking what's in s3...Don't quit here...")
    information_csv["Uploaded"] = False # don't use get_attributes for speed benefit
    for item in bucket.objects.all():
        if 'mp4' in item.key:
            # print(item.key)
            create_or_update_entry({"UUID":item.key[item.key.rfind("/")+1:-4], "Uploaded":True}, shouldSave=False)
    saveCSV(CSV_PATH)

def categorize_video_wrapper(video_id):
    try:
        return categorize_video(video_id)
    except Exception, e:
        print_and_log("Error in categorization on id: " + video_id + ": " + str(e)+"\n"+traceback.format_exc(), error=True)
        return None

def download_video_wrapper(video_id):
    try:
        return download_video(video_id)
    except Exception, e:
        print_and_log("Error in downloading video on id: " + video_id + ": " + str(e)+"\n"+traceback.format_exc(), error=True)
        return None

def uploadToS3_wrapper(args, video_id):
    try:
        return uploadToS3(args, video_id)
    except Exception, e:
        print_and_log("Error in uploading video on id: " + video_id + ": " + str(e)+"\n"+traceback.format_exc(), error=True)
        return None

def convert_wrapper(id_):
    try:
        return convertVideo(id_)
    except Exception, e:
        print_and_log("Error in converting video: on id: " + video_id + ": " + str(e)+"\n"+traceback.format_exc(), error=True)
        return None

def print_error(e):
    print_and_log(str(e), error=True)

def main():
    global information_csv, NUM_VIDS, BACKUP_EVERY_N_VIDEOS, OPEN_ON_DOWNLOAD, QUERIES, bucket, graph, sess
    ######################### Initialize #########################
    args = parse_args()
    if os.path.exists("out/tmp/"):
        shutil.rmtree("out/tmp/") # Remove temporary files
    createOutputDirs()
    createBotoDirs()
    #### Make remote connections and load models if necessary
    print_and_log("Connecting to s3...")
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(DATA_BUCKET_NAME)
    print_and_log("Created Boto3 directories if not already there")

    if args.categorize:
        print_and_log("Loading Face Tracker...")
        graph = load_model_pb(FACE_DETECTION_MODEL)
        sess = tf.Session(graph=graph)
    NUM_VIDS = int(args.num_vids)
    BACKUP_EVERY_N_VIDEOS = int(args.backup_every)

    OPEN_ON_DOWNLOAD = args.openOnDownload
    if args.query != None:
        QUERIES = [args.query] if ',' not in args.query else args.query.split(",")
        QUERIES = [x.strip() for x in QUERIES]
    start_logger(args)
    recover_or_get_youtube_id_dictionary(args)
    information_csv = information_csv.replace(np.nan, "")
    saveCSV(CSV_PATH)
    # Create output folder if it's not there
    createOutputDirs()
    print_and_log("Created output directories if not already there")

    ################################ Run ################################
    if args.clean:
        clean_downloads()
    pool = Pool(processes=int(args.num_threads))
    if args.query != None:
        print_and_log("Switching to download new videos...")
        for q in QUERIES:
            for _id in information_csv[(information_csv["Query"] == q) & (is_empty_or_false("File Path"))]["UUID"].tolist()[:NUM_VIDS]:
                # download_video(_id)
                pool.apply_async(download_video, args=(_id,))
    if args.convert:
        print_and_log("Starting Conversion...")
        for _id in tqdm(information_csv[(information_csv['File Path'].str.contains("webm")) &
                                         (~is_empty_or_false("File Path")) &
                                         (information_csv["Worker"] == WORKER_UUID)]["UUID"].tolist()):
            convert_wrapper(_id)

    if args.categorize:
        print_and_log("Switching to Categorize...")
        for _id in tqdm(information_csv.loc[(information_csv['File Path'].str.contains("toCheck")) &
                                            (information_csv["Worker"] == WORKER_UUID)]["UUID"].tolist()):
            create_or_update_entry(categorize_video_wrapper(_id))
            # pool.apply_async(categorize_video_wrapper, args=(_id,), callback=create_or_update_entry, error_callback=print_error)
    if args.upload:
        print_and_log("Switching to Uploading...")
        for _id in tqdm(information_csv[(~is_empty_or_false("File Path")) &
                                          ((information_csv["Uploaded"] == False) | (is_empty_or_false("Uploaded"))) &
                                          (information_csv['File Path'].str.contains("Multimodal") |
                                           information_csv['File Path'].str.contains("Conversation") |
                                           information_csv['File Path'].str.contains("Faces")) &
                                           (information_csv["Worker"] == WORKER_UUID)]["UUID"].tolist()):
            pool.apply_async(uploadToS3_wrapper, args=(args, _id), callback=create_or_update_entry)
    saveCSV(CSV_PATH)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
