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
import cv
from uuid import uuid4 as uuid
import httplib2
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
from CheckMovement import MotionDetectorInstantaneous
from CheckFaces import load_model_pb, checkForFace
import VadCollector
import traceback


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
YOUTUBE_API_VERSION = "v3"
YOUTUBE_MAX_SEARCH_REASULTS = 50
BACKUP_EVERY_N_VIDEOS = 5  # Backup to the CSV every this number of videos
FACE_DETECTION_MODEL = "./zf4_tiny_3900000.pb"  # Face tracking model
NUM_VIDS = 10  # Number of videos that should be downloaded on default
SPEECH_TRHESHHOLD = .5  # percentage of video that contains speech
# The tolerance of the speech model. Options are 1, 2, or 3.
CONVERSATION_AGGRESSIVENESS = 2
CSV_PATH = None  # items.csv should be in out/
QUERIES = []  # Queries given as command line arguments split up.
OPEN_ON_DOWNLOAD = False # Should the program open the videos once downloaded?
bucket, graph, sess = None, None, None  # Initializing variables globally
parser = HTMLParser.HTMLParser()

# Create dataframe
columns = ["Url", "UUID", "Date Updated", "Format", "File Location", "Dimensions", "Bitrate", "Downloaded",
           "Query", "Rating", "Title", "Description", "Duration",
           "Captions", "Size(bytes)", "Keywords", "Length(seconds)", "Viewcount", "Faces", "Conversation", "Author", "Uploaded"]

columnTypes = [str, str, str, str, str, str, float, bool, str, float, str, str, float, str, float, str, float, float, bool, bool, str, bool]
information_csv = pd.DataFrame(columns=columns)
backup_counter = 0


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


def saveCSV(path):
    """
    Saves information_csv, as a csv, to the path in the argument provided
    """
    print_and_log("Saving the CSV...")
    try:
        with open(path, 'wb') as fout:
            information_csv.to_csv(path, index=False, encoding='utf-8')
        print_and_log("Saved")
    except Exception, e:
        logging.error("Failed to save csv:"+str(e)+"\n"+traceback.format_exc())


def convertDataTypes():
    """
    Converts all the columns in information_csv to their proper
    datatype. Otherwise, they all become `object`
    """
    global information_csv
    for i in range(len(columns)):
        column = columns[i]
        information_csv[[column]] = information_csv[
            [column]].astype(columnTypes[i])


def recover_or_get_youtube_id_dictionary(args):
    """
    Used during the startup. It will either create a new csv or load the old one.
    If it loads an old csv, it will check the folder and update the csv based on
    which files are downloaded.
    """
    global information_csv, CSV_PATH
    # Create JSON file if not there
    CSV_PATH = os.path.join("out/", 'items.csv')
    if not os.path.exists(CSV_PATH):
        if args.query != None:
            scrape_ids(args)
        saveCSV(CSV_PATH)
    else:
        logging.info("Opening cached search results: %s" % CSV_PATH)
        logging.info("Checking cached search results: %s" % CSV_PATH)
        with open(CSV_PATH, 'rb') as fin:
            information_csv = pd.read_csv(CSV_PATH)
            # convertDataTypes()
        for key in QUERIES:
            try:
                if len(information_csv[(information_csv['Downloaded'] != True) &\
                            (information_csv['Query'].str.contains(key))]["Query"].tolist()) > NUM_VIDS\
                            and not args.rebuild:
                    logging.info("Found query:" + key +
                                 " in cached search results, using cached search")
                else:
                    if args.query != None:
                        logging.info("Didn't find query:" + key +
                                     " in cached search results, scraping now...")
                        scrape_ids(args)
            except:
                pdb.set_trace()


def convert_caption_to_str(trackList):
    """
    Converts a list of `Track` objects to a string
    """
    if trackList == None:
        return ""
    retStr = ""
    for track in trackList:
        retStr += "Starts at " + str(track.start) + "s and lasts " + str(
            track.duration) + "s: " + parser.unescape(track.text) + "\n"
    # Make the characters solely ascii
    return retStr.encode('ascii', 'ignore').decode('ascii')


@retry(wait_fixed=600000, stop_max_attempt_number=5)
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


def create_or_update_entry(infoDict, success=True, shouldSave=True):
    """
    Creates or updates the entry in information_csv, and, by proxy, the csv
    Also backs up every N videos to the CSV.

    infoDict Complete keys: UUID, Length, Keywords,
    ViewCount, Title, Author, Bitrate, Dimensions, Format, Description
    Duration
    """
    global information_csv, backup_counter
    if not success:
        return
    if infoDict is None:
        print_and_log("Invalid entry blocked. Infodict is none. " + "\n", error=True)
        traceback.print_stack()
        return
    if "UUID" not in infoDict.keys() or len(infoDict["UUID"]) != 11: # all infoDicts need a UUID entry for row identification
                                    # and all UUIDs are 11 characters long.
        print_and_log("Invalid entry blocked: " + str(infoDict.keys()) + "\n", error=True)
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
        columns_except_url_and_uid = columns[3:]
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
            for column in columns:
                added = False
                for row in rows:
                    if row[column] == "" or pd.isnull(row[column]) or pd.isnan(row[column]):
                        newRow.append(row[column])
                        added = True
                        break
                if not added:
                    newRow.append("")
            information_csv.drop(information_csv.index[list(rows.index.values)], inplace=True)
            information_csv.loc[len(information_csv)] = pd.Series(
                newRow, index=columns)
        elif len(row_in_csv) == 0:  # If it isn't already in CSV
            newRow = [url, uid, date]
            for column in columns_except_url_and_uid:
                if column == "Downloaded":
                    newRow.append(
                        False if "Downloaded" not in infoDict.keys() else infoDict["Downloaded"])
                elif column == "Duration" and "File Location" in infoDict.keys() and (infoDict["File Location"] != "" or row_in_csv["File Location"] != ""):
                    cap = cv2.VideoCapture(row_in_csv["File Location"].tolist()[0] if infoDict[
                                           "File Location"] == "" else infoDict["File Location"])
                    newRow.append(cap.get(cv.CV_CAP_PROP_FRAME_COUNT))
                    cap.release()
                else:
                    # If no data, leave cell blank
                    newRow.append(
                        infoDict[column] if column in infoDict.keys() else "")
            assert len(information_csv.keys()) == len(newRow)
            information_csv.loc[len(information_csv)] = pd.Series(
                newRow, index=columns)
        else:
            print_and_log("Error in updating an entry", error=True)
    except Exception, e:
        print_and_log("ERROR ON THREAD: FAILED TO ADD OBJECT!!!!!" +
                    str(e)+"\n"+traceback.format_exc(), error=True)
        pdb.set_trace()
    information_csv = information_csv[pd.notnull(information_csv['UUID'])] # Remove all null UUID entries from csv, they are useless

@retry(wait_fixed=600000, stop_max_attempt_number=5)
def scrape_ids(args):
    """
    Scrapes youtube and creates or updates entries.
    """
    youtube_api = build(YOUTUBE_API_SERVICE_NAME,
                        YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    # Call the search.list method to retrieve results matching the specified
    # query term.
    counter = 0
    for key in QUERIES:
        search = youtube_api.search().list(
            q=key,
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
                    create_or_update_entry({"UUID": uid, "Query": str(key)}, shouldSave=False)
                    if uid not in information_csv["UUID"]:
                        counter += 1
                except Exception, e:
                    print("Error on item: ", str(e)+"\n"+traceback.format_exc())
                if counter >= NUM_VIDS:
                    counter = 0
                    allResultsRead = True
                    break
            try:
                search = youtube_api.search().list(
                    q=key,
                    type="video",
                    part="id",
                    videoDuration="medium",
                    maxResults=YOUTUBE_MAX_SEARCH_REASULTS,
                    pageToken=searchResponse["nextPageToken"]
                )
            except KeyError:
                allResultsRead = True


# @retry(wait_fixed=600000, stop_max_attempt_number=5)
def download_video(uid):
    """
    Downloads a video of a specific uid
    """
    video_url = "youtube.com/watch?v=" + uid  # Not calling uid_to_url as it causes problems when multithreaded
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
        infoDict = {"UUID": uid, "Length(seconds)": video_object.length, "Keywords": str(video_object.keywords),
                "Viewcount": video_object.viewcount, "Title": video_object.title, "Author": video_object.author,
                "Bitrate": stream.rawbitrate, "Dimensions": stream.resolution, "Format": stream.extension,
                "Size(bytes)": stream.get_filesize(), "Downloaded": True, "File Location": filepath,
                "Duration": parser.unescape(video_object.duration).encode('ascii', 'ignore').decode('ascii'),
                "Description": parser.unescape(video_object.description).encode('ascii', 'ignore').decode('ascii'), "Captions": captions}
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
    row = information_csv[information_csv["UUID"] == video_id]
    oldPath = row["File Location"].tolist()[0]
    newPath = "out/toCheck/"+row["UUID"].tolist()[0]+".mp4"
    ff = ffmpy.FFmpeg(
        inputs={oldPath: "-y"},
        outputs={newPath: None}
    )
    ff.run()
    information_csv[information_csv["UUID"] == video_id]["Format"] = ".mp4"
    information_csv[information_csv["UUID"] == video_id]["File Location"] = newPath


def stripAudio(video_id):
    """
    Strips wav from mp4
    """
    row = information_csv[information_csv["UUID"] == video_id]
    oldPath = row["File Location"].tolist()[0]
    newPath = "out/tmp/"+row["UUID"].tolist()[0]+".wav"
    if not os.path.exists(newPath):
        ff = ffmpy.FFmpeg(
            inputs={oldPath: None},
            outputs={newPath: "-y -codec:v copy -af pan=\"mono: c0=FL\" -ar 32000"}
        )
        ff.run()
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
    if not os.path.exists(path):
        print_and_log("isMoving got passed invalid path: " + path, error=True)
        pdb.set_trace()
        return
    return MotionDetectorInstantaneous(path)()


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


def hasFaces(path):
    """
    Check if a video
    """
    global graph, sess
    if not os.path.exists(path):
        print_and_log("isMoving got passed invalid path: " + path, error=True)
        return
    return checkForFace(path, graph, sess)


def moveFromTo(from_, to_):
    """
    Moves a file from path to another path
    """
    os.rename(from_, to_)


def uploadToS3(args, video_id):
    global bucket
    """
    Upload to S3 and update information_csv
    """
    global information_csv
    row = information_csv[information_csv["UUID"] == video_id]
    infoDict = {"UUID": video_id}
    if information_csv["File Location"].tolist()[0] == "":
        return infoDict
    path = row["File Location"].tolist()[0]
    type_ = row["Format"].tolist()[0]
    if path == None or not os.path.exists(path):
        path = findFile(video_id+"."+type_)
        infoDict["File Location"] = path
        create_or_update_entry(infoDict)
    if path == None:
        infoDict["Downloaded"] = False
        return infoDict
    # get second to last occurence
    s3path = path[path.rfind("/", 0, path.rfind("/"))+1:]
    print_and_log("Uploading " + path + " to " + s3path)
    bucket.upload_file(path, s3path)
    infoDict["Uploaded"] = True
    return infoDict

def findFile(fileName):
    for root, subdirs, files in os.walk("out/"):
        for file in files:
            if fileName == file:
                return str(root+"/"+fileName)

def categorize_video(args, video_id):
    """
    Categorize a video, move to correct folder, and return new infoDict
    """
    print_and_log("Categorizing "+video_id)
    row = information_csv[information_csv["UUID"] == video_id]
    infoDict = {"UUID": video_id}
    fileLoc = row["File Location"].tolist()[0]
    fileName = video_id+"."+row["Format"].tolist()[0]
    type_ = row["Format"].tolist()[0]
    if not os.path.exists(fileLoc):
        fileLoc = findFile(video_id+"."+type_)
        infoDict["File Location"] = fileLoc
        create_or_update_entry(infoDict)
    if fileLoc == None:
        infoDict["Downloaded"] = False
        return infoDict
    if "toConvert" in fileLoc:
        print_and_log("Needs to be converted before categorization..."+video_id)
        return infoDict
    if "toCheck" not in fileLoc:
        print_and_log("Already checked..."+video_id)
        return infoDict
    if "" == row["File Location"].tolist()[0]:
        print_and_log("Video not where expected..."+video_id, error=True)
        return
    if "webm" in row["File Location"].tolist()[0]:
        if args.convert:
            convertVideo(row["File Location"].tolist()[0])
        else:
            print_and_log("Convert argument not selected. Will not convert this video: " + str(video_id))
        return row.to_dict(orient='records')[0]
    path = "out/toCheck/"+fileName
    print_and_log("Checking movement for " + video_id + "...")
    doesHaveMovement = isMoving(path)
    print_and_log(video_id + " moves? " + str(doesHaveMovement))
    print_and_log("Checking conversation for " + video_id + "...")
    doesHaveConversation = hasConversation(video_id)
    print_and_log(video_id + " converses? " + str(doesHaveConversation))
    doesHaveFaces = False
    newPath = "out/Trash/"+fileName
    if doesHaveMovement:
        print_and_log("Checking faces in " + video_id + "...")
        doesHaveFaces = hasFaces(path)
        print_and_log(video_id + " Faces? " + str(doesHaveFaces))
    print("Face, "+str(doesHaveFaces)+" | Speech, "+str(doesHaveConversation))
    if doesHaveConversation and doesHaveFaces:
        newPath = "out/Multimodal/"+fileName
    elif doesHaveConversation:
        newPath = "out/Conversation/"+fileName
    elif doesHaveFaces:
        newPath = "out/Faces/"+fileName
    moveFromTo(path, newPath)
    infoDict["Faces"] = doesHaveFaces
    infoDict["Conversation"] = doesHaveConversation
    infoDict["File Location"] = newPath
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
    bucket.put_object(
        Bucket='youtube-video-data',
        Body='',
        Key=folder
        )

def createBotoDirs():
    folders = ["Conversation/", "Multimodal/", "Faces/", "Trash/"]
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


def clean_downloads():
    """
    Cleans the downloads and syncs the information_csv.
    If this process is interupted, please restart it
    and wait until completion.
    """
    print_and_log("Cleaning time!!!.....")
    createOutputDirs()
    # Go through the folders and update the csv to reflect file structure.
    information_csv["Downloaded"] = False
    for root, subdirs, files in os.walk("out/"):
        if len(files) > 0:
            for file in tqdm(files):
                if ".mp4" in file or ".webm" in file:
                    uid = file[:-4] if ".mp4" in file else file[:-5]
                    path = os.path.join(root, file)
                    if "temp" in file or "tmp" in root:  # Temp file
                        remPath = os.path.join(root, file)
                        os.remove(remPath)
                        logging.info("Removing "+str(remPath) +
                                     "as it is a partial download")
                    else:
                        create_or_update_entry({"UUID": uid, "Downloaded": True, "File Location": path, "Format": path[
                                                      path.rindex(".")+1:]}, shouldSave=False)
                    if "Multimodal" in root:
                        create_or_update_entry({"UUID": uid, "Downloaded": True, "Conversation": True, "Faces": True, "File Location": path, "Format": path[
                                                      path.rindex(".")+1:]}, shouldSave=False)
                    elif "Conversation" in root:
                        create_or_update_entry({"UUID": uid, "Downloaded": True, "Conversation": True, "File Location": path, "Format": path[
                                                      path.rindex(".")+1:]}, shouldSave=False)
                    elif "Faces" in root:
                        create_or_update_entry({"UUID": uid, "Downloaded": True, "Faces": True, "File Location": path, "Format": path[
                                                      path.rindex(".")+1:]}, shouldSave=False)
                    elif "Trash" in root:
                        create_or_update_entry({"UUID": uid, "Downloaded": True, "Faces": False, "Conversation":False, "File Location": path, "Format": path[
                                                      path.rindex(".")+1:]}, shouldSave=False)
                    elif "toCheck" in root:
                        create_or_update_entry({"UUID": uid, "Downloaded": True, "Faces": None, "Conversation":None, "File Location": path, "Format": path[
                                                      path.rindex(".")+1:]}, shouldSave=False)
    saveCSV(CSV_PATH)
    print_and_log("Finished updating CSV. Moving files to propper place now.")
    # Go through information CSV and update file locations based on information_csv.
    for _id in tqdm(information_csv[information_csv["Downloaded"] == True]["UUID"].tolist()):
        row = information_csv[information_csv["UUID"] == _id]
        oldPath = row["File Location"].tolist()[0]
        newPath = "out/"
        if "webm" in oldPath:
            newPath += "toConvert/"
        elif (row["Conversation"] == True).tolist()[0] and (row["Faces"] == True).tolist()[0]:
            newPath += "Multimodal/"
        elif (row["Conversation"] == True).tolist()[0]:
            newPath += "Conversation/"
        elif (row["Faces"] == True).tolist()[0]:
            newPath += "Faces/"
        elif ((row["Conversation"] == "").tolist()[0] or row["Conversation"].tolist()[0] is None) and\
                    ((row["Faces"] == "").tolist()[0] or row["Faces"].tolist()[0] is None):
            newPath += "toCheck/"
        else:
            newPath += "Trash/"
        if len(oldPath) > 1:
            newPath += oldPath[oldPath.rindex("/")+1:]
            if oldPath != newPath:
                print_and_log(
                    "Moving "+row["File Location"].tolist()[0]+" to "+newPath)
                moveFromTo(row["File Location"].tolist()[0], newPath)
                create_or_update_entry({"UUID": uid,"File Location": newPath}, shouldSave=False)
        else:
            create_or_update_entry({"UUID": _id, "Downloaded": False})
    print_and_log("Finished cleaning directory.")
    saveCSV(CSV_PATH)


def categorize_video_wrapper(args, video_id):
    try:
        return (categorize_video(args, video_id), True)
    except Exception, e:
        print_and_log("Error in categorization: " + str(e)+"\n"+traceback.format_exc(), error=True)
        return (None, False)

def download_video_wrapper(video_id):
    try:
        return (download_video(video_id), True)
    except Exception, e:
        print_and_log("Error in downloading video: " + str(e)+"\n"+traceback.format_exc(), error=True)
        return (None, False)

def uploadToS3_wrapper(args, video_id):
    try:
        return (uploadToS3(args, video_id), True)
    except Exception, e:
        print_and_log("Error in uploading video: " + str(e)+"\n"+traceback.format_exc(), error=True)
        return (None, False)

def convert_wrapper(id_):
    try:
        return (convertVideo(id_), True)
    except Exception, e:
        print_and_log("Error in converting video: " + str(e)+"\n"+traceback.format_exc(), error=True)
        return (None, False)

def complete_wrapper(args, _id):
    infoDict, success = download_video_wrapper(_id)
    create_or_update_entry(infoDict)
    if success:
        if args.convert:
            infoDict, success = convert_wrapper(_id)
        if success and args.categorize:
            infoDict, success = categorize_video_wrapper(args, _id)
            if success and args.upload:
                infoDict, success = uploadToS3_wrapper(args, _id)
                if success:
                    return infoDict

def main():
    global information_csv, NUM_VIDS, BACKUP_EVERY_N_VIDEOS, OPEN_ON_DOWNLOAD, QUERIES, bucket, graph, sess
    ######################### Initialize ####################################
    args = parse_args()
    createOutputDirs()
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

    if args.upload:
        print_and_log("Connecting to s3...")
        s3 = boto3.resource('s3')
        bucket = s3.Bucket("youtube-video-data")
        createBotoDirs()
        print_and_log("Created Boto3 directories if not already there")

    if args.categorize:
        print_and_log("Loading Face Tracker...")
        graph = load_model_pb(FACE_DETECTION_MODEL)
        sess = tf.Session(graph=graph)
        print_and_log("Processing downloaded Videos first...")

    # Create output folder if it's not there
    createOutputDirs()
    print_and_log("Created output directories if not already there")


    ################################ Run ################################
    if args.clean:
        clean_downloads()
    pool = Pool(processes=int(args.num_threads))

    if args.convert:
        print_and_log("Starting Conversion...")
        for _id in tqdm(information_csv[((information_csv['File Location'].str.contains("webm")) &
                                         (information_csv["Downloaded"] == True))]["UUID"].tolist()):
            convert_wrapper(_id)

    counter = 0
    if args.query != None:
        print_and_log("Switching to download new videos...")
        for q in QUERIES:
            for _id in information_csv[(information_csv["Query"] == q) & (information_csv["Downloaded"] == False)]["UUID"].tolist():
                if counter >= NUM_VIDS:
                    break
                complete_wrapper(args, _id)
                # pool.apply_async(download_video_wrapper, args=(_id, ), callback=create_or_update_entry)
                # if args.categorize:
                #     pool.apply_async(categorize_video_wrapper, args=(args, _id), callback=create_or_update_entry)
                # if args.upload and args.categorize:
                #     pool.apply_async(uploadToS3_wrapper, args=(args, _id), callback=create_or_update_entry)
                counter += 1

    if args.categorize:
        for _id in tqdm(information_csv.loc[(information_csv["Downloaded"] == True) & (information_csv['File Location'].str.contains("toCheck"))]["UUID"].tolist()):
            create_or_update_entry(*categorize_video_wrapper(args, _id))
            # pool.apply_async(categorize_video_wrapper, args=(args, _id), callback=create_or_update_entry)


    if args.upload:
        for _id in tqdm(information_csv[((information_csv["Downloaded"] == True) &
                                          ((information_csv["Uploaded"] == False) | (information_csv["Uploaded"] == "")) &
                                          (information_csv['File Location'].str.contains("Multimodal") |
                                           information_csv['File Location'].str.contains("Conversation") |
                                           information_csv['File Location'].str.contains("Faces")))]["UUID"].tolist()):
            create_or_update_entry(*uploadToS3_wrapper(args, _id))
            # pool.apply_async(uploadToS3_wrapper, args=(args, _id), callback=create_or_update_entry)
    saveCSV(CSV_PATH)
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
