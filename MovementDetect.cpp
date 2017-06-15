//
//  Created by Cedric Verstraeten on 18/02/14.
//  Copyright (c) 2014 Cedric Verstraeten. All rights reserved.
//

#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <time.h>
#include <dirent.h>
#include <sstream>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdlib.h>

using namespace std;
using namespace cv;



// Check if the directory exists, if not create it
// This function will create a new directory if the image is the first
// image taken for a specific day
inline void directoryExistsOrCreate(const char* pzPath)
{
    DIR *pDir;
    // directory doesn't exists -> create it
    if ( pzPath == NULL || (pDir = opendir (pzPath)) == NULL)
        mkdir(pzPath, 0777);
    // if directory exists we opened it and we
    // have to close the directory again.
    else if(pDir != NULL)
        (void) closedir (pDir);
}


// Check if there is motion in the result matrix
// count the number of changes and return.
inline int detectMotion(const Mat & motion, Mat & result, Mat & result_cropped,
                 int x_start, int x_stop, int y_start, int y_stop,
                 int max_deviation,
                 Scalar & color)
{
    // calculate the standard deviation
    Scalar mean, stddev;
    meanStdDev(motion, mean, stddev);
    // if not to much changes then the motion is real (neglect agressive snow, temporary sunlight)
    if(stddev[0] < max_deviation)
    {
        int number_of_changes = 0;
        int min_x = motion.cols, max_x = 0;
        int min_y = motion.rows, max_y = 0;
        // loop over image and detect changes
        for(int j = y_start; j < y_stop; j+=2){ // height
            for(int i = x_start; i < x_stop; i+=2){ // width
                // check if at pixel (j,i) intensity is equal to 255
                // this means that the pixel is different in the sequence
                // of images (prev_frame, current_frame, next_frame)
                if(static_cast<int>(motion.at<uchar>(j,i)) == 255)
                {
                    number_of_changes++;
                    if(min_x>i) min_x = i;
                    if(max_x<i) max_x = i;
                    if(min_y>j) min_y = j;
                    if(max_y<j) max_y = j;
                }
            }
        }
        if(number_of_changes){
            //check if not out of bounds
            if(min_x-10 > 0) min_x -= 10;
            if(min_y-10 > 0) min_y -= 10;
            if(max_x+10 < result.cols-1) max_x += 10;
            if(max_y+10 < result.rows-1) max_y += 10;
            // draw rectangle round the changed pixel
            Point x(min_x,min_y);
            Point y(max_x,max_y);
            Rect rect(x,y);
            Mat cropped = result(rect);
            cropped.copyTo(result_cropped);
            rectangle(result,rect,color,1);
        }
        return number_of_changes;
    }
    return 0;
}
bool success = false;
Mat getFrame(VideoCapture cap, int skipFrames) {
    Mat ret;
    int counter = 0;
    for (int i = 0; i < 100; i++){
        success = cap.read(ret);
        if (success) counter += 1;
        if (success and counter >= skipFrames) return ret;
    }
    return ret;
}


bool does_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

bool is_video_file_good(std::string path) { // Will give false negative on videos under CV_CAP_PROP_FRAME_COUNT/numToCheck frames
    std::ifstream stream;
    stream.open(path);
    if (!stream) return false; // Check if file is there
    unsigned int curLine = 0;
    string line;
    bool testedFile = true;
    while(getline(stream, line)) { // I changed this, see below
        curLine++;
        if (line.find("moov", 0) != string::npos) {
            testedFile = false;
            break;
        }
    }
    return !testedFile;
}

int checkFile (string fileName)
{
    // if (!is_video_file_good(fileName)){ cout << "Bad video file..." << endl; return -1;}
    // Set up capture
    VideoCapture capture(fileName);
    if ( !capture.isOpened() )  // if not success, exit program
    {
         cout << "Cannot open the video file" << endl;
         return -1;
    }
    // Take images and convert them to gray
    const int DELAY = 500; // in mseconds, take a picture every 1/2 second
    Mat result, result_cropped;
    Mat prev_frame = result = getFrame(capture, 1);
    Mat current_frame = getFrame(capture, 1);
    Mat next_frame = getFrame(capture, 1);
    if (!success) return -1;
    cvtColor(current_frame, current_frame, CV_RGB2GRAY);
    cvtColor(prev_frame, prev_frame, CV_RGB2GRAY);
    cvtColor(next_frame, next_frame, CV_RGB2GRAY);

    // d1 and d2 for calculating the differences
    // result, the result of and operation, calculated on d1 and d2
    // number_of_changes, the amount of changes in the result matrix.
    // color, the color for drawing the rectangle when something has changed.
    Mat d1, d2, motion;
    int number_of_changes, number_of_sequence = 0;
    Scalar mean_, color(0,255,255); // yellow

    // Detect motion in window
    int x_start = 1, x_stop = current_frame.cols-1;
    int y_start = 1, y_stop = current_frame.rows-1;
    // If more than 'there_is_motion' pixels are changed, we say there is motion
    // and store an image on disk
    int there_is_motion = 5;

    // Maximum deviation of the image, the higher the value, the more motion is allowed
    int max_deviation = 50;

    // Erode kernel
    Mat kernel_ero = getStructuringElement(MORPH_RECT, Size(2,2));

    // All settings have been set, now go in endless loop and
    // take as many pictures you want..
    int numToCheck = 100;
    int skipFrames = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_COUNT)/numToCheck);
    int counter = 0;
    int countThresh = 10;
    for (int i = 0; i < numToCheck; i++){
        // Take a new image
        prev_frame = current_frame;
        current_frame = next_frame;
        next_frame = getFrame(capture, skipFrames);
        if(next_frame.empty()) break;
        result = next_frame;
        if (!success) break;
        cvtColor(next_frame, next_frame, CV_RGB2GRAY);
        // Calc differences between the images and do AND-operation
        // threshold image, low differences are ignored (ex. contrast change due to sunlight)
        absdiff(prev_frame, next_frame, d1);
        absdiff(next_frame, current_frame, d2);
        bitwise_and(d1, d2, motion);
        threshold(motion, motion, 35, 255, CV_THRESH_BINARY);
        erode(motion, motion, kernel_ero);
        number_of_changes = detectMotion(motion, result, result_cropped,  x_start, x_stop, y_start, y_stop, max_deviation, color);
        // If a lot of changes happened, we assume something changed.
        if(number_of_changes>=there_is_motion)
        {
            counter += 1;
            number_of_sequence++;
        }
        if (counter >= countThresh) return 1;
    }
    return 0;    
}
int main (int argc, char * const argv[]){cout << checkFile(argv[1]) << endl;}