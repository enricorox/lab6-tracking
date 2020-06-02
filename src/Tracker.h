#include <string>
#include <vector>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include "Points.h"
#include "Structs.h"

//#define PATTERN "*.png"
#define PATTERN "*.png"

// Max number of match
#define K_MATCH 1

// distance refine
#define RATIO 2
// RANSAC
#define THRESHOLD 1.25
#define THRESHOLD_DYN 15

// ORB parameters
//#define N_FEATURE_OBJECT 4500
#define N_FEATURE_OBJECT 9000
//#define N_FEATURE_FRAME 10000
#define N_FEATURE_FRAME 100000
#define SCALE_FACTOR 2

// Lukas-Kanade parameters
#define WIN_SIZE 7,7//21,21//7,7
#define MAX_PYR_LV 3

#define THICKNESS 3

#define GREEN 0,255,0
#define BLUE 255,0,0
#define RED 0,0,255
#define CYANO 255,255,0

// framerate in milliseconds
#define FRAMERATE 16

// build color vector
const cv::Scalar colors[] = { cv::Scalar(RED), cv::Scalar(BLUE),
	                           cv::Scalar(GREEN), cv::Scalar(CYANO)};

cv::Mat drawRect(cv::Mat img, cv::Scalar color, int thickness, cv::Point2f pt1, cv::Point2f pt2, cv::Point2f pt3, cv::Point2f pt4);

cv::Mat drawRect(cv::Mat img, cv::Scalar color, int thickness, std::vector<cv::Point2f> points);

cv::Point2f project(cv::Mat H, cv::Point2f p);

std::vector<cv::Point2f> project(cv::Mat H, std::vector<cv::Point2f> vecs);

std::vector<cv::Point2f> extractCorners(cv::Rect2f r);

std::vector<cv::Point2f> discardPoints(std::vector<cv::Point2f> points, std::vector<bool> mask, int idx=-1);

std::vector<bool> toBool(std::vector<uchar> vec);

std::vector<bool> toBool(std::vector<char> v);

class Tracker{
// members
private:
	std::vector<cv::Mat> obj_img;
    cv::VideoCapture cap;
    cv::Mat first_frame;

// functions
public:
	Tracker(std::string path_video, std::string path_objs);
	void showTracking();

private:
	std::vector<Matching> findFeatures();
};
