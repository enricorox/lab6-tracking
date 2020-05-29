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

//#define PATTERN "*.png"
#define PATTERN "*.png"

// Max number of match
#define K_MATCH 1

// distance refine
#define RATIO 2
// RANSAC
#define THRESHOLD 1.25
#define THRESHOLD_DYN 1.25

// ORB parameters
#define N_FEATURE_OBJECT 4500//5000//4800
#define N_FEATURE_FRAME 10000//20000//37500
#define SCALE_FACTOR 2

// Lukas-Kanade parameters
#define WIN_SIZE 7,7//21,21//7,7
#define MAX_PYR_LV 3

// Corners obj 1
#define X_TOP1 30//346//30
#define Y_TOP1 20//132//20
#define X_BOTTOM1 925//2707//925
#define Y_BOTTOM1 1235//3928//1235
// Corners obj 2
#define X_TOP2 30//214//30
#define Y_TOP2 25//99//25
#define X_BOTTOM2 833//2938//833
#define Y_BOTTOM2 1249//3994//1249
// Corners obj 3
#define X_TOP3 25//165//25
#define Y_TOP3 40//594//40
#define X_BOTTOM3 884//2905//884
#define Y_BOTTOM3 1244//3994//1244
// Corners obj 4
#define X_TOP4 25//297//25
#define Y_TOP4 40//165//40
#define X_BOTTOM4 881//2905//881
#define Y_BOTTOM4 1259//3928//1259

#define THICKNESS 3

#define GREEN 0,255,0
#define BLUE 255,0,0
#define RED 0,0,255
#define CYANO 255,255,0

// framerate in milliseconds
#define FRAMERATE 16

struct Matching{
	std::vector<cv::Point2f> obj_features, video_features;
};

struct TrackRect{
	cv::Point2f p1,p2,p3,p4;
};

// build color vector
const cv::Scalar colors[] = { cv::Scalar(RED), cv::Scalar(BLUE),
	                           cv::Scalar(GREEN), cv::Scalar(CYANO)};

const std::vector<std::vector<cv::Point2f>> two_corners = {
	{cv::Point2f(X_TOP1, Y_TOP1), cv::Point2f(X_BOTTOM1, Y_BOTTOM1)},
	{cv::Point2f(X_TOP2, Y_TOP2), cv::Point2f(X_BOTTOM2, Y_BOTTOM2)},
	{cv::Point2f(X_TOP3, Y_TOP3), cv::Point2f(X_BOTTOM3, Y_BOTTOM3)},
	{cv::Point2f(X_TOP4, Y_TOP4), cv::Point2f(X_BOTTOM4, Y_BOTTOM4)}
};

cv::Mat drawRect(cv::Mat img, cv::Scalar color, int thickness, cv::Point2f pt1, cv::Point2f pt2, cv::Point2f pt3, cv::Point2f pt4);

cv::Mat drawRect(cv::Mat img, cv::Scalar color, int thickness, TrackRect t);

cv::Mat drawRect(cv::Mat img, cv::Scalar color, int thickness, std::vector<cv::Point2f> points);

cv::Point2f project(cv::Mat H, cv::Point2f p);

std::vector<cv::Point2f> project(cv::Mat H, std::vector<cv::Point2f> vecs);

std::vector<cv::Point2f> extractCorners(cv::Rect2f r);

class Tracker{
// members
private:
	std::vector<cv::Mat> src_video, obj_img;

// functions
public:
	Tracker(std::string path_video, std::string path_objs);
	std::vector<cv::Mat> computeTracking();

private:
	std::vector<Matching> init();
	std::vector<cv::Mat> track(std::vector<Matching> t);


};
