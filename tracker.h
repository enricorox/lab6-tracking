#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/photo.hpp>

#include "Util.h"


using namespace std;

typedef struct
{
	vector<cv::Point2f> obj_features;
	vector<cv::Point2f> video_features;
	vector<cv::Point2f> video_obj_corners;

} Matching;


class Tracker
{
	
public:

	Tracker(cv::String video_path, cv::String obj_imgs_path, cv::String out_video_path, double obj_resize_ratio);

	bool isLoaded();
	void computeTracking();

private:

	vector<Matching> init();
	void track(vector<Matching> matchings);

	static vector<cv::Point2f> findObjCorners(cv::Mat obj_img, double reg_of_int_ratio);
	static void drawPoly(cv::Mat img, vector<cv::Point2f> corners, cv::Vec3i color, int thickness);

	cv::VideoCapture src_video;
	vector<cv::Mat> obj_imgs;
	vector<cv::Vec3i> obj_shape_colors;

	cv::VideoWriter out_video;

};