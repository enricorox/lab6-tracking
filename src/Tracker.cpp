#include "Tracker.h"

using namespace cv;
using namespace std;

Tracker::Tracker(std::string path_video, std::string path_objs){
	// collect images
	// I assume thy are *.png
	std::vector<string> im_files;
	cv::utils::fs::glob(path_objs, PATTERN, im_files);
	for(auto& name : im_files)
		obj_img.push_back(imread(name));

	// collect video frame
	VideoCapture cap(path_video);
	if(cap.isOpened()){ // check if we succeeded
		while(true){
			// extract frame
			Mat frame;
			cap >> frame;

			// save frame
			src_video.push_back(frame);
		}
	}
}

std::vector<cv::Mat> Tracker::computeTracking(){
	init();
	track();
	vector<cv::Mat> v;
	return v;
}

std::vector<Matching> void Tracker::init(){

}

std::vector<cv::Mat> Tracker::track(){
	std::vector<cv::Mat> v;
	return v;
}
