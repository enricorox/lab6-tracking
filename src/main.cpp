#include <opencv2/highgui.hpp>
#include "Tracker.h"

#define PATH_VIDEO "../data/video.mov"
#define PATH_OBJECTS "../data/objects/"

int main(){
	// do the hard work
	Tracker t(PATH_VIDEO, PATH_OBJECTS);
	std::vector<cv::Mat> video = t.computeTracking();t.computeTracking();

	// show frames
	for(auto& f : video){
		cv::imshow("tracking", f);
		cv::waitKey(5);
	}
}
