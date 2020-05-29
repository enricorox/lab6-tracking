#include "Lab6.h"

using namespace std;

int main()
{
	
	cv::String data_path = DATA_PATH;
	cv::String video_path = data_path + VIDEO_PATH;
	cv::String tracked_path = data_path + TRACKED_PATH;
	cv::String objs_path = data_path + OBJS_PATH;

	
	Tracker tracker(video_path, objs_path, tracked_path, 2);

	tracker.computeTracking();

	return 0;

}
