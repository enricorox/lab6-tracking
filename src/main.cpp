#include <opencv2/highgui.hpp>
#include "Tracker.h"

#define PATH_VIDEO "../data/video.mov"
#define PATH_OBJECTS "../data/objects/"

#define PATH_VIDEO1 "../data1/video.mp4"
#define PATH_OBJECTS1 "../data1/objects/"

int main(){
	// do the hard work
	Tracker t(PATH_VIDEO, PATH_OBJECTS);
	t.showTracking();
    std::cout<<"Exit."<<std::endl;
}
