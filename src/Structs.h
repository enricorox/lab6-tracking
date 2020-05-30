#include <vector>
#include <opencv2/core.hpp>

struct Matching{
    std::vector<cv::Point2f> obj_features, video_features;
};

struct TrackRect{
    cv::Point2f p1,p2,p3,p4;
};