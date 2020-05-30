//
// Created by enrico on 30/05/20.
//

#ifndef LAB6_POINTS_H
#define LAB6_POINTS_H

#endif //LAB6_POINTS_H

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

const std::vector<std::vector<cv::Point2f>> two_corners = {
        {cv::Point2f(X_TOP1, Y_TOP1), cv::Point2f(X_BOTTOM1, Y_BOTTOM1)},
        {cv::Point2f(X_TOP2, Y_TOP2), cv::Point2f(X_BOTTOM2, Y_BOTTOM2)},
        {cv::Point2f(X_TOP3, Y_TOP3), cv::Point2f(X_BOTTOM3, Y_BOTTOM3)},
        {cv::Point2f(X_TOP4, Y_TOP4), cv::Point2f(X_BOTTOM4, Y_BOTTOM4)}
};