#include "Tracker.h"

using namespace cv;
using namespace std;

Tracker::Tracker(std::string path_video, std::string path_objs){
	// collect images
	// I assume thy are *.png
	cout<<"Reading images..."<<endl;
	std::vector<string> im_files;
	cv::utils::fs::glob(path_objs, PATTERN, im_files);
	for(auto& name : im_files){
		obj_img.push_back(imread(name));
	}
	cout<<"Read "<<obj_img.size()<<" images."<<endl;

	// collect video frame
	cout<<"Reading video..."<<endl;
	VideoCapture cap(path_video);
	if(cap.isOpened()){ // check if we succeeded
		while(true){
			// extract frame
			Mat frame;
			cap >> frame;

			// exit if video ends
			if(frame.empty())
				break;
			// save frame
			src_video.push_back(frame);
		}
	}
	cout<<"Read "<<src_video.size()<<" frames."<<endl;
}

void Tracker::showTracking(){
	vector<Matching> points_vecs = findFeatures();

    vector<vector<Point2f>> next_frame_pts(points_vecs.size());
    vector<vector<Point2f>> obj_pts(points_vecs.size());

    // prepare first frame
    Mat first = src_video[0].clone();
    for(int oIdx = 0; oIdx < points_vecs.size(); oIdx++){
        // initialize points
        next_frame_pts[oIdx] = points_vecs[oIdx].video_features;
        obj_pts[oIdx] = points_vecs[oIdx].obj_features;

        // find homography
        Mat H = findHomography(obj_pts[oIdx], next_frame_pts[oIdx]);

        // compute static corners
        vector<Point2f> corners = extractCorners(cv::Rect2f(two_corners[oIdx][0], two_corners[oIdx][1]));

        // project corners and draw rectangle
        first = drawRect(first, colors[oIdx], THICKNESS, project(H, corners));
    }

    // show frame
    imshow("Video OUT", first);
    waitKey(FRAMERATE);

    int delay = FRAMERATE;
    bool quit = false;
    // for every frame
    for(int fIdx = 1; (fIdx < src_video.size()) && !quit; fIdx++){
        // update points
        vector<vector<Point2f>> prev_frame_pts(next_frame_pts); // deep copy of all arrays

        // for every object
        Mat frame = src_video[fIdx].clone();
        for(int oIdx = 0; oIdx < points_vecs.size(); oIdx++){
            // compute flow
            vector<uchar> status;
            vector<float> err;
            next_frame_pts[oIdx].clear();
            calcOpticalFlowPyrLK(src_video[fIdx-1], src_video[fIdx], prev_frame_pts[oIdx], next_frame_pts[oIdx],
                                 status, err, Size(WIN_SIZE), MAX_PYR_LV);

            // update points
            next_frame_pts[oIdx] = discardPoints(next_frame_pts[oIdx], toBool(status));
            obj_pts[oIdx] = discardPoints(obj_pts[oIdx], toBool(status));



            // find homography
            vector<char> mask;
            Mat H = findHomography(obj_pts[oIdx], next_frame_pts[oIdx], RANSAC, THRESHOLD_DYN, mask);

            // refine points with ransac mask
            next_frame_pts[oIdx] = discardPoints(next_frame_pts[oIdx], toBool(mask));
            obj_pts[oIdx] = discardPoints(obj_pts[oIdx], toBool(mask));

            vector<Point2f> corners = extractCorners(cv::Rect2f(two_corners[oIdx][0], two_corners[oIdx][1]));
            frame = drawRect(frame, colors[oIdx], THICKNESS, project(H, corners));
        }

        // show frame
        imshow("Video OUT", frame);
        switch(waitKey(delay)){
            case 'q': quit = true; break; // q --> exit
            case 'f': delay = 0; break; // f--> frame by frame
            default: delay = FRAMERATE;
        };
    }
}

std::vector<Matching> Tracker::findFeatures(){
	vector<Matching> result;

	// extract first frame
	Mat frame = src_video.at(0);

	// use orb detector
	//Ptr<ORB> orb = ORB::create(N_FEATURE_FRAME,SCALE_FACTOR);
    Ptr<ORB> orb = ORB::create(N_FEATURE_FRAME);

	// find frame features
	vector<KeyPoint> frame_keypoints;
	orb->detect(frame, frame_keypoints);
	Mat frame_descriptors;
	orb->compute(frame, frame_keypoints, frame_descriptors);

	// find object features and matches
	int obj_counter = 0; // object counter
	for(auto& obj : obj_img){
		//orb = ORB::create(N_FEATURE_OBJECT);
        orb = ORB::create(N_FEATURE_OBJECT, SCALE_FACTOR);

		// compute keypoints
		vector<KeyPoint> obj_keypoints;
		orb->detect(obj, obj_keypoints);

		// compute keypoints' descriptors
		Mat obj_descriptors;
		orb->compute(obj, obj_keypoints, obj_descriptors);

		// find matches
		BFMatcher matcher = BFMatcher(NORM_HAMMING, true);
		vector<vector<DMatch>> knn_matches;
		matcher.knnMatch(obj_descriptors, frame_descriptors, knn_matches, K_MATCH);

		//find min distance
		float min_distance = FLT_MAX;
		for(auto& v : knn_matches){
			// skip if no match
			if(v.empty()) continue;

			if(min_distance > v[0].distance)
				min_distance = v[0].distance;
		}

		// extract points from knn_matches
		// and refine match with ratio test
		Matching points;
		vector<DMatch> matches;
		for(auto& v : knn_matches){
			// skip if no match or bad matches
			if(v.empty() || v[0].distance > RATIO*min_distance ) continue;

			// save match
			matches.push_back(v.at(0));

			// find indexes
			int objIdx = v.at(0).queryIdx;
			int frameIdx = v.at(0).trainIdx;

			// find points for homography
			Point2f obj_p = obj_keypoints.at(objIdx).pt;
			points.obj_features.push_back(obj_p);

			Point2f frame_p = frame_keypoints.at(frameIdx).pt;
			points.video_features.push_back(frame_p);
		}

		cout<<"Number of matches with object "<<obj_counter<<": "<<points.obj_features.size()<<endl;

		// find homography
		Mat H;
		vector<char> mask;
		H = findHomography(points.obj_features, points.video_features, RANSAC, THRESHOLD, mask);

		// refine matches with RANSAC mask
		Matching good_matches;
		good_matches.obj_features = discardPoints(points.obj_features, toBool(mask), obj_counter);
        good_matches.video_features = discardPoints(points.video_features, toBool(mask), obj_counter);
		cout<<"Saved only "<<good_matches.obj_features.size()<<" matches"<<endl;

		// save points
		result.push_back(good_matches);

		// ---draw rect on obj---
		// extract corners
		vector<Point2f> corners = extractCorners(cv::Rect2f(two_corners[obj_counter][0], two_corners[obj_counter][1]));
		Mat obj_rect = drawRect(obj, colors[obj_counter], THICKNESS, corners);

		// ---draw rect on frame---
		// project corners
		vector<Point2f> projected_corners = project(H, corners);
		Mat frame_rect = drawRect(frame, colors[obj_counter], THICKNESS, projected_corners);

		// ---draw matches between frame and object ---
		Mat drawn_matches_image;
		drawMatches(obj_rect, obj_keypoints, frame_rect, frame_keypoints, matches,
				drawn_matches_image, Scalar::all(-1), Scalar::all(-1), mask);

		// write drawn_matches_image to disk
		char filename[32];
		sprintf(filename, "%d_matches.png", obj_counter);
		imwrite(filename, drawn_matches_image);

		// increase counter
		obj_counter++;
	}
	return result;
}

cv::Mat drawRect(Mat img, Scalar color, int thickness, Point2f pt1, Point2f pt2, Point2f pt3, Point2f pt4){
	Mat result = img.clone();

	// convert to Point2i
	Point2i p1 = static_cast<Point2i>(pt1);
	Point2i p2 = static_cast<Point2i>(pt2);
	Point2i p3 = static_cast<Point2i>(pt3);
	Point2i p4 = static_cast<Point2i>(pt4);

	// draw lines
	line(result, p1, p2, color, thickness);
	line(result, p2, p3, color, thickness);
	line(result, p3, p4, color, thickness);
	line(result, p4, p1, color, thickness);

	return result;
}

cv::Mat drawRect(Mat img, Scalar color, int thickness, TrackRect t){
	return drawRect(img, color, thickness, t.p1, t.p2, t.p3, t.p4);
}

cv::Mat drawRect(Mat img, Scalar color, int thickness, vector<Point2f> points){
	return drawRect(img, color, thickness, points[0], points[1], points[2], points[3]);
}

Point2f project(Mat H, Point2f p){
	if(H.empty()){
		cout<<"Homography matrix is empty!"<<endl;
		exit(-1);
	}
	// add one component
	Vec3f q(p.x, p.y, 1);

	// convert to floating point
	H.convertTo(H, CV_32FC1);

	// matrix multiplication
	Mat mul = H*q;

	//cout<<"Projected point:"<<endl<<mul<<endl<<endl;

	// discard last component (should be 1)
	Point2f result = Point2f(
			(int) mul.at<float>(0,0),
			(int) mul.at<float>(0,1)
	);

	return result;
}

vector<Point2f> project(Mat H, vector<Point2f> vecs){
	vector<Point2f> result;
	for(auto& v : vecs)
		result.push_back(project(H,v));
	return result;
}

vector<Point2f> extractCorners(Rect2f r){
	Point2f p1(r.x, r.y);
	Point2f p2(r.x, r.y + r.height);
	Point2f p3(r.x + r.width, r.y + r.height);
	Point2f p4(r.x + r.width, r.y);

	return {p1,p2,p3,p4};
}

std::vector<cv::Point2f> discardPoints(std::vector<cv::Point2f> points, std::vector<bool> mask, int idx){
    if(points.size() != mask.size()){
        cout<<"Error: vector and its mask must have same dimensions!"<<endl;
        exit(-1);
    }

    vector<Point2f> result;
    for(int i = 0; i < points.size(); i++)
        if(mask[i])
            result.push_back(points[i]);

    if(points.size() != result.size()){
        if(result.empty()) {
            cout << "Error: all points has been discarded"
                 << ((idx > -1) ? " from object " + to_string(idx) : ".") << endl;
            exit(-1);
        } else
            cout << "Warning: Discarded " <<points.size() - result.size() << " points"
                 << ((idx > -1) ? " from object " + to_string(idx) : ".") << endl;
    }
    return result;
}

std::vector<bool> toBool(vector<char> v){
    vector<bool> r;
    for(auto& e : v)
        r.push_back((e) ? true : false);
    return r;
}

std::vector<bool> toBool(vector<uchar> v){
    vector<bool> r;
    for(auto& e : v)
        r.push_back((e) ? true : false);
    return r;
}