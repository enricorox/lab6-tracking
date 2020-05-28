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
		//namedWindow("images", WINDOW_NORMAL);
		//imshow("images", imread(name));
		//while(waitKey() != 'n');
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
			//imshow("Video IN", frame);
			//waitKey(FRAMERATE);
		}
	}
	cout<<"Read "<<src_video.size()<<" frames."<<endl;
}

std::vector<cv::Mat> Tracker::computeTracking(){
	vector<TrackRect> t = init();
	vector<Mat> v = track(t);
	return v;
}

std::vector<TrackRect> Tracker::init(){
	vector<TrackRect> result;

	// build rect vectors
	Rect2i rects[4];
	rects[0] = Rect2i(Vec2i(X_TOP1, Y_TOP1), Vec2i(X_BOTTOM1, Y_BOTTOM1));
	rects[1] = Rect2i(Vec2i(X_TOP2, Y_TOP2), Vec2i(X_BOTTOM2, Y_BOTTOM2));
	rects[2] = Rect2i(Vec2i(X_TOP3, Y_TOP3), Vec2i(X_BOTTOM3, Y_BOTTOM3));
	rects[3] = Rect2i(Vec2i(X_TOP4, Y_TOP4), Vec2i(X_BOTTOM4, Y_BOTTOM4));

	// extract first frame
	Mat frame = src_video.at(0);

	// use orb detector
	Ptr<ORB> orb = ORB::create(N_FEATURE_FRAME,SCALE_FACTOR);

	// find frame features
	vector<KeyPoint> frame_keypoints;
	orb->detect(frame, frame_keypoints);
	Mat frame_descriptors;
	orb->compute(frame, frame_keypoints, frame_descriptors);

	// draw keypoint
	Mat key_im;
	drawKeypoints(frame, frame_keypoints, key_im);

	// show keypoints
	//imshow("Keypoints", key_im);
	//waitKey();

	// find object features and matches
	int obj_counter = 0; // object counter
	for(auto& obj : obj_img){
		orb = ORB::create(N_FEATURE_OBJECT);

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

		cout<<"Number of good matches with object "<<obj_counter<<": "<<points.obj_features.size()<<endl;

		// apply RANSAC
		Mat H, mask;
		H = findHomography(points.obj_features, points.video_features, RANSAC, THRESHOLD, mask);

		// refine matches
		Matching good_matches;
		for(int i = 0; i < points.obj_features.size(); i++)
			if(mask.at<bool>(i)){
				good_matches.obj_features.push_back(points.obj_features.at(i));
				good_matches.video_features.push_back(points.video_features.at(i));
			}
		cout<<"Saved only "<<good_matches.obj_features.size()<<" matches"<<endl;

		// draw rect on obj
		Vec2f p1 = Vec2i(rects[obj_counter].x, rects[obj_counter].y);
		Vec2f p2 = Vec2i(rects[obj_counter].x, rects[obj_counter].y + rects[obj_counter].height);
		Vec2f p3 = Vec2i(rects[obj_counter].x + rects[obj_counter].width, rects[obj_counter].y + rects[obj_counter].height);
		Vec2f p4 = Vec2i(rects[obj_counter].x + rects[obj_counter].width, rects[obj_counter].y);
		Mat obj_rect = drawRect(obj, colors[obj_counter], THICKNESS, p1, p2, p3, p4);

		// draw rect on frame
		Vec2f q1 = project(H, p1);
		Vec2f q2 = project(H, p2);
		Vec2f q3 = project(H, p3);
		Vec2f q4 = project(H, p4);
		Mat frame_rect = drawRect(frame, colors[obj_counter], THICKNESS, q1, q2, q3, q4);

		// draw matches
		Mat drawn_matches_image;
		drawMatches(obj_rect, obj_keypoints, frame_rect, frame_keypoints, matches,
				drawn_matches_image, Scalar::all(-1), Scalar::all(-1), mask);

		// write drawn_matches_image to disk
		char filename[32];
		sprintf(filename, "%d_matches.png", obj_counter);
		imwrite(filename, drawn_matches_image);

		// save points
		TrackRect t;
		t.p1 = q1;
		t.p2 = q2;
		t.p3 = q3;
		t.p4 = q4;
		result.push_back(t);

		// increase counter
		obj_counter++;
	}
	return result;
}

std::vector<cv::Mat> Tracker::track(vector<TrackRect> t){
	vector<Mat> v;
	vector<vector<Point2f>> prevPts(t.size()), nextPts(t.size());

	// prepare first frame
	Mat f = src_video[0];
	for(int oIdx = 0; oIdx < t.size(); oIdx++){
		nextPts[oIdx].push_back(t[oIdx].p1);
		nextPts[oIdx].push_back(t[oIdx].p2);
		nextPts[oIdx].push_back(t[oIdx].p3);
		nextPts[oIdx].push_back(t[oIdx].p4);
		f = drawRect(f, colors[oIdx], THICKNESS, nextPts[oIdx]);
	}
	v.push_back(f);



	// for every frame
	for(int fIdx = 1; fIdx < src_video.size(); fIdx++){
		// update rects
		prevPts = nextPts;

		// for every object
		Mat f = src_video[fIdx];
		for(int oIdx = 0; oIdx < t.size(); oIdx++){
			// compute flow
			Mat status, err;
			calcOpticalFlowPyrLK(src_video[fIdx-1], src_video[fIdx], prevPts[oIdx], nextPts[oIdx], status, err);
			f = drawRect(f, colors[oIdx], THICKNESS, nextPts[oIdx]);
		}
		cout<<"frame added!"<<endl;
		v.push_back(f); // TODO uncomment line
		imshow("Video OUT", f); // TODO comment line
		if(waitKey(0.5*FRAMERATE) == 'q')
			break;
	}
	return v;
}

cv::Mat drawRect(Mat img, Scalar color, int thickness, Vec2f pt1, Vec2f pt2, Vec2f pt3, Vec2f pt4){
	Mat result;
	img.copyTo(result);

	// convert to Point2i
	Vec2i p1 = static_cast<Point2i>(pt1);
	Vec2i p2 = static_cast<Point2i>(pt2);
	Vec2i p3 = static_cast<Point2i>(pt3);
	Vec2i p4 = static_cast<Point2i>(pt4);
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

Vec2i project(Mat H, Vec2f p){
	Mat mul;

	// add one component
	Vec3f q(p.val[0], p.val[1], 1);

	// convert to floating point
	H.convertTo(H, CV_32FC1);

	// matrix multiplication
	mul = H*q;
	cout<<"Projected point:"<<endl<<mul<<endl<<endl;

	// discard last component (should be 1)
	Vec2i result = Vec2i(
			(int) mul.at<float>(0,0),
			(int) mul.at<float>(0,1)
	);

	return result;
}
