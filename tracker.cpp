#include "tracker.h"

Tracker::Tracker(cv::String video_path, cv::String obj_imgs_path, cv::String out_video_path, double obj_resize_ratio)
{
	// ***************** READ VIDEO *****************
	src_video.open(video_path);
	// if no video is found, no frames will be loaded

	cv::Size s((int)src_video.get(cv::CAP_PROP_FRAME_WIDTH),
				(int)src_video.get(cv::CAP_PROP_FRAME_HEIGHT));

	out_video.open(out_video_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G') , src_video.get(cv::CAP_PROP_FPS), s);


	// ***************** READ OBJS *****************

	vector<cv::String> img_paths;
	cv::utils::fs::glob(obj_imgs_path, "*.png", img_paths);

	// if no images are found,no objects will be loaded

	cv::Mat current_img;
	cv::Vec3i random_color;

	for (auto& img_path : img_paths)
	{
		current_img = cv::imread(img_path);
		cv::resize(
			current_img, current_img,
			cv::Size(
				current_img.cols / obj_resize_ratio,
				current_img.rows / obj_resize_ratio
			)
		);

		obj_imgs.push_back(current_img);

		random_color = { rand() % 256, rand() % 256, rand() % 256 };
		obj_shape_colors.push_back(random_color);

	}

}

bool Tracker::isLoaded()
{
	return !obj_imgs.empty() && src_video.isOpened();
}

vector<cv::Point2f> Tracker::findObjCorners(cv::Mat obj_img, double reg_of_int_ratio)
{
	vector<cv::Point2f> obj_corners;
	int width = obj_img.cols;
	int height = obj_img.rows;

	cv::Mat obj_img_grey;
	cv::cvtColor(obj_img, obj_img_grey, cv::COLOR_BGR2GRAY);

	int reg_of_int_width  = (int)(width  * reg_of_int_ratio);
	int reg_of_int_height = (int)(height * reg_of_int_ratio);

	vector<cv::Rect> regions_of_interest;

	cv::Rect top_left(
		0, 0,
		reg_of_int_width, reg_of_int_height
	);
	regions_of_interest.push_back(top_left);
	cv::Rect top_right(
		width - reg_of_int_width, 0,
		reg_of_int_width, reg_of_int_height
	);
	regions_of_interest.push_back(top_right);
	cv::Rect bottom_right(
		width - reg_of_int_width, height - reg_of_int_height,
		reg_of_int_width, reg_of_int_height
	);
	regions_of_interest.push_back(bottom_right);
	cv::Rect bottom_left(
		0, height - reg_of_int_height,
		reg_of_int_width, reg_of_int_height
	);
	regions_of_interest.push_back(bottom_left);

	vector<cv::Point2f> corner_found;

	for (int i = 0; i < regions_of_interest.size(); i++)
	{
		cv::Rect reg_of_int = regions_of_interest[i];
		cv::Point2f tl = reg_of_int.tl();


		cv::goodFeaturesToTrack(cv::Mat(obj_img_grey(reg_of_int)), corner_found,
			1, 0.05, min(width, height) / 3);

		if (corner_found.size() == 1)
			obj_corners.push_back(cv::Point2f(corner_found[0].x + tl.x, corner_found[0].y + tl.y));

	}
	
	

	return obj_corners;
}

void Tracker::drawPoly(cv::Mat img, vector<cv::Point2f> corners, cv::Vec3i color, int thickness)
{
	int n_pts = corners.size();

	for (int i = 0; i < n_pts; i++)
	{
		cv::Point pt1 = corners[i];
		cv::Point pt2 = corners[(i + 1) % n_pts];
		cv::line(img, pt1, pt2, color, thickness);
	}
}

vector<Matching> Tracker::init()
{
	// if something is not properly loaded, exit with an empty vector
	if (!isLoaded()) return vector<Matching>();

	// resulting vector of matchings
	vector<Matching> matchings;

	//cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
	cv::Ptr<cv::ORB> sift = cv::ORB::create();
	cv::BFMatcher matcher;

	// ***************** VIDEO FEATURES DETECTION *****************
	vector<cv::KeyPoint> frame_keypoints;
	cv::Mat frame_descriptors;
	cv::Mat first_frame;
	src_video >> first_frame;
	cv::Mat frame_img_view = first_frame.clone();
	

	// detect features of the first video frame
	sift->detectAndCompute(first_frame, cv::Mat(), frame_keypoints, frame_descriptors);

	// ******************** OBJ IMAGES TREATMENT ********************
	vector<cv::KeyPoint> obj_keypoints;
	cv::Mat obj_descriptors;
	cv::Mat obj_img;
	cv::Vec3i obj_shape_color;

	cv::Mat homography;

	for (int i = 0; i < obj_imgs.size(); i++)
	{

		// get data on the current object image
		obj_img			= obj_imgs[i];
		obj_shape_color = obj_shape_colors[i];

		cv::Mat obj_img_view = obj_img.clone();

		// ************ OBJ IMG FEATURES DETECTION ************
		sift->detectAndCompute(obj_img, cv::Mat(), obj_keypoints, obj_descriptors);

		// ******** KEYPOINTS MATCHING (OBJ <-> VIDEO) ********
		vector<cv::DMatch> matches;
		// match current obj features with the frame features
		matcher.match(obj_descriptors, frame_descriptors, matches, cv::Mat());

		// show matches
		cv::Mat matches_img;
		cv::drawMatches(obj_img, obj_keypoints, first_frame, frame_keypoints, matches, matches_img);
		//cv::imshow("Matches " + to_string(i), matches_img);

		// *************** MATCHES THRESHOLDING ***************
		// introduced because some objects were not well identified
		// using this thresholding operation all objects are found

		// find minimum distance to use in the threshold
		double min_dist = INFINITY;

		for (auto& match : matches)
		{
			if (match.distance < min_dist)
				min_dist = match.distance;

		}

		double threshold = RATIO * min_dist;

		// select couples of features which match with a distance inside the threshold
		vector<cv::Point2f> selected_obj_pts;
		vector<cv::Point2f> selected_frame_pts;

		for (auto& match : matches)
		{
			if (match.distance < threshold)
			{
				selected_obj_pts.push_back(obj_keypoints[match.queryIdx].pt);
				selected_frame_pts.push_back(frame_keypoints[match.trainIdx].pt);
			}
		}

		// ********** KEYPOINTS FILTERING WITH RANSAC **********
		vector<uchar> inliers;
		// set threshold to 10
		homography = cv::findHomography(selected_obj_pts, selected_frame_pts, inliers, cv::RANSAC, 3.0);

		vector<cv::Point2f> inlier_obj_pts;
		vector<cv::Point2f> inlier_frame_pts;

		for (int j = 0; j < inliers.size(); j++)
		{
			if (inliers[j]) {
				inlier_obj_pts.push_back(selected_obj_pts[j]);
				inlier_frame_pts.push_back(selected_frame_pts[j]);
				
				cv::circle(obj_img_view, selected_obj_pts[j], 5, obj_shape_color, 1);
				cv::circle(frame_img_view, selected_frame_pts[j], 5, obj_shape_color, 1);
				
			}

		}
		
		// ************** OBJ 4 CORNERS DETECTION **************
		vector<cv::Point2f> obj_corners;
		vector<cv::Point2f> video_obj_corners;

		int scaling = 1;
		do
		{
			obj_corners = Tracker::findObjCorners(obj_img, REG_OF_INT_RATIO * scaling);
			scaling++;
		} while (obj_corners.size() != 4);


		// project corners on the video through the homography
		cv::perspectiveTransform(obj_corners, video_obj_corners, homography);

		for (auto& pt : obj_corners)
		{
			cv::circle(obj_img_view, pt, 5, obj_shape_color, cv::FILLED);
		}

		for (auto& pt : video_obj_corners)
		{
			cv::circle(frame_img_view, pt, 5, obj_shape_color, cv::FILLED);
		}


		Matching matching = { inlier_obj_pts, inlier_frame_pts, video_obj_corners };

		matchings.push_back(matching);
		
		cv::imshow("Object image " + to_string(i), obj_img_view);

	}

	cv::imshow(WIN_VIDEO, frame_img_view);

	return matchings;
}


void Tracker::track(vector<Matching> matchings)
{
	// reroll the src_video to the first frame
	src_video.set(cv::CAP_PROP_POS_FRAMES, 0);

	cv::Mat next_frame;
	src_video.read(next_frame);
	cv::Mat prev_frame = next_frame.clone();

	cv::Mat next_grey_frame, prev_grey_frame;

	src_video.set(cv::CAP_PROP_POS_FRAMES, 0);

	vector<cv::Point2f> moving_pts;
	vector<cv::Point2f> moving_corners;

	cv::Mat draw_frame;

	// for each frame
	while (src_video.read(next_frame))
	{

		
		cv::cvtColor(prev_frame, prev_grey_frame, cv::COLOR_BGR2GRAY);
		cv::cvtColor(next_frame, next_grey_frame, cv::COLOR_BGR2GRAY);

		draw_frame = next_frame.clone();

		// for each object
		for (int i = 0; i < matchings.size(); i++)
		{

			vector<uchar> flow_found;
			vector<float> err;

			cv::calcOpticalFlowPyrLK(
				prev_grey_frame, next_grey_frame,
				matchings[i].video_features, moving_pts,
				flow_found, err
			);

			double min_err = INFINITY;

			for (auto& error : err)
			{
				if (error < min_err)
					min_err = error;

			}

			double threshold = ERR_RATIO * min_err;

			vector<cv::Point2f> selected_prev_pts;
			vector<cv::Point2f> selected_next_pts;

			for (int j = 0; j < flow_found.size(); j++)
			{
				// for now put to true
				if (true || (flow_found[i] && err[i] <= threshold))
				{
					selected_prev_pts.push_back(matchings[i].video_features[j]);
					selected_next_pts.push_back(moving_pts[j]);
				}
			}

			cv::Mat homography = cv::findHomography(selected_prev_pts, selected_next_pts);


			for (auto& pt : selected_next_pts)
			{
				cv::circle(draw_frame, pt, 5, obj_shape_colors[i], 2);
			}

			cv::perspectiveTransform(matchings[i].video_obj_corners, moving_corners, homography);

			Tracker::drawPoly(draw_frame, moving_corners, obj_shape_colors[i], 3);

			matchings[i].video_features = selected_next_pts;
			matchings[i].video_obj_corners = moving_corners;
		}

		out_video << draw_frame;

		cv::imshow(WIN_VIDEO, draw_frame);
		cv::waitKey(16);

		prev_frame = next_frame.clone();
	}

}

void Tracker::computeTracking()
{
	vector<Matching> matchings = init();
	track(matchings);
}