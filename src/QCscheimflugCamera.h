// QCCalibration.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>

using namespace std;
using namespace cv;

class QCscheimflugCamera {
public:

	QCscheimflugCamera(int boardWidth, int boardHeight);
	~QCscheimflugCamera();
	void setPictureNums(int& imageNums);
	void changeBoardWidth(int& boardWidth);
	void changeBoardHeight(int& boardHeight);
	void changeSquareSize(int& squareSize);
	void changeBoardSize(int& boardWidth, int& boardHeight);
	void clear();
	bool fastCheckCorners(Mat image, vector<Point2f> corners);
	bool extractCorners(Mat leftImage, Mat rightImage);
	void findCircleCenter(Mat Image);

	bool extractUndisCorners(Mat leftImage, Mat rightImage);
	void calObjectPoints();
	bool cameraCalibration();
	void reconstruction(Mat leftImage, Mat rightImage, int number);
	void convert2Object();
	bool checkAndDrawCorners(cv::Mat& image);
	void undistortion_iterate(Mat& Image);
	void undistortion_gaussionNewton(Mat& leftImage, Mat& rightImage);
	void readData();
	void vertify_undistortion();
	void vertify_triangular(int number);
	void vertify_projection(int number);
	void vertify();


private:

	int pictureNums;
	int cornerNums;
	int squareSize;
	int boardWidth, boardHeight;

	Size board_Size;
	Size cameraSize;
	string result_yml = "result.yml";

	Mat cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0));
	Mat Cd = Mat(1, 14, CV_64FC1, Scalar::all(0));



	vector<Point3f> storeLine;
	Mat R, T, E, F;
	Mat trans;

	vector<Point2f> corners;
	vector<vector<Point2f>>  allCorners;

	vector<Point2f> undisCorners;
	vector<vector<Point2f>>  allUndisCorners;

	vector<Point3f> objectPoints;
	vector<vector<Point3f>>  allObjectPoints;

	
};


// TODO: 在此处引用程序需要的其他标头。
