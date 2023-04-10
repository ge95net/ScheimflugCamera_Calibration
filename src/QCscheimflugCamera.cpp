#include "QCscheimflugCamera.h"
#include <filesystem>
#include <algorithm>

using namespace std;
using namespace cv;

QCscheimflugCamera::QCscheimflugCamera(int boardWidth, int boardHeight)
{

	this->boardWidth = boardWidth;
	this->boardHeight = boardHeight;

	cornerNums = boardWidth * boardHeight;
	board_Size = Size(boardWidth, boardHeight);

	clear();


}


QCscheimflugCamera::~QCscheimflugCamera() {}


void QCscheimflugCamera::setPictureNums(int& imageNums)
{
	int default_pictureNums = 3;
	if (imageNums < default_pictureNums)
	{
		cout << "images are not enough" << endl;
	}
	else
	{
		pictureNums = imageNums;
	}
}

void QCscheimflugCamera::changeBoardWidth(int& width)
{
	boardWidth = width;
}

void QCscheimflugCamera::changeBoardHeight(int& height)
{
	boardHeight = height;

}

void QCscheimflugCamera::changeSquareSize(int& squareSize)
{
	this->squareSize = squareSize;
}


void QCscheimflugCamera::changeBoardSize(int& width, int& height)
{
	board_Size = Size(width, height);
}


void QCscheimflugCamera::clear()
{
	corners.clear();


}


bool QCscheimflugCamera::fastCheckCorners(Mat image, vector<Point2f> corners)
{
	return cv::findChessboardCorners(image, board_Size, corners);
	
}


void QCscheimflugCamera::findCircleCenter(Mat image)
{
	SimpleBlobDetector::Params params;
	params.blobColor = 255;
	cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);
	findCirclesGrid(image, board_Size, corners, CALIB_CB_SYMMETRIC_GRID | CALIB_CB_CLUSTERING, detector);
	//cornerSubPix(image, corners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 30, 0.1));
	allCorners.push_back(corners);
	cameraSize = Size(image.cols, image.rows);
}





void QCscheimflugCamera::calObjectPoints()
{


	for (int i = 0; i < boardHeight; i++)
	{
		for (int j = 0; j < boardWidth; j++)
		{

			objectPoints.push_back(Point3f(float(j * 0.55), float(i * 0.55), 0));
		}
	}


	for (int k = 0; k < pictureNums; k++)
	{
		allObjectPoints.push_back(objectPoints);
	}
	cout << "size " << allObjectPoints.size() << endl;

}


bool QCscheimflugCamera::cameraCalibration()
{
	if (allCorners.size() < pictureNums)
	{
		cout << "images are not enough" << endl;
		return false;
	}
	else
	{
		cout << "starting calibrating cameras" << endl;
	}

	calObjectPoints();
	Mat perViewErrors;
	Mat stdDeviationsIntrinsics;
	Mat stdDeviationsExtrinsics;
	cameraMatrix.at<double>(0, 0) = 6221;
	cameraMatrix.at<double>(0, 2) = 1295;
	cameraMatrix.at<double>(1, 1) = 6270;
	cameraMatrix.at<double>(1, 2) = 1026;
	cameraMatrix.at<double>(2, 2) = 1;
	double CameraError = cv::calibrateCamera(allObjectPoints, allCorners, cameraSize, cameraMatrix, Cd, R, T, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors, CALIB_RATIONAL_MODEL + CALIB_THIN_PRISM_MODEL + CALIB_TILTED_MODEL,
		TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 150, DBL_EPSILON));

	


	//store
	FileStorage fs(result_yml, FileStorage::WRITE);
	if (!fs.isOpened())
	{
		cout << "fail to open file" << endl;
	}
	else
	{
		cout << "open file" << endl;
		fs << "cameraMatrix" << cameraMatrix << "Cd" << Cd << "R" << R << "T" << T
			<< "stdDeviationsIntrinsics" << stdDeviationsIntrinsics << "stdDeviationsExtrinsics" << stdDeviationsExtrinsics
			<< "perViewErrors" << perViewErrors
			<< "CameraError" << CameraError;
			



		fs.release();

	}

	cout << "CameraError" << CameraError << endl;
	cout << "Calibration Finished" << endl;
	return true;
}


bool QCscheimflugCamera::checkAndDrawCorners(cv::Mat& image)
{
	std::vector<cv::Point2f> corners;
	bool found = cv::findChessboardCorners(image, board_Size, corners, cv::CALIB_CB_FAST_CHECK);
	if (found) { drawChessboardCorners(image, board_Size, corners, found); }

	return found;
}



void QCscheimflugCamera::undistortion_iterate(Mat& Image)
{
	cout << "start undistortion" << endl;

	double k1 = Cd.at<double>(0, 0);
	double k2 = Cd.at<double>(0, 1);
	double p1 = Cd.at<double>(0, 2);
	double p2 = Cd.at<double>(0, 3);
	double k3 = Cd.at<double>(0, 4);
	double k4 = Cd.at<double>(0, 5);
	double k5 = Cd.at<double>(0, 6);
	double k6 = Cd.at<double>(0, 7);
	double s1 = Cd.at<double>(0, 8);
	double s2 = Cd.at<double>(0, 9);
	double s3 = Cd.at<double>(0, 10);
	double s4 = Cd.at<double>(0, 11);
	double taux = Cd.at<double>(0, 12);
	double tauy = Cd.at<double>(0, 13);

	double fx = cameraMatrix.at<double>(0, 0);
	double fy = cameraMatrix.at<double>(1, 1);
	double cx = cameraMatrix.at<double>(0, 2);
	double cy = cameraMatrix.at<double>(1, 2);

	//rotation
	Mat Rtau = Mat(3, 3, CV_64FC1);
	Mat S = Mat(3, 3, CV_64FC1, Scalar::all(0));
	Mat Rtau_invert;
	Mat S_invert;
	Mat cameraMatrix_invert;
	Mat TiltMatrix;
	Mat TiltMatrix_invert;
	Rtau.at<double>(0, 0) = cos(tauy);
	Rtau.at<double>(0, 1) = sin(tauy) * sin(taux);
	Rtau.at<double>(0, 2) = -sin(tauy) * cos(taux);
	Rtau.at<double>(1, 0) = 0;
	Rtau.at<double>(1, 1) = cos(taux);
	Rtau.at<double>(1, 2) = sin(taux);
	Rtau.at<double>(2, 0) = sin(tauy);
	Rtau.at<double>(2, 1) = -cos(tauy) * sin(taux);
	Rtau.at<double>(2, 2) = cos(tauy) * cos(taux);


	S.at<double>(0, 0) = Rtau.at<double>(2, 2);
	S.at<double>(0, 2) = -Rtau.at<double>(0, 2);
	S.at<double>(1, 1) = Rtau.at<double>(2, 2);
	S.at<double>(1, 2) = -Rtau.at<double>(1, 2);
	S.at<double>(2, 2) = 1;

	TiltMatrix = S * Rtau ;
	

	invert(TiltMatrix, TiltMatrix_invert);
	invert(cameraMatrix, cameraMatrix_invert);


	
	//Mat tiltCoordinate = Mat(3, 1, CV_64FC1);
	//Mat idealCoordinate = Mat(3, 1, CV_64FC1);
	//Mat imageCoordinate_H = Mat(3, 1, CV_64FC1);
	


	// distort the left image
	for (int i = 0; i < corners.size(); i++)
	{
		double u = corners[i].x;
		double v = corners[i].y;
		Mat imageCoordinate_H = (Mat_<double>(3, 1) << u, v, 1);

		Mat tiltCoordinate = cameraMatrix_invert * imageCoordinate_H;
		Mat idealCoordinate = TiltMatrix_invert * tiltCoordinate;
		/*
		Mat idealCoordinateImage = cameraMatrix * idealCoordinate;
		idealCoordinateImage.at<double>(0, 0) = idealCoordinateImage.at<double>(0, 0) / idealCoordinateImage.at<double>(2, 0);
		idealCoordinateImage.at<double>(1, 0) = idealCoordinateImage.at<double>(1, 0) / idealCoordinateImage.at<double>(2, 0);
		idealCoordinateImage.at<double>(2, 0) = idealCoordinateImage.at<double>(2, 0) / idealCoordinateImage.at<double>(2, 0);
		storeLine.push_back(Point3f(idealCoordinateImage.at<double>(0, 0), idealCoordinateImage.at<double>(1, 0), 1));
		*/
		double invPro = idealCoordinate.at<double>(2,0) ? 1. / idealCoordinate.at<double>(2, 0) : 1;

		double x = invPro * idealCoordinate.at<double>(0, 0);
		double y = invPro * idealCoordinate.at<double>(1, 0);



		double x0 = x;
		double y0 = y;
		
		double r_square = x * x + y * y;
		double first_item = (1 + k1 * r_square + k2 * r_square * r_square + k3 * r_square * r_square * r_square)/(1 + k4 * r_square + k5 * r_square * r_square + k6 * r_square * r_square * r_square);

		double x_second_item = 2 * p1 * x * y + p2 * (r_square + 2 * x * x) +s1 * r_square + s2 * r_square * r_square;
		double y_second_item = 2 * p2 * x * y + p1 * (r_square + 2 * y * y) +s3 * r_square + s4 * r_square * r_square;

		double error = 100.12165;

		for (int j = 0; j < 3000; j++)
		{
			if (error < DBL_EPSILON)
			{
				break;
			}


			x = (x0 - x_second_item) / first_item;
			y = (y0 - y_second_item) / first_item;

			r_square = x * x + y * y;

			first_item = (1 + k1 * r_square + k2 * r_square * r_square + k3 * r_square * r_square * r_square)/(1 + k4 * r_square + k5 * r_square * r_square + k6 * r_square * r_square * r_square);

			x_second_item = 2 * p1 * x * y + p2 * (r_square + 2 * x * x) + s1 * r_square +s2 * r_square * r_square;
			y_second_item = 2 * p2 * x * y + p1 * (r_square + 2 * y * y) + s3 * r_square +s4 * r_square * r_square;

			double x_dis = x * first_item + x_second_item;
			double y_dis = y * first_item + y_second_item;

			Mat dis_idealCoordinate = (Mat_<double>(3, 1) << x_dis, y_dis, 1);
			Mat dis_tiltCoordinate = TiltMatrix * dis_idealCoordinate;
			invPro = dis_tiltCoordinate.at<double>(2, 0) ? 1. / dis_tiltCoordinate.at<double>(2, 0) : 1;

			Mat dis_idealImageCoordinate = cameraMatrix * invPro * dis_tiltCoordinate;
			double u_dis = dis_idealImageCoordinate.at<double>(0, 0);
			double v_dis = dis_idealImageCoordinate.at<double>(1, 0);


			error = sqrt(pow((u - u_dis), 2) + pow((v - v_dis), 2));
			//cout << "error=" << error << endl;
		}

		idealCoordinate = (Mat_<double>(3, 1) << x, y, 1);
		
		Mat idealImageCoordinate = cameraMatrix * idealCoordinate;
		double u_proj = idealImageCoordinate.at<double>(0, 0);
		double v_proj = idealImageCoordinate.at<double>(1, 0);

		undisCorners.push_back(Point2d(u_proj, v_proj));
		
	}

	allUndisCorners.push_back(undisCorners);


	cout << "end" << endl;


}



void QCscheimflugCamera::readData()
{
	cv::FileStorage fs(result_yml, cv::FileStorage::READ);
	if (fs.isOpened())
	{
		cout << "read the data" << endl;
		fs["cameraMatrix"] >> cameraMatrix;
		fs["Cd"] >> Cd;
		fs["R"] >> R;
		fs["T"] >> T;
		cout << "read the data successfully" << endl;
	}
	else {
		cout << "result doesn't exist" << endl;
	}
}


void QCscheimflugCamera::vertify_undistortion()
{
	undisCorners.clear();

	cv::undistortPoints(corners, undisCorners, cameraMatrix, Cd, cv::noArray(), cameraMatrix);

	cout << "end" << endl;
}


void QCscheimflugCamera::vertify()
{
	calObjectPoints();
	for (int i = 0; i < pictureNums; i++)
	{
		vector<Point2d> line;
		Mat Rotation;
		Mat Trans = Mat(3, 1, CV_64FC1);
		Mat ExternMat = Mat(3,4, CV_64FC1);

		Mat R1;
		Rodrigues(R.row(i), Rotation);

		Trans.at<double>(0, 0) = T.at<double>(i, 0);
		Trans.at<double>(1, 0) = T.at<double>(i, 1);
		Trans.at<double>(2, 0) = T.at<double>(i, 2);

		Rotation.convertTo(Rotation,CV_64FC1);
		Trans.convertTo(Trans, CV_64FC1);
		ExternMat.convertTo(ExternMat, CV_64FC1);
		hconcat(Rotation, Trans, ExternMat);
		for (int j = 0; j < allObjectPoints[i].size(); j++)
		{
			Mat X = Mat(4, 1, CV_64FC1);
			X.at<double>(0, 0) = allObjectPoints[i][j].x;
			X.at<double>(1, 0) = allObjectPoints[i][j].y;
			X.at<double>(2, 0) = allObjectPoints[i][j].z;
			X.at<double>(3, 0) = 1;
			Mat cameraCoordinate = cameraMatrix* ExternMat* X;
			cameraCoordinate.at<double>(0, 0) = cameraCoordinate.at<double>(0, 0) / cameraCoordinate.at<double>(2, 0);
			cameraCoordinate.at<double>(1, 0) = cameraCoordinate.at<double>(1, 0) / cameraCoordinate.at<double>(2, 0);
			cameraCoordinate.at<double>(2, 0) = cameraCoordinate.at<double>(2, 0) / cameraCoordinate.at<double>(2, 0);
			line.push_back(Point2d(cameraCoordinate.at<double>(0, 0), cameraCoordinate.at<double>(1, 0)));
		}
		cout << "endl" << endl;
	}
}