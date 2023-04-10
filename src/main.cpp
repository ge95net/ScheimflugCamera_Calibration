#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "QCscheimflugCamera.h"
using namespace std;
using namespace cv;

int main()
{
	int pictureNums = 10;
	QCscheimflugCamera QCscheimflugCamera(18, 23);
	QCscheimflugCamera.setPictureNums(pictureNums);
	cv::FileStorage fs("result.yml", FileStorage::READ);
	bool calibration;
	if (fs.isOpened()) {
		calibration = false;
	}
	else {
		calibration = true;
	}

	for (int i = 0; i < pictureNums; i++)
	{


		Mat Image = imread("D:\\research\\SheimflugCamera\\ScheimflugCamera\\TiltLens_Data\\Data\\" + to_string(i + 1)+ ".bmp", 0);
		QCscheimflugCamera.findCircleCenter(Image);
		cout << "Image Sequence " << i + 1 << " Success" << endl;

		if (calibration == false)
		{
			QCscheimflugCamera.readData();
			QCscheimflugCamera.undistortion_iterate(Image);
			//QCscheimflugCamera.vertify_undistortion();
			//QCscheimflugCamera.vertify();
		}

	}



	if (calibration == true)
	{
		QCscheimflugCamera.cameraCalibration();
	}

	cout << "Camera Calibration Finished" << endl;
	system("pause");
	return 0;

}