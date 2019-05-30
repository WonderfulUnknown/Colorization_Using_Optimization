#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

double Threshold = 0.01;
Mat rgb2ntsc(Mat src)
{
	//输入rgb的mat型图
	Mat dst = src.clone();
	src.convertTo(src, CV_32FC3);
	dst.convertTo(dst, CV_32FC3);
	//逐行逐列逐像素点的转换
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++) 
		{
			dst.at<Vec3f>(i, j)[2] = saturate_cast<float>((0.299*src.at<Vec3f>(i, j)[2] +
				0.587*src.at<Vec3f>(i, j)[1] +
				0.114*src.at<Vec3f>(i, j)[0]));
			dst.at<Vec3f>(i, j)[1] = saturate_cast<float>((0.596*src.at<Vec3f>(i, j)[2] +
				-0.274*src.at<Vec3f>(i, j)[1] +
				-0.322*src.at<Vec3f>(i, j)[0]));
			dst.at<Vec3f>(i, j)[0] = saturate_cast<float>((0.211*src.at<Vec3f>(i, j)[2] +
				-0.523*src.at<Vec3f>(i, j)[1] +
				0.312*src.at<Vec3f>(i, j)[0]));
		}
	}
	return dst;
}

int main()
{
	Mat src = imread("example.bmp");
	Mat marked = imread("mine.bmp");
	//转变为float类型的灰度图,float类型0-1就是uchar的0-255
	src.convertTo(src, CV_64F);
	marked.convertTo(marked, CV_64F);
	divide(src, 255, src);
	divide(marked, 255, marked);

	Mat color = abs(src - marked);

	vector<Mat> channels;
	Mat imgBlueChannel;
	Mat imgGreenChannel;
	Mat imgRedChannel;
	split(color, channels);
	imgBlueChannel = channels.at(0);
	imgGreenChannel = channels.at(1);
	imgRedChannel = channels.at(2);

	Mat colorIM = imgBlueChannel + imgGreenChannel + imgRedChannel;
	for (int i = 0; i < colorIM.rows; i++)
	{
		for (int j = 0; j < colorIM.cols; j++)
		{
			if (colorIM.at<double>(i, j) > Threshold)
				colorIM.at<double>(i, j) = 1;
			else
				colorIM.at<double>(i, j) = 0;
		}
	}
	Mat YIQ_gray, YIQ_color;
	YIQ_gray = rgb2ntsc(src);
	YIQ_color = rgb2ntsc(marked);

	Mat YUV = YIQ_color.clone();
	for (int i = 0; i < YUV.rows; i++)
		for (int j = 0; j < YUV.cols; j++)
			YUV.at<Vec3f>(i, j)[0] = YIQ_gray.at<Vec3f>(i, j)[0];
	waitKey(0);
	return 0;
}