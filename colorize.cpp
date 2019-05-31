#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

#define Max 5000
double Threshold = 0.01;
int window_size = 1;//窗口半径

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

void getColorExact(Mat colorIm,Mat ntscIm)
{
	int n, m, imgSize;
	n = ntscIm.rows;
	m = ntscIm.cols;
	imgSize = n * m;
	Mat indsM = Mat::zeros(n, m, CV_64F);//储存1-imgSize
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			indsM.at<double>(i, j) = (i + 1)*(j + 1);
	int lbInds[Max];//存储所有colorIm中大于0的值的索引
	int temp = 0;
	for (int i = 0; i < colorIm.rows; i++)
	{
		for (int j = 0; j < colorIm.cols; j++)
		{
			if (colorIm.at<double>(i, j) > 0)
				lbInds[temp++] = i * (j + 1);
		}
	}
	int number;//需要选取的所有点数量
	number = (2 * window_size + 1) * 2 * 2;
	Mat col_inds = Mat::zeros((imgSize*number), 1, CV_64F);
	Mat row_inds = Mat::zeros((imgSize*number), 1, CV_64F);
	Mat vals = Mat::zeros(number, 1, CV_64F);
	Mat gvals = Mat::zeros(1, number, CV_64F);

	//==========interaction
	int len = 0, consts_len = 0;
	for (int j = 1; j < m; j++)
	{
		for (int i = 1; i < n; i++)
		{
			int tlen;
			consts_len++;
			if (!colorIm.at<double>(i, j))//如果没被上色，值为0
			{
				tlen = 0;
				for (int x = max(1, i - window_size); x <= min(i + window_size, n); x++)
				{
					for (int y = max(1, j - window_size); y <= min(j + window_size, n); y++)
					{
						if (x != i || y != j)
						{
							len++;
							tlen++;
							//可能会越界，不知是否支持默认为1，后面考虑全部改为数组
							row_inds.at<double>(len) = consts_len;
							col_inds.at<double>(len) = indsM.at<double>(x, y);
							gvals.at<double>(tlen) = ntscIm.at<Vec3f>(x, y)[0];//取灰度值
						}
					}
				}
				double t_val = ntscIm.at<Vec3f>(i, j)[0];
				gvals.at<double>(tlen + 1) = t_val;
				//int c_var = mean((gvals.at<double>))
			}
		}
	}
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