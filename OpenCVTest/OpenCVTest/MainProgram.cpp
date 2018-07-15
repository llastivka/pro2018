#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdint.h>
#include <math.h>

using namespace cv;
using namespace std;

Mat src;
Mat srcGray;
Point2f inputQuad[4];
Point2f outputQuad[4];

Mat perspectiveTransform(Mat input)
{
	Mat lambda(2, 4, CV_32FC1);
	Mat output;
	lambda = Mat::zeros(input.rows, input.cols, input.type());
	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	warpPerspective(input, output, lambda, output.size());
	return output;
}

vector<Vec2f> calculateHoughLines(Mat src)
{
	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec2f> result;
	HoughLines(dst, result, 1, CV_PI / 180, 150, 0, 0);
	return result;
}

void drawLines(Mat img, vector<Vec2f> lines) {
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(img, pt1, pt2, Scalar(0, 0, 0), 3, CV_AA);
	}
}

vector<Vec2i> regionForPixel(Mat img, int row, int column)
{
	vector<Vec2i> result = {};
	return result;
}

vector<vector<Vec2i>> detectLightRegions(Mat img) {
	vector<vector<Vec2i>> result = {};
	Vec3b threshold = Vec3b(25, 25, 25);
	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			if (img.at<Vec3b>(r, c)[0] < threshold[0] && img.at<Vec3b>(r, c)[1] < threshold[1] && img.at<Vec3b>(r, c)[2] < threshold[2])
			{
				result.push_back(regionForPixel(img, r, c));
			}
		}
	}
	return result;
}

int main(int argc, char** argv)
{
	src = imread("white.png", CV_LOAD_IMAGE_UNCHANGED);
	srcGray = imread("white.png", CV_LOAD_IMAGE_GRAYSCALE);
	vector<Vec2f> lines = calculateHoughLines(srcGray);
	medianBlur(srcGray, srcGray, 5);
	//drawLines(srcGray, lines);
	imshow("with lines", srcGray);
	imwrite("lines.png", srcGray);

	double thres = 220;
	double color = 255;
	threshold(srcGray, srcGray, thres, color, CV_THRESH_BINARY);

	// Execute erosion to improve the detection
	int erosion_size = 4;
	Mat element = getStructuringElement(MORPH_CROSS,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(srcGray, srcGray, element);
	imshow("output", srcGray);

	//vector<vector<Vec2i>> lightRegions = detectLightRegions(srcGray);

	//imshow("output", src);
	waitKey();
}