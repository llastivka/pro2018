#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdint.h>
#include <math.h>

using namespace cv;
using namespace std;

struct color
{
	float r, g, b;
};
/*
int main(int argc, char** argv)
{
	
	color blue;
	blue.r = 80;
	blue.g = 213;
	blue.b = 246;

	color yellow = { 253, 243, 0 };
	color pink = { 251, 73, 153 };
	color green = { 25, 182, 79 };

	Mat image = imread("code.png", CV_LOAD_IMAGE_COLOR);
	Mat output;
	color col = blue;
	float diff = 1;
	inRange(image, Scalar(col.b - diff, col.g - diff, col.r - diff), Scalar(col.b + diff, col.g + diff, col.r + diff), output);

	imshow("output", output);
	imwrite("output2.png", output);
	

	Mat image = cv::imread("code.png");
	Mat output;
	
	color blue;
	blue.r = 80;
	blue.g = 213;
	blue.b = 246;

	float diff = 20;
	inRange(image, Scalar(blue.b - diff, blue.g - diff, blue.r - diff), Scalar(blue.b + diff, blue.g + diff, blue.r + diff), output);
	imshow("output", output);

	waitKey();
}
*/