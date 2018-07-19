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

Mat threasholdImage(Mat img)
{
	medianBlur(img, img, 5);

	double thres = 240;
	double color = 255;
	threshold(img, img, thres, color, CV_THRESH_BINARY);

	// Execute erosion to improve the detection
	int erosion_size = 4;
	Mat element = getStructuringElement(MORPH_CROSS,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(img, img, element);
	return img;
}

void findAngles(Mat img)
{
	vector<Vec2i> angles = {};

	vector<vector<Point>> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;
	findContours(srcGray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

	vector<double> largestArea = { 0, 0, 0, 0 };
	vector<int> largestContourIndexes = { 0, 0, 0, 0 };
	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		for (int j = 0; j < 4; j++)
		{
			if (a > largestArea[j]) {
				largestArea.insert(largestArea.begin() + j, a);
				//for inserting with shift of other elements, refactor later
				for (int k = largestContourIndexes.size() - 1; k > j; k--)
				{
					largestContourIndexes[k] = largestContourIndexes[k - 1];
				}
				largestContourIndexes[j] = i; //Store the index of largest contour
				break;
			}
		}
	}

	//x and y for each of 5 points (upper-left, upper-right, lower-right, lower-left, centroid) for each of 4 largest conrours
	vector<vector<Vec2i>> countoursImportantPointsCoordinates = {}; //final size should be 4
	Point2f generalUpperLeft = Point2f(src.cols, src.rows);
	Point2f generalUpperRight = Point2f(0, src.rows);
	Point2f generalLowerRight = Point2f(0, 0);
	Point2f generalLowerLeft = Point2f(src.cols, 0);
	for (int i = 0; i < largestContourIndexes.size(); i++)
	{
		int maxX = 0;
		int maxY = 0;
		int minX = src.cols;
		int minY = src.rows;
		
		for (int j = 0; j < contours[largestContourIndexes[i]].size(); j++)
		{
			int x = contours[largestContourIndexes[i]][j].x;
			int y = contours[largestContourIndexes[i]][j].y;
			if (x > maxX)
			{
				maxX = x;
			} 
			if (x < minX)
			{
				minX = x;
			}
			if (y > maxY)
			{
				maxY = y;
			}
			if (y < minY)
			{
				minY = y;
			}
		}
		cout << i << "(" << largestContourIndexes[i] << "): minX = " << minX << ", maxX = " << maxX << ", minY = " << minY << ", maxY = " << maxY << endl;
		Vec2i upperLeft = Vec2i(minX, minY);
		Vec2i upperRight = Vec2i(maxX, minY);
		Vec2i lowerRight = Vec2i(maxX, maxY);
		Vec2i lowerLeft = Vec2i(minX, maxY);
		Vec2i centroid = Vec2i((minX + maxX)/2, (minY + maxY)/2);
		countoursImportantPointsCoordinates.push_back({ upperLeft, upperRight, lowerRight, lowerLeft, centroid });

		if (pow(upperLeft[0] - 0, 2) + pow(upperLeft[1] - 0, 2) < pow(generalUpperLeft.x - 0, 2) + pow(generalUpperLeft.y - 0, 2))
		{
			generalUpperLeft.x = upperLeft[0];
			generalUpperLeft.y = upperLeft[1];
		}
		if (pow(upperRight[0] - src.cols - 1, 2) + pow(upperRight[1] - 0, 2) < pow(generalUpperRight.x - src.cols - 1, 2) + pow(generalUpperRight.y- 0, 2))
		{
			generalUpperRight.x = upperRight[0];
			generalUpperRight.y = upperRight[1];
		}
		if (pow(lowerRight[0] - src.cols - 1, 2) + pow(lowerRight[1] - src.rows - 1, 2) < pow(generalLowerRight.x - src.cols - 1, 2) + pow(generalLowerRight.y - src.rows - 1, 2))
		{
			generalLowerRight.x = lowerRight[0];
			generalLowerRight.y = lowerRight[1];
		}
		if (pow(lowerLeft[0] - 0, 2) + pow(lowerLeft[1] - src.rows - 1, 2) < pow(generalLowerLeft.x - 0, 2) + pow(generalLowerLeft.y - src.rows - 1, 2))
		{
			generalLowerLeft.x = lowerLeft[0];
			generalLowerLeft.y = lowerLeft[1];
		}

		//for testing purposes
		inputQuad[0] = Point2f(generalUpperLeft.x, generalUpperLeft.y);
		inputQuad[1] = Point2f(generalUpperRight.x, generalUpperRight.y);
		inputQuad[2] = Point2f(generalLowerRight.x, generalLowerRight.y);
		inputQuad[3] = Point2f(generalUpperLeft.x, generalLowerLeft.y);
		for (int i = 0; i < 4; i++)
		{
			circle(src, inputQuad[i], 3, Scalar(0, 0, 255), 1, 8, 0);
		}
		imshow("points", src);
		cout << "Press any key" << endl;
		waitKey(0);
	}

	//for now i'll just assign it but i'll have to rewrite it eventually
	inputQuad[0] = Point2f(generalUpperLeft.x, generalUpperLeft.y);
	inputQuad[1] = Point2f(generalUpperRight.x, generalUpperRight.y);
	inputQuad[2] = Point2f(generalLowerRight.x, generalLowerRight.y);
	inputQuad[3] = Point2f(generalUpperLeft.x, generalLowerLeft.y);

}

int main(int argc, char** argv)
{
	src = imread("white.png", CV_LOAD_IMAGE_UNCHANGED);
	srcGray = imread("white.png", CV_LOAD_IMAGE_GRAYSCALE);
	//vector<Vec2f> lines = calculateHoughLines(srcGray);
	//drawLines(srcGray, lines);
	//imshow("with lines", srcGray);
	//imwrite("lines.png", srcGray);
	
	srcGray = threasholdImage(srcGray);
	imshow("white regions", srcGray);
	findAngles(srcGray);

	outputQuad[0] = Point2f(0, 0);
	outputQuad[1] = Point2f(src.cols - 1, 0);
	outputQuad[2] = Point2f(src.cols - 1, src.rows - 1);
	outputQuad[3] = Point2f(0, src.rows - 1);

	Mat output = perspectiveTransform(src);

	imshow("output", output);
	imwrite("output.png", output);
	waitKey();
}