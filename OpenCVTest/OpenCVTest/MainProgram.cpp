#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdint.h>
#include <math.h>

using namespace cv;
using namespace std;

Mat src;
Mat linesImg;
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

Mat drawLines(vector<Vec2f> lines) {
	Mat linesImg = Mat::zeros(src.rows, src.cols, CV_LOAD_IMAGE_GRAYSCALE);
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
		line(linesImg, pt1, pt2, Scalar(255, 255, 255), 1, CV_AA);
	}
	return linesImg;
}

//TODO will have to think about it more bacause if the image size is smaller it probably will affect this function.
//Probably will have to make max rhoDiff and max thetaDiff dependent on the size of initial picture
vector<Vec2f> unifyCloseLines(vector<Vec2f> lines)
{
	vector<Vec2f> modified = {};
	for (int i = 0; i < lines.size(); i++)
	{
		if (!(lines[i][0] == 0 && lines[i][1] == 0))
		{
			vector<int> closeList = {};
			closeList.push_back(i);
			float rho1 = lines[i][0], theta1 = lines[i][1];
			for (int j = i + 1; j < lines.size(); j++)
			{
				if (!(lines[j][0] == 0 && lines[j][1] == 0))
				{
					float rho2 = lines[j][0], theta2 = lines[j][1];
					float rhoDiff = abs(rho1 - rho2);
					float thetaDiff = abs(theta1 - theta2);
					if (rhoDiff < 20 && thetaDiff < 2 * CV_PI * 0.2) //rhoDiff and thetaDiff here
					{
						closeList.push_back(j);
					}
				}
			}

			//for testing purposes (to see close lines visually)
			//vector<Vec2f> expLines = {};
			//for (int k = 0; k < closeList.size(); k++)
			//{
			//	expLines.push_back(lines[closeList.at(k)]);
			//}
			//Mat exp = drawLines(expLines);
			//imshow("mid", exp);
			//waitKey(0);

			float rhoSum = 0;
			float thetaSum = 0;
			for (int j = 0; j < closeList.size(); j++)
			{
				rhoSum += lines[closeList.at(j)][0];
				thetaSum += lines[closeList.at(j)][1];
				lines[j][0] = 0;
				lines[j][1] = 0;
			}
			Vec2f* averageLine = new Vec2f(rhoSum / closeList.size(), thetaSum / closeList.size());
			modified.push_back(*averageLine);
		}
	}
	return modified;
}

float calculateSumOfCloseness(vector<Vec2f> lines)
{
	float sum = 0;
	for (int i = 0; i < lines.size(); i++) {
		float theta = lines[i][1];
		while (theta > 2 * CV_PI)
		{
			theta = theta - 2 * CV_PI;
		}
		while (theta < 0)
		{
			theta += 2 * CV_PI;
		}
		if (theta >= CV_PI / 4 && theta < 3 * CV_PI / 4)
		{
			sum += abs(theta - CV_PI / 2);
		}
		else if (theta >= 3 * CV_PI / 4 && theta < 5 * CV_PI / 4)
		{
			sum += abs(theta - CV_PI);
		}
		else if (theta >= 5 * CV_PI / 4 && theta < 7 * CV_PI / 4)
		{
			sum += abs(theta - 3 * CV_PI / 2);
		}
		else
		{
			sum += abs(theta);
		}
	}
	return sum;
}

void transform(vector<Vec2f> lines)
{
	float initialPointsCoordinates[4][2] = { { 0, 0 },{ src.cols - 1, 0 },{ src.cols - 1, src.rows - 1 },{ 0, src.rows - 1 } };
	inputQuad[0] = Point2f(initialPointsCoordinates[0][0], initialPointsCoordinates[0][1]);
	inputQuad[1] = Point2f(initialPointsCoordinates[1][0], initialPointsCoordinates[1][1]);
	inputQuad[2] = Point2f(initialPointsCoordinates[2][0], initialPointsCoordinates[2][1]);
	inputQuad[3] = Point2f(initialPointsCoordinates[3][0], initialPointsCoordinates[3][1]);


	Mat currentMinTransformed = linesImg;
	vector<Vec2f> currentMinLines = lines;
	float currentMinSumOfCloseness = calculateSumOfCloseness(lines);
	int shift = 1;
	int counterFor8DifferentTypesOfChange = 0;

	while (counterFor8DifferentTypesOfChange >= 0 && counterFor8DifferentTypesOfChange < 12 && calculateSumOfCloseness(lines) > 0) {
		outputQuad[0] = inputQuad[0];
		outputQuad[1] = inputQuad[1];
		outputQuad[2] = inputQuad[2];
		outputQuad[3] = inputQuad[3];

		switch (counterFor8DifferentTypesOfChange) {
		case 0: outputQuad[0] = Point2f(inputQuad[0].x + shift, inputQuad[0].y);
			break;
		case 1: outputQuad[0] = Point2f(inputQuad[0].x, inputQuad[0].y + shift);
			break;
		case 2: outputQuad[1] = Point2f(inputQuad[1].x - shift, inputQuad[1].y);
			break;
		case 3: outputQuad[1] = Point2f(inputQuad[1].x, inputQuad[1].y + shift);
			break;
		case 4: outputQuad[2] = Point2f(inputQuad[2].x - shift, inputQuad[2].y);
			break;
		case 5: outputQuad[2] = Point2f(inputQuad[2].x, inputQuad[2].y - shift);
			break;
		case 6: outputQuad[3] = Point2f(inputQuad[3].x + shift, inputQuad[3].y);
			break;
		case 7: outputQuad[3] = Point2f(inputQuad[3].x, inputQuad[3].y - shift);
			break;
		case 8: outputQuad[0] = Point2f(inputQuad[0].x + shift, inputQuad[0].y);
			outputQuad[1] = Point2f(inputQuad[1].x - shift, inputQuad[1].y);
			break;
		case 9: outputQuad[1] = Point2f(inputQuad[1].x, inputQuad[1].y + shift);
			outputQuad[2] = Point2f(inputQuad[2].x, inputQuad[2].y - shift);
			break;
		case 10: outputQuad[2] = Point2f(inputQuad[2].x - shift, inputQuad[2].y);
			outputQuad[3] = Point2f(inputQuad[3].x + shift, inputQuad[3].y);
			break;
		case 11: outputQuad[0] = Point2f(inputQuad[0].x, inputQuad[0].y + shift);
			outputQuad[3] = Point2f(inputQuad[3].x, inputQuad[3].y - shift);
			break;
		default:
			break;
		}

		cout << "Current MIN sum of angle closeness: " << currentMinSumOfCloseness << endl;
		Mat transformed = perspectiveTransform(linesImg);
		vector<Vec2f> newLines = calculateHoughLines(transformed);
		float newSumOfCloseness = calculateSumOfCloseness(newLines);
		cout << "New current sum of angle closeness: " << newSumOfCloseness << endl;
		//for testing purposes
		imshow("mid", transformed);
		waitKey();

		shift++;
		if (newSumOfCloseness - currentMinSumOfCloseness < 0.5) {
			if (newSumOfCloseness < currentMinSumOfCloseness) {
				currentMinSumOfCloseness = newSumOfCloseness;
				currentMinTransformed = transformed;
			}
			//for testing purposes
			imwrite("transformed.png", transformed);
		}
		else
		{
			counterFor8DifferentTypesOfChange++;
			cout << "!!! New type of change: " << counterFor8DifferentTypesOfChange << endl;
			shift = 1;
			linesImg = currentMinTransformed;
		}
	}
}

int main(int argc, char** argv)
{
	src = imread("real.png", 1);
	imshow("source", src);
	vector<Vec2f> lines = calculateHoughLines(src);
	linesImg = drawLines(lines);
	imwrite("lines.png", linesImg);
	vector<Vec2f> lessLines = unifyCloseLines(lines);
	Mat linesImg1 = drawLines(lessLines);
	imwrite("lines1.png", linesImg1);
	transform(lessLines);

	imshow("output", linesImg);

	waitKey();
}