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
const double pi = 3.14159265358979323846;
const int C = 4;
const int N = 100;
Mat modules[N];
color palette[C];
String codes[C] = { "00","01","10","11" };
color paletteAverages[C][N];
int numOfPaletteColors[C] = { 0, 0, 0, 0 };
String messages[N];
vector<Vec2f> lines;
// Input Quadilateral or Image plane coordinates
Point2f inputQuad[4];
// Output Quadilateral or World plane coordinates
Point2f outputQuad[4];
int currentNumOfOddLines = 0;
float currentSumOfAngleCloseness = 0;

void split(Mat image)
{
	Size size(1000, 1000);
	Mat resized;
	resize(image, resized, size);
	Size smallSize(100, 100);
	int counter = 0;
	for (int r = 0; r < resized.rows; r += smallSize.height)
	{
		for (int c = 0; c < resized.cols; c += smallSize.width)
		{
			Rect rect = Rect(c, r, smallSize.width, smallSize.height);
			modules[counter] = Mat(resized, rect);
			counter++;
		}
	}
}

color average(Mat module) {

	Scalar meanVal = mean(module);
	color average;
	average.b = meanVal.val[0];
	average.g = meanVal.val[1];
	average.r = meanVal.val[2];
	return average;
}

void printMessage() {
	ofstream file;
	file.open("example.txt");
	for (int i = 0; i < N; i++) {
		file << messages[i] << endl;
	}
	file.close();
}

int indexOfMax(int array[], int size)
{
	int maxIndex = 0;
	int max = array[0];
	for (int i = 0; i < size; i++)
	{
		if (array[i] > max)
		{
			max = array[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

void averagePaletteColors(int paletteIndex) {
	int sumB = 0;
	int sumG = 0;
	int sumR = 0;
	for (int i = 0; i < numOfPaletteColors[paletteIndex]; i++)
	{
		sumB += paletteAverages[paletteIndex][i].b;
		sumG += paletteAverages[paletteIndex][i].g;
		sumR += paletteAverages[paletteIndex][i].r;
	}
	palette[paletteIndex].b = sumB / numOfPaletteColors[paletteIndex];
	palette[paletteIndex].g = sumG / numOfPaletteColors[paletteIndex];
	palette[paletteIndex].r = sumR / numOfPaletteColors[paletteIndex];
}

void decode(Mat image) {
	split(image);

	for (int i = 0; i <4; i++)
	{
		palette[i] = average(*(modules + i));
		messages[i] = codes[i];
		paletteAverages[i][numOfPaletteColors[i]] = palette[i];
		numOfPaletteColors[i] += 1;
		cout << messages[i] << endl;
	}

	int diff = 40;
	for (int i = 4; i < 100; i++)
	{
		int nonZeroCounts[C];
		for (int k = 0; k < 4; k++)
		{
			color col = palette[k];
			Mat output;
			inRange(modules[i], Scalar(col.b - diff, col.g - diff, col.r - diff), Scalar(col.b + diff, col.g + diff, col.r + diff), output);
			nonZeroCounts[k] = countNonZero(output);
		}
		int index = indexOfMax(nonZeroCounts, C);
		messages[i] = codes[index];
		cout << messages[i] << endl;
		cv::imwrite(to_string(i + 1) + ".png", modules[i]);

		color currentAverage = average(modules[i]);
		paletteAverages[index][numOfPaletteColors[index]] = currentAverage;
		numOfPaletteColors[index] += 1;

		averagePaletteColors(index);
	}

	printMessage();
}

//help method for testing purposes
bool checkCorrectness() {
	String correctResult = "00011011101101001000010011010010010110010110011111100011001110000010010011011100010011100010000010100010110001011100010000011010001100110110011100001001101011001001110011010001101100000110100111000001";
	int index = 0;
	bool result = true;
	int numOfFail = 0;
	for (int i = 0; i < N; i++)
	{
		if (messages[i][0] != correctResult[index] || messages[i][1] != correctResult[index + 1])
		{
			result = false;
			numOfFail++;
			if (messages[i][0] != correctResult[index])
			{
				cout << index << endl;
			}
			if (messages[i][1] != correctResult[index + 1])
			{
				cout << index + 1 << endl;
			}
			//break;
		}
		index += 2;
	}
	numOfFail;
	cout << "Fails: " << numOfFail << endl;
	return result;
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
	Point2f &r)
{
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

bool lessLinesLeftToChange(vector<Vec2f> lines)
{
	int count = 0;
	float permittedShift = 0.01; //in radians
	for (int i = 0; i < lines.size(); i++) {
		float theta = lines[i][1];
		if (!(theta > 0 - permittedShift && theta < 0 + permittedShift) &&
			!(theta > 90 * pi / 180 - permittedShift && theta < 90 * pi / 180 - permittedShift) &&
			!(theta > 180 * pi / 180 - permittedShift && theta < 180 * pi / 180 - permittedShift) &&
			!(theta > 270 * pi / 180 - permittedShift && theta < 270 * pi / 180 - permittedShift))
		{
			count++;
		}
	}
	bool less = count < currentNumOfOddLines;
	if (less)
	{
		currentNumOfOddLines = count;
	}
	return less;
}

float calculateClosenessForHorizontal(vector<Vec2f> lines) {
	float sum = 0;
	for (int i = 0; i < lines.size(); i++) {
		float theta = lines[i][1];
		while (theta > 2 * pi)
		{
			theta -= 2 * pi;
		}
		while (theta < 0)
		{
			theta += 2 * pi;
		}
		if (theta >= 3 * pi / 4 && theta < 5 * pi / 4)
		{
			sum += abs(theta - pi);
		}
		else if (theta >= 7 * pi / 4 || theta < pi / 4)
		{
			sum += abs(theta);
		}
	}
	return sum;
}

float calculateClosenessForVertical(vector<Vec2f> lines)
{
	float sum = 0;
	for (int i = 0; i < lines.size(); i++) {
		float theta = lines[i][1];
		while (theta > 2 * pi)
		{
			theta -= 2 * pi;
		}
		while (theta < 0)
		{
			theta += 2 * pi;
		}
		if (theta >= pi / 4 && theta < 3 * pi / 4)
		{
			sum += abs(theta - pi / 2);
		}
		else if (theta >= 5 * pi / 4 && theta < 7 * pi / 4)
		{
			sum += abs(theta - 3 * pi / 2);
		}
	}
	return sum;
}

float sumOfClosenessToIdealAngle(vector<Vec2f> lines)
{
	float sum = 0;
	for (int i = 0; i < lines.size(); i++) {
		float theta = lines[i][1];
		while (theta > 2 * pi)
		{
			theta = theta - 2 * pi;
		}
		while (theta < 0)
		{
			theta += 2 * pi;
		}
		if (theta >= pi / 4 && theta < 3 * pi / 4)
		{
			sum += abs(theta - pi / 2);
		}
		else if (theta >= 3 * pi / 4 && theta < 5 * pi / 4)
		{
			sum += abs(theta - pi);
		}
		else if (theta >= 5 * pi / 4 && theta < 7 * pi / 4)
		{
			sum += abs(theta - 3 * pi / 2);
		}
		else
		{
			sum += abs(theta);
		}
	}
	return sum;
}

Mat perspectiveTransform(Mat input)
{
	// Lambda Matrix
	Mat lambda(2, 4, CV_32FC1);
	//Input and Output Image;
	Mat output;

	// Set the lambda matrix the same type and size as input
	lambda = Mat::zeros(input.rows, input.cols, input.type());

	/*
	// The 4 points that select quadilateral on the input , from top-left in clockwise order
	// These four pts are the sides of the rect box used as input
	inputQuad[0] = Point2f(-30, -60);
	inputQuad[1] = Point2f(input.cols + 50, -50);
	inputQuad[2] = Point2f(input.cols + 100, input.rows + 50);
	inputQuad[3] = Point2f(-50, input.rows + 50);
	// The 4 points where the mapping is to be done , from top-left in clockwise order
	outputQuad[0] = Point2f(0, 0);
	outputQuad[1] = Point2f(input.cols - 1, 0);
	outputQuad[2] = Point2f(input.cols - 1, input.rows - 1);
	outputQuad[3] = Point2f(0, input.rows - 1);
	*/

	// Get the Perspective Transform Matrix i.e. lambda 
	lambda = getPerspectiveTransform(inputQuad, outputQuad);
	// Apply the Perspective Transform just found to the src image
	warpPerspective(input, output, lambda, output.size());

	//Display input and output
	//imshow("Input", input);
	//imshow("Output", output);

	return output;
}

vector<Vec2f> calculateHoughLines(Mat src)
{
	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	vector<Vec2f> result;
	HoughLines(dst, result, 1, CV_PI / 180, 150, 0, 90);
	return result;
}

void detectAnglePoints(Mat src) {
	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	// detect lines
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 90);
	int linesSize = lines.size();
	cout << linesSize << endl;
	vector<Point> linePoints1;
	vector<Point> linePoints2;
	// draw lines
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
		linePoints1.push_back(pt1);
		linePoints2.push_back(pt2);
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}

	//intersection experimenting
	/*
	vector<Point2f> intersectionPoints;
	for (int i = 0; i < linePoints1.size(); i++)
	{
	for (int j = i + 1; j < linePoints2.size(); j++)
	{
	Point2f intersectionPoint;
	intersection(linePoints1[i], linePoints2[i], linePoints1[j], linePoints2[j], intersectionPoint);
	int thickness = -1;
	int lineType = 8;
	circle(cdst, intersectionPoint, 2.0, Scalar(0, 255, 0), thickness, lineType);
	intersectionPoints.push_back(intersectionPoint);
	}
	}

	vector<float> distances;
	float distanceSum = 0;
	for (int i = 0; i < intersectionPoints.size(); i++)
	{
	for (int j = i + 1; j < intersectionPoints.size(); j++)
	{
	Point diff = intersectionPoints[i] - intersectionPoints[j];
	distanceSum += sqrt(diff.x*diff.x + diff.y*diff.y);
	}
	}
	float averageDistance = distanceSum / intersectionPoints.size();
	cout << distanceSum << endl;
	cout << averageDistance << endl;
	*/


	cv::imshow("source", src);
	cv::imshow("detected lines", cdst);

	cv::imwrite("lines.png", cdst);
}

/*int main(int argc, char** argv)
{
	//decoding !!!
	//Mat imageCode = imread("code.png", CV_LOAD_IMAGE_COLOR);
	//decode(imageCode);
	//cout << checkCorrectness();

	Mat src = imread("ideal code.png", 1);
	//detectAnglePoints(src);


	lines = calculateHoughLines(src);
	currentNumOfOddLines = lines.size();
	int initialPointsCoordinates[4][2] = { { 0, 0 },{ src.cols - 1, 0 },{ src.cols - 1, src.rows - 1 },{ 0, src.rows - 1 } };
	inputQuad[0] = Point2f(initialPointsCoordinates[0][0], initialPointsCoordinates[0][1]);
	inputQuad[1] = Point2f(initialPointsCoordinates[1][0], initialPointsCoordinates[1][1]);
	inputQuad[2] = Point2f(initialPointsCoordinates[2][0], initialPointsCoordinates[2][1]);
	inputQuad[3] = Point2f(initialPointsCoordinates[3][0], initialPointsCoordinates[3][1]);

	float thr = lines.size() * 0.01;
	int counterFor8DifferentTypesOfChange = 0;

	//current min stuff
	float currentMinSumOfCloseness = sumOfClosenessToIdealAngle(lines);
	float currentMinHorizontal = calculateClosenessForHorizontal(lines);
	float currentMinVertical = calculateClosenessForVertical(lines);
	Mat currentMinTransformed = src;
	vector<Vec2f> currentMinLines = lines;

	int shift = 1;

	int counterForTesting = 0;
	while (counterFor8DifferentTypesOfChange >= 0 && counterFor8DifferentTypesOfChange <12 && sumOfClosenessToIdealAngle(lines)>0) {
		counterForTesting++;
		cout << counterForTesting << endl;

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

		cout << "Current min sum of angle closeness: " << currentMinSumOfCloseness << endl;
		cout << "Perspective transform start! " << endl;
		Mat transformed = perspectiveTransform(src);
		cout << "Perspective transform end! " << endl;
		cout << "Hough lines start!" << endl;
		vector<Vec2f> newLines = calculateHoughLines(transformed);
		cout << "Hough lines end! " << endl;
		cout << "Sum of closeness calculation start! " << endl;
		float newSumOfCloseness = sumOfClosenessToIdealAngle(newLines);
		cout << "Sum of closeness calculation end! " << endl;
		float newHorizontal = calculateClosenessForHorizontal(newLines);
		float newVertical = calculateClosenessForVertical(newLines);
		cout << "New current sum of angle closeness: " << newSumOfCloseness << endl;
		cout << "New current horizontal closeness: " << newHorizontal << endl;
		cout << "New current vertical closeness: " << newVertical << endl;
		//imshow("mid", transformed);
		//waitKey();

		shift++;
		if (newSumOfCloseness - currentMinSumOfCloseness < 0.5) {
			//if ((newHorizontal - currentMinHorizontal < 0.2) ||	(newVertical - currentMinVertical < 0.2)) {
			if (newSumOfCloseness < currentMinSumOfCloseness) {
				currentMinSumOfCloseness = newSumOfCloseness;
				currentMinTransformed = transformed;
			}
			*/
			/*
			if (newHorizontal < currentMinHorizontal) {
			currentMinHorizontal = newHorizontal;
			currentMinTransformed = transformed;
			}
			if (newVertical < currentMinVertical) {
			currentMinVertical = newVertical;
			currentMinTransformed = transformed;
			}
			*/
/*
			cv::imwrite("transformed.png", transformed);
		}
		else
		{
			counterFor8DifferentTypesOfChange++;
			cout << "Type of change: " << counterFor8DifferentTypesOfChange << endl;
			shift = 1;
			src = currentMinTransformed;

			//for testing purposes
			//imshow("mid transformed", currentMinTransformed);
			//if (counterFor8DifferentTypesOfChange > 0) {
			//	waitKey();
			//}
		}
	}

	imwrite("transformed.png", src);
	imshow("output", src);

	cv::waitKey();
	return 0;
}
*/