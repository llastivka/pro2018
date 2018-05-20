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
const int C = 4;
const int N = 100;
Mat modules[N];
color palette[C];
String codes[C] = { "00","01","10","11" };
color paletteAverages[C][N];
int numOfPaletteColors[C] = {0, 0, 0, 0};
String messages[N];

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
		imwrite(to_string(i+1) + ".png", modules[i]);

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
		if (messages[i][0] != correctResult[index] || messages[i][1] != correctResult[index+1])
		{
			result = false;
			numOfFail++;
			if (messages[i][0] != correctResult[index])
			{
				cout << index << endl;
			}
			if (messages[i][1] != correctResult[index+1])
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

void detectAnglePoints(Mat src) {
	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	vector<Vec2f> lines;
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
	
	imshow("source", src);
	imshow("detected lines", cdst);

	imwrite("lines.png", cdst);
}

int main(int argc, char** argv)
{
	//decoding !!!
	//Mat imageCode = imread("code.png", CV_LOAD_IMAGE_COLOR);
	//decode(imageCode);
	//cout << checkCorrectness();

	//angle points detection !!!
	Mat src = imread("real.png", 0);
	detectAnglePoints(src);

	waitKey();
	return 0;
}