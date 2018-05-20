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

int main(int argc, char** argv)
{
	//decoding !!!
	//Mat imageCode = imread("code.png", CV_LOAD_IMAGE_COLOR);
	//decode(imageCode);
	//cout << checkCorrectness();

	//angle points detection !!!
	Mat src = imread("real.png", 0);

	Mat dst, cdst;
	Canny(src, dst, 50, 200, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);

	vector<Vec2f> lines;
	// detect lines
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 90);

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
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
	}

	imshow("source", src);
	imshow("detected lines", cdst);

	imwrite("lines.png", cdst);

	waitKey();
	return 0;
}