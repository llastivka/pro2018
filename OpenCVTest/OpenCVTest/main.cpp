#include <opencv2/opencv.hpp>
#include <stdint.h>

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
	/*
	Mat channels[3];
	split(module, channels);
	int sumB = 0;
	int sumG = 0;
	int sumR = 0;
	for (int r = 0; r < module.rows; r++)
	{
		for (int c = 0; c < module.cols; c++)
		{
			sumB += module.at<Vec3b>(r, c)[0];
			sumG += module.at<Vec3b>(r, c)[1];
			sumR += module.at<Vec3b>(r, c)[2];
		}
	}
	color average;
	average.b = sumB / N;
	average.g = sumG / N;
	average.r = sumR / N;
	*/
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
	for (int i = 0; i <= size - 1; i++)
	{
		if (array[i] > max)
		{
			max = array[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

void decode(Mat image) {
	split(image);
	
	for (int i = 0; i <4; i++)
	{
		palette[i] = average(*(modules + i));
		messages[i] = codes[i];
		cout << messages[i] << endl;
	}

	int threshold = 1000;
	for (int i = 4; i < 100; i++)
	{
		int nonZeroCounts[C];
		for (int k = 0; k < 4; k++)
		{
			color col = palette[k];
			Mat output;
			inRange(modules[i], Scalar(col.b - 50, col.g - 50, col.r - 50), Scalar(col.b + 50, col.g + 50, col.r + 50), output);
			nonZeroCounts[k] = countNonZero(output);
		}
		int index = indexOfMax(nonZeroCounts, C);
		messages[i] = codes[index];
		cout << messages[i] << endl;
	}

	printMessage();
}

//help method for testing purposes
bool checkCorrectness() {
	String correctResult = "00011011101101001000010011010010010110010110011111100011001110000010010011011100010011100010000010100010110001011100010000011010001100110110011100001001101011001001110011010001101100000110100111000001";
	int index = 0;
	bool result = true;
	for (int i = 0; i < N; i++) 
	{
		if (messages[i][0] != correctResult[index] || messages[i][1] != correctResult[index+1])
		{
			result = false;
			break;
		}
		index += 2;
	}
	return result;
}

int main(int argc, char** argv)
{
	Mat imageCode = imread("code.png", CV_LOAD_IMAGE_COLOR);
	decode(imageCode);
	cout << checkCorrectness();

	imshow("image", imageCode);
	waitKey();
}