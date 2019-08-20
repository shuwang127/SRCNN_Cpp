/* Program  : Image Super-Resolution using deep Convolutional Neural Networks
 * Author   : Wang Shu
 * Date     : Sun 13 Sep, 2015
 * Descrip. :
* */

#include <iostream>
#include <iomanip>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/* Marco Definition */
#define UP_SCALE        2       // the scale of super-resolution
#define CONV1_FILTERS   64      // the first convolutional layer
#define CONV2_FILTERS   32      // the second convolutional layer
#define INFO            1       // show the debug info
//#define DEBUG           1       // show the debug info
//#define DISPLAY         1       // display the temp image
//#define SAVE            1       // save the temp image
#define CUBIC           1

/* Load the convolutional data */
#include "convdata.h"

void UsageShow(char* pname)
{
    cout << "Usage: " << pname << " image_file [image_output]" << endl;
}

static int IntTrim(int a, int b, int c)
{
    int buff[3] = {a, c, b};
    return buff[ (int)(c > a) + (int)(c > b) ];
}

/***
 * FuncName : Convolution99
 * Function : Complete one cell in the first Convolutional Layer
 * Parameter    : src - the original input image
 *        dst - the output image
 *        kernel - the convolutional kernel
 *        bias - the cell bias
 * Output   : <void>
***/
void Convolution99(Mat& src, Mat& dst, float kernel[9][9], float bias)
{
    int width, height, row, col, i, j;
    float temp;
    height = dst.rows;
    width = dst.cols;
    int rowf[height + 8], colf[width + 8];

    /* Expand the src image */
    for (row = 0; row < height + 8; row++)
    {
        rowf[row] = IntTrim(0, height - 1, row - 4);
    }
    for (col = 0; col < width + 8; col++)
    {
        colf[col] = IntTrim(0, width - 1, col - 4);
    }

    /* Complete the Convolution Step */
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            /* Convolution */
            temp = 0;

            for (i = 0; i < 9; i++)
            {
                for (j = 0; j < 9; j++)
                {
                    temp += kernel[i][j] * src.at<unsigned char>(rowf[row + i], colf[col + j]);
                }
            }

            temp += bias;

            /* Threshold */
            temp = (temp < 0) ? 0 : temp;

            dst.at<float>(row, col) = temp;
        }
    }

    return;
}

/***
 * FuncName : Convolution11
 * Function : Complete one cell in the second Convolutional Layer
 * Parameter    : src - the first layer data
 *        dst - the output data
 *        kernel - the convolutional kernel
 *        bias - the cell bias
 * Output   : <void>
***/
void Convolution11(vector<Mat>& src, Mat& dst, float kernel[CONV1_FILTERS], float bias)
{
    int width, height, row, col, i;
    float temp;
    height = dst.rows;
    width = dst.cols;

    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            /* Process with each pixel */
            temp = 0;

            for (i = 0; i < CONV1_FILTERS; i++)
            {
                temp += src[i].at<float>(row, col) * kernel[i];
            }
            temp += bias;

            /* Threshold */
            temp = (temp < 0) ? 0 : temp;

            dst.at<float>(row, col) = temp;
        }
    }

    return;
}

/***
 * FuncName : Convolution55
 * Function : Complete the cell in the third Convolutional Layer
 * Parameter    : src - the second layer data
 *        dst - the output image
 *        kernel - the convolutional kernel
 *        bias - the cell bias
 * Output   : <void>
***/
void Convolution55(vector<Mat>& src, Mat& dst, float kernel[32][5][5], float bias)
{
    int width, height, row, col, i, m, n;
    float temp;
    double temppixel;
    height = dst.rows;
    width = dst.cols;
    int rowf[height + 4], colf[width + 4];

    /* Expand the src image */
    for (row = 0; row < height + 4; row++)
    {
        rowf[row] = IntTrim(0, height - 1, row - 2);
    }
    for (col = 0; col < width + 4; col++)
    {
        colf[col] = IntTrim(0, width - 1, col - 2);
    }

    /* Complete the Convolution Step */
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            temp = 0;

            for (i = 0; i < CONV2_FILTERS; i++)
            {
                temppixel = 0;
                for (m = 0; m < 5; m++)
                {
                    for (n = 0; n < 5; n++)
                    {
                        temppixel += \
                        kernel[i][m][n] * src[i].at<float>(rowf[row + m], colf[col + n]);
                    }
                }

                temp += temppixel;
            }

            temp += bias;

            /* Threshold */
            temp = IntTrim(0, 255, temp);

            dst.at<unsigned char>(row, col) = (unsigned char)temp;
        }
#ifdef DEBUG
        cout << "Convolutional Layer III : " << setw(4) << row + 1 << '/' << dst.rows << " Complete ..." << endl;
#endif
    }

    return;
}

/***
 * FuncName : main
 * Function : the entry of the program
 * Parameter    : argc - the number of the initial parameters
 *        argv - the entity of the initial parameters
 * Output   : int 0 for normal / int 1 for failed
***/
int main(int argc, char** argv)
{
    if (argc > 1)
    {
        /* Read the original image */
        Mat pImgOrigin;
        //pImgOrigin = imread("Pictures/butterfly_GT.bmp");
        pImgOrigin = imread(argv[1]);
#ifdef INFO
        cout << "Read the Original Image Successfully ..." << endl;
#endif
#ifdef SAVE
        imwrite("Result/Origin.png", pImgOrigin);
#endif

        /* Convert the image from BGR to YCrCb Space */
        Mat pImgYCrCb;
        cvtColor(pImgOrigin, pImgYCrCb, CV_BGR2YCrCb);
#ifdef INFO
        cout << "Convert the Image to YCrCb Sucessfully ..." << endl;
#endif

        /* Split the Y-Cr-Cb channel */
        vector<Mat> pImgYCrCbCh(3);
        split(pImgYCrCb, pImgYCrCbCh);
#ifdef DISPLAY
        imshow("Luma", pImgYCrCbCh[0]);
#endif
#ifdef SAVE
        imwrite("Result/Luma.png", pImgYCrCbCh[0]);
#endif
#ifdef INFO
        cout << "Spliting the Y-Cr-Cb Channel ..." << endl;
#endif

        /* Resize the Y-Cr-Cb Channel with Bicubic Interpolation */
        vector<Mat> pImg(3);
        for (int i = 0; i < 3; i++)
        {
            resize(pImgYCrCbCh[i], pImg[i], pImgYCrCbCh[i].size()*UP_SCALE, 0, 0, CV_INTER_CUBIC);
        }
#ifdef DISPLAY
        imshow("LumaCubic", pImg[0]);
#endif
#ifdef SAVE
        imwrite("Result/LumaCubic.png", pImg[0]);
#endif
#ifdef INFO
        cout << "Completed Bicubic Interpolation ..." << endl;
#endif

#ifdef CUBIC
        /* Output the Cubic Inter Result (Optional) */
        Mat pImgCubic;
        resize(pImgYCrCb, pImgCubic, pImgYCrCb.size()*UP_SCALE, 0, 0, CV_INTER_CUBIC);
        cvtColor(pImgCubic, pImgCubic, CV_YCrCb2BGR);
#ifdef DISPLAY
        imshow("Cubic", pImgCubic);
#endif
#ifdef SAVE
        imwrite("Result/Cubic.png", pImgCubic);
#endif
#ifdef DEBUG
        cout << "Completed Bicubic Version (Optional) ..." << endl;
#endif
#endif

        /******************* The First Layer *******************/
        vector<Mat> pImgConv1(CONV1_FILTERS);
        for (int i = 0; i < CONV1_FILTERS; i++)
        {
            pImgConv1[i].create(pImg[0].size(), CV_32F);
            Convolution99(pImg[0], pImgConv1[i], weights_conv1_data[i], biases_conv1[i]);
#ifdef DEBUG
            cout << "Convolutional Layer I : " << setw(2) << i + 1 << "/64 Cell Completed ..." << endl;
#endif
        }
#ifdef DISPLAY
        imshow("Conv1", pImgConv1[8]);
#endif
#ifdef SAVE
        imwrite("Result/Conv1.png", pImgConv1[8]);
#endif
#ifdef INFO
        cout << "Convolutional Layer I : 100% Complete ..." << endl;
#endif

        /******************* The Second Layer *******************/
        vector<Mat> pImgConv2(CONV2_FILTERS);
        for (int i = 0; i < CONV2_FILTERS; i++)
        {
            pImgConv2[i].create(pImg[0].size(), CV_32F);
            Convolution11(pImgConv1, pImgConv2[i], weights_conv2_data[i], biases_conv2[i]);
#ifdef DEBUG
            cout << "Convolutional Layer II : " << setw(2) << i + 1 << "/32 Cell Complete..." << endl;
#endif
    }
#ifdef DISPLAY
        imshow("Conv2", pImgConv2[31]);
#endif
#ifdef SAVE
        imwrite("Result/Conv2.png", pImgConv2[31]);
#endif
#ifdef INFO
        cout << "Convolutional Layer II : 100% Complete ..." << endl;
#endif

        /******************* The Third Layer *******************/
        Mat pImgConv3;
        pImgConv3.create(pImg[0].size(), CV_8U);
        Convolution55(pImgConv2, pImgConv3, weights_conv3_data, biases_conv3);
#ifdef DISPLAY
        imshow("Conv3", pImgConv3);
#endif
#ifdef SAVE
        imwrite("Result/Conv3.png", pImgConv3);
#endif
#ifdef INFO
        cout << "Convolutional Layer III : 100% Complete ..." << endl;
#endif

        /* Merge Ycinv and Cr-Cb Channel*/
        pImg[0] = pImgConv3;

        /* Merge the Y-Cr-Cb Channel into an image */
        Mat pImgYCrCbOut;
        merge(pImg, pImgYCrCbOut);
#ifdef INFO
        cout << "Merge Image Complete..." << endl;
#endif

        /* Convert the image from YCrCb to BGR Space */
        Mat pImgBGROut;
        cvtColor(pImgYCrCbOut, pImgBGROut, CV_YCrCb2BGR);
#ifdef DISPLAY
        imshow("Output", pImgBGROut);
#endif
//#ifdef SAVE
        if (argc > 2)
        {
            imwrite(argv[2], pImgBGROut);
        } else {
            imwrite("srcnn_output.png", pImgBGROut);
        }

//#endif
#ifdef INFO
        cout << "Convert the Image to BGR Sucessfully ..." << endl;
#endif

        cvWaitKey();

        return 0;
    } else {
        UsageShow(argv[0]);
    }
}
