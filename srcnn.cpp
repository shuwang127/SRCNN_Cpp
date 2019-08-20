/* Program  : Image Super-Resolution using deep Convolutional Neural Networks
 * Author   : Wang Shu
 * Date     : Sun 13 Sep, 2015
 * Descrip. :
* */

#include <iostream>
#include <iomanip>
#include <cstdlib>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

/* Marco Definition */
#define UP_SCALE        2       // the scale of super-resolution
#define INFO            1       // show the debug info
//#define DEBUG           1       // show the debug info
//#define DISPLAY         1       // display the temp image
//#define SAVE            1       // save the temp image
#define CUBIC           1

/* Load the convolutional data */
#include "convdata.h"

float partCNN = 0.5;

void UsageShow(char* pname)
{
    cout << "Usage: " << pname << " image_file [image_output] [partCNN=0.5 {0.0-1.0}]" << endl;
}

static int IntTrim(int a, int b, int c)
{
    int buff[3] = {a, c, b};
    return buff[ (int)(c > a) + (int)(c > b) ];
}

/***
 * FuncName : Convolution99x11
 * Function : Complete one cell in the first and second Convolutional Layer
 * Parameter    : src - the original input image
 *        dst - the output image
 *        kernel - the convolutional kernel
 *        bias - the cell bias
 * Output   : <void>
***/
void Convolution99x11(Mat& src, vector<Mat>& dst, float kernel99[CONV1_FILTERS][9][9], float bias99[CONV1_FILTERS], float kernel11[CONV2_FILTERS][CONV1_FILTERS], float bias11[CONV2_FILTERS])
{
    int width, height, row, col, i, j, k;
    float temp[CONV1_FILTERS], result;
    height = src.rows;
    width = src.cols;
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
            for (k = 0; k < CONV1_FILTERS; k++)
            {
                /* Convolution */
                temp[k] = 0.0;

                for (i = 0; i < 9; i++)
                {
                    for (j = 0; j < 9; j++)
                    {
                        temp[k] += kernel99[k][i][j] * src.at<unsigned char>(rowf[row + i], colf[col + j]);
                    }
                }

                temp[k] += bias99[k];

                /* Threshold */
                temp[k] = (temp[k] < 0) ? 0 : temp[k];
            }

            /* Process with each pixel */
            for (k = 0; k < CONV2_FILTERS; k++)
            {
                result = 0.0;

                for (i = 0; i < CONV1_FILTERS; i++)
                {
                    result += temp[i] * kernel11[k][i];
                }
                result += bias11[k];

                /* Threshold */
                result = (result < 0) ? 0 : result;

                dst[k].at<float>(row, col) = result;
            }
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
            temp = IntTrim(0, 255, (int)(temp + 0.5));

            dst.at<unsigned char>(row, col) = (unsigned char)temp;
        }
#ifdef DEBUG
        cout << "Convolutional Layer III : " << setw(4) << row + 1 << '/' << dst.rows << " Complete ..." << endl;
#endif
    }

    return;
}

void ConvolutionA(Mat& src, Mat& dst, float part)
{
    int width, height, row, col;
    float cnn, cub;
    height = src.rows;
    width = src.cols;

    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            /* Process with each pixel */
            cnn = src.at<unsigned char>(row, col);
            cub = dst.at<unsigned char>(row, col);
            cnn *= part;
            cnn += (1.0 - part) * cub;

            /* Threshold */
            cnn = IntTrim(0, 255, (int)(cnn + 0.5));

            dst.at<unsigned char>(row, col) = (unsigned char)cnn;
        }
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
        cout << "Scale : " << UP_SCALE << endl;
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
        vector<Mat> pImgConv2(CONV2_FILTERS);
        for (int i = 0; i < CONV2_FILTERS; i++)
        {
            pImgConv2[i].create(pImg[0].size(), CV_32F);
        }
        Convolution99x11(pImg[0], pImgConv2, weights_conv1_data, biases_conv1, weights_conv2_data, biases_conv2);
#ifdef DEBUG
        cout << "Convolutional Layer I and II: " << setw(2) << i + 1 << "/64x32 Cell Completed ..." << endl;
#endif
#ifdef DISPLAY
        imshow("Conv1x2", pImgConv2[31]);
#endif
#ifdef SAVE
        imwrite("Result/Conv1x2.png", pImgConv2[31]);
#endif
#ifdef INFO
        cout << "Convolutional Layer I and II : 100% Complete ..." << endl;
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

        /* Merge Yconv and Cr-Cb Channel*/
        if (argc > 3)
        {
            partCNN = atof(argv[3]);
            partCNN = (partCNN < 0.0) ? 0 : ((partCNN > 1.0) ? 1.0 : partCNN);
        }
        ConvolutionA(pImgConv3, pImg[0], partCNN);
#ifdef INFO
        cout << "Average, Part CNN : " << partCNN << endl;
#endif

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
