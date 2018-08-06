/*******************************************************************************
 * SRCNN: Super-Resolution with deep Convolutional Neural Networks
 * ----------------------------------------------------------------------------
 * Current Author : Raphael Kim ( rageworx@gmail.com )
 * Pre-Author     : Wang Shu
 * Origin-Date    @ Sun 13 Sep, 2015
 * Descriptin ..
 *                 This source code modified version from Origianl code of Wang
 *                Shu's. All license following from origin.
*******************************************************************************/

////////////////////////////////////////////////////////////////////////////////
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <omp.h>

#include "SRCNN.h"

/* pre-calculated convolutional data */
#include "convdata.h"

////////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace cv;
using namespace SRCNN;

////////////////////////////////////////////////////////////////////////////////

/* Marco Definition */
#define USE_CUBIC

static float    image_mulifly   = 2.0f;
static unsigned image_width     = 0;
static unsigned image_height    = 0;
static bool     opt_verbose     = false;

static string   path_me;
static string   file_me;
static string   file_src;
static string   file_dst;

////////////////////////////////////////////////////////////////////////////////

#define STR_VERSION     "0.1.1.5"

////////////////////////////////////////////////////////////////////////////////

/* Function Declaration */
void Convolution99( Mat& src, Mat& dst, \
                    float kernel[9][9], float bias);

void Convolution11( vector<Mat>& src, Mat& dst, \
                    float kernel[CONV1_FILTERS], float bias);

void Convolution55( vector<Mat>& src, Mat& dst, \
                    float kernel[32][5][5], float bias);

////////////////////////////////////////////////////////////////////////////////

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
    /* Expand the src image */
    Mat src2;
    src2.create(Size(src.cols + 8, src.rows + 8), CV_8U);
    
    for (int row = 0; row < src2.rows; row++)
    {
        for (int col = 0; col < src2.cols; col++)
        {
            int tmpRow = row - 4;
            int tmpCol = col - 4;

            if (tmpRow < 0)
            {
                tmpRow = 0;
            }
            else 
            if (tmpRow >= src.rows)
            {
                tmpRow = src.rows - 1;
            }

            if (tmpCol < 0)
            {
                tmpCol = 0;
            }
            else 
            if (tmpCol >= src.cols)
            {
                tmpCol = src.cols - 1;
            }

            src2.at<unsigned char>(row, col) = \
                src.at<unsigned char>(tmpRow, tmpCol);
        }
    }

    /* Complete the Convolution Step */
    for (int row = 0; row < dst.rows; row++)
    {
        for (int col = 0; col < dst.cols; col++)
        {
            /* Convolution */
            float temp = 0;
            for (int i = 0; i < 9; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    temp += kernel[i][j] * src2.at<unsigned char>(row + i, col + j);
                }
            }
            
            temp += bias;

            /* Threshold */
            temp = (temp >= 0) ? temp : 0;

            dst.at<float>(row, col) = temp;
        }
    }
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
    for (int row = 0; row < dst.rows; row++)
    {
        for (int col = 0; col < dst.cols; col++)
        {
            /* Process with each pixel */
            float temp = 0;
            for (int i = 0; i < CONV1_FILTERS; i++)
            {
                temp += src[i].at<float>(row, col) * kernel[i];
            }
            temp += bias;

            /* Threshold */
            temp = (temp >= 0) ? temp : 0;

            dst.at<float>(row, col) = temp;
        }
    }
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
    /* Expand the src image */
    vector<Mat> src2(CONV2_FILTERS);
    for (int i = 0; i < CONV2_FILTERS; i++)
    {
        src2[i].create(Size(src[i].cols + 4, src[i].rows + 4), CV_32F);
        for (int row = 0; row < src2[i].rows; row++)
        {
            for (int col = 0; col < src2[i].cols; col++)
            {
                int tmpRow = row - 2;
                int tmpCol = col - 2;

                if (tmpRow < 0)
                {
                    tmpRow = 0;
                }
                else 
                if (tmpRow >= src[i].rows)
                {
                    tmpRow = src[i].rows - 1;
                }

                if (tmpCol < 0)
                {
                    tmpCol = 0;
                }
                else 
                if (tmpCol >= src[i].cols)
                {
                    tmpCol = src[i].cols - 1;
                }

                src2[i].at<float>(row, col) = \
                    src[i].at<float>(tmpRow, tmpCol);
            }
        }
    }

    /* Complete the Convolution Step */
    for (int row = 0; row < dst.rows; row++)
    {
        for (int col = 0; col < dst.cols; col++)
        {
            float temp = 0;

            for (int i = 0; i < CONV2_FILTERS; i++)
            {
                double temppixel = 0;
                for (int m = 0; m < 5; m++)
                {
                    for (int n = 0; n < 5; n++)
                    {
                        temppixel += \
                        kernel[i][m][n] * src2[i].at<float>(row + m, col + n);
                    }
                }

                temp += temppixel;
            }

            temp += bias;

            /* Threshold */
            temp = (temp >= 0) ? temp : 0;
            temp = (temp <= 255) ? temp : 255;

            dst.at<unsigned char>(row, col) = (unsigned char)temp;
        }

        if ( opt_verbose == true )
        {
            printf( "Convolutional Layer III : %04d/%04d Completed ...  \r",
                    row + 1,
                    dst.rows );
        }
    }

    if ( opt_verbose == true )
    {
        printf( "\n" );
    }
    
    return;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef EXPORTLIB

bool parseArgs( int argc, char** argv )
{
    for( int cnt=0; cnt<argc; cnt++ )
    {
        string strtmp = argv[ cnt ];
        size_t fpos   = string::npos;

        if ( cnt == 0 )
        {
            fpos = strtmp.find_last_of( "\\" );

            if ( fpos == string::npos )
            {
                fpos = strtmp.find_last_of( "/" );
            }

            if ( fpos != string::npos )
            {
                path_me = strtmp.substr( 0, fpos );
                file_me = strtmp.substr( fpos + 1 );
            }
            else
            {
                file_me = strtmp;
            }
        }
        else
        {
            if ( strtmp.find( "--scale=" ) == 0 )
            { 
                string strval = strtmp.substr( 8 );
                if ( strval.size() > 0 )
                {
                    float tmpfv = atof( strval.c_str() );
                    if ( tmpfv > 0.f )
                    {
                        image_mulifly = tmpfv;
                    }
                }
            }
            else
            if ( strtmp.find( "--verbose" ) == 0 )
            {
                opt_verbose = true;
            }
            else
            if ( file_src.size() == 0 )
            {
                file_src = strtmp;
            }
            else
            if ( file_dst.size() == 0 )
            {
                file_dst = strtmp;
            }
        }
    }
    
    if ( ( file_src.size() > 0 ) && ( file_dst.size() == 0 ) )
    {
        string convname = file_src;
        string srcext;
        
        // changes name without file extention.
        size_t posdot = file_src.find_last_of( "." );
        if ( posdot != string::npos )
        {
            convname = file_src.substr( 0, posdot );
            srcext   = file_src.substr( posdot );
        }
        
        convname += "_resized";
        if ( srcext.size() > 0 )
        {
            convname += srcext;
        }
        
        file_dst = convname;
    }
    
    if ( ( file_src.size() > 0 ) && ( file_dst.size() > 0 ) )
    {
        return true;
    }
    
    return false;
}

void printTitle()
{
    printf( "%s : Super-Resolution with deep Convolutional Neural Networks\n",
            file_me.c_str() );
    printf( "(C)2018 Raphael Kim, pre-author : Wang Shu., Program version %s\n",
            STR_VERSION );
}

void printHelp()
{
    printf( "\n" );
    printf( "    usage : %s (options) [source file name] ([output file name])\n" );
    printf( "\n" );
    printf( "    _options_\n" );
    printf( "\n" );
    printf( "        --scale=(ratio: 0.1 to .. ) : scaling by ratio.\n" );
    printf( "        --verbose                   : turns on verbose\n" );
    printf( "\n" );
}

/***
 * FuncName : main
 * Function : the entry of the program
 * Parameter    : argc - the number of the initial parameters
 *        argv - the entity of the initial parameters
 * Output   : int 0 for normal / int 1 for failed
***/
int main( int argc, char** argv )
{
    if ( parseArgs( argc, argv ) == false )
    {
        printTitle();
        printHelp();
        fflush( stdout );
        return 0;
    }
    
    printTitle();
    printf( "\n" );
    printf( "Scale raitio : %.2f\n", image_mulifly );
    fflush( stdout );
    
    /* Read the original image */
    Mat pImgOrigin;

    pImgOrigin = imread( file_src.c_str() );

    if ( pImgOrigin.empty() == false )
    {
        printf( "Image %s loaded.\n", file_src.c_str() );
		fflush( stdout );
    }
    else
    {
        printf( "Cannot load file \"%s\"\n", file_src.c_str() );
        return -1;
    }
	

    /* Convert the image from BGR to YCrCb Space */
    Mat pImgYCrCb;
    cvtColor(pImgOrigin, pImgYCrCb, CV_BGR2YCrCb);
    
    if ( pImgYCrCb.empty() == true )
    {
        printf( "YCrCb Covert failure.\n" );
        return -2;
    }
    else
    if ( opt_verbose == true )
    {
        printf( "Converting Image to YCrCb done.\n" );
		fflush( stdout );
    }

	
    /* Split the Y-Cr-Cb channel */
    vector<Mat> pImgYCrCbCh(3);
    split(pImgYCrCb, pImgYCrCbCh);

    if ( opt_verbose == true )
    {
        printf( "Spliting the Y-Cr-Cb Channel.\n" );
		fflush( stdout );
    }

    /* Resize the Y-Cr-Cb Channel with Bicubic Interpolation */
    vector<Mat> pImg(3);
    
    for (int i = 0; i < 3; i++)
    {
        Size newsz = pImgYCrCbCh[i].size();
        newsz.width  *= image_mulifly;
        newsz.height *= image_mulifly;
        
        resize( pImgYCrCbCh[i], 
                pImg[i], 
                newsz, 
                0, 
                0, 
                CV_INTER_CUBIC );
    }

    if ( opt_verbose == true )
    {
        printf( "Completed Bicubic Interpolation.\n" );
		fflush( stdout );
    }

#ifdef USE_CUBIC
    /* Output the Cubic Inter Result (Optional) */
    Mat pImgCubic;
    
    Size newsz = pImgYCrCb.size();
    newsz.width  *= image_mulifly;
    newsz.height *= image_mulifly;
    
    resize( pImgYCrCb, 
            pImgCubic, 
            newsz,
            0, 
            0, 
            CV_INTER_CUBIC );
            
    cvtColor(pImgCubic, pImgCubic, CV_YCrCb2BGR);

    if ( opt_verbose == true )
    {
        printf( "Completed Bicubic Version (Optional).\n" );
		fflush( stdout );
    }
#endif

    /******************* The First Layer *******************/
    vector<Mat> pImgConv1(CONV1_FILTERS);
    for (int i = 0; i < CONV1_FILTERS; i++)
    {
        pImgConv1[i].create( pImg[0].size(), CV_32F );

        Convolution99( pImg[0], 
                       pImgConv1[i], 
                       weights_conv1_data[i], 
                       biases_conv1[i] );

        if ( opt_verbose == true )
        {
            printf( "Convolutional Layer I : %03d/%03d Cell Completed ... \r",
                    i + 1,
                    64 );                   
        }
		fflush( stdout );
    }

    if ( opt_verbose == true )
    {
        printf( "\n" );
        printf( "Convolutional Layer I : 100%% Complete.\n " );
		fflush( stdout );
    }
    
    /******************* The Second Layer *******************/
    vector<Mat> pImgConv2(CONV2_FILTERS);
    for (int i = 0; i < CONV2_FILTERS; i++)
    {
        pImgConv2[i].create(pImg[0].size(), CV_32F);
        Convolution11(pImgConv1, pImgConv2[i], weights_conv2_data[i], biases_conv2[i]);
        
        if ( opt_verbose == true )
        {
            printf( "Convolutional Layer II : %03d/%03d Cell Completed ... \r",
                    i + 1,
                    32 );                   
        }
		fflush( stdout );
    }

    if ( opt_verbose == true )
    {
        printf( "\n" );
        printf( "Convolutional Layer II : 100%% Complete.\n " );
		fflush( stdout );
    }

    /******************* The Third Layer *******************/
    Mat pImgConv3;
    pImgConv3.create(pImg[0].size(), CV_8U);
    Convolution55(pImgConv2, pImgConv3, weights_conv3_data, biases_conv3);

    if ( opt_verbose == true )
    {
        printf( "\n" );
        printf( "Convolutional Layer III : 100% Complete.\n " );
		fflush( stdout );
    }
    
    /* Merge the Y-Cr-Cb Channel into an image */
    Mat pImgYCrCbOut;
    merge(pImg, pImgYCrCbOut);

    if ( opt_verbose == true )
    {
        printf( "Merge Image Complete.\n" );
		fflush( stdout );
    }

    /* Convert the image from YCrCb to BGR Space */
    Mat pImgBGROut;
    cvtColor(pImgYCrCbOut, pImgBGROut, CV_YCrCb2BGR);

    if ( pImgBGROut.empty() == false )
    {
        imwrite( file_dst.c_str() , pImgBGROut);
    
        printf( "Image written as %s\n ", file_dst.c_str() );
    }
    else
    {
        printf( "Failed to convert image to BGR.\n" );
        return -10;
    }
    
    fflush( stdout );
        
    return 0;
}
#endif /// of EXPORTLIB