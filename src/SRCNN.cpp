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

#include <unistd.h>
#if !defined(NO_OMP)
    #include <omp.h>
#endif
#include <pthread.h>

#include "SRCNN.h"

/* pre-calculated convolutional data */
#include "convdata.h"

////////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace cv;
using namespace SRCNN;

////////////////////////////////////////////////////////////////////////////////

static float    image_multiply  = 2.0f;
static unsigned image_width     = 0;
static unsigned image_height    = 0;
static bool     opt_verbose     = true;
static bool     opt_cubicfilter = true;
static bool     opt_debug       = false;
static int      t_exit_code     = 0;

static string   path_me;
static string   file_me;
static string   file_src;
static string   file_dst;

////////////////////////////////////////////////////////////////////////////////

#define DEF_STR_VERSION		"0.1.1.6"

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
    
    #pragma omp parallel for
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

            #pragma omp parallel for
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

            #pragma omp parallel for
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

    int cnt = 0;

    #pragma omp parallel for private(cnt)
    for ( cnt=0; cnt<CONV2_FILTERS; cnt++)
    {
        src2[cnt].create( Size( src[cnt].cols + 4, 
                                src[cnt].rows + 4 ), 
                          CV_32F );

        for (int row = 0; row < src2[cnt].rows; row++)
        {
            for (int col = 0; col < src2[cnt].cols; col++)
            {
                int tmpRow = row - 2;
                int tmpCol = col - 2;

                if (tmpRow < 0)
                {
					tmpRow = 0;
				}
                else 
				if (tmpRow >= src[cnt].rows)
                {
					tmpRow = src[cnt].rows - 1;
				}

                if (tmpCol < 0)
                {
					tmpCol = 0;
				}
                else 
				if (tmpCol >= src[cnt].cols)
                {
					tmpCol = src[cnt].cols - 1;
				}

                src2[cnt].at<float>(row, col) = \
					src[cnt].at<float>(tmpRow, tmpCol);
            }
        }
    }

    int row = 0;

    /* Complete the Convolution Step */
    #pragma omp parallel for private( row )
    for ( row=0; row<dst.rows; row++ )
    {
        #pragma omp parallel for
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
            #pragma omp critical
			printf( "- Processing convolutional Layer III : %04d/%04d ...  \r",
			        row + 1,
					dst.rows );
            fflush( stdout );
		}
    }

	if ( opt_verbose == true )
	{
		printf( "\n" );
        fflush( stdout );
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
						image_multiply = tmpfv;
					}
				}
			}
			else
			if ( strtmp.find( "--noverbose" ) == 0 )
			{
				opt_verbose = false;
			}
			else
            if ( strtmp.find( "--nocubicfilter" ) == 0 )
            {
                opt_cubicfilter = false;
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
			DEF_STR_VERSION );
}

void printHelp()
{
	printf( "\n" );
	printf( "    usage : %s (options) [source file name] ([output file name])\n", file_me.c_str() );
	printf( "\n" );
	printf( "    _options_\n" );
	printf( "\n" );
	printf( "        --scale=( ratio: 0.1 to .. ) : scaling by ratio.\n" );
	printf( "        --noverbose                  : turns off all verbose\n" );
	printf( "        --nocubicfilter              : do not use cubic filter\n" );
    printf( "\n" );
}

void* pthreadcall( void* p )
{
     if ( opt_verbose == true )
    {
    	printTitle();
    	printf( "\n" );
    	printf( "- Scale multiply ratio : %.2f\n", image_multiply );
    	fflush( stdout );
    }
	
    /* Read the original image */
    Mat pImgOrigin;

    pImgOrigin = imread( file_src.c_str() );

	if ( pImgOrigin.empty() == false )
    {
        if ( opt_verbose == true )
        {
            printf( "- Image load : %s\n", file_src.c_str() );
            fflush( stdout );
        }
	}
	else
	{
        if ( opt_verbose == true )
        {
            printf( "- load failure : %s\n", file_src.c_str() );
        }

        t_exit_code = -1;
        pthread_exit( &t_exit_code );
	}

    // Test image resize target ...
    Size testsz = pImgOrigin.size();
    if ( ( ( (float)testsz.width * image_multiply ) <= 0.f ) ||
         ( ( (float)testsz.height * image_multiply ) <= 0.f ) )
    {
        if ( opt_verbose == true )
        {
            printf( "- Image scale error : ratio too small.\n" );
        }

        t_exit_code = -1;
        pthread_exit( &t_exit_code );
    }

    // -------------------------------------------------------------

    if ( opt_verbose == true )
    {
        printf( "- Image converting to Y-Cr-Cb : " );
        fflush( stdout );
    }

    /* Convert the image from BGR to YCrCb Space */
    Mat pImgYCrCb;
    cvtColor(pImgOrigin, pImgYCrCb, CV_BGR2YCrCb);
	
	if ( pImgYCrCb.empty() == false )
    {
        if ( opt_verbose == true )
        {
            printf( "Ok.\n" );
            fflush( stdout );
        }
    }
    else
	{
        if ( opt_verbose == true )
        {
            printf( "Failure.\n" );
        }

		t_exit_code = -2;
        pthread_exit( &t_exit_code );
	}

    // ------------------------------------------------------------

    if ( opt_verbose == true )
    {
        printf( "- Splitting channels : " );
        fflush( stdout );
    }

    /* Split the Y-Cr-Cb channel */
    vector<Mat> pImgYCrCbCh(3);
    split(pImgYCrCb, pImgYCrCbCh);

    if ( pImgYCrCb.empty() == false )
    {
        if ( opt_verbose == true )
        {
            printf( "Ok.\n" );
            fflush( stdout );
        }
    }
    else
    {
        if ( opt_verbose == true )
        {
            printf( "Failure.\n" );
            t_exit_code = -3;
            pthread_exit( &t_exit_code );
        }
    }

    // ------------------------------------------------------------

    if ( opt_verbose == true )
    {
        printf( "- Resizing slitted channels with bicublic interpolation : " );
    }

    /* Resize the Y-Cr-Cb Channel with Bicubic Interpolation */
    vector<Mat> pImg(3);
	
    #pragma omp parallel for
    for (int i = 0; i < 3; i++)
    {
		Size newsz = pImgYCrCbCh[i].size();
		newsz.width  *= image_multiply;
		newsz.height *= image_multiply;
		
        resize( pImgYCrCbCh[i], 
		        pImg[i], 
				newsz, 
				0, 
				0, 
				CV_INTER_CUBIC );
    }

    if ( opt_verbose == true )
    {
        printf( "Ok.\n" );
    }

    // -----------------------------------------------------------
    
    if ( opt_cubicfilter == true )
    {
        if ( opt_verbose == true )
        {
            printf( "- Optional processing bicubic filter : " );
            fflush( stdout );
        }
    
        /* Output the Cubic Inter Result (Optional) */
        Mat pImgCubic;
	
    	Size newsz = pImgYCrCb.size();
    	newsz.width  *= image_multiply;
    	newsz.height *= image_multiply;
	
        resize( pImgYCrCb, 
    	        pImgCubic, 
    			newsz,
    			0, 
    			0, 
    			CV_INTER_CUBIC );
			
        cvtColor(pImgCubic, pImgCubic, CV_YCrCb2BGR);

        if ( opt_verbose == true )
        {
            printf( "Ok.\n" );
            fflush( stdout );
        }
    }

    int cnt = 0;

    /******************* The First Layer *******************/
    vector<Mat> pImgConv1(CONV1_FILTERS);
    #pragma omp parallel for private( cnt )
    for ( cnt=0; cnt<CONV1_FILTERS; cnt++)
    {
        pImgConv1[cnt].create( pImg[0].size(), CV_32F );

        Convolution99( pImg[0], 
		               pImgConv1[cnt], 
					   weights_conv1_data[cnt], 
					   biases_conv1[cnt] );

		if ( opt_verbose == true )
        {
            #pragma omp critical
			printf( "- Processing convolutional layer I : %03d/%03d ... \r",
			        cnt + 1,
                    CONV1_FILTERS );
            fflush( stdout );
		}

    }

    if ( opt_verbose == true )
    {
        printf( "\n" );
        fflush( stdout );
    }
	
    /******************* The Second Layer *******************/
    vector<Mat> pImgConv2(CONV2_FILTERS);
    #pragma omp parallel for private( cnt )
    for ( cnt=0; cnt<CONV2_FILTERS; cnt++ )
    {
        pImgConv2[cnt].create(pImg[0].size(), CV_32F);
        Convolution11( pImgConv1, 
                       pImgConv2[cnt], 
                       weights_conv2_data[cnt], 
                       biases_conv2[cnt]);
		
		if ( opt_verbose == true )
        {
            #pragma omp critical
			printf( "- Processing convolutional layer II : %03d/%03d ... \r",
			        cnt + 1,
                    CONV2_FILTERS );
            fflush( stdout );
		}
    }

    if ( opt_verbose == true )
    {
        printf( "\n" );
        fflush( stdout );
    }

    /******************* The Third Layer *******************/
    Mat pImgConv3;
    pImgConv3.create(pImg[0].size(), CV_8U);
    Convolution55(pImgConv2, pImgConv3, weights_conv3_data, biases_conv3);
   
    if ( opt_verbose == true )
    {
        printf( "- Merging images : " );
        fflush( stdout );
    }

    /* Merge the Y-Cr-Cb Channel into an image */
    Mat pImgYCrCbOut;
    merge(pImg, pImgYCrCbOut);

	if ( opt_verbose == true )
    {
		printf( "Ok.\n" );
        fflush( stdout );
	}

    // ---------------------------------------------------------

    if ( opt_verbose == true )
    {
        printf( "- Converting channel to BGR : " );
        fflush ( stdout );
    }

    /* Convert the image from YCrCb to BGR Space */
    Mat pImgBGROut;
    cvtColor(pImgYCrCbOut, pImgBGROut, CV_YCrCb2BGR);

	if ( pImgBGROut.empty() == false )
	{
        printf( "Ok.\n" );
        printf( "- Writing result to %s : ", file_dst.c_str() );
        fflush( stdout );

		imwrite( file_dst.c_str() , pImgBGROut);
	
		printf( "Ok.\n" );
	}
	else
	{
		printf( "Failure.\n" );
		t_exit_code = -10;
        pthread_exit( &t_exit_code );
	}
	
	fflush( stdout );
		
    t_exit_code = 0;
    pthread_exit( NULL );
    return NULL;
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

    pthread_t ptt;
    int       tid = 0;

    if ( pthread_create( &ptt, NULL, pthreadcall, &tid ) == 0 )
    {
        // Adjust pthread elevation.
        int         ptpol = 99;
        struct \
        sched_param ptpar = {0};

        pthread_getschedparam( ptt, &ptpol, &ptpar );
        ptpar.sched_priority = sched_get_priority_max( ptpol );
        pthread_setschedparam( ptt, ptpol, &ptpar );
        pthread_join( ptt, NULL );
    }
    else
    {
        printf( "Error: pthread failure.\n" );
    }
	
    return 0;
}
#endif /// of EXPORTLIB
