/*******************************************************************************
** libSRCNN: Library of Super-Resolution with deep Convolutional Neural Networks
** ----------------------------------------------------------------------------
** Current Author : Raphael Kim ( rageworx@gmail.com )
*******************************************************************************/
#ifdef EXPORTLIB
////////////////////////////////////////////////////////////////////////////////
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifndef NO_OMP
    #include <omp.h>
#endif

#include "libsrcnn.h"

/* pre-calculated convolutional data */
#include "convdata.h"

////////////////////////////////////////////////////////////////////////////////

using namespace std;

////////////////////////////////////////////////////////////////////////////////

#define DEF_PRECALC_KR	0.299f
#define DEF_PRECALC_KG	0.587f
#define DEF_PRECALC_KB	0.114f

////////////////////////////////////////////////////////////////////////////////

typedef struct
{
	unsigned char* buff;
	unsigned       width;
	unsigned       height;
	unsigned       depth;
}ImgU8;

typedef struct
{
	float*         buff;
	unsigned       width;
	unsigned       height;
	unsigned       depth;
}ImgF32;


////////////////////////////////////////////////////////////////////////////////

static bool opt_cubicfilter = true;

////////////////////////////////////////////////////////////////////////////////

void convolution99( ImgU8 &src, ImgF32 &dst, KernelMat99 kernel, float bias );
void convolution11( vector<ImgF32*> &src, ImgF32 &dst, ConvKernel1 kernel, float bias );
void convolution55( vector<ImgF32*> &src, ImgU8 &dst, ConvKernel32_55 kernel, float bias );

////////////////////////////////////////////////////////////////////////////////

// some utility functions here ...

void resetImgU8( ImgU8 &img )
{
	img.width = 0;
	img.height = 0;
	img.depth = 0;
	
	if ( img.buff != NULL )
	{
		delete[] img.buff;
		img.buff = NULL;
	}
}

void resetImgF32( ImgF32 &img )
{
	img.width = 0;
	img.height = 0;
	img.depth = 0;
	
	if ( img.buff != NULL )
	{
		delete[] img.buff;
		img.buff = NULL;
	}
}

// RGB->YCbCr refered to 
// http://atrahasis.tistory.com/entry/YCbCr-%EA%B3%BC-RGB-%EA%B0%84%EC%9D%98-%EC%83%81%ED%98%B8%EB%B3%80%ED%99%98-%EA%B3%B5%EC%8B%9D%EC%9D%98-%EC%9D%BC%EB%B0%98%ED%99%94%EB%90%9C-%ED%91%9C%EC%A4%80%EA%B3%B5%EC%8B%9D-%EC%A0%95%EB%A6%AC
void converImgU8toYCbCr( ImgU8 &src, vector<ImgF32*> &out )
{
	if ( src.depth < 3 )
		return;
	
	// create 3 channels of Y-Cb-Cr
	for( unsigned cnt=0; cnt<3; cnt++ )
	{
		ImgF32* imgT = new ImgF32;
		imgT->width  = src.width;
		imgT->height = src.height;
		imgT->depth  = 1;
		imgT->buff   = new float[ src.width * src.height ];
		out.push_back( imgT );
	}
	
	unsigned imgsz = src.width * src.height;
	
	for( unsigned cnt=0; cnt<imgsz; cnt++ )
	{
		float fR = src.buffer[ ( cnt + 0 ) * src.depth ];
		float fG = src.buffer[ ( cnt + 1 ) * src.depth ];
		float fB = src.buffer[ ( cnt + 2 ) * src.depth ];
		
		// Y
		out[0]->buffer[cnt] = ( DEF_PRECALC_KR * fR ) +
	                          ( DEF_PRECALC_KG * fG ) +
							  ( DEF_PRECALC_KB * fB );
		
		// Cb
		out[1]->buffer[cnt] = ( -0.16874f * fR ) -
		                      ( 0.33126f * ( 0.5f * fB ) );

		// Cr
		out[2]->buffer[cnt] = ( 0.5f * fR ) - 
		                      ( 0.41869f * fG ) -
							  ( 0.08131f * fB );
	}
}

void convertYCbCrtoImgU8( vector<ImgF32*> &src, ImgU8* &out )
{
	if ( src.size() != 3 )
		return;
	
	out = new ImgU8;
	
	if ( out == NULL )
		return;
	
	out->width  = src[0].width;
	out->height = src[0].height;
	out->depth  = 3;
	out->buff   = new unsigned char[ out.width * out.height * 3 ];
	
	if ( out->buff == NULL )
		return;
	
	unsigned imgsz = src.width * src.height;
	
	for( unsigned cnt=0; cnt<imgsz; cnt++ )
	{
		float fY  = src[0]->buffer[cnt];
		float fCb = src[1]->buffer[cnt];
		float fCr = src[2]->buffer[cnt];

		// Red -> Green -> Blue ...
		out->buffer[( cnt + 0 ) * 3] = fY + ( 1.402f * fCr );
		out->buffer[( cnt + 1 ) * 3] = fY - ( 0.34414f * fCb ) - ( 0.71414 * fCr );
		out->buffer[( cnt + 2 ) * 3] = fY + ( 1.772 * fCb );
	}
}

////////////////////////////////////////////////////////////////////////////////

void convolution99( ImgU8 &src, ImgF32 &dst, KernelMat99 kernel, float bias )
{
    /* Expand the src image */
	dst.width  = src.width + 8;
	dst.height = src.height + 8;
	dst.depth  = src.depth;
	unsigned dstsz = dst.width * dst.heigfht * dst.dpeth;
	dst.buffer = new float[ dstsz ];
	
	if ( dst.buffer == NULL )
	{
		resetImgF32( dst );
		return;
	}
	    
    #pragma omp parallel for
    for ( unsigned row = 0; row<dst.height; row++ )
    {
        for ( unsigned col = 0; col<dst.width; col++ )
        {
            int tmpRow = (int)row - 4;
            int tmpCol = (int)col - 4;

            if ( tmpRow < 0 )
            {
				tmpRow = 0;
			}
            else 
		    if ( tmpRow >= src.height )
            {
				tmpRow = src.height - 1;
			}

            if ( tmpCol < 0 )
            {
				tmpCol = 0;
			}
            else 
			if ( tmpCol >= src.width )
            {
				tmpCol = src.width - 1;
			}

			for( unsigned cnt=0; cnt<src.depth; cnt++ )
			{
				dst.buff[ ( row * dst.width + col ) * cnt  ] = \
					ctx.refbuff[ ( tmpRow * src.width + tmpCol ) * cnt ];
			}
        }
    }

    /* Complete the Convolution Step */
    for (int row = 0; row < dst.height; row++)
    {
        for (int col = 0; col < dst.width; col++)
        {
            /* Convolution */
            float temp = 0;

            #pragma omp parallel for
			for( unsigned d=0; d<ctx.depth; d++ )
			{
				for ( unsigned x=0; x<9; x++ )
				{
					for ( unsigned y=0; y<9; y++ )
					{
						unsigned pos = ( ( row  + x ) * expsz_w +  ( col + y ) ) * d;
						temp += kernel[x][y] * dst.buff[ pos ];						
					}
				}
				
				temp += bias;

				/* Threshold */
				temp = (temp >= 0) ? temp : 0;

				dst.buff[ ( row * dst.width + col ) * d ] = temp;
			}
        }
    }
}

void convolution11( vector<ImgF32> &src, ImgF32 &dst, ConvKernel1 kernel, float bias )
{
	//#pragma omp parallel for
	for( unsigned d=0; d<dst.depth; d++ )
	{
		for ( unsigned row=0; row<dst.height; row++ )
		{
			for ( unsigned col=0; col<dst.width; col++ )
			{
				/* Process with each pixel */
				float temp = 0;

				#pragma omp parallel for
				for ( unsigned fc=0; fc<CONV1_FILTERS; fc++ )
				{
					temp += src[fc].buff[ ( row * src[fc].width + col ) * d ] \
					        * kernel[fc];
				}
				
				temp += bias;

				/* Threshold */
				temp = (temp >= 0) ? temp : 0;

				dst.buff[ ( row * dst.width + col ) * d ] = temp;
			}
		}
	}
}

void convolution55( vector<ImgF32> &src, ImgU8 &dst, ConvKernel32_55 kernel, float bias )
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
    }

    return;
}

////////////////////////////////////////////////////////////////////////////////

int ProcessSRNN( const unsigned char* refbuff, 
                 unsigned w, unsigned h, unsigned d,
                 float muliply,
                 unsigned char* &outbuff,
                 unsigned &outbuffsz );
{
	if ( ( refbuff == NULL ) || ( w == 0 ) || ( h == 0 ) || ( d == 0 ) )
		return -1;
	
    if ( ( ( (float)w * image_multiply ) <= 0.f ) ||
         ( ( (float)h * image_multiply ) <= 0.f ) )
	{
		return -2;
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

    if ( opt_verbose == true )
    {
        printf( "- Processing convolutional layer I ... " );
        fflush( stdout );
    }

    vector<Mat> pImgConv1(CONV1_FILTERS);
    #pragma omp parallel for private( cnt )
    for ( cnt=0; cnt<CONV1_FILTERS; cnt++)
    {
        pImgConv1[cnt].create( pImg[0].size(), CV_32F );

        Convolution99( pImg[0], 
		               pImgConv1[cnt], 
					   weights_conv1_data[cnt], 
					   biases_conv1[cnt] );
    }

    if ( opt_verbose == true )
    {
        printf( "completed.\n" );
        fflush( stdout );
    }

    /******************* The Second Layer *******************/

    if ( opt_verbose == true )
    {
        printf( "- Processing convolutional layer II ... " );
        fflush( stdout );
    }

    vector<Mat> pImgConv2(CONV2_FILTERS);
    #pragma omp parallel for private( cnt )
    for ( cnt=0; cnt<CONV2_FILTERS; cnt++ )
    {
        pImgConv2[cnt].create(pImg[0].size(), CV_32F);
        Convolution11( pImgConv1, 
                       pImgConv2[cnt], 
                       weights_conv2_data[cnt], 
                       biases_conv2[cnt]);	
    }

    if ( opt_verbose == true )
    {
        printf( "completed.\n" );
        fflush( stdout );
    }

    /******************* The Third Layer *******************/

    if ( opt_verbose == true )
    {
        printf( "- Processing convolutional layer III ... " );
        fflush( stdout );
    }

    Mat pImgConv3;
    pImgConv3.create(pImg[0].size(), CV_8U);
    Convolution55(pImgConv2, pImgConv3, weights_conv3_data, biases_conv3);
   
    if ( opt_verbose == true )
    {
        printf( "completed.\n");
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
        if ( opt_verbose == true )
        {
            printf( "Ok.\n" );
            printf( "- Writing result to %s : ", file_dst.c_str() );
            fflush( stdout );
        }

		imwrite( file_dst.c_str() , pImgBGROut);
	
        if ( opt_verbose == true )
        {
            printf( "Ok.\n" );
        }
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
#endif /// of EXPORTLIB