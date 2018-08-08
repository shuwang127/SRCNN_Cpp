/*******************************************************************************
** libSRCNN: Library of Super-Resolution with deep Convolutional Neural Networks
** ----------------------------------------------------------------------------
** Current Author : Raphael Kim ( rageworx@gmail.com )
** Previous Author : Wang Shu ( https://github.com/shuwang127/SRCNN_Cpp )
** Referenced to : http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
**
** [ Updates ]
**
** - 2018-08-08 -
**     First C++ code ( non-OpenCV ) code released.
**     Tested with MinGW-W64 and G++ @ AARCH64 ( nVidia Jetson TX2 )
**
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
#include "frawscale.h"

/* pre-calculated convolutional data */
#include "convdata.h"

////////////////////////////////////////////////////////////////////////////////

using namespace std;

////////////////////////////////////////////////////////////////////////////////

namespace libsrcnn {

////////////////////////////////////////////////////////////////////////////////

typedef struct
{
    unsigned       width;
    unsigned       height;
    unsigned       depth;
    unsigned char* buff;
}ImgU8;

typedef struct
{
    unsigned       width;
    unsigned       height;
    unsigned       depth;
    float*         buff;
}ImgF32;

typedef struct
{
    ImgF32      Y;
    ImgF32      Cb;
    ImgF32      Cr;
}ImgYCbCr;

typedef ImgF32  ImgConv1Layers[CONV1_FILTERS];
typedef ImgF32  ImgConv2Layers[CONV2_FILTERS];

////////////////////////////////////////////////////////////////////////////////

static bool opt_cubicfilter = true;

////////////////////////////////////////////////////////////////////////////////

void convolution99( ImgF32 &src, ImgF32 &dst, const KernelMat99 kernel, float bias );
void convolution11( ImgConv1Layers &src, ImgYCbCr &dst, const ConvKernel1 kernel, float bias );
void convolution55( ImgConv2Layers &src, ImgU8 &dst, const ConvKernel32_55 kernel, float bias );

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

void initImgU8( ImgU8 &img, unsigned w, unsigned h )
{
    img.width  = w;
    img.height = h;
    img.depth  = 3;

    unsigned imgsz = w * h * 3;
    img.buff = new unsigned char[ imgsz ];
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

void initImgF32( ImgF32 &img, unsigned w, unsigned h )
{
    img.width = w;
    img.height = h;
    img.depth = 1;
    
    unsigned buffsz = w * h;
    img.buff = new float[ buffsz ];
}

void initImgConvLayers( ImgF32* img, unsigned w, unsigned h, unsigned count )
{
    if ( img != NULL )
    {
        for( unsigned cnt=0; cnt<count; cnt++ )
        {
            img[cnt].width  = w;
            img[cnt].height = h;
            img[cnt].depth  = 1;

            unsigned buffsz = w * h;
            img[cnt].buff = new float[ buffsz ];
        }
    }
}

void discardConvLayers( ImgF32* img, unsigned count )
{
    if ( img != NULL )
    {
        for( unsigned cnt=0; cnt<count; cnt++ )
        {
            if ( img[cnt].buff != NULL )
            {
                delete[] img[cnt].buff;
                img[cnt].buff = NULL;
            }
        }
    }
}

void discardImgYCbCr( ImgYCbCr &img )
{
    resetImgF32( img.Y );
    resetImgF32( img.Cb );
    resetImgF32( img.Cr );
}

void initImgYCbCr( ImgYCbCr &img, unsigned w, unsigned h )
{
    initImgF32( img.Y, w, h );
    initImgF32( img.Cb, w, h );
    initImgF32( img.Cr, w, h );
}

void converImgU8toYCbCr( ImgU8 &src, ImgYCbCr &out )
{
    if ( src.depth < 3 )
        return;
    
    initImgYCbCr( out, src.width, src.height );
    
    unsigned imgsz = src.width * src.height;
    
    for( unsigned cnt=0; cnt<imgsz; cnt++ )
    {
        float fR = src.buff[ ( cnt * src.depth ) + 0 ];
        float fG = src.buff[ ( cnt * src.depth ) + 1 ];
        float fB = src.buff[ ( cnt * src.depth ) + 2 ];
        
        // Y
        out.Y.buff[cnt] = ( 0.299f * fR ) + ( 0.587f * fG ) + ( 0.114f * fB );
        
        // Cb
        out.Cb.buff[cnt] = ( -0.16874f * fR ) - ( 0.33126f * fG ) + ( 0.5f * fB );

        // Cr
        out.Cr.buff[cnt] = ( 0.5f * fR ) - ( 0.41869f * fG ) - ( 0.08131f * fB );
    }
}

void convertImgF32x3toImgU8( ImgF32* src, ImgU8 &out )
{
    if ( src == NULL )
        return;
    
    unsigned imgsz = src[0].width * src[0].height;

    out.width  = src[0].width;
    out.height = src[0].height;
    out.depth  = 3;
    out.buff   = new unsigned char[ imgsz * 3 ];        
        
    for( unsigned cnt=0; cnt<imgsz; cnt++ )
    {
        float fY  = src[0].buff[cnt];
        float fCb = src[1].buff[cnt];
        float fCr = src[2].buff[cnt];

        // Red -> Green -> Blue ...
        out.buff[( cnt * 3 ) + 0] = \
                (unsigned char)(fY + ( 1.402f * fCr ));
        out.buff[( cnt * 3 ) + 1] = \
                (unsigned char)(fY - ( 0.34414f * fCb ) - ( 0.71414 * fCr ));
        out.buff[( cnt * 3 ) + 2] = \
                (unsigned char)(fY + ( 1.772 * fCb ));
    }
}

void convertYCbCrtoImgU8( ImgYCbCr &src, ImgU8* &out )
{
    out = new ImgU8;
    
    if ( out == NULL )
        return;
    
    unsigned imgsz = src.Y.width * src.Y.height;

    out->width  = src.Y.width;
    out->height = src.Y.height;
    out->depth  = 3;
    out->buff   = new unsigned char[ imgsz * 3 ];
    
    if ( out->buff == NULL )
        return;
    
    for( unsigned cnt=0; cnt<imgsz; cnt++ )
    {
        float fY  = src.Y.buff[cnt];
        float fCb = src.Cb.buff[cnt];
        float fCr = src.Cr.buff[cnt];

        // Red -> Green -> Blue ...
        out->buff[( cnt * 3 ) + 0] = \
                (unsigned char)(fY + ( 1.402f * fCr ));
        out->buff[( cnt * 3 ) + 1] = \
                (unsigned char)(fY - ( 0.34414f * fCb ) - ( 0.71414 * fCr ));
        out->buff[( cnt * 3 ) + 2] = \
                (unsigned char)(fY + ( 1.772 * fCb ));
    }
}

////////////////////////////////////////////////////////////////////////////////

void convolution99( ImgF32 &src, ImgF32 &dst, const KernelMat99 kernel, float bias )
{
    /* Expand the src image */
    ImgF32 src2;
    initImgF32( src2, src.width + 8, src.height + 8 );
    
    if ( src2.buff == NULL )
    {
        resetImgF32( src2 );
        return;
    }
    
    unsigned row = 0;
    
    #pragma omp parallel for private(row)
    for ( row = 0; row<src2.height; row++ )
    {
        for ( unsigned col = 0; col<src2.width; col++ )
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

            src2.buff[ row * dst.width + col ] = \
                                src.buff[ tmpRow * src.width + tmpCol ];
        }
    }

    /* Complete the Convolution Step */
    #pragma omp parallel for private(row)
    for ( row=0; row<dst.height; row++ )
    {
        for ( unsigned col=0; col<dst.width; col++ )
        {
            /* Convolution */
            float temp = 0;

            for ( unsigned x=0; x<9; x++ )
            {
                for ( unsigned y=0; y<9; y++ )
                {
                    unsigned pos = ( row  + x ) * src2.width + ( col + y );
    
                    temp += kernel[x][y] * src2.buff[pos];                      
                }
            }
            
            temp += bias;

            /* Threshold */
            temp = (temp >= 0) ? temp : 0;

            dst.buff[ row * dst.width + col ] = temp;
        }
    }
    
    delete[] src2.buff;
}

void convolution11( ImgConv1Layers &src, ImgF32 &dst, const ConvKernel1 kernel, float bias )
{
    //#pragma omp parallel for
    for ( unsigned row=0; row<dst.height; row++ )
    {
        for ( unsigned col=0; col<dst.width; col++ )
        {
            /* Process with each pixel */
            float temp = 0;

            #pragma omp parallel for
            for ( unsigned fc=0; fc<CONV1_FILTERS; fc++ )
            {
                temp += src[fc].buff[ row * src[fc].width + col ] * kernel[fc];                     
            }
            
            temp += bias;

            /* Threshold */
            temp = (temp >= 0) ? temp : 0;

            dst.buff[ row * dst.width + col ] = temp;
        }
    }
}

void convolution55( ImgConv2Layers &src, ImgU8 &dst, const ConvKernel32_55 kernel, float bias )
{
    /* Expand the src image */
    ImgConv2Layers src2;
    initImgConvLayers( &src2[0], 
                       src[0].width + 4, 
                       src[0].height + 4, 
                       CONV2_FILTERS );

    unsigned cnt = 0;

    #pragma omp parallel for private(cnt)
    for ( cnt=0; cnt<CONV2_FILTERS; cnt++ )
    {
        for ( unsigned row=0; row<src2[cnt].height; row++ )
        {
            for ( unsigned col=0; col<src2[cnt].width; col++ )
            {
                int tmpRow = (int)row - 2;
                int tmpCol = (int)col - 2;

                if (tmpRow < 0)
                {
                    tmpRow = 0;
                }
                else 
                if (tmpRow >= src[cnt].height)
                {
                    tmpRow = src[cnt].height - 1;
                }

                if (tmpCol < 0)
                {
                    tmpCol = 0;
                }
                else 
                if (tmpCol >= src[cnt].width)
                {
                    tmpCol = src[cnt].width - 1;
                }

                src2[cnt].buff[ row * src2[cnt].width + col ] = \
                        src[cnt].buff[ tmpRow * src[cnt].width + tmpCol ];
            }
        }
    }
    
    int row = 0;

    /* Complete the Convolution Step */
    #pragma omp parallel for private( row )
    for ( row=0; row<dst.height; row++ )
    {
        #pragma omp parallel for
        for ( unsigned col=0; col<dst.width; col++ )
        {
            float temp = 0;

            for ( unsigned i=0; i<CONV2_FILTERS; i++ )
            {
                double temppixel = 0;
                
                for ( unsigned y=0; y<5; y++ )
                {
                    for ( unsigned x=0; x<5; x++ )
                    {
                        unsigned pos = (row + y) * src2[i].width + (col + x);
                        temppixel += kernel[i][x][y] * src2[i].buff[pos];
                    }
                }

                temp += temppixel;
            }

            temp += bias;

            /* Threshold */
            temp = (temp >= 0) ? temp : 0;
            temp = (temp <= 255) ? temp : 255;

            dst.buff[ row * dst.width + col ] = (unsigned char)temp;
        }
    }
    
    discardConvLayers( &src2[0], CONV2_FILTERS );   
}

////////////////////////////////////////////////////////////////////////////////

}; /// of namespace libsrcnn

////////////////////////////////////////////////////////////////////////////////

int ProcessSRCNN( const unsigned char* refbuff, 
                  unsigned w, unsigned h, unsigned d,
                  float muliply,
                  unsigned char* &outbuff,
                  unsigned &outbuffsz )
{
    if ( ( refbuff == NULL ) || ( w == 0 ) || ( h == 0 ) || ( d == 0 ) )
        return -1;
    
    if ( ( ( (float)w * muliply ) <= 0.f ) ||
         ( ( (float)h * muliply ) <= 0.f ) )
    {
        return -2;
    }

    // -------------------------------------------------------------
    // Convert RGB to Y-Cb-Cr
    
    // warning: imgSrc is referenced, don't remove from memory !
    libsrcnn::ImgU8     imgSrc = { w ,h ,d, (unsigned char*)refbuff };
    libsrcnn::ImgYCbCr  imgYCbCr;
    
    converImgU8toYCbCr( imgSrc, imgYCbCr );
    
    /* Resize the Y-Cr-Cb Channel with Bicubic Interpolation */
    libsrcnn::ImgF32 imgResized[3];
    const float* refimgbuf[3] = { imgYCbCr.Y.buff, 
                                  imgYCbCr.Cb.buff, 
                                  imgYCbCr.Cr.buff };
    
    unsigned rs_w = imgYCbCr.Y.width  * muliply;
    unsigned rs_h = imgYCbCr.Y.height * muliply;
        
    #pragma omp parallel for
    for ( unsigned cnt=0; cnt<3; cnt++ )
    {
        imgResized[cnt].width  = rs_w;
        imgResized[cnt].height = rs_h;
        imgResized[cnt].depth  = 1;
        imgResized[cnt].buff   = NULL;

        FRAWBicubicFilter bcfilter;
        FRAWResizeEngine  szf( &bcfilter );
            
        szf.scale( refimgbuf[cnt], 
                   imgYCbCr.Y.width,
                   imgYCbCr.Y.height,
                   rs_w,
                   rs_h,
                   &imgResized[cnt].buff );
    }
    
    // Release splitted image of Y-Cr-Cb --
    discardImgYCbCr( imgYCbCr );

    /******************* The First Layer *******************/

    libsrcnn::ImgConv1Layers imgConv1;
    
    libsrcnn::initImgConvLayers( &imgConv1[0], 
                                 imgResized[0].width,
                                 imgResized[0].height,
                                 CONV1_FILTERS );
        
    #pragma omp parallel for
    for ( unsigned cnt=0; cnt<CONV1_FILTERS; cnt++)
    {
        libsrcnn::convolution99( imgResized[0], 
                                 imgConv1[cnt], 
                                 weights_conv1_data[cnt], 
                                 biases_conv1[cnt] );
    }

    /******************* The Second Layer *******************/

    libsrcnn::ImgConv2Layers imgConv2;
    
    libsrcnn::initImgConvLayers( &imgConv2[0], 
                                 imgResized[0].width,
                                 imgResized[0].height,
                                 CONV2_FILTERS );
    
    #pragma omp parallel for
    for ( unsigned cnt=0; cnt<CONV2_FILTERS; cnt++ )
    {
        libsrcnn::convolution11( imgConv1, 
                                 imgConv2[cnt], 
                                 weights_conv2_data[cnt], 
                                 biases_conv2[cnt]);    
    }
    
    /******************* The Third Layer *******************/

    libsrcnn::ImgU8 imgConv3;
        
    libsrcnn::initImgU8( imgConv3, 
                         imgResized[0].width, 
                         imgResized[0].height );
    
    libsrcnn::convolution55( imgConv2, imgConv3, 
                             weights_conv3_data, 
                             biases_conv3 );
                                
    // ---------------------------------------------------------

    /* Convert the image from YCrCb to RGB Space */
    libsrcnn::ImgU8 imgRGB;
    libsrcnn::convertImgF32x3toImgU8( imgResized, imgRGB );
    
    // discard used image of Resized Y-Cr-Cb.
    libsrcnn::discardConvLayers( imgResized, 3 );
    
    // discard used buffers ..
    libsrcnn::discardConvLayers( &imgConv1[0], CONV1_FILTERS );
    libsrcnn::discardConvLayers( &imgConv2[0], CONV2_FILTERS );
    
    if ( imgRGB.buff != NULL )
    {
        outbuffsz = imgRGB.width * imgRGB.height * imgRGB.depth;
        outbuff = new unsigned char[ outbuffsz ];
        if ( outbuff != NULL )
        {
            memcpy( outbuff, imgRGB.buff, outbuffsz );
            resetImgU8( imgRGB );
            
            return 0;
        }
    }
    
    return -100;
}
#endif /// of EXPORTLIB