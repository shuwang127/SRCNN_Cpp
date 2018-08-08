#ifdef FORTESTINGBIN

#ifdef _WIN32
	#include <windows.h>
#endif

#include <unistd.h>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Image.H>
#include <FL/Fl_RGB_Image.H>
#include <FL/Fl_BMP_Image.H>
#include <Fl/Fl_PNG_Image.H>
#include <Fl/Fl_JPEG_Image.H>

#include <FL/images/png.h>

#include "libsrcnn.h"
#include "fl_imgtk.h"
#include "tick.h"

bool convImage( Fl_RGB_Image* src, Fl_RGB_Image* &dst )
{
    if ( src != NULL )
    {
        unsigned img_w = src->w();
        unsigned img_h = src->h();
        unsigned img_d = src->d();
        unsigned imgsz = img_w * img_h;

        uchar* cdata = NULL;

        switch( img_d )
        {
            case 1: /// single gray
                {
                    const uchar* pdata = (const uchar*)src->data()[0];
                    cdata = new uchar[ imgsz * 3 ];
                    if ( cdata != NULL )
                    {
                        #pragma omp parallel for
                        for( unsigned cnt=0; cnt<imgsz; cnt++ )
                        {
                            cdata[ cnt*3 + 0 ] = pdata[ cnt ];
                            cdata[ cnt*3 + 1 ] = pdata[ cnt ];
                            cdata[ cnt*3 + 2 ] = pdata[ cnt ];
                        }

                        dst = new Fl_RGB_Image( cdata, img_w, img_h, 3 );

                        if ( dst != NULL )
                        {
                            return true;
                        }
                    }
                }
                break;

            case 2: /// Must be RGB565
                {
                    const unsigned short* pdata = (const unsigned short*)src->data()[0];
                    cdata = new uchar[ imgsz * 3 ];
                    if ( cdata != NULL )
                    {
                        #pragma omp parallel for
                        for( unsigned cnt=0; cnt<imgsz; cnt++ )
                        {
                            cdata[ cnt*3 + 0 ] = ( pdata[ cnt ] & 0xF800 ) >> 11;
                            cdata[ cnt*3 + 1 ] = ( pdata[ cnt ] & 0x07E0 ) >> 5;
                            cdata[ cnt*3 + 2 ] = ( pdata[ cnt ] & 0x001F );
                        }

                        dst = new Fl_RGB_Image( cdata, img_w, img_h, 3 );

                        if ( dst != NULL )
                        {
                            return true;
                        }
                    }
                }
                break;
				
			case 4: /// removing alpha ...
                {
                    const unsigned short* pdata = (const unsigned short*)src->data()[0];
                    cdata = new uchar[ imgsz * 3 ];
                    if ( cdata != NULL )
                    {
                        #pragma omp parallel for
                        for( unsigned cnt=0; cnt<imgsz; cnt++ )
                        {
							float alp = (float)( pdata[ cnt * 4 + 3 ] ) / 255.f;
                            cdata[ cnt*3 + 0 ] = pdata[ cnt * 4 + 0 ] * alp;
                            cdata[ cnt*3 + 1 ] = pdata[ cnt * 4 + 1 ] * alp;
                            cdata[ cnt*3 + 2 ] = pdata[ cnt * 4 + 2 ] * alp;
                        }

                        dst = new Fl_RGB_Image( cdata, img_w, img_h, 3 );

                        if ( dst != NULL )
                        {
                            return true;
                        }
                    }
                }
                break;			

            default:
                {
                    dst = (Fl_RGB_Image*)src->copy();

                    if ( dst != NULL )
                    {
                        return true;
                    }
                }
                break;
        }
    }

    return false;
}

int testImageFile( const char* imgfp, uchar** buff,size_t* buffsz )
{
    int reti = -1;

    if ( imgfp != NULL )
    {
        FILE* fp = fopen( imgfp, "rb" );
        if ( fp != NULL )
        {
            fseek( fp, 0L, SEEK_END );
            size_t flen = ftell( fp );
            fseek( fp, 0L, SEEK_SET );

            if ( flen > 32 )
            {
                // Test
                char testbuff[32] = {0,};

                fread( testbuff, 1, 32, fp );
                fseek( fp, 0, SEEK_SET );

                const uchar jpghdr[3] = { 0xFF, 0xD8, 0xFF };

                // is JPEG ???
                if( strncmp( &testbuff[0], (const char*)jpghdr, 3 ) == 0 )
                {
                    reti = 1; /// JPEG.
                }
                else
                if( strncmp( &testbuff[1], "PNG", 3 ) == 0 )
                {
                    reti = 2; /// PNG.
                }
                else
                if( strncmp( &testbuff[0], "BM", 2 ) == 0 )
                {
                    reti = 3; /// BMP.
                }

                if ( reti > 0 )
                {
                    *buff = new uchar[ flen ];
                    if ( *buff != NULL )
                    {
                        fread( *buff, 1, flen, fp );

                        if( buffsz != NULL )
                        {
                            *buffsz = flen;
                        }
                    }
                }
            }

            fclose( fp );
        }
    }

    return reti;
}

bool savetocolorpng( Fl_RGB_Image* imgcached, const char* fpath )
{
	if ( imgcached == NULL )
		return false;

	if ( imgcached->d() < 3 )
		return false;

	FILE* fp = fopen( fpath, "wb" );
    if ( fp == NULL )
        return false;

    png_structp png_ptr     = NULL;
    png_infop   info_ptr    = NULL;
    png_bytep   row         = NULL;

    png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
    if ( png_ptr != NULL )
    {
        info_ptr = png_create_info_struct( png_ptr );
        if ( info_ptr != NULL )
        {
            if ( setjmp( png_jmpbuf( (png_ptr) ) ) == 0 )
            {
                int mx = imgcached->w();
                int my = imgcached->h();
                int pd = 3;

                png_init_io( png_ptr, fp );
                png_set_IHDR( png_ptr,
                              info_ptr,
                              mx,
                              my,
                              8,
                              PNG_COLOR_TYPE_RGB,
                              PNG_INTERLACE_NONE,
                              PNG_COMPRESSION_TYPE_BASE,
                              PNG_FILTER_TYPE_BASE);

                png_write_info( png_ptr, info_ptr );

                row = (png_bytep)malloc( imgcached->w() * sizeof( png_byte ) * 3 );
                if ( row != NULL )
                {
                    const char* buf = imgcached->data()[0];
                    int bque = 0;

                    for( int y=0; y<my; y++ )
                    {
                        for( int x=0; x<mx; x++ )
                        {
                            row[ (x*3) + 0 ] = buf[ bque + 0 ];
                            row[ (x*3) + 1 ] = buf[ bque + 1 ];
                            row[ (x*3) + 2 ] = buf[ bque + 2 ];
                            bque += pd;
                        }

                        png_write_row( png_ptr, row );
                    }

                    png_write_end( png_ptr, NULL );

                    fclose( fp );

                    free(row);
                }

                png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
                png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

                return true;
            }
        }
    }
}


int main( int argc, char** argv )
{	
	printf( "Test for SRCNN with FLTK-1.3.4-1-ts\n" );
	printf( "(C)2018 Raphael Kim\n\n" );
	fflush( stdout );
	
	const char imgtestpath[] = "Pictures/butterfly_GT.bmp";
	
	Fl_RGB_Image* imgTest = NULL;
	
	uchar* imgbuff = NULL;
	size_t imgsz = 0;
	
    int imgtype = testImageFile( imgtestpath, &imgbuff, &imgsz );
	if ( imgtype > 0 )
	{
		printf( "- Image loaded : ");
		
		switch( imgtype )
		{
			case 1: /// JPEG
				printf( "JPEG | ");
				
				imgTest = new Fl_JPEG_Image( "JPGIMG",
											(const uchar*)imgbuff );
				break;

			case 2: /// PNG
				printf( "PNG | " );
				
				imgTest = new Fl_PNG_Image( "PNGIMAGE",
										   (const uchar*)imgbuff, imgsz );
				break;

			case 3: /// BMP
				printf( "BMP | " );
				imgTest = fl_imgtk::createBMPmemory( (const char*)imgbuff, imgsz );
				break;
		}
		
		if ( imgTest != NULL )
		{
			printf( "%u x %u x %u\n", imgTest->w(), imgTest->h(), imgTest->d() );
			fflush( stdout );
		}
	}
	
	if ( imgTest != NULL )
	{
		Fl_RGB_Image* imgRGB = NULL;
		
		convImage( imgTest, imgRGB );
		
		delete imgTest;
		
		if ( ( imgRGB->w() > 0 ) && ( imgRGB->h() > 0 ) )
		{
			const uchar* refbuff = (const uchar*)imgRGB->data()[0];
			unsigned     ref_w   = imgRGB->w();
			unsigned     ref_h   = imgRGB->h();
			unsigned     ref_d   = imgRGB->d();
			float mulf = 2.0f;
			uchar*       outbuff = NULL;
			unsigned	 outsz   = 0;
			
			printf( "- Processing SRCNN ... " );
			fflush( stdout );
			
			unsigned     tick0 = tick::getTickCount();
			
			int reti = ProcessSRCNN( refbuff,
			                         ref_w,
							 		 ref_h,
									 ref_d,
						 			 mulf,
					 				 outbuff,
									 outsz );
			
			unsigned     tick1 = tick::getTickCount();
			
			if ( ( reti == 0 ) && ( outsz > 0 ) )
			{
				unsigned new_w = ref_w * mulf;
				unsigned new_h = ref_h * mulf;
				
				printf( "Test Ok, took %u ms.\n", tick1 - tick0 );
			
				Fl_RGB_Image* imgDump = new Fl_RGB_Image( outbuff, new_w, new_h, 3 );
				if ( imgDump != NULL )
				{
					savetocolorpng( imgDump, "testout.png" );
					fl_imgtk::discard_user_rgb_image( imgDump );
				}
			}
			else
			{
				printf( "Failed, error code = %d\n", reti );
			}
			
			delete imgRGB;
		}
		else
		{
			printf( "error: load failre - %s\n",  imgtestpath );
		}
	}
	else
	{
		printf( "Failed to load bitmap.\n" );
		printf( "Failed to load bitmap.\n" );
	}

	// let check memory leak before program terminated.
	printf( "- Input any number and press ENTER to terminate, check memory state.\n" );
	fflush( stdout );
	unsigned meaningless = 0;
	scanf( "%d", &meaningless );
	
	return 0;
}

#endif /// of FORTESTINGBIN