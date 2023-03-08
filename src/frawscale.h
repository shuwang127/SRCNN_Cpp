#ifndef __RAWSCALE_H__
#define __RAWSCALE_H__

#include <cstddef>
#include <cmath>
#include <cstring>

////////////////////////////////////////////////////////////////////////////////
//
// F-RAWSCALE ( Part of librawprocessor project )
// - https://github.com/rageworx/librawprocessor
// ============================================================================
// Reprogrammed by Raphael Kim (rageworx@gmail.com) for librawprocessor.
//
// * This class was belongs to below project:
//
//    ==========================================================
//    FreeImage 3
//    ----------------------------------------------------------
//    Design and implementation by
//    - Floris van den Berg (flvdberg@wxs.nl)
//    - Herv?Drolon (drolon@infonie.fr)
//
//    ==========================================================
//
// * History of updates *
//
// [2016-11-11]
//   - Modified for librawprocessor.
//
// [2018-08-08]
//   - Modified for libsrcnn, processing float vectors.
//   - Removed some filters : Lanczos3, B-Spline, Blackman
//
////////////////////////////////////////////////////////////////////////////////

// Filters
class FRAWGenericFilter
{
    protected:
        #define FILTER_PI  double (3.1415926535897932384626433832795)
        #define FILTER_2PI double (2.0 * 3.1415926535897932384626433832795)
        #define FILTER_4PI double (4.0 * 3.1415926535897932384626433832795)

    protected:
        double  _dWidth;

    public:
        FRAWGenericFilter (double dWidth) : _dWidth (dWidth) {}
        virtual ~FRAWGenericFilter() {}

        double GetWidth()                   { return _dWidth; }
        void   SetWidth (double dWidth)     { _dWidth = dWidth; }
        virtual double Filter (double dVal) = 0;
};

class FRAWBoxFilter : public FRAWGenericFilter
{
    public:
        // Default fixed width = 0.5
        FRAWBoxFilter() : FRAWGenericFilter(0.5) {}
        virtual ~FRAWBoxFilter() {}

    public:
        double Filter (double dVal)
        { return ( fabs(dVal) <= _dWidth ? 1.0 : 0.0 ); }
};

class FRAWBilinearFilter : public FRAWGenericFilter
{
    public:
        FRAWBilinearFilter () : FRAWGenericFilter(1) {}
        virtual ~FRAWBilinearFilter() {}

    public:
        double Filter (double dVal)
        {
            dVal = fabs( dVal );
            return ( dVal < _dWidth ? _dWidth - dVal : 0.0 );
        }
};

class FRAWBicubicFilter : public FRAWGenericFilter
{
    protected:
        // data for parameterized Mitchell filter
        double p0, p2, p3;
        double q0, q1, q2, q3;

    public:
        // Default fixed width = 2
        FRAWBicubicFilter ( double b = ( 1 / (double)3 ), double c = ( 1 / (double)3 ) )
         : FRAWGenericFilter(2)
        {
            p0 = (   6 - 2 * b ) / 6;
            p2 = ( -18 + 12 * b + 6 * c ) / 6;
            p3 = (  12 - 9 * b - 6 * c ) / 6;
            q0 = (   8 * b + 24 * c ) / 6;
            q1 = ( -12 * b - 48 * c ) / 6;
            q2 = (   6 * b + 30 * c ) / 6;
            q3 = (  -b - 6 * c ) / 6;
        }
        virtual ~FRAWBicubicFilter() {}

    public:
        double Filter(double dVal)
        {
            dVal = fabs(dVal);

            if(dVal < 1)
                return ( p0 + dVal * dVal * ( p2 + dVal * p3 ) );

            if(dVal < 2)
                return ( q0 + dVal * ( q1 + dVal * ( q2 + dVal * q3 ) ) );

            return 0;
        }
};


////////////////////////////////////////////////////////////////////////////////
// Resize relations.

class FRawScaleWeightsTable
{
    typedef struct
    {
        double*     Weights;
        unsigned    Left;
        unsigned    Right;
    }Contribution;

    private:
        Contribution*   _WeightTable;
        unsigned        _WindowSize;
        unsigned        _LineLength;

    public:
        FRawScaleWeightsTable( FRAWGenericFilter* pFilter = NULL, 
		                       unsigned uDstSize = 0, 
							   unsigned uSrcSize = 0 );
        ~FRawScaleWeightsTable();

    public:
        double   getWeight( unsigned dst_pos, unsigned src_pos );
        unsigned getLeftBoundary( unsigned dst_pos );
        unsigned getRightBoundary( unsigned dst_pos );
};

class FRAWResizeEngine
{
    private:
        FRAWGenericFilter* _pFilter;

    public:
        FRAWResizeEngine( FRAWGenericFilter* filter = NULL );
        virtual ~FRAWResizeEngine() {}

    public:
        unsigned scale( const float* src, unsigned src_width, unsigned src_height,
                        unsigned dst_width, unsigned dst_height, float** dst );

    private:
        void horizontalFilter( const float* src, const unsigned height, const unsigned src_width,
                               const unsigned src_offset_x, const unsigned src_offset_y,
                               float* dst, const unsigned dst_width);
        void verticalFilter( const float* src, const unsigned width, const unsigned src_height,
                             const unsigned src_offset_x, const unsigned src_offset_y,
                             float* dst, const unsigned dst_width, const unsigned dst_height);
};


#endif /// of __RAWSCALE_H__
