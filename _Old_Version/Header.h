#ifndef __HEADER_H_INLCUDED
#define __HEADER_H_INCLUDED

#include <cv.h>
#include <highgui.h>
#include <iostream>
using namespace std;

void ShowImgData(IplImage* src)
{
	if (src->depth == 8)
	{
		printf("Image for HEARER 10*10 \n");
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				unsigned char temp = (src->imageData + i*src->widthStep)[j];
				printf("%d\t", temp);
			}
		}
		printf("Image for TAIL 10*10 \n");
		for (int i = src->height - 10; i < src->height; i++)
		{
			for (int j = src->width - 10; j < src->width; j++)
			{
				unsigned char temp = (src->imageData + i*src->widthStep)[j];
				printf("%d\t", temp);
			}
		}
		printf("\n");
	}
	else
	{
		printf("Image for HEARER 10*10 \n");
		for (int i = 0; i < 10; i++)
		{
			for (int j = 0; j < 10; j++)
			{
				unsigned int temp = 0;
				temp += (unsigned char)(src->imageData + i*src->widthStep)[2 * j];
				temp <<= 8;
				temp += (unsigned char)(src->imageData + i*src->widthStep)[2 * j + 1];
				printf("%d\t", temp);
			}
		}
		printf("Image for TAIL 10*10 \n");
		for (int i = src->height - 10; i < src->height; i++)
		{
			for (int j = src->width - 10; j < src->width; j++)
			{
				unsigned int temp = 0;
				temp += (unsigned char)(src->imageData + i*src->widthStep)[2 * j];
				temp <<= 8;
				temp += (unsigned char)(src->imageData + i*src->widthStep)[2 * j + 1];
				printf("%d\t", temp);
			}
		}
		printf("\n");
	}
}

#endif //__HEADER_H_INCLUDED