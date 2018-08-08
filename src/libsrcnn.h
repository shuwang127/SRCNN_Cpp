#ifndef __LIBSRCNN_H__
#define __LIBSRCNN_H__

#ifdef EXPORTLIB

int ProcessSRCNN( const unsigned char* refbuff, 
                  unsigned w, unsigned h, unsigned d,
                  float muliply,
                  unsigned char* &outbuff,
                  unsigned &outbuffsz );

#endif /// of EXPORTLIB

#endif /// of __SRCNN_H__