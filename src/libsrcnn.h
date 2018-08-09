#ifndef __LIBSRCNN_H__
#define __LIBSRCNN_H__

#ifdef EXPORTLIBSRCNN

int ProcessSRCNN( const unsigned char* refbuff, 
                  unsigned w, unsigned h, unsigned d,
                  float muliply,
                  unsigned char* &outbuff,
                  unsigned &outbuffsz );

#endif /// of EXPORTLIBSRCNN

#endif /// of __SRCNN_H__