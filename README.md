# SRCNN OpenCV GCC
This project was forked from https://github.com/shuwang127/SRCNN_Cpp, An Open source project of **"C++ Implementation of Super-Resolution resizing with Convolutional Neural Network"**.


### Introduction
This is an open source project from original of this:
**SRCNN_Cpp** is a C++ Implementation of Image Super-Resolution using SRCNN which is proposed by Chao Dong in 2014.
 - If you want to find the details of SRCNN algorithm, please read the paper:  

   Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Learning a Deep Convolutional Network for Image Super-Resolution, in Proceedings of European Conference on Computer Vision (ECCV), 2014
 - If you want to download the training code(caffe) or test code(Matlab) for SRCNN, please open your browse and visit http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html for more details.
 - And thank you very much for Chao's work in SRCNN.

### What changed ?
1. Will support FLTK GUI
1. Code modified many things from original.
1. Supports almost of platform - POSIX compatibled.
    - MSYS2 and MinGW-W64
    - GCC of Linux
    - LLVM or CLANG of MacOSX.

### License
Follows original SRCNN_Cpp, and it is released under the GPL v2 License (refer to the LICENSE file for details).

### Requirements
1. Windows may need [MSYS2](https://www.msys2.org/) and [MinGW-W64](https://github.com/msys2/msys2/wiki/MSYS2-installation).
1. You need to install latest version of *OpenCV* to your build environments,
   install opencv libraries into your system with one of these:
    - MSYS2: ```pacman -S /mingw-w64-x86_64-opencv```
    - Debian: ```sudo apt-get install libopencv-dev```
    - MacOS
        1. Xcode : 
        ```
        sudo xcode-select --install 
        sudo xcodebuild -license
        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
        ```
        1. Brew :
        ```
        brew update
        brew install opencv3
        ```

### Compile
You can compile the C/C++ files on the command line in your POSIX shell. 

``` Shell
make
```
If the compile is successful, you will see linked binary in 'bin' directory.

