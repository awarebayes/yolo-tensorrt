/**
MIT License

Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*
*/
#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "trt_utils.h"

struct BBoxInfo;

class DsImage
{
public:
    DsImage() = default;
    DsImage(const std::string& path, const std::string &s_net_type_, const int& inputH, const int& inputW);
    DsImage(const cv::Mat& mat_image_, const std::string &s_net_type_, const int& inputH, const int& inputW);
    int getImageHeight() const { return m_Height; }
    int getImageWidth() const { return m_Width; }
    cv::Mat& getLetterBoxedImage() { return m_LetterboxImage; }
    cv::Mat getOriginalImage() const { return m_OrigImage; }
    std::string getImageName() const { return m_ImageName; }
    void addBBox(BBoxInfo box, const std::string& labelName);
    void showImage() const;
    void saveImageJPEG(const std::string& dirPath) const;
    std::string exportJson() const;
	void letterbox(const int& inputH, const int& inputW);

private:
    int m_Height = 0;
    int m_Width = 0;
    int m_XOffset = 0;
    int m_YOffset = 0;
    float m_ScalingFactor = 0.0f;
    std::string m_ImagePath;
    cv::RNG m_RNG { cv::RNG(unsigned(std::time(0))) };
    std::string m_ImageName;
    std::vector<BBoxInfo> m_Bboxes;

    // unaltered original Image
    cv::Mat m_OrigImage;
    // letterboxed Image given to the network as input
    cv::Mat m_LetterboxImage;
    // final image marked with the bounding boxes
    cv::Mat m_MarkedImage;
};

class CudaPipeline
{
public:
    CudaPipeline(const std::string &s_net_type, unsigned char *gpu_blob, const int& inputH_, const int& inputW_);
    int getImageHeight() const { return m_Height; }
    int getImageWidth() const { return m_Width; }
    cv::cuda::GpuMat& getResult() { await(); return *m_Float; }
    std::shared_ptr<cv::cuda::GpuMat> getOriginalImage() { await(); return m_OrigImage; }
    void preprocess(const cv::Mat &image);
    void preprocess(const cv::cuda::HostMem &image);
    void await() { m_Stream.waitForCompletion(); };

private:
    int m_Height = 0;
    int m_Width = 0;
    int m_XOffset = 0;
    int m_YOffset = 0;
    float m_ScalingFactor = 0.0f;
    std::string m_ImagePath;
    cv::RNG m_RNG { cv::RNG(unsigned(std::time(0))) };
    std::string m_ImageName;
    std::vector<BBoxInfo> m_Bboxes;
    std::string s_net_type_;
    const int inputH;
    const int inputW;

    // unaltered original Image
    std::shared_ptr<cv::cuda::GpuMat> m_OrigImage;
    std::shared_ptr<cv::cuda::GpuMat> m_LetterboxImage;
    std::shared_ptr<cv::cuda::GpuMat> m_Float;
    std::vector<cv::cuda::GpuMat> m_chw;
    cv::cuda::Stream m_Stream;
};

#endif
