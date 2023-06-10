#include <omp.h>
#include "net.h"
#include <iostream>
#include <iomanip>

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#endif
#include <stdio.h>
#include <vector>
#include <InUpsize32.h>
#include <NewZeros.h>
#include <MeshgridStack.h>
#include <ScaleGridFlowX.h>
#include <ScaleGridFlowY.h>
#include <FlowStack.h>
#include <AdjustFlow.h>

void visualizeFlow(const ncnn::Mat& flow)
{
    // Convert flow data to OpenCV format
    cv::Mat Flow_UV(flow.h, flow.w, CV_32FC2);
    memcpy((uchar*)Flow_UV.data, flow.data, flow.w * flow.h * 2 * sizeof(float));

    cv::Mat flow_x(flow.h, flow.w, CV_32F, (float*)flow.data);
    cv::Mat flow_y(flow.h, flow.w, CV_32F, (float*)flow.data + flow.w * flow.h);

    // Calculate magnitude
    cv::Mat magnitude;
    cv::magnitude(flow_x, flow_y, magnitude);


    // Normalize magnitude to [0, 255]
    cv::Mat magnitude_normalized;
    cv::normalize(magnitude, magnitude_normalized, 0, 255, cv::NORM_MINMAX);

    // Convert magnitude to uint8
    cv::Mat magnitude_uint8;
    magnitude_normalized.convertTo(magnitude_uint8, CV_8U);

    // Display the flow visualization
    cv::imshow("Flow Visualization", magnitude_uint8);
    cv::waitKey(0);
}
