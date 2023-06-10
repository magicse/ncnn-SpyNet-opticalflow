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

static int opticalflow(const cv::Mat& inframe0, const cv::Mat& inframe1)
//static int opticalflow()
{
	ncnn::Net net;
	net.opt.num_threads = 4;
	net.opt.use_vulkan_compute = true;
	net.register_custom_layer("InUpsize32", InUpsize32_layer_creator);
	net.register_custom_layer("NewZeros", NewZeros_layer_creator);
	net.register_custom_layer("MeshgridStack", MeshgridStack_layer_creator);
	net.register_custom_layer("ScaleGridFlowX", ScaleGridFlowX_layer_creator);
	net.register_custom_layer("ScaleGridFlowY", ScaleGridFlowY_layer_creator);
	net.register_custom_layer("FlowStack", FlowStack_layer_creator);
	net.register_custom_layer("AdjustFlow", AdjustFlow_layer_creator);

	if (net.load_param("./models/SpyNet.ncnn.param"))
		exit(-1);
	if (net.load_model("./models/SpyNet.ncnn.bin"))
		exit(-1);

    cv::Mat frame0, frame1;
    //default memory order is HWC which means that the offset is computed as offset = h * im.rows * im.elemSize() + w * im.elemSize() + c
    frame0 = inframe0.clone();
    frame1 = inframe1.clone();
    //normalize levels
    frame0.convertTo(frame0, CV_32F, 1.0/255);
    //printMat(frame0);
    frame1.convertTo(frame1, CV_32F, 1.0/255);

    // cv::Mat (h, w, CV_32FC3) to NCNN::MAT;
    ncnn::Mat Frm0, Frm1;
    ncnn::Mat in_fr0_pack3(frame0.cols, frame0.rows, 1, (void*)frame0.data, (size_t)4u * 3, 3);
    ncnn::convert_packing(in_fr0_pack3, Frm0, 1);

    ncnn::Mat in_fr1_pack3(frame1.cols, frame1.rows, 1, (void*)frame1.data, (size_t)4u * 3, 3);
    ncnn::convert_packing(in_fr1_pack3, Frm1, 1);


    Frm0 = Frm0.clone();
    Frm1 = Frm1.clone();

    std::cout << "Frm1 shape (" << Frm1.dims << ", " << Frm1.c << ", " << Frm1.w << ", " << Frm1.h << ")" << std::endl;

    float mean_data[] = {0.485f, 0.456f, 0.406f};
    float std_data[] = {0.229f, 0.224f, 0.225f};

    ncnn::Mat mean_value(1, 1, 3);
    mean_value = ncnn::Mat(3, 1, 1, mean_data).reshape(1, 1, 3);
    ncnn::Mat std_value(1, 1, 3);
    std_value = ncnn::Mat(3, 1, 1, std_data).reshape(1, 1, 3);

    ncnn::Extractor ex = net.create_extractor();

    ex.input("mean", mean_value);
    ex.input("std", std_value);
    ex.input("7", mean_value);
    ex.input("5", std_value);

    //set input, output lyers
    ex.input("in0", Frm0);
    //set input, output lyers
    ex.input("in1", Frm1);

    //inference network
    ncnn::Mat out;
    ex.extract("adjout", out);
    std::cout << "Out matrix size W x H = " << out.w << " x "<< out.h <<" number of channels " << out.c <<std::endl;
    visualizeFlow(out);

    return 0;
}

int spynet_main(int argc, char** argv)
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath0 = argv[1];
    const char* imagepath1 = argv[3];
    printf("imagepath0: %s\n", imagepath0);
    printf("imagepath1: %s\n", imagepath1);
    printf("argv[0]: %s\n", argv[0]);
    printf("argv[1]: %s\n", argv[1]);
    printf("argv[2]: %s\n", argv[2]);
    printf("argv[3]: %s\n", argv[3]);

    cv::Mat fr0 = cv::imread(imagepath0, 1);
    if (fr0.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath0);
        return -1;
    }

    cv::Mat fr1 = cv::imread(imagepath1, 1);
    if (fr1.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath1);
        return -1;
    }

	opticalflow(fr0, fr1);

    return 0;
}
