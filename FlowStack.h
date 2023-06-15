#include "layer.h"

class FlowStack : public ncnn::Layer
{
public:
    FlowStack()
    {
        one_blob_only = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {   //output grid 2x8x8
        int w_in = bottom_blobs[0].w;
        int h_in = bottom_blobs[0].h;
        int channels = 2;
        size_t elemsize = bottom_blobs[0].elemsize;

        // Allocate memory for the top blob
        //top_blob.create(w_in, h_in, channels, elemsize, opt.blob_allocator);
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(channels, w_in, h_in, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;



        float* top_data = top_blob;
        //int size = w_in * h_in;
        int size = w_in * h_in;

        const float* bottom_data0 = bottom_blobs[0];
        const float* bottom_data1 = bottom_blobs[1];

        for (int i = 0; i < size; i++)
        {
            int x = i % w_in;
            int y = i / w_in;

            top_data[i*channels] = bottom_data0[i];
            top_data[i*channels+1] = bottom_data1[i];
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(FlowStack)
