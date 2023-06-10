#include "layer.h"

class AdjustFlow : public ncnn::Layer
{
public:
    AdjustFlow()
    {
        one_blob_only = false;
    }
    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;
        size_t elemsize = bottom_blobs[0].elemsize;
        int channels = 2;
        int w_up = bottom_blobs[1].w;
        int h_up = bottom_blobs[1].h;

        float scale_x = static_cast<float>(w) / static_cast<float>(w_up);
        float scale_y = static_cast<float>(h) / static_cast<float>(h_up);

        // Allocate memory for the top blob
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(channels, w, h, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;
        float* top_data = top_blob;
        top_blob = bottom_blobs[0];
        int size = w * h;

        for (int i = 0; i < size; i++)
        {
                top_blob[i] *= scale_x;
                top_blob[i+size] *= scale_y;
        }

        //
        return 0;
    }

};

DEFINE_LAYER_CREATOR(AdjustFlow)
