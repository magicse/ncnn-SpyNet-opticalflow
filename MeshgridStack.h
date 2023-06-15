#include "layer.h"

class MeshgridStack : public ncnn::Layer
{
public:
    MeshgridStack()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {   //output grid 2x8x8
        int w_in = bottom_blob.w;
        int h_in = bottom_blob.h;
        int channels = 2;
        size_t elemsize = bottom_blob.elemsize;

        //top_blob.create(w_in, h_in, channels, elemsize, opt.blob_allocator);
        top_blob.create(channels, w_in, h_in, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        float* top_data = top_blob;
        int size = w_in * h_in;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++)
        {
            int x = i % w_in;
            int y = i / w_in;

            top_data[i*channels] = static_cast<float>(x);
            top_data[i*channels+1] = static_cast<float>(y);
            //top_blob.reshape(channels, w, h, d);
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(MeshgridStack)
