#include <layer.h>
#include <cmath>
#include <algorithm>


class ScaleGridFlowY : public ncnn::Layer
{
public:
    ScaleGridFlowY()
    {
        one_blob_only = false;
    }


    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        // Get dimensions
        int w = bottom_blobs[1].w;
        int h = bottom_blobs[1].h;
        int channels = bottom_blobs[0].c;
        size_t elemsize = bottom_blobs[0].elemsize;


        // Allocate memory for the top blob
        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(w, h, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // Perform grid flow transformation
        const float* bottom_data = bottom_blobs[0];
        float* top_data = top_blob;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            for (int i = 0; i < w * h; i++)
            {
                // Compute grid flow x
                top_data[i] = 2.0 * bottom_data[i] / std::max(h - 1, 1) - 1.0;
            }

            // Move to the next channel
            bottom_data += w * h;
            top_data += w * h;
        }
        return 0;
    }
};

DEFINE_LAYER_CREATOR(ScaleGridFlowY)
