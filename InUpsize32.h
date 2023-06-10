#include "layer.h"

class InUpsize32 : public ncnn::Layer
{
	public:
    InUpsize32()
    {
        // one input and one output
        // typical one_blob_only type: Convolution, Pooling, ReLU, Softmax ...
        // typical non-one_blob_only type: Eltwise, Split, Concat, Slice ...
        // one_blob_only = true;
        one_blob_only = true;
		//support_inplace = true;
    }
	virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
	{
		//
		int w_in = bottom_blob.w;
        int h_in = bottom_blob.h;
        int channels = 1;
        int mult = 32;

        size_t elemsize = bottom_blob.elemsize;
        int w_up = (w_in % mult) == 0 ? w_in : mult * ((w_in / mult) + 1);
        int h_up = (h_in % mult) == 0 ? h_in : mult * ((h_in / mult) + 1);
        top_blob.create(w_up, h_up, channels, elemsize, opt.blob_allocator);

        if (top_blob.empty())
            return -100;

        float* top_data = top_blob;

        //top_data[0] = w_up;
        //top_data[1] = h_up;
        return 0;
	}
};

DEFINE_LAYER_CREATOR(InUpsize32)
