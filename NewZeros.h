#include "layer.h"

class NewZeros : public ncnn::Layer
{
	//
	public:
    NewZeros()
    {
        one_blob_only = true;
		//support_inplace = true;
    }
	virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
	{
	    int w_in = bottom_blob.w;
        int h_in = bottom_blob.h;
        int channels = 2;
        int div = 32;
        size_t elemsize = bottom_blob.elemsize;

        //int w_up = w_in/div;
        //int h_up = h_in/div;
		int w_up = (w_in + div - 1) / div;
        int h_up = (h_in + div - 1) / div;
        //std::cout << "w_up: " << w_up << " h_up: " << h_up << std::endl;
        top_blob.create(w_up, h_up, channels, elemsize, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        // Fill the top_blob with zeros
		top_blob.fill(0.0f);
        return 0;
	}
};

DEFINE_LAYER_CREATOR(NewZeros)
