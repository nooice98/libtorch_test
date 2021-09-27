#include <torch/script.h>
#include <opencv2/opencv.hpp>

//对图片首先进行处理，返回张量
torch::Tensor process( cv::Mat& image,torch::Device device,int img_size)
{
    //首先对输入的图片进行处理
    cv::cvtColor(image, image, CV_BGR2RGB);// bgr -> rgb
    // image.convertTo(image, CV_32FC3);
    cv::resize(image, image, cv::Size(img_size, img_size));

    std::vector<int64_t> dims = {1, img_size, img_size, 3};

    torch::Tensor img_var = torch::from_blob(image.data, dims, torch::kByte).to(device);//将图像转化成张量
    img_var = img_var.permute({0,3,1,2});//将张量的参数顺序转化为 torch输入的格式 N*C*H*W
    img_var = img_var.toType(torch::kFloat32);
    img_var = img_var.div(1.0);
    
    return img_var;
 
}

torch::Tensor process( cv::Mat& image,torch::Device device)
{
    //首先对输入的图片进行处理
    cv::cvtColor(image, image, CV_BGR2RGB);// bgr -> rgb
    // image.convertTo(image, CV_32FC3);

    std::vector<int64_t> dims = {1, image.rows, image.cols, 3};

    torch::Tensor img_var = torch::from_blob(image.data, dims, torch::kByte).to(device);//将图像转化成张量
    img_var = img_var.permute({0,3,1,2});//将张量的参数顺序转化为 torch输入的格式 N*C*H*W
    img_var = img_var.toType(torch::kFloat32);
    img_var = img_var.div(1.0);
    
    return img_var;
 
}

void save_image(const torch::Tensor image, const std::string path, const std::pair<float, float> range, const size_t cols, const size_t padding, const size_t bits=8){

    // (0) Initialization and Declaration
    size_t i, j, k, l;
    size_t i_dev, j_dev;
    size_t width, height, channels, mini_batch_size;
    size_t width_out, height_out;
    size_t ncol, nrow;
    int mtype_in, mtype_out;
    cv::Mat float_mat, normal_mat, bit_mat;
    cv::Mat sample, output;
    std::vector<cv::Mat> samples;
    torch::Tensor tensor_sq, tensor_per, tensor_con;

    // (1) Get Tensor Size
    mini_batch_size = image.size(0);
    channels = image.size(1);
    height = image.size(2);
    width = image.size(3);

    // (2) Judge the number of channels and bits
    if (channels == 1){
        mtype_in = CV_32FC1;
        if (bits == 8){
            mtype_out = CV_8UC1;
        }
        else if (bits == 16){
            mtype_out = CV_16UC1;
        }
        else{
            std::cerr << "Error : Bits of the image to be saved is inappropriate." << std::endl;
            std::exit(1);
        }
    }
    else if (channels == 3){
        mtype_in = CV_32FC3;
        if (bits == 8){
            mtype_out = CV_8UC3;
        }
        else if (bits == 16){
            mtype_out = CV_16UC3;
        }
        else{
            std::cerr << "Error : Bits of the image to be saved is inappropriate." << std::endl;
            std::exit(1);
        }
    }
    else{
        std::cerr << "Error : Channels of the image to be saved is inappropriate." << std::endl;
        std::exit(1);
    }

    // (3) Add images to the array
    i = 0;
    samples = std::vector<cv::Mat>(mini_batch_size);
    auto mini_batch = image.clamp(/*min=*/range.first, /*max=*/range.second).to(torch::kCPU).chunk(mini_batch_size, /*dim=*/0);  // {N,C,H,W} ===> {1,C,H,W} + {1,C,H,W} + ...
    for (auto &tensor : mini_batch){
        tensor_sq = torch::squeeze(tensor, /*dim=*/0);  // {1,C,H,W} ===> {C,H,W}
        tensor_per = tensor_sq.permute({1, 2, 0});  // {C,H,W} ===> {H,W,C}
        tensor_con = tensor_per.contiguous();
        float_mat = cv::Mat(cv::Size(width, height), mtype_in, tensor_con.data_ptr<float>());  // torch::Tensor ===> cv::Mat
        normal_mat = (float_mat - range.first) / (float)(range.second - range.first);  // [range.first, range.second] ===> [0,1]
        bit_mat = normal_mat * (std::pow(2.0, bits) - 1.0);  // [0,1] ===> [0,255] or [0,65535]
        bit_mat.convertTo(sample, mtype_out);  // {32F} ===> {8U} or {16U}
        if (channels == 3){
            cv::cvtColor(sample, sample, cv::COLOR_RGB2BGR);  // {R,G,B} ===> {B,G,R}
        }
        sample.copyTo(samples.at(i));
        i++;
    }

    // (4) Output Image Information
    ncol = (mini_batch_size < cols) ? mini_batch_size : cols;
    width_out = width * ncol + padding * (ncol + 1);
    nrow = 1 + (mini_batch_size - 1) / ncol;
    height_out =  height * nrow + padding * (nrow + 1);

    // (5) Value Substitution for Output Image
    output = cv::Mat(cv::Size(width_out, height_out), mtype_out, cv::Scalar::all(0));
    for (l = 0; l < mini_batch_size; l++){
        sample = samples.at(l);
        i_dev = (l % ncol) * width + padding * (l % ncol + 1);
        j_dev = (l / ncol) * height + padding * (l / ncol + 1);
        for (j = 0; j < height; j++){
            for (i = 0; i < width; i++){
                for (k = 0; k < sample.elemSize(); k++){
                    output.data[(j + j_dev) * output.step + (i + i_dev) * output.elemSize() + k] = sample.data[j * sample.step + i * sample.elemSize() + k];
                }
            }
        }
    }

    // (6) Image Output
    cv::imwrite(path, output);

    // End Processing
    return;

}