#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <string>
//https://pytorch.org/tutorials/advanced/cpp_export.html
#include "utils.cpp"

std::string image_path = "/home/chengq/libtorch_test/test.jpg";
std::string pretrained_model_path = "/home/chengq/test_model.pt";
std::string fname = "../out.jpg";
constexpr std::pair<float, float> range = {0, 255};
auto device = at::kCUDA;
int upscale = 4;
int bits = 8;

int main(int argc, const char* argv[]) {

    cv::Mat floatMat, resultImg;
    int height, width;

    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::cout << "loading model...\n";
    torch::jit::script::Module module = torch::jit::load(pretrained_model_path);
    module.to(device);
    module.eval();
    std::cout << "loading model done.\n";

    //输入图像
    std::cout << "processing image...\n";
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    std::cout << "input image size: " << image.size() << std::endl;
    torch::Tensor input = process(image, device);
    std::cout << "processing image done\n";

    // 网络前向计算
    // Execute the model and turn its output into a tensor.
    std::cout << "forwarding...\n";
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    torch::Tensor output = module.forward({input}).toTensor();  //前向传播获取结果
    std::cout << "output tensor shape: " << output.sizes() << std::endl;
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::cout << "forwarding done\n";
    std::cout << "Forwarding time = " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000000.0 << " sec" <<std::endl;

    height = output.size(2);
    width = output.size(3);
    // 将输出转为Mat并保存
    output = output.clamp_(range.first, range.second).to(torch::kCPU).squeeze().permute({1,2,0}).contiguous();
    floatMat = cv::Mat(cv::Size(width, height), CV_32FC3, output.data_ptr<float>());  // torch::Tensor ===> cv::Mat
    floatMat.convertTo(resultImg, CV_8UC3);  // {32F} ===> {8U} or {16U}
    cv::cvtColor(resultImg, resultImg, cv::COLOR_RGB2BGR);  // {R,G,B} ===> {B,G,R}
    cv::imwrite(fname, resultImg);

    // save_image(output.detach(), fname, /*range=*/range, /*cols=*/1, /*padding=*/0);

    

}