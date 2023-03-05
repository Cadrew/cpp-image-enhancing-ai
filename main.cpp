#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "cppflow/cppflow.h"
#include <tensorflow/core/public/session.h>

int main() {
    // Read the graph
    cppflow::model model("models/regular");

    // Load an image
    auto input = cppflow::decode_jpeg(cppflow::read_file(std::string("input.png"))); // 640x444

    // Cast it to float, normalize to range [0, 1], and add batch_dimension
    input = cppflow::cast(input, TF_UINT8, TF_FLOAT);
    input = input / 255.f;
    input = cppflow::expand_dims(input, 0);

    // Run
    // auto output = model(input);
    auto output = model({{"serving_default_input_0:0", input}},{"StatefulPartitionedCall:0"});

    // Show the predicted class
    // std::cout << "Content of output:" << output[0] << std::endl;
    std::cout << cppflow::arg_max(output[0], 1) << std::endl;

    // Save it into an image    
    std::vector<float> output_data = output[0].get_data<float>(); 
    const auto* output_tensor = &output[0];
    int rows = 444 * 4;
    int cols = 640 * 4;
    std::cout << output_data.size() << std::endl;
    // if(output_data.size() == rows*cols) // check that the rows and cols match the size of your vector
    // {
        cv::Mat image = cv::Mat(rows, cols, CV_8UC3); // initialize matrix of uchar of 1-channel where you will store vec data

        //copy vector to mat
        // memcpy(image.data, output.data(), output.size()*sizeof(cppflow::tensor));
        memcpy(image.data, output_data.data(), output_data.size()*sizeof(uchar)); // change uchar to any type of data values that you want to use instead
        // memcpy(image.data, output_tensor, rows*cols);
        // cv::imshow("Display image", image);
        cv::imwrite("output.png", image);
    // }
}