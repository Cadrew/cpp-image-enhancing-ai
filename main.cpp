#include <iostream>
#include "cppflow/cppflow.h"

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
    // TODO

    return 0;
}