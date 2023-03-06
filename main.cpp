#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include "cppflow/cppflow.h"
// #include <tensorflow/cc/client/client_session.h>
// #include <tensorflow/cc/ops/standard_ops.h>
// #include <tensorflow/cc/saved_model/loader.h>
// #include <tensorflow/core/framework/tensor.h>

// #include <tensorflow/core/public/session.h>
// #include <tensorflow/core/platform/env.h>

namespace tf = tensorflow;

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

// int test() {
//     // Chargement du modèle TensorFlow sauvegardé
//     tf::GraphDef graph_def;
//     tf::Status status = tf::ReadBinaryProto(tf::Env::Default(), "models/regular/saved_model.pb", &graph_def);
//     if (!status.ok()) {
//         std::cerr << "Erreur lors du chargement du modèle: " << status.ToString() << std::endl;
//         return 1;
//     }
//     tf::SessionOptions session_options;
//     // tf::RunOptions run_options;
//     std::unique_ptr<tf::Session> session(tf::NewSession(session_options));
//     status = session->Create(graph_def);
//     if (!status.ok()) {
//         std::cerr << "Erreur lors de la création de la session: " << status.ToString() << std::endl;
//         return 1;
//     }

//     // Chargement de l'image à traiter avec OpenCV
//     cv::Mat input_image = cv::imread("input.png");
//     if (input_image.empty()) {
//         std::cerr << "Erreur lors du chargement de l'image d'entrée" << std::endl;
//         return 1;
//     }
//     int height = input_image.rows;
//     int width = input_image.cols;
//     int channels = input_image.channels();

//     // Convertir l'image en un tenseur TensorFlow
//     tf::Tensor input_tensor(tf::DT_FLOAT, {1, height, width, channels});
//     float* input_data = input_tensor.flat<float>().data();
//     cv::Mat input_image_float;
//     input_image.convertTo(input_image_float, CV_8UC3, 1.0f / 255.0f);
//     std::memcpy(input_data, input_image_float.data, height * width * channels * sizeof(float));

//     // Préparation des opérations TensorFlow pour effectuer une inférence sur l'image
//     // Ajouter les opérations nécessaires pour effectuer une inférence sur l'image
//     std::vector<std::pair<std::string, tf::Tensor>> inputs = {{"serving_default_input_0:0", input_tensor}};
//     std::vector<std::string> output_node_names = {"StatefulPartitionedCall:0"};
//     std::vector<tf::Tensor> output_tensors;

//     // Exécution de l'inférence sur l'image et récupération de l'image traitée
//     status = session->Run(inputs, output_node_names, {}, &output_tensors);
//     if (!status.ok()) {
//         std::cerr << "Erreur lors de l'exécution de l'inférence: " << status.ToString() << std::endl;
//         return 1;
//     }
//     tf::Tensor output_tensor = output_tensors[0];

//     // Convertir le tenseur TensorFlow en une image OpenCV
//     // cv::Mat output_image(height, width, CV_8UC3, output_tensor.flat<float>().data());
//     // output_image.convertTo(output_image
//     return 0;
// }

// int test() {
//     // Load image
//     cv::Mat image = cv::imread("input.png", cv::IMREAD_COLOR);

//     // Setup TensorFlow to use GPU
//     Session* session;
//     SessionOptions options;
//     GPUOptions gpu_options;
//     // TODO: make it work
//     // gpu_options.mutable_per_process_gpu_memory_fraction()->set_allocated_fraction(0.5);
//     // options.config.mutable_gpu_options()->CopyFrom(gpu_options);
//     Status status = NewSession(options, &session);
//     if (!status.ok()) {
//         std::cerr << "Erreur lors de la création de la session TensorFlow: " << status.ToString() << std::endl;
//         return -1;
//     }

//     // Load ESRGAN model
//     std::string model_path = "models/regular/saved_model.pb";
//     GraphDef graph_def;
//     status = ReadBinaryProto(Env::Default(), model_path, &graph_def);
//     if (!status.ok()) {
//         std::cerr << "Erreur lors de la lecture du modèle ESRGAN: " << status.ToString() << std::endl;
//         return -1;
//     }

//     // Add ESRGAN model to TensorFlow graph
//     std::unique_ptr<Session> esrgan_session(NewSession(SessionOptions()));
//     status = esrgan_session->Create(graph_def);
//     if (!status.ok()) {
//         std::cerr << "Erreur lors de la création de la session ESRGAN: " << status.ToString() << std::endl;
//         return -1;
//     }

//     // Transform image to a TensorFlow tensor
//     Tensor input_tensor(DT_UINT8, TensorShape({1, image.rows, image.cols, image.channels()}));
//     auto input_tensor_mapped = input_tensor.tensor<uint8_t, 4>();
//     cv::Mat image_float;
//     image.convertTo(image_float, CV_32F, 1.0/255.0);
//     for (int y = 0; y < image.rows; ++y) {
//         const auto* row_ptr = image_float.ptr<float>(y);
//         for (int x = 0; x < image.cols; ++x) {
//             for (int c = 0; c < image.channels(); ++c) {
//                 input_tensor_mapped(0, y, x, c) = static_cast<uint8_t>(255.0 * row_ptr[x * image.channels() + c]);
//             }
//         }
//     }

//     // Prepare ESRGAN model inputs and outputs
//     std::vector<std::pair<string, Tensor>> inputs = {{"serving_default_input_0:0", input_tensor}};
//     std::vector<string> output_node_names = {"StatefulPartitionedCall:0"};
//     std::vector<Tensor> output_tensors;

//     // Run ESRGAN model
//     status = esrgan_session->Run(inputs, output_node_names, {}, &output_tensors);
//     if (!status.ok()) {
//         std::cerr << "Erreur lors de l'exécution du modèle ESRGAN: " << status.ToString() << std::endl;
//         return -1;
//     }

//     // Get an OpenCV image
//     const auto& output_tensor = output_tensors[0];
//     cv::Mat output_image(image.rows * 4, image.cols * 4, CV_8UC3);
//     cv::imwrite("output.png", output_image);
//     return 0;
// }