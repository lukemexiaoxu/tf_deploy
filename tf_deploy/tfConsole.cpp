#include "stdafx.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <Eigen\Dense>

#include "tfConsole.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"

#include <chrono>
#include <iomanip> 
#include <iostream>
#include <string>

#include "tensorflowSegment.h"

using namespace std;
using namespace ConsoleSpace;
//using namespace FunctionSpace;


using namespace tensorflow;



int tensorSegment()
{
	string modal_path = "D:/Work/Tools/TestTensorflow/TestTensorflow/ck15model.pb";
	GraphDef graph_def;         // graph from frozen protobuf
  
	SessionOptions opts;        // gpu options
	Session* session = 0;       // session to run the graph

    // graph nodes for i/o
	std::string input_node;     // name of input node in tf graph
	std::string output_node;    // name of output node in tf graph

	// read in graph
	Status status = ReadBinaryProto(Env::Default(), modal_path, &graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 1;
	}
	std::cout << "Successfully imported frozen protobuf." << std::endl;

	// Set options for session
	//string device = "/cpu:0";
	//graph::SetDefaultDevice(device, &graph_def);
    
	// Start a session
	status = NewSession(opts, &session);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 1;
	}
	std::cout << "Session successfully created.\n";

	// Add the graph to the session
	status = session->Create(graph_def);
	if (!status.ok()) {
		std::cerr << status.ToString() << std::endl;
		return 1;
	}
	std::cout << "Successfully added graph to session." << std::endl;

	// Open an image
	string image_path = "D:/Work/Tools/TestTensorflow/TestTensorflow/test.png";
	cv::Mat cv_img = cv::imread(image_path);
	if (!cv_img.data) {
		std::cerr << "Could not open or find the image" << std::endl;
		return 1;
	}
	if (!cv_img.data) {
		std::cout << "Could find content in the image" << std::endl;
		return 1;
	}

	// get the start time to report
	auto start_total = std::chrono::high_resolution_clock::now();

	// Get dimensions
	unsigned int cv_img_h = cv_img.rows;
	unsigned int cv_img_w = cv_img.cols;
	unsigned int cv_img_d = cv_img.channels();

	// Set up inputs to run the graph
	// tf tensor for feeding the graph
	Tensor x_pl(DT_FLOAT, { 1, cv_img_h, cv_img_w, cv_img_d });

	// tf pointer for init of fake cv mat
	float* x_pl_pointer = x_pl.flat<float>().data();

	// fake cv mat (avoid copy)
	cv::Mat x_pl_cv(cv_img_h, cv_img_w, CV_32FC3, x_pl_pointer);
	cv_img.convertTo(x_pl_cv, CV_32FC3);

	std::cout << "shape of input: (0):" << x_pl.shape().dim_size(0) << ",(1):" << x_pl.shape().dim_size(1)
		<< ",(2):" << x_pl.shape().dim_size(2) << ",(3):" << x_pl.shape().dim_size(3) << std::endl;

	// feed the input
	string input_layer = "IteratorGetNext:0";
	string output_layer = "y_hat/cast_pred:0";
	string is_training = "is_training:0";

	tensorflow::Tensor training(tensorflow::DT_BOOL, tensorflow::TensorShape());
	training.scalar<bool>()() = false;

	std::vector<std::pair<std::string, Tensor>> inputs = {
		{ input_layer, x_pl },{ "is_training:0",training } };

	// The session will initialize the outputs automatically
	std::vector<Tensor> outputs;

	// Run the session, evaluating our all operation from the graph
	auto start_inference = std::chrono::high_resolution_clock::now();

	Status run_status = session->Run( inputs,{ output_layer }, {}, &outputs);
	if (!run_status.ok()) {
		std::cerr << run_status.ToString() << std::endl;
		return 1;
	}
	auto elapsed_inference =
		std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start_inference).count();

	session->Close();

	// Process the output with map
	// Get output dimensions
	unsigned int output_img_n = outputs[0].shape().dim_size(0);
	unsigned int output_img_h = outputs[0].shape().dim_size(1);
	unsigned int output_img_w = outputs[0].shape().dim_size(2);
	
	std::cout << "shape of output: n:" << output_img_n << ",h:" << output_img_h
		<< ",w:" << output_img_w << std::endl;
	
	// fake cv mat (avoid copy)
	uint8* flat = outputs[0].flat<uint8>().data();	
	cv::Mat mask_argmax(output_img_h, output_img_w, CV_8U, flat);

	auto elapsed_total =
		std::chrono::duration_cast<std::chrono::milliseconds>(
			std::chrono::high_resolution_clock::now() - start_total)
		.count();

	std::cout << "Successfully run prediction from session." << std::endl;
	std::cout << "Time to infer: " << elapsed_inference << "ms." << std::endl;
	std::cout << "Time in total: " << elapsed_total << "ms." << std::endl;

	// save image
	std::string image_log_path = "D:/Work/Tools/TestTensorflow/TestTensorflow/output.png";
	std::cout << "Saving this image to " << image_log_path << std::endl;
	cv::imwrite(image_log_path, mask_argmax);

	// print the output
	cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE); 
	cv::imshow("Display window", mask_argmax);
	cv::waitKey(0);
	
	return 0;


}


// NaviKeenConsoleDemo.cpp : Defines the entry point for the console application.
//

void help_onMain()
{
	printf("\tHELP:\n");
	printf("\t\tSSD Recognize Single Image: -func SSD_S -i <file> -o <file> -GPU <gpu number> -v?\n");
	printf("\t\tSSD Recognize Multi Image: -func SSD_F -i <DIR> -o <DIR> -GPU <gpu number> -v?\n");
	printf("\t\tSSD Recognize WebService Image: -func SSD_W -u <URL> -o <file> -GPU <gpu number> -v?\n");
	printf("\t\tSSD Recognize A Video: -func SSD_V -i <file> -o <DIR> -GPU <gpu number> -v?\n");
	printf("\t\tSegmentation FrontView Single Image: -func ST_FV_S -i <FILE> -o <FILE> -f <txtFILE> -v?\n");
	printf("\t\tSegmentation FrontView Folder Image: -func ST_FV_F -i <FOLDER> -o <FOLDER> -f <txtFILE> -v?\n");
	printf("\t\tSegmentation FrontView WebService Image: -func ST_FV_W -u <URL> -o <file> -v?\n");
	printf("\t\tSegmentation FrontView Video: -func ST_FV_V -i <FILE> -o <DIR> -f <txtFILE> -v?\n");
	printf("\t\tSegmentation BirdEye Single Image: -func ST_BE_S -i <FILE> -o <FILE> -v?\n");
	printf("\t\tSegmentation BirdEye Folder Image: -func ST_BE_F -i <FOLDER> -o <FOLDER> -v?\n");
	printf("\t\tSegmentation Vectorization FrontView Single Image: -func STVEC_FV_S -i <FILE> -o <FILE> -f <txtFILE> -v?\n");
	printf("\t\tSegmentation Vectorization FrontView Folder Image: -func STVEC_FV_F -i <FOLDER> -o <FOLDER> -f <txtFILE> -l <use sdk?>\n");
	printf("\t\tAIMAP Server Rrocess: -func AIMAP_SERVER -p <pid> -s <startSeq> -e <endSeq> -c <channelcode> -t <type>\n");
}

int main(int argc, char** argv)
{

	ConsoleFunction funMode = SegmentationBirdEyeSingleImage;

	if (argc < 2)
	{
		help_onMain();
		cin.ignore();
		exit(EXIT_FAILURE);
	}

	if (!strcmp(argv[1], "-func"))
	{
		string fun = argv[2];
		if (fun == "SSD_S")
			funMode = ConsoleFunction::SSDSingleImage;
		else if (fun == "SSD_F")
			funMode = ConsoleFunction::SSDFolderImage;
		else if (fun == "SSD_V")
			funMode = ConsoleFunction::SSDVideo;
		else if (fun == "ST_FV_S")
			funMode = ConsoleFunction::SegmentationFrontViewSingleImage;
		else if (fun == "ST_FV_F")
			funMode = ConsoleFunction::SegmentationFrontViewFolderImage;
		else if (fun == "ST_FV_V")
			funMode = ConsoleFunction::SegmentationFrontViewVideo;
		else if (fun == "ST_BE_S")
			funMode = ConsoleFunction::SegmentationBirdEyeSingleImage;
		else if (fun == "ST_BE_F")
			funMode = ConsoleFunction::SegmentationBirdEyeFolderImage;
		else if (fun == "STVEC_FV_S")
			funMode = ConsoleFunction::SegmentationVectorizationFrontViewSingleImage;
		else if (fun == "STVEC_FV_F")
			funMode = ConsoleFunction::SegmentationVectorizationFrontViewFolderImage;
		else if (fun == "STVEC_FV_AIMAP")
			funMode = ConsoleFunction::SegmentationVectorizationFrontViewAIMAP;
		else if (fun == "SSD_W")
			funMode = ConsoleFunction::SSDSingleWebServiceImage;
		else if (fun == "SSD_AIMAP")
			funMode = ConsoleFunction::SSDAIMAP;
		else if (fun == "STLF_FVFI")
			funMode = ConsoleFunction::SegmentationToLaneFunctionFrontViewFolderImage;
		else if (fun == "STLF_WEIYASDK")
			funMode = ConsoleFunction::SegmentationToLaneWeiYaSDK;
		else if (fun == "AIMAP_SERVER")
			funMode = ConsoleFunction::AIMapServerProcess;
	}
	else if (!strcmp(argv[1], "-help"))
	{
		help_onMain();
		cin.ignore();
		exit(EXIT_FAILURE);
	}

	switch (funMode)
	{
	case ConsoleFunction::SSDSingleImage:
	{
		//SSDDetect ssd(argc, argv);
		//ssd.SSDSingleImageDetect();
		break;
	}

	case ConsoleFunction::SSDFolderImage:
	{
		//SSDDetect ssd(argc, argv);
		//ssd.SSDBatchImagesDetect();
		break;
	}

	case ConsoleFunction::SSDSingleWebServiceImage:
	{
		//SSDDetect ssd(argc, argv);
		//ssd.SSDSingleWebServiceImageDetect();
		break;
	}

	case ConsoleFunction::SSDVideo:
	{
		//SSDDetect ssd(argc, argv);
		//ssd.SSDVideoDetect();
		break;
	}

	case ConsoleFunction::SegmentationBirdEyeSingleImage:
	{
		string modal_path = "D:/Work/Tools/TestTensorflow/TestTensorflow/ck15model.pb";
		// Open an image
		string image_path = "D:/Work/Tools/TestTensorflow/TestTensorflow/test.png";
		cv::Mat cv_img = cv::imread(image_path);
		if (!cv_img.data) {
			std::cerr << "Could not open or find the image" << std::endl;
			return 1;
		}
		if (!cv_img.data) {
			std::cout << "Could find content in the image" << std::endl;
			return 1;
		}
		//tensorSegment();
		tensorflowSegment segment;
		segment.init(modal_path);
		cv::Mat mask;
		bool verbose = true;
		segment.segmentImg(cv_img, mask, verbose);
		break;
	}
	case ConsoleFunction::SegmentationBirdEyeFolderImage:
	{
		//SegmentationDetect seg(argc, argv);
		//seg.SegmentationBirdEyeFolderImageDetect();
		break;
	}
	case ConsoleFunction::SegmentationFrontViewSingleImage:
	{
		//SegmentationDetect seg(argc, argv);
		//seg.SegmentationFrontViewSingleImageDetect();
		break;
	}
	case ConsoleFunction::SegmentationFrontViewFolderImage:
	{
		//SegmentationDetect seg(argc, argv);
		//seg.SegmentationFrontViewFolderImageDetect();
		break;
	}
	case ConsoleFunction::SegmentationFrontViewVideo:
	{
		//SegmentationDetect seg(argc, argv);
		//seg.SegmentationFrontViewVideoDetect();
		break;
	}
	case ConsoleFunction::SegmentationVectorizationFrontViewSingleImage:
	{
		//SegmentationVectorizationFunction vecg(argc, argv);
		//vecg.SegmentationFrontViewSingleImageVectorizationGroup();
		break;
	}
	case ConsoleFunction::SegmentationVectorizationFrontViewFolderImage:
	{
		//SegmentationVectorizationFunction vecg(argc, argv);
		//vecg.SegmentationFrontViewFolderImageVectorizationGroup();
		break;
	}
	case ConsoleFunction::SegmentationVectorizationFrontViewAIMAP:
	{
		//AIMapSegmentToLaneFunction vecAiMap(argc, argv);
		//vecAiMap.Process();
		break;
	}
	case ConsoleFunction::SSDAIMAP:
	{
		//AIMapSSDFunction ssdAiMap(argc, argv);
		//ssdAiMap.Process();
		break;
	}
	case ConsoleFunction::SegmentationToLaneFunctionFrontViewFolderImage:
	{
		//SegmentToLaneFunction seg(argc, argv);
		//seg.Process();
		break;
	}
	case ConsoleFunction::SegmentationToLaneWeiYaSDK:
	{
		//WeiyaSDK weiya(argc, argv);
		//weiya.Process();
		break;
	}
	case ConsoleFunction::AIMapServerProcess:
	{
		//AIMapServerProcessFunction AiMapServer(argc, argv);
		//AiMapServer.Process();
		break;
	}
	default:
		break;
	}
	//#ifdef WIN32
	//	system("pause");
	//#endif
	return 0;
}

