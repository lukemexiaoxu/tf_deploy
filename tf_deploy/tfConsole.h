#pragma once

#define COMPILER_MSVC
#define NOMINMAX

namespace ConsoleSpace
{
	enum ConsoleFunction
	{
		Unknown,
		LenetSingleImage,
		LenetFolderImage,
		SSDSingleImage,
		SSDFolderImage,
		SSDSingleWebServiceImage,
		SSDVideo,
		SegmentationFrontViewSingleImage,
		SegmentationFrontViewFolderImage,
		SegmentationFrontViewVideo,
		SegmentationBirdEyeSingleImage,
		SegmentationBirdEyeFolderImage,
		SegmentationVectorizationFrontViewSingleImage,
		SegmentationVectorizationFrontViewFolderImage,
		SegmentationVectorizationFrontViewAIMAP,
		SegmentationToLaneFunctionFrontViewFolderImage,
		SSDAIMAP,
		SegmentationToLaneWeiYaSDK,
		AIMapServerProcess,
	};

	enum ConsoleDisplayMode
	{
		Hidden,
		ShowImage,
	};
}