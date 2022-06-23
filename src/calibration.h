#include <glm/mat4x4.hpp>
#include <librealsense2/rs.hpp>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/aruco.hpp>

extern cv::Ptr<cv::aruco::Dictionary> openCVCalibrationDictionary;
extern cv::Ptr<cv::aruco::GridBoard> openCVCalibrationBoard;

bool tryCalibrateCameraPosition(glm::mat4& transform, rs2::video_frame& frame);