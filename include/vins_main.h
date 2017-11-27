#include<chrono>
#include <Eigen/Dense>
#include "feature_tracker.h"
#include "parameters.h"

class ProcessImage
{
public:
	void processImage();
	bool grabAllCameras(VideoCapture* caps,cv::Mat* frames);

private:
	double m_d_first_image_time = 0.0;
	bool m_b_first_image_flag = true;
	int m_pub_count = 1;
	FeatureTracker trackerData[NUM_OF_CAM];

	//for stereo
	vector<uchar> r_status;
	vector<float> r_err;
};

struct IMU_MSG {
	double header;
	Eigen::Vector3d acc;
	Eigen::Vector3d gyr;
};

struct IMG_MSG {
	double header;
	std::map<int, Eigen::Vector3d> point_clouds;
};

typedef shared_ptr <IMG_MSG const > ImgConstPtr;
typedef shared_ptr <IMU_MSG const > ImuConstPtr;