#include<chrono>
#include <eigen3/Eigen/Dense>
#include <opencv2\opencv.hpp>

#include "feature_tracker.h"
#include "parameters.h"
#include "estimator.h"

class ProcessImage
{
public:
	void process();
	bool grabAllCameras(cv::VideoCapture* caps,cv::Mat* frames);

private:
	double m_d_first_image_time = 0.0;
	bool m_b_first_image_flag = true;
	int m_pub_count = 1;
	FeatureTracker trackerData[NUM_OF_CAM];

	//for stereo
	vector<uchar> r_status;
	vector<float> r_err;
};

class ProcessIMU
{
public:
	void process();
};

struct IMU_MSG {
	double header;
	Eigen::Vector3d acc;
	Eigen::Vector3d gyr;
};

struct IMG_MSG {
	double header;
	std::vector<int> id_of_point;
	std::vector<Eigen::Vector3d> point_clouds; //id and corrsponding pts
};

typedef shared_ptr <IMG_MSG const > ImgConstPtr;
typedef shared_ptr <IMU_MSG const > ImuConstPtr;

class VIO
{
public:
	VIO();
	void processVIO();

private:
	/*
	Send imu data and visual data into VINS
	*/
	std::vector<std::pair<std::vector<ImuConstPtr>, ImgConstPtr>> getMeasurements();
	void send_imu(const ImuConstPtr &imu_msg);
private:
	double current_time = -1.0;
	Estimator* estimator;

	std::mutex i_buf;
	std::mutex m_retrive_data_buf;

	queue<RetriveData> retrive_data_buf;
};