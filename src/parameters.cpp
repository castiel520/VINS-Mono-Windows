#include "parameters.h"

std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;

double FOCUS_LENGTH_Y;
double PY;
double FOCUS_LENGTH_X;
double PX;

int MAX_CNT;
int MIN_DIST;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;
int ESTIMATE_EXTRINSIC;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_FOLDER_PATH;
std::string VINS_RESULT_PATH;
int LOOP_CLOSURE = 0;
std::string VOC_FILE;
std::string PATTERN_FILE;
int MIN_LOOP_NUM;
int NUM_ITERATIONS;
double SOLVER_TIME;
double MIN_PARALLAX;
double INIT_DEPTH;

double ACC_N, ACC_W;
double GYR_N, GYR_W;

Eigen::Vector3d G{ 0.0, 0.0, 9.8 };

void readParameters()
{
	std::string config_file;
	config_file = "../config/sample.yaml" ;
	cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
	if (!fsSettings.isOpened())
	{
		std::cerr << "ERROR: Wrong path to settings" << std::endl;
	}


	MAX_CNT = fsSettings["max_cnt"];
	MIN_DIST = fsSettings["min_dist"];
	ROW = fsSettings["image_height"];
	COL = fsSettings["image_width"];
	FREQ = fsSettings["freq"];
	F_THRESHOLD = fsSettings["F_threshold"];
	SHOW_TRACK = fsSettings["show_track"];
	EQUALIZE = fsSettings["equalize"];
	FISHEYE = fsSettings["fisheye"];
	if (FISHEYE == 1)
		FISHEYE_MASK = "../config/fisheye_mask.jpg";
	CAM_NAMES.push_back(config_file);

	STEREO_TRACK = false;
	FOCAL_LENGTH = 460;
	PUB_THIS_FRAME = false;

	if (FREQ == 0)
		FREQ = 100;

	ESTIMATE_EXTRINSIC = 2;
	fsSettings.release();

	if (ESTIMATE_EXTRINSIC == 2)
	{
		RIC.push_back(Eigen::Matrix3d::Identity());
		TIC.push_back(Eigen::Vector3d::Zero());
		fsSettings["ex_calib_result_path"] >> EX_CALIB_RESULT_PATH;
		EX_CALIB_RESULT_PATH = VINS_FOLDER_PATH + EX_CALIB_RESULT_PATH;

	}
	else
	{
		if (ESTIMATE_EXTRINSIC == 1)
		{
			fsSettings["ex_calib_result_path"] >> EX_CALIB_RESULT_PATH;
			EX_CALIB_RESULT_PATH = VINS_FOLDER_PATH + EX_CALIB_RESULT_PATH;
		}

		cv::Mat cv_R, cv_T;
		fsSettings["extrinsicRotation"] >> cv_R;
		fsSettings["extrinsicTranslation"] >> cv_T;
		Eigen::Matrix3d eigen_R;
		Eigen::Vector3d eigen_T;
		cv::cv2eigen(cv_R, eigen_R);
		cv::cv2eigen(cv_T, eigen_T);
		Eigen::Quaterniond Q(eigen_R);
		eigen_R = Q.normalized();
		RIC.push_back(eigen_R);
		TIC.push_back(eigen_T);
		//ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]);
		//ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
	}
	LOOP_CLOSURE = fsSettings["loop_closure"];
	if (LOOP_CLOSURE == 1)
	{
		fsSettings["voc_file"] >> VOC_FILE;
		fsSettings["pattern_file"] >> PATTERN_FILE;
		VOC_FILE = VINS_FOLDER_PATH + VOC_FILE;
		PATTERN_FILE = VINS_FOLDER_PATH + PATTERN_FILE;
		MIN_LOOP_NUM = fsSettings["min_loop_num"];
	}
	SOLVER_TIME = fsSettings["max_solver_time"];
	NUM_ITERATIONS = fsSettings["max_num_iterations"];

	MIN_PARALLAX = fsSettings["keyframe_parallax"];
	MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

	ACC_N = fsSettings["acc_n"];
	ACC_W = fsSettings["acc_w"];
	GYR_N = fsSettings["gyr_n"];
	GYR_W = fsSettings["gyr_w"];
	G.z() = fsSettings["g_norm"];

	INIT_DEPTH = 5.0;
}
