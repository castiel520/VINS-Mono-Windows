#include "parameters.h"

std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;

double FOCUS_LENGTH_Y;
double PY;
double FOCUS_LENGTH_X;
double PX;

int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
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

	WINDOW_SIZE = 20;
	STEREO_TRACK = false;
	FOCAL_LENGTH = 460;
	PUB_THIS_FRAME = false;

	if (FREQ == 0)
		FREQ = 100;

	fsSettings.release();


}
