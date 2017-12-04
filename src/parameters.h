#pragma once
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#include "utility/utility.h"

extern int ROW;
extern int COL;
extern int FOCAL_LENGTH;
const int NUM_OF_CAM = 1;

extern double FOCUS_LENGTH_Y;
extern double PY;
extern double FOCUS_LENGTH_X;
extern double PX;

extern std::string FISHEYE_MASK;
extern std::vector<std::string> CAM_NAMES;
extern int MAX_CNT;
extern int MIN_DIST;
extern const int WINDOW_SIZE;
extern int FREQ;
extern double F_THRESHOLD;
extern int SHOW_TRACK;
extern int STEREO_TRACK;
extern int EQUALIZE;
extern int FISHEYE;
extern bool PUB_THIS_FRAME;
extern int ESTIMATE_EXTRINSIC;

extern std::vector<Eigen::Matrix3d> RIC;
extern std::vector<Eigen::Vector3d> TIC;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string VINS_FOLDER_PATH;
extern int LOOP_CLOSURE;
extern std::string PATTERN_FILE;
extern std::string VOC_FILE;
extern int MIN_LOOP_NUM;
extern int NUM_ITERATIONS;
extern double SOLVER_TIME;
extern double MIN_PARALLAX;
extern double INIT_DEPTH;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;
extern Eigen::Vector3d G;

enum StateOrder
{
	O_P = 0,
	O_R = 3,
	O_V = 6,
	O_BA = 9,
	O_BG = 12
};

#define WINDOW_SIZE 10
#define SIZE_POSE 7
#define SIZE_SPEEDBIAS 9
#define SIZE_FEATURE 1
#define NUM_OF_F 1000


void readParameters();
