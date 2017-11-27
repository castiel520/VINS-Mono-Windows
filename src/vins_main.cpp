#include <iostream>
#include <stdio.h>
#include <memory>
#include <windows.h>
#include <mutex>
#include <opencv2\opencv.hpp>

#include "vins_main.h"

using namespace cv;

VideoCapture caps[NUM_OF_CAM];

// Store the IMU data for vins
queue<ImuConstPtr> imu_msg_buf;
// Store the fesature data processed by featuretracker
queue<ImgConstPtr> img_msg_buf;

// Lock the feature and imu data buffer
std::mutex m_buf;
std::condition_variable con;

std::thread img_thread;
int main(int, char**)
{
	for (int i = 0; i < NUM_OF_CAM; i++)
	{
		caps[i].open(0);
		if (!caps[i].isOpened())
		{
			std::cerr << "Unable to open camera\n";
			return -1;
		}
	}

	img_thread = std::thread(&ProcessImage::processImage, ProcessImage());
	return 0;
}

bool ProcessImage::grabAllCameras(VideoCapture* caps, cv::Mat* frames)
{
	for (int i = 0; i < NUM_OF_CAM; i++)
	{
		if (caps[i].grab())
		{
			caps[i].retrieve(frames[i]);
			if (frames[i].empty())
			{
				return false;
			}
		}
		else
		{
			return false;
		}
	}
	return true;
}

void ProcessImage::processImage()
{
	Mat frames[NUM_OF_CAM];

	//鱼眼相机的mask,追踪时候会用到
	if (FISHEYE)
	{
		for (int i = 0; i < NUM_OF_CAM; i++)
		{
			trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
			if (!trackerData[i].fisheye_mask.data)
			{
				break;
			}
		}
	}

	while (true)
	{
		if (grabAllCameras(caps, frames))
		{
			std::shared_ptr<IMG_MSG> img_msg = std::make_shared<IMG_MSG>();
			img_msg->header = std::chrono::duration<double>(GetTickCount64()).count();

			if (m_b_first_image_flag)
			{
				m_b_first_image_flag = false;
				m_d_first_image_time = img_msg->header;
			}

			// frequency control
			if (round(1.0 * m_pub_count / (img_msg->header - m_d_first_image_time)) <= FREQ)
			{
				PUB_THIS_FRAME = true;
				// reset the frequency control
				if (abs(1.0 * m_pub_count / (img_msg->header - m_d_first_image_time) - FREQ) < 0.01 * FREQ)
				{
					m_d_first_image_time = img_msg->header;
					m_pub_count = 0;
				}
			}
			else
				PUB_THIS_FRAME = false;

			for (int i = 0; i < NUM_OF_CAM; i++)
			{
				if (i != 1 || !STEREO_TRACK)
					trackerData[i].readImage(frames[i]);
				else
				{
					//双目
					if (EQUALIZE)
					{
						cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
						clahe->apply(frames[i], trackerData[i].cur_img);
					}
					else
						trackerData[i].cur_img = frames[i];
				}
			}

			//双目
			if (PUB_THIS_FRAME && STEREO_TRACK && trackerData[0].cur_pts.size() > 0)
			{
				m_pub_count++;
				r_status.clear();
				r_err.clear();
				cv::calcOpticalFlowPyrLK(trackerData[0].cur_img, trackerData[1].cur_img, trackerData[0].cur_pts, trackerData[1].cur_pts, r_status, r_err, cv::Size(21, 21), 3);
				vector<cv::Point2f> ll, rr;
				vector<int> idx;
				for (unsigned int i = 0; i < r_status.size(); i++)
				{
					if (!inBorder(trackerData[1].cur_pts[i]))
						r_status[i] = 0;

					if (r_status[i])
					{
						idx.push_back(i);

						//@baitao commented the undistortion code for now

						//Eigen::Vector3d tmp_p;
						//trackerData[0].m_camera->liftProjective(Eigen::Vector2d(trackerData[0].cur_pts[i].x, trackerData[0].cur_pts[i].y), tmp_p);
						//tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
						//tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
						ll.push_back(cv::Point2f(trackerData[0].cur_pts[i].x, trackerData[0].cur_pts[i].y));

						//trackerData[1].m_camera->liftProjective(Eigen::Vector2d(trackerData[1].cur_pts[i].x, trackerData[1].cur_pts[i].y), tmp_p);
						//tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
						//tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
						rr.push_back(cv::Point2f(trackerData[1].cur_pts[i].x, trackerData[1].cur_pts[i].y));
					}
				}
				if (ll.size() >= 8)
				{
					vector<uchar> status;
					cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 1.0, 0.5, status);
					int r_cnt = 0;
					for (unsigned int i = 0; i < status.size(); i++)
					{
						if (status[i] == 0)
							r_status[idx[i]] = 0;
						r_cnt += r_status[idx[i]];
					}
				}
			}

			//更新全局ID
			for (unsigned int i = 0;; i++)
			{
				bool completed = false;
				for (int j = 0; j < NUM_OF_CAM; j++)
					if (j != 1 || !STEREO_TRACK)
						completed |= trackerData[j].updateID(i);
				if (!completed)
					break;
			}

			//发布当前帧，包括id和undistorted后的点，和u,v点
			if (PUB_THIS_FRAME)
			{
				m_pub_count++;

				vector<set<int>> hash_ids(NUM_OF_CAM);
				for (int i = 0; i < NUM_OF_CAM; i++)
				{
					if (i != 1 || !STEREO_TRACK)
					{
						auto &cur_pts = trackerData[i].cur_pts;
						auto &ids = trackerData[i].ids;
						for (unsigned int j = 0; j < ids.size(); j++)
						{
							int p_id = ids[j];
							hash_ids[i].insert(p_id);
							Eigen::Vector3d p;
							p.x = (cur_pts[i].x - PX) / FOCUS_LENGTH_X;
							p.y = (cur_pts[i].y - PY) / FOCUS_LENGTH_Y;
							p.z = 1;

							img_msg->point_clouds[p_id * NUM_OF_CAM + i] = p;
						}
					}
					else if (STEREO_TRACK)
					{
						//双目
						//auto r_un_pts = trackerData[1].undistortedPoints();
						auto &cur_pts = trackerData[1].cur_pts;
						auto &ids = trackerData[0].ids;
						for (unsigned int j = 0; j < ids.size(); j++)
						{
							if (r_status[j])
							{
								int p_id = ids[j];
								hash_ids[i].insert(p_id);
								Eigen::Vector3d p;
								p.x = (cur_pts[i].x - PX) / FOCUS_LENGTH_X;
								p.y = (cur_pts[i].y - PY) / FOCUS_LENGTH_Y;
								p.z = 1;

								img_msg->point_clouds[p_id * NUM_OF_CAM + i] = p;
							}
						}
					}
					//send to img_msg_buf
					m_buf.lock();
					img_msg_buf.push(img_msg);
					m_buf.unlock();
					con.notify_one();
				}

				if (SHOW_TRACK)
				{
					Mat stereo_img(frames[0].rows * 2, frames[0].cols, CV_8UC3);
					Mat up(stereo_img, Rect(0, 0, frames[0].cols, frames[0].rows));
					Mat down(stereo_img, Rect(0, frames[0].rows, frames[0].cols, frames[0].rows));
					frames[0].copyTo(up);
					frames[1].copyTo(down);

					for (int i = 0; i < NUM_OF_CAM; i++)
					{
						cv::Mat tmp_img = frames[i];
						cv::cvtColor(tmp_img, tmp_img, CV_GRAY2RGB);
						if (i != 1 || !STEREO_TRACK)
						{
							//显示追踪状态，越红越好，越蓝越不行
							for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
							{
								double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
								cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
								//char name[10];
								//sprintf(name, "%d", trackerData[i].ids[j]);
								//cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
							}
						}
						else
						{
							//双目
							for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
							{
								if (r_status[j])
								{
									cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(0, 255, 0), 2);
									cv::line(stereo_img, trackerData[i - 1].cur_pts[j], trackerData[i].cur_pts[j] + cv::Point2f(0, ROW), cv::Scalar(0, 255, 0));
								}
							}
						}
					}
				}

			}

		}
	}

}