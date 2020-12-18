/*********************************************************************
*
* Software License Agreement (GPLv3 License)
*
*  Authors: Mariano Jaimez Tarifa and Javier Monroy
*           MAPIR group, University of Malaga, Spain
*           http://mapir.uma.es
*
*  Date: January 2016
*
* This pkgs offers a fast and reliable estimation of 2D odometry based on planar laser scans.
* SRF is a fast and precise method to estimate the planar motion of a lidar from consecutive range scans. 
* SRF presents a dense method for estimating planar motion with a laser scanner. Starting from a symmetric 
* representation of geometric consistency between scans, we derive a precise range flow constraint and 
* express the motion of the scan observations as a function of the rigid motion of the scanner. 
* In contrast to existing techniques, which align the incoming scan with either the previous one or the last 
* selected keyscan, we propose a combined and efficient formulation to jointly align all these three scans at 
* every iteration. This new formulation preserves the advantages of keyscan-based strategies but is more robust 
* against suboptimal selection of keyscans and the presence of moving objects.
*
*  More Info: http://mapir.isa.uma.es/work/SRF-Odometry
*********************************************************************/


// std header
#include <iostream>
#include <fstream>
#include <numeric>
#include <cmath>

// ROS headers
#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

// Eigen headers
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

namespace SRF_LaserOdometry {
    using Scalar = float;

    using Pose2d = Eigen::Isometry2d;
    using Pose3d = Eigen::Isometry3d;
    using MatrixS31 = Eigen::Matrix<Scalar, 3, 1>;
    using IncrementCov = Eigen::Matrix<Scalar, 3, 3>;

    template <typename T>
    inline T sign(const T x) { return x<T(0) ? -1:1; }

    template <typename Derived>
    inline typename Eigen::MatrixBase<Derived>::Scalar
    getYaw(const Eigen::MatrixBase<Derived>& r)
    {
    return std::atan2( r(1, 0), r(0, 0) );
    }

    template<typename T>
    inline Eigen::Matrix<T, 3, 3> matrixRollPitchYaw(const T roll,
                                                    const T pitch,
                                                    const T yaw)
    {
    const Eigen::AngleAxis<T> ax = Eigen::AngleAxis<T>(roll,  Eigen::Matrix<T, 3, 1>::UnitX());
    const Eigen::AngleAxis<T> ay = Eigen::AngleAxis<T>(pitch, Eigen::Matrix<T, 3, 1>::UnitY());
    const Eigen::AngleAxis<T> az = Eigen::AngleAxis<T>(yaw,   Eigen::Matrix<T, 3, 1>::UnitZ());

    return (az * ay * ax).toRotationMatrix().matrix();
    }

    template<typename T>
    inline Eigen::Matrix<T, 3, 3> matrixYaw(const T yaw)
    {
    return matrixRollPitchYaw<T>(0, 0, yaw);
    }

    template <typename T>
    inline T square(const T num){return pow(num,2);}

class SRF_RefS {
public:


    //Scans and cartesian coordinates: 1 - New, 2 - Old, 3 - Ref
    std::vector<Eigen::MatrixXf> range_1, range_2, range_3;
    std::vector<Eigen::MatrixXf> range_12, range_13, range_warped;
    std::vector<Eigen::MatrixXf> xx_1, xx_2, xx_3, xx_12, xx_13, xx_warped;
    std::vector<Eigen::MatrixXf> yy_1, yy_2, yy_3, yy_12, yy_13, yy_warped;
    std::vector<Eigen::MatrixXf> range_3_warpedTo2, xx_3_warpedTo2, yy_3_warpedTo2;

    //Rigid transformations and velocities (twists: vx, vy, w)
    std::vector<Eigen::MatrixXf> transformations; //T13
    Eigen::Matrix3f overall_trans_prev; // T23
    MatrixS31 kai_abs, kai_loc;
    MatrixS31 kai_loc_old, kai_loc_level;

    //Solver
    Eigen::MatrixXf A,Aw;
    Eigen::MatrixXf B,Bw;
    Eigen::Matrix3f cov_odo;
    Eigen::MatrixXf range_wf;
	
    //Aux variables
    Eigen::MatrixXf dtita_12, dtita_13;
    Eigen::MatrixXf dt_12, dt_13;
    Eigen::MatrixXf rtita_12, rtita_13;

    Eigen::MatrixXf weights_12, weights_13;
    Eigen::MatrixXi null_12, null_13;
    Eigen::MatrixXi outliers;

	float fovh;
    unsigned int cols, cols_i;
	unsigned int width;
	unsigned int ctf_levels;
	unsigned int image_level, level;
	unsigned int num_valid_range;
	unsigned int iter_irls;
	float g_mask[5];
    bool no_ref_scan;
    bool new_ref_scan;


    //Laser poses (most recent and previous)
    Pose3d laser_pose;
    Pose3d laser_oldpose;
	bool test;
    unsigned int method; //0 - consecutive scan alignment, 1 - keyscan alignment, 2 - multi-scan (hybrid) alignment

    //To measure runtimes
    ros::WallDuration    runtime;


    //Methods
    void initialize(unsigned int size, float FOV_rad, unsigned int odo_method);
    void createScanPyramid();
	void calculateCoord();
	void performWarping();
    void performBestWarping();
    void warpScan3To2();
    void calculateRangeDerivatives();
	void computeWeights();
    void solveSystemQuadResiduals3Scans();
    void solveSystemSmoothTruncQuad3Scans();
    void solveSystemSmoothTruncQuadOnly13();
    void solveSystemSmoothTruncQuadOnly12();
    void solveSystemMCauchy();
    void solveSystemMTukey();
    void solveSystemTruncatedQuad();

	bool filterLevelSolution();
	void PoseUpdate();
    void updateReferenceScan();
	void odometryCalculation();
};
}