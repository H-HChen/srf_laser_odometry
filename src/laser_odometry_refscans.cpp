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

#include "laser_odometry_refscans.h"

using namespace Eigen;
using namespace std;
using namespace SRF_LaserOdometry;


void SRF_RefS::initialize(unsigned int size, float FOV_rad, unsigned int odo_method)
{
    method = odo_method;
    cols = size;
    width = size;
    fovh = FOV_rad;//*size/(size-1);           //Exact for simulation, but I don't know how the datasets are given...
    ctf_levels = ceilf(log2(cols) - 4.3f);
    iter_irls = 8;
    no_ref_scan = true;
    new_ref_scan = true;
	
    //Resize original range scan
    range_wf = Eigen::MatrixXf::Constant(1, width, 1);

    //Resize the transformation matrices
    transformations.resize(ctf_levels);
    for (unsigned int i = 0; i < ctf_levels; i++)
    {
        transformations[i].resize(3,3);
        //transformations[i].setIdentity();
    }

	//Resize pyramid
	unsigned int s, cols_i;
    const unsigned int pyr_levels = round(log2(round(float(width)/float(cols)))) + ctf_levels;
    range_1.resize(pyr_levels); range_2.resize(pyr_levels); range_3.resize(pyr_levels);
    range_12.resize(pyr_levels); range_13.resize(pyr_levels);
    xx_1.resize(pyr_levels); xx_2.resize(pyr_levels); xx_3.resize(pyr_levels);
    xx_12.resize(pyr_levels); xx_13.resize(pyr_levels);
    yy_1.resize(pyr_levels); yy_2.resize(pyr_levels); yy_3.resize(pyr_levels);
    yy_12.resize(pyr_levels); yy_13.resize(pyr_levels);
    range_warped.resize(pyr_levels); xx_warped.resize(pyr_levels); yy_warped.resize(pyr_levels);
    range_3_warpedTo2.resize(pyr_levels); xx_3_warpedTo2.resize(pyr_levels); yy_3_warpedTo2.resize(pyr_levels);
    
	for (unsigned int i = 0; i<pyr_levels; i++)
    {
        s = pow(2.f,int(i));
        cols_i = ceil(float(width)/float(s));

        range_1[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
        range_2[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
        range_3[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);

        range_12[i].resize(1, cols_i); 
        range_13[i].resize(1, cols_i);

        xx_1[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
        xx_2[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
        xx_3[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);

        xx_12[i].resize(1, cols_i); 
        xx_13[i].resize(1, cols_i);

        yy_1[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
        yy_2[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
        yy_3[i] = Eigen::MatrixXf::Constant(1, cols_i, 0.f);

        yy_12[i].resize(1, cols_i); 
        yy_13[i].resize(1, cols_i);

		if (cols_i <= cols)
		{
            range_warped[i].resize(1, cols_i); xx_warped[i].resize(1, cols_i); yy_warped[i].resize(1, cols_i);
            range_3_warpedTo2[i].resize(1, cols_i); xx_3_warpedTo2[i].resize(1, cols_i); yy_3_warpedTo2[i].resize(1, cols_i);
		}
    }

    //Resize aux variables
    dt_12.resize(1, cols); dt_13.resize(1, cols);
    dtita_12.resize(1, cols); dtita_13.resize(1, cols);
    weights_12.resize(1, cols); weights_13.resize(1, cols);
    null_12    = Eigen::MatrixXi::Constant(1, cols, 0);
    null_13    = Eigen::MatrixXi::Constant(1, cols, 0);

    cov_odo = IncrementCov::Zero();
    outliers = Eigen::MatrixXi::Constant(1, cols, 0);

	//Compute gaussian mask
	g_mask[0] = 1.f/16.f; 
    g_mask[1] = 0.25f; 
    g_mask[2] = 6.f/16.f; 
    g_mask[3] = g_mask[1]; 
    g_mask[4] = g_mask[0];

    //Initialize "last velocity" as zero
	kai_abs     = MatrixS31::Zero();
	kai_loc_old = MatrixS31::Zero();
    overall_trans_prev.setIdentity(); //haven't modify
}


void SRF_RefS::createScanPyramid()
{
	const float max_range_dif = 0.3f;
	
    //Push scan back
    range_1.swap(range_2); xx_1.swap(xx_2); yy_1.swap(yy_2);

    //The number of levels of the pyramid does not match the number of levels used
    //in the odometry computation (because we sometimes want to finish with lower resolutions)

    unsigned int pyr_levels = round(log2(round(float(width)/float(cols)))) + ctf_levels;

    //Generate levels
    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        unsigned int s = pow(2.f,int(i));
        cols_i = ceil(float(width)/float(s));
		const unsigned int i_1 = i-1;

        //              First level -> Filter, not downsample
        //------------------------------------------------------------------------
        if (i == 0)
		{
			for (unsigned int u = 0; u < cols_i; u++)
            {	
				const float dcenter = range_wf(u);
					
				//Inner pixels
                if ((u>1)&&(u<cols_i-2))
                {		
					if (std::isfinite(dcenter) && dcenter > 0.f)
					{	
                        float sum = 0.f, weight = 0.f;

						for (int l=-2; l<3; l++)
						{
							const float abs_dif = abs(range_wf(u+l)-dcenter);
							if (abs_dif < max_range_dif)
							{
								const float aux_w = g_mask[2+l]*(max_range_dif - abs_dif);
								weight += aux_w;
								sum += aux_w*range_wf(u+l);
							}
						}
                        range_1[i](u) = sum/weight;
					}
					else
                        range_1[i](u) = 0.f;

                }

                //Boundary
                else
                {
                    if (std::isfinite(dcenter) && dcenter > 0.f)
					{						
                        float sum = 0.f, weight = 0.f;

						for (int l=-2; l<3; l++)	
						{
							const int indu = u+l;
							if ((indu>=0)&&(indu<cols_i))
							{
								const float abs_dif = abs(range_wf(indu)-dcenter);										
								if (abs_dif < max_range_dif)
								{
									const float aux_w = g_mask[2+l]*(max_range_dif - abs_dif);
									weight += aux_w;
									sum += aux_w*range_wf(indu);
								}
							}
						}
                        range_1[i](u) = sum/weight;
					}
					else
                        range_1[i](u) = 0.f;

                }
            }
		}

        //                              Downsampling
        //-----------------------------------------------------------------------------
        else
        {            
            const unsigned int cols_prev_level = range_1[i_1].rows();

            //Odd number of elements in the previous level
            if ((cols_prev_level % 2) == 1)
                for (unsigned int u = 0; u < cols_i; u++)
                {
                    const int u2 = 2*u;
                    const float dcenter = range_1[i_1](u2);

                    //Inner pixels
                    if ((u>0)&&(u<cols_i-1))
                    {
                        if (std::isfinite(dcenter) && dcenter > 0.f)
                        {
                            float sum = 0.f, weight = 0.f;

                            for (int l=-2; l<3; l++)
                            {
                                const float abs_dif = abs(range_1[i_1](u2+l)-dcenter);
                                if (abs_dif < max_range_dif)
                                {
                                    const float aux_w = g_mask[2+l]*(max_range_dif - abs_dif);
                                    weight += aux_w;
                                    sum += aux_w*range_1[i_1](u2+l);
                                }
                            }
                            range_1[i](u) = sum/weight;
                        }
                        else
                            range_1[i](u) = 0.f;

                    }

                    //Boundary
                    else
                    {
                        if (std::isfinite(dcenter) && dcenter > 0.f)
                        {
                            float sum = 0.f, weight = 0.f;
                            const unsigned int cols_i2 = range_1[i_1].rows();

                            for (int l=-2; l<3; l++)
                            {
                                const int indu = u2+l;

                                if ((indu>=0)&&(indu<cols_i2))
                                {
                                    const float abs_dif = abs(range_1[i_1](indu)-dcenter);
                                    if (abs_dif < max_range_dif)
                                    {
                                        const float aux_w = g_mask[2+l]*(max_range_dif - abs_dif);
                                        weight += aux_w;
                                        sum += aux_w*range_1[i_1](indu);
                                    }
                                }
                            }
                            range_1[i](u) = sum/weight;

                        }
                        else
                            range_1[i](u) = 0.f;
                    }
                }
            //Even number of elements in the previous level
            else
            {
                for (unsigned int u = 0; u < cols_i; u++)
                {
                    const int u2 = 2*u;

                    if ((range_1[i_1](u2) == 0.f)&&(range_1[i_1](u2+1) == 0.f))
                        range_1[i](u) = 0.f;
                    else if (range_1[i_1](u2) == 0.f)
                        range_1[i](u) = range_1[i_1](u2+1);
                    else if (range_1[i_1](u2+1) == 0.f)
                        range_1[i](u) = range_1[i_1](u2);
                    else
                        range_1[i](u) = 0.5f*(range_1[i_1](u2) + range_1[i_1](u2+1));
                }
            }
        }

        //Calculate coordinates "xy" of the points
        for (unsigned int u = 0; u < cols_i; u++) 
		{
            if (range_1[i](u) > 0.f)
			{
                const float tita = -0.5f*fovh + (float(u) + 0.5f)*fovh/float(cols_i);
                xx_1[i](u) = range_1[i](u)*cos(tita);
                yy_1[i](u) = range_1[i](u)*sin(tita);
			}
			else
			{
                xx_1[i](u) = 0.f;
                yy_1[i](u) = 0.f;
			}
		}
    }

    if (no_ref_scan)
    {
        range_3 = range_1;
        xx_3 = xx_1;
        yy_3 = yy_1;
        no_ref_scan = false;
    }
}

void SRF_RefS::calculateCoord()
{		
    null_12.fill(0);
    null_13.fill(0);
    num_valid_range = 0;

    for (unsigned int u = 0; u < cols_i; u++)
	{
        //Coordinates 12
        if ((range_2[image_level](u) == 0.f) || (range_warped[image_level](u) == 0.f))
		{
            range_12[image_level](u) = 0.f;
            xx_12[image_level](u) = 0.f;
            yy_12[image_level](u) = 0.f;
            null_12(u) = 1;
		}
		else
		{
            range_12[image_level](u) = 0.5f*(range_2[image_level](u) + range_warped[image_level](u));
            xx_12[image_level](u) = 0.5f*(xx_2[image_level](u) + xx_warped[image_level](u));
            yy_12[image_level](u) = 0.5f*(yy_2[image_level](u) + yy_warped[image_level](u));
            if ((u>0)&&(u<cols_i-1))
                num_valid_range++;
		}

        //Coordinates 13
        if ((range_3_warpedTo2[image_level](u) == 0.f) || (range_warped[image_level](u) == 0.f))
        {
            range_13[image_level](u) = 0.f;
            xx_13[image_level](u) = 0.f;
            yy_13[image_level](u) = 0.f;
            null_13(u) = 1;
        }
        else
        {
            range_13[image_level](u) = 0.5f*(range_3_warpedTo2[image_level](u) + range_warped[image_level](u));
            xx_13[image_level](u) = 0.5f*(xx_3_warpedTo2[image_level](u) + xx_warped[image_level](u));
            yy_13[image_level](u) = 0.5f*(yy_3_warpedTo2[image_level](u) + yy_warped[image_level](u));
            if ((u>0)&&(u<cols_i-1))
                num_valid_range++;
        }
	}
}

void SRF_RefS::calculateRangeDerivatives()
{	
    //Compute distances between points
    rtita_12 = Eigen::MatrixXf::Constant(1, cols_i, 1.f);
    rtita_13 = Eigen::MatrixXf::Constant(1, cols_i, 1.f);

	for (unsigned int u = 0; u < cols_i-1; u++)
    {
        const float dist_12 = square(xx_12[image_level](u+1) - xx_12[image_level](u))
                            + square(yy_12[image_level](u+1) - yy_12[image_level](u));

        const float dist_13 = square(xx_13[image_level](u+1) - xx_13[image_level](u))
                            + square(yy_13[image_level](u+1) - yy_13[image_level](u));

        if (dist_12  > 0.f)
            rtita_12(u) = sqrtf(dist_12);

        if (dist_13  > 0.f)
            rtita_13(u) = sqrtf(dist_13);
	}

    //Spatial derivatives
    for (unsigned int u = 1; u < cols_i-1; u++)
    {
        dtita_12(u) = (rtita_12(u-1)*(range_12[image_level](u+1)-range_12[image_level](u)) + rtita_12(u)*(range_12[image_level](u) - range_12[image_level](u-1)))/(rtita_12(u)+rtita_12(u-1));
        dtita_13(u) = (rtita_13(u-1)*(range_13[image_level](u+1)-range_13[image_level](u)) + rtita_13(u)*(range_13[image_level](u) - range_13[image_level](u-1)))/(rtita_13(u)+rtita_13(u-1));
    }

    dtita_12(0) = dtita_12(1);
    dtita_12(cols_i-1) = dtita_12(cols_i-2);

    dtita_13(0) = dtita_13(1);
    dtita_13(cols_i-1) = dtita_13(cols_i-2);

	//Temporal derivative
	for (unsigned int u = 0; u < cols_i; u++)
    {
        dt_12(u) = range_warped[image_level](u) - range_2[image_level](u);
        dt_13(u) = range_warped[image_level](u) - range_3_warpedTo2[image_level](u);
    }
}

void SRF_RefS::computeWeights()
{
	//The maximum weight size is reserved at the constructor
    weights_12.setConstant(1, cols, 0.f);
    weights_13.setConstant(1, cols, 0.f);
	
    //Parameters for error_linearization - (kd = 1.f, k2d = 0.02f, ssigma = 100*e-4f works!)
    const float kd = 0.01f;
    const float k2d = 2e-4f;
    const float sensor_sigma = 4e-4f;
//    const float kd = 1.f;
//    const float k2d = 0.02f; //0.5 no, 0.3 no, 0.1 no, 0.05 no, 0.02 no (better between 0.2 and 0.05)
//    const float sensor_sigma = 100.f*4e-4f; //100.f*4e-4f, 200 is too much
	
	for (unsigned int u = 1; u < cols_i-1; u++)
    {
        if (null_12(u) == 0)
		{	
			//							Compute derivatives
			//-----------------------------------------------------------------------
            //const float ini_dtita = range_2[image_level](u+1) - range_2[image_level](u-1);
            //const float final_dtita = range_warped[image_level](u+1) - range_warped[image_level](u-1);
            //const float dtitat = ini_dtita - final_dtita;
            const float dtita2 = dtita_12(u+1) - dtita_12(u-1);
            const float w_der = kd*(square(dt_12(u)) + square(dtita_12(u))) + k2d*square(dtita2) + sensor_sigma;

            weights_12(u) = sqrtf(1.f/w_der);
		}

        if (null_13(u) == 0)
        {
            //							Compute derivatives
            //-----------------------------------------------------------------------
            //const float ini_dtita = range_3_warpedTo2[image_level](u+1) - range_3_warpedTo2[image_level](u-1);
            //const float final_dtita = range_warped[image_level](u+1) - range_warped[image_level](u-1);
            //const float dtitat = ini_dtita - final_dtita;
            const float dtita2 = dtita_13(u+1) - dtita_13(u-1);
            const float w_der = kd*(square(dt_13(u)) + square(dtita_13(u))) + k2d*square(dtita2) + sensor_sigma;

            weights_13(u) = sqrtf(1.f/w_der);
        }
    }

    const float max_w = max(weights_12.maxCoeff(), weights_13.maxCoeff());
    const float inv_max_w = 1.f/max_w;
    weights_12 = inv_max_w*weights_12;
    weights_13 = inv_max_w*weights_13;
}


void SRF_RefS::solveSystemQuadResiduals3Scans()
{
    A.resize(num_valid_range, 3);
    B.resize(num_valid_range, 1);
    unsigned int cont = 0;
    const float kdtita = (cols_i)/fovh;

    //Fill the matrix A and the vector B
    //The order of the variables will be (vx, vy, wz)

    for (unsigned int u = 1; u < cols_i-1; u++)
    {
        if (null_12(u) == 0)
        {
            // Precomputed expressions
            const float tw = weights_12(u);
            const float tita = -0.5f*fovh + (float(u) + 0.5f)/kdtita;

            //Fill the matrix A
            A(cont, 0) = tw*(cos(tita) + dtita_12(u)*kdtita*sin(tita)/range_12[image_level](u));
            A(cont, 1) = tw*(sin(tita) - dtita_12(u)*kdtita*cos(tita)/range_12[image_level](u));
            A(cont, 2) = tw*(-yy_12[image_level](u)*cos(tita) + xx_12[image_level](u)*sin(tita) - dtita_12(u)*kdtita); //?????
            B(cont) = tw*(-dt_12(u));

            cont++;
        }

        if (null_13(u) == 0)
        {
            // Precomputed expressions
            const float tw = weights_13(u);
            const float tita = -0.5f*fovh + (float(u) + 0.5f)/kdtita;

            //Fill the matrix A
            A(cont, 0) = tw*(cos(tita) + dtita_13(u)*kdtita*sin(tita)/range_13[image_level](u));
            A(cont, 1) = tw*(sin(tita) - dtita_13(u)*kdtita*cos(tita)/range_13[image_level](u));
            A(cont, 2) = tw*(-yy_13[image_level](u)*cos(tita) + xx_13[image_level](u)*sin(tita) - dtita_13(u)*kdtita);
            B(cont) = tw*(-dt_13(u));

            cont++;
        }
    }

    //Solve the linear system of equations using a minimum least squares method
    Eigen::MatrixXf AtA, AtB;
    AtA = A.transpose()*A;
    AtB = A.transpose()*B;
    kai_loc_level = AtA.ldlt().solve(AtB);

    //Covariance matrix calculation
    Eigen::MatrixXf res(num_valid_range,1);
    res = A*kai_loc_level - B;
    cov_odo = (1.f/float(num_valid_range-3))*AtA.inverse()*res.squaredNorm();
}

void SRF_RefS::solveSystemSmoothTruncQuad3Scans()
{
    A.resize(num_valid_range, 3); Aw.resize(num_valid_range, 3);
    B.resize(num_valid_range, 1); Bw.resize(num_valid_range, 1);
    unsigned int cont = 0;
    const float kdtita = float(cols_i)/fovh;
    const float inv_kdtita = 1.f/kdtita;

    //Fill the matrix A and the vector B
    //The order of the variables will be (vx, vy, wz)

    for (unsigned int u = 1; u < cols_i-1; u++)
    {
        const float tita = -0.5f*fovh + (float(u) + 0.5f)*inv_kdtita;
        const float cos_tita = cos(tita);
        const float sin_tita = sin(tita);

        if (null_12(u) == 0)
        {
            // Precomputed expressions
            const float tw = weights_12(u);


            //Fill the matrix A
            A(cont, 0) = tw*(cos_tita + dtita_12(u)*kdtita*sin_tita/range_12[image_level](u));
            A(cont, 1) = tw*(sin_tita - dtita_12(u)*kdtita*cos_tita/range_12[image_level](u));
            A(cont, 2) = tw*(-yy_12[image_level](u)*cos_tita + xx_12[image_level](u)*sin_tita - dtita_12(u)*kdtita);
            B(cont) = tw*(-dt_12(u));

            cont++;
        }

        if (null_13(u) == 0)
        {
            // Precomputed expressions
            const float tw = weights_13(u);

            //Fill the matrix A
            A(cont, 0) = tw*(cos_tita + dtita_13(u)*kdtita*sin_tita/range_13[image_level](u));
            A(cont, 1) = tw*(sin_tita - dtita_13(u)*kdtita*cos_tita/range_13[image_level](u));
            A(cont, 2) = tw*(-yy_13[image_level](u)*cos_tita + xx_13[image_level](u)*sin_tita - dtita_13(u)*kdtita);
            B(cont) = tw*(-dt_13(u));

            cont++;
        }
    }

    //Solve the linear system of equations using a minimum least squares method
    Eigen::MatrixXf AtA, AtB;
    AtA = A.transpose()*A;
    AtB = A.transpose()*B;
    kai_loc_level = AtA.ldlt().solve(AtB);
    Eigen::MatrixXf res(num_valid_range,1);
    res = A*kai_loc_level - B;
    /*
    //Compute the median of res
    vector<float> aux_vector;
    for (unsigned int k = 0; k<res.rows(); k++)
        aux_vector.push_back(res(k));
    std::sort(aux_vector.begin(), aux_vector.end());
    float res_median = aux_vector.at(res.rows()/2);

    //Compute the median absolute deviation
    aux_vector.clear();
    for (unsigned int k = 0; k<res.rows(); k++)
        aux_vector.push_back(abs(res(k) - res_median));
    std::sort(aux_vector.begin(), aux_vector.end());
    float mad = aux_vector.at(res.rows()/2);

    //Find the m-estimator constant
    const float c = 4.f*mad;
    const float inv_c = 1.f/c;
    const float squared_c = square(c);

    //Compute the energy
    float new_energy = 0.f, last_energy;

    for (unsigned int i=0; i<res.rows(); i++)
    {
        if (abs(res(i)) < c)     new_energy += 0.5f*square(res(i))*(1.f - 0.5f*square(res(i)*inv_c));
        else                     new_energy += 0.25f*squared_c;
    }
    //printf("\n\nEnergy(0) = %f", new_energy);
    last_energy = 2.f*new_energy;
    unsigned int iter = 1;
    
    //Solve iterative reweighted least squares
    //===================================================================
    //while ((new_energy < 0.995f*last_energy)&&(iter < 10))
    */
    float aver_dt = 0.f, aver_res = 0.f; unsigned int ind = 0;
    for (unsigned int u = 1; u < cols_i-1; u++){
        if (null_12(u) == 0)
        {
        aver_dt  += std::abs(dt_12(u));
        aver_res += std::abs(res(ind++));
        }
        if (null_13(u) == 0)
        {
        aver_dt  += std::abs(dt_13(u));
        aver_res += std::abs(res(ind++));
        }
    }
    aver_dt /= cont; aver_res /= cont;
    const float k = 10.f/aver_dt; //200

    for (unsigned int i=1; i<=iter_irls; i++)
    {
        cont = 0;
        //last_energy = new_energy;

        for (unsigned int u = 1; u < cols_i-1; u++)
        {
            if (null_12(u) == 0)
            {
                //float res_weight;
                //if (abs(res(cont)) <= c)    res_weight = 1.f - square(res(cont)*inv_c);
                //else                        res_weight = 0.f;
                const float res_weight = std::sqrt(1.f/(1.f + ((k*res(cont))*(k*res(cont)))));

                //Fill the matrix Aw
                Aw(cont,0) = res_weight*A(cont,0);
                Aw(cont,1) = res_weight*A(cont,1);
                Aw(cont,2) = res_weight*A(cont,2);
                Bw(cont) = res_weight*B(cont);
                cont++;
            }

            if (null_13(u) == 0)
            {
                const float res_weight = std::sqrt(1.f/(1.f + ((k*res(cont))*(k*res(cont)))));

                //float res_weight;
                //if (abs(res(cont)) <= c)    res_weight = 1.f - square(res(cont)*inv_c);
                //else                        res_weight = 0.f;

                //Fill the matrix Aw
                Aw(cont,0) = res_weight*A(cont,0);
                Aw(cont,1) = res_weight*A(cont,1);
                Aw(cont,2) = res_weight*A(cont,2);
                Bw(cont) = res_weight*B(cont);
                cont++;
            }
        }

        //Solve the linear system of equations using a minimum least squares method
        AtA = Aw.transpose()*Aw;
        AtB = Aw.transpose()*Bw;
        kai_loc_level = AtA.ldlt().solve(AtB);
        res = A*kai_loc_level - B;

        //Compute the energy
        //new_energy = 0.f;
        /*
        for (unsigned int i=0; i<res.rows(); i++)
        {
            if (abs(res(i)) < c)    new_energy += 0.5f*square(res(i))*(1.f - 0.5f*square(res(i)*inv_c));
            else                    new_energy += 0.25f*squared_c;
        }
        //printf("\nEnergy(%d) = %f", iter, new_energy);
        iter++;
        */
        //Recompute c
        //-------------------------------------------------
        //Compute the median of res
//        aux_vector.clear();
//        for (unsigned int k = 0; k<res.rows(); k++)
//            aux_vector.push_back(res(k));
//        std::sort(aux_vector.begin(), aux_vector.end());
//        res_median = aux_vector.at(res.rows()/2);

//        //Compute the median absolute deviation
//        aux_vector.clear();
//        for (unsigned int k = 0; k<res.rows(); k++)
//            aux_vector.push_back(abs(res(k) - res_median));
//        std::sort(aux_vector.begin(), aux_vector.end());
//        mad = aux_vector.at(res.rows()/2);

//        //Find the m-estimator constant
//        c = 4.f*mad;
//        c_inv = 1.f/c;
    }

    //Covariance calculation
    cov_odo = (1.f/float(num_valid_range-3))*AtA.inverse()*res.squaredNorm();
    //ROS_INFO_STREAM("COV_ODO:\n" << cov_odo);

}

void SRF_RefS::solveSystemSmoothTruncQuadOnly13()
{
    unsigned int valid_here = 0;
    for (unsigned int u = 1; u < cols_i-1; u++)
    {
        if (null_13(u) == 0)
            valid_here++;
    }

    A.resize(valid_here, 3); Aw.resize(valid_here, 3);
    B.resize(valid_here, 1); Bw.resize(valid_here, 1);
    unsigned int cont = 0;
    const float kdtita = float(cols_i)/fovh;
    const float inv_kdtita = 1.f/kdtita;

    //Fill the matrix A and the vector B
    //The order of the variables will be (vx, vy, wz)

    for (unsigned int u = 1; u < cols_i-1; u++)
    {
        const float tita = -0.5f*fovh + (float(u) + 0.5f)*inv_kdtita;
        const float cos_tita = cos(tita);
        const float sin_tita = sin(tita);

        if (null_13(u) == 0)
        {
            // Precomputed expressions
            const float tw = weights_13(u);

            //Fill the matrix A
            A(cont, 0) = tw*(cos_tita + dtita_13(u)*kdtita*sin_tita/range_13[image_level](u));
            A(cont, 1) = tw*(sin_tita - dtita_13(u)*kdtita*cos_tita/range_13[image_level](u));
            A(cont, 2) = tw*(-yy_13[image_level](u)*cos_tita + xx_13[image_level](u)*sin_tita - dtita_13(u)*kdtita);
            B(cont) = tw*(-dt_13(u));

            cont++;
        }
    }

    //Solve the linear system of equations using a minimum least squares method
    Eigen::MatrixXf AtA, AtB;
    AtA = A.transpose()*A;
    AtB = A.transpose()*B;
    kai_loc_level = AtA.ldlt().solve(AtB);
    Eigen::MatrixXf res(num_valid_range,1);
    res = A*kai_loc_level - B;
    //cout << endl << "max res: " << res.maxCoeff();
    //cout << endl << "min res: " << res.minCoeff();

    //Compute the median of res
    /*
    vector<float> aux_vector;
    for (unsigned int k = 0; k<res.rows(); k++)
        aux_vector.push_back(res(k));
    std::sort(aux_vector.begin(), aux_vector.end());
    const float res_median = aux_vector.at(res.rows()/2);

    //Compute the median absolute deviation
    aux_vector.clear();
    for (unsigned int k = 0; k<res.rows(); k++)
        aux_vector.push_back(abs(res(k) - res_median));
    std::sort(aux_vector.begin(), aux_vector.end());
    const float mad = aux_vector.at(res.rows()/2);

    //Find the m-estimator constant
    const float c = 4.f*mad;
    const float inv_c = 1.f/c;
    const float squared_c = square(c);

    //Compute the energy
    float new_energy = 0.f, last_energy;
    for (unsigned int i=0; i<res.rows(); i++)
    {
        if (abs(res(i)) < c)     new_energy += 0.5f*square(res(i))*(1.f - 0.5f*square(res(i)*inv_c));
        else                     new_energy += 0.25f*squared_c;
    }
    //printf("\n\nEnergy(0) = %f", new_energy);
    last_energy = 2.f*new_energy;
    unsigned int iter = 1;

    //Solve iterative reweighted least squares
    //===================================================================
    while ((new_energy < 0.995f*last_energy)&&(iter < 10))
    */

    float aver_dt = 0.f, aver_res = 0.f; unsigned int ind = 0;
    for (unsigned int u = 1; u < cols_i-1; u++)
        if (null_13(u) == 0)
        {
        aver_dt  += std::abs(dt_13(u));
        aver_res += std::abs(res(ind++));
        }
    aver_dt /= cont; aver_res /= cont;
  //    printf("\n Aver dt = %f, aver res = %f", aver_dt, aver_res);


    const float k = 10.f/aver_dt; //200

    for (unsigned int i=1; i<=iter_irls; i++)
    {
        cont = 0;
        //last_energy = new_energy;

        for (unsigned int u = 1; u < cols_i-1; u++)
        {
            if (null_13(u) == 0)
            {
                const float res_weight = std::sqrt(1.f/(1.f + ((k*res(cont))*(k*res(cont)))));

                //float res_weight;
                //if (abs(res(cont)) <= c)    res_weight = 1.f - square(res(cont)*inv_c);
                //else                        res_weight = 0.f;

                //Fill the matrix Aw
                Aw(cont,0) = res_weight*A(cont,0);
                Aw(cont,1) = res_weight*A(cont,1);
                Aw(cont,2) = res_weight*A(cont,2);
                Bw(cont) = res_weight*B(cont);
                cont++;
            }
        }

        //Solve the linear system of equations using a minimum least squares method
        AtA = Aw.transpose()*Aw;
        AtB = Aw.transpose()*Bw;
        kai_loc_level = AtA.ldlt().solve(AtB);
        res = A*kai_loc_level - B;

        //Compute the energy
        /*
        new_energy = 0.f;
        for (unsigned int i=0; i<res.rows(); i++)
        {
            if (abs(res(i)) < c)    new_energy += 0.5f*square(res(i))*(1.f - 0.5f*square(res(i)*inv_c));
            else                    new_energy += 0.25f*squared_c;
        }
        //printf("\nEnergy(%d) = %f", iter, new_energy);
        */
        //iter++;

    }

    //Covariance calculation
    cov_odo = (1.f/float(valid_here-3))*AtA.inverse()*res.squaredNorm();
}

void SRF_RefS::solveSystemSmoothTruncQuadOnly12()
{
    unsigned int valid_here = 0;
    for (unsigned int u = 1; u < cols_i-1; u++)
    {
        if (null_12(u) == 0)
            valid_here++;
    }

    A.resize(valid_here, 3); Aw.resize(valid_here, 3);
    B.resize(valid_here, 1); Bw.resize(valid_here, 1);
    unsigned int cont = 0;
    const float kdtita = float(cols_i)/fovh;
    const float inv_kdtita = 1.f/kdtita;

    //Fill the matrix A and the vector B
    //The order of the variables will be (vx, vy, wz)

    for (unsigned int u = 1; u < cols_i-1; u++)
    {
        const float tita = -0.5f*fovh + (float(u) + 0.5f)*inv_kdtita;
        const float cos_tita = cos(tita);
        const float sin_tita = sin(tita);

        if (null_12(u) == 0)
        {
            // Precomputed expressions
            const float tw = weights_12(u);

            //Fill the matrix A
            A(cont, 0) = tw*(cos_tita + dtita_12(u)*kdtita*sin_tita/range_12[image_level](u));
            A(cont, 1) = tw*(sin_tita - dtita_12(u)*kdtita*cos_tita/range_12[image_level](u));
            A(cont, 2) = tw*(-yy_12[image_level](u)*cos_tita + xx_12[image_level](u)*sin_tita - dtita_12(u)*kdtita);
            B(cont) = tw*(-dt_12(u));

            cont++;
        }
    }

    //Solve the linear system of equations using a minimum least squares method
    Eigen::MatrixXf AtA, AtB;
    AtA = A.transpose()*A;
    AtB = A.transpose()*B;
    kai_loc_level = AtA.ldlt().solve(AtB);
    Eigen::MatrixXf res(num_valid_range,1);
    res = A*kai_loc_level - B;
    //cout << endl << "max res: " << res.maxCoeff();
    //cout << endl << "min res: " << res.minCoeff();

    //Compute the median of res
    /*
    vector<float> aux_vector;
    for (unsigned int k = 0; k<res.rows(); k++)
        aux_vector.push_back(res(k));
    std::sort(aux_vector.begin(), aux_vector.end());
    const float res_median = aux_vector.at(res.rows()/2);

    //Compute the median absolute deviation
    aux_vector.clear();
    for (unsigned int k = 0; k<res.rows(); k++)
        aux_vector.push_back(abs(res(k) - res_median));
    std::sort(aux_vector.begin(), aux_vector.end());
    const float mad = aux_vector.at(res.rows()/2);

    //Find the m-estimator constant
    const float c = 4.f*mad;
    const float inv_c = 1.f/c;
    const float squared_c = square(c);

    //Compute the energy
    float new_energy = 0.f, last_energy;
    for (unsigned int i=0; i<res.rows(); i++)
    {
        if (abs(res(i)) < c)     new_energy += 0.5f*square(res(i))*(1.f - 0.5f*square(res(i)*inv_c));
        else                     new_energy += 0.25f*squared_c;
    }
    //printf("\n\nEnergy(0) = %f", new_energy);
    last_energy = 2.f*new_energy;
    unsigned int iter = 1;

    //Solve iterative reweighted least squares
    //===================================================================
    while ((new_energy < 0.995f*last_energy)&&(iter < 10))
    */


    float aver_dt = 0.f, aver_res = 0.f; unsigned int ind = 0;
    for (unsigned int u = 1; u < cols_i-1; u++)
        if (null_12(u) == 0)
        {
        aver_dt  += std::abs(dt_12(u));
        aver_res += std::abs(res(ind++));
        }
    aver_dt /= cont; aver_res /= cont;
  //    printf("\n Aver dt = %f, aver res = %f", aver_dt, aver_res);

    const float k = 10.f/aver_dt; //200

    for (unsigned int i=1; i<=iter_irls; i++)
    {
        cont = 0;
        //last_energy = new_energy;

        for (unsigned int u = 1; u < cols_i-1; u++)
        {
            if (null_12(u) == 0)
            {
                //float res_weight;
                //if (abs(res(cont)) <= c)    res_weight = 1.f - square(res(cont)*inv_c);
                //else                        res_weight = 0.f;
                const float res_weight = std::sqrt(1.f/(1.f + ((k*res(cont))*(k*res(cont)))));

                //Fill the matrix Aw
                Aw(cont,0) = res_weight*A(cont,0);
                Aw(cont,1) = res_weight*A(cont,1);
                Aw(cont,2) = res_weight*A(cont,2);
                Bw(cont) = res_weight*B(cont);
                cont++;
            }
        }

        //Solve the linear system of equations using a minimum least squares method
        AtA = Aw.transpose()*Aw;
        AtB = Aw.transpose()*Bw;
        kai_loc_level = AtA.ldlt().solve(AtB);
        res = A*kai_loc_level - B;

        //Compute the energy
        /*
        new_energy = 0.f;
        for (unsigned int i=0; i<res.rows(); i++)
        {
            if (abs(res(i)) < c)    new_energy += 0.5f*square(res(i))*(1.f - 0.5f*square(res(i)*inv_c));
            else                    new_energy += 0.25f*squared_c;
        }
        //printf("\nEnergy(%d) = %f", iter, new_energy);
        iter++;
        */
    }

    //Covariance calculation
    cov_odo = (1.f/float(valid_here-3))*AtA.inverse()*res.squaredNorm();
}


void SRF_RefS::performWarping()
{
    Eigen::Matrix3f acu_trans;
    acu_trans.setIdentity();
    for (unsigned int i=0; i<=level; i++)
        acu_trans = transformations[i]*acu_trans;

    Eigen::MatrixXf wacu = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
    range_warped[image_level].setConstant(1, cols_i, 0.f);

    const float cols_lim = float(cols_i-1);
    const float kdtita = cols_i/fovh;

    for (unsigned int j = 0; j<cols_i; j++)
    {
        if (range_1[image_level](j) > 0.f)
        {
            //Transform point to the warped reference frame
            const float x_w = acu_trans(0,0)*xx_1[image_level](j) + acu_trans(0,1)*yy_1[image_level](j) + acu_trans(0,2);
            const float y_w = acu_trans(1,0)*xx_1[image_level](j) + acu_trans(1,1)*yy_1[image_level](j) + acu_trans(1,2);
            const float tita_w = atan2(y_w, x_w);
            const float range_w = sqrt(x_w*x_w + y_w*y_w);

            //Calculate warping
            const float uwarp = kdtita*(tita_w + 0.5*fovh) - 0.5f;

            //The warped pixel (which is not integer in general) contributes to all the surrounding ones
            if ((uwarp >= 0.f)&&(uwarp < cols_lim))
            {
                const int uwarp_l = uwarp;
                const int uwarp_r = uwarp_l + 1;
                const float delta_r = float(uwarp_r) - uwarp;
                const float delta_l = uwarp - float(uwarp_l);

                //Very close pixel
                if (abs(round(uwarp) - uwarp) < 0.05f)
                {
                    range_warped[image_level](round(uwarp)) += range_w;
                    wacu(round(uwarp)) += 1.f;
                }
                else
                {
                    const float w_r = square(delta_l);
                    range_warped[image_level](uwarp_r) += w_r*range_w;
                    wacu(uwarp_r) += w_r;

                    const float w_l = square(delta_r);
                    range_warped[image_level](uwarp_l) += w_l*range_w;
                    wacu(uwarp_l) += w_l;
                }
            }
        }
    }

    //Scale the averaged range and compute coordinates
    for (unsigned int u = 0; u<cols_i; u++)
    {
        if (wacu(u) > 0.f)
        {
            const float tita = -0.5f*fovh + (float(u) + 0.5f)/kdtita;
            range_warped[image_level](u) /= wacu(u);
            xx_warped[image_level](u) = range_warped[image_level](u)*cos(tita);
            yy_warped[image_level](u) = range_warped[image_level](u)*sin(tita);
        }
        else
        {
            range_warped[image_level](u) = 0.f;
            xx_warped[image_level](u) = 0.f;
            yy_warped[image_level](u) = 0.f;
        }
    }
}

void SRF_RefS::performBestWarping()
{
    Eigen::Matrix3f acu_trans;
    acu_trans.setIdentity();
    for (unsigned int i=0; i<=level; i++)
        acu_trans = transformations[i]*acu_trans;
    Eigen::MatrixXf x_trans = Eigen::MatrixXf::Constant(1, cols_i, 0.f); 
    Eigen::MatrixXf y_trans = Eigen::MatrixXf::Constant(1, cols_i, 0.f); 
    Eigen::MatrixXf u_trans(1, cols_i);
    Eigen::MatrixXf range_trans = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
    range_warped[image_level].setConstant(1, cols_i, 0.f);

    const float kdtita = float(cols_i)/fovh;

    for (unsigned int u = 0; u<cols_i; u++)
    {
        if (range_1[image_level](u) > 0.f)
        {
            //Transform point to the warped reference frame
            x_trans(u) = acu_trans(0,0)*xx_1[image_level](u) + acu_trans(0,1)*yy_1[image_level](u) + acu_trans(0,2);
            y_trans(u) = acu_trans(1,0)*xx_1[image_level](u) + acu_trans(1,1)*yy_1[image_level](u) + acu_trans(1,2);
            range_trans(u) = sqrtf(square(x_trans(u)) + square(y_trans(u)));
            const float tita_trans = atan2(y_trans(u), x_trans(u));
            u_trans(u) = kdtita*(tita_trans + 0.5f*fovh) - 0.5f;
        }
    }

    //Check projection for each segment
    for (unsigned int u = 0; u<cols_i-1; u++)
    {
        if ((range_trans(u) == 0.f) || (range_trans(u+1) == 0.f))
            continue;
        else if (floorf(u_trans(u)) != floorf(u_trans(u+1)))
        {
           const float range_l = (floorf(u_trans(u)) > floorf(u_trans(u+1))) ? range_trans(u+1) : range_trans(u);
           const float range_r = (floorf(u_trans(u)) > floorf(u_trans(u+1))) ? range_trans(u) : range_trans(u+1);
           const float u_trans_l = (floorf(u_trans(u)) > floorf(u_trans(u+1))) ? u_trans(u+1) : u_trans(u);
           const float u_trans_r = (floorf(u_trans(u)) > floorf(u_trans(u+1))) ? u_trans(u) : u_trans(u+1);
           const int u_l = min(floorf(u_trans(u)), floorf(u_trans(u+1)));
           const int u_r = max(floorf(u_trans(u)), floorf(u_trans(u+1)));

           for (unsigned int u_segment=u_l+1; (u_segment<=u_r)&&(u_segment<cols_i)&&(u_segment>=0); u_segment++)
           {
               const float range_interp = ((u_segment - u_trans_l)*range_r + (u_trans_r - u_segment)*range_l)/(u_trans_r - u_trans_l);
               if ((range_warped[image_level](u_segment) == 0.f)||(range_interp < range_warped[image_level](u_segment)))
                   range_warped[image_level](u_segment) = range_interp;
           }
        }
    }

    //Compute coordinates
    for (unsigned int u = 0; u<cols_i; u++)
    {
        const float tita = -0.5f*fovh + (float(u) + 0.5f)/kdtita;
        xx_warped[image_level](u) = range_warped[image_level](u)*cos(tita);
        yy_warped[image_level](u) = range_warped[image_level](u)*sin(tita);
    }
}

void SRF_RefS::warpScan3To2()
{
    //Use the previous transformation to warp the scan 3 forward (to 2)
    Eigen::Matrix3f acu_trans_inv = overall_trans_prev.inverse();


    //Create forward-warped scans for every level
    for (unsigned int i=0; i<ctf_levels; i++)
    {
        unsigned int s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = ceil(float(cols)/float(s));
        image_level = ctf_levels - i + round(log2(round(float(width)/float(cols)))) - 1;

        Eigen::MatrixXf x_trans = Eigen::MatrixXf::Constant(1, cols_i, 0.f); 
        Eigen::MatrixXf y_trans = Eigen::MatrixXf::Constant(1, cols_i, 0.f); 
        Eigen::MatrixXf u_trans(1, cols_i);
        Eigen::MatrixXf range_trans = Eigen::MatrixXf::Constant(1, cols_i, 0.f);
        range_3_warpedTo2[image_level].setConstant(1, cols_i, 0.f);

        const float kdtita = float(cols_i)/fovh;

        //Compute transformed coordinates
        for (unsigned int u = 0; u<cols_i; u++)
        {
            if (range_3[image_level](u) > 0.f)
            {
                //Transform point to the warped reference frame
                x_trans(u) = acu_trans_inv(0,0)*xx_3[image_level](u) + acu_trans_inv(0,1)*yy_3[image_level](u) + acu_trans_inv(0,2);
                y_trans(u) = acu_trans_inv(1,0)*xx_3[image_level](u) + acu_trans_inv(1,1)*yy_3[image_level](u) + acu_trans_inv(1,2);
                range_trans(u) = sqrtf(square(x_trans(u)) + square(y_trans(u)));
                const float tita_trans = atan2(y_trans(u), x_trans(u));
                u_trans(u) = kdtita*(tita_trans + 0.5f*fovh) - 0.5f;
            }
        }

        //Check projection for each segment
        for (unsigned int u = 0; u<cols_i-1; u++)
        {
            if ((range_trans(u) == 0.f) || (range_trans(u+1) == 0.f))
                continue;
            else if (floorf(u_trans(u)) != floorf(u_trans(u+1)))
            {
               const float range_l = (floorf(u_trans(u)) > floorf(u_trans(u+1))) ? range_trans(u+1) : range_trans(u);
               const float range_r = (floorf(u_trans(u)) > floorf(u_trans(u+1))) ? range_trans(u) : range_trans(u+1);
               const float u_trans_l = (floorf(u_trans(u)) > floorf(u_trans(u+1))) ? u_trans(u+1) : u_trans(u);
               const float u_trans_r = (floorf(u_trans(u)) > floorf(u_trans(u+1))) ? u_trans(u) : u_trans(u+1);
               const int u_l = min(floorf(u_trans(u)), floorf(u_trans(u+1)));
               const int u_r = max(floorf(u_trans(u)), floorf(u_trans(u+1)));

               for (unsigned int u_segment=u_l+1; (u_segment<=u_r)&&(u_segment<cols_i)&&(u_segment>=0); u_segment++)
               {
                   const float range_interp = ((u_segment - u_trans_l)*range_r + (u_trans_r - u_segment)*range_l)/(u_trans_r - u_trans_l);

                   //Condition to obtain the right projection (hiding the occluded parts)
//                   if ((range_3_warpedTo2[image_level](u_segment) == 0.f)||(range_interp < range_3_warpedTo2[image_level](u_segment)))
//                       range_3_warpedTo2[image_level](u_segment) = range_interp;

                   //Condition to obtain the structure of the environment (I retain the furthest point for each pixel)
                   if (range_interp > range_3_warpedTo2[image_level](u_segment))
                       range_3_warpedTo2[image_level](u_segment) = range_interp;
               }
            }
        }

        //Compute coordinates
        for (unsigned int u = 0; u<cols_i; u++)
        {
            const float tita = -0.5f*fovh + (float(u) + 0.5f)/kdtita;
            xx_3_warpedTo2[image_level](u) = range_3_warpedTo2[image_level](u)*cos(tita);
            yy_3_warpedTo2[image_level](u) = range_3_warpedTo2[image_level](u)*sin(tita);
        }
    }
}


void SRF_RefS::odometryCalculation()
{
    //==================================================================================
    //						DIFERENTIAL  ODOMETRY  MULTILEVEL
    //==================================================================================
    ros::WallTime start = ros::WallTime::now();

    createScanPyramid();

    if (new_ref_scan)
    {
        range_3_warpedTo2 = range_3;
        xx_3_warpedTo2 = xx_3;
        yy_3_warpedTo2 = yy_3;
    }
    else{
        warpScan3To2();
    }
    //Coarse-to-fine scheme
    for (unsigned int i=0; i<ctf_levels; i++)
    {
        
        //Previous computations
        transformations[i].setIdentity();

        level = i;
        unsigned int s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = ceil(float(cols)/float(s));
        image_level = ctf_levels - i + round(log2(round(float(width)/float(cols)))) - 1;

        for (unsigned int k=0; k<3; k++)
        {
            //1. Perform warping
            if ((i == 0)&&(k == 0))
            {
                range_warped[image_level] = range_1[image_level];
                xx_warped[image_level] = xx_1[image_level];
                yy_warped[image_level] = yy_1[image_level];
            }
            else{

                performBestWarping();

            }

            //2. Calculate inter coords
            calculateCoord();

            //3. Compute derivatives
            calculateRangeDerivatives();

            //4. Compute weights
            computeWeights();

            //5. Solve odometry
            
            if (num_valid_range > 3)
            {
                if (method == 0)
                    solveSystemSmoothTruncQuadOnly12();

                else if (method == 1){
                    solveSystemSmoothTruncQuadOnly13();
                }

                else
                {
                    if  (new_ref_scan == true){
                        solveSystemSmoothTruncQuadOnly13();
                    }
                    else{
                        solveSystemSmoothTruncQuad3Scans();
                    }
                }
            }
            
            //6. Filter solution
            if (!filterLevelSolution()){
                if  (new_ref_scan == true)
                    new_ref_scan = false;
                runtime = ros::WallTime::now() - start;
                cout << endl << "Time odometry (ms): " << runtime;
                updateReferenceScan();
                return;}
            //filterLevelSolution();
            if (kai_loc_level.norm() < 0.05f)
            {
                printf("\n Number of non-linear iterations: %d", k+1);
                break;
            }

        }
    }

    //Clear flag
    if  (new_ref_scan == true)
        new_ref_scan = false;

    runtime = ros::WallTime::now() - start;
    cout << endl << "Time odometry (ms): " << runtime;
    //Update poses
    PoseUpdate();

    //Update the reference scan if necessary
    updateReferenceScan();

}

bool SRF_RefS::filterLevelSolution()
{
    //		Calculate Eigenvalues and Eigenvectors
    //----------------------------------------------------------
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eigensolver(cov_odo);
    ROS_INFO_STREAM("COV_ODO:\n" << cov_odo);

    if (eigensolver.info() != Eigen::Success)
    {
        printf("\n Eigensolver couldn't find a solution. Pose is not updated\n");
        return false;
    }
	
    //First, we have to describe both the new linear and angular speeds in the "eigenvector" basis
    //-------------------------------------------------------------------------------------------------
    Eigen::Matrix<float,3,3> Bii = eigensolver.eigenvectors();
    Eigen::Matrix<float,3,1> kai_b = Bii.colPivHouseholderQr().solve(kai_loc_level);

    assert((kai_loc_level).isApprox(Bii*kai_b, 1e-5) && "Ax=b has no solution." && __LINE__);

    //Second, we have to describe both the old linear and angular speeds in the "eigenvector" basis too
    //-------------------------------------------------------------------------------------------------
    MatrixS31 kai_loc_sub;

    //Important: we have to substract the solutions from previous levels
    Eigen::Matrix3f acu_trans = Matrix3f::Identity();
    for (unsigned int i=0; i<=level; i++)
        acu_trans = transformations[i]*acu_trans;

    kai_loc_sub(0) = -acu_trans(0,2);
    kai_loc_sub(1) = -acu_trans(1,2);
    if (acu_trans(0,0) > 1.f)
        kai_loc_sub(2) = 0.f;
    else
        kai_loc_sub(2) = -acos(acu_trans(0,0))*sign(acu_trans(1,0));
    kai_loc_sub += kai_loc_old;

    Eigen::Matrix<float,3,1> kai_b_old = Bii.colPivHouseholderQr().solve(kai_loc_sub);

    assert((kai_loc_sub).isApprox(Bii*kai_b_old, 1e-5) && "Ax=b has no solution." && __LINE__);


    //Filter speed
    const float cf = 15e3f*expf(-int(level)), df = 0.05f*expf(-int(level));
    //const float cf = 5e3f*expf(-int(level)), df = 0.02f*expf(-int(level));
    //const float cf = 0*expf(-int(level)), df = 0.f*expf(-int(level));

    Eigen::Matrix<float,3,1> kai_b_fil;
    for (unsigned int i=0; i<3; i++)
    {
        kai_b_fil(i,0) = (kai_b(i,0) + (cf*eigensolver.eigenvalues()(i,0) + df)*kai_b_old(i,0))/(1.f + cf*eigensolver.eigenvalues()(i,0) + df);
        //kai_b_fil_f(i,0) = (1.f*kai_b(i,0) + 0.f*kai_b_old_f(i,0))/(1.0f + 0.f);
    }

    //Transform filtered speed to local reference frame and compute transformation
    Eigen::Matrix<float,3,1> kai_loc_fil = Bii.inverse().colPivHouseholderQr().solve(kai_b_fil);
    
    assert((kai_b_fil).isApprox(Bii.inverse()*kai_loc_fil, 1e-5) && "Ax=b has no solution." && __LINE__);


    //transformation
    const float incrx = kai_loc_fil(0);
    const float incry = kai_loc_fil(1);
    const float rot   = kai_loc_fil(2);

    Matrix3f new_trans = Matrix3f::Identity();
    new_trans(0,0) = cos(rot);
    new_trans(0,1) = -sin(rot);
    new_trans(1,0) = sin(rot);
    new_trans(1,1) = cos(rot);

    Matrix2f V = Matrix2f::Identity();
    Vector2f incr; incr << incrx, incry;
    if (abs(rot) > 0.001f)
    {
        const float V1 = sin(rot)/rot;
        const float V2 = (1.f - cos(rot))/rot;
        V << V1, -V2, V2, V1;
    }

    new_trans(0,2) = (V*incr)(0);
    new_trans(1,2) = (V*incr)(1);

    transformations[level] = new_trans*transformations[level];
    
    return true;
}

void SRF_RefS::PoseUpdate()
{
    //First, compute the overall transformation
    //---------------------------------------------------
    Eigen::Matrix3f acu_trans;
    acu_trans.setIdentity();

    for (unsigned int i=1; i<=ctf_levels; i++)
        acu_trans = transformations[i-1]*acu_trans;


    overall_trans_prev = overall_trans_prev*acu_trans;

    //				Compute kai_loc and kai_abs
    //--------------------------------------------------------
    kai_loc(0) = acu_trans(0,2);
    kai_loc(1) = acu_trans(1,2);
    if (acu_trans(0,0) > 1.f)
        kai_loc(2) = 0.f;
    else{
        kai_loc(2) = acos(acu_trans(0,0))*sign(acu_trans(1,0));
    }

    float phi = getYaw(laser_pose.rotation());

    kai_abs(0) = kai_loc(0)*cos(phi) - kai_loc(1)*sin(phi);
    kai_abs(1) = kai_loc(0)*sin(phi) + kai_loc(1)*cos(phi);
    kai_abs(2) = kai_loc(2);

    //cout << endl << "Estimated twist: " << kai_abs.transpose();


    //						Update poses
    //-------------------------------------------------------
    laser_oldpose = laser_pose;

    Pose3d pose_aux_2D = Pose3d::Identity();

    pose_aux_2D = matrixYaw(double(kai_loc(2)));
    pose_aux_2D.translation()(0) = acu_trans(0,2);
    pose_aux_2D.translation()(1) = acu_trans(1,2);
    laser_pose = laser_pose * pose_aux_2D;

    ROS_INFO("LASERodom = [%f %f %f]",
                laser_pose.translation()(0),
                laser_pose.translation()(1),
                getYaw(laser_pose.rotation()));

    //                  Compute kai_loc_old
    //-------------------------------------------------------
    phi = getYaw(laser_pose.rotation());
    kai_loc_old(0) =  kai_abs(0)*cos(phi) + kai_abs(1)*sin(phi);
    kai_loc_old(1) = -kai_abs(0)*sin(phi) + kai_abs(1)*cos(phi);
    kai_loc_old(2) =  kai_abs(2);

}

void SRF_RefS::updateReferenceScan()
{
    //Threshold small scanner (180, 360, 30 m): r = -14.92*t⁴ - 7.617*t³ + 2.307*t² - 0.3149*t + 0.1054
    //Threshold big scanner (270, 541, 80 m): r = 3.072*t⁴ - 3.916*t³ + 0.1799*t² + 0.0883*t + 0.2849
    //Threshold very big scanner (270, 1080, 30 m): r = -10.66*t⁴ + 11.81*t³ - 4.371*t² + 0.5319*t + 0.2042
    //const float threshold = 0.5f;

    const float trans = sqrtf(square(overall_trans_prev(0,2)) + square(overall_trans_prev(1,2)));
    const float rot = abs(acos(overall_trans_prev(0,0)));
    //printf("\n Trans = %f, rot = %f", trans, rot);

    float keyscan_out_region;
    const float trans_2 = square(trans);
    if (cols < 500)
        keyscan_out_region = -14.92*square(trans_2) - 7.617*trans_2*trans + 2.307*trans_2 - 0.3149*trans + 0.1054 - rot;
    else
        keyscan_out_region = -10.66*square(trans_2) + 11.81*trans_2*trans - 4.371*trans_2 + 0.5319*trans + 0.2042 - rot;

    ROS_INFO("cols:%d ,keyscan_out_region:%f, trans:%f, trans_2:%f", cols, keyscan_out_region, trans, trans_2);
    if (keyscan_out_region < 0.f) //(trans + rot > threshold)
    {
        //ref_scan = old_scan
        range_3 = range_1; xx_3 = xx_1; yy_3 = yy_1;

        //Overall_trans_prev = T12
        Eigen::Matrix3f acu_trans = Matrix3f::Identity();
//        for (unsigned int i=1; i<=ctf_levels; i++)
//            acu_trans = transformations[i-1]*acu_trans;
        overall_trans_prev = acu_trans;

        printf("\n New keyframe inserted!!!");
        new_ref_scan = true;
    }
}

