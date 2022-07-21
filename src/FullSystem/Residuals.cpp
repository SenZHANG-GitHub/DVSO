/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
int PointFrameResidual::instanceCounter = 0;


long runningResID=0;


PointFrameResidual::PointFrameResidual(){assert(false); instanceCounter++;}

PointFrameResidual::~PointFrameResidual(){assert(efResidual==0); instanceCounter--; delete J;}

PointFrameResidual::PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_) :
	point(point_),
	host(host_),
	target(target_)
{
	efResidual=0;
	instanceCounter++;
	resetOOB();
	J = new RawResidualJacobian();
	assert(((long)J)%16==0);

	isNew=true;
}


// // SenFlagVS
// Sen ->  J_I will be modified to virtual stereo version
// Sen -> For other vars, only set J w.r.t. idepth 
// Sen -> J w.r.t. othet vars will be set zero!
double PointFrameResidual::linearizeVirtualStereo(CalibHessian* HCalib)
{
	state_NewEnergyWithOutlier=-1;

	if(state_state == ResState::OOB)
		{ state_NewState = ResState::OOB; return state_energy; }

	// Get some precalculated parameters of the current frame w.r.t. the reference frame
    // Sen -> We don't need info w.r.t. T, c, l (only idepth is optimized here)
	// FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);
	float energyLeft=0;
	
	// Sen -> for virtual stereo residuals, target is host->frame_right
	const Eigen::Vector3f* dIl_left= host->dI;
	const Eigen::Vector3f* dDisp_right = target->dDisp;
	//const float* const Il = target->I;

    // Sen -> The only T is baseline
    Mat33f PRE_RTll_0 = Mat33f::Identity();
	Vec3f PRE_tTll_0 = Vec3f::Zero();
	PRE_tTll_0 << -baseline, 0, 0;
	
	// PRE_tTll_0 << -0.54f, 0, 0;
	// PRE_tTll_0 << w_bl * baseline, 0, 0;

	Mat33f PRE_RTll = Mat33f::Identity();
	Vec3f PRE_tTll = PRE_tTll_0;

	Mat33f K = Mat33f::Zero();
	K(0,0) = HCalib->fxl();
	K(1,1) = HCalib->fyl();
	K(0,2) = HCalib->cxl();
	K(1,2) = HCalib->cyl();
	K(2,2) = 1;

	float fxt = HCalib->fxl();
 
	Mat33f PRE_KRKiTll = K * PRE_RTll * K.inverse();
	Mat33f PRE_RKiTll = PRE_RTll * K.inverse();
	Vec3f PRE_KtTll = K * PRE_tTll;
    
	// const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	// const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	// const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0; 
	// const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
	

	// Sen -> point->color is sampled from host->dI
	const float * const color = point->color;
	const float * const weights = point->weights;

	// Vec2f affLL = precalc->PRE_aff_mode; 
	// float b0 = precalc->PRE_b0_mode; 

    // Sen -> We also don't optimize affLL here
    Vec2f affLL;
	affLL << 1,0;
	float b0 = 0;

	// Vec6f d_xi_x, d_xi_y;
	// Vec4f d_C_x, d_C_y;
	float d_d_x, d_d_y;
	{
		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;


        // Sen -> Will give values to: derescale, u, v, new_idepth, Ku, Kv, KliP
		// Sen -> Difference with the latter projectPoint 
		//		-> Here use point->u, point-v, point->idepth_zero_scaled, PRE_RTll_0, PRE_tTll_0
		//			-> FEJ! and only use the center point for idepth, \xi, c (FEJ params) 
		//		-> Latter use point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll
		//			-> Use all pattern points for JI, Jab, and resisual and use current state (not FEJ)
		if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
				PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{ state_NewState = ResState::OOB; return state_energy; }

		centerProjectedTo = Vec3f(Ku, Kv, new_idepth);


		// diff d_idepth
        // Sen -> drescale, u, v are updated in projectPoint()
		d_d_x = drescale * (PRE_tTll_0[0] - PRE_tTll_0[2] * u) * SCALE_IDEPTH * HCalib->fxl();
		d_d_y = drescale * (PRE_tTll_0[1] - PRE_tTll_0[2] * v) * SCALE_IDEPTH * HCalib->fyl();

	}


	{
		J->Jpdxi[0] = Vec6f::Zero();  // d_xi_x
		J->Jpdxi[1] = Vec6f::Zero();  // d_xi_y

		J->Jpdc[0] = Vec4f::Zero();   // d_C_x
		J->Jpdc[1] = Vec4f::Zero();   // d_C_y

		// Sen -> Jpdd only used here
		J->Jpdd[0] = d_d_x;
		J->Jpdd[1] = d_d_y;

	}


	

	float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
 	// float JabJab_00=0, JabJab_01=0, JabJab_11=0;

	float wJI2_sum = 0;

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;

        // Sen -> Difference: (1) point->idepth_zero_scaled -> point->idepth_scaled
        //        -> (2) PRE_RTll_0, PRE_tTll_0 -> PRE_KRKiTll, PRE_KtTll
        // Sen -> Only update Ku and Kv -> Use (Ku, Kv) to fetch J_I_j
		if(!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{ state_NewState = ResState::OOB; return state_energy; }

		projectedTo[idx][0] = Ku;
		projectedTo[idx][1] = Kv;


        // Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        // float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);

		// printf("==============\nhit_Ku: %2.f (wG[0]: %d)\nhit_Kv: %.2f (hG[0]: %d)\n", Ku, wG[0], Kv, hG[0]);

		Vec3f hitDisp_right = (getInterpolatedElement33(dDisp_right, Ku, Kv, wG[0]));

		// printf("=> tmp: %.2f\n", hitDisp_right[0]);

		Vec3f hitColor_left = (getInterpolatedElement33(dIl_left, Ku + hitDisp_right[0], Kv, wG[0]));
		// Vec3f hitColor_left = (getInterpolatedElement33(dIl_left, Ku - hitDisp_right[0], Kv, wG[0]));
		

		// float residual = hitColor_left[0] - (float)(affLL[0] * color[idx] + affLL[1]);
		float residual = hitColor_left[0] - color[idx];


		// float drdA = (color[idx] - b0); // Sen -> Not used in stereo
		if(!std::isfinite((float)hitColor_left[0]) || !std::isfinite((float)hitDisp_right[0]))
		{ 
			// printf("Error: Virtual stereo residual not used!!!");
			// exit(1);
			state_NewState = ResState::OOB; 
			return state_energy; 
		}


		// Sen -> hitColor[1], hitColor[2] refer to JI (d_I_x, d_I_y)
		float d_I_x, d_I_y;

		d_I_x = (1 + hitDisp_right[1]) * hitColor_left[1];
		d_I_y = hitDisp_right[2] * hitColor_left[1] + hitColor_left[2];

		// printf("=============\nhitColor_left[0]: %.2f\nhitColor_left[1]: %.2f\nhitColor_left[2]: %.2f\nhitDisp_right[0]: %.2f\nhitDisp_right[1]: %.2f\nhitDisp_right[2]: %.2f\n", hitColor_left[0], hitColor_left[1], hitColor_left[2], hitDisp_right[0], hitDisp_right[1], hitDisp_right[2]);

		float w;

		// printf("=====================\n
		// 	   Hit w: %.2f\n
		// 	   Grad w: %.2f\n", 
		// 	   sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor_left.tail<2>().squaredNorm())), 
		// 	   sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + d_I_x * d_I_x + d_I_y * d_I_y)));

		// Sen -> Compare the change of w
		if (wGradFlag == 1)
		{
			w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor_left.tail<2>().squaredNorm()));
		}
		if (wGradFlag == 2)
		{
			w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + d_I_x * d_I_x + d_I_y * d_I_y));
		}
		

        w = 0.5f*(w + weights[idx]);

		// huber energy
		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);

		// Sen -> Bug? Default: hw = hw*2 -> then energyLeft=0 if residual < setting_huberTH?
		// hw *= 2;

		// Sen -> energyLeft is not used in optimization，only used to determine outliers 
        //     -> outlier determination does matter for the results
		energyLeft += w * w * hw * residual * residual * (2 - hw);

		// // Sen -> if we use wStereo here -> give larger weights when hw=1!
		if (wStereoPosFlag == 1) hw *= w_energyStereo;

		{
			// Sen -> Bug? Inconsistent with original Huber and energyLeft
			if (wCorrectedFlag == 1)
			{	
				if (judgeSqrtHW)
				{
					if(hw < 1)  hw = sqrtf(hw);
				}
				else
					hw = sqrtf(hw);
			}

			if (wCorrectedFlag == 2)
			{
				//Sen -> corrected
				if (judgeSqrtHW)
				{
					if(hw < 1)  hw = sqrtf(hw * (2-hw)); 
				}
				else
					hw = sqrtf(hw * (2-hw)); 
			}
			
			if (wStereoPosFlag == 2) hw *= sqrtf(w_energyStereo);

			hw *= w;

			d_I_x *= hw;	// hitColor[1]*=hw;
			d_I_y *= hw;	// hitColor[2]*=hw;

			J->resF[idx] = residual*hw;

			J->JIdx[0][idx] = d_I_x;	// hitColor[1];
			J->JIdx[1][idx] = d_I_y;	// hitColor[2];

			J->JabF[0][idx] = 0;    // drdA * hw;
			J->JabF[1][idx] = 0;    // hw;
 
			JIdxJIdx_00 += d_I_x * d_I_x;    // hitColor[1]*hitColor[1];
			JIdxJIdx_11 += d_I_y * d_I_y;	 // hitColor[2]*hitColor[2];
			JIdxJIdx_10 += d_I_x * d_I_y;	 // hitColor[1]*hitColor[2];

			// JabJIdx_00 += drdA*hw * hitColor[1];
			// JabJIdx_01 += drdA*hw * hitColor[2];
			// JabJIdx_10 += hw * hitColor[1];
			// JabJIdx_11 += hw * hitColor[2];

			// JabJab_00 += drdA*drdA*hw*hw;
			// JabJab_01 += drdA*hw*hw;
			// JabJab_11 += hw*hw;


			// wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]);
			wJI2_sum += hw * hw * (d_I_x * d_I_x + d_I_y * d_I_y);

			if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;
			if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;

		}
	}

	J->JIdx2(0,0) = JIdxJIdx_00;
	J->JIdx2(0,1) = JIdxJIdx_10;
	J->JIdx2(1,0) = JIdxJIdx_10;
	J->JIdx2(1,1) = JIdxJIdx_11;

	J->JabJIdx(0,0) = 0; // JabJIdx_00;
	J->JabJIdx(0,1) = 0; // JabJIdx_01;
	J->JabJIdx(1,0) = 0; // JabJIdx_10;
	J->JabJIdx(1,1) = 0; // JabJIdx_11;
	
    J->Jab2(0,0) = 0;    // JabJab_00;
	J->Jab2(0,1) = 0;    // JabJab_01;
	J->Jab2(1,0) = 0;    // JabJab_01;
	J->Jab2(1,1) = 0;    // JabJab_11;


	// Sen -> Compare the energy of vs residual and normal residuals
	state_NewEnergyWithOutlier = energyLeft;

    // printf("*> Virtual stereo residual energyLeft: %.2f\n", energyLeft);

	if(energyLeft > scaleEnergyLeftTHR * std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2 * w_energyStereo * scaleWJI2SumTHR)
	{
		// Sen -> This will seriously affect results if we set too many outliers
		energyLeft = scaleEnergyLeftTHR * std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
		state_NewState = ResState::OUTLIER;
	}
	// if(energyLeft > 2*std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2*2)
	// {
	// 	energyLeft = 2*std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
	// 	state_NewState = ResState::OUTLIER;
	// }
	else
	{
		state_NewState = ResState::IN;
	}

	state_NewEnergy = energyLeft;
	return energyLeft;
}




// Sen -> All Jacobians are calculated here in linearize()
// Sen -> linearize() will be called at each update during optimization
// Sen -> linearization and FEJ are achieved by using precalc->PRE_R/tTll_0
double PointFrameResidual::linearize(CalibHessian* HCalib)
{
	state_NewEnergyWithOutlier=-1;

	if(state_state == ResState::OOB)
		{ state_NewState = ResState::OOB; return state_energy; }

	FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);
	float energyLeft=0;
	const Eigen::Vector3f* dIl = target->dI;
	//const float* const Il = target->I;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;

	// Sen -> PRE_R/tTll_0 -> inc of fixed linearizalition points
	const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0; 
	const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;

	const float * const color = point->color;
	const float * const weights = point->weights;

	Vec2f affLL = precalc->PRE_aff_mode; 
	float b0 = precalc->PRE_b0_mode; 


	Vec6f d_xi_x, d_xi_y;
	Vec4f d_C_x, d_C_y;
	float d_d_x, d_d_y;
	{
		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
				PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{ state_NewState = ResState::OOB; return state_energy; }

		centerProjectedTo = Vec3f(Ku, Kv, new_idepth);


		// diff d_idepth
		d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*SCALE_IDEPTH*HCalib->fxl();
		d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*SCALE_IDEPTH*HCalib->fyl();



		// diff calib
		d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));
		d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) * HCalib->fyli();
		d_C_x[0] = KliP[0]*d_C_x[2];
		d_C_x[1] = KliP[1]*d_C_x[3];

		d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) * HCalib->fxli();
		d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
		d_C_y[0] = KliP[0]*d_C_y[2];
		d_C_y[1] = KliP[1]*d_C_y[3];

		d_C_x[0] = (d_C_x[0]+u)*SCALE_F;
		d_C_x[1] *= SCALE_F;
		d_C_x[2] = (d_C_x[2]+1)*SCALE_C;
		d_C_x[3] *= SCALE_C;

		d_C_y[0] *= SCALE_F;
		d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
		d_C_y[2] *= SCALE_C;
		d_C_y[3] = (d_C_y[3]+1)*SCALE_C;


		d_xi_x[0] = new_idepth*HCalib->fxl();
		d_xi_x[1] = 0;
		d_xi_x[2] = -new_idepth*u*HCalib->fxl();
		d_xi_x[3] = -u*v*HCalib->fxl();
		d_xi_x[4] = (1+u*u)*HCalib->fxl();
		d_xi_x[5] = -v*HCalib->fxl();

		d_xi_y[0] = 0;
		d_xi_y[1] = new_idepth*HCalib->fyl();
		d_xi_y[2] = -new_idepth*v*HCalib->fyl();
		d_xi_y[3] = -(1+v*v)*HCalib->fyl();
		d_xi_y[4] = u*v*HCalib->fyl();
		d_xi_y[5] = u*HCalib->fyl();
	}


	{
		J->Jpdxi[0] = d_xi_x;
		J->Jpdxi[1] = d_xi_y;

		J->Jpdc[0] = d_C_x;
		J->Jpdc[1] = d_C_y;

		J->Jpdd[0] = d_d_x;
		J->Jpdd[1] = d_d_y;

	}






	float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
	float JabJIdx_00=0, JabJIdx_01=0, JabJIdx_10=0, JabJIdx_11=0;
	float JabJab_00=0, JabJab_01=0, JabJab_11=0;

	float wJI2_sum = 0;

	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;

		if(!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{ state_NewState = ResState::OOB; return state_energy; }

		projectedTo[idx][0] = Ku;
		projectedTo[idx][1] = Kv;


        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);



		float drdA = (color[idx]-b0);
		if(!std::isfinite((float)hitColor[0]))
		{ state_NewState = ResState::OOB; return state_energy; }


		float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));


        w = 0.5f*(w + weights[idx]);


		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);

        float tmpE = w*w*hw *residual*residual*(2-hw);
		energyLeft += tmpE;

		{
			// Sen -> Bug? Inconsistent with original Huber and energyLeft
			if (wCorrectedFlag == 1)
			{
				if(hw < 1) hw = sqrtf(hw);
			}

			if (wCorrectedFlag == 2)
			{
				//Sen -> corrected
				if(hw < 1) hw = sqrtf(hw * (2-hw)); 
			}

			hw = hw*w;

			hitColor[1]*=hw;
			hitColor[2]*=hw;

			J->resF[idx] = residual*hw;

			J->JIdx[0][idx] = hitColor[1];
			J->JIdx[1][idx] = hitColor[2];

			//! Ij - a*Ii - b  (a = tj*e^aj / ti*e^ai,   b = bj - a*bi) 
			J->JabF[0][idx] = drdA*hw;
			J->JabF[1][idx] = hw;

			JIdxJIdx_00+=hitColor[1]*hitColor[1];
			JIdxJIdx_11+=hitColor[2]*hitColor[2];
			JIdxJIdx_10+=hitColor[1]*hitColor[2];

			JabJIdx_00+= drdA*hw * hitColor[1];
			JabJIdx_01+= drdA*hw * hitColor[2];
			JabJIdx_10+= hw * hitColor[1];
			JabJIdx_11+= hw * hitColor[2];

			JabJab_00+= drdA*drdA*hw*hw;
			JabJab_01+= drdA*hw*hw;
			JabJab_11+= hw*hw;


			wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]);

			if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;
			if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;

		}
	}

	J->JIdx2(0,0) = JIdxJIdx_00;
	J->JIdx2(0,1) = JIdxJIdx_10;
	J->JIdx2(1,0) = JIdxJIdx_10;
	J->JIdx2(1,1) = JIdxJIdx_11;
	J->JabJIdx(0,0) = JabJIdx_00;
	J->JabJIdx(0,1) = JabJIdx_01;
	J->JabJIdx(1,0) = JabJIdx_10;
	J->JabJIdx(1,1) = JabJIdx_11;
	J->Jab2(0,0) = JabJab_00;
	J->Jab2(0,1) = JabJab_01;
	J->Jab2(1,0) = JabJab_01;
	J->Jab2(1,1) = JabJab_11;

	state_NewEnergyWithOutlier = energyLeft;

    // printf("=> Normal residual energyLeft: %.2f\n", energyLeft);

	if(energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2)
	{
		energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
		state_NewState = ResState::OUTLIER;
	}
	else
	{
		state_NewState = ResState::IN;
	}

	state_NewEnergy = energyLeft;
	return energyLeft;
}



void PointFrameResidual::debugPlot()
{
	if(state_state==ResState::OOB) return;
	Vec3b cT = Vec3b(0,0,0);

	if(freeDebugParam5==0)
	{
		float rT = 20*sqrt(state_energy/9);
		if(rT<0) rT=0; if(rT>255)rT=255;
		cT = Vec3b(0,255-rT,rT);
	}
	else
	{
		if(state_state == ResState::IN) cT = Vec3b(255,0,0);
		else if(state_state == ResState::OOB) cT = Vec3b(255,255,0);
		else if(state_state == ResState::OUTLIER) cT = Vec3b(0,0,255);
		else cT = Vec3b(255,255,255);
	}

	for(int i=0;i<patternNum;i++)
	{
		if((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0]-3 && projectedTo[i][1] < hG[0]-3 ))
			target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1],cT);
	}
}



void PointFrameResidual::applyRes(bool copyJacobians)
{
	if(copyJacobians)
	{
		if(state_state == ResState::OOB)
		{
			assert(!efResidual->isActiveAndIsGoodNEW);
			return;	// can never go back from OOB
		}
		if(state_NewState == ResState::IN)// && )
		{
			efResidual->isActiveAndIsGoodNEW=true;
			efResidual->takeDataF();
		}
		else
		{
			efResidual->isActiveAndIsGoodNEW=false;
		}
	}

	setState(state_NewState);
	state_energy = state_NewEnergy;
}
}
