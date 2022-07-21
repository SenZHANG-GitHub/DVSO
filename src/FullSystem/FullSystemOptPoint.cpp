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
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace dso
{


PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
	// residuals: ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
	int nres = 0;
	for(FrameHessian* fh : frameHessians)
	{
		if(fh != point->host) 
		{
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
            // // SenFlagVSImmature
			// residuals[nres].host = point->host;
			nres++; 
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

    // // SenFlagVSImmature
	// Sen -> Add virtual stereo residual
	// residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
	// residuals[nres].state_NewState = ResState::OUTLIER;
	// residuals[nres].state_state = ResState::IN;
	// residuals[nres].target = point->host->frame_right;
	// residuals[nres].host = point->host;
	// nres++;

	bool print = false;//rand()%50==0;

	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;

	// Sen ->  Use epipolar line search results for initialization
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;


	float tmpOutlierTHSlack = 1000;
	for(int i=0;i<nres;i++)
	{
		// Sen -> d in Hdd and bd refers to inverse depth
		// Sen -> GN optimization here: (1)fix frame-related variables (2) only update idepth
        // // SenFlagVSImmature
		// if (i < nres - 1)
        {
			lastEnergy += point->linearizeResidual(&Hcalib, tmpOutlierTHSlack, residuals+i, lastHdd, lastbd, currentIdepth);
        }
		// else
		// {
		// 	// lastEnergy += point->linearizeVirtualStereoResidual(&Hcalib, tmpOutlierTHSlack, residuals+i, lastHdd, lastbd, currentIdepth);
		// 	// Sen -> We don't use vs residual energy for lastEnergy and newEnergy comparison
		// 	float tmpHdd = lastHdd;
		// 	float tmpbd = lastbd;
		
		// 	float tmpE = point->linearizeVirtualStereoResidual(&Hcalib, tmpOutlierTHSlack, residuals+i, lastHdd, lastbd, currentIdepth);

		// 	float deltaHdd = (lastHdd - tmpHdd) / (tmpHdd / (nres-1));
		// 	float deltabd = (lastbd - tmpbd) / (tmpbd / (nres-1));
		// 	float deltaE = tmpE / (lastEnergy / (nres-1));

		// 	printf("init-> deltaHdd=%.2f, deltabd=%.2f, deltaE=%.2f\n", deltaHdd, deltabd, deltaE);
		// 	float deltaStop = 1;
		// }

		residuals[i].state_state = residuals[i].state_NewState;
		residuals[i].state_energy = residuals[i].state_NewEnergy;
	}

	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
	{
		if(print)
			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
			nres, lastHdd,lastEnergy,currentIdepth);

	float lambda = 0.1;
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
	{
		float H = lastHdd;
		H *= 1+lambda;
		float step = (1.0/H) * lastbd;
		float newIdepth = currentIdepth - step;

		float newHdd=0; float newbd=0; float newEnergy=0;

		tmpOutlierTHSlack = 1;
		for(int i=0;i<nres;i++)
		{
            // // SenFlagVSImmature
			// if (i < nres - 1)
            {
				newEnergy += point->linearizeResidual(&Hcalib, tmpOutlierTHSlack, residuals+i, newHdd, newbd, newIdepth);
            }
            // // SenFlagVSImmature
			// else 
			// { 
			// 	float tmpHdd = newHdd;
			// 	float tmpbd = newbd;

			// 	float tmpE = point->linearizeVirtualStereoResidual(&Hcalib, tmpOutlierTHSlack, residuals+i, newHdd, newbd, newIdepth);

			// 	float deltaHdd = (newHdd - tmpHdd) / (tmpHdd / (nres-1));
			// 	float deltabd = (newbd - tmpbd) / (tmpbd / (nres-1));
			// 	float deltaE = tmpE / (newEnergy / (nres-1));

			// 	printf("gn opt-> deltaHdd=%.2f, deltabd=%.2f, deltaE=%.2f\n", deltaHdd, deltabd, deltaE);
			// 	float deltaStop = 1;

			// }
		}

		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
		{
			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		if(newEnergy < lastEnergy)
		{
			currentIdepth = newIdepth;
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;
			for(int i=0;i<nres;i++)
			{
				residuals[i].state_state = residuals[i].state_NewState;
				residuals[i].state_energy = residuals[i].state_NewEnergy;
			}

			lambda *= 0.5;
		}
		else
		{
			lambda *= 5;
		}

		if(fabsf(step) < 0.0001*currentIdepth)
			break;
	}

	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}


	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	if(numGoodRes < minObs)
	{
		if(print) printf("OptPoint: OUTLIER!\n");
		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}



	PointHessian* p = new PointHessian(point, &Hcalib);
	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));}

	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;

	// Sen -> It's important we set idepth for PointHessian here
	//     -> Inside PointHessian(pt, Hcalib): 
	//	    	-> setIdepthScaled((pt->idepth_max + pt->idepth_min)*0.5);
	//			-> u = pt->u; v = pt->v; energyTH = pt->energyTH;

	// Sen -> (1) setIdepthScaled and setIdepth do the same thing：Get values for this->idepth, this->idepth_scaled
    //     -> (2) difference: multiplied by scaled_factor or scale_factor_inverse
    //     -> (3) setIdepthZero() and setIdepth() both use scale_idepth for *_scaled
	p->setIdepthZero(currentIdepth);
	p->setIdepth(currentIdepth);
	p->setPointStatus(PointHessian::ACTIVE);


	// // SenFlagVS
	// Sen -> Assume point p will exist in frame_right
    bool pValid = true;
    if (checkWarpValid && !p->isWarpValid())
        pValid = false;

	if (useVirtualStereo && pValid)
	{
		PointFrameResidual* r = new PointFrameResidual(p, p->host, p->host->frame_right);
		r->state_NewEnergy = r->state_energy = 0;
		r->state_NewState = ResState::OUTLIER;
		r->setState(ResState::IN);
		r->virtualStereoResFlag = true; // Sen -> set as stereo residue
		p->residuals.push_back(r);
	}
	
    // // SenFlagVSImmature
	// for (int i=0; i < nres - 1; i++)
    for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN)
		{
			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
			r->state_NewEnergy = r->state_energy = 0;
			r->state_NewState = ResState::OUTLIER;
			r->setState(ResState::IN);
			p->residuals.push_back(r);

			if(r->target == frameHessians.back())
			{
				p->lastResiduals[0].first = r;
				p->lastResiduals[0].second = ResState::IN;
			}
			else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]))
			{
				p->lastResiduals[1].first = r;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	if(print) printf("point activated!\n");

	statistics_numActivatedPoints++;
	return p;
}


// // Sen -> Original optimizeImmaturePoint in DSO for backup
// PointHessian* FullSystem::optimizeImmaturePoint(
// 		ImmaturePoint* point, int minObs,
// 		ImmaturePointTemporaryResidual* residuals)
// {
// 	int nres = 0;
// 	for(FrameHessian* fh : frameHessians)
// 	{
// 		if(fh != point->host) 
// 		{
// 			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
// 			residuals[nres].state_NewState = ResState::OUTLIER;
// 			residuals[nres].state_state = ResState::IN;
// 			residuals[nres].target = fh;
// 			nres++; 
// 		}
// 	}
// 	assert(nres == ((int)frameHessians.size())-1);

// 	bool print = false;//rand()%50==0;

// 	float lastEnergy = 0;
// 	float lastHdd=0;
// 	float lastbd=0;

// 	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;


// 	for(int i=0;i<nres;i++)
// 	{
// 		lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i, lastHdd, lastbd, currentIdepth);
// 		residuals[i].state_state = residuals[i].state_NewState;
// 		residuals[i].state_energy = residuals[i].state_NewEnergy;
// 	}

// 	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
// 	{
// 		if(print)
// 			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
// 				nres, lastHdd, lastEnergy);
// 		return 0;
// 	}

// 	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
// 			nres, lastHdd,lastEnergy,currentIdepth);

// 	float lambda = 0.1;
// 	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
// 	{
// 		float H = lastHdd;
// 		H *= 1+lambda;
// 		float step = (1.0/H) * lastbd;
// 		float newIdepth = currentIdepth - step;

// 		float newHdd=0; float newbd=0; float newEnergy=0;
// 		for(int i=0;i<nres;i++)
// 			newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i, newHdd, newbd, newIdepth);

// 		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
// 		{
// 			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
// 					nres,
// 					newHdd,
// 					lastEnergy);
// 			return 0;
// 		}

// 		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
// 				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
// 				iteration,
// 				log10(lambda),
// 				"",
// 				lastEnergy, newEnergy, newIdepth);

// 		if(newEnergy < lastEnergy)
// 		{
// 			currentIdepth = newIdepth;
// 			lastHdd = newHdd;
// 			lastbd = newbd;
// 			lastEnergy = newEnergy;
// 			for(int i=0;i<nres;i++)
// 			{
// 				residuals[i].state_state = residuals[i].state_NewState;
// 				residuals[i].state_energy = residuals[i].state_NewEnergy;
// 			}

// 			lambda *= 0.5;
// 		}
// 		else
// 		{
// 			lambda *= 5;
// 		}

// 		if(fabsf(step) < 0.0001*currentIdepth)
// 			break;
// 	}

// 	if(!std::isfinite(currentIdepth))
// 	{
// 		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
// 		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
// 	}


// 	int numGoodRes=0;
// 	for(int i=0;i<nres;i++)
// 		if(residuals[i].state_state == ResState::IN) numGoodRes++;

// 	if(numGoodRes < minObs)
// 	{
// 		if(print) printf("OptPoint: OUTLIER!\n");
// 		return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
// 	}



// 	PointHessian* p = new PointHessian(point, &Hcalib);
// 	if(!std::isfinite(p->energyTH)) {delete p; return (PointHessian*)((long)(-1));}

// 	p->lastResiduals[0].first = 0;
// 	p->lastResiduals[0].second = ResState::OOB;
// 	p->lastResiduals[1].first = 0;
// 	p->lastResiduals[1].second = ResState::OOB;

// 	// Sen -> It's important we set idepth for PointHessian here
// 	//     -> Inside PointHessian(pt, Hcalib): 
// 	//	    	-> setIdepthScaled((pt->idepth_max + pt->idepth_min)*0.5);
// 	//			-> u = pt->u; v = pt->v; energyTH = pt->energyTH;

// 	p->setIdepthZero(currentIdepth);
// 	p->setIdepth(currentIdepth);
// 	p->setPointStatus(PointHessian::ACTIVE);


// 	// // SenFlagVS
// 	// // Sen -> Assume point p will exist in frame_right
// 	// PointFrameResidual* r = new PointFrameResidual(p, p->host, p->host->frame_right);
// 	// r->state_NewEnergy = r->state_energy = 0;
// 	// r->state_NewState = ResState::OUTLIER;
// 	// r->setState(ResState::IN);
// 	// r->virtualStereoResFlag = true; // Sen -> set as stereo residue
// 	// p->residuals.push_back(r);

// 	for(int i=0;i<nres;i++)
// 		if(residuals[i].state_state == ResState::IN)
// 		{
// 			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
// 			r->state_NewEnergy = r->state_energy = 0;
// 			r->state_NewState = ResState::OUTLIER;
// 			r->setState(ResState::IN);
// 			p->residuals.push_back(r);

// 			if(r->target == frameHessians.back())
// 			{
// 				p->lastResiduals[0].first = r;
// 				p->lastResiduals[0].second = ResState::IN;
// 			}
// 			else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]))
// 			{
// 				p->lastResiduals[1].first = r;
// 				p->lastResiduals[1].second = ResState::IN;
// 			}
// 		}

// 	if(print) printf("point activated!\n");

// 	statistics_numActivatedPoints++;
// 	return p;
// }



}
