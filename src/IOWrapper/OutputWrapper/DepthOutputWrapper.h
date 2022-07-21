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


#pragma once
#include "boost/thread.hpp"
#include "util/MinimalImage.h"
#include "IOWrapper/Output3DWrapper.h"
#include <opencv2/highgui/highgui.hpp>

#include <iomanip>
#include <string>
#include <sstream>


#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"

namespace dso
{

class FrameHessian;
class CalibHessian;
class FrameShell;


namespace IOWrap
{

class DepthOutputWrapper : public Output3DWrapper
{
public:
        MinimalImageB16* depth;

        inline DepthOutputWrapper(int w, int h)
        {
            printf("OUT: Created DepthOutputWrapper\n");
            this->depth = new MinimalImageB16(w, h);
        }

        virtual ~DepthOutputWrapper()
        {
            printf("OUT: Destroyed DepthOutputWrapper\n");
            delete this->depth;
        }



        virtual void publishKeyframes( std::vector<FrameHessian*> &frames, bool final, CalibHessian* HCalib) override
        {
            if (final)
            {
                int nframe = 0;
                for(FrameHessian* f : frames)
                {
                    nframe += 1;
                    // printf("OUT: KF %d (%s) (id %d, tme %f): %d active, %d marginalized, %d out points, %d immature points. CameraToWorld:\n",
                    //     f->frameID,
                    //     final ? "final" : "non-final",
                    //     f->shell->incoming_id,
                    //     f->shell->timestamp,
                    //     (int)f->pointHessians.size(), (int)f->pointHessiansMarginalized.size(), (int) f->pointHessiansOut.size(), (int)f->immaturePoints.size());
                    // std::cout << f->shell->id <<
                    //     " " << f->shell->camToWorld.translation().transpose()<<
                    //     " " << f->shell->camToWorld.so3().unit_quaternion().x()<<
                    //     " " << f->shell->camToWorld.so3().unit_quaternion().y()<<
                    //     " " << f->shell->camToWorld.so3().unit_quaternion().z()<<
                    //     " " << f->shell->camToWorld.so3().unit_quaternion().w() << "\n";


                    // Sen -> Change this part to save depth images (sparse though)
                    // int maxWrite = 5;
                    // for(PointHessian* p : f->pointHessiansMarginalized)
                    // {
                    //     printf("OUT: Example Point x=%.1f, y=%.1f, idepth=%f, idepth std.dev. %f, %d inlier-residuals\n",
                    //         p->u, p->v, p->idepth_scaled, sqrt(1.0f / p->idepth_hessian), p->numGoodResiduals );
                    //     maxWrite--;
                    //     if(maxWrite==0) break;
                    // }

                    this->depth->setBlack();
                    for (PointHessian* p : f->pointHessiansMarginalized)
                    {
                        this->depth->setPixel1(p->u, p->v, (unsigned short)(256 / p->idepth_scaled));
                    }
                    std::ostringstream oss;
                    oss << std::setfill('0') << std::setw(6) << f->shell->id << ".png";
                    std::string outname = oss.str();
                    cv::imwrite(outname, cv::Mat(depth->h, depth->w, CV_16U, depth->data));
                }
                if (nframe > 1)
                {
                    printf("Error: %d frames processed in one run of publishKeyframes\n", nframe);
                }
            }
            
        }


};



}



}
