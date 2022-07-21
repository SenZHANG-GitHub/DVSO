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
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/globalCalib.h"

#include "util/cnpy.h"

#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>

#include "util/Undistort.h"

#include <boost/thread.hpp>

using namespace dso;

class DispReader
{
	// Sen -> The disp npy (512 x 256, float32) from monodepth
    // Sen -> Important: We don't use undistort to distort for disp 512 x256
    //     -> The pixels might be unmatched for disp and image (undistort from 1200+x300+)
    //     -> One solution: Generate kitti 512x256 images in the same way as monodepth
    //     ->     Requires recalculation of K

public:
	DispReader(std::string filepath, std::string calibFile, std::string gammaFile, std::string vignetteFile, std::string path)
	{
		// Sen -> path is the same as DatasetReader and used for loading timestamps
		this->path = path;
		this->filepath = filepath;
		this->calibfile = calibFile;

        // Sen -> The disparities are saved as .npy with dtype=float32
        assert(CHAR_BIT * sizeof (float) == 32);
        dispNpy = cnpy::npy_load(filepath);
		dataNpy = dispNpy.data<float>();

        nImgs = dispNpy.shape[0];
        heightNpy = dispNpy.shape[1];  // 256
        widthNpy = dispNpy.shape[2];   // 512


		undistort = Undistort::getUndistorterForFile(calibFile, gammaFile, vignetteFile);


		widthOrg = undistort->getOriginalSize()[0];
		heightOrg = undistort->getOriginalSize()[1];
		width = undistort->getSize()[0];
		height = undistort->getSize()[1];


        // Should both be (512, 256)
        assert(width == widthNpy);
        assert(height == heightNpy);


		// load timestamps if possible.
		loadTimestamps();
		printf("DispFolderReader: got %d files in %s!\n", nImgs, filepath.c_str());

	}
	~DispReader()
	{
		delete undistort;
		// delete dataNpy;
	};

	Eigen::VectorXf getOriginalCalib()
	{
		return undistort->getOriginalParameter().cast<float>();
	}
	Eigen::Vector2i getOriginalDimensions()
	{
		return  undistort->getOriginalSize();
	}

	void getCalibMono(Eigen::Matrix3f &K, int &w, int &h)
	{
		K = undistort->getK().cast<float>();
		w = undistort->getSize()[0];
		h = undistort->getSize()[1];
	}

	void setGlobalCalibration()
	{
		int w_out, h_out;
		Eigen::Matrix3f K;
		getCalibMono(K, w_out, h_out);
		setGlobalCalib(w_out, h_out, K);
	}

	int getNumImages()
	{
		return nImgs;
	}

	double getTimestamp(int id)
	{
		if(timestamps.size()==0) return id*0.1f;
		if(id >= (int)timestamps.size()) return 0;
		if(id < 0) return 0;
		return timestamps[id];
	}


	ImageAndDisp* getDisp(int id)
	{
        // Sen -> Will new float[w*h] inside
		ImageAndDisp* disp = new ImageAndDisp(widthNpy, heightNpy, timestamps.size() == 0 ? 0.0 : timestamps[id]);

        float* data = disp->data;
        assert(data != 0);

        int wh = widthNpy * heightNpy;

        for (int i = 0; i < wh; i++)
        {
            // Sen -> disp->data: given w0 and h0, [h0 * w + w0]
            data[i] = dataNpy[id * wh + i];
        }
		

		// Sen -> dispNpy.data<float>() returns shared_ptr -> If we delete dataNpy, dispNpy.data will also be cleared!!!
        // delete dataNpy;
        
        return disp;
	}


	inline float* getPhotometricGamma()
	{
        printf("ERROR: we should not call this func for depth!\n");
        exit(1);

		if(undistort==0 || undistort->photometricUndist==0) return 0;
		return undistort->photometricUndist->getG();
	}


	// undistorter. [0] always exists, [1-2] only when MT is enabled.
	Undistort* undistort;
private:

	inline void loadTimestamps()
	{
		std::ifstream tr;
		std::string timesFile = path.substr(0,path.find_last_of('/')) + "/times.txt";
		tr.open(timesFile.c_str());
		while(!tr.eof() && tr.good())
		{
			std::string line;
			char buf[1000];
			tr.getline(buf, 1000);

			int id;
			double stamp;
			float exposure = 0;

			if(3 == sscanf(buf, "%d %lf %f", &id, &stamp, &exposure))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}

			else if(2 == sscanf(buf, "%d %lf", &id, &stamp))
			{
				timestamps.push_back(stamp);
				exposures.push_back(exposure);
			}
		}
		tr.close();

		// check if exposures are correct, (possibly skip)
		bool exposuresGood = ((int)exposures.size()==(int)getNumImages()) ;
		for(int i=0;i<(int)exposures.size();i++)
		{
			if(exposures[i] == 0)
			{
				// fix!
				float sum=0,num=0;
				if(i>0 && exposures[i-1] > 0) {sum += exposures[i-1]; num++;}
				if(i+1<(int)exposures.size() && exposures[i+1] > 0) {sum += exposures[i+1]; num++;}

				if(num>0)
					exposures[i] = sum/num;
			}

			if(exposures[i] == 0) exposuresGood=false;
		}


		if((int)getNumImages() != (int)timestamps.size())
		{
			printf("set timestamps and exposures to zero!\n");
			exposures.clear();
			timestamps.clear();
		}

		if((int)getNumImages() != (int)exposures.size() || !exposuresGood)
		{
			printf("set EXPOSURES to zero!\n");
			exposures.clear();
		}

		printf("got %d images and %d timestamps and %d exposures.!\n", (int)getNumImages(), (int)timestamps.size(), (int)exposures.size());
	}

    // Sen -> cnpy::NpyArray uses c++11 sharedpointer, which will auto manage its memory
    cnpy::NpyArray dispNpy;
	float* dataNpy;

	std::vector<double> timestamps;
	std::vector<float> exposures;

	int width, height;
	int widthOrg, heightOrg;

    // Sen -> shape of npy disp
    int nImgs, widthNpy, heightNpy;

	std::string path; // Sen -> Used for loading timestamps
	std::string filepath;
	std::string calibfile;

	bool isZipped;


};

