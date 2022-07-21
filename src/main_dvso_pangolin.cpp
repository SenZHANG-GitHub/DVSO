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



#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "util/settings.h"
#include "util/globalFuncs.h"
#include "util/DatasetReader.h"
#include "util/DispReader.h"
#include "util/globalCalib.h"

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"


#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"
#include "IOWrapper/OutputWrapper/DepthOutputWrapper.h"


std::string vignette = "";
std::string gammaCalib = "";
std::string source = "";
// std::string depths = "";
std::string disps_left = "";
std::string disps_right = "";
std::string calib = "";
double rescale = 1;
bool reverse = false;
bool disableROS = false;
int start=0;
int end=100000;
bool prefetch = false;
float playbackSpeed=0;	// 0 for linearize (play as fast as possible, while sequentializing tracking & mapping). otherwise, factor on timestamps.
bool preload=false;
bool useSampleOutput=false;
bool useDepthOutput=false;
// bool loadDepth=false;
bool loadDispLeft=false;
bool loadDispRight=false;

int mode=0;

bool firstRosSpin=false;

using namespace dso;


void my_exit_handler(int s)
{
	printf("Caught signal %d\n",s);
	exit(1);
}

void exitThread()
{
	struct sigaction sigIntHandler;
	sigIntHandler.sa_handler = my_exit_handler;
	sigemptyset(&sigIntHandler.sa_mask);
	sigIntHandler.sa_flags = 0;
	sigaction(SIGINT, &sigIntHandler, NULL);

	firstRosSpin=true;
	while(true) pause();
}



void settingsDefault(int preset)
{
	printf("\n=============== PRESET Settings: ===============\n");
	if(preset == 0 || preset == 1)
	{
		printf("DEFAULT settings:\n"
				"- %s real-time enforcing\n"
				"- 2000 active points\n"
				"- 5-7 active frames\n"
				"- 1-6 LM iteration each KF\n"
				"- original image resolution\n", preset==0 ? "no " : "1x");

		playbackSpeed = (preset==0 ? 0 : 1);
		preload = preset==1;
		setting_desiredImmatureDensity = 1500;
		setting_desiredPointDensity = 2000;
		setting_minFrames = 5;
		setting_maxFrames = 7;
		setting_maxOptIterations=6;
		setting_minOptIterations=1;

		setting_logStuff = false;
	}

	if(preset == 2 || preset == 3)
	{
		printf("FAST settings:\n"
				"- %s real-time enforcing\n"
				"- 800 active points\n"
				"- 4-6 active frames\n"
				"- 1-4 LM iteration each KF\n"
				"- 424 x 320 image resolution\n", preset==0 ? "no " : "5x");

		playbackSpeed = (preset==2 ? 0 : 5);
		preload = preset==3;
		setting_desiredImmatureDensity = 600;
		setting_desiredPointDensity = 800;
		setting_minFrames = 4;
		setting_maxFrames = 6;
		setting_maxOptIterations=4;
		setting_minOptIterations=1;

		benchmarkSetting_width = 424;
		benchmarkSetting_height = 320;

		setting_logStuff = false;
	}

	printf("==============================================\n");
}






void parseArgument(char* arg)
{
	int option;
	float foption;
	char buf[1000];


    if(1==sscanf(arg,"sampleoutput=%d",&option))
    {
        if(option==1)
        {
            useSampleOutput = true;
            printf("USING SAMPLE OUTPUT WRAPPER!\n");
        }
        return;
    }

	if(1==sscanf(arg,"depthoutput=%d",&option))
    {
        if(option==1)
        {
            useDepthOutput = true;
            printf("USING DEPTH OUTPUT WRAPPER!\n");
        }
        return;
    }

    if(1==sscanf(arg,"quiet=%d",&option))
    {
        if(option==1)
        {
            setting_debugout_runquiet = true;
            printf("QUIET MODE, I'll shut up!\n");
        }
        return;
    }

	if(1==sscanf(arg,"preset=%d",&option))
	{
		settingsDefault(option);
		return;
	}


	if(1==sscanf(arg,"rec=%d",&option))
	{
		if(option==0)
		{
			disableReconfigure = true;
			printf("DISABLE RECONFIGURE!\n");
		}
		return;
	}



	if(1==sscanf(arg,"noros=%d",&option))
	{
		if(option==1)
		{
			disableROS = true;
			disableReconfigure = true;
			printf("DISABLE ROS (AND RECONFIGURE)!\n");
		}
		return;
	}

	if(1==sscanf(arg,"nolog=%d",&option))
	{
		if(option==1)
		{
			setting_logStuff = false;
			printf("DISABLE LOGGING!\n");
		}
		return;
	}
	if(1==sscanf(arg,"reverse=%d",&option))
	{
		if(option==1)
		{
			reverse = true;
			printf("REVERSE!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nogui=%d",&option))
	{
		if(option==1)
		{
			disableAllDisplay = true;
			printf("NO GUI!\n");
		}
		return;
	}
	if(1==sscanf(arg,"nomt=%d",&option))
	{
		if(option==1)
		{
			multiThreading = false;
			printf("NO MultiThreading!\n");
		}
		return;
	}
	if(1==sscanf(arg,"prefetch=%d",&option))
	{
		if(option==1)
		{
			prefetch = true;
			printf("PREFETCH!\n");
		}
		return;
	}
	if(1==sscanf(arg,"start=%d",&option))
	{
		start = option;
		printf("START AT %d!\n",start);
		return;
	}
	if(1==sscanf(arg,"end=%d",&option))
	{
		end = option;
		printf("END AT %d!\n",start);
		return;
	}

	if(1==sscanf(arg,"files=%s",buf))
	{
		source = buf;
		printf("loading data from %s!\n", source.c_str());
		return;
	}

	if(1==sscanf(arg,"disps_left=%s",buf))
	{
		disps_left = buf;
        loadDispLeft = true;
		printf("loading disps_left data from %s!\n", disps_left.c_str());
		return;
	}
    
	if(1==sscanf(arg,"disps_right=%s",buf))
	{
		disps_right = buf;
        loadDispRight = true;
		printf("loading disps_right data from %s!\n", disps_right.c_str());
		return;
	}

    // if(1==sscanf(arg,"depths=%s",buf))
	// {
	// 	depths = buf;
    //     loadDepth = true;
	// 	printf("loading depths data from %s!\n", depths.c_str());
	// 	return;
	// }



	if(1==sscanf(arg,"calib=%s",buf))
	{
		calib = buf;
		printf("loading calibration from %s!\n", calib.c_str());
		return;
	}

	if(1==sscanf(arg,"vignette=%s",buf))
	{
		vignette = buf;
		printf("loading vignette from %s!\n", vignette.c_str());
		return;
	}

	if(1==sscanf(arg,"gamma=%s",buf))
	{
		gammaCalib = buf;
		printf("loading gammaCalib from %s!\n", gammaCalib.c_str());
		return;
	}

	if(1==sscanf(arg,"rescale=%f",&foption))
	{
		rescale = foption;
		printf("RESCALE %f!\n", rescale);
		return;
	}

	if(1==sscanf(arg,"speed=%f",&foption))
	{
		playbackSpeed = foption;
		printf("PLAYBACK SPEED %f!\n", playbackSpeed);
		return;
	}

	if(1==sscanf(arg,"wStereo=%f",&foption))
	{
		dso::w_energyStereo = foption;
		// dso::benchmark_initializerSlackFactor = 1 + dso::w_energyStereo;
		dso::benchmark_initializerSlackFactor = 2;
		printf("w_energyStereo: %f!\n", dso::w_energyStereo);
		printf("benchmark_initializerSlackFactor: %f!\n", dso::benchmark_initializerSlackFactor);
		return;
	}

	if(1==sscanf(arg,"wStereoPosFlag=%s",buf))
	{
		std::string tmp = buf;
		if (tmp == "Before") dso::wStereoPosFlag = 1;
		if (tmp == "After") dso::wStereoPosFlag = 2;
		printf("wStereoPosFlag: %s!\n", tmp.c_str());
		return;
	}

	if(1==sscanf(arg,"wCorrectedFlag=%s",buf))
	{
		std::string tmp = buf;
		if (tmp == "Ori") dso::wCorrectedFlag = 1;
		if (tmp == "Corr") dso::wCorrectedFlag = 2;
		printf("wCorrectedFlag: %s!\n", tmp.c_str());
		return;
	}

	if(1==sscanf(arg,"wGradFlag=%s",buf))
	{
		std::string tmp = buf;
		if (tmp == "Hit") dso::wGradFlag = 1;
		if (tmp == "Grad") dso::wGradFlag = 2;
		printf("wGradFlag: %s!\n", tmp.c_str());
		return;
	}

	if(1==sscanf(arg,"scaleEnergyLeftTHR=%f",&foption))
	{
		dso::scaleEnergyLeftTHR = foption;
		printf("scaleEnergyLeftTHR: %f!\n", dso::scaleEnergyLeftTHR);
		return;
	}

	if(1==sscanf(arg,"scaleWJI2SumTHR=%f",&foption))
	{
		dso::scaleWJI2SumTHR = foption;
		printf("scaleWJI2SumTHR: %f!\n", dso::scaleWJI2SumTHR);
		return;
	}

	if(1==sscanf(arg,"useVS=%d",&option))
	{
		// useVS=0: useVirtualStereo = false; useVS=1: useVirtualStereo=true
		if(option==1)
		{
			useVirtualStereo = true;
			printf("We are now using useVS!\n");
		} 
		else if (option == 0)
		{
			useVirtualStereo = false;
			printf("We do not use useVS!\n");
		}
		else
		{
			printf("Error: useVS must be 0 or 1!\n");
			exit(1);
		}
		
		return;
	} //judgeSqrtHW

	if(1==sscanf(arg,"judgeHW=%d",&option))
	{
		if(option==1)
		{
			judgeSqrtHW = true;
			printf("We are now using judgeHW!\n");
		} 
		else if (option == 0)
		{
			judgeSqrtHW = false;
			printf("We do not use judgeHW!\n");
		}
		else
		{
			printf("Error: judgeHW must be 0 or 1!\n");
			exit(1);
		}
		
		return;
	}

    if(1==sscanf(arg,"checkWarpValid=%d",&option))
	{
		if(option==1) 
		{
            // true is the default of checkWarpValid
			checkWarpValid = true;
			printf("We are now using checkWarpValid!\n");
		} 
		else if (option == 0)
		{
			checkWarpValid = false;
			printf("We do not use checkWarpValid!\n");
		}
		
		return;
	}

    if(1==sscanf(arg,"maskWarpGrad=%d",&option))
	{
		if(option==1) 
		{
            // true is the default of checkWarpValid
			maskWarpGrad = true;
			printf("We are now using maskWarpGrad!\n");
		} 
		else if (option == 0)
		{
			maskWarpGrad = false;
			printf("We do not use maskWarpGrad!\n");
		}
		
		return;
	}


	if(1==sscanf(arg,"save=%d",&option))
	{
		if(option==1)
		{
			debugSaveImages = true;
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("rm -rf images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			if(42==system("mkdir images_out")) printf("system call returned 42 - what are the odds?. This is only here to shut up the compiler.\n");
			printf("SAVE IMAGES!\n");
		}
		return;
	}

	if(1==sscanf(arg,"mode=%d",&option))
	{

		mode = option;
		if(option==0)
		{
			printf("PHOTOMETRIC MODE WITH CALIBRATION!\n");
		}
		if(option==1)
		{
			printf("PHOTOMETRIC MODE WITHOUT CALIBRATION!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = 0; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = 0; //-1: fix. >=0: optimize (with prior, if > 0).
		}
		if(option==2)
		{
			printf("PHOTOMETRIC MODE WITH PERFECT IMAGES!\n");
			setting_photometricCalibration = 0;
			setting_affineOptModeA = -1; //-1: fix. >=0: optimize (with prior, if > 0).
			setting_affineOptModeB = -1; //-1: fix. >=0: optimize (with prior, if > 0).
            setting_minGradHistAdd=3;
		}
		return;
	}

	printf("could not parse argument \"%s\"!!!!\n", arg);
}



int main( int argc, char** argv )
{
	//setlocale(LC_ALL, "");
	for(int i=1; i<argc;i++)
		parseArgument(argv[i]);

	// Sen -> Debug args must be given
	{
		if (abs(dso::w_energyStereo + 1) < 1e-6)
		{
			printf("ERROR: wStereo must be given!");
			exit(1);
		}
		if (!(dso::wStereoPosFlag==1 || dso::wStereoPosFlag==2))
		{
			printf("ERROR: wStereoPosFlag must be given!");
			exit(1);
		}
		if (!(dso::wCorrectedFlag==1 || dso::wCorrectedFlag==2))
		{
			printf("ERROR: wCorrectedFlag must be given!");
			exit(1);
		}
		if (!(dso::wGradFlag==1 || dso::wGradFlag==2))
		{
			printf("ERROR: wGradFlag must be given!");
			exit(1);
		}
	}

    // //////////////////////////////////////////
	// // Sen -> Test cnpy
	// cnpy::NpyArray arr = cnpy::npy_load("/home/szha2609/data/kitti/odometry/monodepth_disps/disparities.npy");
    // assert(CHAR_BIT * sizeof (float) == 32);
    // float* tmpdata = arr.data<float>();
	// float a2 = tmpdata[0];
	// std::cout<<"a2: "<<a2<<std::endl;

	// // Sen -> arr.data<float>() returns shared_ptr, we should not delete tmpdata!!!
    // // delete tmpdata;
    // float* tmpdata2 = arr.data<float>();
	// float a1 = tmpdata2[0];
	// std::cout<<"a1: "<<a1<<std::endl;
    
	// if (abs(a1 - a2) < 1e-6)
	// {
	// 	std::cout<<"good"<<std::endl;
	// } else
	// {
	// 	std::cout<<"bad"<<std::endl;
	// }

	// /////////////////////////////////////////

	// hook crtl+C.
	boost::thread exThread = boost::thread(exitThread);

    
	ImageFolderReader* reader = new ImageFolderReader(source, calib, gammaCalib, vignette);
	reader->setGlobalCalibration();

    bool loadDisp = loadDispLeft && loadDispRight;
	if (!loadDisp)
	{
		printf("Error: disps_left and disps_right must be given!");
		exit(1);
	}

	// // Sen => Do not use calib for depth_reader currently
    // if (!(loadDepth || loadDisp))
    // {
    //     printf("Error: Need at least specify either depths or disps_left and disps_right!");
    //     exit(1);
    // }
    // if(loadDepth && loadDisp)
    // {
    //     printf("Error: Can only specify one of either depths or disps_left and disps_right!");
    //     exit(1);
    // }

    // if (loadDepth)
    // {
    //     printf("Error: Disable depth_reader for stereo concern in FullSystem(..)!");
    //     exit(1);
    //     DepthFolderReader* depth_reader = new DepthFolderReader(depths, calib, gammaCalib, vignette);

    //     if (reader->getNumImages() != depth_reader->getNumImages()) 
    //     {
    //         printf("ERROR: Loaded different numbers of images and depths!");
    //         exit(1);
    //     }
    // }


	DispReader* disp_left_reader = new DispReader(disps_left, calib, gammaCalib, vignette, source);
	DispReader* disp_right_reader = new DispReader(disps_right, calib, gammaCalib, vignette, source);
	assert(disp_left_reader->getNumImages() == disp_right_reader->getNumImages());

	if (reader->getNumImages() != disp_left_reader->getNumImages()) 
	{
		printf("ERROR: Loaded different numbers of images and disparities!");
		exit(1);
	}

	
	

	if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
	{
		printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
		exit(1);
	}


	int lstart=start;
	int lend = end;
	int linc = 1;
	if(reverse)
	{
		printf("REVERSE!!!!");
		lstart=end-1;
		if(lstart >= reader->getNumImages())
			lstart = reader->getNumImages()-1;
		lend = start;
		linc = -1;
	}



	FullSystem* fullSystem = new FullSystem();
	fullSystem->setGammaFunction(reader->getPhotometricGamma());
	fullSystem->linearizeOperation = (playbackSpeed==0);






    IOWrap::PangolinDSOViewer* viewer = 0;
	if(!disableAllDisplay)
    {
        viewer = new IOWrap::PangolinDSOViewer(wG[0],hG[0], false);
        fullSystem->outputWrapper.push_back(viewer);
    }


	if(useDepthOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::DepthOutputWrapper(wG[0],hG[0]));


    if(useSampleOutput)
        fullSystem->outputWrapper.push_back(new IOWrap::SampleOutputWrapper());




    // to make MacOS happy: run this in dedicated thread -- and use this one to run the GUI.
    std::thread runthread([&]() {
        std::vector<int> idsToPlay;
        std::vector<double> timesToPlayAt;
        for(int i=lstart;i>= 0 && i< reader->getNumImages() && linc*i < linc*lend;i+=linc)
        {
            idsToPlay.push_back(i);
            if(timesToPlayAt.size() == 0)
            {
                timesToPlayAt.push_back((double)0);
            }
            else
            {
                double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size()-1]);
                double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size()-2]);
                timesToPlayAt.push_back(timesToPlayAt.back() +  fabs(tsThis-tsPrev)/playbackSpeed);
            }
        }


        std::vector<ImageAndExposure*> preloadedImages;
		// std::vector<ImageAndDepth*> preloadedDepths; // Sen -> May not be used 
        std::vector<ImageAndDisp*> preloadedDispsLeft;
        std::vector<ImageAndDisp*> preloadedDispsRight;

        if(preload)
        {
            printf("LOADING ALL IMAGES!\n");
            for(int ii=0;ii<(int)idsToPlay.size(); ii++)
            {
                int i = idsToPlay[ii];
                preloadedImages.push_back(reader->getImage(i));
                // if (loadDepth) preloadedDepths.push_back(depth_reader->getDepth(i));

				preloadedDispsLeft.push_back(disp_left_reader->getDisp(i));
				preloadedDispsRight.push_back(disp_right_reader->getDisp(i));
            }
        }

        struct timeval tv_start;
        gettimeofday(&tv_start, NULL);
        clock_t started = clock();
        double sInitializerOffset=0;


        for(int ii=0;ii<(int)idsToPlay.size(); ii++)
        {
            if(!fullSystem->initialized)	// if not initialized: reset start time.
            {
                gettimeofday(&tv_start, NULL);
                started = clock();
                sInitializerOffset = timesToPlayAt[ii];
            }

            int i = idsToPlay[ii];


            ImageAndExposure* img;
			// if (loadDepth) ImageAndDepth* depth;
			ImageAndDisp* disp_left;
			ImageAndDisp* disp_right;

            if(preload)
			{
				img = preloadedImages[ii];
                // if (loadDepth) depth = preloadedDepths[ii];
				disp_left = preloadedDispsLeft[ii];
				disp_right = preloadedDispsRight[ii];
			}
            else
			{
				img = reader->getImage(i);
				// if (loadDepth) depth = depth_reader->getDepth(i);
				disp_left = disp_left_reader->getDisp(i);
				disp_right = disp_right_reader->getDisp(i);
			}
                



			// Sen -> skipFrame will only be set true if playbackSpeed > 0
            bool skipFrame=false;
            if(playbackSpeed!=0)
            {
                struct timeval tv_now; gettimeofday(&tv_now, NULL);
                double sSinceStart = sInitializerOffset + ((tv_now.tv_sec-tv_start.tv_sec) + (tv_now.tv_usec-tv_start.tv_usec)/(1000.0f*1000.0f));

                if(sSinceStart < timesToPlayAt[ii])
                    usleep((int)((timesToPlayAt[ii]-sSinceStart)*1000*1000));
                else if(sSinceStart > timesToPlayAt[ii]+0.5+0.1*(ii%2))
                {
                    printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                    skipFrame=true;
                }
            }


			assert(img->timestamp == disp_left->timestamp);
			assert(disp_left->timestamp == disp_right->timestamp);
            
            if(!skipFrame) fullSystem->addActiveFrame(img, disp_left->data, disp_right->data, i);




            delete img;
			// if (loadDepth) delete depth;
			
			delete disp_left;
			delete disp_right;

            if(fullSystem->initFailed || setting_fullResetRequested)
            {
                if(ii < 250 || setting_fullResetRequested)
                {
                    printf("RESETTING!\n");

                    std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                    delete fullSystem;

                    for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                    fullSystem = new FullSystem();
                    fullSystem->setGammaFunction(reader->getPhotometricGamma());
                    fullSystem->linearizeOperation = (playbackSpeed==0);


                    fullSystem->outputWrapper = wraps;

                    setting_fullResetRequested=false;
                }
            }

            if(fullSystem->isLost)
            {
                    printf("LOST!!\n");
                    break;
            }

        }

        fullSystem->blockUntilMappingIsFinished();
		
		// clock() -> clock_t: time of current thread by CLOCKS_PER_SEC
		// gettimeofdata -> timeval: time of the system shared by all threads
        clock_t ended = clock();
        struct timeval tv_end;
        gettimeofday(&tv_end, NULL);

		// Sen -> We print results after all images haee
        fullSystem->printResult("result.txt");


        int numFramesProcessed = abs(idsToPlay[0]-idsToPlay.back());
        double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0])-reader->getTimestamp(idsToPlay.back()));
        double MilliSecondsTakenSingle = 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC);
        double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f);
        printf("\n======================"
                "\n%d Frames (%.1f fps)"
                "\n%.2fms per frame (single core); "
                "\n%.2fms per frame (multi core); "
                "\n%.3fx (single core); "
                "\n%.3fx (multi core); "
                "\n======================\n\n",
                numFramesProcessed, numFramesProcessed/numSecondsProcessed,
                MilliSecondsTakenSingle/numFramesProcessed,
                MilliSecondsTakenMT / (float)numFramesProcessed,
                1000 / (MilliSecondsTakenSingle/numSecondsProcessed),
                1000 / (MilliSecondsTakenMT / numSecondsProcessed));
        //fullSystem->printFrameLifetimes();
        if(setting_logStuff)
        {
            std::ofstream tmlog;
            tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
            tmlog << 1000.0f*(ended-started)/(float)(CLOCKS_PER_SEC*reader->getNumImages()) << " "
                  << ((tv_end.tv_sec-tv_start.tv_sec)*1000.0f + (tv_end.tv_usec-tv_start.tv_usec)/1000.0f) / (float)reader->getNumImages() << "\n";
            tmlog.flush();
            tmlog.close();
        }

    });


    if(viewer != 0)
        viewer->run();

	// block the main thread until runthread has finished
    runthread.join();

	for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
	{
		ow->join();
		delete ow;
	}



	printf("DELETE FULLSYSTEM!\n");
	delete fullSystem;

	printf("DELETE READER!\n");
	delete reader;

    // if (loadDepth)
    // {
    //     printf("DELETE DEPTH READER!\n");
	//     delete depth_reader;
    // }

	printf("DELETE DISP READERS (LEFT & RIGHT)!\n");
	delete disp_left_reader;
	delete disp_right_reader;
	

	printf("EXIT NOW!\n");
	return 0;
}
