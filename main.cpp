#include <iostream>
#include <fstream>
#include <omp.h>
#include "ImagePro.h"
#include "IlluminationFlowUlti.h"
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "glog/logging.h"

using namespace std;
using namespace cv;
Point2f point;
bool addRemovePt = false;
int VisTrack(char *PATH)
{
	char Fname[200];
	Mat gray, prevGray, tImg, backGround;

	int nframe = 0, count = 0, width, height, x, y;
	vector<Point2f> points[2];
	for (int ii = 155; ii < 180; ii++)
	{
		sprintf(Fname, "%s/In/(%d).png", PATH, ii);
		Mat gray = imread(Fname, 0);
		if (gray.empty())
			continue;
		count++;
		width = gray.cols, height = gray.rows;
		cvtColor(gray, backGround, CV_GRAY2RGB);

		sprintf(Fname, "%s/Sep/ (%d).png", PATH, ii);
		tImg = imread(Fname, 0);
		if (tImg.empty())
			continue;

		sprintf(Fname, "%s/WO/IllumTrack_%05d.txt", PATH, ii);
		FILE *fp = fopen(Fname, "r");
		Point2f xy;
		while (fscanf(fp, "%f %f\n", &xy.x, &xy.y) != EOF)
			points[1].push_back(xy);
		fclose(fp);

		for (int jj = 0; jj < points[1].size(); jj++)
		{
			x = (int)points[1][jj].x, y = (int)points[1][jj].y;
			if (x < 400 || x > 1250 || y < 250 || y>height - 150)
				continue;
			if (tImg.data[x + y*width] < 2)
				continue;
			circle(backGround, points[1][jj], 5, Scalar(83, 185, 255), -1, 8);
		}

		sprintf(Fname, "%s/TrackIllum_%d.png", PATH, ii);
		imwrite(Fname, backGround);

		points[1].clear();
	}
	return 0;

}
int TrackOpenCVLK(int start, int nframes, char *PATH)
{
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(21, 21), winSize(31, 31);

	const int MAX_COUNT = 5000;
	bool needToInit = false;

	Mat gray, prevGray, tImg, backGround;
	vector<Point2f> points[2];
	//Mat white(1080,1920, CV_8UC3, Scalar(255,255,255));
	char Fname[200];

	int nframe = 0, count = 0, width, height, x, y;
	while (nframe < nframes)
	{
		sprintf(Fname, "%s/Sep/ (%d).png", PATH, nframe + start);
		Mat gray = imread(Fname, 0);
		if (gray.empty())
			continue;
		count++;
		width = gray.cols, height = gray.rows;

		//Create background
		//sprintf(Fname, "%s/Sep/ (%d).png", PATH, nframe + start);
		sprintf(Fname, "%s/In/(%d).png", PATH, nframe + start);
		tImg = imread(Fname, 0);
		if (tImg.empty())
			continue;
		cvtColor(tImg, backGround, CV_GRAY2RGB);

		if (nframe == 0)
		{
			/*{
				sprintf(Fname, "%s/IllumTrack_%05d.txt", PATH, nframe + start);
				FILE *fp = fopen(Fname, "r");
				Point2f xy;
				for (int jj = 0; jj < 914; jj++)
				{
				fscanf(fp, "%f %f\n", &xy.x, &xy.y);
				points[1].push_back(xy);
				}
				fclose(fp);
				}*/

			// automatic initialization
			goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, points[1], subPixWinSize, Size(-1, -1), termcrit);
			addRemovePt = false;

			for (int jj = 0; jj < points[1].size(); jj++)
			{
				x = (int)points[1][jj].x, y = (int)points[1][jj].y;
				if (x < 400 || x > 1500 || y < 100 || y>height - 150)
					continue;
				if (tImg.data[x + y*width] < 2)
					continue;
				circle(backGround, points[1][jj], 5, Scalar(83, 185, 255), -1, 8);
			}

			//sprintf(Fname, "%s/TextTrack_%05d.txt", PATH, nframe+start);
			sprintf(Fname, "%s/IllumTrack_%05d.txt", PATH, nframe + start);
			FILE *fp = fopen(Fname, "w+");
			for (int jj = 0; jj < points[1].size(); jj++)
				fprintf(fp, "%.4f %.4f\n", points[1].at(jj).x, points[1].at(jj).y);
			fclose(fp);
		}
		else if (!points[0].empty())
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 1, termcrit, 0, 0.001);

			sprintf(Fname, "%s/ID_%05d.txt", PATH, nframe + start);
			FILE *fp = fopen(Fname, "w+");
			for (int jj = 0; jj < points[1].size(); jj++)
				fprintf(fp, "%d\n", (int)status[jj]);
			fclose(fp);

			size_t i, k;
			for (i = k = 0; i < points[1].size(); i++)
			{
				if (addRemovePt)
				{
					if (norm(point - points[1][i]) <= 5)
					{
						addRemovePt = false;
						continue;
					}
				}

				if (!status[i])
					continue;

				points[1][k++] = points[1][i];
				x = (int)points[1][i].x, y = (int)points[1][i].y;
				if (x < 400 || x > 1500 || y < 100 || y>height - 150)
					continue;
				if (tImg.data[x + y*width] < 2)
					continue;
				circle(backGround, points[1][i], 5, Scalar(83, 185, 255), -1, 8);
			}
			points[1].resize(k);

			//sprintf(Fname, "%s/TextTrack_%05d.txt", PATH, nframe+start);
			sprintf(Fname, "%s/IllumTrack_%05d.txt", PATH, nframe + start);
			fp = fopen(Fname, "w+");
			for (int jj = 0; jj < points[1].size(); jj++)
				fprintf(fp, "%.4f %.4f\n", points[1].at(jj).x, points[1].at(jj).y);
			fclose(fp);
		}

		if (addRemovePt && points[1].size() < (size_t)MAX_COUNT)
		{
			vector<Point2f> tmp;
			tmp.push_back(point);
			cornerSubPix(gray, tmp, winSize, cvSize(-1, -1), termcrit);
			points[1].push_back(tmp[0]);
			addRemovePt = false;
		}

		needToInit = false;
		imshow("LK Demo", backGround);
		sprintf(Fname, "%s/TrackIllum_%d.png", PATH, nframe + start);
		imwrite(Fname, backGround);

		char c = (char)waitKey(10);
		if (c == 27)
			break;

		std::swap(points[1], points[0]);
		swap(prevGray, gray);
		nframe++;
	}

	return 0;
}
int TrackDIC(int start, int nframes, int width, int height, LKParameters LKarg, char *PATH)
{
	TermCriteria termcrit(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03);
	Size subPixWinSize(21, 21), winSize(31, 31);

	const int MAX_COUNT = 5000;
	bool needToInit = false;

	Mat gray, prevGray, tImg, backGround;
	vector<Point2f> points, npoints, fnpoints; Point2f pts;

	char Fname[200];
	int ii, jj, length = width*height, nframe = 0, count = 0, x, y;
	int hsubset = LKarg.hsubset, InterpAlgo = LKarg.InterpAlgo;

	CPoint2 PR, PT;
	unsigned char *RefImg = new unsigned char[length], *TarImg = new unsigned char[length];
	double *RefPara = new double[length], *TarPara = new double[length];
	while (nframe < nframes)
	{
		sprintf(Fname, "%s/Sep/wLC1P1_%05d.png", PATH, nframe + start);
		Mat gray = imread(Fname, 0);
		if (gray.empty())
			continue;
		else
		{
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					TarImg[ii + jj*width] = gray.data[ii + (height - 1 - jj)*width];
			Generate_Para_Spline(TarImg, TarPara, width, height, InterpAlgo);
		}

		//Create background
		sprintf(Fname, "%s/Sep/wLC1P1_%05d.png", PATH, nframe + start);
		tImg = imread(Fname, 0);
		if (tImg.empty())
			continue;
		cvtColor(tImg, backGround, CV_GRAY2RGB);

		count++;
		if (nframe == 0)
		{
			{
				sprintf(Fname, "%s/TextTrack_%05d.txt", PATH, nframe + start);
				FILE *fp = fopen(Fname, "r");

				for (int jj = 0; jj < 764; jj++)
				{
					fscanf(fp, "%f %f\n", &pts.x, &pts.y);
					npoints.push_back(pts);
				}
				fclose(fp);
			}
			/*// automatic initialization
			goodFeaturesToTrack(gray, npoints, MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
			cornerSubPix(gray, npoints, subPixWinSize, Size(-1,-1), termcrit);

			for(int jj=0; jj<npoints.size(); jj++)
			{
			x = (int)npoints[jj].x, y = (int) npoints[jj].y;
			if(x<10 || x > width - 10 || y < 10 || y>height -10)
			continue;
			if(tImg.data[x+y*width] >250)
			continue;
			circle( backGround, npoints[jj], 5, Scalar(83,185,255), -1, 8);
			}*/
			sprintf(Fname, "%s/IllumTrack_%05d.txt", PATH, nframe + start);
			FILE *fp = fopen(Fname, "w+");
			for (int jj = 0; jj < npoints.size(); jj++)
				fprintf(fp, "%.4f %.4f\n", npoints.at(jj).x, npoints.at(jj).y);
			fclose(fp);
		}
		else
		{
			vector<uchar> status;
			vector<float> err;
			if (prevGray.empty())
				gray.copyTo(prevGray);
			calcOpticalFlowPyrLK(prevGray, gray, points, npoints, status, err, winSize, 1, termcrit, 0, 0.001);

			fnpoints.clear();
			for (ii = 0; ii < npoints.size(); ii++)
			{
				if (points.at(ii).x < 1 || points.at(ii).y < 1)
					continue;
				PR.x = points.at(ii).x, PR.y = height - 1 - points.at(ii).y;
				PT.x = npoints.at(ii).x, PT.y = height - 1 - npoints.at(ii).y;
				//if(TMatching(RefPara, TarPara, hsubset, width, height, width, height, 1, PR, PT,LKarg.DIC_Algo, LKarg.Convergence_Criteria, LKarg.ZNCCThreshold, LKarg.IterMax, InterpAlgo, fufv, true) > LKarg.ZNCCThreshold)
				{
					//PT.x += fufv[0], PT.y += fufv[1];
					pts.x = PT.x, pts.y = height - 1 - PT.y;
					fnpoints.push_back(pts);
				}
			}

			npoints.clear();
			for (ii = 0; ii < fnpoints.size(); ii++)
			{
				pts.x = fnpoints[ii].x, pts.y = fnpoints[ii].y;
				npoints.push_back(pts);
				x = (int)fnpoints[ii].x, y = (int)fnpoints[ii].y;
				if (x<10 || x > width - 10 || y < 10 || y>height - 10)
					continue;
				if (tImg.data[x + y*width] > 250)
					continue;
				circle(backGround, npoints[ii], 5, Scalar(83, 185, 255), -1, 8);
			}

			sprintf(Fname, "%s/IllumTrack_%05d.txt", PATH, nframe + start);
			FILE *fp = fopen(Fname, "w+");
			for (int jj = 0; jj < points.size(); jj++)
				fprintf(fp, "%.4f %.4f\n", points.at(jj).x, points.at(jj).y);
			fclose(fp);
		}

		imshow("DIC tracking", backGround);
		sprintf(Fname, "%s/Sep/TrackIllum_%d.png", PATH, nframe + start);
		imwrite(Fname, backGround);

		char c = (char)waitKey(10);
		if (c == 27)
			break;

		for (ii = 0; ii < length; ii++)
			RefPara[ii] = TarPara[ii];
		nframe++;

		std::swap(npoints, points);
		swap(prevGray, gray);
	}
	return 0;
}

int CamProMatchingBK(char *PATH, int nCams, int nPros, int frameID, int width, int height, int pwidth, int pheight, int nchannels, LKParameters LKArg, CPoint *ROI, double triThresh = 2.0, bool saveWarping = false, bool Simulation = false)
{
	//Double check checker size!!!
	int ii, jj, kk, camID, proID;
	int length = width*height, plength = pwidth*pheight;

	int  step = LKArg.step, hsubset = LKArg.hsubset;
	int Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, Convergence_Criteria = LKArg.Convergence_Criteria, Analysis_Speed = LKArg.Analysis_Speed, DICAlgo = LKArg.DIC_Algo, InterpAlgo = LKArg.InterpAlgo;
	double PSSDab_thresh = LKArg.PSSDab_thresh, Gsigma = LKArg.Gsigma;
	int nSeedPoints = 5000; //should be enough

	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}
	double ProEpipole[3];

	//Load images
	char *Img1 = new char[length*nchannels];
	char *Img2 = new char[plength*nchannels];
	bool *cROI = new bool[length];
	bool *lpROI_calculated = new bool[length];
	for (ii = 0; ii < length; ii++)
	{
		cROI[ii] = false; // 1 - Valid, 0 - Other
		lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
	}

	char Fname[100];
	IplImage *view = 0;
	CPoint2 *SparseCorres1 = new CPoint2[nSeedPoints];
	CPoint2 *SparseCorres2 = new CPoint2[nSeedPoints];
	CPoint2 *disparity = new CPoint2[length];
	float *WarpingParas = 0;
	float *CDepth = new float[length];
	for (camID = 0; camID < nCams; camID++)
	{
#pragma omp critical
		cout << "Run CamPro on frame " << frameID << " of Cam #" << camID + 1;
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID + 1, frameID); view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
		if (view == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 2;
		}
		else
		{
			for (kk = 0; kk < nchannels; kk++)
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						Img1[ii + (height - 1 - jj)*width + kk*length] = view->imageData[nchannels*ii + nchannels*jj*width + kk];
			cvReleaseImage(&view);
		}

		sprintf(Fname, "%s/Results/CamPro/C%d_%05d.xyz", PATH, camID + 1, frameID); FILE *fp = fopen(Fname, "w+"); 	fclose(fp);
		for (proID = 0; proID < nPros; proID++)
		{
#pragma omp critical
			cout << " and Projector #" << proID + 1 << endl;
			sprintf(Fname, "%s/ProjectorPattern.png", PATH);  view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
			if (view == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 2;
			}
			else
			{
				for (kk = 0; kk < nchannels; kk++)
					for (jj = 0; jj < pheight; jj++)
						for (ii = 0; ii < pwidth; ii++)
							Img2[ii + (pheight - 1 - jj)*pwidth + kk*plength] = view->imageData[nchannels*ii + nchannels*jj*pwidth + kk];
				cvReleaseImage(&view);
			}

			//Load scale information
			double CamProScale[10];
			sprintf(Fname, "%s/CamProScale.txt", PATH);
			fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 3;
			}
			else
			{
				for (ii = 0; ii < nCams; ii++)
				{
					fscanf(fp, "%lf ", &CamProScale[ii]);
					CamProScale[ii] = 1.0 / CamProScale[ii];
				}
				fclose(fp);
			}

			//Load seeds points
			sprintf(Fname, "%s/Sparse/C%dP%d_%05d.txt", PATH, camID + 1, proID + 1, frameID);
			fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 3;
			}
			else
			{
				nSeedPoints = 0;
				while (fscanf(fp, "%lf %lf ", &SparseCorres1[nSeedPoints].x, &SparseCorres1[nSeedPoints].y) != EOF)
				{
					SparseCorres1[nSeedPoints].y = SparseCorres1[nSeedPoints].y;
					nSeedPoints++;
				}
				fclose(fp);

				sprintf(Fname, "%s/Sparse/P%dC%d_%05d.txt", PATH, proID + 1, camID + 1, frameID); fp = fopen(Fname, "r");
				if (fp == NULL)
				{
					cout << "Cannot load " << Fname << endl;
					return 3;
				}
				else
				{
					for (ii = 0; ii < nSeedPoints; ii++)
						fscanf(fp, "%d %lf %lf ", &jj, &SparseCorres2[ii].x, &SparseCorres2[ii].y);
					fclose(fp);
				}
			}

			//Customize the ROI
			int maxX = 0, maxY = 0, minX = width, minY = height;
			for (ii = 0; ii<nSeedPoints; ii++)
			{
				if (SparseCorres1[ii].x > maxX)
					maxX = (int)SparseCorres1[ii].x;
				else if (SparseCorres1[ii].x < minX)
					minX = (int)SparseCorres1[ii].x;
				if (SparseCorres1[ii].y > maxY)
					maxY = (int)SparseCorres1[ii].y;
				else if (SparseCorres1[ii].y < minY)
					minY = (int)SparseCorres1[ii].y;
			}
			maxX = maxX>width - 2 ? maxX : maxX + 2;
			minX = minX < 2 ? minX : minX - 2;
			maxY = maxY > height - 2 ? maxY : maxY + 2;
			minY = minY < 2 ? minY : minY - 2;

			for (ii = 0; ii < length; ii++)
				cROI[ii] = false;
			if (ROI == NULL)
				for (jj = minY; jj < maxY; jj++)
					for (ii = minX; ii < maxX; ii++)
						cROI[ii + jj*width] = true;
			else
				for (jj = ROI[0].y; jj < ROI[1].y; jj++)
					for (ii = ROI[0].x; ii < ROI[1].x; ii++)
						cROI[ii + jj*width] = true;

			if (Simulation)
			{
				double *BlurImage = new double[length];
				sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID + 1, frameID);
				view = cvLoadImage(Fname, 0);
				if (!view)
				{
					cout << "Cannot load Images" << Fname << endl;
					return 2;
				}
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						BlurImage[ii + (height - 1 - jj)*width] = (double)((int)((unsigned char)view->imageData[nchannels*ii + jj*width]));
				Gaussian_smooth_Double(BlurImage, BlurImage, height, width, 255.0, 2.0); //severely smooth out the image to determine the boundary

				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii<width; ii++)
						if (BlurImage[ii + jj*width] > 5.0)
							cROI[ii + jj*width] = true;
				delete[]BlurImage;
			}

			for (ii = 0; ii < length; ii++)
			{
				disparity[ii].x = 0.0;
				disparity[ii].y = 0.0;
			}

			if (saveWarping)
			{
				WarpingParas = new float[length * 6];
				for (ii = 0; ii < 6 * length; ii++)
					WarpingParas[ii] = 0.0f;
			}

			double Pmat[24], K[18], distortion[26];
			if (proID == 0)
			{
				Pmat[0] = DInfo.K[0], Pmat[1] = DInfo.K[1], Pmat[2] = DInfo.K[2], Pmat[3] = 0.0,
					Pmat[4] = DInfo.K[3], Pmat[5] = DInfo.K[4], Pmat[6] = DInfo.K[5], Pmat[7] = 0.0,
					Pmat[8] = DInfo.K[6], Pmat[9] = DInfo.K[7], Pmat[10] = DInfo.K[8], Pmat[11] = 0.0;
			}
			else
			{
				Pmat[0] = DInfo.P[12 * (proID - 1)], Pmat[1] = DInfo.P[12 * (proID - 1) + 1], Pmat[2] = DInfo.P[12 * (proID - 1) + 2], Pmat[3] = DInfo.P[12 * (proID - 1) + 3],
					Pmat[4] = DInfo.P[12 * (proID - 1) + 4], Pmat[5] = DInfo.P[12 * (proID - 1) + 5], Pmat[6] = DInfo.P[12 * (proID - 1) + 6], Pmat[7] = DInfo.P[12 * (proID - 1) + 7],
					Pmat[8] = DInfo.P[12 * (proID - 1) + 8], Pmat[9] = DInfo.P[12 * (proID - 1) + 9], Pmat[10] = DInfo.P[12 * (proID - 1) + 10], Pmat[11] = DInfo.P[12 * (proID - 1) + 11];
			}
			for (ii = 0; ii < 12; ii++)
				Pmat[12 + ii] = DInfo.P[12 * (camID - 1 + nPros) + ii];
			for (ii = 0; ii < 9; ii++)
				K[ii] = DInfo.K[9 * proID + ii], K[ii + 9] = DInfo.K[9 * (camID + nPros) + ii];
			for (ii = 0; ii < 13; ii++)
				distortion[ii] = DInfo.distortion[13 * proID + ii], distortion[13 + ii] = DInfo.distortion[13 * (camID + nPros) + ii];

			GreedyMatching(Img1, Img2, disparity, lpROI_calculated, cROI, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, nchannels, width, height, pwidth, pheight, CamProScale[camID], ProEpipole, WarpingParas, Pmat, K, distortion, triThresh);

#pragma omp critical
			{
				CPoint2 Ppts, Cpts; CPoint3 WC;
				sprintf(Fname, "%s/Results/CamPro/C%d_%05d.xyz", PATH, camID + 1, frameID); fp = fopen(Fname, "a");
				for (jj = 0; jj < height; jj++)
				{
					for (ii = 0; ii < width; ii++)
					{
						if (abs(WarpingParas[ii + jj*width] + WarpingParas[ii + jj*width + length]) < 0.001)
							CDepth[ii + jj*width] = 0.0f;
						else
						{
							Cpts.x = 1.0*ii, Cpts.y = 1.0*jj;
							Ppts.x = WarpingParas[ii + jj*width] + ii, Ppts.y = WarpingParas[ii + jj*width + length] + jj;

							Undo_distortion(Cpts, K, distortion);
							Undo_distortion(Ppts, K + 9, distortion + 13);
							Stereo_Triangulation2(&Ppts, &Cpts, Pmat, Pmat + 12, &WC);
							CDepth[ii + jj*width] = (float)WC.z;
							fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
						}
					}
				}
				fclose(fp);
				//sprintf(Fname, "%s/Results/CamPro/C%dP%d_%05d.ijz", PATH, camID+1, proID+1, frameID); WriteGridBinary(Fname, CDepth, width, height);
			}

			if (saveWarping)
			{
				for (ii = 0; ii < 6; ii++)
				{
					sprintf(Fname, "%s/Results/CamPro/C%dP%dp%d_%05d.dat", PATH, camID + 1, proID + 1, ii, frameID);
					WriteGridBinary(Fname, WarpingParas + ii*length, width, height);
				}
			}
		}
	}

	delete[]disparity;
	delete[]CDepth;
	delete[]cROI;
	delete[]lpROI_calculated;
	delete[]Img1;
	delete[]Img2;
	delete[]SparseCorres1;
	delete[]SparseCorres2;
	delete[]WarpingParas;

	return 0;
}
int CamProMatching(char *PATH, int nCams, int nPros, int frameID, int width, int height, int pwidth, int pheight, int nchannels, LKParameters LKArg, CPoint *ROI, bool CheckMatching = false, bool FlowVerification = false, double triThresh = 2.0, bool saveWarping = false, bool Simulation = false)
{
	//Double check checker size!!!
	int ii, jj, kk, ll, camID;
	int length = width*height, plength = pwidth*pheight;

	int  step = LKArg.step, hsubset = LKArg.hsubset;
	int Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, Convergence_Criteria = LKArg.Convergence_Criteria, Analysis_Speed = LKArg.Analysis_Speed, DICAlgo = LKArg.DIC_Algo, InterpAlgo = LKArg.InterpAlgo;
	double PSSDab_thresh = LKArg.PSSDab_thresh, Gsigma = LKArg.Gsigma;
	int nSeedPoints = 5000; //should be enough

	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		//return 1;
	}
	double ProEpipole[3];

	//Load images
	char *Img1 = new char[length*nchannels];
	char *Img2 = new char[plength*nchannels];
	bool *cROI = new bool[length];
	bool *lpROI_calculated = new bool[length];

	char Fname[100];
	IplImage *view = 0;
	CPoint2 *SparseCorres1 = new CPoint2[nSeedPoints];
	CPoint2 *SparseCorres2 = new CPoint2[nSeedPoints];
	CPoint2 *disparity = new CPoint2[length];
	float *WarpingParas = 0;
	float *CDepth = new float[length];

	if (saveWarping)
	{
		WarpingParas = new float[length * 6 * nPros];
		for (ii = 0; ii < 6 * nPros*length; ii++)
			WarpingParas[ii] = 0.0f;
	}

	for (camID = 0; camID < nCams; camID++)
	{
#pragma omp critical
		cout << "Run CamPro on frame " << frameID << " of Cam #" << camID + 1;

		//Read camera image
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID + 1, frameID); view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
		if (view == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 2;
		}
		else
		{
			for (kk = 0; kk < nchannels; kk++)
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						Img1[ii + (height - 1 - jj)*width + kk*length] = view->imageData[nchannels*ii + nchannels*jj*width + kk];
			cvReleaseImage(&view);
		}

		if (nPros == 2)
			LKArg.checkZNCC = true;

		for (int proID = 0; proID < nPros; proID++)
		{
#pragma omp critical
			cout << " and Projector #" << proID + 1 << endl;
			//Read projector image
			sprintf(Fname, "%s/ProjectorPattern%d.png", PATH, proID + 1);  view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
			if (view == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 2;
			}
			else
			{
				for (kk = 0; kk < nchannels; kk++)
					for (jj = 0; jj < pheight; jj++)
						for (ii = 0; ii < pwidth; ii++)
							Img2[ii + (pheight - 1 - jj)*pwidth + kk*plength] = view->imageData[nchannels*ii + nchannels*jj*pwidth + kk];
				cvReleaseImage(&view);
			}

			//Load scale information
			double CamProScale[10];
			sprintf(Fname, "%s/CamProScale.txt", PATH);
			FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 3;
			}
			else
			{
				for (ii = 0; ii < nCams; ii++)
				{
					fscanf(fp, "%lf ", &CamProScale[ii]);
					CamProScale[ii] = 1.0 / CamProScale[ii];
				}
				fclose(fp);
			}

			//Load seeds points
			sprintf(Fname, "%s/Sparse/C%dP%d_%05d.txt", PATH, camID + 1, proID + 1, frameID);
			fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 3;
			}
			else
			{
				nSeedPoints = 0;
				while (fscanf(fp, "%lf %lf ", &SparseCorres1[nSeedPoints].x, &SparseCorres1[nSeedPoints].y) != EOF)
				{
					SparseCorres1[nSeedPoints].y = SparseCorres1[nSeedPoints].y;
					nSeedPoints++;
				}
				fclose(fp);

				sprintf(Fname, "%s/Sparse/P%dC%d_%05d.txt", PATH, proID + 1, camID + 1, frameID); fp = fopen(Fname, "r");
				if (fp == NULL)
				{
					cout << "Cannot load " << Fname << endl;
					return 3;
				}
				else
				{
					for (ii = 0; ii < nSeedPoints; ii++)
						fscanf(fp, "%d %lf %lf ", &jj, &SparseCorres2[ii].x, &SparseCorres2[ii].y);
					fclose(fp);
				}
			}

			bool breakflag = false;
			float *tW = new float[6 * length];
			for (ii = 0; ii < 6 && !breakflag; ii++)
			{
				sprintf(Fname, "%s/Results/CamPro/C%dP%dp%d_%05d.dat", PATH, camID + 1, proID + 1, ii, frameID);
				if (!ReadGridBinary(Fname, tW + ii*length, width, height))
					breakflag = true;
			}
			if (!breakflag)
				for (ii = 0; ii < 6 && !breakflag; ii++)
					for (jj = 0; jj < length; jj++)
						WarpingParas[jj + (6 * proID + ii)*length] = tW[jj + ii*length];
			delete[]tW;

			//Customize the ROI
			int maxX = 0, maxY = 0, minX = width, minY = height;
			for (ii = 0; ii<nSeedPoints; ii++)
			{
				if (SparseCorres1[ii].x > maxX)
					maxX = (int)SparseCorres1[ii].x;
				else if (SparseCorres1[ii].x < minX)
					minX = (int)SparseCorres1[ii].x;
				if (SparseCorres1[ii].y > maxY)
					maxY = (int)SparseCorres1[ii].y;
				else if (SparseCorres1[ii].y < minY)
					minY = (int)SparseCorres1[ii].y;
			}
			maxX = maxX>width - 2 ? maxX : maxX + 2;
			minX = minX < 2 ? minX : minX - 2;
			maxY = maxY > height - 2 ? maxY : maxY + 2;
			minY = minY < 2 ? minY : minY - 2;

			for (ii = 0; ii < length; ii++)
				cROI[ii] = false;
			if (ROI == NULL)
				for (jj = minY; jj < maxY; jj++)
					for (ii = minX; ii < maxX; ii++)
						cROI[ii + jj*width] = true;
			else
				for (jj = ROI[0].y; jj < ROI[1].y; jj++)
					for (ii = ROI[0].x; ii < ROI[1].x; ii++)
						cROI[ii + jj*width] = true;

			if (Simulation)
			{
				double *BlurImage = new double[length];
				sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID + 1, frameID);
				view = cvLoadImage(Fname, 0);
				if (!view)
				{
					cout << "Cannot load Images" << Fname << endl;
					return 2;
				}
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						BlurImage[ii + (height - 1 - jj)*width] = (double)((int)((unsigned char)view->imageData[nchannels*ii + jj*width]));
				Gaussian_smooth_Double(BlurImage, BlurImage, height, width, 255.0, 2.0); //severely smooth out the image to determine the boundary

				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii<width; ii++)
						if (BlurImage[ii + jj*width] > 5.0)
							cROI[ii + jj*width] = true;
				delete[]BlurImage;
			}

			for (ii = 0; ii < length; ii++)
			{
				if (abs(WarpingParas[ii + 6 * proID*length]) + abs(WarpingParas[ii + (1 + 6 * proID)*length]) > 0.1)
				{
					lpROI_calculated[ii] = true; 	// 1 - Calculated, 0 - Other
					disparity[ii].x = WarpingParas[ii + 6 * proID*length], disparity[ii].y = WarpingParas[ii + (1 + 6 * proID)*length];
				}
				else
				{
					lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
					disparity[ii].x = 0.0, disparity[ii].y = 0.0;
				}
			}

			double Pmat[24], K[18], distortion[26];
			if (proID == 0)
			{
				Pmat[0] = DInfo.K[0], Pmat[1] = DInfo.K[1], Pmat[2] = DInfo.K[2], Pmat[3] = 0.0,
					Pmat[4] = DInfo.K[3], Pmat[5] = DInfo.K[4], Pmat[6] = DInfo.K[5], Pmat[7] = 0.0,
					Pmat[8] = DInfo.K[6], Pmat[9] = DInfo.K[7], Pmat[10] = DInfo.K[8], Pmat[11] = 0.0;
			}
			else
			{
				Pmat[0] = DInfo.P[12 * (proID - 1)], Pmat[1] = DInfo.P[12 * (proID - 1) + 1], Pmat[2] = DInfo.P[12 * (proID - 1) + 2], Pmat[3] = DInfo.P[12 * (proID - 1) + 3],
					Pmat[4] = DInfo.P[12 * (proID - 1) + 4], Pmat[5] = DInfo.P[12 * (proID - 1) + 5], Pmat[6] = DInfo.P[12 * (proID - 1) + 6], Pmat[7] = DInfo.P[12 * (proID - 1) + 7],
					Pmat[8] = DInfo.P[12 * (proID - 1) + 8], Pmat[9] = DInfo.P[12 * (proID - 1) + 9], Pmat[10] = DInfo.P[12 * (proID - 1) + 10], Pmat[11] = DInfo.P[12 * (proID - 1) + 11];
			}
			for (ii = 0; ii < 12; ii++)
				Pmat[12 + ii] = DInfo.P[12 * (camID - 1 + nPros) + ii];
			for (ii = 0; ii < 9; ii++)
				K[ii] = DInfo.K[9 * proID + ii], K[ii + 9] = DInfo.K[9 * (camID + nPros) + ii];
			for (ii = 0; ii < 13; ii++)
				distortion[ii] = DInfo.distortion[13 * proID + ii], distortion[13 + ii] = DInfo.distortion[13 * (camID + nPros) + ii];

			GreedyMatching(Img1, Img2, disparity, lpROI_calculated, cROI, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, nchannels, width, height, pwidth, pheight, CamProScale[camID], ProEpipole, WarpingParas + 6 * proID*length, Pmat, K, distortion, triThresh);
			if (CheckMatching)
				MatchingCheck(Img1, Img2, WarpingParas + 6 * proID*length, LKArg, 1.0 / CamProScale[camID], nchannels, width, height, pwidth, pheight);

			/*if(saveWarping)
			{
			for(ii=0; ii<6; ii++)
			{
			sprintf(Fname, "%s/Results/CamPro/C%dP%dp%d_%05d.dat", PATH, camID+1, proID+1, ii, frameID);
			WriteGridBinary(Fname, WarpingParas+(ii+6*proID)*length, width, height);
			//ReadGridBinary(Fname, WarpingParas+(ii+6*proID)*length, width, height);
			}
			}*/

			if (FlowVerification)
			{
				//Load illum flow to verify the pure illum regions
				double start = omp_get_wtime();
				char Fname1[200], Fname2[200];
				float *IllumFlow = new float[2 * length];
				sprintf(Fname1, "%s/Flow/FX1_%05d.dat", PATH, frameID), sprintf(Fname2, "%s/Flow/FY1_%05d.dat", PATH, frameID);
				if (!ReadFlowBinary(Fname1, Fname2, IllumFlow, IllumFlow + length, width, height))
				{
					cout << "Cannot load illumination flow for frame" << frameID << endl;
					delete[]IllumFlow;
					return 1;
				}
				cout << "Loaded illumination flow in " << omp_get_wtime() - start << endl;

				//Clean campro warping
				for (jj = 0; jj < length; jj++)
					if (abs(IllumFlow[jj]) + abs(IllumFlow[jj + length]) < 0.01)
						for (ii = 0; ii < 6; ii++)
							WarpingParas[jj + (ii + proID * 6)*length] = 0.0;

				delete[]IllumFlow;
			}
			/*#pragma omp critical
						{
						CPoint2 Ppts, Cpts; CPoint3 WC;
						sprintf(Fname, "%s/Results/CamPro/C%dP%d_%05d.xyz", PATH, camID + 1, proID + 1, frameID); fp = fopen(Fname, "w+");
						for (jj = 0; jj < height; jj++)
						{
						for (ii = 0; ii < width; ii++)
						{
						if (abs(WarpingParas[ii + jj*width + 6 * proID*length] + WarpingParas[ii + jj*width + (1 + 6 * proID)*length]) < 0.001)
						CDepth[ii + jj*width] = 0.0f;
						else
						{
						Cpts.x = 1.0*ii, Cpts.y = 1.0*jj;
						Ppts.x = WarpingParas[ii + jj*width + (6 * proID)*length] + ii, Ppts.y = WarpingParas[ii + jj*width + (1 + 6 * proID)*length] + jj;

						Undo_distortion(Ppts, K, distortion), Undo_distortion(Cpts, K + 9, distortion + 13);
						Stereo_Triangulation2(&Ppts, &Cpts, Pmat, Pmat + 12, &WC);
						CDepth[ii + jj*width] = (float)WC.z;
						fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
						}
						}
						}
						fclose(fp);
						}*/
		}

		//Create 3D + clean all possible bad points
		sprintf(Fname, "%s/Results/CamPro/C%d_%05d.xyz", PATH, camID + 1, frameID); FILE *fp = fopen(Fname, "w+");
		int nfails, nvalidViews, validView[4];
		double u, v, denum, reprojectionError, thresh, ValidPmat[12 * 4];
		CPoint2 ValidPts[4]; CPoint3 WC;
		for (jj = 0; jj < height; jj++)
		{
			for (ii = 0; ii < width; ii++)
			{
				nvalidViews = 0;
				for (int proID = 0; proID < nPros; proID++)
				{
					validView[proID] = 0;
					if (abs(WarpingParas[ii + jj*width + 6 * proID*length]) + abs(WarpingParas[ii + jj*width + (1 + 6 * proID)*length]) > 0.001)
					{
						validView[nvalidViews] = proID;
						ValidPts[nvalidViews].x = WarpingParas[ii + jj*width + 6 * proID*length] + ii, ValidPts[nvalidViews].y = WarpingParas[ii + jj*width + (1 + 6 * proID)*length] + jj;
						Undo_distortion(ValidPts[nvalidViews], DInfo.K + 9 * proID, DInfo.distortion + 13 * proID);

						if (proID == 0)
						{
							ValidPmat[0] = DInfo.K[0], ValidPmat[1] = DInfo.K[1], ValidPmat[2] = DInfo.K[2], ValidPmat[3] = 0.0,
								ValidPmat[4] = DInfo.K[3], ValidPmat[5] = DInfo.K[4], ValidPmat[6] = DInfo.K[5], ValidPmat[7] = 0.0,
								ValidPmat[8] = DInfo.K[6], ValidPmat[9] = DInfo.K[7], ValidPmat[10] = DInfo.K[8], ValidPmat[11] = 0.0;
						}
						else
						{
							ValidPmat[12 * nvalidViews] = DInfo.P[12 * (proID - 1)], ValidPmat[12 * nvalidViews + 1] = DInfo.P[12 * (proID - 1) + 1], ValidPmat[12 * nvalidViews + 2] = DInfo.P[12 * (proID - 1) + 2], ValidPmat[12 * nvalidViews + 3] = DInfo.P[12 * (proID - 1) + 3],
								ValidPmat[12 * nvalidViews + 4] = DInfo.P[12 * (proID - 1) + 4], ValidPmat[12 * nvalidViews + 5] = DInfo.P[12 * (proID - 1) + 5], ValidPmat[12 * nvalidViews + 6] = DInfo.P[12 * (proID - 1) + 6], ValidPmat[12 * nvalidViews + 7] = DInfo.P[12 * (proID - 1) + 7],
								ValidPmat[12 * nvalidViews + 8] = DInfo.P[12 * (proID - 1) + 8], ValidPmat[12 * nvalidViews + 9] = DInfo.P[12 * (proID - 1) + 9], ValidPmat[12 * nvalidViews + 10] = DInfo.P[12 * (proID - 1) + 10], ValidPmat[12 * nvalidViews + 11] = DInfo.P[12 * (proID - 1) + 11];
						}
						nvalidViews++;
					}
				}
				if (nvalidViews < 1)
					continue;

				ValidPts[nvalidViews].x = 1.0*ii, ValidPts[nvalidViews].y = 1.0*jj;
				Undo_distortion(ValidPts[nvalidViews], DInfo.K + 9 * nPros, DInfo.distortion + 13 * nPros);
				ValidPmat[12 * nvalidViews] = DInfo.P[12 * (nPros - 1)], ValidPmat[12 * nvalidViews + 1] = DInfo.P[12 * (nPros - 1) + 1], ValidPmat[12 * nvalidViews + 2] = DInfo.P[12 * (nPros - 1) + 2], ValidPmat[12 * nvalidViews + 3] = DInfo.P[12 * (nPros - 1) + 3],
					ValidPmat[12 * nvalidViews + 4] = DInfo.P[12 * (nPros - 1) + 4], ValidPmat[12 * nvalidViews + 5] = DInfo.P[12 * (nPros - 1) + 5], ValidPmat[12 * nvalidViews + 6] = DInfo.P[12 * (nPros - 1) + 6], ValidPmat[12 * nvalidViews + 7] = DInfo.P[12 * (nPros - 1) + 7],
					ValidPmat[12 * nvalidViews + 8] = DInfo.P[12 * (nPros - 1) + 8], ValidPmat[12 * nvalidViews + 9] = DInfo.P[12 * (nPros - 1) + 9], ValidPmat[12 * nvalidViews + 10] = DInfo.P[12 * (nPros - 1) + 10], ValidPmat[12 * nvalidViews + 11] = DInfo.P[12 * (nPros - 1) + 11];
				nvalidViews++;

				NviewTriangulation(ValidPts, ValidPmat, &WC, nvalidViews);
				reprojectionError = 0.0; nfails = 0;
				for (kk = 0; kk < nvalidViews; kk++)
				{
					denum = ValidPmat[12 * kk + 8] * WC.x + ValidPmat[12 * kk + 9] * WC.y + ValidPmat[12 * kk + 10] * WC.z + ValidPmat[12 * kk + 11];
					u = (ValidPmat[12 * kk] * WC.x + ValidPmat[12 * kk + 1] * WC.y + ValidPmat[12 * kk + 2] * WC.z + ValidPmat[12 * kk + 3]) / denum;
					v = (ValidPmat[12 * kk + 4] * WC.x + ValidPmat[12 * kk + 5] * WC.y + ValidPmat[12 * kk + 6] * WC.z + ValidPmat[12 * kk + 7]) / denum;

					reprojectionError = 0.5*(abs(u - ValidPts[kk].x) + abs(v - ValidPts[kk].y));
					if (nvalidViews>2)
						thresh = triThresh / (nvalidViews + 2);
					else
						thresh = triThresh;
					if (reprojectionError > thresh)
					{
						nfails++;
						if (validView[kk] > nPros)
							continue;
						for (ll = 0; ll < 6; ll++)
							WarpingParas[ii + jj*width + (ll + 6 * validView[kk])*length] = 0.0;
					}
				}
				if (nfails < 1)
					fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
			}
		}
		fclose(fp);

		for (int proID = 0; proID < nPros; proID++)
		{
			for (ii = 0; ii < 6; ii++)
			{
				sprintf(Fname, "%s/Results/CamPro/C%dP%dp%d_%05d.dat", PATH, camID + 1, proID + 1, ii, frameID);
				WriteGridBinary(Fname, WarpingParas + (ii + 6 * proID)*length, width, height);
			}
		}
	}

	delete[]disparity;
	delete[]CDepth;
	delete[]cROI;
	delete[]lpROI_calculated;
	delete[]Img1;
	delete[]Img2;
	delete[]SparseCorres1;
	delete[]SparseCorres2;
	delete[]WarpingParas;

	return 0;
}
int CamProMatching2(char *PATH, int nCams, int nPros, int frameID, int width, int height, int pwidth, int pheight, int nchannels, LKParameters LKArg, CPoint *ROI, bool CheckMatching = false, bool FlowVerification = false, double triThresh = 2.0, bool saveWarping = false, bool Simulation = false)
{
	//Double check checker size!!!
	int ii, jj, kk, camID;
	int length = width*height, plength = pwidth*pheight;

	int  step = LKArg.step, hsubset = LKArg.hsubset;
	int Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, Convergence_Criteria = LKArg.Convergence_Criteria, Analysis_Speed = LKArg.Analysis_Speed, DICAlgo = LKArg.DIC_Algo, InterpAlgo = LKArg.InterpAlgo;
	double PSSDab_thresh = LKArg.PSSDab_thresh, Gsigma = LKArg.Gsigma;
	int nSeedPoints = 5000; //should be enough

	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		//return 1;
	}
	double ProEpipole[3];

	//Load images
	char *Img1 = new char[length*nchannels];
	char *Img2 = new char[plength*nchannels];
	bool *cROI = new bool[length];
	bool *lpROI_calculated = new bool[length];

	char Fname[100];
	IplImage *view = 0;
	CPoint2 *SparseCorres1 = new CPoint2[nSeedPoints];
	CPoint2 *SparseCorres2 = new CPoint2[nSeedPoints];
	CPoint2 *disparity = new CPoint2[length];
	float *WarpingParas = 0;
	float *CDepth = new float[length];

	if (saveWarping)
	{
		WarpingParas = new float[length * 6 * nPros];
		for (ii = 0; ii < 6 * nPros*length; ii++)
			WarpingParas[ii] = 0.0f;
	}

	for (camID = 0; camID < nCams; camID++)
	{
#pragma omp critical
		cout << "Run CamPro on frame " << frameID << " of Cam #" << camID + 1;

		//Read camera image
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID + 1, frameID); view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
		if (view == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 2;
		}
		else
		{
			for (kk = 0; kk < nchannels; kk++)
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						Img1[ii + (height - 1 - jj)*width + kk*length] = view->imageData[nchannels*ii + nchannels*jj*width + kk];
			cvReleaseImage(&view);
		}

		if (nPros == 2)
			LKArg.checkZNCC = true;

		for (int proID = 0; proID < nPros; proID++)
		{
#pragma omp critical
			cout << " and Projector #" << proID + 1 << endl;
			//Read projector image
			sprintf(Fname, "%s/ProjectorPattern%d.png", PATH, proID + 1);  view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
			if (view == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 2;
			}
			else
			{
				for (kk = 0; kk < nchannels; kk++)
					for (jj = 0; jj < pheight; jj++)
						for (ii = 0; ii < pwidth; ii++)
							Img2[ii + (pheight - 1 - jj)*pwidth + kk*plength] = view->imageData[nchannels*ii + nchannels*jj*pwidth + kk];
				cvReleaseImage(&view);
			}

			//Load scale information
			double CamProScale[10];
			sprintf(Fname, "%s/CamProScale.txt", PATH);
			FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 3;
			}
			else
			{
				for (ii = 0; ii < nCams; ii++)
				{
					fscanf(fp, "%lf ", &CamProScale[ii]);
					CamProScale[ii] = 1.0 / CamProScale[ii];
				}
				fclose(fp);
			}

			//Load seeds points
			sprintf(Fname, "%s/Sparse/C%dP%d_%05d.txt", PATH, camID + 1, proID + 1, frameID);
			fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 3;
			}
			else
			{
				nSeedPoints = 0;
				while (fscanf(fp, "%lf %lf ", &SparseCorres1[nSeedPoints].x, &SparseCorres1[nSeedPoints].y) != EOF)
				{
					SparseCorres1[nSeedPoints].y = SparseCorres1[nSeedPoints].y;
					nSeedPoints++;
				}
				fclose(fp);

				sprintf(Fname, "%s/Sparse/P%dC%d_%05d.txt", PATH, proID + 1, camID + 1, frameID); fp = fopen(Fname, "r");
				if (fp == NULL)
				{
					cout << "Cannot load " << Fname << endl;
					return 3;
				}
				else
				{
					for (ii = 0; ii < nSeedPoints; ii++)
						fscanf(fp, "%lf %lf ", &SparseCorres2[ii].x, &SparseCorres2[ii].y);//Supreeth
					fclose(fp);
				}
			}

			//Customize the ROI
			int maxX = 0, maxY = 0, minX = width, minY = height;
			for (ii = 0; ii<nSeedPoints; ii++)
			{
				if (SparseCorres1[ii].x > maxX)
					maxX = (int)SparseCorres1[ii].x;
				else if (SparseCorres1[ii].x < minX)
					minX = (int)SparseCorres1[ii].x;
				if (SparseCorres1[ii].y > maxY)
					maxY = (int)SparseCorres1[ii].y;
				else if (SparseCorres1[ii].y < minY)
					minY = (int)SparseCorres1[ii].y;
			}
			maxX = maxX>width - 2 ? maxX : maxX + 2;
			minX = minX < 2 ? minX : minX - 2;
			maxY = maxY > height - 2 ? maxY : maxY + 2;
			minY = minY < 2 ? minY : minY - 2;

			for (ii = 0; ii < length; ii++)
				cROI[ii] = false;
			if (ROI == NULL)
				for (jj = minY; jj < maxY; jj++)
					for (ii = minX; ii < maxX; ii++)
						cROI[ii + jj*width] = true;
			else
				for (jj = ROI[0].y; jj < ROI[1].y; jj++)
					for (ii = ROI[0].x; ii < ROI[1].x; ii++)
						cROI[ii + jj*width] = true;

			if (Simulation)
			{
				double *BlurImage = new double[length];
				sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID + 1, frameID);
				view = cvLoadImage(Fname, 0);
				if (!view)
				{
					cout << "Cannot load Images" << Fname << endl;
					return 2;
				}
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						BlurImage[ii + (height - 1 - jj)*width] = (double)((int)((unsigned char)view->imageData[nchannels*ii + jj*width]));
				Gaussian_smooth_Double(BlurImage, BlurImage, height, width, 255.0, 2.0); //severely smooth out the image to determine the boundary

				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii<width; ii++)
						if (BlurImage[ii + jj*width] > 5.0)
							cROI[ii + jj*width] = true;
				delete[]BlurImage;
			}

			for (ii = 0; ii < length; ii++)
			{
				if (abs(WarpingParas[ii + 6 * proID*length]) + abs(WarpingParas[ii + (1 + 6 * proID)*length]) > 0.1)
				{
					lpROI_calculated[ii] = true; 	// 1 - Calculated, 0 - Other
					disparity[ii].x = WarpingParas[ii + 6 * proID*length], disparity[ii].y = WarpingParas[ii + (1 + 6 * proID)*length];
				}
				else
				{
					lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
					disparity[ii].x = 0.0, disparity[ii].y = 0.0;
				}
			}

			double Pmat[24], K[18], distortion[26];
			if (proID == 0)
			{
				Pmat[0] = DInfo.K[0], Pmat[1] = DInfo.K[1], Pmat[2] = DInfo.K[2], Pmat[3] = 0.0,
					Pmat[4] = DInfo.K[3], Pmat[5] = DInfo.K[4], Pmat[6] = DInfo.K[5], Pmat[7] = 0.0,
					Pmat[8] = DInfo.K[6], Pmat[9] = DInfo.K[7], Pmat[10] = DInfo.K[8], Pmat[11] = 0.0;
			}
			else
			{
				Pmat[0] = DInfo.P[12 * (proID - 1)], Pmat[1] = DInfo.P[12 * (proID - 1) + 1], Pmat[2] = DInfo.P[12 * (proID - 1) + 2], Pmat[3] = DInfo.P[12 * (proID - 1) + 3],
					Pmat[4] = DInfo.P[12 * (proID - 1) + 4], Pmat[5] = DInfo.P[12 * (proID - 1) + 5], Pmat[6] = DInfo.P[12 * (proID - 1) + 6], Pmat[7] = DInfo.P[12 * (proID - 1) + 7],
					Pmat[8] = DInfo.P[12 * (proID - 1) + 8], Pmat[9] = DInfo.P[12 * (proID - 1) + 9], Pmat[10] = DInfo.P[12 * (proID - 1) + 10], Pmat[11] = DInfo.P[12 * (proID - 1) + 11];
			}
			for (ii = 0; ii < 12; ii++)
				Pmat[12 + ii] = DInfo.P[12 * (camID - 1 + nPros) + ii];
			for (ii = 0; ii < 9; ii++)
				K[ii] = DInfo.K[9 * proID + ii], K[ii + 9] = DInfo.K[9 * (camID + nPros) + ii];
			for (ii = 0; ii < 13; ii++)
				distortion[ii] = DInfo.distortion[13 * proID + ii], distortion[13 + ii] = DInfo.distortion[13 * (camID + nPros) + ii];

			GreedyMatching(Img1, Img2, disparity, lpROI_calculated, cROI, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, nchannels, width, height, pwidth, pheight, CamProScale[camID], ProEpipole, WarpingParas + 6 * proID*length, Pmat, K, distortion, triThresh);

			int maxdisparity = 70;
			double direction[2] = { 1, 0 };
			double *SImg1 = new double[width*height*nchannels];
			double *SImg2 = new double[pwidth*pheight*nchannels];
			double *Img2Para = new double[pwidth*pheight*nchannels];

			if (Gsigma > 0.0)
			{
				for (kk = 0; kk < nchannels; kk++)
				{
					Gaussian_smooth(Img1 + kk*width*height, SImg1 + kk*width*height, height, width, 255.0, Gsigma / sqrt(2));
					Gaussian_smooth(Img2 + kk*pwidth*pheight, SImg2 + kk*pwidth*pheight, pheight, pwidth, 255.0, Gsigma*sqrt(2));
				}
			}
			else
			{
				for (kk = 0; kk < nchannels; kk++)
				{
					for (ii = 0; ii < width*height; ii++)
						SImg1[ii + kk*width*height] = (double)((int)((unsigned char)(Img1[ii + kk*width*height])));
					for (ii = 0; ii < pwidth*pheight; ii++)
						SImg2[ii + pwidth*pheight*kk] = (double)((int)((unsigned char)(Img2[ii + kk*pwidth*pheight])));
				}
			}
			for (kk = 0; kk < nchannels; kk++)
				Generate_Para_Spline(SImg2 + kk*pwidth*pheight, Img2Para + kk*pwidth*pheight, pwidth, pheight, LKArg.InterpAlgo);

			double *tPatch = new double[2 * nchannels*(LKArg.hsubset * 2 + 1)*(LKArg.hsubset * 2 + 1)],
				*tZNCC = new double[2 * (LKArg.hsubset * 2 + 1)*(LKArg.hsubset * 2 + 1)*nchannels],
				*Znssd_reqd = new double[6 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];

#pragma omp parallel for
			for (int jj = ROI[0].y; jj < ROI[1].y; jj++)
			{
				for (int ii = ROI[0].x; ii < ROI[1].x; ii++)
				{
					if (abs(WarpingParas[ii + jj*width]) + abs(WarpingParas[ii + jj*width + length]) > 0.1)
						continue;
					CPoint2 From(ii, jj), Target(ii, jj);
					double score = BruteforceMatchingEpipolar(From, Target, direction, maxdisparity, SImg1, SImg2, Img2Para, nchannels, width, height, pwidth, pheight, LKArg, tPatch, tZNCC, Znssd_reqd);
					if (score > LKArg.ZNCCThreshold - 0.1)
					{
						WarpingParas[ii + jj*width] = Target.x - From.x;
						WarpingParas[ii + jj*width + length] = Target.y - From.y;
					}
				}
			}

			delete[]SImg1, delete[]SImg2, delete[]Img2Para;
			delete[]tPatch, delete[]tZNCC, delete[]Znssd_reqd;

			if (FlowVerification)
			{
				//Load illum flow to verify the pure illum regions
				double start = omp_get_wtime();
				char Fname1[200], Fname2[200];
				float *IllumFlow = new float[2 * length];
				sprintf(Fname1, "%s/Flow/FX1_%05d.dat", PATH, frameID), sprintf(Fname2, "%s/Flow/FY1_%05d.dat", PATH, frameID);
				if (!ReadFlowBinary(Fname1, Fname2, IllumFlow, IllumFlow + length, width, height))
				{
					cout << "Cannot load illumination flow for frame" << frameID << endl;
					delete[]IllumFlow;
					return 1;
				}
				cout << "Loaded illumination flow in " << omp_get_wtime() - start << endl;

				//Clean campro warping
				for (jj = 0; jj < length; jj++)
					if (abs(IllumFlow[jj]) + abs(IllumFlow[jj + length]) < 0.01)
						for (ii = 0; ii < 6; ii++)
							WarpingParas[jj + (ii + proID * 6)*length] = 0.0;

				delete[]IllumFlow;
			}
		}

		for (jj = 0; jj < nPros; jj++)
		{
			for (ii = 0; ii < 2; ii++)//supreeth
			{
				sprintf(Fname, "%s/Results/CamPro/C%dP%dp%d_%05d.dat", PATH, camID + 1, jj + 1, ii, frameID);
				WriteGridBinary(Fname, WarpingParas + (ii + 6 * jj)*length, width, height);
			}
		}

		double S[3], xx, yy;
		unsigned char *LuminanceImg = new unsigned char[length*nchannels];
		double *ParaIllumSource = new double[length];
		Generate_Para_Spline(Img2, ParaIllumSource, width, height, InterpAlgo);

		for (jj = 0; jj < height; jj++)
		{
			for (ii = 0; ii < width; ii++)
			{
				LuminanceImg[ii + jj*width] = (unsigned char)(0);
				if (!cROI[ii + jj*width])
					for (kk = 0; kk < nchannels; kk++)
						LuminanceImg[ii + jj*width + kk*length] = (unsigned char)(0);

				if (abs(WarpingParas[ii + jj*width]) + abs(WarpingParas[ii + jj*width + length]) > 0.01)
				{
					xx = WarpingParas[ii + jj*width] + ii, yy = WarpingParas[ii + jj*width + length] + jj;
					if (xx<0.0 || xx > pwidth - 1 || yy < 0.0 || yy>pheight - 1)
						continue;
					for (kk = 0; kk<nchannels; kk++)
					{
						Get_Value_Spline(ParaIllumSource, pwidth, pheight, xx, yy, S, -1, InterpAlgo);
						if (S[0] > 255.0)
							S[0] = 255.0;
						else if (S[0] < 0.0)
							S[0] = 0.0;
						LuminanceImg[ii + jj*width + kk*length] = (unsigned char)(int)(S[0] + 0.5);
					}
				}
			}
		}
		sprintf(Fname, "%s/Results/swL_%05d.png", PATH, frameID);
		printf("%s\n", Fname);
		SaveDataToImage(Fname, LuminanceImg, width, height);
		delete[]ParaIllumSource;
		delete[]LuminanceImg;
	}

	delete[]disparity;
	delete[]CDepth;
	delete[]cROI;
	delete[]lpROI_calculated;
	delete[]Img1;
	delete[]Img2;
	delete[]SparseCorres1;
	delete[]SparseCorres2;
	delete[]WarpingParas;

	return 0;
}
int ProCamMatching(char *PATH, int nCams, int frameID, int startID, int width, int height, int pwidth, int pheight, int nchannels, LKParameters LKArg, CPoint *ROI = 0)
{
	//Double check checker size!!!
	int ii, jj, kk;
	int length = width*height, plength = pwidth*pheight;

	int  step = LKArg.step, hsubset = LKArg.hsubset;
	int Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, Convergence_Criteria = LKArg.Convergence_Criteria, Analysis_Speed = LKArg.Analysis_Speed, DICAlgo = LKArg.DIC_Algo, InterpAlgo = LKArg.InterpAlgo;
	double PSSDab_thresh = LKArg.PSSDab_thresh, Gsigma = LKArg.Gsigma;
	int nSeedPoints = 5000; //should be enough

	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	//Load images
	char *Img1 = new char[plength*nchannels];
	char *Img2 = new char[length*nchannels];
	bool *cROI = new bool[plength];
	bool *lpROI_calculated = new bool[plength];
	for (ii = 0; ii < plength; ii++)
	{
		cROI[ii] = false; // 1 - Valid, 0 - Other
		lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
	}

	char Fname[100];
	IplImage *view = 0;
	CPoint2 *SparseCorres1 = new CPoint2[nSeedPoints];
	CPoint2 *SparseCorres2 = new CPoint2[nSeedPoints];
	CPoint2 *disparity = new CPoint2[plength];
	float *PDepth = new float[plength];

	sprintf(Fname, "%s/ProjectorPattern.png", PATH);  view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
	if (view == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 2;
	}
	else
	{
		for (kk = 0; kk < nchannels; kk++)
			for (jj = 0; jj < pheight; jj++)
				for (ii = 0; ii < pwidth; ii++)
					Img1[ii + (pheight - 1 - jj)*pwidth + kk*plength] = view->imageData[nchannels*ii + nchannels*jj*pwidth + kk];
		cvReleaseImage(&view);
	}

	for (int camID = 1; camID <= nCams; camID++)
	{
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID, frameID); view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
		if (view == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 2;
		}
		else
		{
			for (kk = 0; kk < nchannels; kk++)
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						Img2[ii + (height - 1 - jj)*width + kk*length] = view->imageData[nchannels*ii + nchannels*jj*width + kk];
			cvReleaseImage(&view);
		}

		//Load scale information
		double CamProScale[10];
		sprintf(Fname, "%s/CamProScale.txt", PATH);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 3;
		}
		else
		{
			for (ii = 0; ii < nCams; ii++)
				fscanf(fp, "%lf ", &CamProScale[ii]);
			fclose(fp);
		}

		//Load seeds points
		sprintf(Fname, "%s/Sparse/P%d_%05d.txt", PATH, camID, frameID);
		fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 3;
		}
		else
		{
			nSeedPoints = 0;
			while (fscanf(fp, "%d %lf %lf ", &ii, &SparseCorres1[nSeedPoints].x, &SparseCorres1[nSeedPoints].y) != EOF)
				nSeedPoints++;
			fclose(fp);

			sprintf(Fname, "%s/Sparse/CC%d_%05d.txt", PATH, camID, frameID); fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 3;
			}
			else
			{
				for (ii = 0; ii < nSeedPoints; ii++)
					fscanf(fp, "%lf %lf ", &SparseCorres2[ii].x, &SparseCorres2[ii].y);
				fclose(fp);
			}
		}

		//Customize the cROI
		int maxX = 0, maxY = 0, minX = pwidth, minY = pheight;
		for (ii = 0; ii<nSeedPoints; ii++)
		{
			if (SparseCorres1[ii].x > maxX)
				maxX = (int)SparseCorres1[ii].x;
			else if (SparseCorres1[ii].x < minX)
				minX = (int)SparseCorres1[ii].x;
			if (SparseCorres1[ii].y > maxY)
				maxY = (int)SparseCorres1[ii].y;
			else if (SparseCorres1[ii].y < minY)
				minY = (int)SparseCorres1[ii].y;
		}
		maxX = maxX>pwidth - 2 ? maxX : maxX + 2;
		minX = minX < 2 ? minX : minX - 2;
		maxY = maxY > pheight - 2 ? maxY : maxY + 2;
		minY = minY < 2 ? minY : minY - 2;

		//minX = 100, maxX = pwidth-100, minY = 100, maxY = pheight-100;
		for (ii = 0; ii < plength; ii++)
			cROI[ii] = false;
		if (ROI == NULL)
			for (jj = minY; jj < maxY; jj++)
				for (ii = minX; ii < maxX; ii++)
					cROI[ii + jj*pwidth] = true;
		else
			for (jj = ROI[0].y; jj < ROI[1].y; jj++)
				for (ii = ROI[0].x; ii < ROI[1].x; ii++)
					cROI[ii + jj*pwidth] = true;

		for (ii = 0; ii < plength; ii++)
			disparity[ii].x = 0.0, disparity[ii].y = 0.0;
		GreedyMatching(Img1, Img2, disparity, lpROI_calculated, cROI, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, nchannels, pwidth, pheight, width, height, CamProScale[camID - 1]);

		double P1mat[12];
		P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0;
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0;
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;

		CPoint2 Ppts, Cpts; CPoint3 WC;
		int x, y;
		sprintf(Fname, "%s/Results/CamPro/PC%d_%05d.xyz", PATH, camID, frameID); fp = fopen(Fname, "w+");
		for (ii = 0; ii < plength; ii++)
		{
			if (lpROI_calculated[ii])
			{
				x = ii%pwidth, y = ii / pwidth;
				Ppts.x = 1.0*x, Ppts.y = 1.0*y;
				Cpts.x = disparity[x + y*pwidth].x + x, Cpts.y = disparity[x + y*pwidth].y + y;
				if (abs(disparity[x + y*pwidth].x) < 0.001 && abs(disparity[x + y*pwidth].y) < 0.001)
				{
					PDepth[x + y*pwidth] = 0.0f;
					continue;
				}

				Undo_distortion(Ppts, DInfo.K, DInfo.distortion);
				Undo_distortion(Cpts, DInfo.K + 9, DInfo.distortion + 13 * camID);
				Stereo_Triangulation2(&Ppts, &Cpts, P1mat, DInfo.P + 12 * (camID - 1), &WC);
				PDepth[x + y*pwidth] = (float)WC.z;
				fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
			}
		}
		fclose(fp);

		//sprintf(Fname, "%s/Results/CamPro/PC%d_%05d.ijz", PATH, camID, frameID);
		//if(!WriteGridBinary(Fname, PDepth, pwidth, pheight))
		//	cout<<"Cannot write "<<Fname<<endl;
	}

	delete[]disparity;
	delete[]cROI;
	delete[]lpROI_calculated;
	delete[]Img1;
	delete[]Img2;
	delete[]PDepth;
	delete[]SparseCorres1;
	delete[]SparseCorres2;

	return 0;
}
int StereoMatching(char *PATH, int frameID, int startID, int width, int height, int nchannels, LKParameters LKArg, CPoint *ROI = 0)
{
	int ii, jj;
	int length = width*height;

	double Gsigma = LKArg.Gsigma;
	int nSeedPoints = 5000; //should be enough

	DevicesInfo DInfo(2);
	//if(!SetUpDevicesInfo(DInfo, PATH))
	//{
	//	cout<<"Cannot load Camera Projector Info"<<endl;
	//	return 1;
	//}

	//Load images
	char *Img1 = new char[length];
	char *Img2 = new char[length];
	bool *cROI = new bool[length];
	bool *lpROI_calculated = new bool[length];
	for (ii = 0; ii < length; ii++)
	{
		cROI[ii] = false; // 1 - Valid, 0 - Other
		lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
	}

	char Fname[100], Fname1[200], Fname2[200];
	IplImage *view = 0;
	sprintf(Fname, "%s/Image/C%d_%02d.png", PATH, 1, frameID);
	view = cvLoadImage(Fname, 0);
	if (view == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 2;
	}
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			Img1[ii + jj*width] = view->imageData[ii + (height - 1 - jj)*width];
	cvReleaseImage(&view);

	sprintf(Fname, "%s/Image/C%d_%02d.png", PATH, 2, frameID);
	view = cvLoadImage(Fname, 0);
	if (view == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 2;
	}
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			Img2[ii + jj*width] = view->imageData[ii + (height - 1 - jj)*width];
	cvReleaseImage(&view);


	//Load seeds points
	nSeedPoints = 0;
	double tx, ty;
	sprintf(Fname, "%s/Sparse/CC%d_%02d.txt", PATH, 1, frameID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	while (fscanf(fp, "%lf %lf ", &tx, &ty) != EOF)
		nSeedPoints++;
	fclose(fp);


	CPoint2 *SparseCorres1 = new CPoint2[nSeedPoints];
	sprintf(Fname, "%s/Sparse/CC%d_%02d.txt", PATH, 1, frameID);
	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	for (ii = 0; ii < nSeedPoints; ii++)
		fscanf(fp, "%lf %lf ", &SparseCorres1[ii].x, &SparseCorres1[ii].y);
	fclose(fp);

	CPoint2 *SparseCorres2 = new CPoint2[nSeedPoints * 2];
	sprintf(Fname, "%s/Sparse/CC%d_%02d.txt", PATH, 2, frameID);
	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	for (ii = 0; ii < nSeedPoints; ii++)
		fscanf(fp, "%lf %lf ", &SparseCorres2[ii].x, &SparseCorres2[ii].y);
	fclose(fp);


	//Customize the cROI
	int ntriangles = 0;
	int maxX = width - 50, maxY = height - 50, minX = 50, minY = 50;
	if (ROI == NULL)
		for (jj = minY; jj < maxY; jj++)
			for (ii = minX; ii < maxX; ii++)
				cROI[ii + jj*width] = true;
	else
		for (jj = ROI[0].y; jj < ROI[1].y; jj++)
			for (ii = ROI[0].x; ii < ROI[1].x; ii++)
				cROI[ii + jj*width] = true;

	/*
	for(ii=0; ii<length; ii++)
	cROI[ii] = false; // 1 - Valid, 0 - Other
	for(jj= 700; jj<1000; jj++)
	for(ii=300; ii<600; ii++)
	cROI[ii+jj*width] = true;
	*/
	CPoint2 *disparity = new CPoint2[length];
	for (ii = 0; ii < length; ii++)
	{
		disparity[ii].x = 0.0;
		disparity[ii].y = 0.0;
	}
	GreedyMatching(Img1, Img2, disparity, lpROI_calculated, cROI, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, nchannels, width, height, width, height, 1.0);

	float *fx = new float[width*height];
	float *fy = new float[width*height];
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			fx[ii + jj*width] = disparity[ii + jj*width].x, fy[ii + jj*width] = disparity[ii + jj*width].y;

	sprintf(Fname1, "%s/disparityX.dat", PATH, frameID);
	sprintf(Fname2, "%s/disparityY.dat", PATH, frameID);
	WriteFlowBinary(Fname1, Fname2, fx, fy, width, height);

	return 0;

	CPoint2 C1pts, C2pts; CPoint3 WC;
	int x, y;
	sprintf(Fname, "%s/Results/Stereo/Stereo_%05d.xyz", PATH, frameID); fp = fopen(Fname, "w+");
	for (ii = 0; ii < length; ii++)
	{
		if (lpROI_calculated[ii])
		{
			x = ii%width;
			y = ii / width;
			C1pts.x = 1.0*x, C1pts.y = 1.0*y;
			C2pts.x = disparity[x + y*width].x + x, C2pts.y = disparity[x + y*width].y + y;
			if (abs(disparity[x + y*width].x) < 0.001 && abs(disparity[x + y*width].y) < 0.001)
				continue;

			Undo_distortion(C1pts, DInfo.K + 9, DInfo.distortion + 13);
			Undo_distortion(C2pts, DInfo.K + 18, DInfo.distortion + 26);
			Stereo_Triangulation2(&C1pts, &C2pts, DInfo.P, DInfo.P + 12, &WC);
			fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
		}
	}
	fclose(fp);

	delete[]disparity;
	delete[]cROI;
	delete[]lpROI_calculated;
	delete[]Img1;
	delete[]Img2;
	delete[]SparseCorres1;
	delete[]SparseCorres2;

	return 0;
}
int FlowMatching(char *PATH, int nCams, int nPros, int camID, int frameID, int frameJump, int width, int height, int nchannels, LKParameters LKArg, bool forward, bool saveWarping = false, int signature = 0, bool BackWarpCheck = false)
{
	int ii, jj, kk;
	int length = width*height;

	double Epipole[3];
	DevicesInfo DInfo(nCams, nPros);
	if (LKArg.DIC_Algo <= 1) //Only when illumination flow is used
	{
		if (!SetUpDevicesInfo(DInfo, PATH))
		{
			cout << "Cannot load Camera Projector Info" << endl;
			return 1;
		}

		//e't*F = 0: e' is the left null space of F
		double U[9], W[9], V[9];
		Matrix F12(3, 3); F12.Matrix_Init(&DInfo.FmatPC[(camID - 1) * 9]);
		F12.SVDcmp(3, 3, U, W, V, CV_SVD_MODIFY_A);

		//last column of U + normalize
		for (ii = 0; ii < 3; ii++)
			Epipole[ii] = U[2 + 3 * ii] / U[8];
	}

	//Load images
	char *Img1 = new char[length*nchannels];
	char *Img2 = new char[length*nchannels];
	bool *ROI = new bool[length];
	bool *lpROI_calculated = new bool[length];
	char Fname[100], Fname1[200], Fname2[200];

	sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID, 1, frameID);
	Mat cvImg = imread(Fname, nchannels == 1 ? 0 : 1);
	if (!cvImg.data)
	{
		cout << "Cannot load: " << Fname << endl;
		return 2;
	}
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				Img1[ii + (height - 1 - jj)*width + length*kk] = (char)cvImg.data[nchannels*ii + kk + jj*nchannels*width];

	sprintf(Fname, "%s/ProjectorPattern.png", PATH);
	cvImg = imread(Fname, nchannels == 1 ? 0 : 1);
	if (!cvImg.data)
	{
		cout << "Cannot load: " << Fname << endl;
		return 2;
	}
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			for (kk = 0; kk < nchannels; kk++)
				Img2[ii + (height - 1 - jj)*width + length*kk] = (char)cvImg.data[nchannels*ii + kk + jj*nchannels*width];

	//Load seeds points
	int nSeedPoints = 0;
	double tx, ty;
	if (signature == 0)
		sprintf(Fname, "%s/Sparse/C%d_%05d.txt", PATH, camID, frameID);
	else
		sprintf(Fname, "%s/Sparse/C%d_%05d_%d.txt", PATH, camID, frameID, signature);

	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	while (fscanf(fp, "%lf %lf ", &tx, &ty) != EOF)
		nSeedPoints++;
	fclose(fp);

	CPoint2 *SparseCorres1 = new CPoint2[nSeedPoints];
	if (signature == 0)
		sprintf(Fname, "%s/Sparse/C%d_%05d.txt", PATH, camID, frameID);
	else
		sprintf(Fname, "%s/Sparse/C%d_%05d_%d.txt", PATH, camID, 1, signature);
	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	for (ii = 0; ii < nSeedPoints; ii++)
		fscanf(fp, "%lf %lf ", &SparseCorres1[ii].x, &SparseCorres1[ii].y);
	fclose(fp);

	CPoint2 *SparseCorres2 = new CPoint2[nSeedPoints];
	if (signature == 0)
		//sprintf(Fname, "%s/Sparse/C%d_%05d.txt", PATH, camID, frameID + frameJump);
		sprintf(Fname, "%s/Sparse/P1_%05d.txt", PATH, frameID);
	else
		sprintf(Fname, "%s/Sparse/C%d_%05d_%d.txt", PATH, camID, frameID + frameJump, signature);
	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	for (ii = 0; ii < nSeedPoints; ii++)
		fscanf(fp, "%lf %lf ", &SparseCorres2[ii].x, &SparseCorres2[ii].y);
	fclose(fp);

	//Customize the ROI
	int minX = 50, minY = 50, maxX = width - 50, maxY = height - 50;
	CPoint2 *flow = new CPoint2[length];
	float *WarpingsPara = new float[length * 6];
	if (forward)//Foward flow
	{
		for (ii = 0; ii < length; ii++)
		{
			ROI[ii] = false; // 1 - Valid, 0 - Other
			lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
		}

		for (jj = minY; jj < maxY; jj++)
			for (ii = minX; ii < maxX; ii++)
				ROI[ii + jj*width] = true;

		for (ii = 0; ii < length; ii++)
			flow[ii].x = 0.0, flow[ii].y = 0.0, WarpingsPara[ii] = 0, WarpingsPara[ii + length] = 0.0;
		if (GreedyMatching(Img1, Img2, flow, lpROI_calculated, ROI, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, nchannels, width, height, width, height, 1.0, Epipole, WarpingsPara) == 1)
			return 1;

		if (BackWarpCheck)
			MatchingCheck(Img1, Img2, WarpingsPara, LKArg, 1.0, nchannels, width, height, width, height);

		sprintf(Fname1, "%s/Flow/FX%d_%05d.dat", PATH, camID, frameID);
		sprintf(Fname2, "%s/Flow/FY%d_%05d.dat", PATH, camID, frameID);
		WriteFlowBinary(Fname1, Fname2, WarpingsPara, WarpingsPara + length, width, height);

		if (saveWarping)
			for (ii = 0; ii < 4; ii++)
			{
				sprintf(Fname, "%s/Flow/F%dp%d_%05d.dat", PATH, camID, ii, frameID);
				WriteGridBinary(Fname, WarpingsPara + (ii + 2)*length, width, height);
			}
	}
	else//Backward flow
	{
		for (ii = 0; ii < length; ii++)
		{
			ROI[ii] = false; // 1 - Valid, 0 - Other
			lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
		}

		for (jj = minY; jj < maxY; jj++)
			for (ii = minX; ii < maxX; ii++)
				ROI[ii + jj*width] = true;

		for (ii = 0; ii < length; ii++)
			flow[ii].x = 0.0, flow[ii].y = 0.0;
		if (GreedyMatching(Img2, Img1, flow, lpROI_calculated, ROI, SparseCorres2, SparseCorres1, nSeedPoints, LKArg, nchannels, width, height, width, height, 1.0, Epipole, WarpingsPara) == 1)
			return 1;

		if (BackWarpCheck)
			MatchingCheck(Img2, Img1, WarpingsPara, LKArg, 1.0, nchannels, width, height, width, height);

		sprintf(Fname1, "%s/Flow/RX%d_%05d.dat", PATH, camID, frameID + frameJump);
		sprintf(Fname2, "%s/Flow/RY%d_%05d.dat", PATH, camID, frameID + frameJump);
		WriteFlowBinary(Fname1, Fname2, WarpingsPara, WarpingsPara + length, width, height);

		if (saveWarping)
			for (ii = 0; ii < 4; ii++)
			{
				sprintf(Fname, "%s/Flow/R%dp%d_%05d.dat", PATH, camID, ii, frameID + frameJump);
				WriteGridBinary(Fname, WarpingsPara + (ii + 2)*length, width, height);
			}
	}

	delete[]flow;
	delete[]ROI;
	delete[]lpROI_calculated;
	delete[]Img1;
	delete[]Img2;
	delete[]SparseCorres1;
	delete[]SparseCorres2;
	delete[]WarpingsPara;
	//delete []triangleList;

	return 0;
}
int ScaleSelection(int frameID, const int CamID, bool forward, int width, int height, double flowThresh, LKParameters &LKArg, FlowScaleSelection ScaleSel, char *DataPATH)
{
	if (forward)
	{
		cout << "Run forward scale selection on: [" << frameID << " -> " << frameID + 1 << "]" << endl;
		cout << "Camera: " << CamID << endl;
	}
	else
	{
		cout << "Run back ward scale selection on: [" << frameID << " -> " << frameID + 1 << "]" << endl;
		cout << "Camera: " << CamID << endl;
	}

	char Fname[100], FnameX[100], FnameY[100];
	int ii, jj, kk, length = width*height;

	int nCpts = 0;

	CPoint DenseFlowBoundary[2];
	double x, y;
	int xmin = width, xmax = 0, ymin = height, ymax = 0;
	sprintf(Fname, "%s/Sparse/CC%d_%05d.txt", DataPATH, CamID, frameID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 1;
	}
	while (fscanf(fp, "%lf %lf ", &x, &y) != EOF)
	{
		fscanf(fp, "%lf %lf ", &x, &y);
		if (MyFtoI(x) < xmin)
			xmin = MyFtoI(x);
		if (MyFtoI(x) > xmax)
			xmax = MyFtoI(x);
		if (MyFtoI(y) < ymin)
			ymin = MyFtoI(y);
		if (MyFtoI(y) > ymax)
			ymax = MyFtoI(y);
		nCpts++;
	}
	DenseFlowBoundary[0].x = xmin < 100 ? xmin : xmin - 10;
	DenseFlowBoundary[0].y = ymin < 100 ? ymin : ymin - 10;
	DenseFlowBoundary[1].x = xmax > width - 100 ? xmax : xmax + 10;
	DenseFlowBoundary[1].y = ymax > height - 100 ? ymax : ymax + 10;

	bool flag1 = false, flag2 = false, flag3 = false;
	IplImage *view = 0;
	double *Img = new double[2 * length];
	float *fu = new float[width*height];
	float *fv = new float[width*height];
	int *DenseScale = new int[length];

	for (kk = 0; kk < 2; kk++)
	{
		sprintf(Fname, "%s/Image/C%d_%05d.png", DataPATH, CamID, frameID + kk);
		view = cvLoadImage(Fname, 0);
		if (view == NULL)
		{
			cout << "Cannot load: " << Fname << endl;
			flag1 = true;
			goto ENDPROG;
		}

		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				Img[ii + (height - 1 - jj)*width + kk*length] = (double)((int)((unsigned char)view->imageData[ii + jj*width]));
		cvReleaseImage(&view);

		Gaussian_smooth_Double(Img + kk*length, Img + kk*length, height, width, 255.0, LKArg.Gsigma);
	}

	if (forward)
	{
		sprintf(FnameX, "%s/Flow/F%dforX_%05d.dat", DataPATH, CamID, frameID + 1);
		sprintf(FnameY, "%s/Flow/F%dforY_%05d.dat", DataPATH, CamID, frameID + 1);
	}
	else
	{
		sprintf(FnameX, "%s/Flow/F%dreX_%05d.dat", DataPATH, CamID, frameID + 1);
		sprintf(FnameY, "%s/Flow/F%dreY_%05d.dat", DataPATH, CamID, frameID + 1);
	}
	if (!ReadFlowBinary(FnameX, FnameY, fu, fv, width, height))
	{
		flag2 = true;
		goto ENDPROG;
	}

	for (ii = 0; ii < length; ii++)
		DenseScale[ii] = 0;

	if (forward)
		DIC_DenseScaleSelection(DenseScale, fu, fv, Img, Img + length, width, height, LKArg, ScaleSel, flowThresh, DenseFlowBoundary);
	else
		DIC_DenseScaleSelection(DenseScale, fu, fv, Img + length, Img, width, height, LKArg, ScaleSel, flowThresh, DenseFlowBoundary);


	if (forward)
		sprintf(Fname, "%s/Scales/S%dfor_%05d.dat", DataPATH, CamID, frameID + 1);
	else
		sprintf(Fname, "%s/Scales/S%dre_%05d.dat", DataPATH, CamID, frameID + 1);
	if (!WriteGridBinary(Fname, DenseScale, width, height))
		flag3 = true;

ENDPROG:
	delete[]Img;
	delete[]fu;
	delete[]fv;
	delete[]DenseScale;

	if (!flag1 && !flag2 && !flag3)
		return 0;
	else if (flag1)
		return 1;
	else if (flag2)
		return 2;
	else
		return 3;
}
int SProCamMatching(char *PATH, int nCams, int frameID, int width, int height, int pwidth, int pheight, int nchannels, LKParameters LKArg, double SR)
{
	int ii, jj, kk;
	char Fname[200];
	int InterpAlgo = LKArg.InterpAlgo;
	double Gsigma = LKArg.Gsigma;
	int nSeedPoints = 5000; //should be enough
	int plength = pwidth*pheight, length = width*height, Spwidth = (int)(1.0*pwidth / SR), Spheight = (int)(1.0*pheight / SR), Splength = Spwidth*Spheight;

	//Load campro info
	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	//Load images
	sprintf(Fname, "%s/ProjectorPattern.png", PATH);
	Mat cvImg1 = imread(Fname, nchannels == 1 ? 0 : 1);
	if (!cvImg1.data)
	{
		cout << "Cannot load: " << Fname << endl;
		return 2;
	}

	CPoint2 *SparseCorres1 = new CPoint2[nSeedPoints];
	CPoint2 *SparseCorres2 = new CPoint2[nSeedPoints];
	char *Img = new char[(plength + length)*nchannels];
	int *hsubset = new int[Splength];
	bool *sROI = new bool[Splength];
	bool *lpROI_calculated = new bool[Splength];
	float *disparity = new float[2 * Splength];
	float *Z = new float[Spheight*Spwidth];

	for (int camID = 1; camID <= nCams; camID++)
	{
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, camID, frameID);
		Mat cvImg2 = imread(Fname, nchannels == 1 ? 0 : 1);
		if (!cvImg2.data)
		{
			cout << "Cannot load: " << Fname << endl;
			return 2;
		}

		for (kk = 0; kk < nchannels; kk++)
			for (jj = 0; jj < pheight; jj++)
				for (ii = 0; ii < pwidth; ii++)
					Img[ii + jj*pwidth + kk*plength] = (char)cvImg1.data[ii*nchannels + (pheight - 1 - jj)*pwidth*nchannels + kk];

		for (kk = 0; kk < nchannels; kk++)
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					Img[ii + jj*width + plength*nchannels + kk*length] = (char)cvImg2.data[ii*nchannels + (height - 1 - jj)*width*nchannels + kk];

		//Load seeds points
		sprintf(Fname, "%s/Sparse/P%d_%05d.txt", PATH, camID, frameID);
		FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 3;
		}
		else
		{
			nSeedPoints = 0;
			while (fscanf(fp, "%d %lf %lf ", &ii, &SparseCorres1[nSeedPoints].x, &SparseCorres1[nSeedPoints].y) != EOF)
				nSeedPoints++;
			fclose(fp);

			sprintf(Fname, "%s/Sparse/CC%d_%05d.txt", PATH, camID, frameID); fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot load " << Fname << endl;
				return 3;
			}
			else
			{
				for (ii = 0; ii < nSeedPoints; ii++)
					fscanf(fp, "%lf %lf ", &SparseCorres2[ii].x, &SparseCorres2[ii].y);
				fclose(fp);
			}
		}

		//Load scale information
		double CamProScale[10];
		sprintf(Fname, "%s/CamProScale.txt", PATH);
		fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 3;
		}
		else
		{
			for (ii = 0; ii < nCams; ii++)
				fscanf(fp, "%lf ", &CamProScale[ii]);
			fclose(fp);
		}

		for (ii = 0; ii < Splength; ii++)
		{
			lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
			hsubset[ii] = LKArg.hsubset;
			disparity[ii] = 0.0f, disparity[ii + Splength] = 0.0f;
		}

		//Customize the ROI
		int maxX = 0, maxY = 0, minX = pwidth, minY = pheight;
		for (ii = 0; ii<nSeedPoints; ii++)
		{
			if (SparseCorres1[ii].x > maxX)
				maxX = (int)SparseCorres1[ii].x;
			else if (SparseCorres1[ii].x < minX)
				minX = (int)SparseCorres1[ii].x;
			if (SparseCorres1[ii].y > maxY)
				maxY = (int)SparseCorres1[ii].y;
			else if (SparseCorres1[ii].y < minY)
				minY = (int)SparseCorres1[ii].y;
		}
		maxX = maxX>pwidth - 2 ? maxX : maxX + 2;
		minX = minX < 2 ? minX : minX - 2;
		maxY = maxY > pheight - 2 ? maxY : maxY + 2;
		minY = minY < 2 ? minY : minY - 2;

		for (ii = 0; ii < Splength; ii++)
			sROI[ii] = false;
		for (jj = (int)(minY / SR); jj < (int)(maxY / SR); jj++)
			for (ii = (int)(minX / SR); ii < (int)(maxX / SR); ii++)
				sROI[ii + jj*Spwidth] = true;

		if (SGreedyMatching(Img, Img + plength*nchannels, disparity, lpROI_calculated, sROI, hsubset, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, SR, nchannels, pwidth, pheight, width, height, CamProScale[camID - 1]) == 1)
			return 1;

		double P1mat[12];
		P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0;
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0;
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;

		CPoint2 Ppts, Cpts; CPoint3 WC;

		sprintf(Fname, "%s/Results/CamPro/PC%d_%05d.xyz", PATH, camID, frameID); fp = fopen(Fname, "w+");
		if (fp == NULL)
		{
			cout << "Cannot write: " << Fname << endl;
			return false;
		}
		for (jj = 0; jj < Spheight; jj++)
		{
			for (ii = 0; ii < Spwidth; ii++)
			{
				if (lpROI_calculated[ii + jj*Spwidth])
				{
					Ppts.x = 1.0*ii*SR, Ppts.y = 1.0*jj*SR;
					Cpts.x = disparity[ii + jj*Spwidth] + Ppts.x, Cpts.y = disparity[ii + jj*Spwidth + Splength] + Ppts.y;
					if (abs(disparity[ii + jj*Spwidth] * SR) < 0.001 && abs(disparity[ii + jj*Spwidth + Splength] * SR) < 0.001)
					{
						Z[ii + jj*Spwidth] = 0.0f;
						continue;
					}

					Undo_distortion(Ppts, DInfo.K, DInfo.distortion);
					Undo_distortion(Cpts, DInfo.K + 9 * camID, DInfo.distortion + 13 * camID);
					Stereo_Triangulation2(&Ppts, &Cpts, P1mat, DInfo.P + 12 * (camID - 1), &WC);

					Z[ii + jj*Spwidth] = (float)WC.z;
					fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
				}
				else
					Z[ii + jj*Spwidth] = 0.0f;
			}
		}
		fclose(fp);

		//sprintf(Fname, "%s/Results/CamPro/PC%d_%05d.ijz", PATH, camID, frameID); 
		//WriteGridBinary(Fname, Z, Spwidth, Spheight);
	}


	delete[]Z;
	delete[]hsubset;
	delete[]Img;
	delete[]SparseCorres1;
	delete[]SparseCorres2;
	delete[]sROI;
	delete[]lpROI_calculated;
	delete[]disparity;

	return 0;
}
int SFlowMatching(char *PATH, int nCams, int camID, int frameID, int frameJump, int nchannels, LKParameters LKArg, double SR, bool forward)
{
	int InterpAlgo = LKArg.InterpAlgo;
	double Gsigma = LKArg.Gsigma;
	int nSeedPoints = 5000; //should be enough

	int ii, jj;
	char Fname[200], Fname1[200], Fname2[200];

	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot load Camera Projector Info" << endl;
		return 1;
	}

	double Epipole[3];
	if (LKArg.DIC_Algo <= 1) //Only when illumination flow is used
	{
		//e't*F = 0: e' is the left null space of F
		double U[9], W[9], V[9];
		Matrix F12(3, 3); F12.Matrix_Init(&DInfo.FmatPC[(camID - 1) * 9]);
		F12.SVDcmp(3, 3, U, W, V, CV_SVD_MODIFY_A);

		//last column of U + normalize
		for (ii = 0; ii < 3; ii++)
			Epipole[ii] = U[2 + 3 * ii] / U[8];
	}

	sprintf(Fname1, "%s/Image/C%d_%05d.png", PATH, camID, frameID);
	sprintf(Fname2, "%s/Image/C%d_%05d.png", PATH, camID, frameID + frameJump);
	Mat cvImg1 = imread(Fname1, 0);
	if (!cvImg1.data)
	{
		cout << "Cannot load: " << Fname1 << endl;
		return 2;
	}
	Mat cvImg2 = imread(Fname2, 0);
	if (!cvImg2.data)
	{
		cout << "Cannot load: " << Fname2 << endl;
		return 2;
	}

	int width = cvImg1.cols, height = cvImg1.rows, length = width*height;
	int Swidth = (int)(1.0*width / SR), Sheight = (int)(1.0*height / SR), Slength = Swidth*Sheight;
	char *Img = new char[2 * length];

	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			Img[ii + jj*width] = (char)cvImg1.data[ii + (height - 1 - jj)*width];
			Img[ii + jj*width + length] = (char)cvImg2.data[ii + (height - 1 - jj)*width];
		}
	}

	//Load seeds points
	nSeedPoints = 0;
	double tx, ty;
	sprintf(Fname, "%s/Sparse/CC%d_%05d.txt", PATH, camID, frameID);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	while (fscanf(fp, "%lf %lf ", &tx, &ty) != EOF)
		nSeedPoints++;
	fclose(fp);

	CPoint2 *SparseCorres1 = new CPoint2[nSeedPoints];
	sprintf(Fname, "%s/Sparse/CC%d_%05d.txt", PATH, camID, frameID);
	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	for (ii = 0; ii < nSeedPoints; ii++)
		fscanf(fp, "%lf %lf ", &SparseCorres1[ii].x, &SparseCorres1[ii].y);
	fclose(fp);

	CPoint2 *SparseCorres2 = new CPoint2[nSeedPoints];
	sprintf(Fname, "%s/Sparse/CC%d_%05d.txt", PATH, camID, frameID + frameJump);
	fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 3;
	}
	for (ii = 0; ii < nSeedPoints; ii++)
		fscanf(fp, "%lf %lf ", &SparseCorres2[ii].x, &SparseCorres2[ii].y);
	fclose(fp);

	int *flowhsubset = new int[length];
	float *flow = new float[2 * Slength];
	bool *sROI = new bool[Slength];
	bool *lpROI_calculated = new bool[Slength];

	//Customize the ROI
	int maxX = 0, maxY = 0, minX = width, minY = height;
	for (ii = 0; ii<nSeedPoints; ii++)
	{
		if (SparseCorres1[ii].x > maxX)
			maxX = (int)SparseCorres1[ii].x;
		else if (SparseCorres1[ii].x < minX)
			minX = (int)SparseCorres1[ii].x;
		if (SparseCorres1[ii].y > maxY)
			maxY = (int)SparseCorres1[ii].y;
		else if (SparseCorres1[ii].y < minY)
			minY = (int)SparseCorres1[ii].y;
	}
	maxX = maxX>width - 50 ? maxX : maxX + 50;
	minX = minX < 50 ? minX : minX - 50;
	maxY = maxY > height - 50 ? maxY : maxY + 50;
	minY = minY < 50 ? minY : minY - 50;

	for (ii = 0; ii < Slength; ii++)
		sROI[ii] = false;
	for (jj = (int)(minY / SR); jj < (int)(maxY / SR); jj++)
		for (ii = (int)(minX / SR); ii < (int)(maxX / SR); ii++)
			sROI[ii + jj*Swidth] = true;

	//Foward flow
	if (forward)
	{
		for (ii = 0; ii < Slength; ii++)
		{
			lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
			flow[ii] = 0.0, flow[ii + Slength] = 0.0;
		}

		sprintf(Fname, "%s/Scales/S%dfor_%05d.dat", PATH, camID, frameID + 1); fp = fopen(Fname, "r");
		if (!ReadGridBinary(Fname, flowhsubset, width, height))
		{
			for (ii = 0; ii < width*height; ii++)
				flowhsubset[ii] = LKArg.hsubset;
			cout << "Cannot load scale information" << endl;
			return 4;
		}

		if (SGreedyMatching(Img, Img + length, flow, lpROI_calculated, sROI, flowhsubset, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, SR, nchannels, width, height, width, height, 1.0, Epipole) == 1)
			return 1;

		sprintf(Fname1, "%s/Flow/%dX2/F%dforX_%05d.dat", PATH, (int)(1.0 / SR), camID, frameID);
		sprintf(Fname2, "%s/Flow/%dX2/F%dforY_%05d.dat", PATH, (int)(1.0 / SR), camID, frameID);
		WriteFlowBinary(Fname1, Fname2, flow, flow + Slength, Swidth, Sheight);
	}
	else//Backward flow
	{
		for (ii = 0; ii < Slength; ii++)
		{
			lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other
			flow[ii] = 0.0, flow[ii + Slength] = 0.0;
		}

		sprintf(Fname, "%s/Scales/S%dfor_%05d.dat", PATH, camID, frameID + 1); fp = fopen(Fname, "r");
		if (!ReadGridBinary(Fname, flowhsubset, width, height))
		{
			for (ii = 0; ii < width*height; ii++)
				flowhsubset[ii] = LKArg.hsubset;
			cout << "Cannot load scale information" << endl;
			return 4;
		}

		if (SGreedyMatching(Img + length, Img, flow, lpROI_calculated, sROI, flowhsubset, SparseCorres2, SparseCorres1, nSeedPoints, LKArg, SR, nchannels, width, height, width, height, 1.0, Epipole) == 1)
			return 1;

		sprintf(Fname1, "%s/Flow/%dX2/F%dreX_%05d.dat", PATH, (int)(1.0 / SR), camID, frameID);
		sprintf(Fname2, "%s/Flow/%dX2/F%dreY_%05d.dat", PATH, (int)(1.0 / SR), camID, frameID);
		WriteFlowBinary(Fname1, Fname2, flow, flow + Slength, Swidth, Sheight);
	}

	delete[]flowhsubset;
	delete[]Img;
	delete[]SparseCorres1;
	delete[]SparseCorres2;
	delete[]sROI;
	delete[]lpROI_calculated;
	delete[]flow;

	return 0;
}
int SVSR_Driver(int frameID, IlluminationFlowImages &IlluminationImages, LKParameters LKArg, SVSRP srp, char *DataPATH)
{
	char Fname[200], FnameX[200], FnameY[200];
	const int nCPpts = 3871, maxTriangles = 10000;
	const double flowThresh = 0.1, SRF = srp.SRF;
	/*IMPORTANT*/

	int ii, jj, kk, ll, mm;
	int width = IlluminationImages.width, height = IlluminationImages.height, pwidth = IlluminationImages.pwidth, pheight = IlluminationImages.pheight, length = width*height, plength = pwidth*pheight;
	int nchannels = IlluminationImages.nchannels, nCams = IlluminationImages.nCams, TemporalW = IlluminationImages.nframes, frameJump = IlluminationImages.frameJump;

	//1: Setup cameras & projector parameters and load images data
	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, DataPATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	IplImage *view = 0;
	sprintf(Fname, "%s/ProjectorPattern.png", DataPATH);
	view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
	if (view == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 2;
	}
	for (kk = 0; kk < nchannels; kk++)
		for (jj = 0; jj < pheight; jj++)
			for (ii = 0; ii < pwidth; ii++)
				IlluminationImages.PImg[ii + jj*pwidth + kk*plength] = (double)((int)((unsigned char)view->imageData[nchannels*ii + (pheight - 1 - jj)*nchannels*pwidth + kk]));
	cvReleaseImage(&view);

	for (kk = 0; kk < nchannels; kk++)
	{
		Gaussian_smooth_Double(IlluminationImages.PImg + kk*plength, IlluminationImages.PImg + kk*plength, pheight, pwidth, 255.0, 0.707);
		Generate_Para_Spline(IlluminationImages.PImg + kk*plength, IlluminationImages.PPara + kk*plength, pwidth, pheight, LKArg.InterpAlgo);
	}

	for (kk = 0; kk < nCams; kk++)
	{
		for (ll = 0; ll < TemporalW; ll++)
		{
			sprintf(Fname, "%s/Image/C%d_%05d.png", DataPATH, kk + 1, frameID + ll*frameJump);
			view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
			if (!view)
			{
				cout << "Cannot load " << Fname << endl;
				return 2;
			}
			for (mm = 0; mm < nchannels; mm++)
			{
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						IlluminationImages.Img[ii + (height - 1 - jj)*width + mm*length + nchannels*(kk*TemporalW + ll)*length] = (double)((int)((unsigned char)view->imageData[nchannels*ii + nchannels*jj*width + mm]));

				Gaussian_smooth_Double(IlluminationImages.Img + mm*length + nchannels*(kk*TemporalW + ll)*length, IlluminationImages.Img + mm*length + nchannels*(kk*TemporalW + ll)*length, height, width, 255.0, LKArg.Gsigma);
				Generate_Para_Spline(IlluminationImages.Img + mm*length + nchannels*(kk*TemporalW + ll)*length, IlluminationImages.Para + mm*length + nchannels*(kk*TemporalW + ll)*length, width, height, LKArg.InterpAlgo);
			}
			cvReleaseImage(&view);
		}
	}

	//2: Compute the flow
	int swidth = (int)width / SRF, sheight = (int)height / SRF, slength = swidth*sheight;
	FlowVect Paraflow(swidth, sheight, nCams);
	float *Fp = new float[6 * nCams*slength], *Rp = new float[6 * nCams*slength];

	for (kk = 0; kk < nCams; kk++)
	{
		if (srp.Precomputed)
		{
			sprintf(FnameX, "%s/Flow/%dX2/FX%d_%05d.dat", DataPATH, (int)(1.0 / srp.SRF), kk + 1, frameID);
			sprintf(FnameY, "%s/Flow/%dX2/FT%d_%05d.dat", DataPATH, (int)(1.0 / srp.SRF), kk + 1, frameID);
		}
		else
		{
			sprintf(FnameX, "%s/Flow/FX%d_%05d.dat", DataPATH, kk + 1, frameID);
			sprintf(FnameY, "%s/Flow/FY%d_%05d.dat", DataPATH, kk + 1, frameID);
		}

		double start = omp_get_wtime();
		if (!ReadFlowBinary(FnameX, FnameY, Fp + 6 * kk*slength, Fp + (6 * kk + 1)*slength, swidth, sheight))
		{
			cout << "Cannot open " << FnameX << " or " << FnameY << endl;
			return 4;
		}
		else
			cout << "Load " << FnameX << ", " << FnameY << " in " << omp_get_wtime() - start << "s" << endl;

		Generate_Para_Spline(Fp + 6 * kk*slength, Paraflow.C12x + kk*length, (int)(width / SRF), (int)(height / SRF), LKArg.InterpAlgo);
		Generate_Para_Spline(Fp + (6 * kk + 1)*slength, Paraflow.C12y + kk*length, (int)(width / SRF), (int)(height / SRF), LKArg.InterpAlgo);

		for (ii = 0; ii < 4; ii++)
		{
			sprintf(Fname, "%s/Flow/F%dp%d_%05d.dat", DataPATH, kk + 1, ii, frameID);
			if (!ReadGridBinary(Fname, Fp + (6 * kk + ii + 2)*slength, swidth, sheight))
				return 0;
		}

		if (srp.Precomputed)
		{
			sprintf(FnameX, "%s/Flow/%dX2/RX%d_%05d.dat", DataPATH, (int)(1.0 / srp.SRF), kk + 1, frameID + frameJump);
			sprintf(FnameY, "%s/Flow/%dX2/RY%d_%05d.dat", DataPATH, (int)(1.0 / srp.SRF), kk + 1, frameID + frameJump);
		}
		else
		{
			sprintf(FnameX, "%s/Flow/RX%d_%05d.dat", DataPATH, kk + 1, frameID + frameJump);
			sprintf(FnameY, "%s/Flow/RY%d_%05d.dat", DataPATH, kk + 1, frameID + frameJump);
		}

		start = omp_get_wtime();
		if (!ReadFlowBinary(FnameX, FnameY, Rp + 6 * kk*slength, Rp + (6 * kk + 1)*slength, swidth, sheight))
		{
			cout << "Cannot open " << FnameX << " or " << FnameY << endl;
			return 4;
		}
		else
			cout << "Load " << FnameX << ", " << FnameY << " in " << omp_get_wtime() - start << "s" << endl;

		for (ii = 0; ii < 4; ii++)
		{
			sprintf(Fname, "%s/Flow/R%dp%d_%05d.dat", DataPATH, kk + 1, ii, frameID + frameJump);
			if (!ReadGridBinary(Fname, Rp + (6 * kk + ii + 2)*slength, swidth, sheight))
				return 0;
		}

		Generate_Para_Spline(Rp + 6 * kk*slength, Paraflow.C21x + kk*length, (int)(width / SRF), (int)(height / SRF), LKArg.InterpAlgo);
		Generate_Para_Spline(Rp + (6 * kk + 1)*slength, Paraflow.C21y + kk*length, (int)(width / SRF), (int)(height / SRF), LKArg.InterpAlgo);
	}

	//Load seoparation results to determine mask if available
	float *TexSep = new float[2 * length];
	float *IllumSep = new float[2 * length];
	bool *TextureMask = new bool[plength*TemporalW * 4]; //because SR for campro is 0.5
	for (ii = 0; ii < plength*TemporalW * 4; ii++)
		TextureMask[ii] = true;

	int x, y;
	for (kk = 0; kk < TemporalW; kk++)
	{
		sprintf(FnameX, "%s/Results/Sep/C1TS%dp0_%05d.dat", DataPATH, 1, frameID);
		sprintf(FnameY, "%s/Results/Sep/C1TS%dp1_%05d.dat", DataPATH, 1, frameID);
		if (ReadFlowBinary(FnameX, FnameY, TexSep, TexSep + length, width, height))
		{
			if (kk == 0)
				cout << "Texture separation available" << endl;
			sprintf(FnameX, "%s/Results/Sep/C%dp0_%05d.dat", DataPATH, 1, frameID);
			sprintf(FnameY, "%s/Results/Sep/C%dp1_%05d.dat", DataPATH, 1, frameID);
			if (ReadFlowBinary(FnameX, FnameY, IllumSep, IllumSep + length, width, height))
			{
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
					{
						if (abs(TexSep[ii + jj*width]) > 0.01 && abs(TexSep[ii + jj*width + length]) > 0.01 && abs(TexSep[ii + jj*width]) < 40 && abs(TexSep[ii + jj*width + length]) < 40)
						{

							x = (int)IllumSep[ii + jj*width] + ii, y = (int)IllumSep[ii + jj*width + length] + jj;
							//TextureMask[x+y*pwidth] = false;
							TextureMask[2 * x + 2 * y * 2 * pwidth + kk*plength * 4] = false;
							TextureMask[2 * x + 1 + 2 * y * 2 * pwidth + kk*plength * 4] = false;
							TextureMask[2 * x + (2 * y + 1) * 2 * pwidth + kk*plength * 4] = false;
							TextureMask[2 * x + 1 + (2 * y + 1) * 2 * pwidth + kk*plength * 4] = false;
						}
					}
			}
		}
	}

	//3. Read sparse correspondence
	int *ntriangles = new int[nCams];
	int *triangleList = new int[nCams * 3 * maxTriangles]; //10000 triangles should be more than enough
	for (kk = 0; kk < nCams; kk++)
	{
		sprintf(Fname, "%s/Sparse/tripletList%d_%05d.txt", DataPATH, kk + 1, frameID + frameJump); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot open tripletList" << endl;
			return 3;
		}
		else
		{
			ii = 0;
			while (fscanf(fp, "%d %d %d ", &triangleList[3 * ii + kk * 3 * maxTriangles], &triangleList[3 * ii + 1 + kk * 3 * maxTriangles], &triangleList[3 * ii + 2 + kk * 3 * maxTriangles]) != EOF)
				ii++;
			fclose(fp);
			ntriangles[kk] = ii;
		}
	}

	int *nPpts = new int[nCams];
	int *CPindex = new int[nCPpts*nCams];
	CPoint2 *Pcorners = new CPoint2[nCPpts*nCams];
	CPoint2 *Ccorners = new CPoint2[nCPpts*TemporalW*nCams];

	CPoint2 DROI[2]; DROI[0].x = pwidth, DROI[1].x = 0, DROI[0].y = pheight, DROI[1].y = 0;
	for (kk = 0; kk < nCams; kk++)
	{
		sprintf(Fname, "%s/Sparse/P%d_%05d.txt", DataPATH, kk + 1, frameID + frameJump); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot open " << Fname << endl;
			return 3;
		}
		ii = 0;
		while (fscanf(fp, "%d %lf %lf ", &CPindex[ii + nCPpts*kk], &Pcorners[ii + nCPpts*kk].x, &Pcorners[ii + nCPpts*kk].y) != EOF)
		{
			if (Pcorners[ii + nCPpts*kk].x < DROI[0].x)
				DROI[0].x = Pcorners[ii + nCPpts*kk].x;
			if (Pcorners[ii + nCPpts*kk].x > DROI[1].x)
				DROI[1].x = Pcorners[ii + nCPpts*kk].x;
			if (Pcorners[ii + nCPpts*kk].y < DROI[0].y)
				DROI[0].y = Pcorners[ii + nCPpts*kk].y;
			if (Pcorners[ii + nCPpts*kk].y > DROI[1].y)
				DROI[1].y = Pcorners[ii + nCPpts*kk].y;
			ii++;
		}
		nPpts[kk] = ii;
		fclose(fp);
	}

	for (kk = 0; kk < nCams; kk++)
	{
		for (jj = 0; jj < TemporalW; jj++)
		{
			sprintf(Fname, "%s/Sparse/CC%d_%05d.txt", DataPATH, kk + 1, frameID + jj*frameJump); FILE *fp = fopen(Fname, "r");
			if (fp == NULL)
			{
				cout << "Cannot open " << Fname << endl;
				return 3;
			}
			ii = 0;
			while (fscanf(fp, "%lf %lf ", &Ccorners[ii + (kk*TemporalW + jj)*nCPpts].x, &Ccorners[ii + (kk*TemporalW + jj)*nCPpts].y) != EOF)
				ii++;
			fclose(fp);
		}
	}
	cout << "Finish loading sparse correspondence" << endl;

	//Load scale information
	double start = omp_get_wtime();
	double *CamProScale = new double[nCams];
	sprintf(Fname, "%s/CamProScale.txt", DataPATH);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 3;
	}
	else
	{
		for (ii = 0; ii < nCams; ii++)
			fscanf(fp, "%lf ", &CamProScale[ii]);
		fclose(fp);
	}

	int *flowhsubset = 0;
	if (!srp.Precomputed)
	{
		flowhsubset = new int[nCams*width*height * 2];
		for (kk = 0; kk < nCams; kk++)
		{
			sprintf(Fname, "%s/Scales/S%dfor_%05d.dat", DataPATH, kk + 1, frameID + frameJump); FILE *fp = fopen(Fname, "r");
			if (!ReadGridBinary(Fname, flowhsubset + 2 * length*kk, width, height))
			{
				for (ii = 0; ii < length; ii++)
					flowhsubset[ii + 2 * length*kk] = LKArg.hsubset;
				cout << "Cannot load scale information" << endl;
				//return 4;
			}

			sprintf(Fname, "%s/Scales/S%dre_%05d.dat", DataPATH, kk + 1, frameID + frameJump); fp = fopen(Fname, "r");
			if (!ReadGridBinary(Fname, flowhsubset + length*(2 * kk + 1), width, height))
			{
				for (ii = 0; ii < length; ii++)
					flowhsubset[ii + length*(2 * kk + 1)] = LKArg.hsubset;
				cout << "Cannot load scale information" << endl;
				//return 4;
			}
		}
	}
	cout << "Finished loading scale information in " << omp_get_wtime() - start << endl;

	//Read campro depth
	start = omp_get_wtime();
	int *CamProMask = new int[plength*nCams*TemporalW * 4]; //because SR for campro is 0.5
	float *CamProDepth = new float[plength*nCams*TemporalW * 4];

	for (ii = 0; ii < plength*nCams*TemporalW * 4; ii++)
		CamProMask[ii] = 0; //not const

	for (ll = 0; ll < nCams; ll++)
	{
		for (kk = 0; kk < TemporalW; kk++)
		{
			sprintf(Fname, "%s/Results/CamPro/PC%d_%05d.ijz", DataPATH, ll + 1, frameID + kk*frameJump); FILE *fp1 = fopen(Fname, "r+");
			if (!ReadGridBinary(Fname, CamProDepth + 4 * kk*plength + ll * 8 * plength, pwidth * 2, pheight * 2))
			{
				cout << "Cannot load " << Fname << endl;
				return 4;
			}
			else
			{
				for (jj = 0; jj < 2 * pheight; jj++)
				{
					for (ii = 0; ii<2 * pwidth; ii++)
					{
						if (abs(CamProDepth[ii + jj*pwidth * 2 + 4 * kk*plength + ll*plength*TemporalW * 4]) > 0.1)
							CamProMask[ii + jj*pwidth * 2 + 4 * kk*plength + ll*plength*TemporalW * 4] = 1;
						else
							CamProMask[ii + jj*pwidth * 2 + 4 * kk*plength + ll*plength*TemporalW * 4] = 0;
					}
				}
			}
		}
	}
	cout << "Finished loading CameraProjector information in " << omp_get_wtime() - start << endl;

	int depthW = (int)((DROI[1].x - DROI[0].x + 1) / srp.Rstep), depthH = (int)((DROI[1].y - DROI[0].y + 1) / srp.Rstep), depthLength = depthW*depthH;
	CPoint2 undistortPt, *IJ = new CPoint2[depthLength];
	double *depth = new double[depthLength*TemporalW];
	int *SubCamProMask = new int[depthLength*TemporalW];
	double *SubCamProDepth = new double[depthLength*TemporalW];
	for (jj = 0; jj < depthH; jj++)
	{
		for (ii = 0; ii < depthW; ii++)
		{
			double proPointX = 1.0*ii*srp.Rstep + DROI[0].x, proPointY = 1.0*jj*srp.Rstep + DROI[0].y;

			IJ[ii + jj*depthW].x = proPointX, IJ[ii + jj*depthW].y = proPointY;
			Undo_distortion(IJ[ii + jj*depthW], DInfo.K, DInfo.distortion);

			int idx = 2 * proPointX, idy = 2 * proPointY;

			SubCamProMask[ii + jj*depthW] = 0, SubCamProMask[ii + jj*depthW + depthW*depthH] = 0;
			SubCamProDepth[ii + jj*depthW] = 0.0, SubCamProDepth[ii + jj*depthW + depthW*depthH] = 0.0;

			if (abs(proPointX * 2 - idx) < 0.1 && abs(proPointY * 2 - idy) < 0.1) //interger or 0.5 type
			{
				for (kk = 0; kk < nCams; kk++)
				{
					if (CamProMask[idx + idy*pwidth * 2 + kk*plength*TemporalW * 4] == 1)
					{
						SubCamProMask[ii + jj*depthW] = CamProMask[idx + idy*pwidth * 2 + kk*TemporalW * 4 * plength];
						SubCamProDepth[ii + jj*depthW] = CamProDepth[idx + idy*pwidth * 2 + kk*TemporalW * 4 * plength];
					}
					if (CamProMask[idx + idy*pwidth * 2 + 4 * plength + kk*plength*TemporalW * 4] == 1)
					{
						SubCamProMask[ii + jj*depthW + depthLength] = CamProMask[idx + idy*pwidth * 2 + 4 * plength + kk*plength*TemporalW * 4];
						SubCamProDepth[ii + jj*depthW + depthLength] = CamProDepth[idx + idy*pwidth * 2 + 4 * plength + kk*plength*TemporalW * 4];
					}
				}
			}
			else
			{
				SubCamProMask[ii + jj*depthW] = 0, SubCamProMask[ii + jj*depthW + depthW*depthH] = 0;
				SubCamProDepth[ii + jj*depthW] = 0.0, SubCamProDepth[ii + jj*depthW + depthW*depthH] = 0.0;
			}
		}
	}
	delete[]CamProMask;
	delete[]CamProDepth;

	SVSR(IlluminationImages, Paraflow, Fp, Rp, flowhsubset, depth, TextureMask, IJ, SubCamProMask, SubCamProDepth, Pcorners, Ccorners, CPindex, nPpts, triangleList, ntriangles, DInfo, srp, LKArg, CamProScale, DROI, width, height, pwidth, pheight, DataPATH, frameID);

	delete[]Fp;
	delete[]Rp;
	delete[]CPindex;
	delete[]nPpts;
	delete[]ntriangles;
	delete[]CamProScale;
	delete[]flowhsubset;
	delete[]Ccorners;
	delete[]Pcorners;
	delete[]SubCamProMask;
	delete[]SubCamProDepth;
	delete[]TextureMask;
	delete[]TexSep;
	delete[]IllumSep;

	return 0;
}
int DepthProgogationDriver(int frameID, int nframes, char *DataPATH)
{
	char Fname[200], FnameX[200], FnameY[200];
	const int width = 1600, height = 1000, pwidth = 1024, pheight = 768, nchannels = 1, length = width*height, plength = pwidth*pheight, nCams = 1, nPros = 1;

	int SparseFlowAlgo = 1, nCPpts = 2962;
	const double flowThresh = 0.1, Gsigma = 1.0;

	LKParameters LKArg;
	LKArg.hsubset = 5, LKArg.Convergence_Criteria = 2, LKArg.DIC_Algo = 2, LKArg.IterMax = 30, LKArg.InterpAlgo = 5, LKArg.ZNCCThreshold = 0.99;
	/*IMPORTANT*/

	int ii, jj, kk, ll, frameJump = 1;

	//1: Setup cameras & projector parameters and load images data
	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, DataPATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	IlluminationFlowImages Fimgs(width, height, pwidth, pheight, nchannels, nCams, nPros, nframes);
	IplImage *view = 0;

	//2. Read initial depth 
	double *InitDepth = new double[plength];
	sprintf(Fname, "%s/GT/D_%05d.ijz", DataPATH, frameID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
		return 2;
	for (ii = 0; ii < plength; ii++)
		fscanf(fp, "%lf ", &InitDepth[ii]);
	fclose(fp);

	float *flowX = new float[length];
	float *flowY = new float[length];
	double *NewDepth = new double[plength];

	double x, y, z, t1, t2;
	CPoint2 DROI[2];
	for (kk = 0; kk < nframes; kk++)
	{
		cout << "Working on frame " << frameID + kk + 1 << endl;

		for (ll = 0; ll < nframes; ll++)
		{
			sprintf(Fname, "%s/Image/C%d_%05d.png", DataPATH, 1, frameID + kk + ll*frameJump);
			view = cvLoadImage(Fname, 0);
			if (!view)
			{
				cout << "Cannot load Images" << endl;
				return 2;
			}
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					Fimgs.Img[ii + (height - 1 - jj)*width + (0 * nframes + ll)*length] = (double)((int)((unsigned char)view->imageData[ii + jj*width]));
			cvReleaseImage(&view);

			Gaussian_smooth_Double(Fimgs.Img + (0 * nframes + ll)*length, Fimgs.Img + (0 * nframes + ll)*length, height, width, 255.0, Gsigma);
			Generate_Para_Spline(Fimgs.Img + (0 * nframes + ll)*length, Fimgs.Para + (0 * nframes + ll)*length, width, height, LKArg.InterpAlgo);
		}

		DROI[0].x = width, DROI[1].x = 0, DROI[0].y = height, DROI[1].y = 0;
		sprintf(Fname, "%s/Sparse/P_%05d.txt", DataPATH, frameID + kk); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot open Projector checker location" << endl;
			return 3;
		}
		fscanf(fp, "%d ", &nCPpts);
		for (ii = 0; ii < nCPpts; ii++)
		{
			fscanf(fp, "%lf %lf ", &t1, &t2);
			if (t1 < DROI[0].x)
				DROI[0].x = t1;
			if (t1 > DROI[1].x)
				DROI[1].x = t1;
			if (t2 < DROI[0].y)
				DROI[0].y = t2;
			if (t2 > DROI[1].y)
				DROI[1].y = t2;
		}
		fclose(fp);
		DROI[1].x = width, DROI[0].x = 0, DROI[1].y = height, DROI[0].y = 0;

		//Read flow
		sprintf(FnameX, "%s/Flow/F%dforX_%05d.dat", DataPATH, 1, frameID + kk + 1);
		sprintf(FnameY, "%s/Flow/F%dforY_%05d.dat", DataPATH, 1, frameID + kk + 1);
		//if(!ReadFlowBinary(FnameX, FnameY, flowX, flowY, width, height))
		//if(!ReadFlowData(FnameX, FnameY, flowX, flowY, width, height))
		{
			cout << "Cannot open flow" << endl;
			return 3;
		}

		//Run depth progogation
		DepthPropagation(Fimgs, flowX, flowY, InitDepth, NewDepth, DInfo, LKArg, width, height, pwidth, pheight, DROI);

		//fprintf new depth
		sprintf(Fname, "%s/Results/D_%05d.ijz", DataPATH, frameID + kk + 1); fp = fopen(Fname, "w+");
		for (jj = 0; jj < pheight; jj++)
		{
			for (ii = 0; ii < pwidth; ii++)
				fprintf(fp, "%.3f ", NewDepth[ii + jj*pwidth]);
			fprintf(fp, "\n");
		}
		fclose(fp);

		//fprintf 3D 
		sprintf(Fname, "%s/Results/D_%05d.xyz", DataPATH, frameID + kk + 1); fp = fopen(Fname, "w+");
		for (jj = 0; jj < pheight; jj++)
		{
			for (ii = 0; ii < pwidth; ii++)
			{
				if (abs(NewDepth[ii + jj*pwidth]) > 0.1)
				{
					x = NewDepth[ii + jj*pwidth] * (DInfo.iK[0] * ii + DInfo.iK[1] * jj + DInfo.iK[2]);
					y = NewDepth[ii + jj*pwidth] * (DInfo.iK[4] * jj + DInfo.iK[5]);
					z = NewDepth[ii + jj*pwidth];
					fprintf(fp, "%.3f %.3f %.3f\n", x, y, z);
				}
			}
		}
		fclose(fp);

		for (ii = 0; ii < plength; ii++)
			InitDepth[ii] = NewDepth[ii];
	}

	delete[]InitDepth;
	delete[]NewDepth;
	delete[]flowX;
	delete[]flowY;

	return 0;
}
int DetectTrackSparseCorners2(int frameID, int nCams, int nPros, int frameJump, int width, int height, int pwidth, int pheight, char *PATH)
{
	int nframes = 2;
	CPoint2 IROI[2], *PROI = new CPoint2[2 * nframes / 2];
	IROI[0].x = 50, IROI[0].y = 10; IROI[1].x = width - 50, IROI[1].y = height - 10;
	for (int ii = 0; ii < nframes / 2; ii++)
	{
		PROI[2 * ii].x = 10, PROI[2 * ii].y = 10;
		PROI[2 * ii + 1].x = pwidth - 10, PROI[2 * ii + 1].y = pheight - 10;
	}

	bool byTracking = false;
	if (0)
	{
		char Fname[200];
		double x0, y0, x1, y1;
		sprintf(Fname, "%s/PROI.txt", PATH);
		FILE *fp = fopen(Fname, "r");
		for (int ii = 0; ii < nframes / 2; ii++)
		{
			fscanf(fp, "%lf %lf %lf %lf ", &x0, &y0, &x1, &y1);
			PROI[2 * ii].x = x0, PROI[2 * ii].y = pheight - y0, PROI[2 * ii + 1].x = x1, PROI[2 * ii + 1].y = pheight - y1;
		}
		fclose(fp);
	}
	else
	{
		for (int ii = 0; ii < nCams; ii++)
			CheckerDetectionTrackingDriver(ii, nCams, nPros, frameJump, frameID, nframes, pwidth, pheight, width, height, PATH, byTracking);
	}

	if (nCams == 2)
		CleanValidCheckerStereo(nCams, frameID, nframes, IROI, PROI, PATH);

	CleanValidChecker(nCams, nPros, frameJump, frameID, nframes, IROI, PROI, PATH);
	TwoDimTriangulation(frameID, nCams, nPros, frameJump, nframes, pwidth, pheight, PATH);

	return 0;
}
int DetectTrackSparseCornersForFlow(int frameID, int nCams, int nPros, int frameJump, int width, int height, int pwidth, int pheight, char *PATH)
{
	const int nframes = 2;
	for (int ii = 0; ii < nframes; ii++)
		CheckerDetectionCorrespondenceDriver(0, nCams, nPros, frameID + ii, pwidth, pheight, width, height, PATH);

	char Fname1[200], Fname2[200];
	int ii, jj, Pid;
	const int nPpts = 936;
	double x, y;

	CPoint2 Cpts[nframes*nPpts];
	int sum, dPind[nframes*nPpts], validCorres[nPpts];

	for (int camID = 0; camID < nCams; camID++)
	{
		sprintf(Fname1, "%s/Sparse/C%d_%05d.txt", PATH, camID + 1, frameID); FILE *fp1 = fopen(Fname1, "w+"); fclose(fp1);
		sprintf(Fname2, "%s/Sparse/C%d_%05d.txt", PATH, camID + 1, frameID + frameJump); FILE *fp2 = fopen(Fname2, "w+"); fclose(fp2);
		for (int proID = 0; proID < nPros; proID++)
		{
			printf("Working on projector %d\n", proID + 1);
			for (ii = 0; ii < nframes*nPpts; ii++)
				dPind[ii] = 0;
			for (ii = 0; ii < nPpts; ii++)
				validCorres[ii] = 0;
			for (jj = 0; jj < 2; jj++)
			{
				sprintf(Fname1, "%s/Sparse/P%dC%d_%05d.txt", PATH, proID + 1, camID + 1, frameID + jj*frameJump); FILE *fp1 = fopen(Fname1, "r");
				sprintf(Fname2, "%s/Sparse/C%dP%d_%05d.txt", PATH, camID + 1, proID + 1, frameID + jj*frameJump); FILE *fp2 = fopen(Fname2, "r");
				if (fp1 == NULL || fp2 == NULL)
					printf("Cannot load %s or %s\n", Fname1, Fname2);
				while (fscanf(fp1, "%d %lf %lf ", &Pid, &x, &y) != EOF)
				{
					dPind[Pid + jj*nPpts] = 1;
					fscanf(fp2, "%lf %lf ", &Cpts[Pid + jj*nPpts].x, &Cpts[Pid + jj*nPpts].y);
				}
				fclose(fp1), fclose(fp2);
			}

			sprintf(Fname1, "%s/Sparse/C%d_%05d.txt", PATH, camID + 1, frameID); FILE *fp1 = fopen(Fname1, "a");
			sprintf(Fname2, "%s/Sparse/C%d_%05d.txt", PATH, camID + 1, frameID + frameJump); FILE *fp2 = fopen(Fname2, "a");
			for (ii = 0; ii < nPpts; ii++)
			{
				if (ii == 685)
					int a = 0;
				sum = 0;
				for (jj = 0; jj < 2; jj++)
					sum += dPind[ii + jj*nPpts];
				if (sum == 2)
				{
					fprintf(fp1, "%.4f %.4f\n", Cpts[ii].x, Cpts[ii].y);
					fprintf(fp2, "%.4f %.4f\n", Cpts[ii + nPpts].x, Cpts[ii + nPpts].y);
				}
			}
			fclose(fp1), fclose(fp2);
		}
	}
	return 0;
}
int ConvertStereoDepthToProjector(char *PATH, int frameID, int nframes, int nCams, int width, int height, int pwidth, int pheight, int nchannels, LKParameters LKArg)
{
	int ii, jj, kk;
	int length = width*height, plength = pwidth*pheight;
	char Fname[200];

	int hsubset = LKArg.hsubset, ConvCriteria = LKArg.Convergence_Criteria, IterMax = LKArg.IterMax, InterpAlgo = LKArg.InterpAlgo, DIC_Algo = LKArg.DIC_Algo;
	double ZNCCThresh = LKArg.ZNCCThreshold;

	float *GroundTruth = new float[4 * plength];
	float *StereoDepth = new float[4 * plength];

	double *Img1 = new double[length*nchannels];
	double *Img2 = new double[length*nchannels];
	double *Img1Para = new double[length*nchannels];
	double *Img2Para = new double[length*nchannels];

	double x, y, z, denum, fufv[2];
	CPoint2 RefPoint, TarPoint;
	CPoint3 WC;

	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	CPoint2 DROI[2];
	//Read ground truth
	sprintf(Fname, "%s/GT/D_%05d.ijz", PATH, frameID);
	if (!ReadGridBinary(Fname, GroundTruth, 2 * pwidth, 2 * pheight))
		return 2;

	//Read Stereo images
	IplImage *view = 0;
	sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 1, frameID);
	view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
	if (!view)
	{
		cout << "Cannot load " << Fname << endl;
		return 4;
	}
	for (kk = 0; kk < nchannels; kk++)
	{
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				Img1[ii + (height - 1 - jj)*width + kk*length] = (double)((int)((unsigned char)view->imageData[nchannels*ii + nchannels*jj*width + kk]));

		Gaussian_smooth_Double(Img1 + kk*length, Img1 + kk*length, height, width, 255.0, LKArg.Gsigma);
		Generate_Para_Spline(Img1 + kk*length, Img1Para + kk*length, width, height, LKArg.InterpAlgo);
	}
	cvReleaseImage(&view);

	sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 2, frameID);
	view = cvLoadImage(Fname, nchannels == 1 ? 0 : 1);
	if (!view)
	{
		cout << "Cannot load " << Fname << endl;
		return 4;
	}
	for (kk = 0; kk < nchannels; kk++)
	{
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				Img2[ii + (height - 1 - jj)*width + kk*length] = (double)((int)((unsigned char)view->imageData[nchannels*ii + nchannels*jj*width + kk]));

		Gaussian_smooth_Double(Img2 + kk*length, Img2 + kk*length, height, width, 255.0, LKArg.Gsigma);
		Generate_Para_Spline(Img2 + kk*length, Img2Para + kk*length, width, height, LKArg.InterpAlgo);
	}
	cvReleaseImage(&view);


	//Project to camera 1 and compute depth
	double start = omp_get_wtime();
	int percent = 10, increment = 10;

	int	nvalidPoints = 0, count = 0, computedPoints = 0;
	for (jj = 0; jj < 2 * pheight; jj++)
		for (ii = 0; ii < 2 * pwidth; ii++)
			if (abs(GroundTruth[ii + jj * 2 * pwidth]) > 1.0)
				count++;

	cout << "Running with appx " << count << " points" << endl;
	nvalidPoints = count, count = 0;
	for (jj = 0; jj < 2 * pheight; jj++)
	{
		for (ii = 0; ii < 2 * pwidth; ii++)
		{
			if (abs(GroundTruth[ii + jj * 2 * pwidth]) < 1.0)
			{
				StereoDepth[ii + jj * 2 * pwidth] = 0.0;
				continue;
			}

			if ((100 * count / nvalidPoints - percent) > 0)
			{
				double elapsed = omp_get_wtime() - start;
				cout << "%" << 100 * count / nvalidPoints << " " << computedPoints << " computed. Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / (percent)*(100.0 - percent) << endl;
				percent += increment;
			}
			count++;

			x = GroundTruth[ii + jj * 2 * pwidth] * (DInfo.iK[0] * ii / 2 + DInfo.iK[1] * jj / 2 + DInfo.iK[2]);
			y = GroundTruth[ii + jj * 2 * pwidth] * (DInfo.iK[4] * jj / 2 + DInfo.iK[5]);
			z = GroundTruth[ii + jj * 2 * pwidth];

			denum = DInfo.P[8] * x + DInfo.P[9] * y + DInfo.P[10] * z + DInfo.P[11];
			RefPoint.x = (DInfo.P[0] * x + DInfo.P[1] * y + DInfo.P[2] * z + DInfo.P[3]) / denum;
			RefPoint.y = (DInfo.P[4] * x + DInfo.P[5] * y + DInfo.P[6] * z + DInfo.P[7]) / denum;

			if (RefPoint.x<10 || RefPoint.x<10 || RefPoint.y>width - 10 || RefPoint.y>height - 10)
				StereoDepth[ii + jj * 2 * pwidth] = 0.0;
			else
			{
				denum = DInfo.P[20] * x + DInfo.P[21] * y + DInfo.P[22] * z + DInfo.P[23];
				TarPoint.x = (DInfo.P[12] * x + DInfo.P[13] * y + DInfo.P[14] * z + DInfo.P[15]) / denum;
				TarPoint.y = (DInfo.P[16] * x + DInfo.P[17] * y + DInfo.P[18] * z + DInfo.P[19]) / denum;

				if (TarPoint.x<10 || TarPoint.x<10 || TarPoint.y>width - 10 || TarPoint.y>height - 10)
					StereoDepth[ii + jj * 2 * pwidth] = 0.0;
				else
				{
					if (TMatching(Img1Para, Img2Para, hsubset, width, height, width, height, nchannels, RefPoint, TarPoint, DIC_Algo, ConvCriteria, ZNCCThresh, IterMax, InterpAlgo, fufv, false) > ZNCCThresh)
					{
						TarPoint.x = TarPoint.x + fufv[0], TarPoint.y = TarPoint.y + fufv[1];
						Undo_distortion(RefPoint, DInfo.K + 9, DInfo.distortion + 13);
						Undo_distortion(TarPoint, DInfo.K + 18, DInfo.distortion + 26);
						Stereo_Triangulation2(&RefPoint, &TarPoint, DInfo.P, DInfo.P + 12, &WC);
						StereoDepth[ii + jj * 2 * pwidth] = (float)WC.z;
						computedPoints++;
					}
					else
						StereoDepth[ii + jj * 2 * pwidth] = 0.0f;
				}
			}
		}
	}
	cout << "Total time: " << setw(2) << omp_get_wtime() - start << " with " << computedPoints << " computed. " << endl;

	sprintf(Fname, "%s/Results/Stereo/PStereo_%05d.xyz", PATH, frameID);  FILE *fp = fopen(Fname, "w+");
	for (jj = 0; jj < 2 * pheight; jj++)
	{
		for (ii = 0; ii < 2 * pwidth; ii++)
			if (abs(StereoDepth[ii + jj * 2 * pwidth]) > 1.0)
			{
				double	rayDirectX = DInfo.iK[0] * ii / 2 + DInfo.iK[1] * jj / 2 + DInfo.iK[2], rayDirectY = DInfo.iK[4] * jj / 2 + DInfo.iK[5];
				fprintf(fp, "%.3f %.3f %.3f\n", rayDirectX*StereoDepth[ii + jj * 2 * pwidth], rayDirectY*StereoDepth[ii + jj * 2 * pwidth], StereoDepth[ii + jj * 2 * pwidth]);
			}
	}
	fclose(fp);

	//Write out the results
	sprintf(Fname, "%s/Results/Stereo/PStereo_%05d.ijz", PATH, frameID);
	WriteGridBinary(Fname, StereoDepth, 2 * pwidth, 2 * pheight);

	delete[]GroundTruth;
	delete[]StereoDepth;
	delete[]Img1;
	delete[]Img2;
	delete[]Img1Para;
	delete[]Img2Para;

	return 0;
}
int CompareWithGroundTruthFlow(char *PATH, int frameID, int nchannels, int width, int height, int pwidth, int pheight, double SR, int Resolution, int noise)
{
	int ii, jj, kk, ll;
	int length = width*height, plength = pwidth*pheight, nCams = 1, nPros = 1, nframes = 2, frameJump = 1;
	char Fname[200];

	double intensity, flowThresh = 0.1, Gsigma = 1.0;
	LKParameters LKArg;
	LKArg.step = 1, LKArg.hsubset = 5, LKArg.InterpAlgo = 1;
	LKArg.Incomplete_Subset_Handling = 0, LKArg.Convergence_Criteria = 0, LKArg.Analysis_Speed = 0, LKArg.IterMax = 20;
	LKArg.PSSDab_thresh = 0.01, LKArg.ZNCCThreshold = 0.99, LKArg.Gsigma = 1.0;

	float *Depth1 = new float[(int)(1.0*plength / SR / SR)];
	float *Depth2 = new float[(int)(1.0*plength / SR / SR)];
	double *FlowDif1 = new double[(int)(1.0*plength / SR / SR)];
	double *FlowDif2 = new double[(int)(1.0*plength / SR / SR)];
	int *code = new int[(int)(1.0*plength / SR / SR)];

	DevicesInfo DInfo(nCams);
	sprintf(Fname, "%s/%d", PATH, Resolution);
	if (!SetUpDevicesInfo(DInfo, Fname))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}
	else
		cout << "Load CamPro Info" << endl;

	IlluminationFlowImages Fimgs(width, height, pwidth, pheight, nchannels, nCams, nPros, nframes);
	Fimgs.frameJump = 1;
	IplImage *view = 0;
	for (kk = 0; kk < nCams; kk++)
	{
		for (ll = 0; ll < nframes; ll++)
		{
			sprintf(Fname, "%s/%d/Image/Luminance%d_%05d.png", PATH, Resolution, kk + 1, frameID + ll*frameJump);
			view = cvLoadImage(Fname, 0);
			if (!view)
			{
				cout << "Cannot load Images" << endl;
				return 2;
			}
			cout << "Load " << Fname << endl;

			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
				{
					intensity = (double)((int)((unsigned char)view->imageData[ii + jj*width])) + gaussian_noise(0, noise);
					if (intensity > 255.0)
						intensity = 255.0;
					else if (intensity < 0.0)
						intensity = 0.0;
					Fimgs.Img[ii + (height - 1 - jj)*width + (kk*nframes + ll)*length] = intensity;
				}
			cvReleaseImage(&view);

			Gaussian_smooth_Double(Fimgs.Img + (kk*nframes + ll)*length, Fimgs.Img + (kk*nframes + ll)*length, height, width, 255.0, Gsigma);
			Generate_Para_Spline(Fimgs.Img + (kk*nframes + ll)*length, Fimgs.Para + (kk*nframes + ll)*length, width, height, LKArg.InterpAlgo);
		}
	}

	for (ii = 0; ii < plength; ii++)
	{
		Depth1[ii] = 0.0;
		Depth2[ii] = 0.0;
		FlowDif1[ii] = 0.0;
		FlowDif2[ii] = 0.0;
	}

	sprintf(Fname, "%s/GT/D_%05d.ijz", PATH, frameID);
	if (!ReadGridBinary(Fname, Depth1, (int)(1.0*pwidth / SR), (int)(1.0*pheight / SR)))
		return 1;
	else
		cout << "Load " << Fname << endl;

	sprintf(Fname, "%s/GT/D_%05d.ijz", PATH, frameID + 1);
	if (!ReadGridBinary(Fname, Depth2, (int)(1.0*pwidth / SR), (int)(1.0*pheight / SR)))
		return 1;
	else
		cout << "Load " << Fname << endl;

	CPoint2 DROI[2];
	/*DROI[0].x = pwidth*SR, DROI[1].x = 0, DROI[0].y = pheight*SR, DROI[1].y = 0;
	sprintf(Fname, "%s/Sparse/P_%05d.txt",PATH, frameID); fp = fopen(Fname, "r");
	if(fp==NULL)
	{
	cout<<"Cannot open Projector checker location"<<endl;
	return 3;
	}
	fscanf(fp, "%d ", &nCPpts);
	for(ii=0; ii<nCPpts; ii++)
	{
	fscanf(fp, "%lf %lf ", &t1, &t2);
	if(t1 < DROI[0].x)
	DROI[0].x = t1;
	if(t1 > DROI[1].x)
	DROI[1].x = t1;
	if(t2 < DROI[0].y)
	DROI[0].y = t2;
	if(t2 > DROI[1].y)
	DROI[1].y = t2;
	}
	fclose(fp);*/
	DROI[1].x = pwidth*SR - 100, DROI[0].x = 100, DROI[1].y = pheight*SR - 100, DROI[0].y = 100;

	double x, y, z, u1, v1, u2, v2, n1, n2, denum, ProP[3], EpiLine[3];
	CPoint2 dPts[2], nPts;
	int nvalidpoints = 0;
	for (jj = 0; jj < (int)(1.0*pheight / SR); jj += (int)(1.0 / SR))
	{
		for (ii = 0; ii < (int)(1.0*pwidth / SR); ii += (int)(1.0 / SR))
		{
			if (abs(Depth1[ii + jj*(int)(1.0*pwidth / SR)]) < 0.1 || abs(Depth2[ii + jj*(int)(1.0*pwidth / SR)]) < 0.1 || ii<DROI[0].x || jj<DROI[0].y || ii>DROI[1].x || jj>DROI[1].y)
				continue;
			nvalidpoints++;
		}
	}

	srand(0);
	int startScale = 3, stopScale = 18, stepScale = 1, count = 0;
	int *ScaleList = new int[4 * plength];
	bool scaleEstimate = false;
	sprintf(Fname, "%s/%d/Scales/%05d.txt", PATH, Resolution, frameID);
	if (!ReadGridBinary(Fname, ScaleList, pwidth, pheight))
		scaleEstimate = true;

	double *RefPatch = new double[(2 * stopScale + 1)*(2 * stopScale + 1)*nchannels];
	double *TarPatch = new double[(2 * stopScale + 1)*(2 * stopScale + 1)*nchannels];
	double *ZNCCStorage = new double[2 * (2 * stopScale + 1)*(2 * stopScale + 1)*nchannels];

	double start = omp_get_wtime();
	int percent = 1, step = 1;

	cout << "Working on resolution " << Resolution << " ... with ADWN " << noise << endl;
	for (jj = 0; jj < (int)(1.0*pheight / SR); jj += (int)(1.0 / SR))
	{
		for (ii = 0; ii < (int)(1.0*pwidth / SR); ii += (int)(1.0 / SR))
		{
			if (100.0*count / nvalidpoints - percent >= 0)
			{
				double elapsed = omp_get_wtime() - start;
				cout << "%" << percent << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / percent*(100.0 - percent) << endl;
				percent += step;
			}

			if (abs(Depth1[ii + jj*(int)(1.0*pwidth / SR)]) < 0.1 || abs(Depth2[ii + jj*(int)(1.0*pwidth / SR)]) < 0.1 || ii<DROI[0].x || jj<DROI[0].y || ii>DROI[1].x || jj>DROI[1].y)
			{
				FlowDif1[ii + jj*(int)(1.0*pwidth / SR)] = 0.0;
				FlowDif2[ii + jj*(int)(1.0*pwidth / SR)] = 0.0;
				continue;
			}
			count++;
			ProP[0] = 1.0*ii, ProP[1] = 1.0*jj, ProP[2] = 1.0;
			mat_mul(DInfo.FmatPC, ProP, EpiLine, 3, 3, 1);

			x = Depth1[ii + jj*(int)(1.0*pwidth / SR)] * (DInfo.iK[0] * ii / SR + DInfo.iK[1] * jj / SR + DInfo.iK[2]);
			y = Depth1[ii + jj*(int)(1.0*pwidth / SR)] * (DInfo.iK[4] * jj / SR + DInfo.iK[5]);
			z = Depth1[ii + jj*(int)(1.0*pwidth / SR)];

			denum = DInfo.P[8] * x + DInfo.P[9] * y + DInfo.P[10] * z + DInfo.P[11];
			u1 = (DInfo.P[0] * x + DInfo.P[1] * y + DInfo.P[2] * z + DInfo.P[3]) / denum;
			v1 = (DInfo.P[4] * x + DInfo.P[5] * y + DInfo.P[6] * z + DInfo.P[7]) / denum;

			x = Depth2[ii + jj*(int)(1.0*pwidth / SR)] * (DInfo.iK[0] * ii / SR + DInfo.iK[1] * jj / SR + DInfo.iK[2]);
			y = Depth2[ii + jj*(int)(1.0*pwidth / SR)] * (DInfo.iK[4] * jj / SR + DInfo.iK[5]);
			z = Depth2[ii + jj*(int)(1.0*pwidth / SR)];

			denum = DInfo.P[8] * x + DInfo.P[9] * y + DInfo.P[10] * z + DInfo.P[11];
			u2 = (DInfo.P[0] * x + DInfo.P[1] * y + DInfo.P[2] * z + DInfo.P[3]) / denum;
			v2 = (DInfo.P[4] * x + DInfo.P[5] * y + DInfo.P[6] * z + DInfo.P[7]) / denum;

			double multiplier = (Resolution == 0) ? 0.7 : 1.0*Resolution;

			n1 = gaussian_noise(0, 0.1*multiplier);
			if (n1 > 0.3*multiplier)
				n1 = 0.3*multiplier;
			else if (n1<-0.3*multiplier)
				n1 = -0.3*multiplier;
			n2 = gaussian_noise(0, 0.1*multiplier);
			if (n2>0.3*multiplier)
				n2 = 0.3*multiplier;
			else if (n2 < -0.3*multiplier)
				n2 = -0.3*multiplier;

			dPts[0].x = u1, dPts[0].y = v1;
			dPts[1].x = u2 + n1, dPts[1].y = v2 + n2;

			//Search for optimal subset size:		
			double difference, minDif = 9e9, ZNCC;
			if (scaleEstimate)
			{
				double fufv[2];
				CPoint2 PR, PT, tpoint;
				for (kk = startScale; kk < stopScale; kk += stepScale)
				{
					PR.x = u1, PR.y = v1, PT.x = u2, PT.y = v2;
					ZNCC = TMatching(Fimgs.Para, Fimgs.Para + length, kk, width, height, width, height, nchannels, PR, PT, 1,
						LKArg.Convergence_Criteria, LKArg.ZNCCThreshold, LKArg.IterMax, LKArg.InterpAlgo, fufv, false, NULL, NULL, RefPatch, ZNCCStorage);
					if (ZNCC < LKArg.ZNCCThreshold)
						continue;

					PT.x = PT.x + fufv[0], PT.y = PT.y + fufv[1], tpoint.x = PR.x, tpoint.y = PR.y;
					ZNCC = TMatching(Fimgs.Para + length, Fimgs.Para, kk, width, height, width, height, nchannels, PT, tpoint, 1,
						LKArg.Convergence_Criteria, LKArg.ZNCCThreshold, LKArg.IterMax, LKArg.InterpAlgo, fufv, false, NULL, NULL, RefPatch, ZNCCStorage);
					if (ZNCC < LKArg.ZNCCThreshold)
						continue;

					tpoint.x = tpoint.x + fufv[0], tpoint.y = tpoint.y + fufv[1];
					difference = abs(tpoint.x - PR.x) + abs(tpoint.y - PR.y);
					if (minDif > difference)
					{
						LKArg.hsubset = kk;
						minDif = difference;
					}
					if (minDif > 1.0)
						LKArg.hsubset = 0;
				}
			}
			else
				LKArg.hsubset = ScaleList[ii + jj*pwidth];

			ScaleList[ii + jj*pwidth] = LKArg.hsubset;
			if (LKArg.hsubset == 0)
			{
				FlowDif1[ii + jj*pwidth] = 0.0;
				FlowDif2[ii + jj*pwidth] = 0.0;
				ScaleList[ii + jj*pwidth] = 0;
				continue;
			}

			nPts.x = u2 + n1, nPts.y = v2 + n2; LKArg.DIC_Algo = 1;
			//Project to epipolar line
			denum = pow(EpiLine[0], 2) + pow(EpiLine[1], 2);
			dPts[1].x = (EpiLine[1] * (EpiLine[1] * nPts.x - EpiLine[0] * nPts.y) - EpiLine[0] * EpiLine[2]) / denum;
			dPts[1].y = (EpiLine[0] * (-EpiLine[1] * nPts.x + EpiLine[0] * nPts.y) - EpiLine[1] * EpiLine[2]) / denum;

			ZNCC = EpipSearchLK(dPts, EpiLine, Fimgs.Img, Fimgs.Img + length, Fimgs.Para, Fimgs.Para + length, nchannels, width, height, width, height, LKArg, RefPatch, ZNCCStorage, TarPatch, NULL, NULL);
			if (ZNCC < LKArg.ZNCCThreshold)
				FlowDif1[ii + jj*pwidth] = 10.0;
			else
				FlowDif1[ii + jj*pwidth] = sqrt(pow(dPts[1].x - u2, 2) + pow(dPts[1].y - v2, 2));

			dPts[1].x = u2 + n1, dPts[1].y = v2 + n2; LKArg.DIC_Algo = 3;
			ZNCC = EpipSearchLK(dPts, EpiLine, Fimgs.Img, Fimgs.Img + length, Fimgs.Para, Fimgs.Para + length, nchannels, width, height, width, height, LKArg, RefPatch, ZNCCStorage, TarPatch, NULL, NULL);
			if (ZNCC < LKArg.ZNCCThreshold)
				FlowDif2[ii + jj*pwidth] = 10.0;
			else
				FlowDif2[ii + jj*pwidth] = sqrt(pow(dPts[1].x - u2, 2) + pow(dPts[1].y - v2, 2));
		}
	}
	cout << "Nprocessed: " << count << "in " << omp_get_wtime() - start << "s" << endl;

	if (scaleEstimate)
	{
		sprintf(Fname, "%s/%d/Scales/%05d.txt", PATH, Resolution, frameID);
		WriteGridBinary(Fname, ScaleList, pwidth, pheight);
	}

	sprintf(Fname, "%s/%d_%d_FlowDif.txt", PATH, Resolution, 1);
	WriteGridBinary(Fname, FlowDif1, pwidth, pheight);

	sprintf(Fname, "%s/%d_%d_FlowDif.txt", PATH, Resolution, 3);
	WriteGridBinary(Fname, FlowDif2, pwidth, pheight);

	delete[]Depth1;
	delete[]Depth2;
	delete[]FlowDif1;
	delete[]FlowDif2;
	delete[]ScaleList;
	delete[]RefPatch;
	delete[]TarPatch;
	delete[]ZNCCStorage;

	return 0;
}
int CompareWithGroundTruth(char *bPATH, char *PATH, int frameID, int nCams, int width, int height, int SuperRes)
{
	double maxErrorCutoff = 0.001;

	int ii, jj, kk;
	int length = width*height;
	double x, y, z;
	char Fname[200];

	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot ProCam Info" << endl;
		return 1;
	}

	float *GroundTruth = new float[length*SuperRes*SuperRes];
	double *ProCam = new double[length*SuperRes*SuperRes];
	float *Stereo = new float[length*SuperRes*SuperRes];
	double *SVSR = new double[length*SuperRes*SuperRes];
	double *ProCamDiff = new double[length*SuperRes*SuperRes];
	double *StereoDiff = new double[length*SuperRes*SuperRes];
	double *SVSRDiff = new double[length*SuperRes*SuperRes];
	bool *mask = new bool[length*SuperRes*SuperRes];
	double *code = new double[length*SuperRes*SuperRes];
	unsigned char *colorCode = new unsigned char[3 * length*SuperRes*SuperRes];

	int nCPpts = 3871;
	CPoint2 *Pcorners = new CPoint2[nCPpts*nCams];
	CPoint2 DROI[2]; DROI[0].x = width, DROI[1].x = 0, DROI[0].y = height, DROI[1].y = 0;
	for (kk = 0; kk < nCams; kk++)
	{
		sprintf(Fname, "%s/Sparse/P%d_%05d.txt", PATH, kk + 1, frameID); FILE *fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 3;
		}
		ii = 0;
		while (fscanf(fp, "%d %lf %lf ", &jj, &Pcorners[ii + nCPpts*kk].x, &Pcorners[ii + nCPpts*kk].y) != EOF)
		{
			if (Pcorners[ii + nCPpts*kk].x < DROI[0].x)
				DROI[0].x = Pcorners[ii + nCPpts*kk].x;
			if (Pcorners[ii + nCPpts*kk].x > DROI[1].x)
				DROI[1].x = Pcorners[ii + nCPpts*kk].x;
			if (Pcorners[ii + nCPpts*kk].y < DROI[0].y)
				DROI[0].y = Pcorners[ii + nCPpts*kk].y;
			if (Pcorners[ii + nCPpts*kk].y > DROI[1].y)
				DROI[1].y = Pcorners[ii + nCPpts*kk].y;
			ii++;
		}
		fclose(fp);
	}

	for (ii = 0; ii < length*SuperRes*SuperRes; ii++)
	{
		GroundTruth[ii] = 0.0;
		ProCam[ii] = 0.0;
		Stereo[ii] = 0.0;
		SVSR[ii] = 0.0;
		SVSRDiff[ii] = 9e9;
	}

	sprintf(Fname, "%s/GT/D_%05d.ijz", bPATH, frameID);
	float *BGroundTruth = new float[length * 4];
	if (!ReadGridBinary(Fname, BGroundTruth, width * 2, height * 2))
	{
		cout << "Cannot load " << Fname << endl;
		return 4;
	}
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			GroundTruth[ii + jj*width] = BGroundTruth[2 * ii + 2 * jj * 2 * width];
		}
	}


	int depthW = (int)((DROI[1].x - DROI[0].x + 1)*SuperRes), depthH = (int)((DROI[1].y - DROI[0].y + 1)*SuperRes);
	sprintf(Fname, "%s/Results/CamPro/PC1_%05d.ijz", PATH, frameID);
	//if(!ReadGridBinary(Fname, ProCam, width*2, height*2))
	if (!ReadGridBinary(Fname, ProCam, depthW, depthH))
	{
		cout << "Cannot load " << Fname << endl;
		return 4;
	}

	/*sprintf(Fname, "%s/Results/Stereo/PStereo_%05d.ijz", PATH, frameID);
	if(!ReadGridBinary(Fname, Stereo,  width*2, height*2))
	{
	cout<<"Cannot load "<<Fname<<endl;
	return 4;
	}*/

	//int depthW = (int)((DROI[1].x-DROI[0].x+1)*2), depthH = (int)((DROI[1].y-DROI[0].y+1)*2);
	sprintf(Fname, "%s/Results/SVSR/L0/0.50-10.00-0.00-1-3_3D_%05d.ijz", PATH, frameID);
	if (!ReadGridBinary(Fname, SVSR, depthW, depthH))
	{
		cout << "Cannot load " << Fname << endl;
		return 4;
	}
	/*{
	sprintf(Fname, "%s/Results/SVSR/L0/0.30-0.00-1.00-3-3_3D_%05d.xyz", PATH, frameID);
	FILE *fp = fopen(Fname, "w+");
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	if(abs(SVSR[ii+jj*depthW]) < 1)
	continue;
	double	rayDirectX = DInfo.iK[0]*(DROI[0].x+ii)+DInfo.iK[1]*(DROI[0].y+jj)+DInfo.iK[2], rayDirectY = DInfo.iK[4]*(DROI[0].y+jj)+DInfo.iK[5];
	fprintf(fp, "%.3f %.3f %.3f \n", rayDirectX*SVSR[ii+jj*depthW], rayDirectY*SVSR[ii+jj*depthW], SVSR[ii+jj*depthW]);
	}
	}
	fclose(fp);
	return 0;
	}*/
	/*float *SubProCam = new float[width*height];
	float *ParaSubProCam = new float[width*height];
	for(jj=0; jj<height; jj++)
	for(ii=0; ii<width; ii++)
	SubProCam[ii+jj*width] =  ProCam[(2*ii+1)+(jj*2+1)*2*width];
	Generate_Para_Spline(SubProCam, ParaSubProCam, width, height, 1);

	double S[3];
	for(jj=0; jj<SuperRes*height; jj++)
	for(ii=0; ii<SuperRes*width; ii++)
	{
	Get_Value_Spline_Float(ParaSubProCam, width, height, 1.0*ii/SuperRes, 1.0*jj/SuperRes, S, -1, 1);
	ProCam[ii+jj*SuperRes*width] = S[0];
	}*/

	for (jj = 0; jj < SuperRes*height; jj++)
	{
		for (ii = 0; ii < SuperRes*width; ii++)
		{
			;//ProCamDiff[ii+jj*SuperRes*width] = ProCam[ii+jj*SuperRes*width]-GroundTruth[ii+jj*SuperRes*width];
			//StereoDiff[ii+jj*width] = Stereo[2*ii+1+(2*jj+1)*width]-GroundTruth[2*ii+1+(2*jj+1)*width];
		}
	}


	for (jj = 0; jj < depthH; jj++)
	{
		for (ii = 0; ii < depthW; ii++)
		{
			double proPointX = SuperRes*DROI[0].x + ii, proPointY = SuperRes*DROI[0].y + jj;
			SVSRDiff[(int)(proPointX)+(int)(proPointY)*SuperRes*width] = SVSR[ii + jj*depthW] - GroundTruth[(int)(proPointX)+(int)(proPointY)*SuperRes*width];
			ProCamDiff[(int)(proPointX)+(int)(proPointY)*SuperRes*width] = ProCam[ii + jj*depthW] - GroundTruth[(int)(proPointX)+(int)(proPointY)*SuperRes*width];
			//SVSR[ii+jj*depthW] = SVSR[ii+jj*depthW]-GroundTruth[(int)(proPointX)+(int)(proPointY)*SuperRes*width];
			//ProCam[ii+jj*depthW] = ProCam[ii+jj*depthW]-GroundTruth[(int)(proPointX)+(int)(proPointY)*SuperRes*width];
		}
	}


	//WriteGridBinary("C:/temp/SVSR.dat", SVSR, depthW, depthH);
	//WriteGridBinary("C:/temp/ProCam.dat", ProCam, depthW, depthH);

	FILE *fp = 0;
	/*sprintf(Fname, "%s/Results/ColorCoded/Label_%05d.xyzc", PATH, frameID); fp = fopen(Fname, "w+");
	for(jj=0; jj<SuperRes*height; jj++)
	{
	for(ii=0; ii<SuperRes*width; ii++)
	{
	if(!(ii>DROI[0].x*SuperRes && ii < DROI[1].x*SuperRes && jj>DROI[0].y*SuperRes && jj<DROI[1].y*SuperRes))
	continue;
	if(abs(GroundTruth[ii+jj*SuperRes*width]) < 1.0)
	continue;

	allDiff[0] = abs(ProCamDiff[ii+jj*SuperRes*width]), allDiff[1] = 1000.0,//abs(StereoDiff[ii+jj*width]),
	allDiff[2] = abs(SVSRDiff[ii+jj*SuperRes*width]);
	if(abs(allDiff[2] - allDiff[0]) < 0.00001)
	allDiff[0] += 0.0001;
	typeCode[0] = 1, typeCode[1] = 2, typeCode[2] = 3;
	Quick_Sort_Double(allDiff, typeCode, 0, 2);

	z = (float)GroundTruth[ii+jj*SuperRes*width];
	x = (float)(z*(DInfo.iK[0]*ii/SuperRes + DInfo.iK[1]*jj/SuperRes + DInfo.iK[2]));
	y = (float)(z*(DInfo.iK[4]*jj/SuperRes+ DInfo.iK[5]));
	if(allDiff[0] < abs(GroundTruth[ii+jj*SuperRes*width]*0.001) )
	{
	if( typeCode[0] == 1)
	fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", x, y ,z, 255, 0, 0); //red
	if( typeCode[0] == 2)
	fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", x, y ,z, 0, 255, 0);
	if( typeCode[0] == 3)
	fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", x, y ,z, 0, 0, 255); //blue
	}
	}
	}
	fclose(fp);*/

	//Determine common range for max and min error
	int ProCamcount = 0, SVSRcount = 0;
	double minErr = 9e9, maxErr = -9e9;
	for (jj = 0; jj < SuperRes*height; jj++)
	{
		for (ii = 0; ii<SuperRes*width; ii++)
		{
			if (!(ii>DROI[0].x*SuperRes && ii < DROI[1].x*SuperRes && jj>DROI[0].y*SuperRes && jj < DROI[1].y*SuperRes))
				continue;
			if (abs(GroundTruth[ii + jj*SuperRes*width]) < 1.0)
				continue;

			if (abs(ProCamDiff[ii + jj*SuperRes*width]) < abs(GroundTruth[ii + jj*SuperRes*width])*maxErrorCutoff)
			{
				ProCamcount++;
				if (ProCamDiff[ii + jj*SuperRes*width] < minErr)
					minErr = ProCamDiff[ii + jj*SuperRes*width];
				if (ProCamDiff[ii + jj*SuperRes*width] > maxErr)
					maxErr = ProCamDiff[ii + jj*SuperRes*width];
			}

			/*if(abs(StereoDiff[ii+jj*width]) > 0.1 && abs(StereoDiff[ii+jj*width]) < maxErrorCutoff )
			{
			if(StereoDiff[ii+jj*width]<minErr)
			minErr = StereoDiff[ii+jj*width];
			if(StereoDiff[ii+jj*width]>maxErr)
			maxErr = StereoDiff[ii+jj*width];
			}*/

			if (abs(SVSRDiff[ii + jj*SuperRes*width]) < abs(GroundTruth[ii + jj*SuperRes*width])*maxErrorCutoff)
			{
				SVSRcount++;
				if (SVSRDiff[ii + jj*SuperRes*width] < minErr)
					minErr = SVSRDiff[ii + jj*SuperRes*width];
				if (SVSRDiff[ii + jj*SuperRes*width] > maxErr)
					maxErr = SVSRDiff[ii + jj*SuperRes*width];
			}
		}
	}
	sprintf(Fname, "%s/DepthCom.txt", bPATH); fp = fopen(Fname, "a+");
	fprintf(fp, "%s: ProCam, SVSR:  %d %d\n", PATH, ProCamcount, SVSRcount);
	fclose(fp);


	//Make color coded error for individual data type
	for (jj = 0; jj < SuperRes*height; jj++)
	{
		for (ii = 0; ii < SuperRes*width; ii++)
		{
			if (abs(ProCamDiff[ii + jj*SuperRes*width]) > abs(GroundTruth[ii + jj*SuperRes*width])*maxErrorCutoff || abs(GroundTruth[ii + jj*SuperRes*width]) < 1.0)
			{
				mask[ii + jj*SuperRes*width] = true;
				code[ii + jj*SuperRes*width] = 0.0;
			}
			else
			{
				mask[ii + jj*SuperRes*width] = false;
				code[ii + jj*SuperRes*width] = ((ProCamDiff[ii + jj*SuperRes*width] - minErr) / (maxErr - minErr) - 0.5)*2.0;
			}
		}
	}
	ConvertToHeatMap(code, colorCode, SuperRes*width, SuperRes*height, mask);

	sprintf(Fname, "%s/Results/ColorCoded/ProCam_%05d.xyzc", PATH, frameID); fp = fopen(Fname, "w+");
	if (fp == NULL)
		return false;
	for (jj = 0; jj < SuperRes*height; jj++)
	{
		for (ii = 0; ii<SuperRes*width; ii++)
		{
			if (ii>DROI[0].x*SuperRes && ii < DROI[1].x*SuperRes && jj>DROI[0].y*SuperRes && jj < DROI[1].y*SuperRes)
			{
				if (abs(GroundTruth[ii + jj*SuperRes*width]) < 1.0)
					continue;
				else
				{
					z = (float)GroundTruth[ii + SuperRes*jj*width];
					x = (float)(z*(DInfo.iK[0] * ii / SuperRes + DInfo.iK[1] * jj / SuperRes + DInfo.iK[2]));
					y = (float)(z*(DInfo.iK[4] * jj / SuperRes + DInfo.iK[5]));
					fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", x, y, z, colorCode[3 * ii + 3 * jj*SuperRes*width], colorCode[3 * ii + 1 + 3 * jj*SuperRes*width], colorCode[3 * ii + 2 + 3 * jj*SuperRes*width]);
				}
			}
		}
	}
	fclose(fp);

	//Stereo:
	/*for(jj=0; jj<height; jj++)
	{
	for(ii=0; ii<width; ii++)
	{
	if(abs(StereoDiff[ii+jj*width])>maxErrorCutoff || abs(GroundTruth[ii+jj*width]) < 1.0)
	{
	mask[ii+jj*width] = true;
	code[ii+jj*width] = 0.0;
	}
	else
	{
	mask[ii+jj*width] = false;
	code[ii+jj*width] = ((StereoDiff[ii+jj*width] - minErr)/(maxErr-minErr)-0.5)*2.0;
	}
	}
	}
	ConvertToHeatMap(code, colorCode, width, height, mask);
	sprintf(Fname, "%s/Results/ColorCoded/Stereo_%05d.xyzc", PATH, frameID); fp = fopen(Fname, "w+");
	if(fp==NULL)
	return false;
	for(jj=0; jj<height; jj++)
	{
	for(ii=0; ii<width; ii++)
	{
	if(ii>DROI[0].x && ii < DROI[1].x && jj>DROI[0].y && jj<DROI[1].y)
	{
	if(abs(GroundTruth[ii+jj*width]) < 1.0)
	continue;
	else
	{
	z = (float)GroundTruth[ii+jj*width];
	x = (float)(z*(DInfo.iK[0]*ii + DInfo.iK[1]*jj + DInfo.iK[2]));
	y = (float)(z*(DInfo.iK[4]*jj+ DInfo.iK[5]));
	fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", x, y ,z, colorCode[3*ii+3*jj*width], colorCode[3*ii+1+3*jj*width], colorCode[3*ii+2+3*jj*width]);
	}
	}
	}
	}
	fclose(fp);*/

	//SVSR
	for (jj = 0; jj < SuperRes*height; jj++)
	{
		for (ii = 0; ii < SuperRes*width; ii++)
		{
			if (abs(SVSRDiff[ii + jj*SuperRes*width]) > abs(GroundTruth[ii + jj*SuperRes*width])*maxErrorCutoff || abs(GroundTruth[ii + jj*SuperRes*width]) < 1.0)
			{
				mask[ii + jj*SuperRes*width] = true;
				code[ii + jj*SuperRes*width] = 0.0;
			}
			else
			{
				mask[ii + jj*SuperRes*width] = false;
				code[ii + jj*SuperRes*width] = ((SVSRDiff[ii + jj*SuperRes*width] - minErr) / (maxErr - minErr) - 0.5)*2.0;
			}
		}
	}

	ConvertToHeatMap(code, colorCode, SuperRes*width, SuperRes*height, mask);
	/*sprintf(Fname, "%s/Results/ColorCoded/SVSR_%05d.xyzc", PATH, frameID); fp = fopen(Fname, "w+");
	if(fp==NULL)
	return false;
	for(jj=0; jj<SuperRes*height; jj++)
	{
	for(ii=0; ii<SuperRes*width; ii++)
	{
	if(ii>DROI[0].x*SuperRes && ii < DROI[1].x*SuperRes && jj>DROI[0].y*SuperRes && jj<DROI[1].y*SuperRes)
	{
	if(abs(GroundTruth[ii+jj*SuperRes*width]) < 1.0)
	continue;
	else
	{
	z = (float)GroundTruth[ii+jj*SuperRes*width];
	x = (float)(z*(DInfo.iK[0]*ii/SuperRes + DInfo.iK[1]*jj/SuperRes + DInfo.iK[2]));
	y = (float)(z*(DInfo.iK[4]*jj/SuperRes+ DInfo.iK[5]));
	fprintf(fp, "%.3f %.3f %.3f %d %d %d\n", x, y ,z, colorCode[3*ii+3*jj*SuperRes*width], colorCode[3*ii+1+3*jj*SuperRes*width], colorCode[3*ii+2+3*jj*SuperRes*width]);
	}
	}
	}
	}
	fclose(fp);*/

	delete[]GroundTruth;
	delete[]ProCam;
	delete[]Stereo;
	delete[]SVSR;
	delete[]ProCamDiff;
	delete[]StereoDiff;
	delete[]SVSRDiff;
	delete[]code;
	delete[]colorCode;
	delete[]mask;
	delete[]Pcorners;

	return 0;
}
int ConvertDepthTo3D(char *PATH, char *PATH2, int frameID, int nCams, int deviceID, int width, int height)
{
	int ii, jj;
	char Fname[200];
	double x, y, z;

	//Load campro info
	int nPros = 2;
	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	float *Depth = new float[4 * width*height];
	sprintf(Fname, "%s_%05d.ijz", PATH2, frameID);
	if (!ReadGridBinary(Fname, Depth, width * 2, height * 2))
	{
		cout << "Cannot load " << Fname << endl;
		return 4;
	}


	sprintf(Fname, "%s_%05d.xyz", PATH2, frameID); FILE *fp = fopen(Fname, "w+");
	if (fp == NULL)
		return false;
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			if (abs(Depth[ii + jj*width]) < 1.0)
				continue;
			else
			{
				z = (float)Depth[ii + jj*width];
				x = (float)(z*(DInfo.iK[9 * deviceID + 0] * ii + DInfo.iK[9 * deviceID + 1] * jj + DInfo.iK[9 * deviceID + 2]));
				y = (float)(z*(DInfo.iK[9 * deviceID + 4] * jj + DInfo.iK[9 * deviceID + 5]));
				fprintf(fp, "%.3f %.3f %.3f \n", x, y, z);
			}
		}
	}
	fclose(fp);

	return 0;
}
int Convert3DPtoC(char *PATH, char *PATH2, int frameID, int nCams, int deviceID, int width, int height)
{
	int ii;
	char Fname[200];

	//Load campro info
	int nPros = 2;
	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	CPoint3 P, C;
	double u, v;
	float *Depth = new float[width*height];
	for (ii = 0; ii < width*height; ii++)
		Depth[ii] = 0.0;

	sprintf(Fname, "%s/%05d_C1.xyz", PATH2, frameID); FILE *fp = fopen(Fname, "r");

	if (fp == NULL)
		return false;
	sprintf(Fname, "%s/_%05d_C1.xyz", PATH2, frameID);  FILE *fp2 = fopen(Fname, "w+");
	while (fscanf(fp, "%lf %lf %lf ", &P.x, &P.y, &P.z) != EOF)
	{
		C.x = DInfo.RT1x[(deviceID - 1) * 12 + 0] * P.x + DInfo.RT1x[(deviceID - 1) * 12 + 1] * P.y + DInfo.RT1x[(deviceID - 1) * 12 + 2] * P.z + DInfo.RT1x[(deviceID - 1) * 12 + 3];
		C.y = DInfo.RT1x[(deviceID - 1) * 12 + 4] * P.x + DInfo.RT1x[(deviceID - 1) * 12 + 5] * P.y + DInfo.RT1x[(deviceID - 1) * 12 + 6] * P.z + DInfo.RT1x[(deviceID - 1) * 12 + 7];
		C.z = DInfo.RT1x[(deviceID - 1) * 12 + 8] * P.x + DInfo.RT1x[(deviceID - 1) * 12 + 9] * P.y + DInfo.RT1x[(deviceID - 1) * 12 + 10] * P.z + DInfo.RT1x[(deviceID - 1) * 12 + 11];
		fprintf(fp2, "%.3f %.3f %.3f\n", C.x, C.y, C.z);

		u = C.x / C.z*DInfo.K[deviceID * 9] + C.y / C.z*DInfo.K[deviceID * 9 + 1] + DInfo.K[deviceID * 9 + 2];
		v = C.y / C.z*DInfo.K[deviceID * 9 + 4] + DInfo.K[deviceID * 9 + 5];

		if (u<1 || u>width - 1 || v < 1 || v>height - 1)
			continue;

		Depth[(int)(u + 0.5) + (int)(v + 0.5)*width] = (float)C.z;
	}
	fclose(fp); fclose(fp2);

	sprintf(Fname, "%s/_%05d_C1.ijz", PATH2, frameID);
	WriteGridBinary(Fname, Depth, width, height);

	return 0;
}
void ProjectBinaryCheckRandom(char *PATH, int squareDistance, int squareSize)
{
	int ii, jj, kk, ll;
	char Fname[200]; sprintf(Fname, "%s/Pattern2.png", PATH);
	IplImage *Random = cvLoadImage(Fname, 0);
	int width = Random->width, height = Random->height;
	cvFlip(Random);

	double x_margin = squareDistance / 2, y_margin = squareDistance / 2;
	int board_width = (int)((1.0*width - 2.0*x_margin) / squareDistance + 0.5);
	int board_height = (int)((1.0*height - 2.0*y_margin) / squareDistance + 0.5);
	CPoint2 *corners = new CPoint2[board_width * board_height];

	int SquareType;
	sprintf(Fname, "%s/ProCorners.txt", PATH);
	FILE *fp = fopen(Fname, "w+");
	fprintf(fp, "%d %d %d\n", board_width, board_height, squareDistance);
	for (jj = 0; jj < board_height; jj++)
	{
		for (ii = 0; ii < board_width; ii++)
		{
			corners[ii + jj*board_width].x = x_margin + 1.0*(ii*squareDistance) + 0.5;
			corners[ii + jj*board_width].y = y_margin + 1.0*(jj*squareDistance) + 0.5;

			if ((jj % 2 == 0) ^ (ii % 2 == 0)) //XOR operator
				SquareType = 0;
			else
				SquareType = 1;

			fprintf(fp, "%d %lf %lf \n", SquareType, corners[ii + jj*board_width].x, 1.0*height - 1.0 - corners[ii + jj*board_width].y);
		}
	}
	fclose(fp);

	CPoint center;
	int h_color_flag = 255, v_color_flag = 255;
	for (jj = 0; jj < board_height; jj++)
	{
		for (ii = 0; ii < board_width; ii++)
		{
			center.x = (int)corners[ii + jj*board_width].x;
			center.y = (int)corners[ii + jj*board_width].y;

			for (kk = -squareSize + 2; kk < squareSize; kk++)
			{
				h_color_flag = v_color_flag;

				for (ll = -squareSize + 2; ll < squareSize; ll++)
				{
					if ((kk - 0.5)*(kk - 0.5) + (ll - 0.5)*(ll - 0.5) > (squareSize - 1)*(squareSize - 1))
						continue;
					Random->imageData[(center.x + ll) + (center.y + kk)*width] = 0;
					if (ll == 1)
						h_color_flag = 255 - h_color_flag;//reverse the color
					Random->imageData[(center.x + ll) + (center.y + kk)*width] = h_color_flag;
				}
				if (kk == 0)
					v_color_flag = 255 - v_color_flag;//reverse the color
			}
		}
	}

	sprintf(Fname, "%s/ProjectorPattern.png", PATH);
	cvSaveImage(Fname, Random);
	//cvShowImage("X", Random);
	//cvMoveWindow("X", 1280*0, 0);
	cvWaitKey(-1);

	return;
}
void ProjectColorCheckRandom(char *PATH, int squareSize)
{
	int ii, jj, kk, ll;
	char Fname[200]; sprintf(Fname, "%s/CPattern.png", PATH);
	IplImage *Random = cvLoadImage(Fname, 1);
	int width = Random->width, height = Random->height;

	double *SImg = new double[3 * width*height];
	for (ii = 0; ii < 3 * width*height; ii++)
		SImg[ii] = (double)(int)(unsigned char)(Random->imageData[ii]);

	//Add random noise
	Gaussian_smooth(Random->imageData, SImg, height, width, 255.0, 0.7);

	double x_margin = squareSize / 2, y_margin = squareSize / 2;
	int board_width = (int)((1.0*width - 2.0*x_margin) / squareSize + 0.5);
	int board_height = (int)((1.0*height - 2.0*y_margin) / squareSize + 0.5);
	CPoint2 *corners = new CPoint2[board_width * board_height];

	int SquareType;
	sprintf(Fname, "%s/ProCorners.txt", PATH);
	FILE *fp = fopen(Fname, "w+");
	fprintf(fp, "%d %d %d\n", board_width, board_height, squareSize);
	for (jj = 0; jj < board_height; jj++)
	{
		for (ii = 0; ii < board_width; ii++)
		{
			corners[ii + jj*board_width].x = x_margin + 1.0*(ii*squareSize) + 0.5;
			corners[ii + jj*board_width].y = y_margin + 1.0*(jj*squareSize) + 0.5;

			if ((jj % 2 == 0) ^ (ii % 2 == 0)) //XOR operator
				SquareType = 0;
			else
				SquareType = 1;

			fprintf(fp, "%d %lf %lf \n", SquareType, corners[ii + jj*board_width].x, 1.0*height - 1.0 - corners[ii + jj*board_width].y);
		}
	}
	fclose(fp);

	CPoint center;
	int h_color_flag = 255, v_color_flag = 255;
	for (jj = 0; jj < board_height; jj++)
	{
		for (ii = 0; ii < board_width; ii++)
		{
			center.x = (int)corners[ii + jj*board_width].x;
			center.y = (int)corners[ii + jj*board_width].y;

			for (kk = -5 + 2; kk < 5; kk++)
			{
				h_color_flag = v_color_flag;

				for (ll = -5 + 2; ll < 5; ll++)
				{
					if ((kk - 0.5)*(kk - 0.5) + (ll - 0.5)*(ll - 0.5) > 16.0)
						continue;
					Random->imageData[3 * (center.x + ll) + 3 * (center.y + kk)*width] = 0;
					Random->imageData[3 * (center.x + ll) + 1 + 3 * (center.y + kk)*width] = 0;
					Random->imageData[3 * (center.x + ll) + 2 + 3 * (center.y + kk)*width] = 0;
					if (ll == 1)
						h_color_flag = 255 - h_color_flag;//reverse the color
					Random->imageData[3 * (center.x + ll) + 3 * (center.y + kk)*width] = h_color_flag;
					Random->imageData[3 * (center.x + ll) + 1 + 3 * (center.y + kk)*width] = h_color_flag;
					Random->imageData[3 * (center.x + ll) + 2 + 3 * (center.y + kk)*width] = h_color_flag;
				}
				if (kk == 0)
					v_color_flag = 255 - v_color_flag;//reverse the color
			}
		}
	}

	sprintf(Fname, "%s/CProjectorPattern.png", PATH);
	cvSaveImage(Fname, Random);
	//cvShowImage("X", Csquare);
	//cvMoveWindow("X", 1280*0, 0);
	//cvWaitKey(-1);

	return;
}
int DetectCorners(char *path, char *path2)
{
	int npts = 10000, hsubset = 8, hsubset2 = 12, searchArea = 1, InterpAlgo = 5;
	double ZNCCcoarseThresh = 0.6, ZNNCThresh = .9;
	vector<double> PatternAngles; PatternAngles.push_back(0.0), PatternAngles.push_back(15.0), PatternAngles.push_back(-15.0), PatternAngles.push_back(25.0), PatternAngles.push_back(-25.0), PatternAngles.push_back(45.0);

	Mat cvImg = imread(path, 0);
	if (cvImg.data == NULL)
	{
		cout << "Cannot load " << path << endl;
		return 1;
	}
	int width = cvImg.cols, height = cvImg.rows, length = width*height;

	double *Img = new double[length];
	double *ParaImg = new double[length];

	for (int jj = 0; jj < height; jj++)
		for (int ii = 0; ii < width; ii++)
			Img[ii + jj*width] = (double)(int)cvImg.data[ii + (height - 1 - jj)*width];
	Generate_Para_Spline(Img, ParaImg, width, height, InterpAlgo);

	CPoint2 *Pts = new CPoint2[npts];
	int *type = new int[npts];

	RunCornersDetector(Pts, type, npts, Img, ParaImg, width, height, PatternAngles, hsubset, hsubset2, searchArea, ZNCCcoarseThresh, ZNNCThresh, InterpAlgo);

	char fname[200]; sprintf(fname, "%s", path2);
	FILE *fp = fopen(fname, "w+");
	if (fp == NULL)
	{
		cout << "Cannot write " << fname << endl;
		return 1;
	}
	for (int ii = 0; ii < npts; ii++)
		fprintf(fp, "%.6f %.6f %d\n", Pts[ii].x, Pts[ii].y, type[ii]);
	fclose(fp);

	delete[]Img;
	delete[]ParaImg;
	delete[]Pts;
	delete[]type;
	return 0;
}
int IllumTextSepDriver(char *PATH, char *TPATH, IlluminationFlowImages &IlluminationImages, int proID, LKParameters LKArg, int frameID, int mode, int TextureColor, CPoint *ROI = 0, bool Simulation = false)
{
	//mode 0: interleaving, mode 1: known texture in the beginging
	const int nthreads = 4;
	if (mode == 0)
		cout << "Run separation with interleaving sequence." << endl;
	else
		cout << "Run separation with known texture surface." << endl;

	char Fname[200];
	int ii, jj, kk, ll, mm, nn;
	int nchannels = IlluminationImages.nchannels, nCams = IlluminationImages.nCams, nPros = IlluminationImages.nPros, TemporalW = IlluminationImages.nframes;
	int width = IlluminationImages.width, height = IlluminationImages.height, pwidth = IlluminationImages.pwidth, pheight = IlluminationImages.pheight;
	int length = width*height, plength = pwidth*pheight;

	//1: Setup cameras & projector parameters and load images data
	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	//2: Load images and texture
	sprintf(Fname, "%s/ProjectorPattern%d.png", PATH, proID + 1);
	Mat view = imread(Fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 2;
	}
	for (kk = 0; kk < nchannels; kk++)
	{
		for (jj = 0; jj < pheight; jj++)
			for (ii = 0; ii < pwidth; ii++)
				IlluminationImages.PImg[ii + jj*pwidth + (proID*nchannels + kk)*plength] = (double)((int)(view.data[nchannels*ii + (pheight - 1 - jj)*nchannels*pwidth + kk]));

		Gaussian_smooth_Double(IlluminationImages.PImg + (proID*nchannels + kk)*plength, IlluminationImages.PImg + (proID*nchannels + kk)*plength, pheight, pwidth, 255.0, 0.707);
		Generate_Para_Spline(IlluminationImages.PImg + (proID*nchannels + kk)*plength, IlluminationImages.PPara + (proID*nchannels + kk)*plength, pwidth, pheight, LKArg.InterpAlgo);
	}

	double *SourceTexture = new double[length*nchannels];
	double *ParaSourceTexture = new double[length*nchannels];
	if (mode == 0)
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 1, frameID - 1);
	else
	{
		int id = frameID%IlluminationImages.iRate == 0 ? frameID - (IlluminationImages.iRate - 1) : frameID / IlluminationImages.iRate*IlluminationImages.iRate + 1;
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 1, id);
	}

	view = imread(Fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 2;
	}
	for (kk = 0; kk < nchannels; kk++)
	{
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				SourceTexture[ii + jj*width + kk*length] = (double)((int)(view.data[nchannels*ii + (height - 1 - jj)*nchannels*width + kk]));

		Gaussian_smooth_Double(SourceTexture + kk*length, SourceTexture + kk*length, height, width, 255.0, 0.707);
		Generate_Para_Spline(SourceTexture + kk*length, ParaSourceTexture + kk*length, width, height, LKArg.InterpAlgo);
	}

	for (kk = 0; kk < nCams; kk++)
	{
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, kk + 1, frameID);
		view = imread(Fname, nchannels == 1 ? 0 : 1);
		if (view.data == NULL)
		{
			cout << "Cannot load image " << Fname << endl;
			return 2;
		}
		cout << "Load Images" << Fname << endl;
		for (mm = 0; mm < nchannels; mm++)
		{
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					IlluminationImages.Img[ii + (height - 1 - jj)*width + mm*length] = (double)((int)(view.data[nchannels*ii + nchannels*jj*width + mm]));

			Gaussian_smooth_Double(IlluminationImages.Img + mm*length, IlluminationImages.Img + mm*length, height, width, 255.0, LKArg.Gsigma);
			Generate_Para_Spline(IlluminationImages.Img + mm*length, IlluminationImages.Para + mm*length, width, height, LKArg.InterpAlgo);
		}
	}

	//Create ROI
	bool *cROI = new bool[length];
	for (ii = 0; ii < length; ii++)
		cROI[ii] = false;
	if (Simulation)
	{
		double *BlurImage = new double[length];
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 1, frameID);
		view = imread(Fname, 0);
		if (view.data == NULL)
		{
			cout << "Cannot load image " << Fname << endl;
			return 2;
		}
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				BlurImage[ii + (height - 1 - jj)*width] = (double)((int)(view.data[nchannels*ii + jj*width]));
		Gaussian_smooth_Double(BlurImage, BlurImage, height, width, 255.0, 2.0); //severely smooth out the image to determine the boundary

		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii<width; ii++)
				if (BlurImage[ii + jj*width] > 5.0)
					cROI[ii + jj*width] = true;

		ROI[0].x = width, ROI[0].y = height, ROI[1].x = 0, ROI[1].y = 0;
		for (jj = 0; jj<height; jj++)
		{
			for (ii = 0; ii<width; ii++)
			{
				if (cROI[ii + jj*width])
				{
					if (ROI[0].x > ii)
						ROI[0].x = ii;
					if (ROI[0].y > jj)
						ROI[0].y = jj;
					if (ROI[1].x < ii)
						ROI[1].x = ii;
					if (ROI[1].y < jj)
						ROI[1].y = jj;
				}
			}
		}

		delete[]BlurImage;
	}
	else
		for (jj = ROI[0].y; jj < ROI[1].y; jj++)
			for (ii = ROI[0].x; ii < ROI[1].x; ii++)
				cROI[ii + jj*width] = true;

	//3: Load precompute campro matching parameters (and Texture matching from previous frame if non-interleaving mode is used)
	double start = omp_get_wtime();
	float *ILWarpingParas = new float[6 * length];
	for (ii = 0; ii < 6; ii++)
	{
		sprintf(Fname, "%s/Results/CamPro/C1P%dp%d_%05d.dat", PATH, proID + 1, ii, frameID);
		if (!ReadGridBinary(Fname, ILWarpingParas + ii*length, width, height))
		{
			cout << "Cannot load Campro warping for Projector " << proID + 1 << endl;
			delete[]ILWarpingParas;
			return 1;
		}
	}
	cout << "Loaded campro warpings in " << omp_get_wtime() - start << endl;

	if (frameID / IlluminationImages.iRate*IlluminationImages.iRate + 2 == frameID)
		mode = 0;

	start = omp_get_wtime();
	bool flag;
	int *PrecomSearchR = new int[length];
	for (ii = 0; ii < length; ii++)
		PrecomSearchR[ii] = 0;
	for (jj = 10; jj < height - 10; jj++)
	{
		for (ii = 10; ii<width - 10; ii++)
		{
			flag = false;
			if (abs(ILWarpingParas[ii + jj*width]) + abs(ILWarpingParas[ii + jj*width + length])>0.001)
				flag = true, PrecomSearchR[ii + jj*width] = 0;
			else
			{
				for (kk = 1; kk < 5 && !flag; kk++)
					for (mm = -kk; mm <= kk && !flag; mm++)
						for (nn = -kk; nn <= kk; nn++)
							if (abs(ILWarpingParas[(ii + nn) + (jj + mm)*width]) + abs(ILWarpingParas[(ii + nn) + (jj + mm)*width + length]) > 0.001)
							{
								PrecomSearchR[ii + jj*width] = LKArg.hsubset - 2; //search more at the boundary
								flag = true; break;
							}
			}
			if (!flag)
				PrecomSearchR[ii + jj*width] = 2;//search less for inside regions
		}
	}
	cout << "Created SearchR in " << omp_get_wtime() - start << endl;

	float *previousTWarpingParas = 0;
	if (mode == 1)
	{
		start = omp_get_wtime();
		previousTWarpingParas = new float[6 * length];
		for (ii = 0; ii < 6; ii++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1TSp%d.dat", PATH, frameID - 1, ii);
			if (!ReadGridBinary(Fname, previousTWarpingParas + ii*length, width, height))
			{
				cout << "Cannot load previous Texture: " << Fname << endl;
				delete[]previousTWarpingParas, delete[]ILWarpingParas;
				return 1;
			}
		}
		cout << "Loaded previous Texture warpings in " << omp_get_wtime() - start << endl;
	}

	//Prepare SSIG on texture and decomposed images
	float *SSIG = new float[length];
	double *GXGY = new double[2 * length], S[3], ssig;
	start = omp_get_wtime();
	sprintf(Fname, "%s/%05d_SSIG.dat", TPATH, frameID);
	if (!ReadGridBinary(Fname, SSIG, width, height))
	{
		for (jj = 0; jj < height; jj++)
		{
			for (ii = 0; ii < width; ii++)
			{
				Get_Value_Spline(ParaSourceTexture, width, height, 1.0*ii, 1.0*jj, S, 0, LKArg.InterpAlgo);
				GXGY[ii + jj*width] = S[1], GXGY[ii + jj*width + length] = S[2];
			}
		}
		for (jj = 10; jj < height - 10; jj++)
		{
			for (ii = 10; ii < width - 10; ii++)
			{
				ssig = 0.0;
				for (mm = -LKArg.hsubset; mm <= LKArg.hsubset; mm++)
					for (nn = -LKArg.hsubset; nn <= LKArg.hsubset; nn++)
						ssig += pow(GXGY[(ii + nn) + (jj + mm)*width], 2) + pow(GXGY[(ii + nn) + (jj + mm)*width + length], 2);
				SSIG[ii + jj*width] = ssig / (2 * LKArg.hsubset + 1) / (2 * LKArg.hsubset + 1);
			}
		}
		WriteGridBinary(Fname, SSIG, width, height);
		cout << "Created SSIG in " << omp_get_wtime() - start << endl;
	}
	else
		cout << "Loaded SSIG in " << omp_get_wtime() - start << endl;
	delete[]GXGY;

	//Seedtype
	int *SeedType = new int[length]; //0: No point, 1: Illum 1, 2: Illum 2, 3, Illum1+Illum2, 4: Illum1+text, 5: Illum2+text, 6: Illums+text
	start = omp_get_wtime();
	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
	if (!ReadGridBinary(Fname, SeedType, width, height))
	{
		cout << "Cannot load Separation type for frame " << frameID << ". Re-intialize it. " << endl;
		for (ii = 0; ii < length; ii++)
			SeedType[ii] = 0;
	}
	else
		cout << "Finish in loading Separation type in " << omp_get_wtime() - start << endl;

	float *TWarpingParas = new float[6 * length];
	for (kk = 0; kk < 6; kk++)
		for (ii = 0; ii < length; ii++)
			TWarpingParas[ii + kk*length] = 0.0;
	const float intentsityFalloff = 1.0f / 255.0f;

	float *PhotoAdj = new float[3 * length];
	for (ii = 0; ii < length; ii++)
		PhotoAdj[ii] = intentsityFalloff, PhotoAdj[ii + length] = 0.0, PhotoAdj[ii + 2 * length] = 0.0;

	//Check if precomputed warping parameters are available.
	start = omp_get_wtime();
	bool precomputedAvail = false;
	float *sILWarpingParas = new float[6 * length];
	float *sTWarpingParas = new float[6 * length];
	for (ll = 0; ll < 6; ll++)
	{
		sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, proID + 1, ll);
		precomputedAvail = ReadGridBinary(Fname, sILWarpingParas + ll*length, width, height);
		if (!precomputedAvail)
			break;
	}
	if (precomputedAvail)
	{
		cout << "Loaded precomputed Illumination warping for frame " << frameID << " in " << omp_get_wtime() - start << endl;
		for (jj = 0; jj < length; jj++)
			for (ll = 0; ll < 6; ll++)
				ILWarpingParas[jj + ll*length] = sILWarpingParas[jj + ll*length];

		start = omp_get_wtime();
		for (ll = 0; ll < 6; ll++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1TSp%d.dat", PATH, frameID, ll);
			precomputedAvail = ReadGridBinary(Fname, sTWarpingParas + ll*length, width, height);
			if (!precomputedAvail)
				break;
		}
	}
	else
		cout << "No precomputed Illumination warping loaded for frame " << frameID << endl;

	if (precomputedAvail)
	{
		cout << "Loaded precomputed Texture warping for frame " << frameID << " in " << omp_get_wtime() - start << endl;
		for (jj = 0; jj < length; jj++)
			for (ll = 0; ll < 6; ll++)
				TWarpingParas[jj + ll*length] = sTWarpingParas[jj + ll*length];

		for (ll = 0; ll < 2; ll++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1PA_%d.dat", PATH, frameID, ll);
			if (!ReadGridBinary(Fname, PhotoAdj + ll*length, width, height))
			{
				for (ii = 0; ii<length; ii++)
					PhotoAdj[ii] = intentsityFalloff, PhotoAdj[ii + length] = 0.0, PhotoAdj[ii + 2 * length] = 0.0;
				break;
			}
		}
	}
	else
		cout << "No precomputed Texture warping loaded for frame " << frameID << endl;
	delete[]sILWarpingParas, delete[]sTWarpingParas;

	//Finallize Sep Type, SearchR, and ROI
	for (jj = ROI[0].y; jj <= ROI[1].y; jj++)
	{
		for (ii = ROI[0].x; ii < ROI[1].x; ii++)
		{
			if (abs(ILWarpingParas[ii + jj*width]) + abs(ILWarpingParas[ii + jj*width + length]) > 0.01)
			{
				cROI[ii + jj*width] = false;
				if (SeedType[ii + jj*width] == 0)
					if (abs(TWarpingParas[ii + jj*width]) + abs(TWarpingParas[ii + jj*width + length]) > 0.01)
						SeedType[ii + jj*width] = 4;
					else
						SeedType[ii + jj*width] = 1;
			}
			else
				cROI[ii + jj*width] = true;
		}
	}

	start = omp_get_wtime();
	if (nthreads > 1)
	{
		omp_set_num_threads(nthreads);

		int ROIH = ROI[1].y - ROI[0].y, subROI = ROIH / nthreads;
		CPoint sROI[nthreads * 2];
		for (ii = 0; ii < nthreads; ii++)
		{
			sROI[2 * ii].x = ROI[0].x, sROI[2 * ii + 1].x = ROI[1].x;
			sROI[2 * ii].y = ROI[0].y + subROI*ii - 1, sROI[2 * ii + 1].y = ROI[0].y + subROI*(ii + 1) + 1;
		}
		sROI[nthreads * 2 - 1].y = ROI[1].y;

		bool *scROI = new bool[nthreads*length];
		for (kk = 0; kk < nthreads; kk++)
		{
			for (ii = 0; ii < length; ii++)
				scROI[ii + kk*length] = false;
			for (jj = sROI[2 * kk].y; jj < sROI[2 * kk + 1].y; jj++)
				for (ii = sROI[2 * kk].x; ii < sROI[2 * kk + 1].x; ii++)
					scROI[ii + jj*width + kk*length] = cROI[ii + jj*width];
		}

		for (jj = 0; jj < LKArg.npass; jj++)
		{
#pragma omp parallel 
			{
#pragma omp for nowait
				for (int ii = 0; ii < nthreads; ii++)
				{
					IllumTextureSeperation(frameID, proID, PATH, TPATH, IlluminationImages, SourceTexture, ParaSourceTexture, SSIG, DInfo, ILWarpingParas, TWarpingParas, previousTWarpingParas, PhotoAdj, SeedType, PrecomSearchR, LKArg, mode, scROI + ii*length, ii, Simulation);
				}
			}
		}
		delete[]scROI;
	}
	else
		IllumTextureSeperation(frameID, proID, PATH, TPATH, IlluminationImages, SourceTexture, ParaSourceTexture, SSIG, DInfo, ILWarpingParas, TWarpingParas, previousTWarpingParas, PhotoAdj, SeedType, PrecomSearchR, LKArg, mode, cROI, nthreads, Simulation);
	cout << "Total time: " << omp_get_wtime() - start << endl;

	//Clean warping paras and save 3D:
	double P1mat[24];
	CPoint2 Ppts[2], Cpts; CPoint3 WC;
	P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0,
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0,
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;
	P1mat[12 + 0] = DInfo.P[0], P1mat[12 + 1] = DInfo.P[1], P1mat[12 + 2] = DInfo.P[2], P1mat[12 + 3] = DInfo.P[3],
		P1mat[12 + 4] = DInfo.P[4], P1mat[12 + 5] = DInfo.P[5], P1mat[12 + 6] = DInfo.P[6], P1mat[12 + 7] = DInfo.P[7],
		P1mat[12 + 8] = DInfo.P[8], P1mat[12 + 9] = DInfo.P[9], P1mat[12 + 10] = DInfo.P[10], P1mat[12 + 11] = DInfo.P[11];

	sprintf(Fname, "%s/Results/Sep/%05d_C1P%d.xyz", PATH, frameID, proID + 1); FILE *fp = fopen(Fname, "w+");
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			if (abs(ILWarpingParas[ii + jj*width]) + abs(ILWarpingParas[ii + jj*width + length]) < 0.001)
				continue;
			else	if (abs(ILWarpingParas[ii + jj*width]) + abs(ILWarpingParas[ii + jj*width + length]) > 0.001)
			{
				Ppts[0].x = ILWarpingParas[ii + jj*width] + ii, Ppts[0].y = ILWarpingParas[ii + jj*width + length] + jj;
				if (CamProGeoVerify(1.0*ii, 1.0*jj, Ppts, WC, P1mat, DInfo, width, height, pwidth, pheight, proID) == 1)
					ILWarpingParas[ii + jj*width] = 0.0, ILWarpingParas[ii + jj*width + length] = 0.0, TWarpingParas[ii + jj*width] = 0.0, TWarpingParas[ii + jj*width + length] = 0.0, SeedType[ii + jj*width] = 0;
				else
					fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
			}
		}
	}
	fclose(fp);

	//Save warping paras
	for (ii = 0; ii < 6; ii++)
	{
		sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, proID + 1, ii);
		WriteGridBinary(Fname, ILWarpingParas + ii*length, width, height);
	}
	for (ii = 0; ii < 6; ii++)
	{
		sprintf(Fname, "%s/Results/Sep/%05d_C1TSp%d.dat", PATH, frameID, ii);
		WriteGridBinary(Fname, TWarpingParas + ii*length, width, height);
	}
	for (ii = 0; ii < 3; ii++)
	{
		sprintf(Fname, "%s/Results/Sep/%05d_C1PA_%d.dat", PATH, frameID, ii);
		WriteGridBinary(Fname, PhotoAdj + ii*length, width, height);
	}
	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID, ii);
	WriteGridBinary(Fname, SeedType, width, height);

	//Synthesize separated images
	if (TextureColor == 1)
	{
		if (mode == 0)
			sprintf(Fname, "%s/Image/C1_%05d.png", PATH, frameID - 1);
		else
			sprintf(Fname, "%s/Image/C1_%05d.png", PATH, frameID / IlluminationImages.iRate*IlluminationImages.iRate + 1);

		double *SourceTexture = new double[length * 3];
		double *ParaSourceTexture = new double[length * 3];

		view = imread(Fname, 1);
		if (view.data == NULL)
		{
			cout << "Cannot load: " << Fname << endl;
			return 2;
		}
		for (kk = 0; kk < 3; kk++)
		{
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					SourceTexture[ii + jj*width + kk*length] = (double)((int)(view.data[3 * ii + (height - 1 - jj) * 3 * width + kk]));

			Gaussian_smooth_Double(SourceTexture + kk*length, SourceTexture + kk*length, height, width, 255.0, 0.707);
			Generate_Para_Spline(SourceTexture + kk*length, ParaSourceTexture + kk*length, width, height, LKArg.InterpAlgo);
		}

		sprintf(Fname, "%s/Results/Sep", PATH);
		UpdateIllumTextureImages(Fname, true, frameID, mode, 1, proID, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, IlluminationImages.PPara + proID*nchannels*plength, ILWarpingParas, ParaSourceTexture, TWarpingParas, NULL, true);
		delete[]SourceTexture, delete[]ParaSourceTexture;
	}
	else
	{
		sprintf(Fname, "%s/Results/Sep", PATH);
		UpdateIllumTextureImages(Fname, true, frameID, mode, 1, proID, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, IlluminationImages.PPara + proID*nchannels*plength, ILWarpingParas, ParaSourceTexture, TWarpingParas, NULL, false);
	}

	delete[]PrecomSearchR, delete[]cROI, delete[]SSIG;
	delete[]SourceTexture, delete[]ParaSourceTexture;
	delete[]ILWarpingParas, delete[]TWarpingParas, delete[]PhotoAdj, delete[]previousTWarpingParas;

	return 0;
}
int IllumSepDriver(char *PATH, char *TPATH, IlluminationFlowImages &IlluminationImages, LKParameters LKArg, int frameID, int mode, CPoint *ROI = 0, bool Simulation = false)
{
	cout << "Run illuminations separation." << endl;
	const int nthreads = 4;

	char Fname[200];
	int ii, jj, kk, ll, mm, nn;
	int nchannels = IlluminationImages.nchannels, nCams = IlluminationImages.nCams, nPros = IlluminationImages.nPros, TemporalW = IlluminationImages.nframes;
	int width = IlluminationImages.width, height = IlluminationImages.height, pwidth = IlluminationImages.pwidth, pheight = IlluminationImages.pheight;
	int length = width*height, plength = pwidth*pheight;

	//1: Setup cameras & projector parameters and load images data
	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	//2: Load images and texture
	Mat view;
	for (int proID = 0; proID < nPros; proID++)
	{
		sprintf(Fname, "%s/ProjectorPattern%d.png", PATH, proID + 1);
		view = imread(Fname, nchannels == 1 ? 0 : 1);
		if (view.data == NULL)
		{
			cout << "Cannot load: " << Fname << endl;
			return 2;
		}
		for (kk = 0; kk < nchannels; kk++)
		{
			for (jj = 0; jj < pheight; jj++)
				for (ii = 0; ii < pwidth; ii++)
					IlluminationImages.PImg[ii + jj*pwidth + kk*plength + proID*nchannels*plength] = (double)((int)(view.data[nchannels*ii + (pheight - 1 - jj)*nchannels*pwidth + kk]));

			Gaussian_smooth_Double(IlluminationImages.PImg + kk*plength + proID*nchannels*plength, IlluminationImages.PImg + kk*plength + proID*nchannels*plength, pheight, pwidth, 255.0, 0.707);
			Generate_Para_Spline(IlluminationImages.PImg + kk*plength + proID*nchannels*plength, IlluminationImages.PPara + kk*plength + proID*nchannels*plength, pwidth, pheight, LKArg.InterpAlgo);
		}
	}

	for (kk = 0; kk < nCams; kk++)
	{
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, kk + 1, frameID);
		view = imread(Fname, nchannels == 1 ? 0 : 1);
		if (view.data == NULL)
		{
			cout << "Cannot load image " << Fname << endl;
			return 2;
		}
		cout << "Load Images " << Fname << endl;
		for (mm = 0; mm < nchannels; mm++)
		{
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					IlluminationImages.Img[ii + (height - 1 - jj)*width + mm*length] = (double)((int)(view.data[nchannels*ii + nchannels*jj*width + mm]));

			Gaussian_smooth_Double(IlluminationImages.Img + mm*length, IlluminationImages.Img + mm*length, height, width, 255.0, LKArg.Gsigma);
			Generate_Para_Spline(IlluminationImages.Img + mm*length, IlluminationImages.Para + mm*length, width, height, LKArg.InterpAlgo);
		}
	}

	//3. Create ROI & SSIG
	bool *cROI = new bool[length];
	bool *bkcROI = new bool[length];
	for (ii = 0; ii < length; ii++)
		cROI[ii] = false;
	if (Simulation)
	{
		double *BlurImage = new double[length];
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 1, frameID);
		view = imread(Fname, 0);
		if (view.data == NULL)
		{
			cout << "Cannot load image " << Fname << endl;
			return 2;
		}
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				BlurImage[ii + (height - 1 - jj)*width] = (double)((int)(view.data[nchannels*ii + jj*width]));
		Gaussian_smooth_Double(BlurImage, BlurImage, height, width, 255.0, 2.0); //severely smooth out the image to determine the boundary

		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii<width; ii++)
				if (BlurImage[ii + jj*width] > 5.0)
					cROI[ii + jj*width] = true;

		ROI[0].x = width, ROI[0].y = height, ROI[1].x = 0, ROI[1].y = 0;
		for (jj = 0; jj<height; jj++)
		{
			for (ii = 0; ii<width; ii++)
			{
				if (cROI[ii + jj*width])
				{
					if (ROI[0].x > ii)
						ROI[0].x = ii;
					if (ROI[0].y > jj)
						ROI[0].y = jj;
					if (ROI[1].x < ii)
						ROI[1].x = ii;
					if (ROI[1].y < jj)
						ROI[1].y = jj;
				}
			}
		}

		delete[]BlurImage;
	}
	else
		for (jj = ROI[0].y; jj < ROI[1].y; jj++)
			for (ii = ROI[0].x; ii < ROI[1].x; ii++)
				cROI[ii + jj*width] = true;
	for (ii = 0; ii < length; ii++)
		bkcROI[ii] = cROI[ii];

	// Prepare SSIG on texture 
	float *SSIG = new float[length];
	double *GXGY = new double[2 * length], S[3], ssig;
	double start = omp_get_wtime();
	sprintf(Fname, "%s/Results/Sep/%05d_SSIG.dat", PATH, frameID);
	if (!ReadGridBinary(Fname, SSIG, width, height))
	{
		for (jj = 0; jj < height; jj++)
		{
			for (ii = 0; ii < width; ii++)
			{
				Get_Value_Spline(IlluminationImages.Para, width, height, 1.0*ii, 1.0*jj, S, 0, LKArg.InterpAlgo);
				GXGY[ii + jj*width] = S[1], GXGY[ii + jj*width + length] = S[2];
			}
		}
		for (jj = 10; jj < height - 10; jj++)
		{
			for (ii = 10; ii < width - 10; ii++)
			{
				ssig = 0.0;
				for (mm = -LKArg.hsubset; mm <= LKArg.hsubset; mm++)
				{
					for (nn = -LKArg.hsubset; nn <= LKArg.hsubset; nn++)
					{
						ssig += pow(GXGY[(ii + nn) + (jj + mm)*width], 2) + pow(GXGY[(ii + nn) + (jj + mm)*width + length], 2);
					}
				}
				SSIG[ii + jj*width] = ssig / (2 * LKArg.hsubset + 1) / (2 * LKArg.hsubset + 1);
			}
		}
		WriteGridBinary(Fname, SSIG, width, height);
		cout << "Created SSIG in " << omp_get_wtime() - start << endl;
	}
	else
		cout << "Loaded SSIG in " << omp_get_wtime() - start << endl;

	//Create Sep Type
	int *SeedType = new int[length]; //0: No touched point, 1: Illum 1, 2: Illum 2, 3, Illum1+Illum2, 4: Illum1+text, 5: Illum2+text, 6: Illums+text
	start = omp_get_wtime();
	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
	if (!ReadGridBinary(Fname, SeedType, width, height))
	{
		cout << "Cannot load Separation type for frame " << frameID << ". Re-intialize it. " << endl;
		for (ii = 0; ii < length; ii++)
			SeedType[ii] = 0;
	}
	else
		cout << "Finish in loading Separation type in " << omp_get_wtime() - start << endl;

	//4: Create working memory
	const float intentsityFalloff = 1.0f / 255.0f;
	float *ILWarpingParas = new float[6 * length*nPros];
	float *PhotoAdj = new float[2 * length];
	for (ii = 0; ii < length; ii++)
		PhotoAdj[ii] = 0.5, PhotoAdj[ii + length] = 0.5;

	//Load precompute campro matching parameters (and Texture matching from previous frame if non-interleaving mode is used)
	start = omp_get_wtime();
	for (jj = 0; jj < nPros; jj++)
	{
		for (ii = 0; ii < 6; ii++)
		{
			sprintf(Fname, "%s/Results/CamPro/C%dP%dp%d_%05d.dat", PATH, 1, jj + 1, ii, frameID);
			if (!ReadGridBinary(Fname, ILWarpingParas + (ii + 6 * jj)*length, width, height))
			{
				cout << "Cannot load Campro warping for Projector " << jj + 1 << endl;
				delete[]ILWarpingParas;
				return 1;
			}
		}
	}
	cout << "Loaded campro warpings in " << omp_get_wtime() - start << endl;

	//Estimate seedtype for reoptim when camera pixel has correspondences in both projectors
	int *reoptim = new int[length];
	for (jj = 0; jj<height; jj++)
	{
		for (ii = 0; ii<width; ii++)
		{
			reoptim[ii + jj*width] = 0;
			if (abs(ILWarpingParas[ii + jj*width]) + abs(ILWarpingParas[ii + jj*width + length])>0.001 && abs(ILWarpingParas[ii + jj*width + 6 * length]) + abs(ILWarpingParas[ii + jj*width + 7 * length])>0.001)
				reoptim[ii + jj*width] = 1;
		}
	}

	//Estimate seedtype for reoptim when 3D projects to both projectors
	start = omp_get_wtime();
	double u, v, denum, P1mat[12 * 2];
	P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0,
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0,
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;
	P1mat[12 + 0] = DInfo.P[0], P1mat[12 + 1] = DInfo.P[1], P1mat[12 + 2] = DInfo.P[2], P1mat[12 + 3] = DInfo.P[3],
		P1mat[12 + 4] = DInfo.P[4], P1mat[12 + 5] = DInfo.P[5], P1mat[12 + 6] = DInfo.P[6], P1mat[12 + 7] = DInfo.P[7],
		P1mat[12 + 8] = DInfo.P[8], P1mat[12 + 9] = DInfo.P[9], P1mat[12 + 10] = DInfo.P[10], P1mat[12 + 11] = DInfo.P[11];

	CPoint2 Cpts, Ppts; CPoint3 WC;
	for (jj = 10; jj < height - 10; jj++)
	{
		for (ii = 10; ii < width - 10; ii++)
		{
			for (int proID = 0; proID < nPros; proID++)
			{
				if (abs(ILWarpingParas[ii + jj*width + 6 * proID*length]) + abs(ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length]) > 0.001)
				{
					Cpts.x = 1.0*ii, Cpts.y = 1.0*jj;
					Ppts.x = ILWarpingParas[ii + jj*width + 6 * proID*length] + ii, Ppts.y = ILWarpingParas[ii + jj*width + (6 * proID + 1)*length] + jj;

					Undo_distortion(Ppts, DInfo.K + 9 * proID, DInfo.distortion + 13 * proID);
					Undo_distortion(Cpts, DInfo.K + 9 * nPros, DInfo.distortion + 13 * nPros);
					Stereo_Triangulation2(&Ppts, &Cpts, P1mat + 12 * proID, DInfo.P + 12 * (nPros - 1), &WC);

					//Project to other projector image 
					if (proID == 0)
					{
						denum = P1mat[8 + 12] * WC.x + P1mat[9 + 12] * WC.y + P1mat[10 + 12] * WC.z + P1mat[11 + 12];
						u = (P1mat[0 + 12] * WC.x + P1mat[1 + 12] * WC.y + P1mat[2 + 12] * WC.z + P1mat[3 + 12]) / denum;
						v = (P1mat[4 + 12] * WC.x + P1mat[5 + 12] * WC.y + P1mat[6 + 12] * WC.z + P1mat[7 + 12]) / denum;
					}
					else if (proID == 1)
					{
						denum = P1mat[8] * WC.x + P1mat[9] * WC.y + P1mat[10] * WC.z + P1mat[11];
						u = (P1mat[0] * WC.x + P1mat[1] * WC.y + P1mat[2] * WC.z + P1mat[3]) / denum;
						v = (P1mat[4] * WC.x + P1mat[5] * WC.y + P1mat[6] * WC.z + P1mat[7]) / denum;
					}

					if (u > 40 && u < pwidth - 40 && v>40 && v < pheight - 40)
						reoptim[ii + jj*width] = 1;
				}
			}
		}
	}
	cout << "Estimate ReOptimPoints in " << omp_get_wtime() - start << endl;

	//Check if precomputed warping parameters are available.
	bool precomputedAvail = false, breakflag = false;
	float *sILWarpingParas = new float[6 * length*nPros];
	start = omp_get_wtime();
	for (kk = 0; kk < nPros && !breakflag; kk++)
	{
		for (ll = 0; ll < 6; ll++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, kk + 1, ll);
			precomputedAvail = ReadGridBinary(Fname, sILWarpingParas + (ll + 6 * kk)*length, width, height, false);
			if (!precomputedAvail)
			{
				breakflag = true;
				break;
			}
		}
	}

	if (precomputedAvail) //If not available, have to run the reoptim processs
	{
		cout << "Loaded precomputed warping for frame " << frameID << " in " << omp_get_wtime() - start << endl;
		for (jj = 0; jj < length; jj++)
			for (ii = 0; ii < nPros; ii++)
				for (ll = 0; ll < 6; ll++)
					ILWarpingParas[jj + (ll + 6 * ii)*length] = sILWarpingParas[jj + (ll + 6 * ii)*length];

		for (ll = 0; ll < 2; ll++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1PA2_%d.dat", PATH, frameID, ll);
			if (!ReadGridBinary(Fname, PhotoAdj + ll*length, width, height, false))
			{
				for (ii = 0; ii < 2 * length; ii++)
					PhotoAdj[ii] = 0.5;
				break;
			}
		}
	}
	else
	{
		printf("No precomputed warping loaded for frame %d. Start reoptim campro correspondence. ", frameID);
		start = omp_get_wtime();
		if (nthreads > 1)
		{
			omp_set_num_threads(nthreads);
			//sprintf(Fname, "%s/Results/Sep", PATH);
			//UpdateIllumTextureImages(Fname, true, frameID, mode, nPros, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, IlluminationImages.PPara, ILWarpingParas, NULL, NULL, bkcROI);

			int ROIH = ROI[1].y - ROI[0].y, subROI = ROIH / nthreads;
			CPoint sROI[nthreads * 2];
			for (ii = 0; ii < nthreads; ii++)
			{
				sROI[2 * ii].x = ROI[0].x, sROI[2 * ii + 1].x = ROI[1].x;
				sROI[2 * ii].y = ROI[0].y + subROI*ii - 1, sROI[2 * ii + 1].y = ROI[0].y + subROI*(ii + 1) + 1;
			}
			sROI[nthreads * 2 - 1].y = ROI[1].y;

			bool *scROI = new bool[nthreads*length];
			for (kk = 0; kk < nthreads; kk++)
			{
				for (ii = 0; ii < length; ii++)
					scROI[ii + kk*length] = false;
				for (jj = sROI[2 * kk].y; jj < sROI[2 * kk + 1].y; jj++)
					for (ii = sROI[2 * kk].x; ii < sROI[2 * kk + 1].x; ii++)
						scROI[ii + jj*width + kk*length] = true;
			}
			for (jj = 0; jj < LKArg.npass; jj++)
			{
				printf("Iteration %d\n", jj + 1);
				if (jj == LKArg.npass - 1)
					printf("Cleaning process activated\n");
#pragma omp parallel sections
				{
#pragma omp section
					{
						int ii = 0;
						IllumsReoptim(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, reoptim, SeedType, SSIG, LKArg, mode, scROI + ii*length, ii, jj == LKArg.npass - 1);
					}
#pragma omp section
					{
						int ii = 1;
						IllumsReoptim(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, reoptim, SeedType, SSIG, LKArg, mode, scROI + ii*length, ii, jj == LKArg.npass - 1);
					}
#pragma omp section
					{
						int ii = 2;
						IllumsReoptim(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, reoptim, SeedType, SSIG, LKArg, mode, scROI + ii*length, ii, jj == LKArg.npass - 1);
					}
#pragma omp section
					{
						int ii = 3;
						IllumsReoptim(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, reoptim, SeedType, SSIG, LKArg, mode, scROI + ii*length, ii, jj == LKArg.npass - 1);
					}
				}
			}
			delete[]scROI;
		}
		else
		{
			IllumsReoptim(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, reoptim, SeedType, SSIG, LKArg, mode, cROI, nthreads, false);
			IllumsReoptim(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, reoptim, SeedType, SSIG, LKArg, mode, cROI, nthreads, true);
		}
		cout << "Total time: " << omp_get_wtime() - start << endl;

#pragma omp critical
		{
			for (int proID = 0; proID < nPros; proID++)
			{
				for (int ll = 0; ll < 6; ll++)
				{
					sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, proID + 1, ll);
					WriteGridBinary(Fname, ILWarpingParas + (6 * proID + ll)*length, width, height);
				}
			}
			for (ll = 0; ll < 2; ll++)
			{
				sprintf(Fname, "%s/Results/Sep/%05d_C1PA2_%d.dat", PATH, frameID, ll);
				WriteGridBinary(Fname, PhotoAdj + ll*length, width, height);
			}
			sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);	WriteGridBinary(Fname, SeedType, width, height);
			sprintf(Fname, "%s/Results/Sep", PATH);
			UpdateIllumTextureImages(Fname, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, IlluminationImages.PPara, ILWarpingParas, NULL, NULL, bkcROI);
		}
	}
	delete[]sILWarpingParas;

	//Update seetype
	int count = 0;
	start = omp_get_wtime();
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			count = 0; SeedType[ii + jj*width] = 0;
			for (int proID = 0; proID < nPros; proID++)
			{
				if (abs(ILWarpingParas[ii + jj*width + 6 * proID*length]) + abs(ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length]) > 0.001)
				{
					SeedType[ii + jj*width] = proID + 1; //Illumination only
					count++;
				}
			}
			if (count == nPros)
				SeedType[ii + jj*width] = 3;
		}
	}
	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
	WriteGridBinary(Fname, SeedType, width, height);
	cout << "Seedtype updated in " << omp_get_wtime() - start << endl;


	//Create SearchRange 
	bool flag;
	int *PrecomSearchR = new int[length];
	start = omp_get_wtime();
	for (ii = 0; ii < length; ii++)
		PrecomSearchR[ii] = 0;
	for (jj = 10; jj < height - 10; jj++)
	{
		for (ii = 10; ii < width - 10; ii++)
		{
			flag = false;
			for (int proID = 0; proID<nPros; proID++)
			{
				if (abs(ILWarpingParas[ii + jj*width + 6 * proID*length]) + abs(ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length])>0.001)
					flag = true, cROI[ii + jj*width] = false, PrecomSearchR[ii + jj*width] = 0;
				else
					for (kk = 1; kk < LKArg.hsubset - 2 && !flag; kk++)
						for (mm = -kk; mm <= kk && !flag; mm++)
							for (nn = -kk; nn <= kk; nn++)
								if (abs(ILWarpingParas[(ii + nn) + (jj + mm)*width + 6 * proID*length]) + abs(ILWarpingParas[(ii + nn) + (jj + mm)*width + (1 + 6 * proID)*length]) > 0.001)
								{
									PrecomSearchR[ii + jj*width] = LKArg.hsubset - 2; //search more at the boundary
									flag = true; break;
								}
			}
			if (!flag)
				PrecomSearchR[ii + jj*width] = 2;//search less for inside textured regions
		}
	}

	//Start decomposition
	start = omp_get_wtime();
	if (nthreads > 1)
	{
		omp_set_num_threads(nthreads);

		int ROIH = ROI[1].y - ROI[0].y, subROI = ROIH / nthreads;
		CPoint sROI[nthreads * 2];
		for (ii = 0; ii < nthreads; ii++)
		{
			sROI[2 * ii].x = ROI[0].x, sROI[2 * ii + 1].x = ROI[1].x;
			sROI[2 * ii].y = ROI[0].y + subROI*ii - 1, sROI[2 * ii + 1].y = ROI[0].y + subROI*(ii + 1) + 1;
		}
		sROI[nthreads * 2 - 1].y = ROI[1].y;

		bool *scROI = new bool[nthreads*length];
		for (kk = 0; kk < nthreads; kk++)
		{
			for (ii = 0; ii < length; ii++)
				scROI[ii + kk*length] = false;
			for (jj = sROI[2 * kk].y; jj < sROI[2 * kk + 1].y; jj++)
				for (ii = sROI[2 * kk].x; ii < sROI[2 * kk + 1].x; ii++)
					scROI[ii + jj*width + kk*length] = cROI[ii + jj*width];
		}

		for (jj = 0; jj < LKArg.npass; jj++)
		{
#pragma omp parallel sections
			{
#pragma omp section
				{
					int ii = 0;
					IllumSeperation(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, SeedType, SSIG, PrecomSearchR, LKArg, mode, scROI + ii*length, ii);
				}
#pragma omp section
				{
					int ii = 1;
					IllumSeperation(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, SeedType, SSIG, PrecomSearchR, LKArg, mode, scROI + ii*length, ii);
				}
#pragma omp section
				{
					int ii = 2;
					IllumSeperation(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, SeedType, SSIG, PrecomSearchR, LKArg, mode, scROI + ii*length, ii);
				}
#pragma omp section
				{
					int ii = 3;
					IllumSeperation(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, SeedType, SSIG, PrecomSearchR, LKArg, mode, scROI + ii*length, ii);
				}
			}
		}
		delete[]scROI;
	}
	else
		for (ii = 0; ii < LKArg.npass; ii++)
			IllumSeperation(frameID, PATH, TPATH, IlluminationImages, DInfo, ILWarpingParas, PhotoAdj, SeedType, SSIG, PrecomSearchR, LKArg, mode, cROI, nthreads);
	cout << "Total time: " << omp_get_wtime() - start << endl;

	//Create 3D + clean all possible bad points
	double u1, v1, u2, v2, reprojectionError;
	sprintf(Fname, "%s/Results/Sep/%05d_C1.xyz", PATH, frameID); FILE *fp = fopen(Fname, "w+");
	for (int proID = 0; proID < nPros; proID++)
	{
		if (proID == 0)
		{
			P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0,
				P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0,
				P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[10] = 0.0;
		}
		else
		{
			P1mat[0] = DInfo.P[12 * (proID - 1)], P1mat[1] = DInfo.P[12 * (proID - 1) + 1], P1mat[2] = DInfo.P[12 * (proID - 1) + 2], P1mat[3] = DInfo.P[12 * (proID - 1) + 3],
				P1mat[4] = DInfo.P[12 * (proID - 1) + 4], P1mat[5] = DInfo.P[12 * (proID - 1) + 5], P1mat[6] = DInfo.P[12 * (proID - 1) + 6], P1mat[7] = DInfo.P[12 * (proID - 1) + 7],
				P1mat[8] = DInfo.P[12 * (proID - 1) + 8], P1mat[9] = DInfo.P[12 * (proID - 1) + 9], P1mat[10] = DInfo.P[12 * (proID - 1) + 10], P1mat[11] = DInfo.P[12 * (proID - 1) + 11];
		}

		for (jj = 0; jj < height; jj++)
		{
			for (ii = 0; ii < width; ii++)
			{
				if (abs(ILWarpingParas[ii + jj*width + 6 * proID*length]) + abs(ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length]) < 0.001)
					continue;
				else
				{
					Cpts.x = 1.0*ii, Cpts.y = 1.0*jj;
					Ppts.x = ILWarpingParas[ii + jj*width + 6 * proID*length] + ii, Ppts.y = ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length] + jj;

					Undo_distortion(Cpts, DInfo.K + 9 * nPros, DInfo.distortion + 13 * nPros);
					Undo_distortion(Ppts, DInfo.K + 9 * proID, DInfo.distortion + 13 * proID);
					Stereo_Triangulation2(&Ppts, &Cpts, P1mat, DInfo.P + 12 * (nPros - 1), &WC);

					//Project to projector image 
					denum = P1mat[8] * WC.x + P1mat[9] * WC.y + P1mat[10] * WC.z + P1mat[11];
					u1 = (P1mat[0] * WC.x + P1mat[1] * WC.y + P1mat[2] * WC.z + P1mat[3]) / denum;
					v1 = (P1mat[4] * WC.x + P1mat[5] * WC.y + P1mat[6] * WC.z + P1mat[7]) / denum;

					//Project to Camera image 
					denum = DInfo.P[8 + 12 * (nPros - 1)] * WC.x + DInfo.P[9 + 12 * (nPros - 1)] * WC.y + DInfo.P[10 + 12 * (nPros - 1)] * WC.z + DInfo.P[11 + 12 * (nPros - 1)];
					u2 = (DInfo.P[0 + 12 * (nPros - 1)] * WC.x + DInfo.P[1 + 12 * (nPros - 1)] * WC.y + DInfo.P[2 + 12 * (nPros - 1)] * WC.z + DInfo.P[3 + 12 * (nPros - 1)]) / denum;
					v2 = (DInfo.P[4 + 12 * (nPros - 1)] * WC.x + DInfo.P[5 + 12 * (nPros - 1)] * WC.y + DInfo.P[6 + 12 * (nPros - 1)] * WC.z + DInfo.P[7 + 12 * (nPros - 1)]) / denum;

					reprojectionError = abs(u2 - Cpts.x) + abs(v2 - Cpts.y) + abs(u1 - Ppts.x) + abs(v1 - Ppts.y);

					if (reprojectionError / 4 > 1 || u1<0 || u1>pwidth - 1 || v1<0 || v1>pheight - 1 || u2 < 0 || u2 > width - 1 || v2<0 || v2 > height - 1)
						SeedType[ii + jj*width] = 0, ILWarpingParas[ii + jj*width + 6 * proID*length] = 0.0, ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length] = 0.0;
					else if (proID != 0 && abs(ILWarpingParas[ii + jj*width]) + abs(ILWarpingParas[ii + jj*width + length]) > 0.001 && abs(ILWarpingParas[ii + jj*width + 6 * length]) + abs(ILWarpingParas[ii + jj*width + 7 * length]) > 0.001)
					{
						SeedType[ii + jj*width] = 0;
						continue;
					}
					else
						fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
				}
			}
		}
	}
	fclose(fp);

	//Save warping paras
	for (jj = 0; jj < nPros; jj++)
	{
		for (ii = 0; ii < 6; ii++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, jj + 1, ii);
			WriteGridBinary(Fname, ILWarpingParas + (ii + 6 * jj)*length, width, height);
		}
	}
	for (ii = 0; ii < 2; ii++)
	{
		sprintf(Fname, "%s/Results/Sep/%05d_C1PA2_%d.dat", PATH, frameID, ii);
		WriteGridBinary(Fname, PhotoAdj + ii*length, width, height);
	}
	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
	WriteGridBinary(Fname, SeedType, width, height);

	//Synthesize separated images
	sprintf(Fname, "%s/Results/Sep", PATH);
	UpdateIllumTextureImages(Fname, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, IlluminationImages.PPara, ILWarpingParas, NULL, NULL, bkcROI);

	delete[]ILWarpingParas, delete[]PhotoAdj;
	delete[]PrecomSearchR, delete[]SeedType;
	delete[]cROI, delete[]bkcROI;
	delete[]SSIG, delete[]GXGY;

	return 0;
}
int TwoIllumTextSepDriver(char *PATH, char *TPATH, IlluminationFlowImages &IlluminationImages, LKParameters LKArg, int frameID, int mode, CPoint *ROI = 0, bool restart = true, bool color = false, bool Simulation = false)
{
	const int nthreads = 4;
	//mode 0: interleaving, mode 1: known texture in the beginging
	if (mode == 0)
		cout << "Run separation with interleaving sequence." << endl;
	else if (mode == 1)
		cout << "Run separation with known texture surface." << endl;
	else
		cout << "Run separation for 2L" << endl;

	char Fname[200];
	int ii, jj, kk, ll, mm, nn;
	int nchannels = IlluminationImages.nchannels, nCams = IlluminationImages.nCams, nPros = IlluminationImages.nPros, TemporalW = IlluminationImages.nframes;
	int width = IlluminationImages.width, height = IlluminationImages.height, pwidth = IlluminationImages.pwidth, pheight = IlluminationImages.pheight;
	int length = width*height, plength = pwidth*pheight;

	//1: Setup cameras & projector parameters and load images data
	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	//2: Projector pattern
	Mat view;
	for (int proID = 0; proID < 2; proID++)
	{
		sprintf(Fname, "%s/ProjectorPattern%d.png", PATH, proID + 1);
		view = imread(Fname, nchannels == 1 ? 0 : 1);
		if (view.data == NULL)
		{
			cout << "Cannot load: " << Fname << endl;
			return 2;
		}
		for (kk = 0; kk < nchannels; kk++)
		{
			for (jj = 0; jj < pheight; jj++)
				for (ii = 0; ii < pwidth; ii++)
					IlluminationImages.PImg[ii + jj*pwidth + (proID*nchannels + kk)*plength] = (double)((int)(view.data[nchannels*ii + (pheight - 1 - jj)*nchannels*pwidth + kk]));

			Gaussian_smooth_Double(IlluminationImages.PImg + (proID*nchannels + kk)*plength, IlluminationImages.PImg + (proID*nchannels + kk)*plength, pheight, pwidth, 255.0, LKArg.ProjectorGsigma);
			Generate_Para_Spline(IlluminationImages.PImg + (proID*nchannels + kk)*plength, IlluminationImages.PPara + (proID*nchannels + kk)*plength, pwidth, pheight, LKArg.InterpAlgo);
		}
	}

	//Texture template
	double *SourceTexture = new double[length*nchannels];
	double *ParaSourceTexture = new double[length*nchannels];
	if (mode < 2)
	{
		if (mode == 0)
			sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 1, frameID - 1);
		else if (mode == 1)
		{
			int id = frameID%IlluminationImages.iRate == 0 ? frameID - (IlluminationImages.iRate - 1) : frameID / IlluminationImages.iRate*IlluminationImages.iRate + 1;
			sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 1, id);
		}

		view = imread(Fname, nchannels == 1 ? 0 : 1);
		if (view.data == NULL)
		{
			cout << "Cannot load: " << Fname << endl;
			return 2;
		}
		for (kk = 0; kk < nchannels; kk++)
		{
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					SourceTexture[ii + jj*width + kk*length] = (double)((int)(view.data[nchannels*ii + (height - 1 - jj)*nchannels*width + kk]));

			Gaussian_smooth_Double(SourceTexture + kk*length, SourceTexture + kk*length, height, width, 255.0, 0.707);
			Generate_Para_Spline(SourceTexture + kk*length, ParaSourceTexture + kk*length, width, height, LKArg.InterpAlgo);
		}
	}

	//Image to be decomposed
	for (kk = 0; kk < nCams; kk++)
	{
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, kk + 1, frameID);
		view = imread(Fname, nchannels == 1 ? 0 : 1);
		if (view.data == NULL)
		{
			cout << "Cannot load image " << Fname << endl;
			return 2;
		}
		cout << "Load Images " << Fname << endl;
		for (mm = 0; mm < nchannels; mm++)
		{
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					IlluminationImages.Img[ii + (height - 1 - jj)*width + mm*length] = (double)((int)(view.data[nchannels*ii + nchannels*jj*width + mm]));

			Gaussian_smooth_Double(IlluminationImages.Img + mm*length, IlluminationImages.Img + mm*length, height, width, 255.0, LKArg.Gsigma);
			Generate_Para_Spline(IlluminationImages.Img + mm*length, IlluminationImages.Para + mm*length, width, height, LKArg.InterpAlgo);
		}
	}

	//3. Create ROI and SSIG
	bool *cROI = new bool[length];
	bool *bkcROI = new bool[length];
	for (ii = 0; ii < length; ii++)
		cROI[ii] = false;

	if (Simulation)
	{
		double *BlurImage = new double[length];
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, 1, frameID);
		view = imread(Fname, 0);
		if (view.data == NULL)
		{
			cout << "Cannot load image " << Fname << endl;
			return 2;
		}
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				BlurImage[ii + (height - 1 - jj)*width] = (double)((int)(view.data[nchannels*ii + jj*width]));
		Gaussian_smooth_Double(BlurImage, BlurImage, height, width, 255.0, 2.0); //severely smooth out the image to determine the boundary

		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii<width; ii++)
				if (BlurImage[ii + jj*width] > 5.0)
					cROI[ii + jj*width] = true;

		ROI[0].x = width, ROI[0].y = height, ROI[1].x = 0, ROI[1].y = 0;
		for (jj = 0; jj<height; jj++)
		{
			for (ii = 0; ii<width; ii++)
			{
				if (cROI[ii + jj*width])
				{
					if (ROI[0].x > ii)
						ROI[0].x = ii;
					if (ROI[0].y > jj)
						ROI[0].y = jj;
					if (ROI[1].x < ii)
						ROI[1].x = ii;
					if (ROI[1].y < jj)
						ROI[1].y = jj;
				}
			}
		}
		delete[]BlurImage;
	}
	else
		for (jj = ROI[0].y; jj < ROI[1].y; jj++)
			for (ii = ROI[0].x; ii < ROI[1].x; ii++)
				cROI[ii + jj*width] = true;
	for (ii = 0; ii < length; ii++)
		bkcROI[ii] = cROI[ii];

	//Prepare SSIG on texture: one for texture source, one for the current image 
	float *SSIG = new float[2 * length];
	double *GXGY = new double[2 * length], S[3], ssig;
	double start = omp_get_wtime();

	if (mode < 2)
	{
		if (mode == 0)//Texture source
			sprintf(Fname, "%s/Results/Sep/%05d_SSIG.dat", PATH, frameID - 1);
		else
		{
			int id = frameID%IlluminationImages.iRate == 0 ? frameID - (IlluminationImages.iRate - 1) : frameID / IlluminationImages.iRate*IlluminationImages.iRate + 1;
			sprintf(Fname, "%s/Results/Sep/%05d_SSIG.dat", PATH, id);
		}
		if (!ReadGridBinary(Fname, SSIG, width, height))
		{
			for (jj = 0; jj < height; jj++)
			{
				for (ii = 0; ii < width; ii++)
				{
					Get_Value_Spline(ParaSourceTexture, width, height, 1.0*ii, 1.0*jj, S, 0, LKArg.InterpAlgo);
					GXGY[ii + jj*width] = S[1], GXGY[ii + jj*width + length] = S[2];
				}
			}
			for (jj = LKArg.hsubset; jj < height - LKArg.hsubset; jj++)
			{
				for (ii = LKArg.hsubset; ii < width - LKArg.hsubset; ii++)
				{
					ssig = 0.0;
					for (mm = -LKArg.hsubset; mm <= LKArg.hsubset; mm++)
						for (nn = -LKArg.hsubset; nn <= LKArg.hsubset; nn++)
							ssig += pow(GXGY[(ii + nn) + (jj + mm)*width], 2) + pow(GXGY[(ii + nn) + (jj + mm)*width + length], 2);
					SSIG[ii + jj*width] = ssig / (2 * LKArg.hsubset + 1) / (2 * LKArg.hsubset + 1);
				}
			}
			WriteGridBinary(Fname, SSIG, width, height);
			cout << "Created source SSIG in " << omp_get_wtime() - start << endl;
		}
		else
			cout << "Loaded source SSIG in " << omp_get_wtime() - start << endl;
	}

	start = omp_get_wtime(); //current image
	sprintf(Fname, "%s/Results/Sep/%05d_SSIG.dat", PATH, frameID);
	if (!ReadGridBinary(Fname, SSIG + length, width, height))
	{
		for (jj = 0; jj < height; jj++)
		{
			for (ii = 0; ii < width; ii++)
			{
				Get_Value_Spline(IlluminationImages.Para, width, height, 1.0*ii, 1.0*jj, S, 0, LKArg.InterpAlgo);
				GXGY[ii + jj*width] = S[1], GXGY[ii + jj*width + length] = S[2];
			}
		}
		for (jj = LKArg.hsubset; jj < height - LKArg.hsubset; jj++)
		{
			for (ii = LKArg.hsubset; ii < width - LKArg.hsubset; ii++)
			{
				ssig = 0.0;
				for (mm = -LKArg.hsubset; mm <= LKArg.hsubset; mm++)
					for (nn = -LKArg.hsubset; nn <= LKArg.hsubset; nn++)
						ssig += pow(GXGY[(ii + nn) + (jj + mm)*width], 2) + pow(GXGY[(ii + nn) + (jj + mm)*width + length], 2);
				SSIG[ii + jj*width + length] = ssig / (2 * LKArg.hsubset + 1) / (2 * LKArg.hsubset + 1);
			}
		}
		WriteGridBinary(Fname, SSIG + length, width, height);
		cout << "Created current SSIG in " << omp_get_wtime() - start << endl;
	}
	else
		cout << "Loaded current SSIG in " << omp_get_wtime() - start << endl;

	//4. Create working memory
	const float intentsityFalloff = 1.0f / 255.0f;
	float *PhotoAdj = new float[6 * length];
	float *ILWarpingParas = new float[6 * length*nPros];
	float *TWarpingParas = new float[6 * length];
	for (ii = 0; ii < length; ii++)
	{
		for (kk = 0; kk < 6; kk++)
			ILWarpingParas[ii + kk*length] = 0.0, ILWarpingParas[ii + (kk + 6)*length] = 0.0, TWarpingParas[ii + kk*length] = 0.0, PhotoAdj[ii + kk*length] = 0.0;
	}

	//Load precompute campro matching parameters (and Texture matching from previous frame if non-interleaving mode is used)
	start = omp_get_wtime();
	for (jj = 0; jj < nPros; jj++)
	{
		for (ii = 0; ii < 6; ii++)
		{
			sprintf(Fname, "%s/Results/CamPro/C1P%dp%d_%05d.dat", PATH, jj + 1, ii, frameID);
			if (!ReadGridBinary(Fname, ILWarpingParas + (ii + 6 * jj)*length, width, height))
			{
				cout << "Cannot load Campro warping for Projector " << jj + 1 << endl;
				delete[]ILWarpingParas;
				return 1;
			}
		}
	}
	cout << "Loaded campro warpings in " << omp_get_wtime() - start << endl;

	float *previousTWarpingParas = 0;
	if (mode < 2)//Load previous frame Texture
	{
		if (frameID / IlluminationImages.iRate*IlluminationImages.iRate + 2 == frameID)
			mode = 0;
		if (mode == 1)
		{
			start = omp_get_wtime();
			previousTWarpingParas = new float[6 * length];
			for (ii = 0; ii < 6; ii++)
			{
				sprintf(Fname, "%s/Results/Sep/%05d_C1TSp%d.dat", PATH, frameID - 1, ii);
				if (!ReadGridBinary(Fname, previousTWarpingParas + ii*length, width, height))
				{
					cout << "Cannot load previous Texture warping paras" << endl;
					delete[]previousTWarpingParas, delete[]ILWarpingParas;
					return 1;
				}
			}
			cout << "Loaded previous Texture warpings in " << omp_get_wtime() - start << endl;
		}
	}

	//Create Sep Type
	start = omp_get_wtime();
	int *SeedType = new int[length]; //0: No touched point, 1: Illum 1, 2: Illum 2, 3, Illum1+Illum2, 4: Illum1+text, 5: Illum2+text, 6: Illums+text
	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
	if (!ReadGridBinary(Fname, SeedType, width, height))
	{
		cout << "Cannot load SeedType for frame " << frameID << ". Re-intialize it. " << endl;
		for (ii = 0; ii < length; ii++)
		{
			if (abs(ILWarpingParas[ii]) > 0.01)
				SeedType[ii] = 1;
			else if (abs(ILWarpingParas[ii + 6 * length]) > 0.01)
				SeedType[ii] = 2;
			else
				SeedType[ii] = 0;
		}
	}
	else
		cout << "Loaded SeedType in " << omp_get_wtime() - start << endl;

	//Create SearchRange for obtaining initial guess
	bool flag;
	int *PrecomSearchR = new int[length];
	start = omp_get_wtime();
	sprintf(Fname, "%s/Results/Sep/%05d_SearchR.dat", PATH, frameID);
	if (!ReadGridBinary(Fname, PrecomSearchR, width, height))
	{
		for (jj = 2 * LKArg.hsubset; jj < height - 2 * LKArg.hsubset; jj++)
		{
			for (ii = 2 * LKArg.hsubset; ii < width - 2 * LKArg.hsubset; ii++)
			{
				flag = false;
				for (ll = 0; ll<nPros; ll++)
				{
					if (abs(ILWarpingParas[ii + jj*width + 6 * ll*length]) + abs(ILWarpingParas[ii + jj*width + (1 + 6 * ll)*length])>0.001)
						flag = true, PrecomSearchR[ii + jj*width] = 0;
					else
					{
						for (kk = 1; kk < LKArg.hsubset - 2 && !flag; kk++)
							for (mm = -kk; mm <= kk && !flag; mm++)
								for (nn = -kk; nn <= kk; nn++)
									if (abs(ILWarpingParas[(ii + nn) + (jj + mm)*width + 6 * ll*length]) + abs(ILWarpingParas[(ii + nn) + (jj + mm)*width + (1 + 6 * ll)*length]) > 0.001)
									{
										PrecomSearchR[ii + jj*width] = kk + 2; //search more at the boundary within a radius of LKArg.hsubset-2 pixels
										flag = true; break;
									}
					}
				}
				if (!flag)
					PrecomSearchR[ii + jj*width] = 2;//search less for mixed regions
			}
		}
		WriteGridBinary(Fname, PrecomSearchR, width, height);
		cout << "Created SearchR in " << omp_get_wtime() - start << endl;
	}
	//else
	//cout << "Loaded SearchR in " << omp_get_wtime() - start << endl;

	//Check if precomputed warping parameters are available.
	bool precomputedAvail = false, breakflag = false;
	float *sILWarpingParas = new float[6 * length*nPros];
	float *sTWarpingParas = new float[6 * length];
	start = omp_get_wtime();
	for (kk = 0; kk < nPros && !breakflag; kk++)
	{
		for (ll = 0; ll < 6; ll++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, kk + 1, ll);
			precomputedAvail = ReadGridBinary(Fname, sILWarpingParas + (ll + 6 * kk)*length, width, height);
			if (!precomputedAvail)
			{
				breakflag = true;
				break;
			}
		}
	}
	if (precomputedAvail)
	{
		cout << "Loaded precomputed Illum for frame " << frameID << " in " << omp_get_wtime() - start << endl;
		for (jj = 0; jj < length; jj++)
		{
			for (ii = 0; ii < nPros; ii++)
				for (ll = 0; ll < 6; ll++)
					if (abs(ILWarpingParas[jj + (ll + 6 * ii)*length]) < 0.001)//there is case when you use small ROI for procam matching and then expand it after sepration finishes
						ILWarpingParas[jj + (ll + 6 * ii)*length] = sILWarpingParas[jj + (ll + 6 * ii)*length];
		}

		if (mode < 2)
		{
			for (ll = 0; ll < 6; ll++)
			{
				sprintf(Fname, "%s/Results/Sep/%05d_C1TSp%d.dat", PATH, frameID, ll);
				precomputedAvail = ReadGridBinary(Fname, sTWarpingParas + ll*length, width, height);
				if (!precomputedAvail)
					break;
			}
		}
		if (precomputedAvail)
		{
			if (mode < 2)
			{
				cout << "Loaded precomputed Texture warping for frame " << frameID << " in " << omp_get_wtime() - start << endl;
				for (jj = 0; jj < length; jj++)
					for (ll = 0; ll < 6; ll++)
						TWarpingParas[jj + ll*length] = sTWarpingParas[jj + ll*length];
			}

			for (ll = 0; ll < 5; ll++)//2Ilums+text
			{
				sprintf(Fname, "%s/Results/Sep/%05d_C1PA_%d.dat", PATH, frameID, ll);
				if (!ReadGridBinary(Fname, PhotoAdj + ll*length, width, height))
					break;
			}
		}
	}
	delete[]sILWarpingParas, delete[]sTWarpingParas;

	//Merge seedtype: 0: No touched point, 1: Illum 1, 2: Illum 2, 3, Illum1+Illum2, 4: Illum1+text, 5: Illum2+text, 6: Illums+text
	for (int ii = 0; ii < length; ii++)
	{
		if (abs(ILWarpingParas[ii]) + abs(ILWarpingParas[ii + length]) > 0.01 &&abs(ILWarpingParas[ii + 6 * length]) + abs(ILWarpingParas[ii + 7 * length]) > 0.01 && abs(TWarpingParas[ii]) + abs(TWarpingParas[ii + length]) > 0.01)
			SeedType[ii] = 6;
		else if (abs(ILWarpingParas[ii]) + abs(ILWarpingParas[ii + length]) > 0.01 && abs(TWarpingParas[ii]) + abs(TWarpingParas[ii + length]) > 0.01)
			SeedType[ii] = 4;
		else if (abs(ILWarpingParas[ii + 6 * length]) + abs(ILWarpingParas[ii + 7 * length]) > 0.01 && abs(TWarpingParas[ii]) + abs(TWarpingParas[ii + length]) > 0.01)
			SeedType[ii] = 5;
		else if (abs(ILWarpingParas[ii]) + abs(ILWarpingParas[ii + length]) > 0.01 && abs(ILWarpingParas[ii + 6 * length]) + abs(ILWarpingParas[ii + 7 * length]) > 0.01)
			SeedType[ii] = 3;
		else if (abs(ILWarpingParas[ii]) > 0.01)
			SeedType[ii] = 1;
		else if (abs(ILWarpingParas[ii + 6 * length]) > 0.01)
			SeedType[ii] = 2;
	}

	/*//Clean Texture & SeedType
	int validCount = 0;
	double *MagText = new double[length];
	for (int jj = 0; jj < height; jj++)
	{
	for (int ii = 0; ii < width; ii++)
	{
	double mag = sqrt(TWarpingParas[ii + jj*width] * TWarpingParas[ii + jj*width] + TWarpingParas[ii + jj*width + length] * TWarpingParas[ii + jj*width + length]);
	if (mag > 0.01)
	{
	MagText[validCount] = mag;
	validCount++;
	}
	}
	}

	double meanText = MeanArray(MagText, validCount);
	double stdText = sqrt(VarianceArray(MagText, validCount, meanText));

	for (int jj = 0; jj < height; jj++)
	{
	for (int ii = 0; ii < width; ii++)
	{
	int ind = ii + jj*width;
	double mag = sqrt(TWarpingParas[ii + jj*width] * TWarpingParas[ii + jj*width] + TWarpingParas[ii + jj*width + length] * TWarpingParas[ii + jj*width + length]);
	if (mag >meanText + 1.2*stdText)
	{
	for (int kk = 0; kk < 6; kk++)
	TWarpingParas[ind + kk*length] = 0.0;
	if (SeedType[ind] == 4)
	SeedType[ind] = 1;
	else if (SeedType[ind] == 5)
	SeedType[ind] = 2;
	else if (SeedType[ind] == 6)
	SeedType[ind] = 3;
	}
	}
	}
	delete[]MagText;*/

	//Start decomposition
	start = omp_get_wtime();
	if (nthreads > 1)
	{
		omp_set_num_threads(nthreads);

		int ROIH = ROI[1].y - ROI[0].y, subROI = ROIH / nthreads;
		CPoint sROI[nthreads * 2];
		for (ii = 0; ii < nthreads; ii++)
		{
			sROI[2 * ii].x = ROI[0].x, sROI[2 * ii + 1].x = ROI[1].x;
			sROI[2 * ii].y = ROI[0].y + subROI*ii - 1, sROI[2 * ii + 1].y = ROI[0].y + subROI*(ii + 1) + 1;
		}
		sROI[nthreads * 2 - 1].y = ROI[1].y;

		bool *scROI = new bool[nthreads*length];
		for (kk = 0; kk < nthreads; kk++)
		{
			for (ii = 0; ii < length; ii++)
				scROI[ii + kk*length] = false;
			for (jj = sROI[2 * kk].y; jj < sROI[2 * kk + 1].y; jj++)
				for (ii = sROI[2 * kk].x; ii < sROI[2 * kk + 1].x; ii++)
					scROI[ii + jj*width + kk*length] = cROI[ii + jj*width];
		}
		int mapSeed[] = { 4 };
		for (kk = 0; kk < 1; kk++)
		{
			for (ll = 0; ll < LKArg.npass2; ll++)
			{
#pragma omp parallel 
			{
#pragma omp for nowait
				for (int ii = 0; ii < nthreads; ii++)
				{
					TwoIllumTextSeperation(frameID, TPATH, IlluminationImages, SourceTexture, ParaSourceTexture, SSIG, DInfo, ILWarpingParas, TWarpingParas, PhotoAdj, previousTWarpingParas, SeedType, PrecomSearchR, LKArg, mode, scROI + ii*length, ii, mapSeed[kk]);
				}
			}
			//LKArg.hsubset -= 2;
			}
		}
		delete[]scROI;
	}
	else
	{
		int mapSeed[] = { 4 };
		for (ll = 0; ll < LKArg.npass2; ll++)
		{
			for (kk = 0; kk < 1; kk++)
				TwoIllumTextSeperation(frameID, TPATH, IlluminationImages, SourceTexture, ParaSourceTexture, SSIG, DInfo, ILWarpingParas, TWarpingParas, PhotoAdj, previousTWarpingParas, SeedType, PrecomSearchR, LKArg, mode, cROI, 0, mapSeed[kk]);
			//LKArg.hsubset -= 2;
		}
	}
	cout << "Total time: " << omp_get_wtime() - start << endl;


	//Create 3D + clean all possible bad points
	double P1mat[24];
	CPoint2 Ppts[2], Cpts; CPoint3 WC;

	P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0,
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0,
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;
	P1mat[12 + 0] = DInfo.P[0], P1mat[12 + 1] = DInfo.P[1], P1mat[12 + 2] = DInfo.P[2], P1mat[12 + 3] = DInfo.P[3],
		P1mat[12 + 4] = DInfo.P[4], P1mat[12 + 5] = DInfo.P[5], P1mat[12 + 6] = DInfo.P[6], P1mat[12 + 7] = DInfo.P[7],
		P1mat[12 + 8] = DInfo.P[8], P1mat[12 + 9] = DInfo.P[9], P1mat[12 + 10] = DInfo.P[10], P1mat[12 + 11] = DInfo.P[11];

	sprintf(Fname, "%s/Results/Sep/%05d_C1.xyz", PATH, frameID); FILE *fp = fopen(Fname, "w+");
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			int seedtype = SeedType[ii + jj*width];
			if (abs(ILWarpingParas[ii + jj*width]) + abs(ILWarpingParas[ii + jj*width + length]) < 0.001 && abs(ILWarpingParas[ii + jj*width + 6 * length]) + abs(ILWarpingParas[ii + jj*width + 7 * length]) < 0.001)
				continue;
			else if (seedtype != 3 && seedtype != 6)
			{
				for (int proID = 0; proID < 2; proID++)
				{
					if (abs(ILWarpingParas[ii + jj*width + 6 * proID*length]) + abs(ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length]) > 0.001)
					{
						Ppts[0].x = ILWarpingParas[ii + jj*width + 6 * proID*length] + ii, Ppts[0].y = ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length] + jj;
						if (CamProGeoVerify(1.0*ii, 1.0*jj, Ppts, WC, P1mat, DInfo, width, height, pwidth, pheight, proID) == 1)
							ILWarpingParas[ii + jj*width + 6 * proID*length] = 0.0, ILWarpingParas[ii + jj*width + (1 + 6 * proID)*length] = 0.0, TWarpingParas[ii + jj*width] = 0.0, TWarpingParas[ii + jj*width + length] = 0.0;
						else
							fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
					}
				}
			}
			else if (seedtype == 3 || seedtype == 6)
			{
				Ppts[0].x = ILWarpingParas[ii + jj*width] + ii, Ppts[0].y = ILWarpingParas[ii + jj*width + length] + jj;
				Ppts[1].x = ILWarpingParas[ii + jj*width + 6 * length] + ii, Ppts[1].y = ILWarpingParas[ii + jj*width + 7 * length] + jj;
				if (CamProGeoVerify(1.0*ii, 1.0*jj, Ppts, WC, P1mat, DInfo, width, height, pwidth, pheight, 2) == 1)
					ILWarpingParas[ii + jj*width] = 0.0, ILWarpingParas[ii + jj*width + length] = 0.0, ILWarpingParas[ii + jj*width + 6 * length] = 0.0, ILWarpingParas[ii + jj*width + 7 * length] = 0.0, TWarpingParas[ii + jj*width] = 0.0, TWarpingParas[ii + jj*width + length] = 0.0;
				else
					fprintf(fp, "%.3f %.3f %.3f \n", WC.x, WC.y, WC.z);
			}
		}
	}
	fclose(fp);

	//Synthesize separated images
	if (color)
	{
		double *SourceTexture = new double[length * 3];
		double *ParaSourceTexture = new double[length * 3];
		if (mode < 2)
		{
			if (mode == 0)
				sprintf(Fname, "%s/Image/C1_%05d.png", PATH, frameID - 1);
			else if (mode == 1)
			{
				if (frameID% IlluminationImages.iRate == 0)
					sprintf(Fname, "%s/Image/C1_%05d.png", PATH, frameID - (IlluminationImages.iRate - 1));
				else
					sprintf(Fname, "%s/Image/C1_%05d.png", PATH, frameID / IlluminationImages.iRate*IlluminationImages.iRate + 1);
			}

			view = imread(Fname, 1);
			if (view.data == NULL)
			{
				cout << "Cannot load: " << Fname << endl;
				return 2;
			}
			for (kk = 0; kk < 3; kk++)
			{
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						SourceTexture[ii + jj*width + kk*length] = (double)((int)(view.data[3 * ii + (height - 1 - jj) * 3 * width + kk]));

				Gaussian_smooth_Double(SourceTexture + kk*length, SourceTexture + kk*length, height, width, 255.0, 0.707);
				Generate_Para_Spline(SourceTexture + kk*length, ParaSourceTexture + kk*length, width, height, LKArg.InterpAlgo);
			}
		}

		sprintf(Fname, "%s/Results/Sep", PATH);
		UpdateIllumTextureImages(Fname, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, IlluminationImages.PPara, ILWarpingParas, ParaSourceTexture, TWarpingParas, bkcROI, color);
		delete[]SourceTexture, delete[]ParaSourceTexture;
	}
	else
	{
		sprintf(Fname, "%s/Results/Sep", PATH);
		UpdateIllumTextureImages(Fname, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, IlluminationImages.PPara, ILWarpingParas, ParaSourceTexture, TWarpingParas, bkcROI);
	}

	//Save warping paras
	for (jj = 0; jj < nPros; jj++)
	{
		for (ii = 0; ii < 6; ii++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, jj + 1, ii);
			WriteGridBinary(Fname, ILWarpingParas + (ii + 6 * jj)*length, width, height);
		}
	}
	if (mode < 2)
	{
		for (ii = 0; ii < 6; ii++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1TSp%d.dat", PATH, frameID, ii);
			WriteGridBinary(Fname, TWarpingParas + ii*length, width, height);
		}
	}
	for (ii = 0; ii < 5; ii++)
	{
		sprintf(Fname, "%s/Results/Sep/%05d_C1PA_%d.dat", PATH, frameID, ii);
		WriteGridBinary(Fname, PhotoAdj + ii*length, width, height);
	}
	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
	WriteGridBinary(Fname, SeedType, width, height);

	delete[]ILWarpingParas, delete[]TWarpingParas, delete[]previousTWarpingParas, delete[]PhotoAdj;
	delete[]PrecomSearchR, delete[]SeedType;
	delete[]SSIG, delete[]SourceTexture, delete[]ParaSourceTexture, delete[]cROI;

	return 0;
}
int UndistortImageDrivers(char *PATH, int nCams, int width, int height, int nchannels, int InterpAlgo, int start, int stop)
{
	int ii, jj, kk, ll, mm, length = width*height;
	char Fname[200];
	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	double S[3];
	CPoint2 uv;
	Mat view;
	double *Img = new double[length*nchannels];
	double *ParaImg = new double[length*nchannels];
	for (mm = 0; mm < nCams; mm++)
	{
		for (ll = start; ll <= stop; ll++)
		{
			sprintf(Fname, "%s/Image/C1%d_%05d.png", PATH, mm + 1, ll);
			view = imread(Fname, nchannels == 1 ? 0 : 1);
			if (view.data == NULL)
			{
				cout << "Cannot load: " << Fname << endl;
				return 2;
			}
			for (kk = 0; kk < nchannels; kk++)
			{
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						Img[ii + jj*width + kk*length] = (double)((int)(view.data[nchannels*ii + (height - 1 - jj)*nchannels*width + kk]));
				Generate_Para_Spline(Img + kk*length, ParaImg + kk*length, width, height, InterpAlgo);

				for (jj = 0; jj < height; jj++)
				{
					for (ii = 0; ii<width; ii++)
					{
						uv.x = 1.0*ii, uv.y = 1.0*jj;
						Undo_distortion(uv, DInfo.K + (mm + 1) * 9, DInfo.distortion + (mm + 1) * 13);
						Get_Value_Spline(ParaImg + kk*length, width, height, uv.x, uv.y, S, -1, InterpAlgo);
						if (S[0]>255.0)
							Img[ii + jj*width + kk*length] = 255.0;
						else if (S[0] < 0.0)
							Img[ii + jj*width + kk*length] = 0.0;
						else
							Img[ii + jj*width + kk*length] = S[0];
					}
				}
			}
			sprintf(Fname, "%s/Image/_C%d_%05d.png", PATH, mm + 1, ll);
			SaveDataToImage(Fname, Img, width, height, nchannels);
		}
	}

	return 0;
}
int SURFMatching(char *Fname1, char *Fname2, bool ShowImage = false, int signature = 0)
{
	char Fname[200];
	sprintf(Fname, "%s.png", Fname1); 	Mat img_1 = imread(Fname, 1);
	sprintf(Fname, "%s.png", Fname2); 	Mat img_2 = imread(Fname, 1);

	if (!img_1.data || !img_2.data)
	{
		std::cout << " --(!) Error reading images " << std::endl; return -1;
	}

	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 1000;

	SiftFeatureDetector detector(minHessian);
	//SurfFeatureDetector detector( minHessian );

	std::vector<KeyPoint> keypoints_1, keypoints_2;

	detector.detect(img_1, keypoints_1);
	detector.detect(img_2, keypoints_2);

	//-- Step 2: Calculate descriptors (feature vectors)
	SiftDescriptorExtractor extractor;
	//SurfDescriptorExtractor extractor;

	Mat descriptors_1, descriptors_2;

	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist)
			min_dist = dist;
		if (dist > max_dist)
			max_dist = dist;
	}

	//printf("-- Max dist : %f \n", max_dist );
	//printf("-- Min dist : %f \n", min_dist );

	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
		if (matches[i].distance <= max(7 * min_dist, 0.02))
			good_matches.push_back(matches[i]);

	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	if (ShowImage)
	{
		namedWindow("Good Matches", WINDOW_AUTOSIZE);
		imshow("Good Matches", img_matches);
		waitKey(0);
	}


	/*if (signature == 0) //Supreeth
		sprintf(Fname, "%s.txt", Fname1);
		else
		sprintf(Fname, "%s_%d.txt", Fname1, signature);
		FILE* fp = fopen(Fname, "w+");
		for (int i = 0; i < (int)good_matches.size(); i++)
		fprintf(fp, "%.2f %2.f \n", keypoints_1.at(good_matches[i].queryIdx).pt.x - 1, 1.0*img_1.rows - 1 - keypoints_1.at(good_matches[i].queryIdx).pt.y);
		fclose(fp);

		if (signature == 0)
		sprintf(Fname, "%s.txt", Fname2);
		else
		sprintf(Fname, "%s_%d.txt", Fname2, signature);
		fp = fopen(Fname, "w+");
		for (int i = 0; i < (int)good_matches.size(); i++)
		fprintf(fp, "%.2f %2.f \n", keypoints_2.at(good_matches[i].trainIdx).pt.x - 1, 1.0*img_1.rows - 1 - keypoints_2.at(good_matches[i].trainIdx).pt.y);
		fclose(fp);*/

	sprintf(Fname, "C:/temp/S/Sparse/C1P1_%05d.txt", signature);
	FILE* fp = fopen(Fname, "w+");
	for (int i = 0; i < (int)good_matches.size(); i++)
		fprintf(fp, "%.2f %2.f \n", keypoints_1.at(good_matches[i].queryIdx).pt.x - 1, 1.0*img_1.rows - 1 - keypoints_1.at(good_matches[i].queryIdx).pt.y);
	fclose(fp);

	sprintf(Fname, "C:/temp/S/Sparse/P1C1_%05d.txt", signature);
	fp = fopen(Fname, "w+");
	for (int i = 0; i < (int)good_matches.size(); i++)
		fprintf(fp, "%.2f %2.f \n", keypoints_2.at(good_matches[i].trainIdx).pt.x - 1, 1.0*img_1.rows - 1 - keypoints_2.at(good_matches[i].trainIdx).pt.y);
	fclose(fp);

	return 0;
}
int ComputeSparseCorresSim(char *bPATH, char *PATH, int frameID, int nCams, int width, int height, int pwidth, int pheight, int SuperRes)
{
	int npts = 500000, hsubset = 5, hsubset2 = 6, searchArea = 1, InterpAlgo = 5;
	double ZNCCcoarseThresh = 0.6, ZNNCThresh = 0.95;
	vector<double> PatternAngles; PatternAngles.push_back(0.0), PatternAngles.push_back(30.0);

	int nframes = 2, frameJump = 1;
	int plength = pwidth*pheight;
	double x, y, z;
	char Fname[200];

	DevicesInfo DInfo(nCams);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot ProCam Info" << endl;
		return 1;
	}

	int a, b, c, nPpts = 3871;
	CPoint2 *Pcorners = new CPoint2[nPpts];
	sprintf(Fname, "%s/ProCorners.txt", PATH);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
		cout << "Cannot load ProCorners" << endl;
	fscanf(fp, "%d %d %d", &a, &b, &c);
	for (int ii = 0; ii < nPpts; ii++)
		fscanf(fp, "%d %lf %lf ", &a, &Pcorners[ii].x, &Pcorners[ii].y);
	fclose(fp);


	CPoint2*projectedCorners = new CPoint2[nPpts];
	for (int mm = 0; mm < nframes; mm++)
	{
		float *GroundTruth = new float[plength*SuperRes*SuperRes];
		sprintf(Fname, "%s/GT/D_%05d.ijz", bPATH, frameID + mm);
		if (!ReadGridBinary(Fname, GroundTruth, pwidth*SuperRes, pheight*SuperRes))
		{
			cout << "Cannot load " << Fname << endl;
			return 4;
		}

		double rayDirect[2], PrayDirect[6], denum;
		for (int kk = 0; kk < nPpts; kk++)
		{
			int ii = (int)(Pcorners[kk].x*SuperRes), jj = (int)(Pcorners[kk].y*SuperRes);
			if (abs(GroundTruth[ii + jj*SuperRes*pwidth]) < 1.0)
				continue;

			z = (float)GroundTruth[ii + jj*SuperRes*pwidth];
			x = (float)(z*(DInfo.iK[0] * ii / SuperRes + DInfo.iK[1] * jj / SuperRes + DInfo.iK[2]));
			y = (float)(z*(DInfo.iK[4] * jj / SuperRes + DInfo.iK[5]));

			//project to image

			rayDirect[0] = DInfo.iK[0] * ii / SuperRes + DInfo.iK[1] * jj / SuperRes + DInfo.iK[2], rayDirect[1] = DInfo.iK[4] * jj / SuperRes + DInfo.iK[5];

			PrayDirect[0] = DInfo.P[12 * 0] * rayDirect[0] + DInfo.P[12 * 0 + 1] * rayDirect[0 + 1] + DInfo.P[12 * 0 + 2], PrayDirect[1] = DInfo.P[12 * 0 + 3];
			PrayDirect[2] = DInfo.P[12 * 0 + 4] * rayDirect[0] + DInfo.P[12 * 0 + 5] * rayDirect[0 + 1] + DInfo.P[12 * 0 + 6], PrayDirect[3] = DInfo.P[12 * 0 + 7];
			PrayDirect[4] = DInfo.P[12 * 0 + 8] * rayDirect[0] + DInfo.P[12 * 0 + 9] * rayDirect[0 + 1] + DInfo.P[12 * 0 + 10], PrayDirect[5] = DInfo.P[12 * 0 + 11];

			denum = GroundTruth[ii + jj*SuperRes*pwidth] * PrayDirect[4] + PrayDirect[5];
			projectedCorners[kk].x = (GroundTruth[ii + jj*SuperRes*pwidth] * PrayDirect[0] + PrayDirect[1]) / denum;
			projectedCorners[kk].y = (GroundTruth[ii + jj*SuperRes*pwidth] * PrayDirect[2] + PrayDirect[3]) / denum;
		}

		sprintf(Fname, "%s/Image/C1_%05d.png", PATH, frameID + mm);
		Mat cvImg = imread(Fname, 0);
		if (cvImg.data == NULL)
		{
			cout << "Cannot load " << Fname << endl;
			return 1;
		}
		else
			cout << "Loaded :" << Fname << endl;

		int width = cvImg.cols, height = cvImg.rows, length = width*height;

		double *Img = new double[length];
		double *ParaImg = new double[length];
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width] = (double)(int)cvImg.data[ii + (height - 1 - jj)*width];
		Generate_Para_Spline(Img, ParaImg, width, height, InterpAlgo);

		int *type = new int[npts];
		CPoint2 *Pts = new CPoint2[npts];
		RunCornersDetector(Pts, type, npts, Img, ParaImg, width, height, PatternAngles, hsubset, hsubset2, searchArea, ZNCCcoarseThresh, ZNNCThresh, InterpAlgo);

		sprintf(Fname, "%s/corners1_%05d.txt", PATH, frameID + mm); fp = fopen(Fname, "w+");
		if (fp == NULL)
		{
			cout << "Cannot write " << Fname << endl;
			return 1;
		}
		for (int ii = 0; ii < npts; ii++)
			fprintf(fp, "%.6f %.6f\n", Pts[ii].x, Pts[ii].y);
		fclose(fp);

		int *CCorresID = new int[nPpts];
		sprintf(Fname, "%s/C1_%d_Ppts.txt", PATH, frameID + mm); 	FILE *fp1 = fopen(Fname, "w+");
		sprintf(Fname, "%s/Sparse/C1_%05d.txt", PATH, frameID + mm); FILE *fp2 = fopen(Fname, "w+");
		for (int kk = 0; kk < nPpts; kk++)
		{
			for (int ll = 0; ll < npts; ll++)
			{
				if ((projectedCorners[kk].x - Pts[ll].x)*(projectedCorners[kk].x - Pts[ll].x) + (projectedCorners[kk].y - Pts[ll].y)*(projectedCorners[kk].y - Pts[ll].y) < 0.5)
				{
					fprintf(fp1, "%d %.8f %.8f \n", kk, Pcorners[kk].x, Pcorners[kk].y);
					fprintf(fp2, "%.8f %.8f \n", projectedCorners[kk].x, projectedCorners[kk].y);
					break;
				}
			}
		}
		fclose(fp1), fclose(fp2);

		delete[]GroundTruth;
		delete[]Img;
		delete[]ParaImg;
		delete[]Pts;
		delete[]type;
	}

	CPoint2 IROI[2], *PROI = new CPoint2[2 * nframes / 2];
	IROI[0].x = 20, IROI[0].y = 20; IROI[1].x = width - 20, IROI[1].y = height - 20;
	for (int ii = 0; ii < nframes / 2; ii++)
	{
		PROI[2 * ii].x = 20, PROI[2 * ii].y = 20;
		PROI[2 * ii + 1].x = pwidth - 20, PROI[2 * ii + 1].y = pheight - 20;
	}

	int nPros = 1;
	CleanValidChecker(nCams, nPros, frameJump, frameID, nframes, IROI, PROI, PATH);
	TwoDimTriangulation(frameID, frameJump, nCams, nPros, nframes, pwidth, pheight, PATH);

	delete[]Pcorners;

	return 0;
}
int TestGTDecomposed(char *PATH, int width, int height, int frameID)
{
	int ii, jj, length = width*height;
	char Fname[200];

	//1: Setup cameras & projector parameters and load images data
	DevicesInfo DInfo(1);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot CamPro Info" << endl;
		return 1;
	}

	float *TextWarping = new float[length * 2];
	for (ii = 0; ii < 2; ii++)
	{
		sprintf(Fname, "%s/Results/Sep/Dogs/C1TS1p%d_%05d.dat", PATH, ii, frameID);
		if (!ReadGridBinary(Fname, TextWarping + ii*length, width, height, false))
			return 1;
	}


	float *OpticalWarping = new float[length * 2];
	char FnameX[200], FnameY[200];
	sprintf(FnameX, "%s/Flow/RX%d_%05d.dat", PATH, 1, frameID);
	sprintf(FnameY, "%s/Flow/RY%d_%05d.dat", PATH, 1, frameID);
	if (!ReadFlowBinary(FnameX, FnameY, OpticalWarping, OpticalWarping + length, width, height))
		return 1;

	FILE *fp = fopen("C:/temp/uvT.txt", "w+");
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			if (abs(TextWarping[ii + jj*width]) + abs(TextWarping[ii + jj*width + length]) > 0.01)
			{
				fprintf(fp, "%.4f %.4f \n", TextWarping[ii + jj*width] - OpticalWarping[ii + jj*width], TextWarping[ii + jj*width + length] - OpticalWarping[ii + jj*width + length]);
			}
		}
	}
	fclose(fp);


	float *IllumWarpingGT = new float[2 * length];
	for (ii = 0; ii < 2; ii++)
	{
		sprintf(Fname, "%s/Results/CamPro/C1p%d_%05d.dat", PATH, ii, frameID);
		if (!ReadGridBinary(Fname, IllumWarpingGT + ii*length, width, height, false))
			return 1;
	}

	float *IllumWarping = new float[length * 2];
	for (ii = 0; ii < 2; ii++)
	{
		sprintf(Fname, "%s/Results/Sep/Dogs/C1p%d_%05d.dat", PATH, ii, frameID);
		if (!ReadGridBinary(Fname, IllumWarping + ii*length, width, height, false))
			return 1;
	}

	fp = fopen("C:/temp/uvL.txt", "w+");
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			if (abs(TextWarping[ii + jj*width]) + abs(TextWarping[ii + jj*width + length]) > 0.01 && abs(IllumWarping[ii + jj*width]) + abs(IllumWarping[ii + jj*width + length]) > 0.01 && abs(IllumWarpingGT[ii + jj*width]) + abs(IllumWarpingGT[ii + jj*width + length]) > 0.01)
			{
				fprintf(fp, "%.4f %.4f \n", IllumWarpingGT[ii + jj*width] - IllumWarping[ii + jj*width], IllumWarpingGT[ii + jj*width + length] - IllumWarping[ii + jj*width + length]);
				//fprintf(fp, "%.4f %.4f \n", u, v);
			}
		}
	}
	fclose(fp);

	return 0;
}
int ModifyProjectorPattern(char *Fname, int pitch, int nchannels)
{
	Mat cvImg = imread(Fname, nchannels == 1 ? 0 : 1);
	if (cvImg.data == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return 1;
	}
	int width = cvImg.cols, height = cvImg.rows, length = width*height;
	unsigned char *Img = new unsigned char[length*nchannels];
	for (int ii = 0; ii < nchannels*length; ii++)
		Img[ii] = (unsigned char)255;

	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = pitch; jj < height - pitch; jj++)
			for (int ii = pitch; ii < width - pitch; ii++)
				Img[ii + jj*width + kk*length] = cvImg.data[nchannels*ii + (height - 1 - jj)*nchannels*width + kk];
	}
	SaveDataToImage(Fname, Img, width, height, nchannels);
	return 0;
}
int DenseTrackingwithPOI(char *PATH, int width, int height, int nchannels, int startID, int stopID, int frameJump, int forward, LKParameters &LKArg)
{
	int numThreads = (stopID - startID + 1); numThreads = numThreads > MAXTHREADS ? MAXTHREADS : numThreads;
	omp_set_num_threads(numThreads);

	if (forward)
		printf("Run forward tracking\n");
	else
		printf("Run backward tracking\n");

#pragma omp parallel 
	{
#pragma omp for nowait
		for (int ii = startID; ii <= stopID; ii += frameJump)
		{
			char Fname1[200], Fname2[200];
#pragma omp critical
			printf("Run POI tracking on %d\n", ii);
			sprintf(Fname1, "%s/Image/RandomTextureMap/C1_%05d", PATH, ii);
			if (forward)
				sprintf(Fname2, "%s/Image/RandomTextureMap/C1_%05d", PATH, ii + 1);
			else
				sprintf(Fname2, "%s/Image/RandomTextureMap/C1_%05d", PATH, 1);

			//SURFMatching(Fname1, Fname2, false, ii);
			FlowMatching(PATH, 1, 1, 1, ii, 0, width, height, nchannels, LKArg, forward == 1, false, ii);
		}
	}

	return 0;
}

int FlowDistanceToTemplate(char *Path)
{
	char FnameX[200], FnameY[200];
	int width = 1920, height = 1080;

	//Cum flow
	float *fX = new float[width*height];
	float *fY = new float[width*height];
	float *PfX = new float[width*height];
	float *PfY = new float[width*height];
	float *Displacement = new float[width*height];
	for (int ii = 0; ii < width*height; ii++)
		Displacement[ii] = 0.0;

	float *cX = new float[width*height];
	float *cY = new float[width*height];
	for (int jj = 0; jj < height; jj++)
		for (int ii = 0; ii < width; ii++)
			cX[ii + jj*width] = ii, cY[ii + jj*width] = jj;


	for (int id = 2; id < 60; id++)
	{
		sprintf(FnameX, "%s/Flow/FX1_%05d.dat", Path, id), sprintf(FnameY, "%s/Flow/FY1_%05d.dat", Path, id);
		if (!ReadFlowBinary(FnameX, FnameY, fX, fY, width, height))
			return 1;

		for (int jj = 0; jj < height; jj++)
		{
			for (int ii = 0; ii < width; ii++)
			{
				cX[ii + jj*width] += fX[ii + jj*width];
				cY[ii + jj*width] += fY[ii + jj*width];

				if (cX[ii + jj*width] < 0)
					cX[ii + jj*width] = 0.0;
				if (cX[ii + jj*width]>width - 1)
					cX[ii + jj*width] = width - 1;
				if (cY[ii + jj*width] < 0)
					cY[ii + jj*width] = 0.0;
				if (cY[ii + jj*width]>height - 1)
					cY[ii + jj*width] = height - 1;
			}
		}

		Generate_Para_Spline(fX, PfX, width, height, 1);
		Generate_Para_Spline(fY, PfY, width, height, 1);

		double Sx[3], Sy[3];
		for (int jj = 0; jj < height; jj++)
		{
			for (int ii = 0; ii < width; ii++)
			{
				Get_Value_Spline(PfX, width, height, cX[ii + jj*width], cY[ii + jj*height], Sx, 0, 1);
				Get_Value_Spline(PfY, width, height, cX[ii + jj*width], cY[ii + jj*height], Sy, 0, 1);
				Displacement[ii + jj*width] += sqrt(Sx[0] * Sx[0] + Sy[0] * Sy[0]);
			}
		}

		sprintf(FnameX, "%s/Flow/Cum_%05d.dat", Path, id);
		WriteGridBinary(FnameX, Displacement, width, height);
	}

	return 0;
}
int MergeSeedType(char *PATH, int nPros, int frameID)
{
	int width = 1920, height = 1080, length = width*height;
	float *ILWarpingParas = new float[2 * length];
	char Fname[200];

	for (int jj = 0; jj < nPros; jj++)
	{
		sprintf(Fname, "%s/Results/CamPro/C1P%dp%d_%05d.dat", PATH, jj + 1, 0, frameID);
		if (!ReadGridBinary(Fname, ILWarpingParas + jj*length, width, height))
		{
			cout << "Cannot load Campro warping for Projector " << jj + 1 << endl;
			delete[]ILWarpingParas;
			return 1;
		}
	}
	cout << "Loaded campro warpings in " << endl;

	//Create Sep Type
	int *SeedType = new int[length]; //0: No touched point, 1: Illum 1, 2: Illum 2, 3, Illum1+Illum2, 4: Illum1+text, 5: Illum2+text, 6: Illums+text
	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
	if (ReadGridBinary(Fname, SeedType, width, height))
	{
		for (int ii = 0; ii < length; ii++)
		{
			if (abs(ILWarpingParas[ii]) > 0.01)
				SeedType[ii] = 1;
			else if (nPros == 2 && abs(ILWarpingParas[ii + length]) > 0.01)
				SeedType[ii] = 2;
		}
	}
	else
		cout << "Cannot load SeedTypefor frame " << frameID << ". Re-intialize it. " << endl;

	sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
	WriteGridBinary(Fname, SeedType, width, height);

	delete[]ILWarpingParas, delete[]SeedType;

	return 0;
}
int DeleteSeedType(char *PATH)
{
	char Fname[200];
	int width = 1920, height = 1080, length = width*height;
	int *SeedType = new int[length]; //0: No touched point, 1: Illum 1, 2: Illum 2, 3, Illum1+Illum2, 4: Illum1+text, 5: Illum2+text, 6: Illums+text
	float *ILWarpingParas = new float[6 * length];
	float *TWarpingParas = new float[2 * length];

	for (int frameID = 62; frameID < 241; frameID++)
	{
		sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
		if (!ReadGridBinary(Fname, SeedType, width, height))
			continue;

		for (int ii = 0; ii < 2; ii++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, 1, ii);
			ReadGridBinary(Fname, ILWarpingParas + ii*length, width, height);
		}
		for (int ii = 0; ii < 2; ii++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1TSp%d.dat", PATH, frameID, ii);
			ReadGridBinary(Fname, TWarpingParas + ii*length, width, height);
		}

		for (int ii = 0; ii < length; ii++)
		{
			if (SeedType[ii] == 1)
				ILWarpingParas[ii] = 0.0, ILWarpingParas[ii + length] = 0.0, TWarpingParas[ii] = 0.0, TWarpingParas[ii + length] = 0.0, SeedType[ii] = 0;
		}

		sprintf(Fname, "%s/Results/Sep/%05d_SeedType.dat", PATH, frameID);
		WriteGridBinary(Fname, SeedType, width, height);
		for (int ii = 0; ii < 2; ii++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1P%dp%d.dat", PATH, frameID, 1, ii);
			WriteGridBinary(Fname, ILWarpingParas + ii*length, width, height);
		}
		for (int ii = 0; ii < 2; ii++)
		{
			sprintf(Fname, "%s/Results/Sep/%05d_C1TSp%d.dat", PATH, frameID, ii);
			WriteGridBinary(Fname, TWarpingParas + ii*length, width, height);
		}
	}
	return 0;

}

bool GrabImage(char *fname, double *Img, int &width, int &height, int nchannels, bool silent)
{
	Mat view = imread(fname, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		if (!silent)
			cout << "Cannot load: " << fname << endl;
		return false;
	}
	if (Img == NULL)
	{
		width = view.cols, height = view.rows;
		Img = new double[width*height*nchannels];
	}
	int length = width*height;
	for (int kk = 0; kk < nchannels; kk++)
	{
		for (int jj = 0; jj < height; jj++)
			for (int ii = 0; ii < width; ii++)
				Img[ii + jj*width + kk*length] = (double)(int)view.data[nchannels*ii + jj*nchannels*width + kk];
	}

	return true;
}
int main(int argc, char* argv[])
{
	/*{
		int width = 1920, height = 1080, nchannels = 3, length = width*height, patternSize = 27, patternLength = patternSize*patternSize, hsubset = 13;
		double *Img1 = new double[length * 3];
		double *Para = new double[length * 3];
		double *Pattern = new double[patternSize *patternSize * 3];

		char Fname[100];
		IplImage *view = 0;
		sprintf(Fname, "E:/Juggling/0/%d.png", 10); GrabImage(Fname, Img1, width, height, nchannels, true);
		for (int kk = 0; kk < nchannels; kk++)
		Generate_Para_Spline(Img1 + kk*length, Para + kk*length, width, height, 1);


		sprintf(Fname, "E:/Juggling/0/Green.png", 1); GrabImage(Fname, Pattern, patternSize, patternSize, nchannels, true);

		CPoint2 POI; POI.x = 986, POI.y = 335;
		TMatchingFine_ZNCC(Pattern, patternSize, hsubset, Para, width, height, nchannels, POI, 0, 1, 0.8, 1);
		return 0;
		}*/

	//VisTrack("C:/temp/X");
	//TrackOpenCVLK(1, 207, "C:/temp/X");
	//ProjectBinaryCheckRandom("C:/temp", 32, 6);
	//SURFMatching("C:/temp/oneshot_pattern", "C:/temp/oneshot_rect_gain", true);
	char DataPATH[] = "E:/ICCV/JumpF";
	char TDataPATH[] = "../../mnt";

	int mode = atoi(argv[1]);
	bool SimulationMode = false;

	int width = 1280, height = 720, pwidth = 800, pheight = 600, nCams = 8, nPros = 1, nchannels = 1, nframes = 2, frameJump = 1;
	CPoint ROI[2]; //ROI[0].x = 200, ROI[0].y = height - 980, ROI[1].x = 1350, ROI[1].y = height - 150;
	//ROI[0].x = 100, ROI[0].y = height-1000, ROI[1].x = 1800, ROI[1].y = height-20;
	//ROI[0].x = 250, ROI[0].y = 50, ROI[1].x = 1600, ROI[1].y = height - 50;
	ROI[0].x = 180, ROI[0].y = 100, ROI[1].x = 760, ROI[1].y = height - 170;

	LKParameters LKArg;
	LKArg.step = 1, LKArg.DisplacementThresh = 30, LKArg.DIC_Algo = 3, LKArg.InterpAlgo = 1, LKArg.EpipEnforce = 0;
	LKArg.Incomplete_Subset_Handling = 0, LKArg.Convergence_Criteria = 0, LKArg.Analysis_Speed = 0, LKArg.IterMax = 15;
	LKArg.Gsigma = 1.0, LKArg.ProjectorGsigma = LKArg.Gsigma, LKArg.ssigThresh = 1.0;

	/*int frameID = 370;
	char Fname1[200], Fname2[200];
	sprintf(Fname1, "%s/Image/C1_%05d", DataPATH, frameID);
	sprintf(Fname2, "%s/ProjectorPattern1", DataPATH);
	SURFMatching(Fname1, Fname2, true, frameID);
	return 0;*/

	if (mode == 0)
	{
		int CamID = atoi(argv[2]);
		int startID = atoi(argv[3]) - 1; //start  from frame 2 instead of 1

		double flowThresh = 0.1;
		LKArg.hsubset = 5, LKArg.PSSDab_thresh = 0.01, LKArg.ZNCCThreshold = 0.99, LKArg.Gsigma = 1.0;

		FlowScaleSelection FlowScaleSel;
		FlowScaleSel.startS = 2, FlowScaleSel.stopS = 20, FlowScaleSel.stepIJ = 2, FlowScaleSel.stepS = 1;

		int firstID = 1;
		//Note: firstID is special, pay attention!

		cout << "From: " << startID << " To: " << startID + 1 << endl;
		omp_set_num_threads(2);

		int status[2];
		double start = omp_get_wtime();
#pragma omp parallel 
		{
#pragma omp for nowait
			for (int ii = 0; ii < 2; ii++)
			{
				status[ii] = ScaleSelection(startID, CamID, ii == 0 ? true : false, width, height, flowThresh, LKArg, FlowScaleSel, DataPATH);
			}
		}
		cout << "Elapse: " << setw(2) << omp_get_wtime() - start << endl;
		cout << "S0:" << status[0] << endl;
		cout << "S1:" << status[1] << endl;
		if (status[0] == 0 && status[1] == 0)
			return 0;
		else if (status[0] != 0 && status[1] == 0)
			return status[0];
		else if (status[1] != 0 && status[0] == 0)
			return status[1];
		else
			return status[0] + status[1];
	}
	else if (mode == 1)
	{
		int forward = atoi(argv[2]),
			backward = atoi(argv[3]),
			startID = atoi(argv[4]),
			stopID = atoi(argv[5]); //start  from frame 2 instead of 1

		TVL1Parameters tvl1arg;
		tvl1arg.lamda = 0.5, tvl1arg.tau = 0.25, tvl1arg.theta = 0.01, tvl1arg.epsilon = 0.005, tvl1arg.iterations = 30, tvl1arg.nscales = 30, tvl1arg.warps = 20;
		
		cout << "Run TVL1 flow from: " << startID << " To: " << stopID << endl;
		for (int ii = startID; ii <= stopID; ii += frameJump)
			TVL1OpticalFlowDriver(ii, 1, nCams, width, height, DataPATH, tvl1arg, forward, backward);

		return 0;
	}
	else if (mode == 2)
	{
		double SR = atof(argv[2]);
		int startID = atoi(argv[3]),
			stopID = atoi(argv[4]),
			backward = atoi(argv[5]);
		cout << "From: " << startID << " To: " << startID + frameJump << endl;

		LKArg.hsubset = 10, LKArg.PSSDab_thresh = 0.01, LKArg.ZNCCThreshold = 0.99;
		LKArg.DIC_Algo = 3;

		int status[20];
		omp_set_num_threads(2);
		double start = omp_get_wtime();
		for (int jj = 0; jj < nCams; jj++)
		{
#pragma omp parallel sections
			{
#pragma omp section
				{
					cout << "Running forward flow" << endl;
					for (int ii = startID; ii <= stopID; ii += frameJump)
					{
						if (abs(SR - 1.0) < 0.001)
							status[jj * 2] = FlowMatching(DataPATH, nCams, nPros, jj + 1, ii, frameJump, width, height, nchannels, LKArg, true, false, 0, true);
						else
							;//status[jj*2] = SFlowMatching(DataPATH, nCams, jj+1, ii, frameJump, nchannels, LKArg, SR, true);
					}
				}
#pragma omp section
				{
					if (backward)
					{
						cout << "Running backward flow" << endl;
						for (int ii = startID; ii <= stopID; ii += frameJump)
						{
							if (abs(SR - 1.0) < 0.001)
								;//status[jj*2+1] = FlowMatching(DataPATH, nCams, jj+1, ii, frameJump, width, height, nchannels, LKArg, false);
							else
								;//status[1+jj*2] = SFlowMatching(DataPATH, nCams, jj+1, ii, frameJump, nchannels, LKArg, SR, false);
						}
					}
				}
			}
		}

		cout << "Elapse: " << setw(2) << omp_get_wtime() - start << endl;

		if (status[0] == 0 && status[1] == 0)
			return 0;
		else if (status[0] != 0 && status[1] == 0)
			return status[0];
		else if (status[1] != 0 && status[0] == 0)
			return status[1];
		else
			return status[0] + status[1];
	}
	else if (mode == 3) //Surf-seeded flow
	{
		int startID = atoi(argv[2]),
			stopID = atoi(argv[3]),
			forward = atoi(argv[4]);

		LKArg.hsubset = 7, LKArg.PSSDab_thresh = 0.025, LKArg.ZNCCThreshold = 0.975;
		LKArg.DIC_Algo = 3;

		DenseTrackingwithPOI(DataPATH, width, height, nchannels, startID, stopID, frameJump, forward, LKArg);

		return 0;
	}
	else if (mode == 4) //Stereo
	{
		int startID = 0, stopID = 1;
		//int startID = atoi(argv[1+1+2]), stopID = atoi(argv[1+1+3]); 
		cout << "From: " << startID << " To: " << stopID << endl;

		LKArg.DIC_Algo = 4, LKArg.hsubset = 11, LKArg.PSSDab_thresh = 0.05, LKArg.ZNCCThreshold = 0.95;

		int firstID = 1; //Note: firstID is special, pay attention!
		for (int frameID = startID; frameID <= stopID; frameID++)
		{
			int returnvalue = StereoMatching(DataPATH, frameID, firstID, width, height, nchannels, LKArg);
			if (returnvalue != 0)
				return returnvalue;
		}

		return 0;
	}
	else if (mode == 5) //CamPro
	{
		bool flowVerification = false;
		double SR = atof(argv[2]);
		int procam = atoi(argv[3]), startID = atoi(argv[4]), stopID = atoi(argv[5]);
		//double SR = 1.0; int procam = 0, startID = 32, stopID = 32;

		cout << "From: " << startID << " To: " << stopID << " with SR: " << SR << endl;

		LKArg.DIC_Algo = 0,
			LKArg.hsubset = 9, LKArg.PSSDab_thresh = .05, LKArg.ZNCCThreshold = .855;
		bool saveWarping = 1;
		double triThresh = 2.0;

		int status, offset = (procam == 1) ? 0 : 1;
		int numThreads = (stopID - startID) / 2 + 1; numThreads = numThreads > MAXTHREADS ? MAXTHREADS : numThreads;
		omp_set_num_threads(numThreads + procam);

#pragma omp parallel for
		for (int frameID = startID; frameID <= stopID; frameID += (2 - procam)*frameJump)
		{
			if (SR == 1.0)
			{
				if (procam == 1)
					status = ProCamMatching(DataPATH, nCams, frameID, 1, width, height, pwidth, pheight, nchannels, LKArg);
				else
					status = CamProMatching2(DataPATH, nCams, nPros, frameID, width, height, pwidth, pheight, nchannels, LKArg, ROI, false, flowVerification, triThresh, saveWarping, SimulationMode);
			}
			else
				status = SProCamMatching(DataPATH, nCams, frameID, width, height, pwidth, pheight, nchannels, LKArg, SR);
			cout << " .... finish with return value " << status << endl;
		}
	}
	else if (mode == 6)
	{
		int SepMode = atoi(argv[2]),//interleaving or flashing
			colorTexture = atoi(argv[3]), startID = atoi(argv[4]), stopID = atoi(argv[5]);
		//int SepMode = 1, colorTexture = 1, startID = 32, stopID = 32;

		LKArg.DIC_Algo = 3, LKArg.step = 1, LKArg.hsubset = 8, LKArg.npass = 1, LKArg.npass2 = 1, LKArg.searchRangeScale = 3, LKArg.searchRangeScale2 = 6, LKArg.searchRange = 8;
		LKArg.PSSDab_thresh = 0.04, LKArg.ZNCCThreshold = 0.975, LKArg.ssigThresh = 30.0;
		LKArg.ProjectorGsigma = 0.707;  //Important: this value may change in simulation or real modes.
		IlluminationFlowImages IlluminationImages(width, height, pwidth, pheight, nchannels, nCams, nPros, 1);
		IlluminationImages.frameJump = frameJump, IlluminationImages.iRate = 30;  //Important: iRate = interleaving rate

		int step = SepMode == 0 ? 2 : 1;
		for (int ii = startID; ii <= stopID; ii += step)
		{
			IllumTextSepDriver(DataPATH, TDataPATH, IlluminationImages, 0, LKArg, ii, SepMode, colorTexture, ROI, SimulationMode);
			IllumTextSepDriver(DataPATH, TDataPATH, IlluminationImages, 1, LKArg, ii, SepMode, colorTexture, ROI, SimulationMode);
			//IllumSepDriver(DataPATH, TDataPATH, IlluminationImages, LKArg, ii, mode, ROI, SimulationMode);
			TwoIllumTextSepDriver(DataPATH, TDataPATH, IlluminationImages, LKArg, ii, SepMode, ROI, true, colorTexture, SimulationMode);
		}

		return 0;
	}
	else if (mode == 7)
	{
		double alpha = atof(argv[2]), beta = atof(argv[3]), gamma = atof(argv[4]);
		int nchannels = atoi(argv[5]), matchingAlgo = atoi(argv[6]);
		int startID = atoi(argv[7]) - frameJump, stopID = atoi(argv[8]);//because input generated is [2, 2] instead of [1, 1]

		/*double alpha = 0.05, beta = 1.0, gamma = 1.0;
		int nchannels = 1, matchingAlgo = 3, startID = 12, stopID = 13;*/

		//Careful with input data
		int camID = 1, firstID = 1;
		LKArg.hsubset = 10, LKArg.DIC_Algo = matchingAlgo;
		LKArg.PSSDab_thresh = 0.025, LKArg.ZNCCThreshold = 0.975;

		SVSRP srp;
		srp.Precomputed = false, srp.SRF = 1.0, srp.maxOuterIter = 10, srp.thresh = 0.05, srp.Rstep = 0.5, srp.CheckerDisk = 16;
		srp.alpha = alpha, srp.beta = beta, srp.gamma = gamma, srp.dataScale = 10.0, srp.regularizationScale = 10.0;

		IlluminationFlowImages IlluminationImages(width, height, pwidth, pheight, nchannels, nCams, nPros, nframes);
		IlluminationImages.frameJump = frameJump;
		//Input data end

		char wording[200]; sprintf(wording, "SVSR_%d", startID);
		google::InitGoogleLogging(wording);

		sprintf(wording, "Run depth upsampling on (%d, %d) with (alpha, beta, gamma) = %.2f, %.2f, %.2f\n", startID, startID + 1, srp.alpha, srp.beta, srp.gamma);
		cout << wording;

		int returnvalue;
		for (int frameID = startID; frameID <= stopID; frameID += 2 * frameJump)
		{
			returnvalue = SVSR_Driver(frameID, IlluminationImages, LKArg, srp, DataPATH);
			if (returnvalue != 0)
				return returnvalue;
		}
		return 0;
	}
	else if (mode == 8)
	{
		int startID = atoi(argv[1 + 2]) - 1, stopID = atoi(argv[1 + 3]); //because input generated is [2, 2] instead of [1, 1]
		//int startID = 1, stopID = 2; 

		cout << "Run depth propagation on [" << startID << ", " << stopID << "]" << endl;
		return DepthProgogationDriver(startID, stopID - startID, DataPATH);
	}
	else if (mode == 9)
	{
		int detection = atoi(argv[2]), startID = atoi(argv[3]), stopID = atoi(argv[4]);
		//int detection = 1, startID = 2, stopID = 2;

		cout << "Run corner detection on [" << startID << ", " << stopID << "]" << endl;

		if (detection == 1)
		{
			int numThreads = (stopID - startID) / 2 + 1; numThreads = numThreads > MAXTHREADS ? MAXTHREADS : numThreads;
			omp_set_num_threads(numThreads);

			for (int jj = 0; jj < nCams; jj++)
#pragma omp parallel for
				for (int ii = startID; ii <= stopID; ii += 2 * frameJump)
					CheckerDetectionCorrespondenceDriver(jj, nCams, nPros, ii, pwidth, pheight, width, height, DataPATH);
		}
		else if (detection == 2)
		{
			int numThreads = (stopID - startID) / 2 + 1; numThreads = numThreads > MAXTHREADS ? MAXTHREADS : numThreads;
			omp_set_num_threads(numThreads);

#pragma omp parallel for
			for (int ii = startID; ii <= stopID; ii += 2 * frameJump)
				DetectTrackSparseCornersForFlow(ii, nCams, nPros, frameJump, width, height, pwidth, pheight, DataPATH);
		}
		return 0;
	}
	else if (mode == 10)
	{
		int startID = atoi(argv[1 + 2]),
			stopID = atoi(argv[1 + 3]); //because input generated is [2, 2] instead of [1, 1]

		//note: it is important to set nchannels to 1 and set the ZNCC low!!!
		LKArg.hsubset = 7, LKArg.DIC_Algo = 1;
		LKArg.PSSDab_thresh = 0.01, LKArg.ZNCCThreshold = 0.97;
		nchannels = 1;
		for (int ii = startID; ii <= stopID; ii++)
		{
			cout << "Working on frame " << ii << endl << endl;;
			ConvertStereoDepthToProjector(DataPATH, ii, 1, 2, width, height, pwidth, pheight, nchannels, LKArg);
		}
	}
	else if (mode == 11)
	{
		int	resolution = 8,//atoi(argv[1+2]), 
			noise = 2;//atoi(argv[1+3]);
		double SR = 1.0;

		CPoint Dimen[9]; Dimen[0].x = 1024, Dimen[0].y = 768, Dimen[1].x = 1184, Dimen[1].y = 888, Dimen[2].x = 1672, Dimen[2].y = 1256;
		Dimen[3].x = 2048, Dimen[3].y = 1536, Dimen[4].x = 2364, Dimen[4].y = 1772, Dimen[5].x = 2644, Dimen[5].y = 1984;
		Dimen[6].x = 2896, Dimen[6].y = 2172, Dimen[7].x = 3128, Dimen[7].y = 2348, Dimen[8].x = 3344, Dimen[8].y = 2508;

		return CompareWithGroundTruthFlow(DataPATH, 1, nchannels, Dimen[resolution].x, Dimen[resolution].y, pwidth, pheight, SR, resolution, noise);
	}
	else if (mode = 12)
	{
		int startID = 1,//atoi(argv[3])-1,
			stopID = 2, //atoi(argv[4]),
			SuperRes = 2;

		for (int ii = startID; ii <= stopID; ii++)
		{
			cout << "Working on frame " << ii << endl << endl;;
			CompareWithGroundTruth(DataPATH, DataPATH, ii, nCams, pwidth, pheight, SuperRes);
		}
	}
	else
	{
		int startID = atoi(argv[4]), stopID = atoi(argv[5]);
		{
			int detection = 1;
			cout << "Run corner detection on [" << startID << ", " << stopID << "]" << endl;

			int numThreads = (stopID - startID + 1) / 2; numThreads = numThreads > MAXTHREADS ? MAXTHREADS : numThreads;
			omp_set_num_threads(numThreads);
			if (detection == 1)
			{
				for (int jj = 0; jj < nCams; jj++)
#pragma omp parallel for
					for (int ii = startID; ii <= stopID; ii++)
						CheckerDetectionCorrespondenceDriver(jj, nCams, nPros, ii, pwidth, pheight, width, height, DataPATH);
			}
		}
	}

	return 0;
}
