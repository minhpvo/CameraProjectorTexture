#include "Ultility.h"
#include "Matrix.h"
#include "ImagePro.h"
//#include <cv.h>
#include <opencv2/opencv.hpp>
//#include <core/core.hpp>
//#include <highgui/highgui.hpp>

#define MAXTHREADS 8

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

using namespace cv;
using namespace std;


struct LKParameters
{
	//DIC_Algo: 
	//0 epipolar search with translation model
	//1 Affine model with epipolar constraint
	//2 translation model
	//3 Affine model without epipolar constraint
	bool checkZNCC;
	int step, hsubset, npass, npass2, searchRangeScale, searchRangeScale2, searchRange, DisplacementThresh, Incomplete_Subset_Handling, Convergence_Criteria, Analysis_Speed, IterMax, InterpAlgo, DIC_Algo, EpipEnforce;
	double ZNCCThreshold, PSSDab_thresh, ssigThresh, Gsigma, ProjectorGsigma;
};

struct TVL1Parameters
{
	bool useInitialFlow;
	int iterations, nscales, warps;
	double tau, lamda, theta, epsilon;
};

struct FlowScaleSelection
{
	int startS, stopS, stepS, stepIJ;
};

class SVSRP
{
public:
	SVSRP(): maxOuterIter(20), thresh(0.1), SRF(0.2), CheckerDisk(16), Rstep(1.0), alpha(0.1), dataScale(0.5), regularizationScale(2.0){};
	SVSRP(int mI, double threshhold, int cDist, double R, double al, double dS, double rS): maxOuterIter(mI), thresh(threshhold), CheckerDisk(cDist), Rstep(R), alpha(al), dataScale(dS), regularizationScale(rS){};
	~SVSRP(){};

	bool Precomputed;
	int maxOuterIter, CheckerDisk;
	double Rstep, SRF, thresh, alpha, beta, gamma, dataScale, regularizationScale;
};

class IJZ3
{
public:
	IJZ3() : i(0), j(0), z1(0.0), z2(0.0), z3(0.0){};
	IJZ3(double vi, double  vj, double vz1, double vz2, double vz3) : i(vi), j(vj), z1(vz1), z2(vz2), z3(vz3){};
	~IJZ3(){};

	double i, j, z1, z2, z3;
};
class DevicesInfo
{
public:
	DevicesInfo(int nC = 1, int nP = 1)
	{
		nCams = nC, nPros = nP;
		K = new double [(nCams+nPros)*9];
		iK = new double [(nCams+nPros)*9];
		P = new double [(nCams+nPros-1)*12];
		RTx1 = new double [(nCams+nPros-1)*12];
		RT1x = new double [(nCams+nPros-1)*12];
		FmatPC = new double [nCams*nPros*9];
		L = new double[3*nCams*nPros];
		distortion = new double[(nCams+nPros)*13];
	}
	~DevicesInfo()
	{
		delete []K;
		delete []iK;
		delete []P;
		delete []RTx1;
		delete []RT1x;
		delete []FmatPC;
		delete []distortion;
	}

	int nCams, nPros;
	double *K, *iK, *P, *RTx1, *RT1x, *FmatPC, *L, *distortion;

};
class FlowVect
{
public: 
	FlowVect(int width, int height, int nC) 
	{
		nframes = 2;
		nCams = nC;
		C21x = new float [width*height*nframes*nCams];
		C21y = new float [width*height*nframes*nCams];
		C12x = new float [width*height*nframes*nCams];
		C12y = new float [width*height*nframes*nCams];
	}

	~FlowVect()
	{
		delete []C12x;
		delete []C12y;
		delete []C21x;
		delete []C21y;
	}

	int nCams, nframes;
	float *C12x;
	float *C12y;
	float *C21x;
	float *C21y;
};
class IlluminationFlowImages
{
public:
	IlluminationFlowImages(int w, int h, int pw, int ph, int numChannels, int nC, int nP, int nf)
	{
		width = w, height = h, pwidth = pw, pheight = ph, nchannels = numChannels, nCams = nC, nPros = nP, nframes = nf;
		Img = new double[width*height*nCams*nchannels*nframes];
		Para = new double[width*height*nCams*nchannels*nframes];
		PImg = new double[pwidth*pheight*nPros*nchannels];
		PPara = new double[pwidth*pheight*nPros*nchannels];
	}

	~IlluminationFlowImages()
	{
		delete []Img;
		delete []Para;
		delete []PImg;
		delete []PPara;
	}

	int width, height, pwidth, pheight, nchannels, nCams, nPros, nframes, frameJump, iRate;
	double *Img;
	double *Para;
	double *PImg;
	double *PPara;
};


bool SetUpDevicesInfo(DevicesInfo &DInfo, char *PATH);
void LensDistortion_Point(CPoint2 &img_point, double *camera, double *distortion);
void LensDistortion_PointL(CPoint2L &img_point, double *camera, double *distortion);
void CC_Calculate_xcn_ycn_from_i_j(double i, double j, double &xcn, double &ycn, double *A, double *K, int Method);

void Undo_distortion(CPoint2 &uv, double *camera, double *distortion);
void Stereo_Triangulation2(CPoint2 *pts1, CPoint2 *pts2, double *P1, double *P2, CPoint3 *WC, int npts = 1);
void Triplet_Triangulation2(CPoint2 *pts1, CPoint2 *pts2, CPoint2 *pts3, double *P1, double *P2, double* P3, CPoint3 *WC, int npts = 1);
void NviewTriangulation(CPoint2 *pts, double *P, CPoint3 *WC, int nview, int npts = 1, double *Cov = 0, double *A = 0, double *B = 0);
//void Project3DtoImg(CPoint3 WC, CPoint2 *pts, double *P, double *camera, int nCam);
double Compute_Homo(CPoint5 *uvxy, int planar_pts, Matrix&Homo, CPoint5 *s_pts = 0, double *LinParaHomo = 0);
double Compute_AffineHomo(CPoint2 *From, CPoint2 *To, int npts, double *Affine, double *A = NULL, double *B = NULL);

void synthesize_square_mask(double *square_mask_smooth, int *pattern_bi_graylevel, int Pattern_size, double sigma, int flag, bool OpenMP = false);
double TMatching(double *RefPara, double *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, CPoint2 PR, CPoint2 PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch = 0, double *ShapePara = 0, double *oPara = 0, double *Timg = 0, double *T = 0);
void TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, char *Image, int width, int height, CPoint POI, int search_area, double thresh, double &zncc);
void TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, double *Image, int width, int height, CPoint POI, int search_area, double thresh, double &zncc);
int TMatchingCoarse(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, CPoint2 &POI, int search_area, double thresh, double &zncc, int InterpAlgo, double *InitPara = 0, double *maxZNCC = 0);
double TMatchingFine_ZNCC(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, CPoint2 &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *Znssd_reqd = 0);
bool TMatchingFine(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, CPoint2 &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *ShapePara = 0, double *maxZNCC = 0);
void DetectCornersHarris(char *img, int width, int height, CPoint2 *HarrisC, int &npts, double sigma, double sigmaD, double thresh, double alpha, int SuppressType = 1, double AMN_thresh = 0.9);
void RunCornersDetector(CPoint2 *CornerPts, int *CornersType , int &nCpts, double *Img, double *IPara, int width, int height, vector<double>PatternAngles, int hsubset1 = 6, int hsubset2 = 8, int searchArea = 1, double ZNCCCoarseThresh = 0.7, double ZNCCThresh = 0.9, int InterpAlgo = 5);

int  CheckerDetectionCorrespondenceDriver(int CamID, int nCams, int nPros, int frameID, int pwidth, int pheight, int width, int height, char *PATH);
void TrackMarkersFlow(char *Img1, char *Img2, double *IPara1, double *IPara2, int width, int height, CPoint2 *Ppts, CPoint2 *Cpts1, CPoint2 *Cpts2, int nCpts, double *Fmat1x, double *flowx, double *flowy, int hsubset, int advancedTech, int ConvCriteria, double ZNCCThresh, int InterpAlgo);
int DetectTrackMarkersandCorrespondence(char *PATH, CPoint2 *Pcorners, int *PcorType, double *PPara, DevicesInfo &DInfo, LKParameters Flowarg, double *CamProScale, int checkerSize, int nPpts, int PSncols, CPoint *ProjectorCorressBoundary, double EpipThresh, int nframes, int startFrame, int CamID, int width, int height, int pwidth, int pheight);
void CorrespondenceByTrackingMarkers(char *PATH, CPoint2 *Pcorners, int *PcorType, double *PPara, DevicesInfo &DInfo, LKParameters Flowarg, double *CamProScale, int numPatterns, int checkerSize, int nPpts, int PSncols, CPoint *ProjectorCorressBoundary, double EpipThresh, int nframes, int startFrame, int CamID, int width, int height, int pwidth, int pheight);
int CheckerDetectionTrackingDriver(int CamID, int nCams, int nPros, int frameJump, int startF, int nframes, int pwidth, int pheight, int width, int height, char *PATH, bool byTracking);
void CleanValidCheckerStereo( const int nCams, int startF, int nframes,CPoint2 *IROI, CPoint2 *PROI, char *PATH);
void CleanValidChecker( const int nCams, int nPros, int frameJump, int startF, int nframes,CPoint2 *IROI, CPoint2 *PROI, char *PATH);
void TwoDimTriangulation(int startF, int nCams, int nPros, int frameJump, int nframes, int width, int height, char *PATH);

double ComputeMeanAbsGrad(double *Para, int x, int y, int hsubset, int width, int height, int nchannels, int InterpAlgo);
double LKtranslation(double *src, double *dst, CPoint *Pts, int hsubset, int width, int height);
double SearchLK(CPoint2 From, CPoint2 &Target, double *Img1Para, double *Img2Para, int nChannels, int width1, int height1, int width2, int height2, LKParameters LKArg,  double *TemplateImg = 0, double *ZNCCStorage = 0, double *iWp = 0, double *direction = 0, double* iCovariance = 0);
double EpipSearchLK(CPoint2 *dPts, double *EpiLine, double *Img1, double *Img2, double *Para1, double *Para2, int nchannels, int width1, int height1, int width2, int height2, LKParameters Flowarg, double *TemplateImg, double *ZNCCStorage, double *TarPatch, double *iWp = 0,double *iCovariance = 0);

double BruteforceMatchingEpipolar(CPoint2 From, CPoint2 &Target, double *direction, int maxdisparity, double *Img1, double *Img2, double *Img2Para, int nchannels, int width1, int height1, int width2, int height2, LKParameters LKArg, double *tPatch, double *tZNCC, double *Znssd_reqd);
int MatchingCheck(char *Img1, char *Img2, float *WarpingParas, LKParameters LKArg, double scale, int nchannels, int width1, int height1, int width2, int height2);
int MatchingCheck2(float *WarpingParas1, float *WarpingParas2, int width1, int height1, int width2, int height2, bool AffineShape = false);
void DIC_DenseScaleSelection(int *Scale, float *Fu, float *Fv, double *Img1, double *Img2, int width, int height, LKParameters Flowarg, FlowScaleSelection ScaleSel, double flowThresh, CPoint *DenseFlowBoundary = NULL);
int GreedyMatching(char *Img1, char *Img2, CPoint2 *displacement, bool *lpROI_calculated, bool *tROI, CPoint2 *SparseCorres1, CPoint2 *SparseCorres2, int nSeedPoints, LKParameters LKarg, int nchannels, int width1, int height1, int width2, int height2, double Scale,double *Epipole = 0, float *WarpingParas = 0, double *Pmat = 0, double *K = 0, double *distortion = 0, double triThresh = 2.0);
int SGreedyMatching(char *Img1, char *Img2, float *displacement, bool *lpROI_calculated, bool *tROI, int *flowhsubset, CPoint2 *SparseCorres1, CPoint2 *SparseCorres2, int nSeedPoints, LKParameters Flowarg, double SR, int nchannels, int width1, int height1, int width2, int height2, double Scale, double *Epipole = 0);

void WarpImageFlow(float *flow, unsigned char *wImg21, unsigned char *Img2, int width, int height, int nchannels, int InterpAlgo, bool removeStatic);
void WarpImageFlowDouble(float *flow, double *wImg21, double *Img2, int width, int height, int nchannels, int InterpAlgo, bool removeStatic);
int WarpImageFlowDriver(char *Fin, char *Fout, char *FnameX, char *FnameY, int nchannels, int Gsigma, int InterpAlgo, bool removeStatic);
int TVL1OpticalFlowDriver(int frameID, int frameJump, int nCams, int width, int height, char *PATH, TVL1Parameters argGF, bool forward, bool backward);

void NonMinimalSupression1D(double *src, int *MinEle, int &nMinEle, int halfWnd, int nele);
double ComputeDepth3DBased(double hypoDepth, double *rayDirect, double *C1Pd, double *C2Pd, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, int *DenseScale, int advancedTech, int ConvCriteria, double znccThresh, double *C1center, double *C2center, FlowVect &ParaOFlow, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo, double &depth1, double &depth3, double *TCost, bool Intersection = false);
double GoldenPointDepth3DBased(IJZ3 &optimDepth, double hdepth, double ldepth, double *rayDirect, double *C1Pd, double *C2Pd, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &ParaOFlow, int *DenseScale, int advancedTech, int ConvCriteria, double znccThresh, double *C1center, double *C2center, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo);
double BrentPointDepth3DBased(IJZ3 &optimDepth, double ldepth, double hdepth, double *rayDirect, double *C1Pd, double *C2Pd, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &ParaOFlow, int *DenseScale, int advancedTech, int ConvCriteria, double znccThresh, double *C1center, double *C2center, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo, double FractError = 1.0e-6);
double ClosetPointToLine(CPoint2 &cp, CPoint2 p, double *line);
void ProjectDepthFlowToImage(double hypoDepth, double *CxPd, DevicesInfo &DInfo, CPoint2 *Cxpos);
double ProjectedDepthCost(double depth, double *CxPd, DevicesInfo &DInfo, CPoint2 *FlowEndpts, double *iCov, int TwoToThree);
double GoldenDepthCost(double &depth, double ldepth, double hdepth, double *CxPd, DevicesInfo &DInfo, CPoint2 *FlowEndpts, double *CxiCov, int TwoToThree);
double BrentDepthCost(double &depth, double ldepth, double hdepth, double *CxPd, DevicesInfo &DInfo, CPoint2 *FlowEndpts, double *CxiCov, int TwoToThree);
double ComputeDepth2DBasedML(IJZ3 &hypoD, double *rayDirect, double *CxPd, double *Cxcenter, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &ParaOFlow, int advancedTech, int hsubset, int ConvCriteria, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo, double *TCost, int searchTech);
double GoldenPointDepth2DBasedML(IJZ3 &optimDepth, double hdepth, double ldepth, double *rayDirect, double *CxPd, double *Cxcenter, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &ParaOFlow, int advancedTech, int hsubset, int ConvCriteria, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo);
double BrentPointDepth2DBasedML(IJZ3 &optimDepth, double ldepth, double hdepth, double *rayDirect, double *CxPd, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &ParaOFlow, int advancedTech, int hsubset, int ConvCriteria, double *Cxcenter, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo);

double PointDepth1DSearchTemporalLK(int currentID, int *notworking, CPoint3 *XYZ, IJZ3 &depth, double *rayDirect, DevicesInfo &DInfo, IlluminationFlowImages &Fimgs, int *DenseScale, int advancedTech, int ConvCriteria, double znccThresh, double *C1center, double *C2center, FlowVect &ParaOFlow, int cwidth, int cheight, double thresh, int SearchTechnique, int InterpAlgo);
void DepthFlow(IJZ3 *depth, DevicesInfo &DInfo, FlowVect &ParaOFlow, IlluminationFlowImages &Fimgs, int hsubset, int advancedTech, int ConvCriteria, double znccThresh, int cwidth, int cheight, int npts, int SearchTechnique, int InterpAlgo);
void FlowDepthOptimization(DevicesInfo &DInfo, CPoint2 *Ccorners, CPoint2 *Pcorners, int Cnpts, int *triangleList, int ntriangles, double step, IlluminationFlowImages &Fimgs, FlowVect &flow, FlowVect &ParaOFlow, int *DenseScale, int advancedTech, int ConvCriteria, double znccThresh, int width, int height, int pwidth, int pheight, int SearchTechnique, int InterpAlgo, int frameID, char *ResultPATH);

int SVSR_solver( IlluminationFlowImages &Fimgs, FlowVect &flow, double *depth, CPoint *IJ, int depthW, int depthH, int ImgW, int ImgH, DevicesInfo &DInfo, LKParameters &Flowarg);
int SVSR(IlluminationFlowImages &Fimgs, FlowVect &Paraflow, float *Fp, float *Rp, int *flowhsubset, double *depth, bool *TextureMask, CPoint2 *IJ, int *ProCamMask, double *ProCamDepth, CPoint2 *Pcorners, CPoint2 *Ccorners, int *CPindex, int *nPpts, int *triangleList, int *ntriangles, DevicesInfo &DInfo, SVSRP &srp, LKParameters LKArg, double *ProCamScale, CPoint2 *DROI, int ImgW, int ImgH, int PimgW, int PimgH, char *DataPath, int frameID);
int DepthPropagation(IlluminationFlowImages &Fimgs, float *flowX, float *flowY, double *Intdepth, double *PropDepth, DevicesInfo &DInfo, LKParameters Flowarg, int ImgW, int ImgH, int PimgW, int PimgH, CPoint2 *DROI);

void ConvertProDepthToCamProMatching(DevicesInfo &DInfo, float *PDepth, int *CamProMatch, int width, int height, int pwidth, int pheight, int camID);
void FlowFieldsWarpring(float *TextureFlow, float *IlluminationFlow, double *TextureImagePara, double *IlluminationImagePara, int width, int height, int InterpAlgo, char *Fname = 0);
int DetermineTextureImageRegion2(double *TextureImage, bool *mask, int width, int height, double sigma, int Ithresh);
void ComputeIlluminationImage(IlluminationFlowImages &Fimgs, DevicesInfo &DInfo, float *Depth, double *Illumnation, int InterpAlgo, int camID);

int MultiViewGeoVerify(CPoint2 *Pts, double *Pmat, double *K, double *distortion, bool *PassedPoints, int width, int height, int pwidth, int pheight, int nviews, int npts = 1, double thresh = 2.0, 
					   CPoint2 *apts = 0, CPoint2 *bkapts = 0, int *DeviceMask = 0, double *tK = 0, double *tdistortion = 0, double *tP = 0, double *A = 0, double *B = 0);
int CamProGeoVerify(double cx, double cy, CPoint2 *Pxy, CPoint3 &WC, double *P1mat, DevicesInfo &DInfo, int width, int height, int pwidth, int pheight, int pview, double thresh = 2.0);
double ComputeSSIG(double *Para, int x, int y, int hsubset, int width, int height, int nchannels, int InterpAlgo);
void UpdateIllumTextureImages(char *PATH, bool silent, int frameID, int mode, int nPros, int proOffset, int width, int height, int pwidth, int pheight, int nchannels, int InterpAlgo, double *ParaIllumSource, float *ILWarping, double *ParaSourceTexture = NULL, float *TWarping = NULL, bool *ROI = NULL, bool color = false, int per = 0);
int TextImageMatchSource(char *PATH, int frameID, LKParameters LKArg, int nchannels, double scale);

int IllumTextureSeperation(int frameID, int proID, char *PATH, char *TPATH, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, DevicesInfo &DInfo, float *ILWarping, float *TWarping, float *preTWapring, float *PhotoAdj, int *SepType, int *PrecomSearchR, LKParameters LKArg, int mode, bool *cROI, int part, bool Simulation);
int IllumsReoptim(int frameID, char *PATH,  char *TPATH, IlluminationFlowImages &Fimgs, DevicesInfo &DInfo, float *ILWarping, float *PhotoAdj, int *reoptim, int *SeedType, float *SSIG, LKParameters LKArg, int mode, bool *cROI, int part, bool deletePoints);
int IllumSeperation(int frameID, char *PATH,  char *TPATH, IlluminationFlowImages &Fimgs, DevicesInfo &DInfo, float *ILWarping, float *PhotoAdj, int *SeedType, float *SSIG, int *PrecomSearchR, LKParameters LKArg, int mode, bool *cROI, int part);
int TwoIllumTextSeperation(int frameID, char *PATH,  IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, DevicesInfo &DInfo, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *SeedType, int *PrecomSearchR, LKParameters LKArg, int mode, bool *cROI, int part, int SepMode);
