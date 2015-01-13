#include <cmath>
#include <cstdlib>

void filter1D_row_Double(double *kernel, int k_size, double *in, double *out, int width, int height);
void filter1D_row(double *kernel, int k_size, char *in, double *out, int width, int height);
void filter1D_col(double *kernel, int k_size, double *in, double *out, int width, int height, double &i_max);
void Gaussian_smooth(char* data, double* out_data, int height, int width, double max_i , double sigma);
void Gaussian_smooth_Double(double* data, double* out_data, int height, int width, double max_i, double sigma);

void Generate_Para_Spline(double *Image, double *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(float *Image, float *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(char *Image, double *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(unsigned char *Image, double *Para, int width, int height, int Interpolation_Algorithm);
void Generate_Para_Spline(int *Image, double *Para, int width, int height, int Interpolation_Algorithm);
void Get_Value_Spline(float *Para, int width,int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm);
void Get_Value_Spline(double *Para, int width,int height, double X, double Y, double *S, int S_Flag, int Interpolation_Algorithm);

//Bicubic Spline
void Generate_Para_BiCubic_Spline_Double(double *Image, double *Para, int width, int height);
void Get_Value_BiCubic_Spline(double *Para, int width_ex, int height_ex, double X, double Y, double *S, int S_Flag);

//Linear interpolation
int LinearInterp(int *data, int width, int height, double u, double v);
double BilinearInterp(double *data, int width, int height, double x, double y);



void Average_Filtering_All(char *lpD, int width, int height, int ni, int HSize, int VSize);
void MConventional_PhaseShifting(char *lpD, char *lpPBM, double* lpFO, int nipf, int length, int Mask_Threshold, double *f_atan2);
void DecodePhaseShift2(char *Image, char *PBM, double *PhaseUW, int width, int height, int *frequency, int nfrequency, int sstep, int LFstep, int half_filter_size, int m_mask);

void RemoveNoiseMedianFilter(float *data, int width, int height, int ksize, float thresh);
void RemoveNoiseMedianFilter(double *data, int width, int height, int ksize, float thresh, float *fdata);

void TransformImage(double *oImg, double *iImg, double *Trans, int width, int height, int nchannels, int interpAlgo, double *iPara);
