#pragma once

#include "Ultility.h"
#include "ImagePro.h"
#include "Matrix.h"

using namespace cv;
using namespace std;

#define Pi 3.141592653589793

int MyFtoI(double W)
{
	if(W>=0.0)
		return (int)(W+0.5);
	else
		return (int)(W-0.5);

	return 0;
}
bool IsNumber(double x) 
{
	// This looks like it should always be true, but it's false if x is a NaN.
	return (x == x); 
}
bool IsFiniteNumber(double x) 
{
	return (x <= DBL_MAX && x >= -DBL_MAX); 
} 

double gaussian_noise(double mean, double std)
{	
	double u1 = 0.0, u2 = 0.0;
	while(abs(u1) < DBL_EPSILON || abs(u2) < DBL_EPSILON) //avoid 0.0 case since log(0) = inf
	{
		u1= 1.0 * rand() / RAND_MAX;
		u2 = 1.0 * rand() / RAND_MAX;
	}


	double normal_noise = sqrt(-2.0 * log(u1)) * cos(2.0 * Pi * u2);

	return mean + std * normal_noise;
}

void normalize(double *x, int dim)
{
	double tt = 0;
	for(int ii=0; ii<dim; ii++)
		tt += x[ii]*x[ii];
	tt = sqrt(tt);
	for(int ii=0; ii<dim; ii++)
		x[ii] = x[ii]/tt;
	return;
}
float MeanArray(float *data, int length)
{
	double mean = 0.0;
	for (int ii = 0; ii < length; ii++)
		mean += data[ii];
	return mean / length;
}
double MeanArray(double *data, int length)
{
	double mean = 0.0;
	for (int ii = 0; ii < length; ii++)
		mean += data[ii];
	return mean / length;
}
double VarianceArray(double *data, int length, double mean)
{
	if (mean == NULL)
		mean = MeanArray(data, length);

	double var = 0.0;
	for (int ii = 0; ii < length; ii++)
		var += pow(data[ii] - mean, 2);
	return var / (length - 1);
}
double MeanArray(vector<double>&data)
{
	double mean = 0.0;
	for (int ii = 0; ii < data.size(); ii++)
		mean += data[ii];
	return mean / data.size();
}
double VarianceArray(vector<double>&data, double mean)
{
	if (mean == NULL)
		mean = MeanArray(data);

	double var = 0.0;
	for (int ii = 0; ii < data.size(); ii++)
		var += pow(data[ii] - mean, 2);
	return var / (data.size() - 1);
}
double norm_dot_product(double *x, double *y, int dim)
{
	double nx = 0.0, ny = 0.0, dxy = 0.0;
	for(int ii=0; ii<dim; ii++)
	{
		nx += x[ii]*x[ii];
		ny += y[ii]*y[ii];
		dxy += x[ii]*y[ii];
	}
	double radian = dxy/sqrt(nx*ny);

	return radian;
}
void cross_product(double *x, double *y, double *xy)
{
	xy[0] = x[1]*y[2] - x[2]*y[1];
	xy[1] = x[2]*y[0] - x[0]*y[2];
	xy[2] = x[0]*y[1] - x[1]*y[0];

	return;
}
void mat_invert(double* mat, double* imat, int dims)
{	
	if(dims == 3)
	{
		// only work for 3x3
		double a=mat[0],b=mat[1],c=mat[2],d=mat[3],e=mat[4],f=mat[5],g=mat[6],h=mat[7],k=mat[8];
		double A=e*k-f*h,B=c*h-b*k,C=b*f-c*e;
		double D=f*g-d*k,E=a*k-c*g,F=c*d-a*f;
		double G=d*h-e*g,H=b*g-a*h,K=a*e-b*d;
		double DET=a*A+b*D+c*G;
		imat[0]=A/DET,imat[1]=B/DET,imat[2]=C/DET;
		imat[3]=D/DET,imat[4]=E/DET,imat[5]=F/DET,
			imat[6]=G/DET,imat[7]=H/DET,imat[8]=K/DET;
	}
	else
	{
		Mat inMat = Mat(dims, dims, CV_64FC1, mat);
		Mat outMat = inMat.inv(DECOMP_SVD);
		for(int jj=0; jj<dims; jj++)
			for(int ii=0; ii<dims; ii++)
				imat[ii+jj*dims] = outMat.at<double>(jj,ii);
	}

	return;
}
void mat_mul(double *aa,double *bb, double *out,int rowa,int col_row,int colb)
{
	int ii,jj,kk;
	for (ii=0;ii<rowa*colb;ii++)
		out[ii]=0;

	for(ii=0;ii<rowa;ii++)
	{
		for(jj=0;jj<colb;jj++)
		{
			for( kk=0;kk<col_row;kk++)
				out[ii*colb+jj]+=aa[ii*col_row+kk]*bb[kk*colb+jj];
		}
	}

	return;		
}
void mat_add(double *aa,double *bb,double* cc,int row,int col, double scale_a, double scale_b)
{
	int ii,jj;

	for(ii=0;ii<row;ii++)
		for(jj=0;jj<col;jj++)
			cc[ii*col+jj] = scale_a*aa[ii*col+jj]+scale_b*bb[ii*col+jj];

	return;		
}
void mat_subtract(double *aa,double *bb,double* cc,int row,int col, double scale_a, double scale_b)
{
	int ii,jj;

	for(ii=0;ii<row;ii++)
		for(jj=0;jj<col;jj++)
			cc[ii*col+jj] = scale_a*aa[ii*col+jj]-scale_b*bb[ii*col+jj];

	return;		
}
void mat_transpose(double *in,double *out,int row_in,int col_in)
{
	int ii,jj;
	for(jj=0;jj<row_in;jj++)
		for(ii=0;ii<col_in;ii++)
			out[ii*row_in+jj]=in[jj*col_in+ii];
	return;
}
void mat_completeSym(double *mat, int size, bool upper)
{
	if(upper)
	{
		for(int jj=0; jj<size; jj++)
			for(int ii=jj; ii<size; ii++)
				mat[jj+ii*size] = mat[ii+jj*size];
	}
	else
	{
		for(int jj=0; jj<size; jj++)
			for(int ii=jj; ii<size; ii++)
				mat[ii+jj*size] = mat[jj+ii*size];
	}
	return;
}
void Rodrigues_trans(double *RT_vec, double *R_mat, bool vec2mat, double *dR_dm)
{

	if(vec2mat== true)
	{
		int ii;
		double n1 = RT_vec[0], n2 = RT_vec[1], n3 = RT_vec[2]; 
		double phi = sqrt(n1*n1+n2*n2+n3*n3);

		if(phi <DBL_EPSILON)
		{
			R_mat[0] = 1.0, R_mat[1] = 0.0, R_mat[2] = 0.0;
			R_mat[3] = 0.0, R_mat[4] = 1.0, R_mat[5] = 0.0;
			R_mat[6] = 0.0, R_mat[7] = 0.0, R_mat[8] = 1.0;

			if(dR_dm == NULL)
				return;

			dR_dm[0] = 0.0, dR_dm[1] = 0.0, dR_dm[2] = 0.0;
			dR_dm[3] = 0.0, dR_dm[4] = 0.0, dR_dm[5] = 1.0;
			dR_dm[6] = 0.0, dR_dm[7] = -1.0, dR_dm[8] = 0.0;
			dR_dm[9] = 0.0, dR_dm[10] = 0.0, dR_dm[11] = -1.0;
			dR_dm[12] = 0.0, dR_dm[13] = 0.0, dR_dm[14] = 0.0;
			dR_dm[15] = 1.0, dR_dm[16] = 0.0, dR_dm[17] = 0.0;
			dR_dm[18] = 0.0, dR_dm[19] = 1.0, dR_dm[20] = 0.0;
			dR_dm[21] = -1.0, dR_dm[22] = 0.0, dR_dm[23] = 0.0;
			dR_dm[24] = 0.0, dR_dm[25] = 0.0, dR_dm[26] = 0.0;
		}
		else
		{
			n1 = n1/phi, n2 = n2/phi, n3 = n3/phi;

			R_mat[0] = n1*n1*(1.0-cos(phi)) + cos(phi);
			R_mat[1] = n1*n2*(1.0-cos(phi)) - n3*sin(phi);
			R_mat[2] = n1*n3*(1.0-cos(phi)) + n2*sin(phi);
			R_mat[3] = n1*n2*(1.0-cos(phi)) + n3*sin(phi);
			R_mat[4] = n2*n2*(1.0-cos(phi)) + cos(phi);
			R_mat[5] = n2*n3*(1.0-cos(phi)) - n1*sin(phi);
			R_mat[6] = n1*n3*(1.0-cos(phi)) - n2*sin(phi);
			R_mat[7] = n2*n3*(1.0-cos(phi)) + n1*sin(phi);
			R_mat[8] = n3*n3*(1.0-cos(phi)) + cos(phi);

			if(dR_dm == NULL)
				return;

			double M[] = {	2.0*n1,							0.0,						0.0,
				n2,								n1,							-sin(phi)/(1.0-cos(phi)),
				n3,								sin(phi)/(1.0-cos(phi)),	n1,
				n2,								n1,							sin(phi)/(1.0-cos(phi)),
				0.0,							2.0*n2,						0.0,
				-sin(phi)/(1.0-cos(phi)),		n3,							n2,
				n3,								-sin(phi)/(1.0-cos(phi)),	n1,
				sin(phi)/(1.0-cos(phi)),		n3,							n2,
				0.0,							0.0,						2.0*n3};

			for(ii=0; ii<27; ii++)
				M[ii] *= (1.0 - cos(phi));

			double X[] = {	1.0-n1*n1,	-n1*n2,		-n1*n3,
				-n1*n2,		1.0-n2*n2,	-n2*n3,
				-n1*n3,		-n2*n3,		1.0-n3*n3};

			for(ii=0; ii<9; ii++)
				X[ii] /= phi;

			double N[] = {	sin(phi) * (n1*n1 - 1.0),
				n1*n2*sin(phi) - n3*cos(phi),
				n1*n3*sin(phi) + n2*cos(phi),
				n1*n2*sin(phi) + n3*cos(phi),
				sin(phi)*(n2*n2-1.0),
				n2*n3*sin(phi) - n1*cos(phi),
				n1*n3*sin(phi) - n2*cos(phi),
				n2*n3*sin(phi) +n1*cos(phi),
				sin(phi)*(n3*n3 - 1.0)};

			double Y[] = {n1, n2 ,n3};

			double MX[27], NY[27];
			mat_mul(M, X, MX, 9, 3, 3);
			mat_mul(N, Y, NY, 9, 1, 3);
			mat_add(MX, NY, dR_dm, 9, 3);
		}
	}
	else
	{
		// this function is from OpenCV
		int ii;
		double U_[9], W_[9], V_[9], R_[9], rx, ry, rz, theta, s, c;
		Matrix R_Mat(3, 3);

		for(ii=0; ii<9; ii++)
			R_Mat[ii] = R_mat[ii];

		R_Mat.SVDcmp(3, 3, U_, W_, V_, CV_SVD_MODIFY_A | CV_SVD_V_T);
		mat_mul(U_, V_, R_, 3, 3, 3);

		rx = R_[7] - R_[5];
		ry = R_[2] - R_[6];
		rz = R_[3] - R_[1];

		s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
		c = (R_[0] + R_[4] + R_[8] - 1)*0.5;
		c = c > 1. ? 1. : c < -1. ? -1. : c;
		theta = acos(c);

		if( s < 1e-5 )
		{
			double t;

			if( c > 0 )
				rx = ry = rz = 0;
			else
			{
				t = (R_[0] + 1)*0.5;
				rx = sqrt(MAX(t,0.));
				t = (R_[4] + 1)*0.5;
				ry = sqrt(MAX(t,0.))*(R_[1] < 0 ? -1. : 1.);
				t = (R_[8] + 1)*0.5;
				rz = sqrt(MAX(t,0.))*(R_[2] < 0 ? -1. : 1.);
				if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R_[5] > 0) != (ry*rz > 0) )
					rz = -rz;
				theta /= sqrt(rx*rx + ry*ry + rz*rz);
				rx *= theta;
				ry *= theta;
				rz *= theta;
			}
		}
		else
		{
			double vth = 1/(2*s);
			vth *= theta;
			rx *= vth; ry *= vth; rz *= vth;
		}

		RT_vec[0] = rx, RT_vec[1] = ry, RT_vec[2] = rz;
	}

	return;
}

void LS_Solution_Double(double *lpA, double *lpB, int m, int n)
{
	if(m==n)
	{
		QR_Solution_Double(lpA, lpB, n, n);
		return;
	}

	int i, j, k, n2=n*n;
	double *A = new double[n2];
	double *B = new double[n];

	for(i=0; i<n2; i++)
		*(A+i) = 0.0;
	for(i=0; i<n; i++)
		*(B+i) = 0.0;

	for(k=0; k<m; k++)
	{
		for(j=0; j<n; j++)
		{
			for(i=0; i<n; i++)
			{
				*(A+j*n+i) += (*(lpA+k*n+i))*(*(lpA+k*n+j));
			}

			*(B+j) += (*(lpB+k))*(*(lpA+k*n+j));
		}
	}

	QR_Solution_Double(A, B, n, n);

	for(i=0; i<n; i++)
		*(lpB+i) = *(B+i);

	delete []B;
	delete []A;
	return;
}
void QR_Solution_Double(double *lpA, double *lpB, int m, int n)
{
	if(m>3000)
	{
		LS_Solution_Double(lpA, lpB, m, n);
		return;
	}

	m_TemplateClass_1<double> m_TempClass;
	m_TempClass.QR_Solution(lpA, lpB, m, n);
	return;
}

void Quick_Sort_Int(int * A, int *B, int low, int high)
{
	m_TemplateClass_1<int> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}
void Quick_Sort_Float(float * A, int *B, int low, int high)
{
	m_TemplateClass_1<float> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}

void Quick_Sort_Double(double * A, int *B, int low, int high)
{
	m_TemplateClass_1<double> m_TempClass;
	m_TempClass.Quick_Sort(A, B, low, high);
	return;
}

bool in_polygon(double u, double v, CPoint2 *vertex, int num_vertex)
{
	int ii;
	bool position;
	double pi=3.1415926535897932384626433832795;

	for(ii=0; ii<num_vertex; ii++)
	{
		if(abs(u-vertex[ii].x)<0.01 && abs(v-vertex[ii].y)<0.01)
			return 1;
	}
	double dot = (u-vertex[0].x)*(u-vertex[num_vertex-1].x) + (v-vertex[0].y)*(v-vertex[num_vertex-1].y);
	double square1 = (u-vertex[0].x)*(u-vertex[0].x) + (v-vertex[0].y)*(v-vertex[0].y);
	double square2 = (u-vertex[num_vertex-1].x)*(u-vertex[num_vertex-1].x) + (v-vertex[num_vertex-1].y)*(v-vertex[num_vertex-1].y);
	double angle = acos(dot/sqrt(square1*square2));

	for(ii=0; ii<num_vertex-1; ii++)
	{
		dot = (u-vertex[ii].x)*(u-vertex[ii+1].x) + (v-vertex[ii].y)*(v-vertex[ii+1].y);
		square1 = (u-vertex[ii].x)*(u-vertex[ii].x) + (v-vertex[ii].y)*(v-vertex[ii].y);
		square2 = (u-vertex[ii+1].x)*(u-vertex[ii+1].x) + (v-vertex[ii+1].y)*(v-vertex[ii+1].y);

		angle += acos(dot/sqrt(square1*square2));
	}

	angle = angle*180/pi;
	if (fabs(angle-360)<=2.0)
		position =1;
	else
		position=0;

	return position;
}

bool myImgReader(char *fname,  unsigned char *Img, int width, int height, int nchannels)
{
	Mat view = imread(fname, nchannels==1?0:1);
	if(view.data == NULL)
	{
		cout<<"Cannot load: "<<fname<<endl;
		return false;
	}
	int length = width*height;
	for(int kk=0; kk<nchannels; kk++)
	{
		for(int jj=0; jj<height; jj++)
			for(int ii=0; ii<width; ii++)
				Img[ii+jj*width+kk*length] = view.data[nchannels*ii+(height-1-jj)*nchannels*width+kk];
	}

	return true;
}
bool myImgReader(char *fname,  float *Img, int width, int height, int nchannels)
{
	Mat view = imread(fname, nchannels==1?0:1);
	if(view.data == NULL)
	{
		cout<<"Cannot load: "<<fname<<endl;
		return false;
	}
	int length = width*height;
	for(int kk=0; kk<nchannels; kk++)
	{
		for(int jj=0; jj<height; jj++)
			for(int ii=0; ii<width; ii++)
				Img[ii+jj*width+kk*length] = (float)(int)view.data[nchannels*ii+(height-1-jj)*nchannels*width+kk];
	}

	return true;
}
bool myImgReader(char *fname,  double *Img, int width, int height, int nchannels)
{
	Mat view = imread(fname, nchannels==1?0:1);
	if(view.data == NULL)
	{
		cout<<"Cannot load: "<<fname<<endl;
		return false;
	}
	int length = width*height;
	for(int kk=0; kk<nchannels; kk++)
	{
		for(int jj=0; jj<height; jj++)
			for(int ii=0; ii<width; ii++)
				Img[ii+jj*width+kk*length] = (double)(int)view.data[nchannels*ii+(height-1-jj)*nchannels*width+kk];
	}

	return true;
}

bool SaveDataToImage(char *fname, char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1?CV_8UC1:CV_8UC3);
	for(jj=0; jj<height; jj++)
		for(ii=0; ii<width; ii++)
			for(kk=0; kk<nchannels; kk++)
				M.data[nchannels*ii+kk+nchannels*jj*width] = (unsigned char)Img[ii+(height-1-jj)*width+kk*length];

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, unsigned char *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1?CV_8UC1:CV_8UC3);
	for(jj=0; jj<height; jj++)
		for(ii=0; ii<width; ii++)
			for(kk=0; kk<nchannels; kk++)
				M.data[nchannels*ii+kk+nchannels*jj*width] = Img[ii+(height-1-jj)*width+kk*length];

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, float *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1?CV_8UC1:CV_8UC3);
	for(jj=0; jj<height; jj++)
		for(ii=0; ii<width; ii++)
			for(kk=0; kk<nchannels; kk++)
				M.data[nchannels*ii+kk+nchannels*jj*width] = (unsigned char) (int)(Img[ii+(height-1-jj)*width+kk*length]+0.5);

	return imwrite(fname, M);
}
bool SaveDataToImage(char *fname, double *Img, int width, int height, int nchannels)
{
	int ii, jj, kk, length = width*height;

	Mat M = Mat::zeros(height, width, nchannels == 1?CV_8UC1:CV_8UC3);
	for(jj=0; jj<height; jj++)
		for(ii=0; ii<width; ii++)
			for(kk=0; kk<nchannels; kk++)
				M.data[nchannels*ii+kk+nchannels*jj*width] = (unsigned char) (int)(Img[ii+(height-1-jj)*width+kk*length]+0.5);

	return imwrite(fname, M);
}

bool WriteFlowBinary(char *fnX, char *fnY, float *fx, float *fy, int width, int height)
{
	float u, v;

	ofstream fout1, fout2; 
	fout1.open(fnX, ios::binary);
	if(!fout1.is_open()) 
	{
		cout<<"Cannot load: "<<fnX<<endl;
		return false;
	}
	fout2.open(fnY, ios::binary);
	if(!fout2.is_open()) 
	{
		cout<<"Cannot load: "<<fnY<<endl;
		return false;
	}

	for (int j=0;j<height;++j)  
		for (int i=0;i<width;++i)
		{
			// (1) pointing up in matlab is pointing down in c++
			// (2) top left coord in matlab vs. bottom left coord. in your code
			//u = fx[i+(height-1-j)*width];			
			//v = -fy[i+(height-1-j)*width];
			u = fx[i+j*width];			
			v = fy[i+j*width];

			fout1.write(reinterpret_cast<char *>(&u), sizeof(float));
			fout2.write(reinterpret_cast<char *>(&v), sizeof(float));
		}
		fout1.close();
		fout2.close();

		return true;
}
bool ReadFlowBinary(char *fnX, char *fnY, float *fx, float *fy, int width, int height)
{
	float u, v;

	ifstream fin1, fin2; 
	fin1.open(fnX, ios::binary);
	if(!fin1.is_open()) 
	{
		cout<<"Cannot load: "<<fnX<<endl;
		return false;
	}
	fin2.open(fnY, ios::binary);
	if(!fin2.is_open()) 
	{
		cout<<"Cannot load: "<<fnY<<endl;
		return false;
	}

	for (int j=0;j<height;++j)  
		for (int i=0;i<width;++i)
		{
			// (1) pointing up in matlab is pointing down in c++
			// (2) top left coord in matlab vs. bottom left coord. in your code
			fin1.read(reinterpret_cast<char *>(&u), sizeof(float));
			fin2.read(reinterpret_cast<char *>(&v), sizeof(float));

			//fx[i+(height-1-j)*width] = u;			
			//fy[i+(height-1-j)*width] = -v;

			fx[i+j*width] = u;
			fy[i+j*width] = v;
		}
		fin1.close();
		fin2.close();

		return true;
}

void ResizeImage(unsigned char *Image, unsigned char *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, double *InPara)
{
	bool createMem = false;
	int length = width*height;
	if(InPara == NULL)
	{
		createMem = true;
		InPara = new double [length*nchannels];
		for(int kk=0; kk<nchannels; kk++)
			Generate_Para_Spline(Image+kk*length, InPara+kk*length, width, height, InterpAlgo);
	}

	double S[3];
	int nwidth = width*Rfactor, nheight = height*Rfactor, nlength = nwidth*nheight;
	for(int kk=0; kk<nchannels; kk++)
		for(int jj=0; jj<nheight; jj++)
		{
			for(int ii=0; ii<nwidth; ii++)
			{
				Get_Value_Spline(InPara+kk*length, width, height, 1.0*ii/Rfactor, 1.0*jj/Rfactor, S, -1, InterpAlgo);
				if(S[0]>255.0)
					OutImage[ii+jj*nwidth+kk*nlength] = 255;
				else if(S[0]<0.0)
					OutImage[ii+jj*nwidth+kk*nlength] = 0;
				else
					OutImage[ii+jj*nwidth+kk*nlength] = (unsigned char)(int)(S[0]+0.5);
			}
		}

		if(createMem)
			delete []InPara;

		return;
}
void ResizeImage(float *Image, float *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, float *InPara)
{
	bool createMem = false;
	int length = width*height;
	if(InPara == NULL)
	{
		createMem = true;
		InPara = new float[width*height*nchannels];
		for(int kk=0; kk<nchannels; kk++)
			Generate_Para_Spline(Image+kk*length, InPara+kk*length, width, height, InterpAlgo);
	}

	double S[3];
	int nwidth = width*Rfactor, nheight = height*Rfactor, nlength = nwidth*nheight;
	for(int kk=0; kk<nchannels; kk++)
		for(int jj=0; jj<nheight; jj++)
		{
			for(int ii=0; ii<nwidth; ii++)
			{
				Get_Value_Spline(InPara+kk*length, width, height, 1.0*ii/Rfactor, 1.0*jj/Rfactor, S, -1, InterpAlgo);
				if(S[0]>255.0)
					OutImage[ii+jj*nwidth+kk*nlength] = 255;
				else if(S[0]<0.0)
					OutImage[ii+jj*nwidth+kk*nlength] = 0;
				else
					OutImage[ii+jj*nwidth+kk*nlength] = (float)S[0];
			}
		}

		if(createMem)
			delete []InPara;

		return;
}
void ResizeImage(double *Image, double *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, double *InPara)
{
	bool createMem = false;
	int length = width*height;
	if(InPara == NULL)
	{
		createMem = true;
		InPara = new double[width*height*nchannels];
		for(int kk=0; kk<nchannels; kk++)
			Generate_Para_Spline(Image+kk*length, InPara+kk*length, width, height, InterpAlgo);
	}

	double S[3];
	int nwidth = width*Rfactor, nheight = height*Rfactor, nlength = nwidth*nheight;
	for(int kk=0; kk<nchannels; kk++)
		for(int jj=0; jj<nheight; jj++)
		{
			for(int ii=0; ii<nwidth; ii++)
			{
				Get_Value_Spline(InPara+kk*length, width, height, 1.0*ii/Rfactor, 1.0*jj/Rfactor, S, -1, InterpAlgo);
				if(S[0]>255.0)
					OutImage[ii+jj*nwidth+kk*nlength] = 255;
				else if(S[0]<0.0)
					OutImage[ii+jj*nwidth+kk*nlength] = 0;
				else
					OutImage[ii+jj*nwidth+kk*nlength] = S[0];
			}
		}

		if(createMem)
			delete []InPara;

		return;
}

double interpolate( double val, double y0, double x0, double y1, double x1 ) 
{
	return (val-x0)*(y1-y0)/(x1-x0) + y0;
}
double base( double val ) 
{
	if ( val <= -0.75 ) return 0;
	else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
	else if ( val <= 0.25 ) return 1.0;
	else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
	else return 0.0;
}
double red( double gray ) 
{
	return base( gray - 0.5 );
}
double green( double gray )
{
	return base( gray );
}
double blue( double gray ) 
{
	return base( gray + 0.5 );
}
void ConvertToHeatMap(double *Map, unsigned char *ColorMap, int width, int height, bool *mask)
{
	int ii, jj;
	double gray;
	if(mask)
	{
		for(jj=0; jj<height; jj++)
			for(ii=0; ii<width; ii++)
			{
				if(mask[ii+jj*width])
				{
					ColorMap[3*ii+3*jj*width] = 0;
					ColorMap[3*ii+3*jj*width+1] = 0;
					ColorMap[3*ii+3*jj*width+2] = 0;
				}
				else
				{
					gray = Map[ii+jj*width];
					ColorMap[3*ii+3*jj*width] = (unsigned char)(int)(255.0*red(gray)+0.5);
					ColorMap[3*ii+3*jj*width+1] = (unsigned char)(int)(255.0*green(gray)+0.5);
					ColorMap[3*ii+3*jj*width+2] = (unsigned char)(int)(255.0*blue(gray)+0.5);
				}
			}
	}
	else
	{
		for(jj=0; jj<height; jj++)
			for(ii=0; ii<width; ii++)
			{
				gray = Map[ii+jj*width];
				ColorMap[3*ii+3*jj*width] = (unsigned char)(int)(255.0*red(gray)+0.5);
				ColorMap[3*ii+3*jj*width+1] = (unsigned char)(int)(255.0*green(gray)+0.5);
				ColorMap[3*ii+3*jj*width+2] = (unsigned char)(int)(255.0*blue(gray)+0.5);
			}
	}

	return;
}
