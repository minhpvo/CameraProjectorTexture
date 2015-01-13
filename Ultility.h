#include <cstdlib>
#include <iostream>
#include <fstream>
#include <float.h>
#include <opencv2/opencv.hpp>
#include "DataStructure.h"
#include "Matrix.h"

using namespace cv;
using namespace std;

int MyFtoI(double W);
bool IsNumber(double x);
bool IsFiniteNumber(double x) ;
double gaussian_noise(double mean, double std);

void normalize(double *x, int dim = 3);
float MeanArray(float *data, int length);
double MeanArray(double *data, int length);
double VarianceArray(double *data, int length, double mean = NULL);
double MeanArray(vector<double>&data);
double VarianceArray(vector<double>&data, double mean = NULL);
double norm_dot_product(double *x, double *y, int dim = 3);
void cross_product(double *x, double *y, double *xy);
void mat_invert(double* mat, double* imat, int dims = 3);
void mat_mul(double *aa,double *bb, double *out,int rowa,int col_row,int colb);
void mat_add(double *aa,double *bb,double* cc,int row,int col, double scale_a = 1.0, double scale_b = 1.0);
void mat_subtract(double *aa,double *bb,double* cc,int row,int col, double scale_a = 1.0, double scale_b = 1.0);
void mat_transpose(double *in,double *out,int row_in,int col_in);
void mat_completeSym(double *mat, int size, bool upper = true);
void Rodrigues_trans(double *RT_vec, double *R_mat, bool vec2mat, double *dR_dm = NULL);


void LS_Solution_Double(double *lpA, double *lpB, int m, int n);
void QR_Solution_Double(double *lpA, double *lpB, int m, int n);
void Quick_Sort_Double(double * A, int *B, int low, int high);
void Quick_Sort_Float(float * A, int *B, int low, int high);
void Quick_Sort_Int(int * A, int *B, int low, int high);

bool in_polygon(double u, double v, CPoint2 *vertex, int num_vertex);

void ConvertToHeatMap(double *Map, unsigned char *ColorMap, int width, int height, bool *mask = 0);

bool myImgReader(char *fname,  unsigned char *Img, int &width, int &height, int nchannels);
bool myImgReader(char *fname,  float *Img, int &width, int &height, int nchannels);
bool myImgReader(char *fname,  double *Img, int &width, int &height, int nchannels);

bool SaveDataToImage(char *fname, char *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, unsigned char *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, float *Img, int width, int height, int nchannels = 1);
bool SaveDataToImage(char *fname, double *Img, int width, int height, int nchannels = 1);

template <class myType>
bool WriteGridBinary(char *fn, myType *data, int width, int height, bool silent = false)
{
	ofstream fout; 
	fout.open(fn, ios::binary);
	if(!fout.is_open()) 
	{
		if(silent)
			cout<<"Cannot write: "<<fn<<endl;
		return false;
	}

	for (int j=0;j<height;++j)  
		for (int i=0;i<width;++i)
			fout.write(reinterpret_cast<char *>(&data[i+j*width]), sizeof(myType));
	fout.close();

	return true;
}
template <class myType>
bool ReadGridBinary(char *fn, myType *data, int width, int height, bool silent = false)
{
	ifstream fin; 
	fin.open(fn, ios::binary);
	if(!fin.is_open()) 
	{
		cout<<"Cannot open: "<<fn<<endl;
		return false;
	}
	if(silent)
		cout<<"Load "<<fn<<endl;

	for (int j=0;j<height;++j)  
		for (int i=0;i<width;++i)
			fin.read(reinterpret_cast<char *>(&data[i+j*width]), sizeof(myType));
	fin.close();

	return true;
}
bool WriteFlowBinary(char *fnX, char *fnY, float *fx, float *fy, int width, int height);
bool ReadFlowBinary(char *fnX, char *fnY, float *fx, float *fy, int width, int height);

void ResizeImage(unsigned char *Image, unsigned char *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, double *InPara = 0);
void ResizeImage(float *Image, float *OutImage, int width, int height, int channels, double Rfactor, int InterpAlgo, float *Para = 0);
void ResizeImage(double *Image, double *OutImage, int width, int height, int nchannels, double Rfactor, int InterpAlgo, double *Para = 0);

template <class m_Type> class m_TemplateClass_1
{
public:
	void Quick_Sort(m_Type* A, int *B, int low, int high);
	void QR_Solution(m_Type *lpA, m_Type *lpB, int m, int n);
	void QR_Solution_2(m_Type *lpA, m_Type *lpB, int m, int n, int k);
};

template <class m_Type> void m_TemplateClass_1<m_Type>::QR_Solution(m_Type *lpA, m_Type *lpB, int m, int n)
{
	int ii,jj,mm,kk;
	m_Type t,d,alpha,u;
	m_Type *lpC = new m_Type[n];
	m_Type *lpQ = new m_Type[m*m];

	for(ii=0;ii<m;ii++)
	{
		for(jj=0;jj<m;jj++)
		{
			*(lpQ+ii*m+jj)=(m_Type)0;
			if(ii==jj)
				*(lpQ+ii*m+jj)=(m_Type)1;
		}
	}

	for(kk=0;kk<n;kk++)
	{
		u=(m_Type)0;
		for(ii=kk;ii<m;ii++)
		{
			if(fabs(*(lpA+ii*n+kk))>u)
				u=(m_Type)(fabs(*(lpA+ii*n+kk)));
		}

		alpha=(m_Type)0;
		for(ii=kk;ii<m;ii++)
		{
			t=*(lpA+ii*n+kk)/u;
			alpha=alpha+t*t;
		}
		if(*(lpA+kk*n+kk)>(m_Type)0)
			u=-u;
		alpha=(m_Type)(u*sqrt(alpha));
		u=(m_Type)(sqrt(2.0*alpha*(alpha-*(lpA+kk*n+kk))));
		if(fabs(u)>1e-8)
		{
			*(lpA+kk*n+kk)=(*(lpA+kk*n+kk)-alpha)/u;
			for(ii=kk+1;ii<m;ii++)
				*(lpA+ii*n+kk)=*(lpA+ii*n+kk)/u;
			for(jj=0;jj<m;jj++)
			{
				t=(m_Type)0;
				for(mm=kk;mm<m;mm++)
					t=t+*(lpA+mm*n+kk)*(*(lpQ+mm*m+jj));
				for(ii=kk;ii<m;ii++)
					*(lpQ+ii*m+jj)=*(lpQ+ii*m+jj)-(m_Type)(2.0*t*(*(lpA+ii*n+kk)));
			}
			for(jj=kk+1;jj<n;jj++)
			{
				t=(m_Type)0;
				for(mm=kk;mm<m;mm++)
					t=t+*(lpA+mm*n+kk)*(*(lpA+mm*n+jj));
				for(ii=kk;ii<m;ii++)
					*(lpA+ii*n+jj)=*(lpA+ii*n+jj)-(m_Type)(2.0*t*(*(lpA+ii*n+kk)));
			}
			*(lpA+kk*n+kk)=alpha;
			for(ii=kk+1;ii<m;ii++)
				*(lpA+ii*n+kk)=(m_Type)0;
		}
	}
	for(ii=0;ii<m-1;ii++)
	{
		for(jj=ii+1;jj<m;jj++)
		{
			t=*(lpQ+ii*m+jj);
			*(lpQ+ii*m+jj)=*(lpQ+jj*m+ii);
			*(lpQ+jj*m+ii)=t;
		}
	}
	//Solve the equation
	for(ii=0;ii<n;ii++)
	{
		d=(m_Type)0;
		for(jj=0;jj<m;jj++)
			d=d+*(lpQ+jj*m+ii)*(*(lpB+jj));
		*(lpC+ii)=d;
	}
	*(lpB+n-1)=*(lpC+n-1)/(*(lpA+(n-1)*n+n-1));
	for(ii=n-2;ii>=0;ii--)
	{
		d=(m_Type)0;
		for(jj=ii+1;jj<n;jj++)
			d=d+*(lpA+ii*n+jj)*(*(lpB+jj));
		*(lpB+ii)=(*(lpC+ii)-d)/(*(lpA+ii*n+ii));
	}

	delete []lpQ;
	delete []lpC;
	return;
}
template <class m_Type> void m_TemplateClass_1<m_Type>::QR_Solution_2(m_Type *lpA, m_Type *lpB, int m, int n, int k)
{
	int ii,jj,mm,kk;
	m_Type t,d,alpha,u;
	m_Type *lpC = new m_Type[n];
	m_Type *lpQ = new m_Type[m*m];

	for(ii=0;ii<m;ii++)
	{
		for(jj=0;jj<m;jj++)
		{
			*(lpQ+ii*m+jj)=(m_Type)0;
			if(ii==jj)
				*(lpQ+ii*m+jj)=(m_Type)1;
		}
	}

	for(kk=0;kk<n;kk++)
	{
		u=(m_Type)0;
		for(ii=kk;ii<m;ii++)
		{
			if(fabs(*(lpA+ii*n+kk))>u)
				u=(m_Type)(fabs(*(lpA+ii*n+kk)));
		}

		alpha=(m_Type)0;
		for(ii=kk;ii<m;ii++)
		{
			t=*(lpA+ii*n+kk)/u;
			alpha=alpha+t*t;
		}
		if(*(lpA+kk*n+kk)>(m_Type)0)
			u=-u;
		alpha=(m_Type)(u*sqrt(alpha));
		u=(m_Type)(sqrt(2.0*alpha*(alpha-*(lpA+kk*n+kk))));
		if(fabs(u)>1e-8)
		{
			*(lpA+kk*n+kk)=(*(lpA+kk*n+kk)-alpha)/u;
			for(ii=kk+1;ii<m;ii++)
				*(lpA+ii*n+kk)=*(lpA+ii*n+kk)/u;
			for(jj=0;jj<m;jj++)
			{
				t=(m_Type)0;
				for(mm=kk;mm<m;mm++)
					t=t+*(lpA+mm*n+kk)*(*(lpQ+mm*m+jj));
				for(ii=kk;ii<m;ii++)
					*(lpQ+ii*m+jj)=*(lpQ+ii*m+jj)-(m_Type)(2.0*t*(*(lpA+ii*n+kk)));
			}
			for(jj=kk+1;jj<n;jj++)
			{
				t=(m_Type)0;
				for(mm=kk;mm<m;mm++)
					t=t+*(lpA+mm*n+kk)*(*(lpA+mm*n+jj));
				for(ii=kk;ii<m;ii++)
					*(lpA+ii*n+jj)=*(lpA+ii*n+jj)-(m_Type)(2.0*t*(*(lpA+ii*n+kk)));
			}
			*(lpA+kk*n+kk)=alpha;
			for(ii=kk+1;ii<m;ii++)
				*(lpA+ii*n+kk)=(m_Type)0;
		}
	}
	for(ii=0;ii<m-1;ii++)
	{
		for(jj=ii+1;jj<m;jj++)
		{
			t=*(lpQ+ii*m+jj);
			*(lpQ+ii*m+jj)=*(lpQ+jj*m+ii);
			*(lpQ+jj*m+ii)=t;
		}
	}
	//Solve the equation

	m_Type *lpBB;
	for(mm=0; mm<k; mm++)
	{
		lpBB = lpB+mm*m;

		for(ii=0;ii<n;ii++)
		{
			d=(m_Type)0;
			for(jj=0;jj<m;jj++)
				d=d+*(lpQ+jj*m+ii)*(*(lpBB+jj));
			*(lpC+ii)=d;
		}
		*(lpBB+n-1)=*(lpC+n-1)/(*(lpA+(n-1)*n+n-1));
		for(ii=n-2;ii>=0;ii--)
		{
			d=(m_Type)0;
			for(jj=ii+1;jj<n;jj++)
				d=d+*(lpA+ii*n+jj)*(*(lpBB+jj));
			*(lpBB+ii)=(*(lpC+ii)-d)/(*(lpA+ii*n+ii));
		}
	}

	delete []lpQ;
	delete []lpC;
	return;
}
template <class m_Type> void m_TemplateClass_1<m_Type>::Quick_Sort(m_Type* A, int *B, int low, int high)
	//A: array to be sorted (from min to max); B: index of the original array; low and high: array range
	//After sorting, A: sorted array; B: re-sorted index of the original array, e.g., the m-th element of
	// new A[] is the original n-th element in old A[]. B[m-1]=n-1;
	//B[] is useless for most sorting, it is added here for the special application in this program.  
{
	m_Type A_pivot,A_S;
	int B_pivot,B_S;
	int scanUp, scanDown;
	int mid;
	if(high-low<=0)
		return;
	else if(high-low==1)
	{
		if(A[high]<A[low])
		{
			//	Swap(A[low],A[high]);
			//	Swap(B[low],B[high]);
			A_S=A[low];
			A[low]=A[high];
			A[high]=A_S;
			B_S=B[low];
			B[low]=B[high];
			B[high]=B_S;
		}
		return;
	}
	mid=(low+high)/2;
	A_pivot=A[mid];
	B_pivot=B[mid];

	//	Swap(A[mid],A[low]);
	//	Swap(B[mid],B[low]);
	A_S=A[mid];
	A[mid]=A[low];
	A[low]=A_S;
	B_S=B[mid];
	B[mid]=B[low];
	B[low]=B_S;

	scanUp=low+1;
	scanDown=high;
	do
	{
		while(scanUp<=scanDown && A[scanUp]<=A_pivot)
			scanUp++;
		while(A_pivot<A[scanDown])
			scanDown--;
		if(scanUp<scanDown)
		{
			//	Swap(A[scanUp],A[scanDown]);
			//	Swap(B[scanUp],B[scanDown]);
			A_S=A[scanUp];
			A[scanUp]=A[scanDown];
			A[scanDown]=A_S;
			B_S=B[scanUp];
			B[scanUp]=B[scanDown];
			B[scanDown]=B_S;
		}
	}while(scanUp<scanDown);

	A[low]=A[scanDown];
	B[low]=B[scanDown];
	A[scanDown]=A_pivot;
	B[scanDown]=B_pivot;
	if(low<scanDown-1)
		Quick_Sort(A,B,low,scanDown-1);
	if(scanDown+1<high)
		Quick_Sort(A,B,scanDown+1,high);
}
