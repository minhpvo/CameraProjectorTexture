#include <omp.h>
#include <math.h>
#include "IlluminationFlowUlti.h"
#include "ceres/ceres.h"
#include "glog/logging.h"
#include "ceres/types.h"

using namespace std;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::CauchyLoss;
using ceres::SoftLOneLoss;
using ceres::HuberLoss;
using ceres::Problem;
using ceres::Solver;

using namespace std;

#define INACTIVE_WC -999.999
#define INACTIVE_IMPTS 1.0

// Calculates log2 of number.  
template <class T> T log2(T a)
{
	T result = log(a) / log(T(2.0));
	return result;
}
struct SoftConstraint {

	SoftConstraint(double bet, double sc) :beta(bet), softConst(sc){};

	template <typename T>	bool operator()(const T* const d, T* residuals) const
	{
		residuals[0] = T(beta)*(d[0] - T(softConst));
		return true;
	}
	static ceres::CostFunction* Create(double beta, double softConst)
	{
		return (new ceres::AutoDiffCostFunction<SoftConstraint, 1, 1>(new SoftConstraint(beta, softConst)));
	}
	double beta, softConst;
};
struct ZRegularizationErr1 {

	ZRegularizationErr1(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d5, const T* const d6, const T* const d8, const T* const d9, T* residuals) const
	{
		residuals[0] = T(alpha)*(d9[0] + d8[0] + d9[0]
			+ d6[0] - T(8.0)*d5[0] + d6[0]
			+ d9[0] + d8[0] + d9[0]);
		//cout<<"("<<ii<<","<<jj<<") "<<d5[0]<<" "<<d6[0]<<" "<<d8[0]<<" "<<d9[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}

	//Lower left
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr1, 1, 1, 1, 1, 1>(new ZRegularizationErr1(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ZRegularizationErr2 {
	ZRegularizationErr2(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d4, const T* const d5, const T* const d7, const T* const d8, T* residuals) const
	{
		residuals[0] = T(alpha)*(d7[0] + d8[0] + d7[0]
			+ d4[0] - T(8.0)*d5[0] + d4[0]
			+ d7[0] + d8[0] + d7[0]);
		//	cout<<"("<<ii<<","<<jj<<") "<<d4[0]<<" "<<d5[0]<<" "<<d7[0]<<" "<<d8[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}

	//Lower right
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr2, 1, 1, 1, 1, 1>(new ZRegularizationErr2(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ZRegularizationErr3 {
	ZRegularizationErr3(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d2, const T* const d3, const T* const d5, const T* const d6, T* residuals) const
	{
		residuals[0] = T(alpha)*(d3[0] + d2[0] + d3[0]
			+ d6[0] - T(8.0)*d5[0] + d6[0]
			+ d3[0] + d2[0] + d3[0]);
		//cout<<"("<<ii<<","<<jj<<") "<<d2[0]<<" "<<d3[0]<<" "<<d5[0]<<" "<<d6[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}

	//Upper left
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr3, 1, 1, 1, 1, 1>(new ZRegularizationErr3(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ZRegularizationErr4 {
	ZRegularizationErr4(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d1, const T* const d2, const T* const d4, const T* const d5, T* residuals) const
	{
		residuals[0] = T(alpha)*(d1[0] + d2[0] + d1[0]
			+ d4[0] - T(8.0)*d5[0] + d4[0]
			+ d1[0] + d2[0] + d1[0]);
		//cout<<"("<<ii<<","<<jj<<") "<<d1[0]<<" "<<d2[0]<<" "<<d4[0]<<" "<<d5[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}

	//Upper right
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr4, 1, 1, 1, 1, 1>(new ZRegularizationErr4(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ZRegularizationErr5 {
	ZRegularizationErr5(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d2, const T* const d3, const T* const d5, const T* const d6, const T* const d8, const T* const d9, T* residuals) const
	{
		residuals[0] = T(alpha)*(d3[0] + d2[0] + d3[0]
			+ d6[0] - T(8.0)*d5[0] + d6[0]
			+ d9[0] + d8[0] + d9[0]);
		//cout<<"("<<ii<<","<<jj<<") "<<d2[0]<<" "<<d3[0]<<" "<<d5[0]<<" "<<d6[0]<<" "<<d8[0]<<" "<<d9[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}

	//Left
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr5, 1, 1, 1, 1, 1, 1, 1>(new ZRegularizationErr5(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ZRegularizationErr6 {
	ZRegularizationErr6(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d1, const T* const d2, const T* const d4, const T* const d5, const T* const d7, const T* const d8, T* residuals) const
	{
		residuals[0] = T(alpha)*(d1[0] + d2[0] + d1[0]
			+ d4[0] - T(8.0)*d5[0] + d4[0]
			+ d7[0] + d8[0] + d7[0]);
		//cout<<"("<<ii<<","<<jj<<") "<<d1[0]<<" "<<d2[0]<<" "<<d4[0]<<" "<<d5[0]<<" "<<d7[0]<<" "<<d8[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}

	//Right
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr6, 1, 1, 1, 1, 1, 1, 1>(new ZRegularizationErr6(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ZRegularizationErr7 {
	ZRegularizationErr7(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d4, const T* const d5, const T* const d6, const T* const d7, const T* const d8, const T* const d9, T* residuals) const
	{
		residuals[0] = T(alpha)*(d7[0] + d8[0] + d9[0]
			+ d4[0] - T(8.0)*d5[0] + d6[0]
			+ d7[0] + d8[0] + d9[0]);
		//cout<<"("<<ii<<","<<jj<<") "<<d4[0]<<" "<<d5[0]<<" "<<d6[0]<<" "<<d7[0]<<" "<<d8[0]<<" "<<d9[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}

	// Bottom
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr7, 1, 1, 1, 1, 1, 1, 1>(new ZRegularizationErr7(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ZRegularizationErr8 {
	ZRegularizationErr8(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d1, const T* const d2, const T* const d3, const T* const d4, const T* const d5, const T* const d6, T* residuals) const
	{
		residuals[0] = T(alpha)*(d1[0] + d2[0] + d3[0]
			+ d4[0] - T(8.0)*d5[0] + d6[0]
			+ d1[0] + d2[0] + d3[0]);
		//cout<<"("<<ii<<","<<jj<<") "<<d1[0]<<" "<<d2[0]<<" "<<d3[0]<<" "<<d4[0]<<" "<<d5[0]<<" "<<d6[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}

	// Top
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr8, 1, 1, 1, 1, 1, 1, 1>(new ZRegularizationErr8(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ZRegularizationErr9 {
	ZRegularizationErr9(double al, int i, int j) : alpha(al), ii(i), jj(j){};

	template <typename T>	bool operator()(const T* const d1, const T* const d2, const T* const d3, const T* const d4,
		const T* const d5, const T* const d6, const T* const d7, const T* const d8, const T* const d9, T* residuals) const
	{
		residuals[0] = T(alpha)*(d1[0] + d2[0] + d3[0]
			+ d4[0] - T(8.0)*d5[0] + d6[0]
			+ d7[0] + d8[0] + d9[0]);
		//cout<<"("<<ii<<","<<jj<<") "<<d1[0]<<" "<<d2[0]<<" "<<d3[0]<<" "<<d4[0]<<" "<<d5[0]<<" "<<d6[0]<<" "<<d7[0]<<" "<<d8[0]<<" "<<d9[0]<<" "<<residuals[0]<<" "<<endl;
		return true;
	}
	//Well surrounded, no need for boundary condition
	static ceres::CostFunction* Create(double alpha, int ii, int jj)
	{
		return (new ceres::AutoDiffCostFunction<ZRegularizationErr9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(new ZRegularizationErr9(alpha, ii, jj)));
	}
	int ii, jj;
	double alpha;
};
struct ReprojectionError {
	ReprojectionError(int r, int c, int gW, int gH, double *flowX, double *flowY, double *prd)
	{
		rID = r, cID = c;
		index = rID + cID*gW;
		fu = flowX[index], fv = flowY[index];
		PrayDirect[0] = prd[0 + 6 * index], PrayDirect[1] = prd[1 + 6 * index], PrayDirect[2] = prd[2 + 6 * index];
		PrayDirect[3] = prd[3 + 6 * index], PrayDirect[4] = prd[4 + 6 * index], PrayDirect[5] = prd[5 + 6 * index];
	}

	template <typename T>	bool operator()(const T* const depth, T* residuals) const
	{
		T u = (depth[0] * PrayDirect[0] + PrayDirect[1]) / (depth[0] * PrayDirect[4] + PrayDirect[5]);
		T v = (depth[0] * PrayDirect[2] + PrayDirect[3]) / (depth[0] * PrayDirect[4] + PrayDirect[5]);
		residuals[0] = u - T(fu);
		residuals[1] = v - T(fv);
		//cout<<"("<<rID<<","<<cID<<") "<<depth[0]<<" "<<fu<<" "<<fv<<" "<<residuals[0]<<" "<<residuals[1]<<endl;
		return true;
	}

	// Factory to hide the construction of the CostFunction object from the client code.
	static ceres::CostFunction* Create(int rID, int cID, int gW, int gH, double *FlowX, double *FlowY, double*PrayDirect)
	{
		return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 1>(new ReprojectionError(rID, cID, gW, gH, FlowX, FlowY, PrayDirect)));
	}

	int rID, cID, index;
	double fu, fv, PrayDirect[6];
};
void LensDistortion_Point(CPoint2 &img_point, double *camera, double *distortion)
{
	double alpha = camera[0], beta = camera[4], gamma = camera[1], u0 = camera[2], v0 = camera[5];

	double ycn = (img_point.y - v0) / beta;
	double xcn = (img_point.x - u0 - gamma*ycn) / alpha;

	double r2 = xcn*xcn + ycn*ycn, r4 = r2*r2, r6 = r2*r4, r8 = r4*r4, r10 = r4*r6, X2 = xcn*xcn, Y2 = ycn*ycn, XY = xcn*ycn;

	double a0 = distortion[0], a1 = distortion[1], a2 = distortion[2], a3 = distortion[3], a4 = distortion[4];
	double p0 = distortion[5], p1 = distortion[6], p2 = distortion[7], p3 = distortion[8];
	double s0 = distortion[9], s1 = distortion[10], s2 = distortion[11], s3 = distortion[12];

	double radial = (1 + a0*r2 + a1*r4 + a2*r6 + a3*r8 + a4*r10);
	double tangential_x = (p0 + p2*r2)*(r2 + 2.0*X2) + 2.0*(p1 + p3*r2)*XY;
	double tangential_y = (p1 + p3*r2)*(r2 + 2.0*Y2) + 2.0*(p0 + p2*r2)*XY;
	double prism_x = s0*r2 + s2*r4;
	double prism_y = s1*r2 + s3*r4;

	double xcn_ = radial*xcn + tangential_x + prism_x;
	double ycn_ = radial*ycn + tangential_y + prism_y;

	img_point.x = alpha*xcn_ + gamma*ycn_ + u0;
	img_point.y = beta*ycn_ + v0;

	return;
}
void LensDistortion_PointL(CPoint2L &img_point, double *camera, double *distortion)
{

	double alpha = camera[0], beta = camera[4], gamma = camera[1], u0 = camera[2], v0 = camera[5];

	double ycn = (img_point.y - v0) / beta;
	double xcn = (img_point.x - u0 - gamma*ycn) / alpha;

	double r2 = xcn*xcn + ycn*ycn, r4 = r2*r2, r6 = r2*r4, r8 = r4*r4, r10 = r4*r6, X2 = xcn*xcn, Y2 = ycn*ycn, XY = xcn*ycn;

	double a0 = distortion[0], a1 = distortion[1], a2 = distortion[2], a3 = distortion[3], a4 = distortion[4];
	double p0 = distortion[5], p1 = distortion[6], p2 = distortion[7], p3 = distortion[8];
	double s0 = distortion[9], s1 = distortion[10], s2 = distortion[11], s3 = distortion[12];

	double radial = (1 + a0*r2 + a1*r4 + a2*r6 + a3*r8 + a4*r10);
	double tangential_x = (p0 + p2*r2)*(r2 + 2.0*X2) + 2.0*(p1 + p3*r2)*XY;
	double tangential_y = (p1 + p3*r2)*(r2 + 2.0*Y2) + 2.0*(p0 + p2*r2)*XY;
	double prism_x = s0*r2 + s2*r4;
	double prism_y = s1*r2 + s3*r4;

	double xcn_ = radial*xcn + tangential_x + prism_x;
	double ycn_ = radial*ycn + tangential_y + prism_y;

	img_point.x = alpha*xcn_ + gamma*ycn_ + u0;
	img_point.y = beta*ycn_ + v0;

	return;
}
void CC_Calculate_xcn_ycn_from_i_j(double i, double j, double &xcn, double &ycn, double *A, double *K, int Method)
{
	int k;
	double Xcn, Ycn, r2, r4, r6, r8, r10, x2, y2, xy, x0, y0, t, t_x, t_y;
	double radial, tangential_x, tangential_y, prism_x, prism_y;
	double a0 = K[0], a1 = K[1], a2 = K[2], a3 = K[3], a4 = K[4];
	double p0 = K[5], p1 = K[6], p2 = K[7], p3 = K[8];
	double s0 = K[9], s1 = K[10], s2 = K[11], s3 = K[12];

	double dxcn_dxcn_radial, dxcn_dxcn_tangential, dxcn_dxcn_prism;
	double dxcn_dycn_radial, dxcn_dycn_tangential, dxcn_dycn_prism;
	double dycn_dxcn_radial, dycn_dxcn_tangential, dycn_dxcn_prism;
	double dycn_dycn_radial, dycn_dycn_tangential, dycn_dycn_prism;
	double Jacobian[4], Err[2], Hessian[4], JErr[2];

	Ycn = (j - A[4]) / A[1];
	Xcn = (i - A[3] - A[2] * Ycn) / A[0];

	xcn = Xcn;
	ycn = Ycn;
	for (k = 0; k < 20; k++)
	{
		x0 = xcn;
		y0 = ycn;
		r2 = xcn*xcn + ycn*ycn, r4 = r2*r2, r6 = r2*r4, r8 = r4*r4, r10 = r4*r6, x2 = xcn*xcn, y2 = ycn*ycn, xy = xcn*ycn;

		radial = 1.0 + a0*r2 + a1*r4 + a2*r6 + a3*r8 + a4*r10;
		tangential_x = (p0 + p2*r2)*(r2 + 2.0*x2) + 2.0*(p1 + p3*r2)*xy;
		tangential_y = (p1 + p3*r2)*(r2 + 2.0*y2) + 2.0*(p0 + p2*r2)*xy;
		prism_x = s0*r2 + s2*r4;
		prism_y = s1*r2 + s3*r4;

		if (Method == 0)
		{
			xcn = (Xcn - tangential_x - prism_x) / radial;
			ycn = (Ycn - tangential_y - prism_y) / radial;
			t_x = xcn - x0;
			t_y = ycn - y0;
		}
		else if (Method == 1)
		{
			//dxcn_/dxcn, dxcn_/dycn, dycn_/dycn, dycn_/dxcn
			dxcn_dxcn_radial = (1 + a0*r2 + a1*r4 + a2*r6 + a3*r8 + a4*r10) + xcn*(2.0*a0*xcn + 4.0*a1*xcn*r2 + 6.0*a2*xcn*r4 + 8.0*a3*xcn*r6 + 10.0*a4*xcn*r8);
			dxcn_dxcn_tangential = 2.0*xcn*p2*(r2 + 2.0*x2) + 6.0*xcn*(p0 + r2*p2) + 4.0*p3*x2*ycn + 2.0*(p1 + r2*p3)*ycn;
			dxcn_dxcn_prism = 2.0*s0*xcn + 4.0*s2*r2*xcn;

			dxcn_dycn_radial = xcn*(2.0*a0*ycn + 4.0*a1*r2*ycn + 6.0*a2*r4*ycn + 8.0*a3*r6*ycn + 10.0*a4*r8*ycn);
			dxcn_dycn_tangential = 2.0*ycn*p2*(r2 + 2.0*x2) + 2.0*ycn*(p0 + r2*p2) + 4.0*p3*xcn*y2 + 2.0*(p1*+r2*p3)*xcn;
			dxcn_dycn_prism = 2.0*s0*ycn + 4.0*s2*r2*ycn;

			dycn_dycn_radial = (1 + a0*r2 + a1*r4 + a2*r6 + a3*r8 + a4*r10) + ycn*(2.0*a0*ycn + 4.0*a1*r2*ycn + 6.0*a2*r4*ycn + 8.0*a3*r6*ycn + 10.0*a4*r8*ycn);
			dycn_dycn_tangential = 2.0*ycn*p3*(r2 + 2.0*y2) + 6.0*ycn*(p1 + r2*p3) + 4.0*p2*xcn*y2 + 2.0*(p0 + r2*p2)*xcn;
			dycn_dycn_prism = 2.0*s1*ycn + 4.0*s3*r2*ycn;

			dycn_dxcn_radial = ycn*(2.0*a0*xcn + 4.0*a1*r2*xcn + 6.0*a2*r4*xcn + 8.0*a3*r6*xcn + 10.0*a4*r8*xcn);
			dycn_dxcn_tangential = 2.0*xcn*p3*(r2 + 2.0*y2) + 2.0*xcn*(p1 + r2*p3) + 4.0*p2*x2*ycn + 2.0*(p0*+r2*p2)*ycn;
			dycn_dxcn_prism = 2.0*s1*xcn + 4.0*s3*r2*xcn;

			Jacobian[0] = dxcn_dxcn_radial + dxcn_dxcn_tangential + dxcn_dxcn_prism;
			Jacobian[1] = dycn_dxcn_radial + dycn_dxcn_tangential + dycn_dxcn_prism;
			Jacobian[2] = dxcn_dycn_radial + dxcn_dycn_tangential + dxcn_dycn_prism;
			Jacobian[3] = dycn_dycn_radial + dycn_dycn_tangential + dycn_dycn_prism;
			Err[0] = radial*xcn + tangential_x + prism_x - Xcn;
			Err[1] = radial*ycn + tangential_y + prism_y - Ycn;

			JErr[0] = Jacobian[0] * Err[0] + Jacobian[1] * Err[1];
			JErr[1] = Jacobian[2] * Err[0] + Jacobian[3] * Err[1];
			Hessian[0] = Jacobian[0] * Jacobian[0] + Jacobian[1] * Jacobian[1];
			Hessian[1] = Jacobian[0] * Jacobian[2] + Jacobian[1] * Jacobian[3];
			Hessian[2] = Hessian[1];
			Hessian[3] = Jacobian[2] * Jacobian[2] + Jacobian[3] * Jacobian[3];

			t = Hessian[0] * Hessian[3] - Hessian[1] * Hessian[2];
			t_x = (Hessian[3] * JErr[0] - Hessian[1] * JErr[1]) / t;
			t_y = (-Hessian[2] * JErr[0] + Hessian[0] * JErr[1]) / t;
			xcn -= t_x;
			ycn -= t_y;
		}

		//	if(fabs(t_x)<fabs(xcn*1e-24) && fabs(t_y)<fabs(ycn*1e-24))
		if (fabs(t_x) < fabs(xcn*1e-16) && fabs(t_y) < fabs(ycn*1e-16))
			break;
	}
	return;
}

void Undo_distortion(CPoint2 &uv, double *camera, double *distortion)
{
	double xcn, ycn, A[5] = { camera[0], camera[4], camera[1], camera[2], camera[5] };

	CC_Calculate_xcn_ycn_from_i_j(uv.x, uv.y, xcn, ycn, A, distortion, 0);

	uv.x = A[0] * xcn + A[2] * ycn + A[3];
	uv.y = A[1] * ycn + A[4];

	return;
}
void Stereo_Triangulation2(CPoint2 *pts1, CPoint2 *pts2, double *P1, double *P2, CPoint3 *WC, int npts)
{
	int ii;
	double A[12], B[4], u1, v1, u2, v2;
	double p11 = P1[0], p12 = P1[1], p13 = P1[2], p14 = P1[3];
	double p21 = P1[4], p22 = P1[5], p23 = P1[6], p24 = P1[7];
	double p31 = P1[8], p32 = P1[9], p33 = P1[10], p34 = P1[11];

	double P11 = P2[0], P12 = P2[1], P13 = P2[2], P14 = P2[3];
	double P21 = P2[4], P22 = P2[5], P23 = P2[6], P24 = P2[7];
	double P31 = P2[8], P32 = P2[9], P33 = P2[10], P34 = P2[11];

	for (ii = 0; ii < npts; ii++)
	{
		u1 = pts1[ii].x, v1 = pts1[ii].y;
		u2 = pts2[ii].x, v2 = pts2[ii].y;

		A[0] = p11 - u1*p31;
		A[1] = p12 - u1*p32;
		A[2] = p13 - u1*p33;
		A[3] = p21 - v1*p31;
		A[4] = p22 - v1*p32;
		A[5] = p23 - v1*p33;

		A[6] = P11 - u2*P31;
		A[7] = P12 - u2*P32;
		A[8] = P13 - u2*P33;
		A[9] = P21 - v2*P31;
		A[10] = P22 - v2*P32;
		A[11] = P23 - v2*P33;

		B[0] = u1*p34 - p14;
		B[1] = v1*p34 - p24;
		B[2] = u2*P34 - P14;
		B[3] = v2*P34 - P24;

		QR_Solution_Double(A, B, 4, 3);

		WC[ii].x = B[0];
		WC[ii].y = B[1];
		WC[ii].z = B[2];
	}

	return;
}
void Triplet_Triangulation2(CPoint2 *pts1, CPoint2 *pts2, CPoint2 *pts3, double *P1, double *P2, double* P3, CPoint3 *WC, int npts)
{
	int ii;
	double A[18], B[6], u1, v1, u2, v2, u3, v3;
	double p11 = P1[0], p12 = P1[1], p13 = P1[2], p14 = P1[3];
	double p21 = P1[4], p22 = P1[5], p23 = P1[6], p24 = P1[7];
	double p31 = P1[8], p32 = P1[9], p33 = P1[10], p34 = P1[11];

	double P11 = P2[0], P12 = P2[1], P13 = P2[2], P14 = P2[3];
	double P21 = P2[4], P22 = P2[5], P23 = P2[6], P24 = P2[7];
	double P31 = P2[8], P32 = P2[9], P33 = P2[10], P34 = P2[11];

	double PP11 = P3[0], PP12 = P3[1], PP13 = P3[2], PP14 = P3[3];
	double PP21 = P3[4], PP22 = P3[5], PP23 = P3[6], PP24 = P3[7];
	double PP31 = P3[8], PP32 = P3[9], PP33 = P3[10], PP34 = P3[11];

	for (ii = 0; ii < npts; ii++)
	{
		u1 = pts1[ii].x, v1 = pts1[ii].y;
		u2 = pts2[ii].x, v2 = pts2[ii].y;
		u3 = pts3[ii].x, v3 = pts3[ii].y;

		A[0] = p11 - u1*p31;
		A[1] = p12 - u1*p32;
		A[2] = p13 - u1*p33;
		A[3] = p21 - v1*p31;
		A[4] = p22 - v1*p32;
		A[5] = p23 - v1*p33;

		A[6] = P11 - u2*P31;
		A[7] = P12 - u2*P32;
		A[8] = P13 - u2*P33;
		A[9] = P21 - v2*P31;
		A[10] = P22 - v2*P32;
		A[11] = P23 - v2*P33;

		A[12] = PP11 - u3*PP31;
		A[13] = PP12 - u3*PP32;
		A[14] = PP13 - u3*PP33;
		A[15] = PP21 - v3*PP31;
		A[16] = PP22 - v3*PP32;
		A[17] = PP23 - v3*PP33;

		B[0] = u1*p34 - p14;
		B[1] = v1*p34 - p24;
		B[2] = u2*P34 - P14;
		B[3] = v2*P34 - P24;
		B[4] = u3*PP34 - PP14;
		B[5] = v3*PP34 - PP24;

		QR_Solution_Double(A, B, 6, 3);

		WC[ii].x = B[0];
		WC[ii].y = B[1];
		WC[ii].z = B[2];
	}

	return;
}
void NviewTriangulation(CPoint2 *pts, double *P, CPoint3 *WC, int nview, int npts, double *Cov, double *A, double *B)
{
	int ii, jj, kk;
	bool MenCreated = false;
	if (A == NULL)
	{
		MenCreated = true;
		A = new double[6 * nview];
		B = new double[2 * nview];
	}
	double u, v;

	if (Cov == NULL)
	{
		for (ii = 0; ii < npts; ii++)
		{
			for (jj = 0; jj < nview; jj++)
			{
				u = pts[ii + jj*npts].x, v = pts[ii + jj*npts].y;

				A[6 * jj + 0] = P[12 * jj] - u*P[12 * jj + 8];
				A[6 * jj + 1] = P[12 * jj + 1] - u*P[12 * jj + 9];
				A[6 * jj + 2] = P[12 * jj + 2] - u*P[12 * jj + 10];
				A[6 * jj + 3] = P[12 * jj + 4] - v*P[12 * jj + 8];
				A[6 * jj + 4] = P[12 * jj + 5] - v*P[12 * jj + 9];
				A[6 * jj + 5] = P[12 * jj + 6] - v*P[12 * jj + 10];
				B[2 * jj + 0] = u*P[12 * jj + 11] - P[12 * jj + 3];
				B[2 * jj + 1] = v*P[12 * jj + 11] - P[12 * jj + 7];
			}

			QR_Solution_Double(A, B, 2 * nview, 3);

			WC[ii].x = B[0];
			WC[ii].y = B[1];
			WC[ii].z = B[2];
		}
	}
	else
	{
		double mse = 0.0;
		double *At = new double[6 * nview];
		double *Bt = new double[2 * nview];
		double *t1 = new double[4 * nview*nview];
		double *t2 = new double[4 * nview*nview];
		double *Identity = new double[4 * nview*nview];
		double AtA[9], iAtA[9];
		for (ii = 0; ii < 4 * nview*nview; ii++)
			Identity[ii] = 0.0;
		for (ii = 0; ii < 2 * nview; ii++)
			Identity[ii + ii * 2 * nview] = 1.0;

		for (ii = 0; ii < npts; ii++)
		{
			for (jj = 0; jj < nview; jj++)
			{
				u = pts[ii + jj*npts].x, v = pts[ii + jj*npts].y;

				A[6 * jj + 0] = P[12 * jj] - u*P[12 * jj + 8];
				A[6 * jj + 1] = P[12 * jj + 1] - u*P[12 * jj + 9];
				A[6 * jj + 2] = P[12 * jj + 2] - u*P[12 * jj + 10];
				A[6 * jj + 3] = P[12 * jj + 4] - v*P[12 * jj + 8];
				A[6 * jj + 4] = P[12 * jj + 5] - v*P[12 * jj + 9];
				A[6 * jj + 5] = P[12 * jj + 6] - v*P[12 * jj + 10];
				B[2 * jj + 0] = u*P[12 * jj + 11] - P[12 * jj + 3];
				B[2 * jj + 1] = v*P[12 * jj + 11] - P[12 * jj + 7];
			}

			mat_transpose(A, At, nview * 2, 3);
			mat_transpose(B, Bt, nview * 2, 1);
			mat_mul(At, A, AtA, 3, 2 * nview, 3);
			mat_invert(AtA, iAtA);
			mat_mul(A, iAtA, t1, 2 * nview, 3, 3);
			mat_mul(t1, At, t2, 2 * nview, 3, 2 * nview);
			mat_subtract(Identity, t2, t1, 2 * nview, 2 * nview);
			mat_mul(Bt, t1, t2, 1, 2 * nview, 2 * nview);
			mat_mul(Bt, t2, t1, 1, 2 * nview, 1);
			mse = t1[0] / (2 * nview - 3);

			for (jj = 0; jj < 3; jj++)
				for (kk = 0; kk < 3; kk++)
					Cov[kk + jj * 3] = iAtA[kk + jj * 3] * mse;

			QR_Solution_Double(A, B, 2 * nview, 3);

			WC[ii].x = B[0];
			WC[ii].y = B[1];
			WC[ii].z = B[2];
		}

		delete[]At;
		delete[]Bt;
		delete[]t1;
		delete[]t2;
		delete[]Identity;
	}

	if (MenCreated)
		delete[]A, delete[]B;

	return;
}
void ProjectandDistort(CPoint3 WC, CPoint2 *pts, double *P, double *camera, double *distortion, int nCam)
{
	int ii;
	double num1, num2, denum;

	for (ii = 0; ii < nCam; ii++)
	{
		num1 = P[ii * 12 + 0] * WC.x + P[ii * 12 + 1] * WC.y + P[ii * 12 + 2] * WC.z + P[ii * 12 + 3];
		num2 = P[ii * 12 + 4] * WC.x + P[ii * 12 + 5] * WC.y + P[ii * 12 + 6] * WC.z + P[ii * 12 + 7];
		denum = P[ii * 12 + 8] * WC.x + P[ii * 12 + 9] * WC.y + P[ii * 12 + 10] * WC.z + P[ii * 12 + 11];

		pts[ii].x = num1 / denum, pts[ii].y = num2 / denum;
		LensDistortion_Point(pts[ii], camera + ii * 9, distortion + ii * 13);
	}

	return;
}
double Compute_Homo(CPoint5 *uvxy, int planar_pts, Matrix&Homo, CPoint5 *s_pts, double *M)
{
	//Map from xy to uv: uv = H*xy
	int ii;
	double scxx, scyy, u_a = 0.0, v_a = 0.0, x_a = 0.0, y_a = 0.0, z_a = 0.0;
	bool createMem = false;
	if (s_pts == NULL)
	{
		M = new double[2 * planar_pts * 9];
		s_pts = new CPoint5[planar_pts];
		createMem = true;
	}
	Matrix Ti(3, 3), Tw(3, 3);

	//1. Normalize all planar pts.
	for (ii = 0; ii < planar_pts; ii++)
	{
		u_a += uvxy[ii].u;
		v_a += uvxy[ii].v;
		x_a += uvxy[ii].x;
		y_a += uvxy[ii].y;
	}

	u_a = u_a / planar_pts, v_a = v_a / planar_pts;
	x_a = x_a / planar_pts, y_a = y_a / planar_pts;

	//a. Normalize (u,v)
	scxx = 0.0, scyy = 0.0;
	for (ii = 0; ii < planar_pts; ii++)
	{
		scxx += abs(uvxy[ii].u - u_a);
		scyy += abs(uvxy[ii].v - v_a);
	}
	scxx /= planar_pts;
	scyy /= planar_pts;

	Ti[0] = 1.0 / scxx, Ti[1] = 0.0, Ti[2] = -u_a / scxx;
	Ti[3] = 0.0, Ti[4] = 1.0 / scyy;	Ti[5] = -v_a / scyy;
	Ti[6] = 0.0, Ti[7] = 0.0, Ti[8] = 1.0;

	for (ii = 0; ii < planar_pts; ii++)
	{
		s_pts[ii].u = Ti[0] * uvxy[ii].u + Ti[2];
		s_pts[ii].v = Ti[4] * uvxy[ii].v + Ti[5];
	}

	//b. Normalize (x,y,z)
	scxx = 0.0, scyy = 0.0;
	for (ii = 0; ii < planar_pts; ii++)
	{
		scxx += abs(uvxy[ii].x - x_a);
		scyy += abs(uvxy[ii].y - y_a);
	}
	scxx /= planar_pts;
	scyy /= planar_pts;

	Tw[0] = 1.0 / scxx, Tw[1] = 0.0, Tw[2] = -x_a / scxx;
	Tw[3] = 0.0, Tw[4] = 1.0 / scyy, Tw[5] = -y_a / scyy;
	Tw[6] = 0.0, Tw[7] = 0.0, Tw[8] = 1.0;

	for (ii = 0; ii < planar_pts; ii++)
	{
		s_pts[ii].x = Tw[0] * uvxy[ii].x + Tw[2];
		s_pts[ii].y = Tw[4] * uvxy[ii].y + Tw[5];
	}

	//2. DLT
	double uu, vv, xx, yy;
	for (ii = 0; ii < planar_pts; ii++)
	{
		uu = s_pts[ii].u;
		vv = s_pts[ii].v;
		xx = s_pts[ii].x;
		yy = s_pts[ii].y;

		M[2 * ii * 9 + 0] = 0.0, M[2 * ii * 9 + 1] = 0.0, M[2 * ii * 9 + 2] = 0.0;
		M[2 * ii * 9 + 3] = -xx, M[2 * ii * 9 + 4] = -yy, M[2 * ii * 9 + 5] = -1.0;
		M[2 * ii * 9 + 6] = vv*xx, M[2 * ii * 9 + 7] = vv*yy, M[2 * ii * 9 + 8] = vv;

		M[(2 * ii + 1) * 9 + 0] = xx, M[(2 * ii + 1) * 9 + 1] = yy, M[(2 * ii + 1) * 9 + 2] = 1.0;
		M[(2 * ii + 1) * 9 + 3] = 0.0, M[(2 * ii + 1) * 9 + 4] = 0.0, M[(2 * ii + 1) * 9 + 5] = 0.0;
		M[(2 * ii + 1) * 9 + 6] = -uu*xx, M[(2 * ii + 1) * 9 + 7] = -uu*yy, M[(2 * ii + 1) * 9 + 8] = -uu;
	}
	Mat cvM = Mat(2 * planar_pts, 9, CV_64FC1, M);
	Mat H = Mat(9, 1, CV_64FC1);
	SVD::solveZ(cvM, H);

	for (ii = 0; ii < 9; ii++)
		Homo[ii] = H.at<double>(ii, 0);

	Homo = Ti.Inversion()*Homo*Tw;

	double wx, wy, u, v, U, V, A, B, C, err = 0;
	for (ii = 0; ii < 9; ii++)
		Homo[ii] /= Homo[8];

	double h11 = Homo[0], h12 = Homo[1], h13 = Homo[2];
	double h21 = Homo[3], h22 = Homo[4], h23 = Homo[5];
	double h31 = Homo[6], h32 = Homo[7], h33 = Homo[8];

	for (ii = 0; ii < planar_pts; ii++)
	{
		wx = uvxy[ii].x, wy = uvxy[ii].y;
		u = uvxy[ii].u, v = uvxy[ii].v;

		A = h11*wx + h12*wy + h13;
		B = h21*wx + h22*wy + h23;
		C = h31*wx + h32*wy + h33;

		U = A / C, V = B / C;
		err += (U - u)*(U - u) + (V - v)*(V - v);
	}

	if (createMem)
	{
		delete[]s_pts;
		delete[]M;
	}
	return sqrt(err / planar_pts);
}
double Compute_AffineHomo(CPoint2 *From, CPoint2 *To, int npts, double *Affine, double *A, double *B)
{
	//To = H*From
	int ii;
	bool createMem = false;
	if (A == NULL)
	{
		createMem = true;
		A = new double[npts * 3];
		B = new double[npts];
	}

	for (ii = 0; ii < npts; ii++)
		A[3 * ii] = From[ii].x, A[3 * ii + 1] = From[ii].y, A[3 * ii + 2] = 1.0, B[ii] = To[ii].x;
	LS_Solution_Double(A, B, npts, 3);
	Affine[0] = B[0], Affine[1] = B[1], Affine[2] = B[2];

	for (ii = 0; ii < npts; ii++)
		A[3 * ii] = From[ii].x, A[3 * ii + 1] = From[ii].y, A[3 * ii + 2] = 1.0, B[ii] = To[ii].y;
	LS_Solution_Double(A, B, npts, 3);
	Affine[3] = B[0], Affine[4] = B[1], Affine[5] = B[2];

	double error = 0.0, errorx, errory;
	for (ii = 0; ii < npts; ii++)
	{
		errorx = (Affine[0] * From[ii].x + Affine[1] * From[ii].y + Affine[2] - To[ii].x);
		errory = (Affine[3] * From[ii].x + Affine[4] * From[ii].y + Affine[5] - To[ii].y);
		error += errorx*errorx + errory*errory;
	}

	if (createMem)
	{
		delete[]A;
		delete[]B;
	}
	return error / npts;
}
double ComputeSSIG(double *Para, int x, int y, int hsubset, int width, int height, int nchannels, int InterpAlgo)
{
	int ii, jj, kk, length = width*height;
	double S[3], ssig = 0.0;

	for (kk = 0; kk < nchannels; kk++)
	{
		for (jj = -hsubset; jj <= hsubset; jj++)
		{
			for (ii = -hsubset; ii <= hsubset; ii++)
			{
				Get_Value_Spline(Para + kk*length, width, height, x + ii, y + jj, S, 0, InterpAlgo);
				ssig += S[1] * S[1] + S[2] * S[2];
			}
		}
	}

	return ssig / (2 * hsubset + 1) / (2 * hsubset + 1);
}
bool IsLocalWarpAvail(float *WarpingParas, double *iWp, int startX, int startY, CPoint &startf, int &xx, int &yy, int &range, int width, int height, int nearestRange = 7)
{
	int id, kk, ll, mm, nn, kk2, length = width*height;
	int sRange = width / 500;

	bool flag = 0;
	for (kk = 0; kk < nearestRange && !flag; kk++)
	{
		kk2 = kk*kk;
		for (mm = -kk; mm <= kk && !flag; mm++)
		{
			for (nn = -kk; nn <= kk; nn++)
			{
				if (mm*mm + nn*nn < kk2)
					continue;
				if (abs(WarpingParas[(startX + nn) + (startY + mm)*width]) + abs(WarpingParas[(startX + nn) + (startY + mm)*width + length]) > 0.001)
				{
					xx = startX + nn + (int)(WarpingParas[(startX + nn) + (startY + mm)*width] + 0.5);
					yy = startY + mm + (int)(WarpingParas[(startX + nn) + (startY + mm)*width + length] + 0.5);
					flag = true, startf.x = startX + nn, startf.y = startY + mm, range = kk; //Adaptively change the search range

					//Get the affine coeffs:
					id = (startX + nn) + (startY + mm)*width;
					for (ll = 2; ll < 6; ll++)
						iWp[ll - 2] = WarpingParas[id + ll*length];
					break;
				}
			}
		}
	}

	if (!flag)
		return false;
	else
		return true;
}

void synthesize_square_mask(double *square_mask_smooth, int *pattern_bi_graylevel, int Pattern_size, double sigma, int flag, bool OpenMP)
{
	//numPatterns = 1: 0 deg squares, numPatterns =2: 0, 45 deg squares
	int ii, jj, Pattern_length = Pattern_size*Pattern_size;
	int es_con_x = Pattern_size / 2;
	int es_con_y = Pattern_size / 2;
	char dark = (char)pattern_bi_graylevel[0];
	char bright = (char)pattern_bi_graylevel[1];
	char mid = (char)((pattern_bi_graylevel[0] + pattern_bi_graylevel[1]) / 2);

	char *square_mask = new char[Pattern_length];

	for (jj = 0; jj < Pattern_size; jj++)
	{
		for (ii = 0; ii < Pattern_size; ii++)
		{
			if ((ii<es_con_x && jj<es_con_y) || (ii>es_con_x && jj>es_con_y))
				square_mask[ii + jj*Pattern_size] = (flag == 0 ? bright : dark);
			else if (ii == es_con_x || jj == es_con_y)
				square_mask[ii + jj*Pattern_size] = mid;
			else
				square_mask[ii + jj*Pattern_size] = (flag == 0 ? dark : bright);
		}
	}
	// Gaussian smooth
	Gaussian_smooth(square_mask, square_mask_smooth, Pattern_size, Pattern_size, pattern_bi_graylevel[1], sigma);
	delete[]square_mask;
	return;
}
double ComputeZNCCPatch(double *RefPatch, double *TarPatch, int hsubset, int nchannels, double *T = NULL)
{
	int i, kk, iii, jjj;

	FILE *fp1, *fp2;
	bool printout = false;
	if (printout)
	{
		fp1 = fopen("C:/temp/src.txt", "w+");
		fp2 = fopen("C:/temp/tar.txt", "w+");
	}

	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	bool createMem = false;
	if (T == NULL)
	{
		createMem = true;
		T = new double[2 * Tlength*nchannels];
	}
	double ZNCC = 0.0;

	int m = 0;
	double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
	for (jjj = 0; jjj < TimgS; jjj++)
	{
		for (iii = 0; iii < TimgS; iii++)
		{
			for (kk = 0; kk < nchannels; kk++)
			{
				i = iii + jjj*TimgS + kk*Tlength;
				T[2 * m] = RefPatch[i], T[2 * m + 1] = TarPatch[i];
				t_f += T[2 * m], t_g += T[2 * m + 1];

				if (printout)
					fprintf(fp1, "%.4f ", T[2 * m]), fprintf(fp2, "%.4f ", T[2 * m + 1]);
				m++;
			}
		}
		if (printout)
		{
			fprintf(fp1, "\n"), fprintf(fp2, "\n");
		}
	}
	if (printout)
	{
		fclose(fp1), fclose(fp2);
	}

	t_f = t_f / m;
	t_g = t_g / m;
	t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
	for (i = 0; i < m; i++)
	{
		t_4 = T[2 * i] - t_f, t_5 = T[2 * i + 1] - t_g;
		t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
	}

	t_2 = sqrt(t_2*t_3);
	if (t_2 < 1e-10)
		t_2 = 1e-10;

	ZNCC = t_1 / t_2; //This is the zncc score
	if (abs(ZNCC) > 1.0)
		ZNCC = 0.0;

	if (createMem)
		delete[]T;

	return ZNCC;
}
double ComputeZNCC(double *src, double *dst, CPoint *Pts, int hsubset, int width1, int height1, int width2, int height2, double *T = 0)
{
	//This is basically a zncc computinf
	int i, j, ii, jj, II, JJ, m;
	double m_F, m_G, t_1, t_2, t_3, t_4, t_5, t_f, t_g;
	int Fx = Pts[0].x, Fy = Pts[0].y, Tx = Pts[1].x, Ty = Pts[1].y;

	m = 0, t_f = 0.0, t_g = 0.0;
	bool createMem = false;
	if (T == NULL)
	{
		createMem = true;
		T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)];
	}
	for (j = -hsubset; j <= hsubset; j++)
	{
		for (i = -hsubset; i <= hsubset; i++)
		{
			ii = Fx + i;
			jj = Fy + j;
			II = Tx + i;
			JJ = Ty + j;

			m_F = src[ii + jj*width1];
			m_G = dst[II + JJ*width2];

			T[2 * m] = m_F;
			T[2 * m + 1] = m_G;
			t_f += m_F;
			t_g += m_G;
			m++;
		}
	}

	t_f = 1.0*t_f / (m + 1);
	t_g = 1.0*t_g / (m + 1);
	t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
	for (i = 0; i < m; i++)
	{
		t_4 = T[2 * i] - t_f;
		t_5 = T[2 * i + 1] - t_g;
		t_1 += 1.0*t_4*t_5;
		t_2 += 1.0*t_4*t_4;
		t_3 += 1.0*t_5*t_5;
	}

	t_2 = sqrt(t_2*t_3);
	if (t_2 < 1e-10)
		t_2 = 1e-10;

	t_3 = t_1 / t_2;
	if (abs(t_3) > 1.0)
		t_3 = 0.0;

	if (createMem)
		delete[]T;

	return t_3;
}
double ComputeZNCCInterp(double *RefPara, double *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, CPoint2 PR, CPoint2 PT, int InterpAlgo)
{
	int i, kk, iii, jjj;
	double II1, JJ1, II2, JJ2, S[3];

	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS, RefLength = widthRef*heightRef, TarLength = widthTar*heightTar;

	FILE *fp1, *fp2;
	bool printout = false;
	if (printout)
	{
		fp1 = fopen("C:/temp/src.txt", "w+");
		fp2 = fopen("C:/temp/tar.txt", "w+");
	}

	double *T = new double[2 * Tlength*nchannels];
	double ZNCC;
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II1 = PR.x + iii;
				JJ1 = PR.y + jjj;

				II2 = PT.x + iii;
				JJ2 = PT.y + jjj;

				if (II1<0.0 || II1>(double)(widthRef - 1) || JJ1<0.0 || JJ1>(double)(heightRef - 1))
					continue;

				if (II2<0.0 || II2>(double)(widthTar - 1) || JJ2<0.0 || JJ2>(double)(heightTar - 1))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(RefPara + kk*RefLength, widthRef, heightRef, II1, JJ1, S, -1, InterpAlgo);
					T[2 * m] = S[0];
					if (printout)
						fprintf(fp1, "%.4f ", S[0]);

					Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II2, JJ2, S, -1, InterpAlgo);
					T[2 * m + 1] = S[0];
					if (printout)
						fprintf(fp2, "%.4f ", S[0]);

					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
			{
				fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
		}
		if (printout)
		{
			fclose(fp1), fclose(fp2);
		}

		t_f = t_f / m;
		t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f;
			t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) > 1.0)
			ZNCC = 0.0;
	}
	delete[]T;

	return ZNCC;
}
double TMatching(double *RefPara, double *TarPara, int hsubset, int widthRef, int heightRef, int widthTar, int heightTar, int nchannels, CPoint2 PR, CPoint2 PT, int advanced_tech, int Convergence_Criteria, double ZNCCThreshold, int Iter_Max, int InterpAlgo, double *fufv, bool greedySearch, double *ShapePara, double *oPara, double *Timg, double *T)
{
	//Also a fine ImgRef matching,.... some differences in the input as compared to TMatchingFine though
	// NOTE: initial guess is of the form of the homography

	int i, j, k, ii, kk, iii, jjj;
	double II, JJ, a, b, gx, gy, PSSDab, PSSDab_min, t_1, t_2, t_3, t_4, t_5, t_6, m_F, m_G, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 8, 14 }, P_Jump_Incr[] = { 1, 2 };
	int nn = NN[advanced_tech - 1], nExtraParas = 2, _iter = 0;
	int p_jump, p_jump_0 = 1, p_jump_incr = P_Jump_Incr[advanced_tech - 1];
	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS, RefLength = widthRef*heightRef, TarLength = widthTar*heightTar;

	double 	AA[196 * 196], BB[14], CC[14], p[14], p_best[14];
	if (ShapePara == NULL)
	{
		for (ii = 0; ii < nn; ii++)
			p[ii] = (ii == nn - 2 ? 1.0 : 0.0);
	}
	else
	{
		if (advanced_tech == 1) //These are basically taylor approximation of the denumerator
		{
			p[0] = ShapePara[2] - PT.x;
			p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = 1.0;
			p[7] = 0.0;
		}
		else
		{
			p[0] = ShapePara[2] - PT.x, p[1] = ShapePara[5] - PT.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = -0.5*ShapePara[0] * ShapePara[6];
			p[7] = -0.5*ShapePara[1] * ShapePara[7];
			p[8] = -(ShapePara[0] * ShapePara[7] + ShapePara[1] * ShapePara[6]);
			p[9] = -0.5*ShapePara[3] * ShapePara[6];
			p[10] = -0.5*ShapePara[4] * ShapePara[7];
			p[11] = -(ShapePara[3] * ShapePara[7] + ShapePara[4] * ShapePara[6]);
			p[12] = 1.0, p[13] = 0.0;
		}
	}

	bool createMem = false;
	if (Timg == NULL)
	{
		Timg = new double[Tlength*nchannels];
		T = new double[2 * Tlength*nchannels];
		createMem = true;
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			II = PR.x + iii;
			JJ = PR.y + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(RefPara + kk*RefLength, widthRef, heightRef, II, JJ, S, -1, InterpAlgo);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp = 0;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	if (greedySearch)
	{
		/// Let's start with only translation and only match the at the highest level of the pyramid
		double zncc, znccMin;
		for (p_jump = p_jump_0; p_jump > 0; p_jump -= (advanced_tech == 0 ? 1 : 2))
		{
			znccMin = 1e10;
			for (k = 0; k < Iter_Max; k++)
			{
				t_1 = 0.0;
				t_2 = 0.0;
				for (i = 0; i < 4; i++)
					AA[i] = 0.0;
				for (i = 0; i < 2; i++)
					BB[i] = 0.0;

				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						II = PT.x + iii + p[0];
						JJ = PT.y + jjj + p[1];

						if (II<0.0 || II>(double)(widthTar - 2) || JJ<0.0 || JJ>(double)(heightTar - 2))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);

							m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
							m_G = S[0];

							t_3 = m_G - m_F;
							CC[0] = S[1];
							CC[1] = S[2];

							for (i = 0; i < 2; i++)
								BB[i] += t_3*CC[i];

							for (j = 0; j < 2; j++)
								for (i = 0; i < 2; i++)
									AA[j * 2 + i] += CC[i] * CC[j];

							t_1 += t_3*t_3;
							t_2 += m_F*m_F;
						}
					}
				}
				zncc = t_1 / t_2;

				QR_Solution_Double(AA, BB, 2, 2);
				for (i = 0; i < 2; i++)
					p[i] -= BB[i];

				if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
				{
					if (createMem)
					{
						delete[]Timg;
						delete[]T;
					}
					return 0.0;
				}

				if (zncc < znccMin)	// If the iteration does not converge, this can be helpful
				{
					znccMin = zncc;
					p_best[0] = p[0], p_best[1] = p[1];
				}

				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					break;
			}
		}
		p[0] = p_best[0], p[1] = p_best[1];
	}

	/// DIC Iteration: Begin
	bool Break_Flag;
	PSSDab_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0;
			t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2];
			b = p[nn - 1];
			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj;
					JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;

					if (II<5.0 || II>(double)(widthTar - 5) || JJ<5.0 || JJ>(double)(heightTar - 5))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, 0, InterpAlgo);
						m_G = S[0], gx = S[1], gy = S[2];

						if (printout)
							fprintf(fp, "%.2f ", m_G);

						t_3 = a*m_G + b - m_F, t_4 = a;

						t_5 = t_4*gx, t_6 = t_4*gy;
						CC[0] = t_5, CC[1] = t_6;
						CC[2] = t_5*iii, CC[3] = t_5*jjj;
						CC[4] = t_6*iii, CC[5] = t_6*jjj;
						CC[6] = m_G, CC[7] = 1.0;

						for (j = 0; j < nn; j++)
							BB[j] += t_3*CC[j];

						for (j = 0; j < nn; j++)
							for (i = 0; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j];

						t_1 += t_3*t_3;
						t_2 += m_F*m_F;
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			PSSDab = t_1 / t_2;
			if (t_2 < 10.0e-9)
				break;

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || !IsFiniteNumber(p[0]))
			{
				if (createMem)
				{
					delete[]Timg;
					delete[]T;
				}
				return 0.0;
			}

			if (PSSDab < PSSDab_min)	// If the iteration does not converge, this can be helpful
			{
				PSSDab_min = PSSDab;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
			}

			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn - nExtraParas; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn - nExtraParas)
					Break_Flag = true;
			}

			if (Break_Flag)
				break;
		}
		_iter += k;

		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! 
	//They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	double ZNCC;
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II = PT.x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ = PT.y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(widthTar - 1) || JJ<0.0 || JJ>(double)(heightTar - 1))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(TarPara + kk*TarLength, widthTar, heightTar, II, JJ, S, -1, InterpAlgo);
					if (printout)
						fprintf(fp, "%.4f ", S[0]);

					T[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
					T[2 * m + 1] = S[0];
					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m;
		t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f;
			t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) > 1.0)
			ZNCC = 0.0;
	}

	if (abs(p[0]) > 0.005*widthTar || abs(p[1]) > 0.005*widthTar || p[0] != p[0] || p[1] != p[1] || ZNCC < ZNCCThreshold)
	{
		if (createMem)
		{
			delete[]Timg;
			delete[]T;
		}
		return 0.0;
	}

	fufv[0] = p_best[0], fufv[1] = p_best[1];
	if (oPara != NULL)
		for (ii = 0; ii < 8; ii++)
			oPara[ii] = p_best[ii];

	if (createMem)
	{
		delete[]Timg;
		delete[]T;
	}
	return ZNCC;
}
void TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, char *Image, int width, int height, CPoint POI, int search_area, double thresh, double &zncc)
{
	//No interpolation at all, just slide the template around to compute the ZNCC
	int m, i, j, ii, jj, iii, jjj, II, JJ, length = width*height;
	double t_f, t_g, t_1, t_2, t_3, t_4, t_5, m_F, m_G;

	CPoint2 w_pt, ima_pt;
	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	FILE *fp1, *fp2;
	bool printout = false;


	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)];
	zncc = 0.0;
	for (j = -search_area; j <= search_area; j++)
	{
		for (i = -search_area; i <= search_area; i++)
		{
			m = -1;
			t_f = 0.0;
			t_g = 0.0;

			if (printout)
			{
				fp1 = fopen("C:/temp/src.txt", "w+");
				fp2 = fopen("C:/temp/tar.txt", "w+");
			}

			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					jj = Pattern_cen_y + jjj;
					ii = Pattern_cen_x + iii;

					JJ = POI.y + jjj + j;
					II = POI.x + iii + i;

					m_F = Pattern[ii + jj*pattern_size];
					m_G = (double)((int)((unsigned char)(Image[II + JJ*width])));

					if (printout)
					{
						fprintf(fp1, "%.2f ", m_F);
						fprintf(fp2, "%.2f ", m_G);
					}
					m++;
					*(T + 2 * m + 0) = m_F;
					*(T + 2 * m + 1) = m_G;
					t_f += m_F;
					t_g += m_G;
				}
				if (printout)
				{
					fprintf(fp1, "\n");
					fprintf(fp2, "\n");
				}
			}
			if (printout)
			{
				fclose(fp1); fclose(fp2);
			}

			t_f = t_f / (m + 1);
			t_g = t_g / (m + 1);
			t_1 = 0.0;
			t_2 = 0.0;
			t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = *(T + 2 * iii + 0) - t_f;
				t_5 = *(T + 2 * iii + 1) - t_g;
				t_1 += (t_4*t_5);
				t_2 += (t_4*t_4);
				t_3 += (t_5*t_5);
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			t_3 = t_1 / t_2;

			if (t_3 > 1.0 || t_3 < -1.0)
				t_3 = 0.0;

			if (t_3>thresh && t_3 > zncc)
				zncc = t_3;
			else if (t_3 < -thresh && t_3 < zncc)
				zncc = t_3;
		}
	}

	zncc = abs(zncc);

	delete[]T;
	return;
}
void TMatchingSuperCoarse(double *Pattern, int pattern_size, int hsubset, double *Image, int width, int height, CPoint POI, int search_area, double thresh, double &zncc)
{
	//No interpolation at all, just slide the template around to compute the ZNCC
	int m, i, j, ii, jj, iii, jjj, II, JJ, length = width*height;
	double t_f, t_g, t_1, t_2, t_3, t_4, t_5, m_F, m_G;

	CPoint2 w_pt, ima_pt;
	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	FILE *fp1, *fp2;
	bool printout = false;


	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)];
	zncc = 0.0;
	for (j = -search_area; j <= search_area; j++)
	{
		for (i = -search_area; i <= search_area; i++)
		{
			m = -1;
			t_f = 0.0;
			t_g = 0.0;

			if (printout)
			{
				fp1 = fopen("C:/temp/src.txt", "w+");
				fp2 = fopen("C:/temp/tar.txt", "w+");
			}

			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					jj = Pattern_cen_y + jjj;
					ii = Pattern_cen_x + iii;

					JJ = POI.y + jjj + j;
					II = POI.x + iii + i;

					m_F = Pattern[ii + jj*pattern_size];
					m_G = Image[II + JJ*width];

					if (printout)
					{
						fprintf(fp1, "%.2f ", m_F);
						fprintf(fp2, "%.2f ", m_G);
					}
					m++;
					*(T + 2 * m + 0) = m_F;
					*(T + 2 * m + 1) = m_G;
					t_f += m_F;
					t_g += m_G;
				}
				if (printout)
				{
					fprintf(fp1, "\n");
					fprintf(fp2, "\n");
				}
			}
			if (printout)
			{
				fclose(fp1); fclose(fp2);
			}

			t_f = t_f / (m + 1);
			t_g = t_g / (m + 1);
			t_1 = 0.0;
			t_2 = 0.0;
			t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = *(T + 2 * iii + 0) - t_f;
				t_5 = *(T + 2 * iii + 1) - t_g;
				t_1 += (t_4*t_5);
				t_2 += (t_4*t_4);
				t_3 += (t_5*t_5);
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			t_3 = t_1 / t_2;

			if (t_3 > 1.0 || t_3 < -1.0)
				t_3 = 0.0;

			if (t_3>thresh && t_3 > zncc)
				zncc = t_3;
			else if (t_3 < -thresh && t_3 < zncc)
				zncc = t_3;
		}
	}

	zncc = abs(zncc);

	delete[]T;
	return;
}
int TMatchingCoarse(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, CPoint2 &POI, int search_area, double thresh, double &zncc, int InterpAlgo, double *InitPara, double *maxZNCC)
{
	//Compute the zncc in a local region (5x5). No iteration is used to solve for shape parameters
	//InitPara: 3x3 homography matrix
	int i, j, ii, jj, iii, jjj, length = width*height, pjump = search_area > 5 ? 2 : 1;
	double II, JJ, t_1, t_2, t_3, t_4, m_F, m_G, S[6];

	CPoint2 w_pt, ima_pt;
	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;
	int m;
	double t_f, t_g, t_5, xxx = 0.0, yyy = 0.0;
	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)];

	zncc = 0.0;
	for (j = -search_area; j <= search_area; j += pjump)
	{
		for (i = -search_area; i <= search_area; i += pjump)
		{
			m = -1;
			t_f = 0.0, t_g = 0.0;

			if (printout)
				fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					jj = Pattern_cen_y + jjj;
					ii = Pattern_cen_x + iii;

					if (InitPara == NULL)
					{
						II = (int)(POI.x + 0.5) + iii + i;
						JJ = (int)(POI.y + 0.5) + jjj + j;
					}
					else
					{
						II = (InitPara[0] * iii + InitPara[1] * jjj + InitPara[2]) / (InitPara[6] * iii + InitPara[7] * jjj + InitPara[8]);
						JJ = (InitPara[3] * iii + InitPara[4] * jjj + InitPara[5]) / (InitPara[6] * iii + InitPara[7] * jjj + InitPara[8]);
					}

					Get_Value_Spline(Para, width, height, II, JJ, S, -1, InterpAlgo);

					m_F = Pattern[ii + jj*pattern_size];
					m_G = S[0];
					m++;
					*(T + 2 * m + 0) = m_F, *(T + 2 * m + 1) = m_G;
					t_f += m_F, t_g += m_G;
					if (printout)
						fprintf(fp1, "%.2f ", m_G), fprintf(fp2, "%.2f ", m_F);
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			t_f = t_f / (m + 1), t_g = t_g / (m + 1);
			t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = *(T + 2 * iii + 0) - t_f, t_5 = *(T + 2 * iii + 1) - t_g;
				t_1 += (t_4*t_5), t_2 += (t_4*t_4), t_3 += (t_5*t_5);
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			t_3 = t_1 / t_2;
			if (t_3 > 1.0 || t_3 < -1.0)
				t_3 = 0.0;

			if (t_3>thresh && t_3 > zncc)
			{
				zncc = t_3;
				xxx = i, yyy = j;
			}
			else if (t_3 < -thresh && abs(t_3) > zncc)
			{
				zncc = t_3;
				xxx = i, yyy = j;
			}
		}
	}
	if (InitPara != NULL)
		maxZNCC[0] = abs(zncc);

	delete[]T;
	if (zncc > thresh)
	{
		POI.x = (int)(POI.x + 0.5) + xxx;
		POI.y = (int)(POI.y + 0.5) + yyy;
		zncc = abs(zncc);
		return 0;
	}
	else if (zncc < -thresh)
	{
		POI.x = (int)(POI.x + 0.5) + xxx;
		POI.y = (int)(POI.y + 0.5) + yyy;
		zncc = abs(zncc);
		return 1;
	}
	else
		return -1;
}
double TMatchingFine_ZNCC(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, CPoint2 &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *Znssd_reqd)
{
	int i, j, k, m, ii, jj, iii, jjj, iii2, jjj2;
	double II, JJ, iii_n, jjj_n, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, m_F, m_G, t_f, t_ff, t_g, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.1;
	int NN[] = { 6, 12 }, P_Jump_Incr[] = { 1, 1 };
	int nn = NN[advanced_tech], nExtraParas = 2, _iter = 0, Iter_Max = 50;
	int p_jump, p_jump_0 = 1, p_jump_incr = P_Jump_Incr[advanced_tech];

	double AA[144], BB[12], CC[12];

	bool createMem = false;
	if (Znssd_reqd == NULL)
	{
		createMem = true;
		Znssd_reqd = new double[6 * (2 * hsubset + 1)*(2 * hsubset + 1)];
	}

	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	double p[12], p_best[12];
	for (i = 0; i < 12; i++)
		p[i] = 0.0;

	nn = NN[advanced_tech];
	int pixel_increment_in_subset[] = { 1, 2, 2, 3 };

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;

	/// Iteration: Begin
	bool Break_Flag = false;
	DIC_Coeff_min = 4.0;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		for (k = 0; k < Iter_Max; k++)
		{
			m = -1;
			t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < 144; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < 12; iii++)
				BB[iii] = 0.0;

			if (printout)
				fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					ii = Pattern_cen_x + iii, jj = Pattern_cen_y + jjj;

					if (ii<0 || ii>(width - 1) || jj<0 || jj>(height - 1))
						continue;

					iii2 = iii*iii, jjj2 = jjj*jjj;
					if (advanced_tech == 0)
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					}
					else if (advanced_tech == 1)
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * iii*jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * iii*jjj;
					}

					if (II<0.0 || II>(double)(width - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height - 1) - (1e-10))
						continue;

					Get_Value_Spline(Para, width, height, II, JJ, S, 0, InterpAlgo);
					m_F = Pattern[ii + jj*pattern_size];
					m_G = S[0], gx = S[1], gy = S[2];
					m++;

					Znssd_reqd[6 * m + 0] = m_F, Znssd_reqd[6 * m + 1] = m_G;
					Znssd_reqd[6 * m + 2] = gx, Znssd_reqd[6 * m + 3] = gy;
					Znssd_reqd[6 * m + 4] = (double)iii, Znssd_reqd[6 * m + 5] = (double)jjj;
					t_1 += m_F, t_2 += m_G;

					if (printout)
						fprintf(fp1, "%e ", m_F), fprintf(fp2, "%e ", m_G);
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			if (k == 0)
			{
				t_f = t_1 / (m + 1);
				t_1 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[6 * iii + 0] - t_f;
					t_1 += t_4*t_4;
				}
				t_ff = sqrt(t_1);
			}

			t_g = t_2 / (m + 1);
			t_2 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_5 = Znssd_reqd[6 * iii + 1] - t_g;
				t_2 += t_5*t_5;
			}
			t_2 = sqrt(t_2);

			DIC_Coeff = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[6 * iii + 0] - t_f;
				t_5 = Znssd_reqd[6 * iii + 1] - t_g;
				t_6 = t_5 / t_2 - t_4 / t_ff;
				t_3 = t_6 / t_2;
				gx = Znssd_reqd[6 * iii + 2], gy = Znssd_reqd[6 * iii + 3];
				iii_n = Znssd_reqd[6 * iii + 4], jjj_n = Znssd_reqd[6 * iii + 5];
				CC[0] = gx, CC[1] = gy;
				CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
				CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
				if (advanced_tech == 1)
				{
					CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
					CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
				}
				for (j = 0; j < nn; j++)
				{
					BB[j] += t_3*CC[j];
					for (i = 0; i < nn; i++)
						AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
				}

				DIC_Coeff += t_6*t_6;
			}

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (!IsNumber(p[0]) || abs(p[0]) > hsubset || abs(p[1]) > hsubset)
			{
				if (createMem)
					delete[]Znssd_reqd;
				return false;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
			}

			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn)
					Break_Flag = true;
			}

			if (Break_Flag)
				break;
		}
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// Iteration: End

	if (createMem)
		delete[]Znssd_reqd;
	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || p[0] != p[0] || p[1] != p[1] || 1.0 - 0.5*DIC_Coeff_min < ZNCCthresh)
		return false;

	POI.x += p[0], POI.y += p[1];

	return 1.0 - 0.5*DIC_Coeff_min;
}
bool TMatchingFine(double *Pattern, int pattern_size, int hsubset, double *Para, int width, int height, CPoint2 &POI, int advanced_tech, int Convergence_Criteria, double ZNCCthresh, int InterpAlgo, double *ShapePara, double *maxZNCC)
{
	// NOTE: initial guess is of the form of the homography

	int i, j, k, ii, jj, iii, jjj, iii2, jjj2;
	double II, JJ, a, b, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, m_F, m_G, S[6];
	double conv_crit_1 = pow(10.0, -Convergence_Criteria - 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 8, 14 }, P_Jump_Incr[] = { 1, 2 };
	int nn = NN[advanced_tech - 1], nExtraParas = 2, _iter = 0, Iter_Max = 100;
	int p_jump, p_jump_0 = 1, p_jump_incr = P_Jump_Incr[advanced_tech - 1];

	int Pattern_cen_x = pattern_size / 2;
	int Pattern_cen_y = pattern_size / 2;

	double *AA = new double[nn*nn];
	double *BB = new double[nn];
	double *CC = new double[nn];
	//double *p = new double[nn];
	//double *p_best = new double[nn];
	double p[14], p_best[14];
	if (ShapePara == NULL)
		for (ii = 0; ii < nn; ii++)
			p[ii] = (ii == nn - 2 ? 1.0 : 0.0);
	else
	{
		if (advanced_tech == 1)
		{
			p[0] = ShapePara[2] - POI.x;
			p[1] = ShapePara[5] - POI.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = 1.0, p[7] = 0.0;
		}
		else
		{
			p[0] = ShapePara[2] - POI.x;
			p[1] = ShapePara[5] - POI.y;
			p[2] = ShapePara[0] - ShapePara[2] * ShapePara[6] - 1.0;
			p[3] = ShapePara[1] - ShapePara[2] * ShapePara[7];
			p[4] = ShapePara[3] - ShapePara[5] * ShapePara[6];
			p[5] = ShapePara[4] - ShapePara[5] * ShapePara[7] - 1.0;
			p[6] = -0.5*ShapePara[0] * ShapePara[6];
			p[7] = -0.5*ShapePara[1] * ShapePara[7];
			p[8] = -(ShapePara[0] * ShapePara[7] + ShapePara[1] * ShapePara[6]);
			p[9] = -0.5*ShapePara[3] * ShapePara[6];
			p[10] = -0.5*ShapePara[4] * ShapePara[7];
			p[11] = -(ShapePara[3] * ShapePara[7] + ShapePara[4] * ShapePara[6]);
			p[12] = 1.0, p[13] = 0.0;
		}
	}

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;

	/// DIC Iteration: Begin
	bool Break_Flag;
	DIC_Coeff_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2], b = p[nn - 1];

			if (printout)
				fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					jj = Pattern_cen_y + jjj;
					ii = Pattern_cen_x + iii;
					iii2 = iii*iii, jjj2 = jjj*jjj;

					if (advanced_tech == 1)
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					}
					else
					{
						II = POI.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * iii*jjj;
						JJ = POI.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * iii*jjj;
					}

					if (II<0.0 || II>(double)(width - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height - 1) - (1e-10))
						continue;

					Get_Value_Spline(Para, width, height, II, JJ, S, 0, InterpAlgo);

					m_F = Pattern[ii + jj*pattern_size];
					m_G = S[0];
					gx = S[1], gy = S[2];

					t_3 = a*m_G + b - m_F;
					t_4 = a;

					if (printout)
						fprintf(fp1, "%e ", m_F), fprintf(fp2, "%e ", m_G);

					t_5 = t_4*gx, t_6 = t_4*gy;
					CC[0] = t_5, CC[1] = t_6, CC[2] = t_5*iii;
					if (advanced_tech == 1)
					{
						CC[3] = t_5*jjj;
						CC[4] = t_6*iii;
						CC[5] = t_6*jjj;
						CC[6] = m_G;
						CC[7] = 1.0;
					}
					else
					{
						CC[3] = t_5*jjj;
						CC[4] = t_6*iii;
						CC[5] = t_6*jjj;
						CC[6] = t_5*iii2*0.5;
						CC[7] = t_5*jjj2*0.5;
						CC[8] = t_5*iii*jjj;
						CC[9] = t_6*iii2*0.5;
						CC[10] = t_6*jjj2*0.5;
						CC[11] = t_6*iii*jjj;
						CC[12] = m_G;
						CC[13] = 1.0;
					}

					for (j = 0; j < nn; j++)
						BB[j] += t_3*CC[j];

					for (j = 0; j < nn; j++)
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];

					t_1 += t_3*t_3;
					t_2 += m_F*m_F;
				}
				if (printout)
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp1), fclose(fp2);

			DIC_Coeff = t_1 / t_2;
			if (t_2 < 1.0e-9)
				break;

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (!IsNumber(p[0]) || abs(p[0]) > hsubset || abs(p[1]) > hsubset)
			{
				delete[]CC, delete[]BB, delete[]AA;
				return false;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
			}

			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn - nExtraParas; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn - nExtraParas)
					Break_Flag = true;
			}

			if (Break_Flag)
				break;
		}
		_iter += k;

		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (iii = 0; iii < nn; iii++)
			p[iii] = p_best[iii];
	}
	/// DIC Iteration: End

	if (ShapePara != NULL)
	{
		for (ii = 0; ii < nn; ii++)
			ShapePara[ii] = p[ii];
		maxZNCC[0] = sqrt(1.0 - DIC_Coeff_min);
	}
	if (abs(p[0]) > 0.005*width || abs(p[1]) > 0.005*width || p[0] != p[0] || p[1] != p[1] || sqrt(1.0 - DIC_Coeff_min) < ZNCCthresh)
	{
		delete[]CC, delete[]BB, delete[]AA;
		return false;
	}

	POI.x += p[0], POI.y += p[1];

	delete[]CC, delete[]BB, delete[]AA;
	return true;
}
void DetectCornersHarris(char *img, int width, int height, CPoint2 *HarrisC, int &npts, double sigma, double sigmaD, double thresh, double alpha, int SuppressType, double AMN_thresh)
{
	int i, j, ii, jj;

	const int nMaxCorners = 100000;
	const double Pi = 3.1415926535897932;

	int size = 2 * (int)(3.0*sigma + 0.5) + 1, kk = (size - 1) / 2;
	double *GKernel = new double[size];
	double *GDKernel = new double[size];
	double t, sigma2 = sigma*sigma, sigma3 = sigma*sigma*sigma;

	for (ii = -kk; ii <= kk; ii++)
	{
		GKernel[ii + kk] = exp(-(ii*ii) / (2.0*sigma2)) / (sqrt(2.0*Pi)*sigma);
		GDKernel[ii + kk] = 1.0*ii*exp(-(ii*ii) / (2.0*sigma2)) / (sqrt(2.0*Pi)*sigma3);
	}

	//Compute Ix, Iy throught gaussian filters
	double *temp = new double[width*height];
	double *Ix = new double[width*height];
	double *Iy = new double[width*height];

	filter1D_row(GDKernel, size, img, temp, width, height);
	filter1D_col(GKernel, size, temp, Ix, width, height, t);

	filter1D_row(GKernel, size, img, temp, width, height);
	filter1D_col(GDKernel, size, temp, Iy, width, height, t);

	//Compute Ix2, Iy2, Ixy throught gaussian filters
	size = 2 * (int)(3.0*sigmaD + 0.5) + 1, kk = (size - 1) / 2;
	double *GKernel2 = new double[size];


	for (ii = -kk; ii <= kk; ii++)
		GKernel2[ii + kk] = exp(-(ii*ii) / (2.0*sigmaD*sigmaD)) / (sqrt(2.0*Pi)*sigmaD);

	double *Ix2 = new double[width*height];
	double *Iy2 = new double[width*height];
	double *Ixy = new double[width*height];

	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			Ix2[ii + jj*width] = Ix[ii + jj*width] * Ix[ii + jj*width];
			Iy2[ii + jj*width] = Iy[ii + jj*width] * Iy[ii + jj*width];
			Ixy[ii + jj*width] = Ix[ii + jj*width] * Iy[ii + jj*width];
		}
	}

	filter1D_row_Double(GKernel2, size, Ix2, temp, width, height);
	filter1D_col(GKernel2, size, temp, Ix2, width, height, t);

	filter1D_row_Double(GKernel2, size, Iy2, temp, width, height);
	filter1D_col(GKernel2, size, temp, Iy2, width, height, t);

	filter1D_row_Double(GKernel2, size, Ixy, temp, width, height);
	filter1D_col(GKernel2, size, temp, Ixy, width, height, t);

	double *tr = new double[width*height];
	double *Det = new double[width*height];
	for (jj = kk + 10; jj < height - kk - 10; jj++)
	{
		for (ii = kk + 10; ii < width - kk - 10; ii++)
		{
			tr[ii + jj*width] = Ix2[ii + jj*width] + Iy2[ii + jj*width];
			Det[ii + jj*width] = Ix2[ii + jj*width] * Iy2[ii + jj*width] - Ixy[ii + jj*width] * Ixy[ii + jj*width];
		}
	}

	double maxRes = 0.0;
	double *Cornerness = new double[width*height];
	for (jj = kk + 10; jj < height - kk - 10; jj++)
	{
		for (ii = kk + 10; ii < width - kk - 10; ii++)
		{
			Cornerness[ii + jj*width] = Det[ii + jj*width] - alpha*tr[ii + jj*width] * tr[ii + jj*width];
			if (maxRes < Cornerness[ii + jj*width])
				maxRes = Cornerness[ii + jj*width];
		}
	}
	/*
	FILE *fp = fopen("C:/temp/cornerness.txt", "w+");
	for(jj=kk+10; jj<height-kk-10; jj++)
	{
	for(ii=kk+10; ii<width-kk-10; ii++)
	{
	fprintf(fp, "%.4f ", Cornerness[ii+jj*width]);
	if(Cornerness[ii+jj*width] <thresh*maxRes)
	Cornerness[ii+jj*width] = 0.0;
	}
	fprintf(fp, "\n");
	}
	fclose(fp);
	*/
	int npotentialCorners = 0;
	int *potentialCorners = new int[2 * nMaxCorners];
	float *Response = new float[nMaxCorners];
	if (SuppressType == 0)
	{
		for (jj = kk + 10; jj < height - kk - 10; jj++)
		{
			for (ii = kk + 10; ii < width - kk - 10; ii++)
			{
				if (Cornerness[ii + jj*width] > maxRes*0.2)
				{
					potentialCorners[2 * npotentialCorners] = ii;
					potentialCorners[2 * npotentialCorners + 1] = jj;
					Response[npotentialCorners] = (float)Cornerness[ii + jj*width];
					npotentialCorners++;
				}
				if (npotentialCorners + 1 > nMaxCorners)
					break;
			}
			if (npotentialCorners + 1 > nMaxCorners)
				break;
		}


		// The orginal paper recommends a threshold of 0.9 but that results in some very close points, which have similar responses, to pass the test.
		float *Mdist = new float[npotentialCorners*(npotentialCorners + 1) / 2]; // Special handling needed for this square symetric matrix due to large nkpts
		float *pt_uv = new float[2 * npotentialCorners];
		int *ind = new int[npotentialCorners];
		float *radius2 = new float[npotentialCorners];

		int pt1u, pt1v, pt2u, pt2v;
		int index = 0;
		for (ii = 0; ii < npotentialCorners; ii++)
		{
			pt1u = potentialCorners[2 * ii];
			pt1v = potentialCorners[2 * ii + 1];
			index++;

			for (jj = ii + 1; jj < npotentialCorners; jj++)
			{
				pt2u = potentialCorners[2 * jj];
				pt2v = potentialCorners[2 * jj + 1];
				Mdist[index] = 1.0f*(pt1u - pt2u)*(pt1u - pt2u) + (pt1v - pt2v)*(pt1v - pt2v);
				index++;
			}
		}

		float response1, response2, r2;
		for (i = 0; i < npotentialCorners; i++)
		{
			r2 = 1.0e9;
			response1 = Response[i];

			for (j = 0; j < npotentialCorners; j++)
			{
				if (i == j)
					continue;

				response2 = Response[j];
				if (response1 < AMN_thresh * response2)
				{
					if (i > j)
					{
						ii = j;
						jj = i;
					}
					else
					{
						ii = i;
						jj = j;
					}
					index = (2 * npotentialCorners - ii + 1)*ii / 2 + jj - ii;
					r2 = min(r2, Mdist[index]);
				}
			}
			radius2[i] = -r2; //descending order
			ind[i] = i;
		}

		int ANM_pts = 1000;
		Quick_Sort_Float(radius2, ind, 0, npotentialCorners - 1);
		for (i = 0; i < ANM_pts; i++)
		{
			HarrisC[i].x = potentialCorners[2 * ind[i]];
			HarrisC[i].y = potentialCorners[2 * ind[i] + 1];
		}
		npts = ANM_pts;

		delete[]Mdist;
		delete[]pt_uv;
		delete[]ind;
		delete[]radius2;
	}
	else if (SuppressType == 1)
	{
		for (jj = kk + 10; jj < height - kk - 10; jj++)
		{
			for (ii = kk + 10; ii < width - kk - 10; ii++)
			{
				if (Cornerness[ii + jj*width] > maxRes*0.2)
				{
					potentialCorners[2 * npotentialCorners] = ii;
					potentialCorners[2 * npotentialCorners + 1] = jj;
					Response[npotentialCorners] = (float)Cornerness[ii + jj*width];
					npotentialCorners++;
				}
				if (npotentialCorners + 1 > nMaxCorners)
					break;
			}
			if (npotentialCorners + 1 > nMaxCorners)
				break;
		}

		bool breakflag;
		int x, y;
		float *Response2 = new float[npotentialCorners];
		for (kk = 0; kk < npotentialCorners; kk++)
		{
			x = potentialCorners[2 * kk];
			y = potentialCorners[2 * kk + 1];
			Response2[kk] = Response[kk];
			breakflag = false;
			for (jj = -1; jj < 2; jj++)
			{
				for (ii = -1; ii < 2; ii++)
				{
					if (Response[kk] < Cornerness[x + ii + (y + jj)*width] - 1.0)
					{
						Response2[kk] = 0.0;
						breakflag = true;
						break;
					}
				}
				if (breakflag == true)
					break;
			}
		}

		npts = 0;
		for (kk = 0; kk<npotentialCorners; kk++)
		{
			if (Response2[kk] > maxRes*0.2)
			{
				HarrisC[npts].x = potentialCorners[2 * kk];
				HarrisC[npts].y = potentialCorners[2 * kk + 1];
				npts++;
			}
			if (npts > nMaxCorners)
				break;
		}
		delete[]Response2;
	}
	else
	{
		npts = 0;
		for (jj = kk + 10; jj < height - kk - 10; jj++)
		{
			for (ii = kk + 10; ii < width - kk - 10; ii++)
			{
				if (Cornerness[ii + jj*width] > maxRes*0.2)
				{
					HarrisC[npts].x = ii;
					HarrisC[npts].y = jj;
					npts++;
				}
				if (npts > nMaxCorners)
					break;
			}
			if (npts > nMaxCorners)
				break;
		}
	}

	delete[]GKernel;
	delete[]GDKernel;
	delete[]GKernel2;
	delete[]temp;
	delete[]Ix;
	delete[]Iy;
	delete[]Ix2;
	delete[]Iy2;
	delete[]Ixy;
	delete[]tr;
	delete[]Det;
	delete[]Cornerness;
	delete[]potentialCorners;
	delete[]Response;

	return;
}
void DetectCornersCorrelation(double *img, int width, int height, CPoint2 *Checker, int &npts, vector<double> PatternAngles, int hsubset, int search_area, double thresh)
{
	int i, j, ii, jj, kk, jump = 2, nMaxCorners = npts, numPatterns = PatternAngles.size();

	int bi_graylevel[2] = { 0, 255 }, PatternSize = 48, PatternLength = PatternSize*PatternSize; //Note that the pattern size is deliberately make bigger than the subset because small size give very blurry checkercorner
	double *maskSmooth = new double[PatternLength*numPatterns];

	synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);
	double trans[9], temp[9], iH1[9], H1[9] = { 1, 0, -PatternSize / 2, 0, 1, -PatternSize / 2, 0, 0, 1 };
	for (ii = 1; ii < PatternAngles.size(); ii++)
	{
		double c = cos(PatternAngles.at(ii)*3.14159265359 / 180), s = sin(PatternAngles.at(ii)*3.14159265359 / 180);
		double H2[9] = { c, -s, 0, s, c, 0, 0, 0, 1 };
		mat_invert(H1, iH1, 3);
		mat_mul(H2, H1, temp, 3, 3, 3);
		mat_mul(iH1, temp, trans, 3, 3, 3);
		TransformImage(maskSmooth + ii*PatternLength, maskSmooth, trans, PatternSize, PatternSize, 1, 1, NULL);
		//SaveDataToImage("C:/temp/rS.png", maskSmooth+ii*PatternLength, PatternSize, PatternSize, 1);
	}

	double *Cornerness = new double[width*height];
	for (ii = 0; ii < width*height; ii++)
		Cornerness[ii] = 0.0;

	double zncc;
	CPoint POI;
	for (kk = 0; kk < numPatterns; kk++)
	{
		for (jj = 3 * hsubset; jj < height - 3 * hsubset; jj += jump)
		{
			for (ii = 3 * hsubset; ii < width - 3 * hsubset; ii += jump)
			{
				POI.x = ii, POI.y = jj;
				TMatchingSuperCoarse(maskSmooth + kk*PatternLength, PatternSize, hsubset, img, width, height, POI, search_area, thresh, zncc);
				Cornerness[ii + jj*width] = max(zncc, Cornerness[ii + jj*width]);
			}
		}
	}

	double *Cornerness2 = new double[width*height];
	for (ii = 0; ii < width*height; ii++)
		Cornerness2[ii] = Cornerness[ii];
	//WriteGridBinary("C:/temp/cornerness.dat", Cornerness, width, height);

	//Non-max suppression
	bool breakflag;
	for (jj = 3 * hsubset; jj < height - 3 * hsubset; jj += jump)
	{
		for (ii = 3 * hsubset; ii < width - 3 * hsubset; ii += jump)
		{
			breakflag = false;
			if (Cornerness[ii + jj*width] < thresh)
			{
				Cornerness[ii + jj*width] = 0.0;
				Cornerness2[ii + jj*width] = 0.0;
			}
			else
			{
				for (j = -jump; j <= jump; j += jump)
				{
					for (i = -jump; i <= jump; i += jump)
					{
						if (Cornerness[ii + jj*width] < Cornerness[ii + i + (jj + j)*width] - 0.001) //avoid comparing with itself
						{
							Cornerness2[ii + jj*width] = 0.0;
							breakflag = true;
							break;
						}
					}
				}
			}
			if (breakflag == true)
				break;
		}
	}

	npts = 0;
	for (jj = 3 * hsubset; jj < height - 3 * hsubset; jj += jump)
	{
		for (ii = 3 * hsubset; ii < width - 3 * hsubset; ii += jump)
		{
			if (Cornerness2[ii + jj*width] > thresh)
			{
				Checker[npts].x = ii;
				Checker[npts].y = jj;
				npts++;
			}
			if (npts > nMaxCorners)
				break;
		}
	}

	delete[]maskSmooth;
	delete[]Cornerness;
	delete[]Cornerness2;

	return;
}
void RefineCorners(double *Para, int width, int height, CPoint2 *Checker, CPoint2 *Fcorners, int *FStype, int &npts, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCthresh, int InterpAlgo)
{
	int ii, jj, kk, boundary = 50;
	int numPatterns = PatternAngles.size();
	int bi_graylevel[2] = { 0, 255 }, PatternSize = 48, PatternLength = PatternSize*PatternSize; //Note that the pattern size is deliberately make bigger than the hsubset because small size give very blurry checkercorner
	double *maskSmooth = new double[PatternLength*numPatterns * 2];

	synthesize_square_mask(maskSmooth, bi_graylevel, PatternSize, 1.0, 0, false);
	synthesize_square_mask(maskSmooth + PatternLength, bi_graylevel, PatternSize, 1.0, 1, false);

	double trans[9], temp[9], iH1[9], H1[9] = { 1, 0, -PatternSize / 2, 0, 1, -PatternSize / 2, 0, 0, 1 };
	for (ii = 1; ii < PatternAngles.size(); ii++)
	{
		double c = cos(PatternAngles.at(ii)*3.14159265359 / 180), s = sin(PatternAngles.at(ii)*3.14159265359 / 180);
		double H2[9] = { c, -s, 0, s, c, 0, 0, 0, 1 };
		mat_invert(H1, iH1, 3), mat_mul(H2, H1, temp, 3, 3, 3), mat_mul(iH1, temp, trans, 3, 3, 3);
		TransformImage(maskSmooth + 2 * ii*PatternLength, maskSmooth, trans, PatternSize, PatternSize, 1, 1, NULL);
		TransformImage(maskSmooth + (2 * ii + 1)*PatternLength, maskSmooth + PatternLength, trans, PatternSize, PatternSize, 1, 1, NULL);
	}
	/*	FILE *fp = fopen("C:/temp/coarse.txt", "w+");
	for(ii=0; ii<npts; ii++)
	fprintf(fp, "%.2f %.2f \n", Checker[ii].x, Checker[ii].y);
	fclose(fp);*/

	//Detect coarse corners:
	int *goodCandiates = new int[npts];
	CPoint2 *goodCorners = new CPoint2[npts];
	int count = 0, ngoodCandiates = 0, squaretype;

	int percent = 10, increP = 10;
	double start = omp_get_wtime(), elapsed;
	cout << "Coarse refinement ..." << endl;
	double zncc, bestzncc;
	for (ii = 0; ii < npts; ii++)
	{
		if ((ii * 100 / npts - percent) >= 0)
		{
			elapsed = omp_get_wtime() - start;
			printf("\r %.2f%% TE: %f TR: %f", 100.0*ii / npts, elapsed, elapsed / percent*(100.0 - percent));
			percent += increP;
		}

		if ((Checker[ii].x < boundary) || (Checker[ii].y < boundary) || (Checker[ii].x > 1.0*width - boundary) || (Checker[ii].y > 1.0*height - boundary))
			continue;

		zncc = 0.0, bestzncc = 0.0;
		for (jj = 0; jj<numPatterns; jj++)
		{
			squaretype = TMatchingCoarse(maskSmooth + 2 * jj*PatternLength, PatternSize, hsubset1, Para, width, height, Checker[ii], searchArea, ZNCCCoarseThresh, zncc, InterpAlgo);
			if (squaretype >-1 && zncc > bestzncc)
			{
				goodCorners[count].x = Checker[ii].x;
				goodCorners[count].y = Checker[ii].y;
				goodCandiates[count] = squaretype + 2 * jj;
				bestzncc = zncc;
			}
		}
		if (bestzncc > ZNCCCoarseThresh)
			count++;
	}
	ngoodCandiates = count;
	elapsed = omp_get_wtime() - start;
	printf("\r %.2f%% TE: %f TR: %f\n", 100.0, elapsed, elapsed / percent*(100.0 - percent));
	cout << "finished with " << ngoodCandiates << " points" << endl;

	/*FILE *fp = fopen("C:/temp/coarseR.txt", "w+");
	for(ii=0; ii<ngoodCandiates; ii++)
	fprintf(fp, "%.2f %.2f %d\n", goodCorners[ii].x, goodCorners[ii].y, goodCandiates[ii]);
	fclose(fp);*/

	//Merege coarsely detected candidates:
	npts = ngoodCandiates;
	int STACK[30]; //Maximum KNN
	int *squareType = new int[npts];
	CPoint2 *mergeCorners = new CPoint2[npts];
	int *marker = new int[2 * npts];
	for (jj = 0; jj < 2 * npts; jj++)
		marker[jj] = -1;

	int flag, KNN;
	double t1, t2, megre_thresh = 5.0;
	count = 0, ngoodCandiates = 0;
	for (jj = 0; jj < npts; jj++)
	{
		KNN = 0;
		flag = 0;
		for (ii = 0; ii < count; ii++)
		{
			if (marker[ii] == jj)
			{
				flag = 1;
				break;
			}
		}
		if (flag == 1)
			continue;

		for (ii = jj + 1; ii < npts; ii++)
		{
			t1 = goodCorners[ii].x - goodCorners[jj].x;
			t2 = goodCorners[ii].y - goodCorners[jj].y;

			if (t1*t1 + t2*t2 < megre_thresh*megre_thresh &&goodCandiates[ii] == goodCandiates[jj])
			{
				STACK[KNN] = ii;
				KNN++;
			}
		}
		STACK[KNN] = jj;// include itself

		for (kk = 0; kk < KNN + 1; kk++)
		{
			marker[count] = STACK[kk];
			count++;
		}

		mergeCorners[ngoodCandiates].x = 0.0, mergeCorners[ngoodCandiates].y = 0.0;
		for (kk = 0; kk <= KNN; kk++)
		{
			mergeCorners[ngoodCandiates].x += goodCorners[STACK[kk]].x;
			mergeCorners[ngoodCandiates].y += goodCorners[STACK[kk]].y;
		}
		mergeCorners[ngoodCandiates].x /= (KNN + 1);
		mergeCorners[ngoodCandiates].y /= (KNN + 1);
		squareType[ngoodCandiates] = goodCandiates[jj];
		ngoodCandiates++;
	}

	/*FILE *fp = fopen("c:/temp/coarseRM.txt", "w+");
	for(ii=0; ii<ngoodCandiates; ii++)
	fprintf(fp, "%lf %lf %d\n", mergeCorners[ii].x, mergeCorners[ii].y, squareType[ii]);
	fclose(fp);*/

	//Refine corners:
	int advanced_tech = 1; // affine only
	count = 0;
	double *Znssd_reqd = new double[6 * PatternLength];

	percent = 10;
	start = omp_get_wtime();
	cout << "Fine refinement ..." << endl;
	for (ii = 0; ii < ngoodCandiates; ii++)
	{
		if ((ii * 100 / ngoodCandiates - percent) >= 0)
		{
			elapsed = omp_get_wtime() - start;
			printf("\r %.2f%% TE: %f TR: %f", 100.0*ii / ngoodCandiates, elapsed, elapsed / percent*(100.0 - percent));
			percent += increP;
		}

		if ((mergeCorners[ii].x < boundary) || (mergeCorners[ii].y < boundary) || (mergeCorners[ii].x > 1.0*width - boundary) || (mergeCorners[ii].y > 1.0*height - boundary))
			continue;

		zncc = TMatchingFine_ZNCC(maskSmooth + squareType[ii] * PatternLength, PatternSize, hsubset2, Para, width, height, mergeCorners[ii], 0, 1, ZNCCthresh, InterpAlgo, Znssd_reqd);
		if (zncc > ZNCCthresh)
		{
			squareType[ii] = squareType[ii];
			count++;
		}
		else
			squareType[ii] = -1;
	}
	delete[]Znssd_reqd;
	elapsed = omp_get_wtime() - start;
	printf("\r %.2f%% TE: %f TR: %f\n", 100.0, elapsed, elapsed / percent*(100.0 - percent));
	cout << "finished with " << count << " points" << endl;

	//Final merging:
	count = 0;
	for (ii = 0; ii < ngoodCandiates; ii++)
	{
		if (squareType[ii] != -1)
		{
			goodCorners[count].x = mergeCorners[ii].x;
			goodCorners[count].y = mergeCorners[ii].y;
			goodCandiates[count] = squareType[ii];
			count++;
		}
	}

	npts = count;
	for (jj = 0; jj < npts; jj++)
		marker[jj] = -1;

	megre_thresh = 4.0, count = 0, ngoodCandiates = 0;
	for (jj = 0; jj < npts; jj++)
	{
		KNN = 0, flag = 0;
		for (ii = 0; ii < count; ii++)
		{
			if (marker[ii] == jj)
			{
				flag = 1;
				break;
			}
		}
		if (flag == 1)
			continue;

		for (ii = jj + 1; ii < npts; ii++)
		{
			t1 = goodCorners[ii].x - goodCorners[jj].x;
			t2 = goodCorners[ii].y - goodCorners[jj].y;
			if (t1*t1 + t2*t2 < megre_thresh*megre_thresh)
			{
				STACK[KNN] = ii;
				KNN++;
			}
		}
		STACK[KNN] = jj;// include itself

		for (kk = 0; kk < KNN + 1; kk++)
		{
			marker[count] = STACK[kk];
			count++;
		}

		Fcorners[ngoodCandiates].x = goodCorners[jj].x, Fcorners[ngoodCandiates].y = goodCorners[jj].y;
		for (kk = 0; kk < KNN; kk++)
		{
			Fcorners[ngoodCandiates].x += goodCorners[STACK[kk]].x;
			Fcorners[ngoodCandiates].y += goodCorners[STACK[kk]].y;
		}
		Fcorners[ngoodCandiates].x /= (KNN + 1);
		Fcorners[ngoodCandiates].y /= (KNN + 1);
		FStype[ngoodCandiates] = goodCandiates[jj];
		ngoodCandiates++;
	}

	npts = ngoodCandiates;
	cout << "After merging points: " << npts << endl;

	delete[]maskSmooth;
	delete[]goodCorners;
	delete[]goodCandiates;
	delete[]marker;
	delete[]squareType;
	delete[]mergeCorners;

	return;
}
void EpipolarCorrespondences(CPoint *CorresID, int &nCorres, double *IPara, int CamID, int ProID, int width, int height, CPoint2 *Icorners, int *Itype, int Inpts, double *PPara, CPoint2 *Pcorners, int *Ptype, int Pnpts, int pwidth, int pheight, DevicesInfo &DInfo, int PSncols, CPoint minP, CPoint maxP, int checkerSize, double Scale, double ZNCCThreshold, int LK_IterMax, double EpipThresh, int InterpAlgo)
{
	//Note: you must ensure that the coor. in in lower left orig for all the computation to be valid.
	// You need to roughly determine the scale to do matching between camera and projector
	//This function supports only grayscale image
	int ii, jj, kk, ll;

	const int nCams = DInfo.nCams, nPros = DInfo.nPros, nchannels = 1, maxcandidates = 100;
	double FmatCP[9], pLine[3], pt[3], dist;
	int *Pcandidates = new int[Inpts*maxcandidates];
	int *PCcount = new int[Inpts];

	//Step 1: search for potential correspondences throught epipolar constraint
	mat_transpose(DInfo.FmatPC + 9 * (CamID + ProID*nCams), FmatCP, 3, 3);

	//Operate on corrected points only:
	CPoint2 *_Icorners = new CPoint2[Inpts];
	for (ii = 0; ii < Inpts; ii++)
	{
		_Icorners[ii].x = Icorners[ii].x;
		_Icorners[ii].y = Icorners[ii].y;
		Undo_distortion(_Icorners[ii], DInfo.K + 9 * (CamID + nPros), DInfo.distortion + 13 * (CamID + nPros));
	}

	CPoint2 *_Pcorners = new CPoint2[Pnpts];
	for (ii = 0; ii < Pnpts; ii++)
	{
		_Pcorners[ii].x = Pcorners[ii].x;
		_Pcorners[ii].y = Pcorners[ii].y;
		Undo_distortion(_Pcorners[ii], DInfo.K + 9 * ProID, DInfo.distortion + 13 * ProID);
	}

	for (ii = 0; ii < Inpts; ii++)
	{
		pt[0] = _Icorners[ii].x, pt[1] = _Icorners[ii].y, pt[2] = 1.0;
		mat_mul(FmatCP, pt, pLine, 3, 3, 1);

		PCcount[ii] = 0;
		for (jj = 0; jj < Pnpts; jj++)
		{
			if (_Pcorners[jj].x <1.0*minP.x || _Pcorners[jj].x >1.0*maxP.x || _Pcorners[jj].y <1.0*minP.y || _Pcorners[jj].y >1.0*maxP.y)
				continue;

			if (Itype[ii] != Ptype[jj])
				continue;

			dist = (_Pcorners[jj].x*pLine[0] + _Pcorners[jj].y*pLine[1] + pLine[2]) / sqrt(pLine[0] * pLine[0] + pLine[1] * pLine[1]);
			if (abs(dist) < EpipThresh)
			{
				Pcandidates[PCcount[ii] + ii*maxcandidates] = jj;
				PCcount[ii]++;
			}
		}
	}

	/*FILE *fp = fopen("C:/temp/Pcan.txt", "w+");
	for (ii = 0; ii < Inpts; ii++)
	{
	fprintf(fp, "%d ", PCcount[ii]);
	for (jj = 0; jj < PCcount[ii]; jj++)
	fprintf(fp, "%d ", Pcandidates[jj + ii*maxcandidates]);
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	//Step 2: Do a correlation on the candiadates
	int clength = width*height;
	int pattern_size = (int)(1.25*checkerSize) / 2 * 2 + 1, pattern_cen = pattern_size / 2;  //test over an area 1.4^2 times bigger than that of corner itseft

	double fufv[2], ShapePara[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
	double score[maxcandidates];
	int index[maxcandidates];
	//Tar: camera image - Ref: projector image

	ZNCCThreshold -= 0.08;
	nCorres = 0;

	double start = omp_get_wtime();
	int percent = 50, increP = 50;
	cout << "Exploit epipolar constrant..." << endl;
	for (kk = 0; kk < Inpts; kk++)
	{
#pragma omp critical
		if ((kk * 100 / Inpts - percent) > 0)
		{
			double elapsed = omp_get_wtime() - start;
			cout << "%" << kk * 100 / Inpts << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / percent*(100.0 - percent) << endl;
			percent += increP;
		}

		for (ll = 0; ll < PCcount[kk]; ll++)
		{
			ShapePara[0] = Scale, ShapePara[4] = Scale, ShapePara[2] = Icorners[kk].x, ShapePara[5] = Icorners[kk].y;
			if ((Icorners[kk].x < 20.0) || (Icorners[kk].y < 20.0) || (Icorners[kk].x > 1.0*width - 20.0) || (Icorners[kk].y > 1.0*height - 20.0))
				continue;
			score[ll] = -TMatching(PPara, IPara, pattern_size / 2, pwidth, pheight, width, height, nchannels, Pcorners[Pcandidates[ll + kk*maxcandidates]], Icorners[kk], 1, 1, ZNCCThreshold, LK_IterMax, InterpAlgo, fufv, false, ShapePara); //negate this because the sorting function is an ascending one
			index[ll] = Pcandidates[ll + kk*maxcandidates];
		}
		if (ll > 0)
		{
			Quick_Sort_Double(score, index, 0, ll - 1);
			if (abs(score[0]) > ZNCCThreshold)
			{
				CorresID[nCorres].x = kk;
				CorresID[nCorres].y = index[0];
				nCorres++;
			}
		}
	}
	cout << "Time elapsed: " << setw(2) << omp_get_wtime() - start << endl;

	/*fp = fopen( "C:/temp/PtoC.txt", "w+");
	for(ii=0; ii<nCorres; ii++)
	fprintf(fp, "%.2f %.2f %.2f %.2f \n", Icorners[CorresID[ii].x].x, Icorners[CorresID[ii].x].y, _Pcorners[CorresID[ii].y].x, _Pcorners[CorresID[ii].y].y);
	fclose(fp);*/

	delete[]PCcount;
	delete[]Pcandidates;
	delete[]_Icorners;
	delete[]_Pcorners;

	return;
}

void RunCornersDetector(CPoint2 *CornerPts, int *CornersType, int &nCpts, double *Img, double *IPara, int width, int height, vector<double>PatternAngles, int hsubset1, int hsubset2, int searchArea, double ZNCCCoarseThresh, double ZNCCThresh, int InterpAlgo)
{
	int npts = 500000;
	CPoint2 *Checker = new CPoint2[npts];

	//int SuppressType = 1;
	//double Gsigma = 2.5, GDsigma = 3.0, ResponseThresh = 0.9;
	//DetectCorners(img, width, height, HarrisC, npts, Gsigma, GDsigma, ResponseThresh, 0.04, SuppressType, 0.9);
#pragma omp critical
	cout << "Sliding window for detection..." << endl;

	DetectCornersCorrelation(Img, width, height, Checker, npts, PatternAngles, hsubset1, searchArea, ZNCCCoarseThresh);
	/*FILE *fp = fopen("C:/temp/cornerCorr.txt", "w+");
	for (int ii = 0; ii < npts; ii++)
	fprintf(fp, "%.1f %1f\n", Checker[ii].x, Checker[ii].y);
	fclose(fp);
	FILE *fp = fopen("C:/temp/cornerCorr.txt", "r");
	npts = 0;
	while (fscanf(fp, "%lf %lf ", &Checker[npts].x, &Checker[npts].y) != EOF)
	npts++;
	fclose(fp);*/

#pragma omp critical
	cout << "finished width " << npts << " points. Refine detected corners..." << endl;

	RefineCorners(IPara, width, height, Checker, CornerPts, CornersType, npts, PatternAngles, hsubset1, hsubset2, searchArea, ZNCCCoarseThresh, ZNCCThresh, InterpAlgo);
	nCpts = npts;

	delete[]Checker;
	return;
}
int DetectMarkersandCorrespondence(char *PATH, CPoint2 *Pcorners, int *PcorType, double *PPara, DevicesInfo &DInfo, LKParameters LKArg, double *CamProScale, vector<double>PatternAngles, int checkerSize, int nPpts, int PSncols, CPoint *ProjectorCorressBoundary, double EpipThresh, int frameID, int CamID, int nPros, int width, int height, int pwidth, int pheight)
{
	cout << "Working on frame " << frameID << " of camera " << CamID + 1 << endl;
	int length = width*height;
	const int maxPts = 5000, SearchArea = 1;
	int CheckerhSubset1 = (int)(CamProScale[CamID] * 0.2*checkerSize + 0.5), CheckerhSubset2 = (int)(CamProScale[CamID] * 0.3*checkerSize + 0.5); //A reasonable checkerhsubset gives better result than using the huge subset

	int  ii, jj, nCPts, nCCorres = maxPts, CType1[maxPts];
	CPoint2 CPts1[maxPts], Ppts[maxPts], CPts[maxPts * 2];
	CPoint CCorresID[maxPts];
	int T[maxPts], TT[maxPts];

	IplImage *view = 0;
	char *Img = new char[2 * width*height];
	double *SImg = new double[width*height];
	double *IPara = new double[2 * width*height];

	char Fname[100];
	sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, CamID + 1, frameID);
	view = cvLoadImage(Fname, 0);
	if (view == NULL)
	{
		cout << "cannot load " << Fname << endl;
		delete[]Img;
		delete[]SImg;
		delete[]IPara;
		return 1;
	}
	cout << "Loaded " << Fname << endl;
	for (jj = 0; jj < height; jj++)
		for (ii = 0; ii < width; ii++)
			Img[ii + (height - 1 - jj)*width] = view->imageData[ii + jj*width];
	cvReleaseImage(&view);

	Gaussian_smooth(Img, SImg, height, width, 255.0, LKArg.Gsigma);
	Generate_Para_Spline(SImg, IPara, width, height, LKArg.InterpAlgo);

	RunCornersDetector(CPts1, CType1, nCPts, SImg, IPara, width, height, PatternAngles, CheckerhSubset1, CheckerhSubset2, SearchArea, LKArg.ZNCCThreshold - 0.35, LKArg.ZNCCThreshold, LKArg.InterpAlgo);
	/*FILE *fp = fopen("C:/temp/pts.txt", "w+");
	for (int ii = 0; ii < nCPts; ii++)
	fprintf(fp, "%f %f %d \n", CPts1[ii].x, CPts1[ii].y, CType1[ii]);
	fclose(fp);

	{
	nCPts = 0;
	FILE *fp = fopen("C:/temp/pts.txt", "r");
	while (fscanf(fp, "lf %lf %d %", &CPts1[nCPts].x, &CPts1[nCPts].y, &CType1[nCPts]) != EOF)
	nCPts++;
	fclose(fp);
	}*/
	for (int ProID = 0; ProID < nPros; ProID++)
	{
		EpipolarCorrespondences(CCorresID, nCCorres, IPara, CamID, ProID, width, height, CPts1, CType1, nCPts, PPara + ProID*pwidth*pheight, Pcorners, PcorType, nPpts, pwidth, pheight, DInfo, PSncols, ProjectorCorressBoundary[0], ProjectorCorressBoundary[1], checkerSize, CamProScale[CamID], LKArg.ZNCCThreshold, LKArg.IterMax, EpipThresh, LKArg.InterpAlgo);
		for (ii = 0; ii < nCCorres; ii++)
		{
			T[ii] = CCorresID[ii].y;
			TT[ii] = ii;
		}
		Quick_Sort_Int(T, TT, 0, nCCorres - 1);

		for (ii = 0; ii < nCCorres; ii++)
		{
			Ppts[ii].x = Pcorners[CCorresID[TT[ii]].y].x;
			Ppts[ii].y = Pcorners[CCorresID[TT[ii]].y].y;
			CPts[ii].x = CPts1[CCorresID[TT[ii]].x].x;
			CPts[ii].y = CPts1[CCorresID[TT[ii]].x].y;
		}

		sprintf(Fname, "%s/Sparse/P%dC%d_%05d.txt", PATH, ProID + 1, CamID + 1, frameID);
		FILE *fp = fopen(Fname, "w+");
		for (ii = 0; ii < nCCorres; ii++)
			fprintf(fp, "%d %.8f %.8f \n", CCorresID[TT[ii]].y, Ppts[ii].x, Ppts[ii].y);
		fclose(fp);

		sprintf(Fname, "%s/Sparse/C%dP%d_%05d.txt", PATH, CamID + 1, ProID + 1, frameID);
		fp = fopen(Fname, "w+");
		for (ii = 0; ii < nCCorres; ii++)
			fprintf(fp, "%.8f %.8f \n", CPts[ii].x, CPts[ii].y);
		fclose(fp);

		if (nCCorres < 10)
			cout << "Something wrong with #sparse points." << endl;
		cout << nCCorres << " pts dected for frame " << frameID << endl;
	}

	delete[]Img;
	delete[]SImg;
	delete[]IPara;

	return 0;
}
int  CheckerDetectionCorrespondenceDriver(int CamID, int nCams, int nPros, int frameID, int pwidth, int pheight, int width, int height, char *PATH)
{
	//1: Set up input arguments
	char Fname[100];

	int ii, jj;
	int PSncols, checkerSize = 10, nPpts = 938, InterpAlgo = 1;
	double GaussianBlur = 0.707, EpipThresh = 5.0;
	vector<double> PatternAngles; PatternAngles.push_back(0.0);// , PatternAngles.push_back(30.0);

	LKParameters LKArg;
	LKArg.Convergence_Criteria = 1, LKArg.DIC_Algo = 1, LKArg.InterpAlgo = InterpAlgo, LKArg.IterMax = 20;
	LKArg.Gsigma = 0.707, LKArg.ZNCCThreshold = 0.965;

	double *CamProScale = new double[nCams];
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

	CPoint ProjectorCorressBoundary[2];
	ProjectorCorressBoundary[0].x = 10, ProjectorCorressBoundary[0].y = 10;
	ProjectorCorressBoundary[1].x = pwidth - 10, ProjectorCorressBoundary[1].y = pheight - 10;

	//2. Set up data input
	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot load Camera Projector info" << endl;
		return 1;
	}

	char *PImg = new char[pwidth*pheight*nPros];
	double *PImgD = new double[pwidth*pheight*nPros];
	double *PPara = new double[pwidth*pheight*nPros];

	for (int ProID = 0; ProID < nPros; ProID++)
	{
		int plength = pheight*pwidth;
		sprintf(Fname, "%s/ProjectorPattern%d.png", PATH, ProID + 1);
		IplImage *view = cvLoadImage(Fname, 0);
		if (view == NULL)
		{
			cout << "Cannot load ProjectorPattern" << endl;
			return 2;
		}
		else
		{
			for (jj = 0; jj < pheight; jj++)
				for (ii = 0; ii < pwidth; ii++)
					PImg[ii + (pheight - 1 - jj)*pwidth + ProID*plength] = view->imageData[ii + jj*pwidth];
			cvReleaseImage(&view);
		}

		Gaussian_smooth(PImg + ProID*plength, PImgD + ProID*plength, pheight, pwidth, 255.0, GaussianBlur);
		Generate_Para_Spline(PImgD + ProID*plength, PPara + ProID*plength, pwidth, pheight, InterpAlgo);
	}

	int *PcorType = new int[nPpts];
	CPoint2 *Pcorners = new CPoint2[nPpts];

	sprintf(Fname, "%s/ProCorners.txt", PATH);
	fp = fopen(Fname, "r");
	if (fp == NULL)
		cout << "Cannot load ProCorners" << endl;

	fscanf(fp, "%d %d %d", &PSncols, &ii, &jj);
	for (ii = 0; ii < nPpts; ii++)
		fscanf(fp, "%d %lf %lf ", &PcorType[ii], &Pcorners[ii].x, &Pcorners[ii].y);
	fclose(fp);

	//3. Run detection tracking function
	DetectMarkersandCorrespondence(PATH, Pcorners, PcorType, PPara, DInfo, LKArg, CamProScale, PatternAngles, checkerSize, nPpts, PSncols, ProjectorCorressBoundary, EpipThresh, frameID, CamID, nPros, width, height, pwidth, pheight);

	delete[]CamProScale;
	delete[]PImg;
	delete[]PImgD;
	delete[]PPara;
	delete[]PcorType;
	delete[]Pcorners;

	return 0;
}

void TrackMarkersFlow(char *Img1, char *Img2, double *IPara1, double *IPara2, int width, int height, CPoint2 *Ppts, CPoint2 *Cpts1, CPoint2 *Cpts2, int nCpts, double *Fmat1x, float *flowx, float *flowy, int hsubset, int advancedTech, int ConvCriteria, double ZNCCThresh, int InterpAlgo)
{
	int ii, jj, kk;
	int  CheckerhSubset = 6, hTsize = hsubset > CheckerhSubset ? hsubset : CheckerhSubset, TSize = 2 * hTsize + 1, searchArea = 5;

	CPoint2 POI;
	double S[3], zncc;
	double *Template = new double[TSize*TSize];

	for (kk = 0; kk < nCpts; kk++)
	{
		if (Cpts1[kk].x < INACTIVE_IMPTS || Cpts1[kk].y < INACTIVE_IMPTS)
		{
			Cpts2[kk].x = INACTIVE_IMPTS - 1.0;
			Cpts2[kk].y = INACTIVE_IMPTS - 1.0;
		}
		else if (Cpts1[kk].x <3 * hsubset || Cpts1[kk].x > width - 3 * hsubset || Cpts1[kk].y <3 * hsubset || Cpts1[kk].y > height - 3 * hsubset)
		{
			Cpts2[kk].x = INACTIVE_IMPTS - 1.0;
			Cpts2[kk].y = INACTIVE_IMPTS - 1.0;
		}
		else
		{
			Cpts2[kk].x = Cpts1[kk].x + flowx[(int)(Cpts1[kk].x + 0.5) + (int)(Cpts1[kk].y + 0.5) * width];
			Cpts2[kk].y = Cpts1[kk].y + flowy[(int)(Cpts1[kk].x + 0.5) + (int)(Cpts1[kk].y + 0.5) * width];
		}

		//FILE *fp = fopen("C:/temp/ref.txt", "w+");
		for (jj = -hTsize; jj <= hTsize; jj++)
		{
			for (ii = -hTsize; ii <= hTsize; ii++)
			{
				Get_Value_Spline(IPara1, width, height, Cpts1[kk].x + ii, Cpts1[kk].y + jj, S, -1, InterpAlgo);
				Template[(ii + hTsize) + (jj + hTsize)*TSize] = S[0];
				//fprintf(fp, "%.2f ", S[0]);
			}
			//fprintf(fp, "\n");
		}
		//fclose(fp);

		if (TMatchingCoarse(Template, TSize, CheckerhSubset, IPara2, width, height, Cpts2[kk], searchArea, ZNCCThresh, zncc, InterpAlgo) == 0)
		{
			if (!TMatchingFine(Template, TSize, hsubset, IPara2, width, height, Cpts2[kk], advancedTech, ConvCriteria, ZNCCThresh, InterpAlgo))
			{
				Cpts2[kk].x = INACTIVE_IMPTS - 1.0;
				Cpts2[kk].y = INACTIVE_IMPTS - 1.0;
			}
		}
		else
		{
			Cpts2[kk].x = INACTIVE_IMPTS - 1.0;
			Cpts2[kk].y = INACTIVE_IMPTS - 1.0;
		}
	}

	delete[]Template;
	return;
}
int DetectTrackMarkersandCorrespondence(char *PATH, int frameJump, CPoint2 *Pcorners, int *PcorType, double *PPara, DevicesInfo &DInfo, LKParameters LKArg, double *CamProScale, vector<double> PatternAngles, int checkerSize, int nPpts, int PSncols, CPoint *ProjectorCorressBoundary, double EpipThresh, int nframes, int startFrame, int CamID, int width, int height, int pwidth, int pheight)
{
	cout << "Working on frame (" << startFrame << "," << startFrame + frameJump << ") of camera " << CamID << endl;
	int length = width*height;
	const int maxPts = 5000, SearchArea = 1;
	int CheckerhSubset1 = (int)(CamProScale[CamID] * 0.3*checkerSize + 0.5), CheckerhSubset2 = (int)(CamProScale[CamID] * 0.45*checkerSize + 0.5); //A reasonable checkerhsubset gives better result than using the huge subset

	omp_set_num_threads(1);
#pragma omp parallel 
	{
		for (int kk = 0; kk < nframes / 2; kk++)
		{
#pragma omp for nowait
			for (int ll = 0; ll < 2; ll++)
			{
				int  ii, jj, nCPts, nCCorres = maxPts, CType1[maxPts];
				CPoint2 CPts1[maxPts], Ppts[maxPts], CPts[maxPts * 2];
				CPoint CCorresID[maxPts];
				int T[maxPts], TT[maxPts];

				IplImage *view = 0;
				char *Img = new char[2 * width*height];
				double *SImg = new double[width*height];
				double *IPara = new double[2 * width*height];

				char Fname[100];
				sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, CamID + 1, startFrame + 2 * kk + ll*frameJump);
				view = cvLoadImage(Fname, 0);
				if (view == NULL)
				{
					cout << "cannot load " << Fname << endl;
					delete[]Img;
					delete[]SImg;
					delete[]IPara;
					continue;
				}
				cout << "Loaded " << Fname << endl;
				for (jj = 0; jj < height; jj++)
					for (ii = 0; ii < width; ii++)
						Img[ii + (height - 1 - jj)*width] = view->imageData[ii + jj*width];
				cvReleaseImage(&view);

				Gaussian_smooth(Img, SImg, height, width, 255.0, LKArg.Gsigma);
				Generate_Para_Spline(SImg, IPara, width, height, LKArg.InterpAlgo);

				RunCornersDetector(CPts1, CType1, nCPts, SImg, IPara, width, height, PatternAngles, CheckerhSubset1, CheckerhSubset2, SearchArea, LKArg.ZNCCThreshold - 0.35, LKArg.ZNCCThreshold, LKArg.InterpAlgo);
				{
					FILE *fp = fopen("C:/temp/pts.txt", "w+");
					for (ii = 0; ii < nCPts; ii++)
						fprintf(fp, "%d %.8f %.8f \n", CType1[ii], CPts1[ii].x, CPts1[ii].y);
					fclose(fp);
				}
				{
					nCPts = 0;
					FILE *fp = fopen("C:/temp/pts.txt", "r");
					while (fscanf(fp, "%d %lf %lf \n", &CType1[nCPts], &CPts1[nCPts].x, &CPts1[nCPts].y) != EOF)
						nCPts++;
					fclose(fp);
				}


				for (int ProID = 0; ProID < DInfo.nPros; ProID++)
				{
					EpipolarCorrespondences(CCorresID, nCCorres, IPara, CamID, ProID, width, height, CPts1, CType1, nCPts, PPara, Pcorners, PcorType, nPpts, pwidth, pheight, DInfo, PSncols, ProjectorCorressBoundary[0], ProjectorCorressBoundary[1], checkerSize, CamProScale[CamID], LKArg.ZNCCThreshold, LKArg.IterMax, EpipThresh, LKArg.InterpAlgo);

					for (ii = 0; ii < nCCorres; ii++)
					{
						T[ii] = CCorresID[ii].y;
						TT[ii] = ii;
					}
					Quick_Sort_Int(T, TT, 0, nCCorres - 1);

					for (ii = 0; ii < nCCorres; ii++)
					{
						Ppts[ii].x = Pcorners[CCorresID[TT[ii]].y].x;
						Ppts[ii].y = Pcorners[CCorresID[TT[ii]].y].y;
						CPts[ii].x = CPts1[CCorresID[TT[ii]].x].x;
						CPts[ii].y = CPts1[CCorresID[TT[ii]].x].y;
					}

					sprintf(Fname, "%s/P%dC%d_%05d.txt", PATH, ProID + 1, CamID + 1, startFrame + 2 * kk + ll*frameJump);
					FILE *fp = fopen(Fname, "w+");
					for (ii = 0; ii < nCCorres; ii++)
						fprintf(fp, "%d %.8f %.8f \n", CCorresID[TT[ii]].y, Ppts[ii].x, Ppts[ii].y);
					fclose(fp);

					sprintf(Fname, "%s/C%dP%d_%05d.txt", PATH, CamID + 1, ProID + 1, startFrame + 2 * kk + ll*frameJump);
					fp = fopen(Fname, "w+");
					for (ii = 0; ii < nCCorres; ii++)
						fprintf(fp, "%.8f %.8f \n", CPts[ii].x, CPts[ii].y);
					fclose(fp);

					if (nCCorres < 10)
						cout << "Something wrong with #sparse points." << endl;
					cout << "Finish detecting sparse points for frame " << startFrame + 2 * kk + ll*frameJump << endl;
				}

				delete[]Img;
				delete[]SImg;
				delete[]IPara;
			}
		}
	}


	return 0;
}
void CorrespondenceByTrackingMarkers(char *PATH, CPoint2 *Pcorners, int *PcorType, double *PPara, DevicesInfo &DInfo, LKParameters LKArg, double *CamProScale, vector<double> PatternAngles, int checkerSize, int nPpts, int PSncols, CPoint *ProjectorCorressBoundary, double EpipThresh, int nframes, int startFrame, int CamID, int width, int height, int pwidth, int pheight)
{
	char Fname[100];
	int  ii, jj, kk, length = width*height, hsubset = 8;//(int)(CamProScale[CamID]*checkerSize+0.5);
	const int maxPts = 2000, CheckerhSubset = 6, SearchArea = 1;  //A reasonable checkerhsubset gives better result than using the huge subset

	//1: Detect corners and establish sparse correspondences
	int nCPts, nCCorres = maxPts, CType1[maxPts];
	CPoint2 CPts1[maxPts], Ppts[maxPts], CPts[maxPts * 2];
	CPoint CCorresID[maxPts];
	//int T[maxPts], TT[maxPts];

	IplImage *view = 0;
	char *Img = new char[2 * width*height];
	double *SImg = new double[width*height];
	double *IPara = new double[2 * width*height];
	float *flowx = new float[width*height];
	float *flowy = new float[width*height];


	for (kk = 0; kk < nframes / 2; kk++)
	{
		//1.1: Do detection corners and establish sparse correspondences for the 1st frame
		sprintf(Fname, "%s/Image/C%d_%05d.png", PATH, CamID + 1, startFrame + 2 * kk + 1);
		view = cvLoadImage(Fname, 0);
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				Img[ii + (height - 1 - jj)*width] = view->imageData[ii + jj*width];
		cvReleaseImage(&view);

		Gaussian_smooth(Img, SImg, height, width, 255.0, LKArg.Gsigma);
		Generate_Para_Spline(SImg, IPara, width, height, LKArg.InterpAlgo);

		RunCornersDetector(CPts1, CType1, nCPts, SImg, IPara, width, height, PatternAngles, CheckerhSubset, CheckerhSubset + 2,
			SearchArea, LKArg.ZNCCThreshold - 0.2, LKArg.ZNCCThreshold, LKArg.InterpAlgo);

		/*EpipolarCorrespondences(CCorresID, nCCorres, IPara, CamID, width, height, CPts1, CType1, nCPts, PPara, Pcorners, PcorType, nPpts, pwidth, pheight, DInfo, PSncols, ProjectorCorressBoundary[0], ProjectorCorressBoundary[1], checkerSize, CamProScale[CamID], LKArg.ZNCCThreshold, LKArg.IterMax, EpipThresh, LKArg.InterpAlgo);

		for(ii=0; ii<nCCorres; ii++)
		{
		T[ii] = CCorresID[ii].y;
		TT[ii] = ii;
		}
		Quick_Sort_Int(T, TT, 0, nCCorres-1);

		for(ii=0; ii<nCCorres; ii++)
		{
		Ppts[ii].x = Pcorners[CCorresID[TT[ii]].y].x;
		Ppts[ii].y = Pcorners[CCorresID[TT[ii]].y].y;
		CPts[ii].x = CPts1[CCorresID[TT[ii]].x].x;
		CPts[ii].y = CPts1[CCorresID[TT[ii]].x].y;
		}

		sprintf(Fname, "%s/C%d_%d_Ppts.txt",PATH, CamID+1, startFrame+2*kk);
		FILE *fp = fopen(Fname, "w+");
		for(ii=0; ii<nCCorres; ii++)
		fprintf(fp, "%d %.8f %.8f \n", CCorresID[TT[ii]].y, Ppts[ii].x, Ppts[ii].y);
		fclose(fp);

		sprintf(Fname, "%s/C%d_%d_Ppts.txt",PATH, CamID+1, startFrame+2*kk+1);
		fp = fopen(Fname, "w+");
		for(ii=0; ii<nCCorres; ii++)
		fprintf(fp, "%d %.8f %.8f \n", CCorresID[TT[ii]].y, Ppts[ii].x, Ppts[ii].y);
		fclose(fp);

		sprintf(Fname, "%s/Sparse/C%d_%05d.txt",PATH,  CamID+1, startFrame+2*kk+1);
		fp = fopen(Fname, "w+");
		for(ii=0; ii<nCCorres; ii++)
		fprintf(fp, "%.8f %.8f \n", CPts[ii].x, CPts[ii].y);
		fclose(fp);

		//1.2: Dectection by tracking
		sprintf(Fname, "%s/Image/C%d_%05d.png",PATH,  CamID+1, startFrame+2*kk+1);
		view= cvLoadImage(Fname, 0);
		for(jj=0; jj<height; jj++)
		for(ii=0; ii<width; ii++)
		Img[ii+(height-1-jj)*width] = view->imageData[ii+jj*width];
		Generate_Para_Spline(Img, IPara, width, height, LKArg.InterpAlgo);

		sprintf(Fname, "%s/Image/C%d_%05d.png",PATH,  CamID+1, startFrame+2*kk);
		view= cvLoadImage(Fname, 0);
		for(jj=0; jj<height; jj++)
		for(ii=0; ii<width; ii++)
		Img[ii+(height-1-jj)*width+length] = view->imageData[ii+jj*width];
		Generate_Para_Spline(Img+length, IPara+length, width, height, LKArg.InterpAlgo);

		sprintf(Fname, "%s/Flow/F%dreX_%05d.dat", PATH,  CamID+1, startFrame+2*kk+1);
		sprintf(Fname2, "%s/Flow/F%dreY_%05d.dat", PATH,  CamID+1, startFrame+2*kk+1);
		ReadFlowBinary(Fname, Fname2, flowx, flowy, width, height);

		//Track previously detected checkers with the help of global flow
		TrackMarkersFlow(Img, Img+length, IPara, IPara+length, width, height, Ppts, CPts, CPts+nCCorres, nCCorres, DInfo.FmatPC+CamID*9, flowx, flowy, hsubset, LKArg.DIC_Algo, LKArg.Convergence_Criteria, LKArg.ZNCCThreshold, LKArg.InterpAlgo);

		sprintf(Fname, "%s/Sparse/C%d_%05d.txt",PATH,  CamID+1, startFrame+2*kk);
		fp = fopen(Fname, "w+");
		for(ii=0; ii<nCCorres; ii++)
		fprintf(fp, "%.8f %.8f \n", CPts[nCCorres+ii].x, CPts[nCCorres+ii].y);
		fclose(fp);*/
	}

	delete[]Img;
	delete[]SImg;
	delete[]IPara;
	delete[]flowx;
	delete[]flowy;

	return;
}
int  CheckerDetectionTrackingDriver(int CamID, int nCams, int  nPros, int frameJump, int startF, int nframes, int pwidth, int pheight, int width, int height, char *PATH, bool byTracking)
{
	//1: Set up input arguments
	char Fname[100];

	int ii, jj;
	int PSncols, checkerSize = 8, nPpts = 3871, InterpAlgo = 5;
	double GaussianBlur = 0.707, EpipThresh = 0.1;
	vector<double> PatternAngles; PatternAngles.push_back(0.0), PatternAngles.push_back(30.0);

	LKParameters LKArg;
	LKArg.Convergence_Criteria = 1, LKArg.DIC_Algo = 1, LKArg.InterpAlgo = InterpAlgo, LKArg.IterMax = 20;
	LKArg.Gsigma = 0.707, LKArg.ZNCCThreshold = 0.98;

	double *CamProScale = new double[nCams];
	sprintf(Fname, "%s/CamProScale.txt", PATH);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load CamProScale" << endl;
		return 3;
	}
	else
	{
		for (ii = 0; ii < nCams; ii++)
			fscanf(fp, "%lf ", &CamProScale[ii]);
		fclose(fp);
	}

	CPoint ProjectorCorressBoundary[2];
	ProjectorCorressBoundary[0].x = 10, ProjectorCorressBoundary[0].y = 10;
	ProjectorCorressBoundary[1].x = pwidth - 10, ProjectorCorressBoundary[1].y = pheight - 10;

	//2. Set up data input
	DevicesInfo DInfo(nCams, nPros);
	if (!SetUpDevicesInfo(DInfo, PATH))
	{
		cout << "Cannot load Camera Projector info" << endl;
		return 1;
	}

	char *PImg = new char[pwidth*pheight];
	double *PImgD = new double[pwidth*pheight];
	double *PPara = new double[pwidth*pheight];

	sprintf(Fname, "%s/ProjectorPattern.png", PATH);
	IplImage *view = cvLoadImage(Fname, 0);
	if (view == NULL)
	{
		cout << "Cannot load ProjectorPattern" << endl;
		return 2;
	}
	else
	{
		for (jj = 0; jj < pheight; jj++)
			for (ii = 0; ii < pwidth; ii++)
				PImg[ii + (pheight - 1 - jj)*pwidth] = view->imageData[ii + jj*pwidth];
		cvReleaseImage(&view);
	}

	Gaussian_smooth(PImg, PImgD, pheight, pwidth, 255.0, GaussianBlur);
	Generate_Para_Spline(PImgD, PPara, pwidth, pheight, InterpAlgo);

	int *PcorType = new int[nPpts];
	CPoint2 *Pcorners = new CPoint2[nPpts];

	sprintf(Fname, "%s/ProCorners.txt", PATH);
	fp = fopen(Fname, "r");
	if (fp == NULL)
		cout << "Cannot load ProCorners" << endl;

	fscanf(fp, "%d %d %d", &PSncols, &ii, &jj);
	for (ii = 0; ii < nPpts; ii++)
		fscanf(fp, "%d %lf %lf ", &PcorType[ii], &Pcorners[ii].x, &Pcorners[ii].y);
	fclose(fp);

	//3. Run detection tracking function
	for (int ii = startF; ii < startF + nframes; ii += 2)
	{
		if (byTracking)
			CorrespondenceByTrackingMarkers(PATH, Pcorners, PcorType, PPara, DInfo, LKArg, CamProScale, PatternAngles, checkerSize, nPpts, PSncols, ProjectorCorressBoundary, EpipThresh, 2, ii, CamID, width, height, pwidth, pheight);
		else
			DetectTrackMarkersandCorrespondence(PATH, frameJump, Pcorners, PcorType, PPara, DInfo, LKArg, CamProScale, PatternAngles, checkerSize, nPpts, PSncols, ProjectorCorressBoundary, EpipThresh, 2, ii, CamID, width, height, pwidth, pheight);
	}

	delete[]CamProScale;
	delete[]PImg;
	delete[]PImgD;
	delete[]PPara;
	delete[]PcorType;
	delete[]Pcorners;

	return 0;
}
void CleanValidChecker(const int nCams, int nPros, int frameJump, int startF, int nframes, CPoint2 *IROI, CPoint2 *PROI, char *PATH)
{
	char Fname[100];
	int ii, jj, kk, mm, Pid;
	const int nPpts = 2962;
	double x, y;

	CPoint2 PointID[3 * nPpts];
	int CXmask[2 * nPpts];
	int mask[nPpts];
	int mask2[nPpts];

	sprintf(Fname, "%s/ProCorners.txt", PATH);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
		cout << "Cannot load " << Fname << endl;
	fscanf(fp, "%d %d %d ", &jj, &jj, &jj);
	for (ii = 0; ii < nPpts; ii++)
		fscanf(fp, "%d %lf %lf ", &jj, &PointID[ii + 2 * nPpts].x, &PointID[ii + 2 * nPpts].y);
	fclose(fp);

	for (kk = 0; kk < nCams; kk++)
	{
		for (int ll = 0; ll < nframes / 2; ll++)
		{
			for (ii = 0; ii < nPpts; ii++)
				mask2[ii] = 0;

			for (jj = 0; jj<2; jj++)
			{
				sprintf(Fname, "%s/C%d_%d_Ppts.txt", PATH, kk + 1, startF + 2 * ll + jj*frameJump);
				FILE *fp = fopen(Fname, "r");
				if (fp == NULL)
					cout << "Cannot load " << Fname << endl;
				ii = 0, mm = 0;
				while (fscanf(fp, "%d %lf %lf ", &Pid, &x, &y) != EOF)
				{
					CXmask[ii + jj*nPpts] = Pid;
					ii++;
					if (x>PROI[2 * ll].x && y > PROI[2 * ll].y && x < PROI[2 * ll + 1].x && y < PROI[2 * ll + 1].y)
						mask2[Pid] = 1;
				}
				fclose(fp);
			}

			for (ii = 0; ii < 2 * nPpts; ii++)
			{
				PointID[ii].x = 0.0;
				PointID[ii].y = 0.0;
			}

			for (jj = 0; jj<2; jj++)
			{
				sprintf(Fname, "%s/Sparse/C%d_%05d.txt", PATH, kk + 1, startF + 2 * ll + jj*frameJump);
				FILE *fp = fopen(Fname, "r");
				if (fp == NULL)
					cout << "Cannot load " << Fname << endl;
				ii = 0;
				while (fscanf(fp, "%lf %lf ", &x, &y) != EOF)
				{
					if (x>IROI[0].x && y > IROI[0].y && x < IROI[1].x && y < IROI[1].y)
					{
						PointID[CXmask[ii + jj*nPpts] + jj*nPpts].x = x;
						PointID[CXmask[ii + jj*nPpts] + jj*nPpts].y = y;
					}
					ii++;
				}
				fclose(fp);
			}

			int nvalid = 0;
			for (ii = 0; ii < nPpts; ii++)
			{
				mask[ii] = 0;
				for (jj = 0; jj < 2; jj++)
					if ((PointID[ii + jj*nPpts].x < 1.0 && PointID[ii + jj*nPpts].y < 1.0) || mask2[ii] == 0)
						break;
				if (jj == 2)
				{
					mask[ii] = 1;
					nvalid++;
				}
			}

			for (jj = 0; jj < 2; jj++)
			{
				sprintf(Fname, "%s/Sparse/P%d_%05d.txt", PATH, kk + 1, startF + 2 * ll + jj*frameJump); FILE *fp = fopen(Fname, "w+");
				if (fp == NULL)
					cout << "Cannot write " << Fname << endl;
				for (ii = 0; ii < nPpts; ii++)
				{
					if (mask[ii] == 1)
						fprintf(fp, "%d %.4f %.4f \n", ii, PointID[ii + 2 * nPpts].x, PointID[ii + 2 * nPpts].y);
				}
				fclose(fp);
			}

			for (jj = 0; jj < 2; jj++)
			{
				sprintf(Fname, "%s/Sparse/CC%d_%05d.txt", PATH, kk + 1, startF + 2 * ll + jj*frameJump); FILE *fp = fopen(Fname, "w+");
				if (fp == NULL)
					cout << "Cannot write " << Fname << endl;
				for (ii = 0; ii < nPpts; ii++)
					if (mask[ii] == 1)
						fprintf(fp, "%.4f %.4f \n", PointID[ii + jj*nPpts].x, PointID[ii + jj*nPpts].y);
				fclose(fp);
			}
		}
	}

	return;
}
void CleanValidCheckerStereo(const int nCams, int startF, int nframes, CPoint2 *IROI, CPoint2 *PROI, char *PATH)
{
	char Fname[100];
	int ii, jj, kk, mm, Pid;
	const int nPpts = 2962;
	double x, y;

	CPoint2 *PointID = new CPoint2[(2 * nCams + 1)*nPpts];
	int *CXmask = new int[2 * nCams*nPpts];
	int *mask = new int[nPpts];
	int *mask2 = new int[nPpts];

	sprintf(Fname, "%s/ProCorners.txt", PATH);
	FILE *fp = fopen(Fname, "r");
	fscanf(fp, "%d %d %d ", &jj, &jj, &jj);
	for (ii = 0; ii < nPpts; ii++)
		fscanf(fp, "%d %lf %lf ", &jj, &PointID[ii + 2 * nPpts*nCams].x, &PointID[ii + 2 * nPpts*nCams].y);
	fclose(fp);

	for (int ll = 0; ll < nframes / 2; ll++)
	{
		for (ii = 0; ii < nPpts; ii++)
			mask2[ii] = 0;

		for (kk = 0; kk < nCams; kk++)
		{
			for (jj = 0; jj<2; jj++)
			{
				sprintf(Fname, "%s/C%d_%d_Ppts.txt", PATH, kk + 1, startF + 2 * ll + jj);
				fp = fopen(Fname, "r");
				if (fp == NULL)
				{
					cout << "Cannot load " << Fname << endl;
					return;
				}
				ii = 0, mm = 0;
				while (fscanf(fp, "%d %lf %lf ", &Pid, &x, &y) != EOF)
				{
					CXmask[ii + (2 * kk + jj)*nPpts] = Pid;
					ii++;
					if (x>PROI[2 * ll].x && y > PROI[2 * ll].y && x < PROI[2 * ll + 1].x && y < PROI[2 * ll + 1].y)
						mask2[Pid] = 1;
				}
				fclose(fp);
			}
		}

		for (ii = 0; ii < 2 * nPpts*nCams; ii++)
		{
			PointID[ii].x = 0.0;
			PointID[ii].y = 0.0;
		}

		for (kk = 0; kk < nCams; kk++)
		{
			for (jj = 0; jj<2; jj++)
			{
				sprintf(Fname, "%s/Sparse/C%d_%05d.txt", PATH, kk + 1, startF + 2 * ll + jj);
				FILE *fp = fopen(Fname, "r");
				if (fp == NULL)
				{
					cout << "Cannot load " << Fname << endl;
					return;
				}
				ii = 0;
				while (fscanf(fp, "%lf %lf ", &x, &y) != EOF)
				{
					if (x>IROI[0].x && y > IROI[0].y && x < IROI[1].x && y < IROI[1].y)
					{
						PointID[CXmask[ii + (2 * kk + jj)*nPpts] + (2 * kk + jj)*nPpts].x = x;
						PointID[CXmask[ii + (2 * kk + jj)*nPpts] + (2 * kk + jj)*nPpts].y = y;
					}
					ii++;
				}
				fclose(fp);
			}
		}

		int nvalid = 0;
		for (ii = 0; ii < nPpts; ii++)
		{
			mask[ii] = 0;
			for (jj = 0; jj < 2 * nCams; jj++)
				if ((PointID[ii + jj*nPpts].x < 1.0 && PointID[ii + jj*nPpts].y < 1.0) || mask2[ii] == 0)
					break;
			if (jj == 2 * nCams)
			{
				mask[ii] = 1;
				nvalid++;
			}
		}

		for (kk = 0; kk < nCams; kk++)
		{
			for (jj = 0; jj < 2; jj++)
			{
				sprintf(Fname, "%s/Sparse/CCS%d_%05d.txt", PATH, kk + 1, startF + 2 * ll + jj); fp = fopen(Fname, "w+");
				for (ii = 0; ii < nPpts; ii++)
					if (mask[ii] == 1)
						fprintf(fp, "%.4f %.4f \n", PointID[ii + (2 * kk + jj)*nPpts].x, PointID[ii + (2 * kk + jj)*nPpts].y);
				fclose(fp);
			}
		}
	}

	delete[]PointID;
	delete[]CXmask;
	delete[]mask;
	delete[]mask2;

	return;
}
void TwoDimTriangulation(int startF, int nCams, int nPros, int frameJump, int nframes, int width, int height, char *PATH)
{
	const double displacementThresh = 0.1;
	const int nPpts = 2962;
	int ii, jj, kk, ll, tt, nvalid;
	char Fname[100];

	CPoint2 ValidCorners[nPpts];
	Scalar delaunay_color(255, 0, 0);
	Rect rect(0, 0, width, height);

	FILE *fp;
	bool flag;
	double tx, ty;
	for (ll = 0; ll < nCams; ll++)
	{
		for (kk = 0; kk < nframes; kk++)
		{
			nvalid = 0;
			sprintf(Fname, "%s/Sparse/P%d_%05d.txt", PATH, ll + 1, startF + kk*frameJump);
			fp = fopen(Fname, "r");
			if (fp == NULL)
				cout << "Cannot load " << Fname << endl;
			while (fscanf(fp, "%d %lf %lf ", &tt, &tx, &ty) != EOF)
			{
				flag = true;
				for (ii = 0; ii < nvalid; ii++)
				{
					if (abs(ValidCorners[ii].x - tx) < 0.1 && abs(ValidCorners[ii].y - ty) < 0.1)
					{
						flag = false;
						break;
					}
				}
				if (flag)
				{
					ValidCorners[ii].x = tx, ValidCorners[ii].y = ty;
					nvalid++;
				}
			}
			fclose(fp);

			Subdiv2D subdiv(rect);
			for (ii = 0; ii < nvalid; ii++)
			{
				Point2f fp((float)ValidCorners[ii].x, (float)ValidCorners[ii].y);
				subdiv.insert(fp);
			}

			vector<Vec6f> triangleList;
			subdiv.getTriangleList(triangleList);
			vector<Point> pt(3);

			sprintf(Fname, "%s/Sparse/tripletList%d_%05d.txt", PATH, ll + 1, startF + kk*frameJump); fp = fopen(Fname, "w+");
			if (fp == NULL)
				cout << "Cannot write " << Fname << endl;
			sprintf(Fname, "%s/Sparse/tripletCoord%d_%05d.txt", PATH, ll + 1, startF + kk*frameJump); FILE *fp2 = fopen(Fname, "w+");
			if (fp2 == NULL)
				cout << "Cannot write " << Fname << endl;
			for (ii = 0; ii < triangleList.size(); ii++)
			{
				bool breakflag = false;
				Vec6f t = triangleList[ii];
				for (jj = 0; jj < 3; jj++)
				{
					if (t[2 * jj] > width || t[2 * jj] < 0)
					{
						breakflag = true;
						break;
					}

					if (t[2 * jj + 1] > height || t[2 * jj + 1] < 0)
					{
						breakflag = true;
						break;
					}
				}
				if (!breakflag)
				{
					for (jj = 0; jj < nvalid; jj++)
					{
						if (abs(ValidCorners[jj].x - t[0]) < displacementThresh &&abs(ValidCorners[jj].y - t[1]) < displacementThresh)
						{
							fprintf(fp, "%d ", jj);
							break;
						}
					}
					for (jj = 0; jj < nvalid; jj++)
					{
						if (abs(ValidCorners[jj].x - t[2]) < displacementThresh &&abs(ValidCorners[jj].y - t[3]) < displacementThresh)
						{
							fprintf(fp, "%d ", jj);
							break;
						}
					}
					for (jj = 0; jj < nvalid; jj++)
					{
						if (abs(ValidCorners[jj].x - t[4]) < displacementThresh &&abs(ValidCorners[jj].y - t[5]) < displacementThresh)
						{
							fprintf(fp, "%d \n", jj);
							break;
						}
					}
					fprintf(fp2, "%.4f %.4f %.4f %.4f %.4f %.4f \n", t[0], t[1], t[2], t[3], t[4], t[5]);
				}
			}
			fclose(fp), fclose(fp2);

			Mat img(rect.size(), CV_8UC3);
			img = Scalar::all(0);
			for (size_t ii = 0; ii < triangleList.size(); ii++)
			{
				bool breakflag = false;
				Vec6f t = triangleList[ii];
				for (jj = 0; jj < 3; jj++)
				{
					if (t[2 * jj] > width || t[2 * jj] < 0)
					{
						breakflag = true;
						break;
					}

					if (t[2 * jj + 1] > height || t[2 * jj + 1] < 0)
					{
						breakflag = true;
						break;
					}
				}
				if (!breakflag)
				{
					pt[0] = Point(cvRound(t[0]), height - cvRound(t[1]));
					pt[1] = Point(cvRound(t[2]), height - cvRound(t[3]));
					pt[2] = Point(cvRound(t[4]), height - cvRound(t[5]));
					line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
					line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
					line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
				}
			}

			if (kk % 2 == 1)
			{
				//imshow("Tesselation", img); waitKey(100);
				sprintf(Fname, "%s/Sparse/Triangulation%d_%05d.jpg", PATH, ll + 1, startF + kk*frameJump);
				imwrite(Fname, img);
			}
		}
	}
	//destroyWindow("Tesselation");

	return;
}

double SearchLK(CPoint2 From, CPoint2 &Target, double *Img1Para, double *Img2Para, int nchannels, int width1, int height1, int width2, int height2, LKParameters LKArg, double *Timg, double *T, double *iWp, double *direction, double* iCovariance)
{
	int i, j, k, kk, iii, jjj, ij, i2, j2;
	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold, pssdabThresh = LKArg.PSSDab_thresh;

	double ii, jj, II, JJ, a, b, gx, gy, DIC_Coeff, DIC_Coeff_min, t_1, t_2, t_3, t_4, t_5, t_6, m_F, m_G, S[9], p_best[14];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2);
	double conv_crit_2 = conv_crit_1*0.01;
	int NN[] = { 3, 7, 4, 8, 9, 13, 10, 14 };
	int jumpStep[2] = { 1, 2 };
	int DIC_Algo2 = DIC_Algo, nn, nExtraParas = 2, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	if (DIC_Algo == 4)
	{
		nn = 7, DIC_Algo2 = DIC_Algo;
		DIC_Algo = 1;
	}
	else if (DIC_Algo == 5)
	{
		nn = 7, DIC_Algo2 = DIC_Algo;
		DIC_Algo = 1;
	}
	else if (DIC_Algo == 6)
	{
		nn = 8, DIC_Algo2 = DIC_Algo;
		DIC_Algo = 3;
	}
	else if (DIC_Algo == 7)
	{
		nn = 8, DIC_Algo2 = DIC_Algo;
		DIC_Algo = 3;
	}
	else
		nn = NN[DIC_Algo];

	double AA[196], BB[14], CC[14], p[14];
	for (i = 0; i < nn; i++)
		p[i] = (i == nn - 2 ? 1.0 : 0.0);

	int length1 = width1*height1, length2 = width2*height2, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;

	bool createMem = false;
	if (Timg == NULL)
	{
		Timg = new double[Tlength*nchannels];
		T = new double[2 * Tlength*nchannels];
	}

	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			ii = From.x + iii;
			jj = From.y + jjj;

			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(Img1Para + kk*length1, width1, height1, ii, jj, S, -1, Interpolation_Algorithm);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false;
	FILE *fp;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
				for (kk = 0; kk < nchannels; kk++)
					fprintf(fp, "%.2f ", Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	bool useInitPara = false;
	if (iWp != NULL)
	{
		useInitPara = true;
		p[2] = iWp[0], p[3] = iWp[1], p[4] = iWp[2], p[5] = iWp[3];
	}

	/// Let's start with only translation and only match the at the highest level of the pyramid
	bool Break_Flag = false;
	DIC_Coeff_min = 1e10;
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= 2)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;
		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0;
			t_2 = 0.0;
			for (i = 0; i < 4; i++)
				AA[i] = 0.0;
			for (i = 0; i < 2; i++)
				BB[i] = 0.0;

			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					if (DIC_Algo == 1 || DIC_Algo == 3)
						II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					else
						II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];

					if (II<0.0 || II>(double)(width1 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height1 - 1) - (1e-10))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(Img2Para + kk*length2, width2, height2, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						m_G = S[3 * kk];
						if (printout)
							fprintf(fp, "%.2f ", m_G);
						t_3 = m_G - m_F;
						CC[0] = S[3 * kk + 1], CC[1] = S[3 * kk + 2];

						for (i = 0; i < 2; i++)
							BB[i] += t_3*CC[i];

						for (j = 0; j < 2; j++)
							for (i = j; i < 2; i++)
								AA[j * 2 + i] += CC[i] * CC[j];

						t_1 += t_3*t_3, t_2 += m_F*m_F;
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			DIC_Coeff = t_1 / t_2;
			mat_completeSym(AA, 2);
			QR_Solution_Double(AA, BB, 2, 2);
			for (i = 0; i < 2; i++)
				p[i] -= BB[i];


			if (DIC_Coeff != DIC_Coeff || DIC_Coeff > 50 || abs(p[0]) > hsubset || abs(p[1]) > hsubset)
			{
				if (createMem)
				{
					delete[]T;
					delete[]Timg;
				}
				return 0.0;
			}

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				p_best[0] = p[0], p_best[1] = p[1];
				if (p[0] != p[0] || p[1] != p[1])
				{
					if (createMem)
					{
						delete[]T;
						delete[]Timg;
					}
					return 0.0;
				}
			}

			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				break;
		}
	}
	p[0] = p_best[0], p[1] = p_best[1];

	if (DIC_Algo <= 1)
	{
		p[0] = 0.5*(p[0] / direction[0] + p[1] / direction[1]);
		if (DIC_Algo == 0)
			p[1] = 1.0, p[2] = 0.0;
		else
			p[1] = 0.0, p[2] = 0.0, p[3] = 0.0, p[4] = 0.0;
	}

	if (useInitPara)
	{
		if (DIC_Algo == 1)
			p[1] = iWp[0], p[2] = iWp[1], p[3] = iWp[2], p[4] = iWp[3], p[5] = 1.0, p[6] = 0.0;
		else if (DIC_Algo == 3)
			p[2] = iWp[0], p[3] = iWp[1], p[4] = iWp[2], p[5] = iWp[3], p[6] = 1.0, p[7] = 0.0;
	}

	//Now, do the full DIC
	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
				fp = fopen("C:/temp/tar.txt", "w+");

			a = p[nn - 2], b = p[nn - 1];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					if (DIC_Algo == 0)
					{
						II = Target.x + iii + p[0] * direction[0];
						JJ = Target.y + jjj + p[0] * direction[1];
					}
					else if (DIC_Algo == 1) //afine
					{
						II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj;
						JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
					}
					else if (DIC_Algo == 2)
					{
						II = Target.x + iii + p[0];
						JJ = Target.y + jjj + p[1];
					}
					else if (DIC_Algo == 3)
					{
						II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj;
						JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
					}

					if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(Img2Para + kk*length2, width2, height2, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

						m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
						m_G = S[3 * kk];

						if (printout)
							fprintf(fp, "%.2f ", m_G);

						gx = S[3 * kk + 1], gy = S[3 * kk + 2];
						t_3 = a*m_G + b - m_F;

						t_4 = a, t_5 = t_4*gx, t_6 = t_4*gy;
						if (DIC_Algo == 0)
						{
							CC[0] = t_5*direction[0] + t_6*direction[1];
							CC[1] = m_G, CC[2] = 1.0;
						}
						else if (DIC_Algo == 1)
						{
							CC[0] = t_5*direction[0] + t_6*direction[1];
							CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
							CC[5] = m_G, CC[6] = 1.0;
						}
						else if (DIC_Algo == 2)
						{
							CC[0] = t_5, CC[1] = t_6;
							CC[2] = m_G, CC[3] = 1.0;
						}
						else if (DIC_Algo == 3)
						{
							CC[0] = t_5, CC[1] = t_6;
							CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
							CC[6] = m_G, CC[7] = 1.0;
						}

						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = j; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j];
						}

						t_1 += t_3*t_3;
						t_2 += m_F*m_F;
					}
				}
				if (printout)
					fprintf(fp, "\n");
			}
			if (printout)
				fclose(fp);

			DIC_Coeff = t_1 / t_2;

			mat_completeSym(AA, nn);
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff != DIC_Coeff || DIC_Coeff > 50)
			{
				if (createMem)
					delete[]T, delete[]Timg;
				return 0.0;
			}
			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
				if (p[0] != p[0])
				{
					if (createMem)
						delete[]T, delete[]Timg;
					return 0.0;
				}
			}

			if (DIC_Algo <= 1)
			{
				if (abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
				{
					if (createMem)
						delete[]T, delete[]Timg;
					return 0.0;
				}
				if (fabs(BB[0]) < conv_crit_1)
				{
					for (i = 1; i < nn - nExtraParas; i++)
						if (fabs(BB[i]) > conv_crit_2)
							break;
					if (i == nn - nExtraParas)
						Break_Flag = true;
				}
			}
			else
			{
				if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
				{
					if (createMem)
					{
						delete[]T;
						delete[]Timg;
					}
					return 0.0;
				}
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (i = 2; i < nn - nExtraParas; i++)
					{
						if (fabs(BB[i]) > conv_crit_2)
							break;
					}
					if (i == nn - nExtraParas)
						Break_Flag = true;
				}
			}

			if (Break_Flag)
				break;
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}

	//Quadratic if needed:
	if (DIC_Algo2 > 3)
	{
		DIC_Algo = DIC_Algo2, nn = NN[DIC_Algo];
		if (DIC_Algo == 4)
		{
			p[7] = p[5], p[8] = p[6];
			for (i = 5; i < 7; i++)
				p[i] = 0.0;
		}
		else if (DIC_Algo == 5)
		{
			p[11] = p[5], p[12] = p[6];
			for (i = 5; i < 11; i++)
				p[i] = 0.0;
		}
		else if (DIC_Algo == 6)
		{
			p[8] = p[6], p[9] = p[7];
			for (i = 6; i < 8; i++)
				p[i] = 0.0;
		}
		else if (DIC_Algo == 7)
		{
			p[12] = p[6], p[13] = p[7];
			for (i = 6; i < 12; i++)
				p[i] = 0.0;
		}

		//p_jump_0 = 1;
		for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
		{
			DIC_Coeff_min = 1e10;
			bool Break_Flag = false;

			for (k = 0; k < Iter_Max; k++)
			{
				t_1 = 0.0, t_2 = 0.0;
				for (i = 0; i < nn*nn; i++)
					AA[i] = 0.0;
				for (i = 0; i < nn; i++)
					BB[i] = 0.0;

				a = p[nn - 2], b = p[nn - 1];

				if (printout)
					fp = fopen("C:/temp/tar.txt", "w+");

				for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
				{
					for (iii = -hsubset; iii <= hsubset; iii += p_jump)
					{
						if (DIC_Algo == 4) //irregular
						{
							ij = iii*jjj;
							II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
							JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
						}
						else if (DIC_Algo == 5) //Quadratic
						{
							ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
							II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
							JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
						}
						else if (DIC_Algo == 6)
						{
							ij = iii*jjj;
							II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij;
							JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
						}
						else if (DIC_Algo == 7)
						{
							ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
							II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
							JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
						}

						if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							Get_Value_Spline(Img2Para + kk*length2, width2, height2, II, JJ, S + 3 * kk, 0, Interpolation_Algorithm);

							m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
							m_G = S[3 * kk];

							if (printout)
								fprintf(fp, "%.2f ", m_G);

							gx = S[3 * kk + 1], gy = S[3 * kk + 2];
							t_3 = a*m_G + b - m_F;

							t_4 = a, t_5 = t_4*gx, t_6 = t_4*gy;
							if (DIC_Algo == 4) //irregular
							{
								CC[0] = t_5*direction[0] + t_6*direction[1];
								CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
								CC[5] = t_5*ij, CC[6] = t_6*ij;
								CC[7] = m_G, CC[8] = 1.0;
							}
							else if (DIC_Algo == 5) //Quadratic
							{
								CC[0] = t_5*direction[0] + t_6*direction[1];
								CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
								CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = t_5*i2, CC[8] = t_5*j2, CC[9] = t_6*i2, CC[10] = t_6*j2;
								CC[11] = m_G, CC[12] = 1.0;
							}
							else if (DIC_Algo == 6)  //irregular
							{
								CC[0] = t_5, CC[1] = t_6;
								CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
								CC[6] = t_5*ij, CC[7] = t_6*ij;
								CC[8] = m_G, CC[9] = 1.0;
							}
							else if (DIC_Algo == 7)
							{
								CC[0] = t_5, CC[1] = t_6;
								CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
								CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = t_5*i2, CC[9] = t_5*j2, CC[10] = t_6*i2, CC[11] = t_6*j2;
								CC[12] = m_G, CC[13] = 1.0;
							}

							for (j = 0; j < nn; j++)
							{
								BB[j] += t_3*CC[j];
								for (i = j; i < nn; i++)
									AA[j*nn + i] += CC[i] * CC[j];
							}

							t_1 += t_3*t_3, t_2 += m_F*m_F;
						}
					}
					if (printout)
						fprintf(fp, "\n");
				}
				if (printout)
					fclose(fp);

				DIC_Coeff = t_1 / t_2;
				mat_completeSym(AA, nn);
				QR_Solution_Double(AA, BB, nn, nn);
				for (i = 0; i < nn; i++)
					p[i] -= BB[i];

				if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
				{
					DIC_Coeff_min = DIC_Coeff;
					for (i = 0; i < nn; i++)
						p_best[i] = p[i];
					if (p[0] != p[0])
					{
						if (createMem)
							delete[]T, delete[]Timg;
						return 0.0;
					}
				}

				if (DIC_Algo <= 1)
				{
					if (abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
					{
						if (createMem)
							delete[]T, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1)
					{
						for (i = 1; i < nn - nExtraParas; i++)
							if (fabs(BB[i]) > conv_crit_2)
								break;
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				else
				{
					if (abs(p[0]) > hsubset || abs(p[1]) > hsubset)
					{
						if (createMem)
							delete[]T, delete[]Timg;
						return 0.0;
					}
					if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
					{
						for (i = 2; i < nn - nExtraParas; i++)
							if (fabs(BB[i]) > conv_crit_2)
								break;
						if (i == nn - nExtraParas)
							Break_Flag = true;
					}
				}
				if (Break_Flag)
					break;
			}
			_iter += k;
			// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
			for (i = 0; i < nn; i++)
				p[i] = p_best[i];
		}
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close on convergence, but in case of trouble, zncc is more reliable.
	if (DIC_Coeff_min < pssdabThresh)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				if (DIC_Algo == 0)
					II = Target.x + iii + p[0] * direction[0], JJ = Target.y + jjj + p[0] * direction[1];
				else if (DIC_Algo == 1)
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				else if (DIC_Algo == 2)
					II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];
				else if (DIC_Algo == 3)
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
				else if (DIC_Algo == 4) //irregular
				{
					ij = iii*jjj;
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
					JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
				}
				else if (DIC_Algo == 5) //Quadratic
				{
					ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
					JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
				}
				else if (DIC_Algo == 6)
				{
					ij = iii*jjj;
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
				}
				else if (DIC_Algo == 7)
				{
					ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
					JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
				}

				if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Img2Para + kk*length2, width2, height2, II, JJ, S + 3 * kk, -1, Interpolation_Algorithm);
					if (printout)
						fprintf(fp, "%.4f ", S[3 * kk]);

					T[2 * m] = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];
					T[2 * m + 1] = S[3 * kk];
					t_f += T[2 * m];
					t_g += T[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / m, t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = T[2 * i] - t_f, t_5 = T[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;
	}

	if (createMem)
	{
		delete[]Timg;
		delete[]T;
	}
	if (DIC_Coeff_min > 1.0)
		return 0.0;

	if (DIC_Algo <= 1)
	{
		if (DIC_Coeff_min< znccThresh || p[0] != p[0] || abs(p[0] * direction[0]) > hsubset || abs(p[1] * direction[0]) > hsubset)
			return DIC_Coeff_min;
	}
	else
	{
		if (DIC_Coeff_min< znccThresh || p[0] != p[0] || p[1] != p[1] || abs(p[0]) > 2.0*hsubset || abs(p[1]) > 2.0*hsubset)
			return DIC_Coeff_min;
	}

	if (iCovariance != NULL)
	{
		a = p[nn - 2], b = p[nn - 1];
		for (i = 0; i < nn*nn; i++)
			AA[i] = 0.0;
		for (i = 0; i < nn; i++)
			BB[i] = 0.0;

		int count = 0;
		int mMinusn = Tlength*nchannels - nn;
		double *B = new double[Tlength];
		double *BtA = new double[nn];
		double *AtA = new double[nn*nn];

		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				if (DIC_Algo == 1)
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				else if (DIC_Algo == 2)
					II = Target.x + iii + p[0], JJ = Target.y + jjj + p[1];
				else if (DIC_Algo == 3)
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj, JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj;
				else if (DIC_Algo == 4) //irregular
				{
					ij = iii*jjj;
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij;
					JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij;
				}
				else if (DIC_Algo == 5) //Quadratic
				{
					ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
					II = Target.x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj + p[5] * ij + p[7] * i2 + p[8] * j2;
					JJ = Target.y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj + p[6] * ij + p[9] * i2 + p[10] * j2;
				}
				else if (DIC_Algo == 6)
				{
					ij = iii*jjj;
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij;
					JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij;
				}
				else if (DIC_Algo == 7)
				{
					ij = iii*jjj, i2 = iii*iii, j2 = jjj*jjj;
					II = Target.x + iii + p[0] + p[2] * iii + p[3] * jjj + p[6] * ij + p[8] * i2 + p[9] * j2;
					JJ = Target.y + jjj + p[1] + p[4] * iii + p[5] * jjj + p[7] * ij + p[10] * i2 + p[11] * j2;
				}

				if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
					continue;
				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Img2Para + kk*length2, width2, height2, II, JJ, S, 0, Interpolation_Algorithm);
					m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength], m_G = S[0];

					gx = S[1], gy = S[2];
					t_3 = a*m_G + b - m_F;
					t_5 = a*gx, t_6 = a*gy;

					B[count] = t_3;
					count++;

					if (DIC_Algo == 1)
					{
						CC[0] = t_5*direction[0] + t_6*direction[1];
						CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
						CC[5] = m_G, CC[6] = 1.0;
					}
					else if (DIC_Algo == 2)
					{
						CC[0] = t_5, CC[1] = t_6;
						CC[2] = m_G, CC[3] = 1.0;
					}
					else if (DIC_Algo == 3)
					{
						CC[0] = t_5, CC[1] = t_6;
						CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
						CC[6] = m_G, CC[7] = 1.0;
					}
					else if (DIC_Algo == 4) //irregular
					{
						CC[0] = t_5*direction[0] + t_6*direction[1];
						CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
						CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = m_G, CC[8] = 1.0;
					}
					else if (DIC_Algo == 5) //Quadratic
					{
						CC[0] = t_5*direction[0] + t_6*direction[1];
						CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
						CC[5] = t_5*ij, CC[6] = t_6*ij, CC[7] = t_5*i2, CC[8] = t_5*j2;
						CC[9] = t_6*i2, CC[10] = t_6*j2, CC[11] = m_G, CC[12] = 1.0;
					}
					else if (DIC_Algo == 6)  //irregular
					{
						CC[0] = t_5, CC[1] = t_6;
						CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
						CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = m_G, CC[9] = 1.0;
					}
					else if (DIC_Algo == 7)
					{
						CC[0] = t_5, CC[1] = t_6;
						CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
						CC[6] = t_5*ij, CC[7] = t_6*ij, CC[8] = t_5*i2, CC[9] = t_5*j2, CC[10] = t_6*i2, CC[11] = t_6*j2, CC[12] = m_G, CC[13] = 1.0;
					}

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = j; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += m_F*m_F;
				}
			}
		}
		DIC_Coeff = t_1 / t_2;

		mat_completeSym(AA, nn);
		for (i = 0; i < nn*nn; i++)
			AtA[i] = AA[i];
		for (i = 0; i < nn; i++)
			BtA[i] = BB[i];

		QR_Solution_Double(AA, BB, nn, nn);

		double BtAx = 0.0, BtB = 0.0;
		for (i = 0; i < count; i++)
			BtB += B[i] * B[i];
		for (i = 0; i < nn; i++)
			BtAx += BtA[i] * BB[i];
		double mse = (BtB - BtAx) / mMinusn;

		Matrix iAtA(nn, nn), Cov(nn, nn);
		iAtA.Matrix_Init(AtA);
		iAtA = iAtA.Inversion(true, true);
		Cov = mse*iAtA;

		double det = Cov[0] * Cov[nn + 1] - Cov[1] * Cov[nn];
		iCovariance[0] = Cov[nn + 1] / det, iCovariance[1] = -Cov[1] / det, iCovariance[2] = iCovariance[1], iCovariance[3] = Cov[0] / det; //actually, this is inverse of the iCovariance

		delete[]B;
		delete[]BtA;
		delete[]AtA;
	}

	if (useInitPara)
	{
		if (DIC_Algo == 1 || DIC_Algo == 4 || DIC_Algo == 5)
			iWp[0] = p[1], iWp[1] = p[2], iWp[2] = p[3], iWp[3] = p[4];
		else if (DIC_Algo == 3 || DIC_Algo == 6 || DIC_Algo == 7)
			iWp[0] = p[2], iWp[1] = p[3], iWp[2] = p[4], iWp[3] = p[5];
	}

	if (DIC_Algo < 2 || (DIC_Algo>3 && DIC_Algo < 6))
		Target.x += p[0] * direction[0], Target.y += p[0] * direction[1];
	else
		Target.x += p[0], Target.y += p[1];

	return DIC_Coeff_min;
}
double EpipSearchLK(CPoint2 *dPts, double *EpiLine, double *Img1, double *Img2, double *Para1, double *Para2, int nchannels, int width1, int height1, int width2, int height2, LKParameters LKArg, double *RefPatch, double *ZNCCStorage, double *TarPatch, double *iWp, double *iCovariance)
{
	//Only make sense when you somehow take the lense distortion into account
	int hsubset = LKArg.hsubset, Tsize = 2 * hsubset + 1, Tlength = Tsize*Tsize, step, range;
	double ZNCCThresh = LKArg.ZNCCThreshold;
	int ii, jj, mm, nn, length1 = width1*height1, length2 = width2*height2;

	//Assme that the line ax+by+c = 0;
	double denum = sqrt(EpiLine[0] * EpiLine[0] + EpiLine[1] * EpiLine[1]);
	double xx, yy, direction[2] = { -EpiLine[1] / denum, EpiLine[0] / denum };

	CPoint iPts[2];
	CPoint2 t1, t2, Pts[2];
	if (dPts[0].x < 10.0 || dPts[0].x > width1 - 10.0 || dPts[0].y < 10.0 || dPts[0].y > height1 - 10.0)
	{
		dPts[1].x = 9e9, dPts[1].y = 9e9;
		if (iCovariance)
		{
			iCovariance[0] = 9e9, iCovariance[1] = 9e9, iCovariance[2] = 9e9, iCovariance[3] = 9e9;
		}
		return 0.0;
	}
	else if (dPts[1].x < 10.0 || dPts[1].x > width2 - 10.0 || dPts[1].y < 10.0 || dPts[1].y > height2 - 10.0)
	{
		dPts[1].x = 9e9, dPts[1].y = 9e9;
		if (iCovariance)
		{
			iCovariance[0] = 9e9, iCovariance[1] = 9e9, iCovariance[2] = 9e9, iCovariance[3] = 9e9;
		}
		return 0.0;
	}

	int bestIncre;
	double zncc, znccBest = -1.0;

	//iWp is also updated in SearcLK if process succeeds.
	zncc = SearchLK(dPts[0], dPts[1], Para1, Para2, nchannels, width1, height1, width2, height2, LKArg, RefPatch, ZNCCStorage, iWp, direction, iCovariance);
	if (zncc < ZNCCThresh)
	{
		iPts[0].x = (int)(dPts[0].x + 0.5), iPts[0].y = (int)(dPts[0].y + 0.5);
		if (iWp != NULL)
		{
			for (jj = -hsubset; jj <= hsubset; jj++)
				for (ii = -hsubset; ii <= hsubset; ii++)
					RefPatch[(ii + hsubset) + (jj + hsubset)*Tsize] = Img1[(iPts[0].x + ii) + (iPts[0].y + jj)*width1];
		}

		//Search along the epipolar line in a multi resolution manner
		CPoint2 startSearchP;
		if (width1 == width2) // optical flow case
		{
			startSearchP.x = dPts[0].x, startSearchP.y = dPts[0].y;
			step = 2, range = hsubset > 8 ? step*hsubset : (step + 2)*hsubset;
		}
		else //stereo
		{
			startSearchP.x = dPts[1].x, startSearchP.y = dPts[1].y;
			step = 1, range = hsubset > 8 ? step*hsubset : (step + 1)*hsubset;
		}

		bool flag;
		for (jj = -range; jj < range; jj += step)
		{
			Pts[1].x = startSearchP.x + jj*direction[0];
			Pts[1].y = startSearchP.y + jj*direction[1];

			if (Pts[1].x <5 * hsubset || Pts[1].x >width2 - 5 * hsubset || Pts[1].y <5 * hsubset || Pts[1].y >height2 - 5 * hsubset)
				continue;

			if (iWp != NULL)
			{
				flag = true;
				for (mm = -hsubset; mm <= hsubset && flag; mm++)
					for (nn = -hsubset; nn <= hsubset; nn++)
					{
						xx = Pts[1].x + nn + nn*iWp[0] + mm*iWp[1];
						yy = Pts[1].y + mm + nn*iWp[2] + mm*iWp[3];
						if (xx<2 || xx>width2 - 2 || yy<2 || yy>height2 - 2)
						{
							flag = false; break;
						}
						else
							TarPatch[(nn + hsubset) + (mm + hsubset)*Tsize] = BilinearInterp(Img2, width2, height2, xx, yy);
						//Get_Value_Spline(Para2, width2, height2, xx, yy, S, -1, LKArg.InterpAlgo);
						//TarPatch[(nn+hsubset) + (mm+hsubset)*Tsize] = S[0];
					}
				if (flag)
					zncc = ComputeZNCCPatch(RefPatch, TarPatch, hsubset, nchannels, ZNCCStorage);
				else
					zncc = 0.0;
			}
			else
			{
				iPts[1].x = (int)(Pts[1].x + 0.5), iPts[1].y = (int)(Pts[1].y + 0.5);
				zncc = ComputeZNCC(Img1, Img2, iPts, hsubset, width1, height1, width2, height2, ZNCCStorage);
			}
			if (znccBest < zncc)
			{
				znccBest = zncc;
				bestIncre = jj;
			}
		}

		if (znccBest < ZNCCThresh - 0.1 && znccBest > 0.6)
		{
			for (jj = -2 * range; jj < -range; jj += step)
			{
				Pts[1].x = startSearchP.x + jj*direction[0], Pts[1].y = startSearchP.y + jj*direction[1];

				if (Pts[1].x <5 * hsubset || Pts[1].x >width2 - 5 * hsubset || Pts[1].y <5 * hsubset || Pts[1].y >height2 - 5 * hsubset)
					continue;

				if (iWp != NULL)
				{
					flag = true;
					for (mm = -hsubset; mm <= hsubset && flag; mm++)
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							xx = Pts[1].x + nn + nn*iWp[0] + mm*iWp[1], yy = Pts[1].y + mm + nn*iWp[2] + mm*iWp[3];
							if (xx<2 || xx>width2 - 2 || yy<2 || yy>height2 - 2)
							{
								flag = false; break;
							}
							else
								TarPatch[(nn + hsubset) + (mm + hsubset)*Tsize] = BilinearInterp(Img2, width2, height2, xx, yy);
							//Get_Value_Spline(Para2, width2, height2, xx, yy, S, -1, LKArg.InterpAlgo);
							//TarPatch[(nn+hsubset) + (mm+hsubset)*Tsize] = S[0];
						}
					if (flag)
						zncc = ComputeZNCCPatch(RefPatch, TarPatch, hsubset, nchannels, ZNCCStorage);
					else
						zncc = 0.0;
				}
				else
				{
					iPts[1].x = (int)(Pts[1].x + 0.5), iPts[1].y = (int)(Pts[1].y + 0.5);
					zncc = ComputeZNCC(Img1, Img2, iPts, hsubset, width1, height1, width2, height2, ZNCCStorage);
				}
				if (znccBest < zncc)
				{
					znccBest = zncc;
					bestIncre = jj;
				}
			}

			for (jj = range; jj < 2 * range; jj += step)
			{
				Pts[1].x = startSearchP.x + jj*direction[0], Pts[1].y = startSearchP.y + jj*direction[1];

				if (Pts[1].x <5 * hsubset || Pts[1].x >width2 - 5 * hsubset || Pts[1].y <5 * hsubset || Pts[1].y >height2 - 5 * hsubset)
					continue;

				if (iWp != NULL)
				{
					flag = true;
					for (mm = -hsubset; mm <= hsubset && flag; mm++)
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							xx = Pts[1].x + nn + nn*iWp[0] + mm*iWp[1], yy = Pts[1].y + mm + nn*iWp[2] + mm*iWp[3];
							if (xx<2 || xx>width2 - 2 || yy<2 || yy>height2 - 2)
							{
								flag = false; break;
							}
							else
								TarPatch[(nn + hsubset) + (mm + hsubset)*Tsize] = BilinearInterp(Img2, width2, height2, xx, yy);
							//Get_Value_Spline(Para2, width2, height2, xx, yy, S, -1, LKArg.InterpAlgo);
							//TarPatch[(nn+hsubset) + (mm+hsubset)*Tsize] = S[0];
						}
					if (flag)
						zncc = ComputeZNCCPatch(RefPatch, TarPatch, hsubset, nchannels, ZNCCStorage);
					else
						zncc = 0.0;
				}
				else
				{
					iPts[1].x = (int)(Pts[1].x + 0.5), iPts[1].y = (int)(Pts[1].y + 0.5);
					zncc = ComputeZNCC(Img1, Img2, iPts, hsubset, width1, height1, width2, height2, ZNCCStorage);
				}
				if (znccBest < zncc)
				{
					znccBest = zncc;
					bestIncre = jj;
				}
			}
		}

		if (znccBest > ZNCCThresh - 0.1)
		{
			if (step > 1)
			{
				startSearchP.x = startSearchP.x + bestIncre*direction[0], startSearchP.y = startSearchP.y + bestIncre*direction[1];
				for (jj = -step; jj <= step; jj++)
				{
					Pts[1].x = startSearchP.x + jj*direction[0], Pts[1].y = startSearchP.y + jj*direction[1];

					if (Pts[1].x <5 * hsubset || Pts[1].x >width2 - 5 * hsubset || Pts[1].y <5 * hsubset || Pts[1].y >height2 - 5 * hsubset)
						continue;

					if (iWp != NULL)
					{
						flag = true;
						for (mm = -hsubset; mm <= hsubset && flag; mm++)
							for (nn = -hsubset; nn <= hsubset; nn++)
							{
								xx = Pts[1].x + nn + nn*iWp[0] + mm*iWp[1], yy = Pts[1].y + mm + nn*iWp[2] + mm*iWp[3];
								if (xx<2 || xx>width2 - 2 || yy<2 || yy>height2 - 2)
								{
									flag = false; break;
								}
								else
									TarPatch[(nn + hsubset) + (mm + hsubset)*Tsize] = BilinearInterp(Img2, width2, height2, xx, yy);
								//Get_Value_Spline(Para2, width2, height2, xx, yy, S, -1, LKArg.InterpAlgo);
								//TarPatch[(nn+hsubset) + (mm+hsubset)*Tsize] = S[0];
							}
						if (flag)
							zncc = ComputeZNCCPatch(RefPatch, TarPatch, hsubset, nchannels, ZNCCStorage);
						else
							zncc = 0.0;
					}
					else
					{
						iPts[1].x = (int)(Pts[1].x + 0.5), iPts[1].y = (int)(Pts[1].y + 0.5);
						zncc = ComputeZNCC(Img1, Img2, iPts, hsubset, width1, height1, width2, height2, ZNCCStorage);
					}
					if (znccBest < zncc)
					{
						znccBest = zncc;
						bestIncre = jj;
					}
				}
			}

			dPts[1].x = startSearchP.x + bestIncre*direction[0], dPts[1].y = startSearchP.y + bestIncre*direction[1];
			return zncc = SearchLK(dPts[0], dPts[1], Para1, Para2, nchannels, width1, height1, width2, height2, LKArg, RefPatch, ZNCCStorage, iWp, direction, iCovariance);
		}
		else
			return znccBest;
	}
	else
		return zncc;
}
void DIC_DenseScaleSelection(int *Scale, float *Fu, float *Fv, double *Img1, double *Img2, int width, int height, LKParameters LKArg, FlowScaleSelection ScaleSel, double flowThresh, CPoint *DenseFlowBoundary)
{
	//This  function only supports grayscale image
	const int nchannels = 1;
	int ii, jj, kk, startI, stopI, startJ, stopJ;
	int startScale = ScaleSel.startS, stopScale = ScaleSel.stopS, stepIJ = ScaleSel.stepIJ, stepS = ScaleSel.stepS;

	if (DenseFlowBoundary != NULL)
	{
		startI = DenseFlowBoundary[0].x, stopI = DenseFlowBoundary[1].x;
		startJ = DenseFlowBoundary[0].y, stopJ = DenseFlowBoundary[1].y;
	}
	else
	{
		startI = width / 50, stopI = width - width / 50;
		startJ = height / 50, stopJ = height - height / 50;
	}
	//startI = 820, stopI = 880, startJ = 820, stopJ = 880;

	double *Img1Para = new double[width*height];
	double *Img2Para = new double[width*height];
	Generate_Para_Spline(Img1, Img1Para, width, height, LKArg.InterpAlgo);
	Generate_Para_Spline(Img2, Img2Para, width, height, LKArg.InterpAlgo);

	CPoint2 PR, PT, tpoint;
	double ZNCC, difference, minDif, fufv[2];

	double start = omp_get_wtime();
	int percent = 5, increment = 2;

	int apts = (stopJ - startJ)*(stopI - startI) / stepIJ / stepIJ, npts = 0;
	for (jj = startJ; jj <= stopJ; jj += stepIJ)
	{
		for (ii = startI; ii <= stopI; ii += stepIJ)
		{
			if (omp_get_thread_num() == 0)
			{
				if ((100 * npts / apts - percent) > 0)
				{
					percent += increment;
					double elapsed = omp_get_wtime() - start;
					cout << "%" << 100 * npts / apts << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / (percent - increment)*(100.0 - percent + increment) << endl;
				}
			}
			npts++;

			minDif = 9e9;
			for (kk = startScale; kk < stopScale; kk += stepS)
			{
				PR.x = 1.0*ii, PR.y = 1.0*jj, PT.x = PR.x + Fu[ii + jj*width], PT.y = PR.y + Fv[ii + jj*width];

				ZNCC = TMatching(Img1Para, Img2Para, kk, width, height, width, height, nchannels, PR, PT, 1, LKArg.Convergence_Criteria, LKArg.ZNCCThreshold, LKArg.IterMax, 1, fufv, false);
				if (ZNCC < LKArg.ZNCCThreshold)
					continue;

				PT.x = PT.x + fufv[0], PT.y = PT.y + fufv[1], tpoint.x = PR.x, tpoint.y = PR.y;
				ZNCC = TMatching(Img2Para, Img1Para, kk, width, height, width, height, nchannels, PT, tpoint, 1, LKArg.Convergence_Criteria, LKArg.ZNCCThreshold, LKArg.IterMax, 1, fufv, false);
				if (ZNCC < LKArg.ZNCCThreshold)
					continue;
				tpoint.x = tpoint.x + fufv[0], tpoint.y = tpoint.y + fufv[1];

				difference = abs(tpoint.x - PR.x) + abs(tpoint.y - PR.y);
				if (minDif > difference)
				{
					Scale[ii + jj*width] = kk;
					minDif = difference;
				}
			}

			if (minDif > 1.0)
				Scale[ii + jj*width] = -1;
		}
	}
	stopI = ii - stepIJ, stopJ = jj - stepIJ;

	int idx, idy;
	double f00, f01, f10, f11;
	for (jj = startJ; jj <= stopJ; jj++)
	{
		for (ii = startI; ii <= stopI; ii++)
		{
			idx = (ii - startI) / stepIJ, idy = (jj - startJ) / stepIJ;

			f00 = 1.0*Scale[(idx*stepIJ + startI) + (idy*stepIJ + startJ)*width];
			f01 = 1.0*Scale[((idx + 1)*stepIJ + startI) + (idy*stepIJ + startJ)*width];
			f10 = 1.0*Scale[(idx*stepIJ + startI) + ((idy + 1)*stepIJ + startJ)*width];
			f11 = 1.0*Scale[((idx + 1)*stepIJ + startI) + ((idy + 1)*stepIJ + startJ)*width];

			int res = MyFtoI((f01 - f00)*(ii - (idx*stepIJ + startI)) / stepIJ + (f10 - f00)*(jj - (idy*stepIJ + startJ)) / stepIJ + (f11 - f01 - f10 + f00)*(ii - (idx*stepIJ + startI)) / stepIJ*(jj - (idy*stepIJ + startJ)) / stepIJ + f00);
			Scale[ii + jj*width] = res;
		}
	}

	delete[]Img1Para;
	delete[]Img2Para;

	return;
}

void DIC_FindROI(char *lpROI, CPoint Start_Point, int width, int height, CPoint *bound)
{
	//bound[0]: bottom left
	//bound[1] = top right
	int m, n, x, y;
	int length = width*height;

	int *Txy = new int[length * 2];
	for (n = 0; n < length; n++)
		*(lpROI + n) = (char)0;

	m = 0;
	x = Start_Point.x;
	y = Start_Point.y;
	*(lpROI + y*width + x) = (char)255;
	*(Txy + 2 * m + 0) = x;
	*(Txy + 2 * m + 1) = y;
	while (m >= 0)
	{
		x = *(Txy + 2 * m + 0);
		y = *(Txy + 2 * m + 1);
		m--;

		if ((y + 1) < bound[1].y && *(lpROI + (y + 1)*width + x) == (char)0)
		{
			m++;
			*(lpROI + (y + 1)*width + x) = (char)255;
			*(Txy + 2 * m + 0) = x;
			*(Txy + 2 * m + 1) = y + 1;
		}
		if (y > bound[0].y && *(lpROI + (y - 1)*width + x) == (char)0)
		{
			m++;
			*(lpROI + (y - 1)*width + x) = (char)255;
			*(Txy + 2 * m + 0) = x;
			*(Txy + 2 * m + 1) = y - 1;
		}
		if (x > bound[0].x && *(lpROI + y*width + x - 1) == (char)0)
		{
			m++;
			*(lpROI + y*width + x - 1) = (char)255;
			*(Txy + 2 * m + 0) = x - 1;
			*(Txy + 2 * m + 1) = y;
		}
		if ((x + 1) < bound[1].x && *(lpROI + y*width + x + 1) == (char)0)
		{
			m++;
			*(lpROI + y*width + x + 1) = (char)255;
			*(Txy + 2 * m + 0) = x + 1;
			*(Txy + 2 * m + 1) = y;
		}
	}

	delete[]Txy;

	return;
}
void DIC_AddtoQueue(double *Coeff, int *Tindex, int M)
{
	int i, j, t;
	double coeff;
	for (i = 0; i <= M - 1; i++)
	{
		if (*(Coeff + M) > *(Coeff + i))
		{
			coeff = *(Coeff + M);
			t = *(Tindex + M);
			for (j = M - 1; j >= i; j--)
			{
				*(Coeff + j + 1) = *(Coeff + j);
				*(Tindex + j + 1) = *(Tindex + j);
			}
			*(Coeff + i) = coeff;
			*(Tindex + i) = t;
			break;
		}
	}

	return;
}
bool DIC_CheckPointValidity(bool *lpROI, int x_n, int y_n, int width, int height, int hsubset, double validity_ratio)
{
	int m = 0, n = 0, ii, jj, iii, jjj;
	for (jjj = -hsubset; jjj <= hsubset; jjj += 2)
	{
		for (iii = -hsubset; iii <= hsubset; iii += 2)
		{
			m++;
			jj = y_n + jjj;
			ii = x_n + iii;

			if (ii<0 || ii>(width - 1) || jj<0 || jj>(height - 1))
				continue;

			if (*(lpROI + jj*width + ii) == false)
				continue;

			n++;
		}
	}

	if (n < int(m*validity_ratio))
		return false;

	return true;
}
void DIC_Initial_Guess(double *lpImageData, int width, int height, double *UV_Guess, CPoint Start_Point, int *IG_subset, int Initial_Guess_Scheme)
{
	int i;

	for (i = 0; i < 14; i++)
		UV_Guess[i] = 0.0;

	if (Initial_Guess_Scheme == 1) //Epipolar line
	{
		;
	}
	else //Just correlation
	{
		int j, k, m, n, ii, jj, II_0, JJ_0, iii, jjj;
		double ratio = 0.2;
		double t_f, t_g, t_1, t_2, t_3, t_4, t_5, m_F, m_G, C_zncc, C_znssd_min, C_znssd_max;
		int hsubset, m_IG_subset[2];
		int length = width*height;

		m_IG_subset[0] = IG_subset[0] / 2;
		m_IG_subset[1] = IG_subset[1] / 2;

		double *C_znssd = new double[length];
		char *TT = new char[length];
		double *T = new double[2 * (2 * m_IG_subset[1] + 1)*(2 * m_IG_subset[1] + 1)];

		C_znssd_min = 1e12;
		C_znssd_max = -1e12;

		for (n = 0; n < length; n++)
		{
			*(TT + n) = (char)0;
			*(C_znssd + n) = 1e2;
		}

		for (k = 0; k < 2; k++)
		{
			hsubset = m_IG_subset[k];
			for (j = hsubset; j < height - hsubset; j++)
			{
				for (i = hsubset; i < width - hsubset; i++)
				{
					if (*(TT + j*width + i) == (char)1)
						continue;

					m = -1;
					t_f = 0.0;
					t_g = 0.0;
					for (jjj = -hsubset; jjj <= hsubset; jjj++)
					{
						for (iii = -hsubset; iii <= hsubset; iii++)
						{
							jj = Start_Point.y + jjj;
							ii = Start_Point.x + iii;
							JJ_0 = j + jjj;
							II_0 = i + iii;

							m_F = *(lpImageData + jj*width + ii);
							m_G = *(lpImageData + length + JJ_0*width + II_0);

							m++;
							*(T + 2 * m + 0) = m_F;
							*(T + 2 * m + 1) = m_G;
							t_f += m_F;
							t_g += m_G;
						}
					}

					t_f = t_f / (m + 1);
					t_g = t_g / (m + 1);
					t_1 = 0.0;
					t_2 = 0.0;
					t_3 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = *(T + 2 * iii + 0) - t_f;
						t_5 = *(T + 2 * iii + 1) - t_g;
						t_1 += (t_4*t_5);
						t_2 += (t_4*t_4);
						t_3 += (t_5*t_5);
					}
					t_2 = sqrt(t_2*t_3);
					if (t_2 < 1e-10)		// Avoid being divided by 0.
						t_2 = 1e-10;

					C_zncc = t_1 / t_2;

					// Testing shows that C_zncc may not fall into (-1, 1) range, so need the following line.
					if (C_zncc > 1.0 || C_zncc < -1.0)
						C_zncc = 0.0;	// Use 0.0 instead of 1.0 or -1.0

					*(C_znssd + j*width + i) = 2.0*(1.0 - C_zncc);

					if (*(C_znssd + j*width + i) < C_znssd_min)
					{
						C_znssd_min = *(C_znssd + j*width + i);
						UV_Guess[0] = i - Start_Point.x;
						UV_Guess[1] = j - Start_Point.y;
					}

					if (*(C_znssd + j*width + i) > C_znssd_max)
						C_znssd_max = *(C_znssd + j*width + i);	// C_znssd_max should be close to 4.0, C_znssd_min should be close to 0.0
				}
			}

			if (k == 0)
			{
				for (n = 0; n<length; n++)
				{
					if (*(C_znssd + n) >(C_znssd_min + ratio*(C_znssd_max - C_znssd_min)))
						*(TT + n) = (char)1;
				}

				C_znssd_min = 1e12;
				C_znssd_max = -1e12;
			}
		}

		delete[]T;
		delete[]TT;
		delete[]C_znssd;
	}

	return;
}
void DIC_Initial_Guess_Refine(int x_n, int y_n, double *lpImageData, double *Znssd_reqd, bool *lpROI, double *p, int nchannels, int width1, int height1, int width2, int height2, int hsubset, int step, int DIC_Algo, double *direction)
{
	/// Refine initial guess of u and v of the starting point with integral-pixel accuracy using ZNSSD

	int d_u, d_v, u0, v0, U0, V0, alpha, alpha0;
	int m, ii, jj, kk, iii, jjj, II_0, JJ_0;
	int length1 = width1*height1, length2 = width2*height2;
	double t_1, t_2, t_3, t_4, t_5, t_F, t_G, mean_F, mean_G;
	double C_zncc, C_znssd, C_znssd_min;
	C_znssd_min = 1.0E12;

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;

	if (DIC_Algo <= 1) //Epipoloar constraint on the flow
	{
		alpha = 0;
		for (alpha0 = -3 * step; alpha0 <= 3 * step; alpha0++)
		{
			u0 = (int)(direction[0] * (p[0] + 0.5*alpha0) + 0.5);
			v0 = (int)(direction[1] * (p[0] + 0.5*alpha0) + 0.5);

			m = -1;
			mean_F = 0.0;
			mean_G = 0.0;
			if (printout)
			{
				fp1 = fopen("C:/temp/src.txt", "w+");
				fp2 = fopen("C:/temp/tar.txt", "w+");
			}
			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					ii = x_n + iii;
					jj = y_n + jjj;

					if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1))
						continue;

					if (lpROI[jj*width1 + ii] == false)
						continue;

					II_0 = ii + u0 + (int)(p[1] * iii + p[2] * jjj + 0.5);
					JJ_0 = jj + v0 + (int)(p[3] * iii + p[4] * jjj + 0.5);

					if (II_0<0 || II_0>(width2 - 1) || JJ_0<0 || JJ_0>(height2 - 1))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						t_F = lpImageData[jj*width1 + ii + kk*length1];
						t_G = lpImageData[nchannels*length1 + kk*length2 + JJ_0*width2 + II_0];

						if (printout)
						{
							fprintf(fp1, "%.2f ", t_F), fprintf(fp2, "%.2f ", t_G);
						}
						m++;
						Znssd_reqd[2 * m] = t_F;
						Znssd_reqd[2 * m + 1] = t_G;
						mean_F += t_F;
						mean_G += t_G;
					}
				}
				if (printout)
				{
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
				}
			}
			if (printout)
			{
				fclose(fp1); fclose(fp2);
			}
			if (m < 10)
				continue;

			mean_F /= (m + 1);
			mean_G /= (m + 1);
			t_1 = 0.0;
			t_2 = 0.0;
			t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[2 * iii] - mean_F;
				t_5 = Znssd_reqd[2 * iii + 1] - mean_G;
				t_1 += (t_4*t_5);
				t_2 += (t_4*t_4);
				t_3 += (t_5*t_5);
			}

			C_zncc = t_1 / sqrt(t_2*t_3);
			C_znssd = 2.0*(1.0 - C_zncc);

			if (C_znssd < C_znssd_min)
			{
				C_znssd_min = C_znssd;
				alpha = alpha0;
			}
		}

		p[0] = p[0] + 0.5*alpha;
	}
	else //Affine shape 
	{
		U0 = 0, V0 = 0;
		for (d_u = -step; d_u <= step; d_u++)
		{
			for (d_v = -step; d_v <= step; d_v++)
			{
				u0 = d_u + (int)(p[0] + 0.5);
				v0 = d_v + (int)(p[1] + 0.5);

				m = -1;
				mean_F = 0.0;
				mean_G = 0.0;
				if (printout)
				{
					fp1 = fopen("C:/temp/src.txt", "w+");
					fp2 = fopen("C:/temp/tar.txt", "w+");
				}
				for (jjj = -hsubset; jjj <= hsubset; jjj++)
				{
					for (iii = -hsubset; iii <= hsubset; iii++)
					{
						ii = x_n + iii, jj = y_n + jjj;

						if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1))
							continue;

						if (lpROI[jj*width1 + ii] == false)
							continue;

						II_0 = ii + u0 + (int)(p[2] * iii + p[3] * jjj);
						JJ_0 = jj + v0 + (int)(p[4] * iii + p[5] * jjj);

						if (II_0<0 || II_0>(width2 - 1) || JJ_0<0 || JJ_0>(height2 - 1))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							t_F = lpImageData[jj*width1 + ii + kk*length1];
							t_G = lpImageData[nchannels*length1 + kk*length2 + JJ_0*width2 + II_0];

							if (printout && kk == 0)
							{
								fprintf(fp1, "%.2f ", t_F), fprintf(fp2, "%.2f ", t_G);
							}
							m++;
							*(Znssd_reqd + 2 * m + 0) = t_F;
							*(Znssd_reqd + 2 * m + 1) = t_G;
							mean_F += t_F;
							mean_G += t_G;
						}
					}
					if (printout)
					{
						fprintf(fp1, "\n"), fprintf(fp2, "\n");
					}
				}
				if (printout)
				{
					fclose(fp1); fclose(fp2);
				}
				if (m < 10)
					continue;

				mean_F /= (m + 1);
				mean_G /= (m + 1);
				t_1 = 0.0;
				t_2 = 0.0;
				t_3 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = *(Znssd_reqd + 2 * iii + 0) - mean_F;
					t_5 = *(Znssd_reqd + 2 * iii + 1) - mean_G;
					t_1 += (t_4*t_5);
					t_2 += (t_4*t_4);
					t_3 += (t_5*t_5);
				}

				C_zncc = t_1 / sqrt(t_2*t_3);
				C_znssd = 2.0*(1.0 - C_zncc);

				if (C_znssd < C_znssd_min)
				{
					C_znssd_min = C_znssd;
					U0 = u0;
					V0 = v0;
				}
			}
		}

		p[0] = U0;
		p[1] = V0;
	};

	return;
}
double DIC_Compute2(int UV_index_n, int UV_index, double *lpImageData, double *Para, double *lpUV, int *lpUV_xy, double *Znssd_reqd, bool *lpROI, int nchannels, int width1, int height1, int width2, int height2, int UV_length, int DIC_Algo, int hsubset, int step, int Iter_Max, int *iteration_check, double conv_crit_1, double conv_crit_2, int Interpolation_Algorithm, int Analysis_Speed, bool firsttime, double *direction, bool checkZNCC = false, double ZNNCThresh = 0.99)
{
	double DIC_Coeff, a, b;
	int i, j, ii, jj, kk, iii, jjj, iii2, jjj2, ij;
	int k, m, nn, nExtraParas;
	int length1 = width1*height1, length2 = width2*height2;
	int NN[] = { 3, 7, 4, 8, 6, 12 };
	double II, JJ, iii_n, jjj_n;
	double m_F, m_G, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g;
	double S[9];
	double p[8], ip[8], p_best[8];// U, V, Ux, Uy, Vx, Vy, (a) and b.
	double AA[144], BB[12], CC[12], gx, gy;

	direction[0] = 1.0, direction[1] = 0;
	int x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
	int x_n = lpUV_xy[2 * UV_index_n], y_n = lpUV_xy[2 * UV_index_n + 1];

	nn = NN[DIC_Algo];
	nExtraParas = 2;

	for (i = 0; i < nn; i++)
		p[i] = lpUV[i], ip[i] = lpUV[i];

	// The following two lines are needed for large rotation cases.
	if (DIC_Algo == 1)
		;//p[0] = 0.5*((p[0] * direction[0] + (p[1] * (x_n - x) + p[2] * (y_n - y))) / direction[0] + (p[0] * direction[1] + (p[3] * (x_n - x) + p[4] * (y_n - y))) / direction[1]);//Supreeth
	else if (DIC_Algo == 3)
		p[0] += (p[2] * (x_n - x) + p[3] * (y_n - y)), p[1] += (p[4] * (x_n - x) + p[5] * (y_n - y));

	// Refine initial guess of u and v of the starting point with integral-pixel accuracy using ZNSSD
	if (firsttime)
		DIC_Initial_Guess_Refine(x_n, y_n, lpImageData, Znssd_reqd, lpROI, p, nchannels, width1, height1, width2, height2, hsubset, step, DIC_Algo, direction);

	bool printout = false; FILE *fp1 = 0, *fp2 = 0;
	int piis, pixel_increment_in_subset[] = { 1, 2, 2, 3 };
	double DIC_Coeff_min = 9e9;
	/// Iteration: Begin
	bool Break_Flag = false;
	for (k = 0; k < Iter_Max; k++)
	{
		m = -1;
		t_1 = 0.0, t_2 = 0.0;
		for (iii = 0; iii < nn*nn; iii++)
			AA[iii] = 0.0;
		for (iii = 0; iii < nn; iii++)
			BB[iii] = 0.0;

		a = p[nn - 2], b = p[nn - 1];

		if (printout)
			fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

		piis = pixel_increment_in_subset[Analysis_Speed];	// Depending on algorithms, Analysis_Speed may be changed during the iteration loop.
		for (jjj = -hsubset; jjj <= hsubset; jjj += piis)
		{
			for (iii = -hsubset; iii <= hsubset; iii += piis)
			{
				ii = x_n + iii, jj = y_n + jjj;
				if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1) || lpROI[jj*width1 + ii] == false)
					continue;

				if (DIC_Algo == 0)
					II = ii + p[0] * direction[0], JJ = jj + p[0] * direction[1];
				else if (DIC_Algo == 1)
					II = ii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = jj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				if (DIC_Algo == 2)
					II = ii + p[0], JJ = jj + p[1];
				else if (DIC_Algo == 3)
					II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1] + p[4] * iii + p[5] * jjj;
				if (DIC_Algo == 4)
					II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1] + p[4] * iii + p[5] * jjj;
				else if (DIC_Algo == 5)
				{
					iii2 = iii*iii, jjj2 = jjj*jjj, ij = iii*jjj;
					II = ii + p[0] + p[2] * iii + p[3] * jjj + p[6] * iii2*0.5 + p[7] * jjj2*0.5 + p[8] * ij;
					JJ = jj + p[1] + p[4] * iii + p[5] * jjj + p[9] * iii2*0.5 + p[10] * jjj2*0.5 + p[11] * ij;
				}

				if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length2, width2, height2, II, JJ, S, 0, Interpolation_Algorithm);

					m_F = lpImageData[ii + jj*width1 + kk*length1];
					m_G = S[0], gx = S[1], gy = S[2];
					m++;

					if (DIC_Algo < 4)
					{
						t_3 = a*m_G + b - m_F, t_4 = a;
						t_1 += t_3*t_3, t_2 += m_F*m_F;

						t_5 = t_4*gx, t_6 = t_4*gy;
						if (DIC_Algo == 0)
							CC[0] = t_5*direction[0] + t_6*direction[1], CC[1] = m_G, CC[2] = 1.0;
						else if (DIC_Algo == 1)
							CC[0] = t_5*direction[0] + t_6*direction[1], CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj, CC[5] = m_G, CC[6] = 1.0;
						else if (DIC_Algo == 2)
							CC[0] = t_5, CC[1] = t_6, CC[2] = m_G, CC[3] = 1.0;
						else if (DIC_Algo == 3)
							CC[0] = t_5, CC[1] = t_6, CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj, CC[6] = m_G, CC[7] = 1.0;

						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = 0; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j];
						}
					}
					else
					{
						Znssd_reqd[6 * m + 0] = m_F, Znssd_reqd[6 * m + 1] = m_G;
						Znssd_reqd[6 * m + 2] = gx, Znssd_reqd[6 * m + 3] = gy;
						Znssd_reqd[6 * m + 4] = (double)iii, Znssd_reqd[6 * m + 5] = (double)jjj;
						t_1 += m_F, t_2 += m_G;
					}

					if (printout)
						fprintf(fp1, "%.2f ", m_F), fprintf(fp2, "%.2f ", m_G);
				}
			}
			if (printout)
				fprintf(fp1, "\n"), fprintf(fp2, "\n");
		}
		if (printout)
			fclose(fp1), fclose(fp2);

		if (DIC_Algo < 4)
			DIC_Coeff = t_1 / t_2;
		else
		{
			if (k == 0)
			{
				t_f = t_1 / (m + 1);
				t_1 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[6 * iii + 0] - t_f;
					t_1 += t_4*t_4;
				}
				t_ff = sqrt(t_1);
			}
			t_g = t_2 / (m + 1);
			t_2 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_5 = Znssd_reqd[6 * iii + 1] - t_g;
				t_2 += t_5*t_5;
			}
			t_2 = sqrt(t_2);

			DIC_Coeff = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[6 * iii + 0] - t_f;
				t_5 = Znssd_reqd[6 * iii + 1] - t_g;
				t_6 = t_5 / t_2 - t_4 / t_ff;
				t_3 = t_6 / t_2;
				gx = Znssd_reqd[6 * iii + 2], gy = Znssd_reqd[6 * iii + 3];
				iii_n = Znssd_reqd[6 * iii + 4], jjj_n = Znssd_reqd[6 * iii + 5];
				CC[0] = gx, CC[1] = gy;
				CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
				CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
				if (DIC_Algo == 5)
				{
					CC[6] = gx*iii_n*iii_n*0.5, CC[7] = gx*jjj_n*jjj_n*0.5, CC[8] = gx*iii_n*jjj_n;
					CC[9] = gy*iii_n*iii_n*0.5, CC[10] = gy*jjj_n*jjj_n*0.5, CC[11] = gy*iii_n*jjj_n;
				}
				for (j = 0; j < nn; j++)
				{
					BB[j] += t_3*CC[j];
					for (i = 0; i < nn; i++)
						AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
				}

				DIC_Coeff += t_6*t_6;
			}
		}

		if (!IsNumber(DIC_Coeff))
			return 9e9;
		if (!IsFiniteNumber(DIC_Coeff))
			return 9e9;

		QR_Solution_Double(AA, BB, nn, nn);
		for (iii = 0; iii < nn; iii++)
			p[iii] -= BB[iii];

		if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
		{
			DIC_Coeff_min = DIC_Coeff;
			for (iii = 0; iii < nn; iii++)
				p_best[iii] = p[iii];
			if (!IsNumber(p[0]) || !IsNumber(p[1]))
				return 9e9;
		}

		if (DIC_Algo <= 1)
		{
			if (abs((p[0] - ip[0])*direction[0]) > hsubset)
				return 9e9;

			if (fabs(BB[0]) < conv_crit_1)
			{
				for (iii = 1; iii < nn - nExtraParas; iii++)
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				if (iii == nn - nExtraParas)
					if (Analysis_Speed == 1)	// For Analysis_Speed==1, need to run a full "normal speed" analysis
						Analysis_Speed = 0;
					else
						Break_Flag = true;
			}
		}
		else if (DIC_Algo <= 3)
		{
			if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
				return 9e9;
			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn - nExtraParas; iii++)
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				if (iii == nn - nExtraParas)
					if (Analysis_Speed == 1)	// For Analysis_Speed==1, need to run a full "normal speed" analysis
						Analysis_Speed = 0;
					else
						Break_Flag = true;
			}
		}
		else
		{
			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn)
					Break_Flag = true;
			}
		}

		if (Break_Flag)
			break;
	}
	if (k < 1)
		k = 1;
	iteration_check[k - 1]++;
	/// Iteration: End

	if (checkZNCC && DIC_Algo < 4)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, ZNCC, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp2 = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				ii = x_n + iii, jj = y_n + jjj;

				if (DIC_Algo == 0)
					II = ii + p[0] * direction[0], JJ = jj + p[0] * direction[1];
				else if (DIC_Algo == 1)
					II = ii + p[0] * direction[0] + p[1] * iii + p[2] * jjj, JJ = jj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				if (DIC_Algo == 2)
					II = ii + p[0], JJ = jj + p[1];
				else if (DIC_Algo == 3)
					II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1] + p[4] * iii + p[5] * jjj;

				if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length2, width2, height2, II, JJ, S, 0, Interpolation_Algorithm);
					if (printout)
						fprintf(fp2, "%.4f ", S[0]);

					Znssd_reqd[2 * m] = lpImageData[ii + jj*width1 + kk*length1], Znssd_reqd[2 * m + 1] = S[0];
					t_f += Znssd_reqd[2 * m], t_g += Znssd_reqd[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp2, "\n");
		}
		if (printout)
			fclose(fp2);

		t_f = t_f / m, t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = Znssd_reqd[2 * i] - t_f, t_5 = Znssd_reqd[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) < ZNNCThresh)
			DIC_Coeff_min = 1.0;
	}

	if (DIC_Algo <= 1)
	{
		if (abs((p[0] - ip[0])*direction[0]) > hsubset)
			return 9e9;
		else
		{
			for (i = 0; i < nn; i++)
				lpUV[i] = p_best[i];
			return DIC_Coeff_min;
		}
	}
	else
	{
		if (abs(p_best[0] - ip[0]) > hsubset || abs(p_best[1] - ip[1]) > hsubset || p_best[0] != p_best[0] || p_best[1] != p_best[1])
			return 9e9;
		else
		{
			for (i = 0; i < nn; i++)
				lpUV[i] = p_best[i];
			return DIC_Coeff_min;
		}
	}

}
double DIC_Compute(int UV_index_n, int UV_index, double *lpImageData, double *Para, double *lpUV, int *lpUV_xy, double *Znssd_reqd, bool *lpROI, int nchannels, int width1, int height1, int width2, int height2, int UV_length, int DIC_Algo, int hsubset, int step, int Iter_Max, int *iteration_check, double conv_crit_1, double conv_crit_2, int Interpolation_Algorithm, int Analysis_Speed, bool firsttime, double *direction, bool checkZNCC = false, double ZNNCThresh = 0.99)
{
	double DIC_Coeff, a, b;
	int i, j, ii, jj, kk, iii, jjj;
	int k, m, nn, nExtraParas;
	int length1 = width1*height1, length2 = width2*height2;
	int NN[] = { 6, 4 };
	double II, JJ, iii_n, jjj_n;
	double m_F, m_G, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g;
	double S[9];
	double p[6], ip[6], p_best[6];// U, V, Ux, Uy, (a) and b.
	double AA[36], BB[6], CC[6], gx, gy;

	int x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
	int x_n = lpUV_xy[2 * UV_index_n], y_n = lpUV_xy[2 * UV_index_n + 1];

	nn = NN[DIC_Algo];
	nExtraParas = 2;

	for (i = 0; i < nn; i++)
		p[i] = lpUV[i], ip[i] = lpUV[i];

	bool printout = false; FILE *fp1 = 0, *fp2 = 0;
	int piis, pixel_increment_in_subset[] = { 1, 2, 2, 3 };
	double DIC_Coeff_min = 9e9;
	/// Iteration: Begin
	bool Break_Flag = false;
	for (k = 0; k < Iter_Max; k++)
	{
		m = -1;
		t_1 = 0.0, t_2 = 0.0;
		for (iii = 0; iii < nn*nn; iii++)
			AA[iii] = 0.0;
		for (iii = 0; iii < nn; iii++)
			BB[iii] = 0.0;

		a = p[nn - 2], b = p[nn - 1];
		if (printout)
			fp1 = fopen("C:/temp/src.txt", "w+"), fp2 = fopen("C:/temp/tar.txt", "w+");

		piis = pixel_increment_in_subset[Analysis_Speed];	// Depending on algorithms, Analysis_Speed may be changed during the iteration loop.
		for (jjj = -hsubset; jjj <= hsubset; jjj += piis)
		{
			for (iii = -hsubset; iii <= hsubset; iii += piis)
			{
				ii = x_n + iii, jj = y_n + jjj;
				if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1) || lpROI[jj*width1 + ii] == false)
					continue;

				II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1];
				if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length2, width2, height2, II, JJ, S, 0, Interpolation_Algorithm);

					m_F = lpImageData[ii + jj*width1 + kk*length1];
					m_G = S[0], gx = S[1], gy = S[2];
					m++;

					if (DIC_Algo == 0)
					{
						t_3 = a*m_G + b - m_F, t_4 = a;
						t_1 += t_3*t_3, t_2 += m_F*m_F;

						t_5 = t_4*gx, t_6 = t_4*gy;
						if (DIC_Algo == 0)
							CC[0] = t_5, CC[1] = t_6, CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = m_G, CC[5] = 1.0;

						for (j = 0; j < nn; j++)
						{
							BB[j] += t_3*CC[j];
							for (i = 0; i < nn; i++)
								AA[j*nn + i] += CC[i] * CC[j];
						}
					}
					else
					{
						Znssd_reqd[6 * m + 0] = m_F, Znssd_reqd[6 * m + 1] = m_G;
						Znssd_reqd[6 * m + 2] = gx, Znssd_reqd[6 * m + 3] = gy;
						Znssd_reqd[6 * m + 4] = (double)iii, Znssd_reqd[6 * m + 5] = (double)jjj;
						t_1 += m_F, t_2 += m_G;
					}

					if (printout)
						fprintf(fp1, "%.2f ", m_F), fprintf(fp2, "%.2f ", m_G);
				}
			}
			if (printout)
				fprintf(fp1, "\n"), fprintf(fp2, "\n");
		}
		if (printout)
			fclose(fp1), fclose(fp2);

		if (DIC_Algo == 0)
			DIC_Coeff = t_1 / t_2;
		else
		{
			if (k == 0)
			{
				t_f = t_1 / (m + 1);
				t_1 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[6 * iii + 0] - t_f;
					t_1 += t_4*t_4;
				}
				t_ff = sqrt(t_1);
			}
			t_g = t_2 / (m + 1);
			t_2 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_5 = Znssd_reqd[6 * iii + 1] - t_g;
				t_2 += t_5*t_5;
			}
			t_2 = sqrt(t_2);

			DIC_Coeff = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[6 * iii + 0] - t_f;
				t_5 = Znssd_reqd[6 * iii + 1] - t_g;
				t_6 = t_5 / t_2 - t_4 / t_ff;
				t_3 = t_6 / t_2;
				gx = Znssd_reqd[6 * iii + 2], gy = Znssd_reqd[6 * iii + 3];
				iii_n = Znssd_reqd[6 * iii + 4], jjj_n = Znssd_reqd[6 * iii + 5];
				CC[0] = gx, CC[1] = gy;
				CC[2] = gx*iii_n, CC[3] = gx*jjj_n;
				CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
				for (j = 0; j < nn; j++)
				{
					BB[j] += t_3*CC[j];
					for (i = 0; i < nn; i++)
						AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
				}

				DIC_Coeff += t_6*t_6;
			}
		}

		if (!IsNumber(DIC_Coeff))
			return 9e9;
		if (!IsFiniteNumber(DIC_Coeff))
			return 9e9;

		QR_Solution_Double(AA, BB, nn, nn);
		for (iii = 0; iii < nn; iii++)
			p[iii] -= BB[iii];

		if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
		{
			DIC_Coeff_min = DIC_Coeff;
			for (iii = 0; iii < nn; iii++)
				p_best[iii] = p[iii];
			if (!IsNumber(p[0]) || !IsNumber(p[1]))
				return 9e9;
		}

		if (DIC_Algo == 0)
		{
			if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
				return 9e9;
			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn - nExtraParas; iii++)
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				if (iii == nn - nExtraParas)
					if (Analysis_Speed == 1)	// For Analysis_Speed==1, need to run a full "normal speed" analysis
						Analysis_Speed = 0;
					else
						Break_Flag = true;
			}
		}
		else
		{
			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn)
					Break_Flag = true;
			}
		}

		if (Break_Flag)
			break;
	}
	if (k < 1)
		k = 1;
	iteration_check[k - 1]++;
	/// Iteration: End

	if (checkZNCC && DIC_Algo < 4)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, ZNCC, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp2 = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				ii = x_n + iii, jj = y_n + jjj;
				II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1];
				if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					Get_Value_Spline(Para + kk*length2, width2, height2, II, JJ, S, 0, Interpolation_Algorithm);
					if (printout)
						fprintf(fp2, "%.4f ", S[0]);

					Znssd_reqd[2 * m] = lpImageData[ii + jj*width1 + kk*length1], Znssd_reqd[2 * m + 1] = S[0];
					t_f += Znssd_reqd[2 * m], t_g += Znssd_reqd[2 * m + 1];
					m++;
				}
			}
			if (printout)
				fprintf(fp2, "\n");
		}
		if (printout)
			fclose(fp2);

		t_f = t_f / m, t_g = t_g / m;
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = Znssd_reqd[2 * i] - t_f, t_5 = Znssd_reqd[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		ZNCC = t_1 / t_2; //This is the zncc score
		if (abs(ZNCC) < ZNNCThresh)
			DIC_Coeff_min = 1.0;
	}


	if (abs(p_best[0] - ip[0]) > hsubset || abs(p_best[1] - ip[1]) > hsubset || p_best[0] != p_best[0] || p_best[1] != p_best[1])
		return 9e9;
	else
	{
		for (i = 0; i < nn; i++)
			lpUV[i] = p_best[i];
		return DIC_Coeff_min;
	}
}
double DIC_Calculation2(int UV_index_n, int UV_index, double *lpImageData, double *Para, double *lpUV, int *lpUV_xy, double *Znssd_reqd, bool *lpROI, int nchannels, int width1, int height1, int width2, int height2, int UV_length, int DIC_Algo, int hsubset, int step, double PSSDab_thresh, double ZNCCthresh, double ssigThresh, int Iter_Max, int *iteration_check, double conv_crit_1, double conv_crit_2, int Interpolation_Algorithm, int Analysis_Speed, bool firsttime, double *direction, double *FlowU = 0, double *FlowV = 0, bool InitFlow = 0, bool checkZNCC = false)
{
	int i;
	int NN[] = { 3, 7, 4, 8, 6, 12 }, nn = NN[DIC_Algo];

	double shapepara[8];
	for (i = 0; i < nn; i++)
		shapepara[i] = lpUV[i*UV_length + UV_index];

	double ssig = ComputeSSIG(Para, lpUV_xy[2 * UV_index_n], lpUV_xy[2 * UV_index_n + 1], hsubset, width1, height1, nchannels, Interpolation_Algorithm);
	if (ssig < ssigThresh)
	{
		for (i = 0; i < nn; i++)
			*(lpUV + i*UV_length + UV_index_n) = 0.0;
		return 9e9;
	}

	double DIC_Coeff = DIC_Compute(UV_index_n, UV_index, lpImageData, Para + width1*height1*nchannels, shapepara, lpUV_xy, Znssd_reqd, lpROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, Interpolation_Algorithm, Analysis_Speed, firsttime, direction, checkZNCC, ZNCCthresh);
	if (DIC_Coeff < PSSDab_thresh)
	{
		for (i = 0; i < nn; i++)
			lpUV[i*UV_length + UV_index_n] = shapepara[i];
		return DIC_Coeff;
	}
	else if (InitFlow)
	{
		for (i = 0; i < nn - 2; i++)
			shapepara[i] = 0.0;
		shapepara[nn - 2] = 1.0, shapepara[nn - 1] = 0.0;

		int x = lpUV_xy[2 * UV_index_n];
		int y = lpUV_xy[2 * UV_index_n + 1];
		if (DIC_Algo == 0)
			shapepara[0] = 0.5*(FlowU[x + y*width1] / direction[0] + FlowV[x + y*width1] / direction[1]);
		else
		{
			shapepara[0] = FlowU[x + y*width1];
			shapepara[1] = FlowV[x + y*width1];
		}

		DIC_Coeff = DIC_Compute(UV_index_n, UV_index, lpImageData, Para, shapepara, lpUV_xy, Znssd_reqd, lpROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, Interpolation_Algorithm, Analysis_Speed, firsttime, direction, checkZNCC, ZNCCthresh);
		if (DIC_Coeff < PSSDab_thresh)
		{
			for (i = 0; i < nn; i++)
				*(lpUV + i*UV_length + UV_index_n) = shapepara[i];
			return DIC_Coeff;
		}
		else
		{
			for (i = 0; i < nn; i++)
				*(lpUV + i*UV_length + UV_index_n) = 0.0;
			return 9e9;
		}
	}
	else
	{
		for (i = 0; i < nn; i++)
			*(lpUV + i*UV_length + UV_index_n) = 0.0;
		return 9e9;
	}
}
double DIC_Calculation(int UV_index_n, int UV_index, double *lpImageData, double *Para, double *lpUV, int *lpUV_xy, double *Znssd_reqd, bool *lpROI, int nchannels, int width1, int height1, int width2, int height2, int UV_length, int DIC_Algo, int hsubset, int step, double PSSDab_thresh, double ZNCCthresh, double ssigThresh, int Iter_Max, int *iteration_check, double conv_crit_1, double conv_crit_2, int Interpolation_Algorithm, int Analysis_Speed, bool firsttime, double *direction, double *FlowU = 0, double *FlowV = 0, bool InitFlow = 0, bool checkZNCC = false)
{
	int i;
	int NN[] = { 6, 4 }, nn = NN[DIC_Algo];

	double shapepara[6];
	for (i = 0; i < nn; i++)
		shapepara[i] = lpUV[i*UV_length + UV_index];

	double ssig = ComputeSSIG(Para, lpUV_xy[2 * UV_index_n], lpUV_xy[2 * UV_index_n + 1], hsubset, width1, height1, nchannels, Interpolation_Algorithm);
	if (ssig < ssigThresh)
	{
		for (i = 0; i < nn; i++)
			*(lpUV + i*UV_length + UV_index_n) = 0.0;
		return 9e9;
	}

	double DIC_Coeff = DIC_Compute(UV_index_n, UV_index, lpImageData, Para + width1*height1*nchannels, shapepara, lpUV_xy, Znssd_reqd, lpROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, Interpolation_Algorithm, Analysis_Speed, firsttime, direction, checkZNCC, ZNCCthresh);
	if (DIC_Coeff < PSSDab_thresh)
	{
		for (i = 0; i < nn; i++)
			lpUV[i*UV_length + UV_index_n] = shapepara[i];
		return DIC_Coeff;
	}
	else
	{
		for (i = 0; i < nn; i++)
			*(lpUV + i*UV_length + UV_index_n) = 0.0;
		return 9e9;
	}
}

double BruteforceMatchingEpipolar(CPoint2 From, CPoint2 &Target, double *direction, int maxdisparity, double *Img1, double *Img2, double *Img2Para, int nchannels, int width1, int height1, int width2, int height2, LKParameters LKArg, double *tPatch, double *tZNCC, double *Znssd_reqd)//double *Znssd_reqd = new double[6 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
{
	int kk, ll, mm, nn, ii, jj;
	int length1 = width1*height1, length2 = width2*height2;
	int hsubset = LKArg.hsubset, patchS = 2 * hsubset + 1, patchLength = patchS*patchS;

	//Take reference patch
	int x0 = (int)(From.x + 0.5), y0 = (int)(From.y + 0.5);
	for ( ll = 0; ll < nchannels; ll++) 
		for ( mm = -hsubset; mm <= hsubset; mm++)
			for ( nn = -hsubset; nn <= hsubset; nn++)
				tPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength] = Img1[(x0 + nn) + (y0 + mm)*width1 + ll*length1];

	//Search along epipolar line
	double zncc, znccMax = 0.0;
	int bestDistparity = 0;
	for (int ii = -maxdisparity; ii < maxdisparity; ii += 2)
	{
		//Take target patch
		int x1 = (int)(From.x + direction[0] * ii + 0.5), y1 = (int)(From.y + direction[1] * ii + 0.5);
		for (ll = 0; ll < nchannels; ll++)
			for (mm = -hsubset; mm <= hsubset; mm++)
				for (nn = -hsubset; nn <= hsubset; nn++)
					tPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength + nchannels*patchLength] = Img2[(x1 + nn) + (y1 + mm)*width2 + ll*length2];

		zncc = ComputeZNCCPatch(tPatch, tPatch + patchLength*nchannels, hsubset, nchannels, tZNCC);
		if (zncc > znccMax)
			znccMax = zncc, bestDistparity = ii;
	}

	//do refinement
	if (znccMax > LKArg.ZNCCThreshold - 0.25)
	{
		double DIC_Coeff, a, b;
		int i, j, ii, jj, kk, iii, jjj;
		int k, m, nn, nExtraParas;
		int NN[] = { 6, 4 };
		double II, JJ, iii_n, jjj_n;
		double m_F, m_G, t_1, t_2, t_3, t_4, t_5, t_6, t_f, t_ff, t_g;
		double S[9], gx, gy, p[6], ip[6], p_best[6], AA[36], BB[6], CC[6];
		int Analysis_Speed = LKArg.Analysis_Speed;
		double conv_crit_1 = LKArg.Convergence_Criteria, conv_crit_2 = conv_crit_1*0.1;

		int x = x0, y = y0;
		int x_n = (int)(From.x + direction[0] * bestDistparity + 0.5), y_n = (int)(From.y+ direction[1] * bestDistparity + 0.5);

		nn = NN[LKArg.DIC_Algo];
		nExtraParas = 2;

		for (i = 0; i < nn; i++)
			p[i] = 0.0, ip[i] = 0.0;
		if (LKArg.DIC_Algo == 0)
			p[nn - 2] = 1.0, p[nn - 1] = 0.0;

		bool printout = false; FILE *fp1 = 0, *fp2 = 0;
		if (printout)
		{
			fp1 = fopen("C:/temp/src.txt", "w+");
			for (int mm = -hsubset; mm <= hsubset; mm++)
			{
				for (int nn = -hsubset; nn <= hsubset; nn++)
				{
					fprintf(fp1, "%.2f ", tPatch[(mm + hsubset)*patchS + (nn + hsubset)]);
				}
				fprintf(fp1, "%\n");
			}
			fclose(fp1);
		}

		int piis, pixel_increment_in_subset[] = { 1, 2, 2, 3 };
		double DIC_Coeff_min = 9e9;
		/// Iteration: Begin
		bool Break_Flag = false;
		for (k = 0; k < LKArg.IterMax; k++)
		{
			m = -1;
			t_1 = 0.0, t_2 = 0.0;
			for (iii = 0; iii < nn*nn; iii++)
				AA[iii] = 0.0;
			for (iii = 0; iii < nn; iii++)
				BB[iii] = 0.0;

			a = p[nn - 2], b = p[nn - 1];
			if (printout)
				fp2 = fopen("C:/temp/tar.txt", "w+");

			piis = pixel_increment_in_subset[LKArg.Analysis_Speed];	// Depending on algorithms, Analysis_Speed may be changed during the iteration loop.
			for (jjj = -hsubset; jjj <= hsubset; jjj += piis)
			{
				for (iii = -hsubset; iii <= hsubset; iii += piis)
				{
					ii = x_n + iii, jj = y_n + jjj;
					if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1))
						continue;

					II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1];
					if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(Img2Para + kk*length2, width2, height2, II, JJ, S, 0, LKArg.InterpAlgo);

						m_F = tPatch[(jjj + hsubset)*patchS + (iii + hsubset) + kk*patchLength];
						m_G = S[0], gx = S[1], gy = S[2];
						m++;

						if (LKArg.DIC_Algo == 0)
						{
							t_3 = a*m_G + b - m_F, t_4 = a;
							t_1 += t_3*t_3, t_2 += m_F*m_F;

							t_5 = t_4*gx, t_6 = t_4*gy;
							if (LKArg.DIC_Algo == 0)
								CC[0] = t_5, CC[1] = t_6, CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = m_G, CC[5] = 1.0;

							for (j = 0; j < nn; j++)
							{
								BB[j] += t_3*CC[j];
								for (i = 0; i < nn; i++)
									AA[j*nn + i] += CC[i] * CC[j];
							}
						}
						else
						{
							Znssd_reqd[6 * m + 0] = m_F, Znssd_reqd[6 * m + 1] = m_G;
							Znssd_reqd[6 * m + 2] = gx, Znssd_reqd[6 * m + 3] = gy;
							Znssd_reqd[6 * m + 4] = (double)iii, Znssd_reqd[6 * m + 5] = (double)jjj;
							t_1 += m_F, t_2 += m_G;
						}

						if (printout)
							fprintf(fp2, "%.2f ", m_G);
					}
				}
				if (printout)
					fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp2);

			if (LKArg.DIC_Algo == 0)
				DIC_Coeff = t_1 / t_2;
			else
			{
				if (k == 0)
				{
					t_f = t_1 / (m + 1);
					t_1 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = Znssd_reqd[6 * iii + 0] - t_f;
						t_1 += t_4*t_4;
					}
					t_ff = sqrt(t_1);
				}
				t_g = t_2 / (m + 1);
				t_2 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_5 = Znssd_reqd[6 * iii + 1] - t_g;
					t_2 += t_5*t_5;
				}
				t_2 = sqrt(t_2);

				DIC_Coeff = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[6 * iii + 0] - t_f;
					t_5 = Znssd_reqd[6 * iii + 1] - t_g;
					t_6 = t_5 / t_2 - t_4 / t_ff;
					t_3 = t_6 / t_2;
					gx = Znssd_reqd[6 * iii + 2], gy = Znssd_reqd[6 * iii + 3];
					iii_n = Znssd_reqd[6 * iii + 4], jjj_n = Znssd_reqd[6 * iii + 5];
					CC[0] = gx, CC[1] = gy;
					CC[2] = gx*iii_n, CC[3] = gx*jjj_n, CC[4] = gy*iii_n, CC[5] = gy*jjj_n;
					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j] / (t_2*t_2);
					}

					DIC_Coeff += t_6*t_6;
				}
			}

			if (!IsNumber(DIC_Coeff))
				return 9e9;
			if (!IsFiniteNumber(DIC_Coeff))
				return 9e9;

			QR_Solution_Double(AA, BB, nn, nn);
			for (iii = 0; iii < nn; iii++)
				p[iii] -= BB[iii];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (iii = 0; iii < nn; iii++)
					p_best[iii] = p[iii];
				if (!IsNumber(p[0]) || !IsNumber(p[1]))
					return 9e9;
			}

			if (LKArg.DIC_Algo == 0)
			{
				if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
					return 9e9;
				if (fabs(BB[0]) < LKArg.Convergence_Criteria && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn - nExtraParas; iii++)
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					if (iii == nn - nExtraParas)
						if (Analysis_Speed == 1)	// For Analysis_Speed==1, need to run a full "normal speed" analysis
							Analysis_Speed = 0;
						else
							Break_Flag = true;
				}
			}
			else
			{
				if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
				{
					for (iii = 2; iii < nn; iii++)
					{
						if (fabs(BB[iii]) > conv_crit_2)
							break;
					}
					if (iii == nn)
						Break_Flag = true;
				}
			}

			if (Break_Flag)
				break;
		}
		if (k < 1)
			k = 1;
		/// Iteration: End

		if (LKArg.DIC_Algo == 0)
		{
			int m = 0;
			double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
			if (printout)
				fp2 = fopen("C:/temp/tar.txt", "w+");
			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					ii = x_n + iii, jj = y_n + jjj;
					II = ii + p[0] + p[2] * iii + p[3] * jjj, JJ = jj + p[1];
					if (II<0.0 || II>(double)(width2 - 1) - (1e-10) || JJ<0.0 || JJ>(double)(height2 - 1) - (1e-10))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						Get_Value_Spline(Img2Para + kk*length2, width2, height2, II, JJ, S, -1, LKArg.InterpAlgo);
						if (printout)
							fprintf(fp2, "%.4f ", S[0]);

						Znssd_reqd[2 * m] = tPatch[(jjj + hsubset)*patchS + (iii + hsubset) + kk*patchLength], Znssd_reqd[2 * m + 1] = S[0];
						t_f += Znssd_reqd[2 * m], t_g += Znssd_reqd[2 * m + 1];
						m++;
					}
				}
				if (printout)
					fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp2);

			t_f = t_f / m, t_g = t_g / m;
			t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
			for (i = 0; i < m; i++)
			{
				t_4 = Znssd_reqd[2 * i] - t_f, t_5 = Znssd_reqd[2 * i + 1] - t_g;
				t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
			}

			t_2 = sqrt(t_2*t_3);
			if (t_2 < 1e-10)
				t_2 = 1e-10;

			zncc = t_1 / t_2; //This is the zncc score
			if (abs(zncc) < LKArg.ZNCCThreshold)
				DIC_Coeff_min = 1.0;
		}
		else
			zncc = DIC_Coeff_min;

		if (abs(p_best[0] - ip[0]) > hsubset || abs(p_best[1] - ip[1]) > hsubset || p_best[0] != p_best[0] || p_best[1] != p_best[1])
			return 0.0;
		if (zncc < LKArg.ZNCCThreshold - 0.05)
			return zncc;
		else
		{
			Target.x = x_n + p[0], Target.y = y_n + p[1];
			return zncc;
		}
	}

	return znccMax;
}
int GreedyMatching2(char *Img1, char *Img2, CPoint2 *displacement, bool *lpROI_calculated, bool *tROI, CPoint2 *SparseCorres1, CPoint2 *SparseCorres2, int nSeedPoints, LKParameters LKArg, int nchannels, int width1, int height1, int width2, int height2, double Scale, double *Epipole, float *WarpingParas, double *Pmat, double *K, double *distortion, double triThresh)
{
	int ii, kk, cp;
	bool debug = false, passed;
	int length1 = width1*height1, length2 = width2*height2;
	double *lpImageData = new double[nchannels*(length1 + length2)];
	double *lpResult_UV = new double[length1 + length2];
	for (ii = 0; ii < length1 + length2; ii++)
		lpResult_UV[ii] = 0.0;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, step = LKArg.step, Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, InterpAlgo = LKArg.InterpAlgo;
	int Convergence_Criteria = LKArg.Convergence_Criteria, Iter_Max = LKArg.IterMax, Analysis_Speed = LKArg.Analysis_Speed;
	double Gsigma = LKArg.Gsigma, PSSDab_thresh = LKArg.PSSDab_thresh, ZNCCThresh = LKArg.ZNCCThreshold;

	// Prepare image data
	if (Gsigma > 0.0)
		for (kk = 0; kk < nchannels; kk++)
		{
			Gaussian_smooth(Img1 + kk*length1, lpImageData + kk*length1, height1, width1, 255.0, Gsigma);
			Gaussian_smooth(Img2 + kk*length2, lpImageData + kk*length2 + nchannels*length1, height2, width2, 255.0, Gsigma);
		}
	else
	{
		for (kk = 0; kk < nchannels; kk++)
		{
			for (ii = 0; ii < length1; ii++)
				lpImageData[ii + kk*length1] = (double)((int)((unsigned char)(Img1[ii + kk*length1])));
			for (ii = 0; ii < length2; ii++)
				lpImageData[nchannels*length1 + ii] = (double)((int)((unsigned char)(Img2[ii + kk*length2])));
		}
	}

	int m, M, x, y, xx, yy, rr, UV_index, UV_index_n;
	double vr_temp[] = { 0.0, 0.65, 0.8, 0.95 };
	double validity_ratio = vr_temp[Incomplete_Subset_Handling];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	double  ImgPt[3], epipline[3], direction[2];

	if (DIC_Algo == 0)
		conv_crit_1 /= 100.0, conv_crit_2 /= 100.0;
	else if (DIC_Algo == 1)
		conv_crit_1 /= 1000.0;

	int total_valid_points = 0, total_calc_points = 0;
	for (ii = 0; ii < length1; ii++)
		if (tROI[ii]) // 1 - Valid, 0 - Other
			total_valid_points++;
	if (total_valid_points == 0)
	{
		delete[]lpImageData;
		delete[]lpResult_UV;
		return 1;
	}

	int NN[] = { 3, 7, 4, 8, 6, 12 }, nParas = NN[DIC_Algo];
	int UV_length = length1;	// The actual value (i.e., total_calc_points, to be determined later) should be smaller.
	double *lpUV = new double[nParas*UV_length];	// U, V, Ux, Uy, Vx, Vy, Uxx, Uyy, Uxy, Vxx, Vyy, Vxy, (a) and b. or alpha, (a), and b
	int *lpUV_xy = new int[2 * UV_length];	// Coordinates of the points corresponding to lpUV

	double *Znssd_reqd = 0;
	if (DIC_Algo < 4)
		Znssd_reqd = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	else if (DIC_Algo == 4)
		Znssd_reqd = new double[6 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	else if (DIC_Algo == 5)
		Znssd_reqd = new double[12 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	double *Coeff = new double[UV_length];
	int *Tindex = new int[UV_length];

	double *Para = new double[(length1 + length2)*nchannels];
	for (kk = 0; kk < nchannels; kk++)
	{
		Generate_Para_Spline(lpImageData + kk*length1, Para + kk*length1, width1, height1, InterpAlgo);
		Generate_Para_Spline(lpImageData + kk*length2 + nchannels*length1, Para + kk*length2 + nchannels*length1, width2, height2, InterpAlgo);
	}

	int *iteration_check = new int[Iter_Max];
	for (m = 0; m < Iter_Max; m++)
		iteration_check[m] = 0;

	int *PointPerSeed = new int[nSeedPoints];
	for (ii = 0; ii < nSeedPoints; ii++)
		PointPerSeed[ii] = 0;

	double start = omp_get_wtime();
	int percent = 5, increment = 5;

	CPoint startF;
	CPoint2 CorresPoints[2], tCorresPoints[2], ttCorresPoints[2];
	int DeviceMask[2];
	double tK[18], tdistortion[26], tP[24], A[12], B[4];

	UV_index = 0;
	UV_index_n = 0;
	bool firstpoint = true;
	double UV_Guess[8], iWp[4], coeff;
	printf("Expanding from precomputed points...\n");
	for (kk = 0; kk < length1; kk++)
	{
		firstpoint = true;
		cp = 0, M = 0;

		x = kk%width1, y = kk / width1;
		if (lpROI_calculated[y*width1 + x] || !tROI[y*width1 + x])
			continue;
		else
		{
			UV_index = UV_index_n;
			lpUV_xy[2 * UV_index] = x;
			lpUV_xy[2 * UV_index + 1] = y;

			if (!IsLocalWarpAvail(WarpingParas, iWp, x, y, startF, xx, yy, rr, width1, height1, 2))
				continue;

			if (DIC_Algo <= 1)
			{
				ImgPt[0] = lpUV_xy[2 * UV_index], ImgPt[1] = lpUV_xy[2 * UV_index + 1], ImgPt[2] = 1.0;
				cross_product(ImgPt, Epipole, epipline);
				direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
				direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);

				UV_Guess[0] = 0.5*((xx - x) / direction[0] + (yy - y) / direction[1]);
				if (DIC_Algo == 1)
					UV_Guess[1] = iWp[0], UV_Guess[2] = iWp[1], UV_Guess[3] = iWp[2], UV_Guess[4] = iWp[3], UV_Guess[5] = 1.0, UV_Guess[6] = 0.0;
			}
			else
			{
				UV_Guess[0] = xx - x, UV_Guess[1] = yy - y;
				UV_Guess[2] = iWp[0], UV_Guess[3] = iWp[1], UV_Guess[4] = iWp[2], UV_Guess[5] = iWp[3], UV_Guess[6] = 1.0, UV_Guess[7] = 0.0;
			}

			for (m = 0; m < nParas; m++)
				lpUV[m*UV_length + UV_index] = UV_Guess[m];

			coeff = DIC_Calculation(UV_index_n, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
			if (WarpingParas != NULL)
				for (m = 0; m < 6; m++)
					WarpingParas[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width1 + m*UV_length] = (float)lpUV[UV_index + m*UV_length];

			if (coeff < PSSDab_thresh)
			{
				if (distortion != NULL)
				{
					passed = false;
					CorresPoints[0].x = lpUV[UV_index] + x, CorresPoints[0].y = lpUV[UV_index + UV_length] + y, CorresPoints[1].x = x, CorresPoints[1].y = y;
					MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
					if (passed)
						Coeff[M] = coeff, Tindex[M] = UV_index;
					else
						M--;
				}
				else
					Coeff[M] = coeff, Tindex[M] = UV_index;
			}
			else
				M--;
			x = lpUV_xy[2 * UV_index];
			y = lpUV_xy[2 * UV_index + 1];
			lpROI_calculated[y*width1 + x] = true;
			firstpoint = false;
		}

		while (M >= 0)
		{
			if ((100 * (UV_index_n + 1)*step*step / total_valid_points - percent) >= 0)
			{
				cout << total_calc_points + cp << " of good points" << endl;
				double elapsed = omp_get_wtime() - start;
				cout << "%" << 100 * (UV_index_n + 1)*step*step / total_valid_points << " Time elapsed: " << setw(2) << elapsed <<
					" Time remaining: " << setw(2) << elapsed / (percent + increment)*(100.0 - percent) << endl;
				percent += increment;
			}

			UV_index = Tindex[M];
			x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
			M--; // Remove from the queque


			if ((y + step) < height1 && tROI[(y + step)*width1 + x] && !lpROI_calculated[(y + step)*width1 + x])
			{
				if (DIC_CheckPointValidity(tROI, x, y + step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y + step;

					if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
						int a = 0;

					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (distortion != NULL)
						{
							passed = false;
							CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
							CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
							MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 6; m++)
									WarpingParas[x + (y + step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[(y + step)*width1 + x] = true;
			}
			if ((y - step) >= 0 && tROI[(y - step)*width1 + x] && !lpROI_calculated[(y - step)*width1 + x])
			{
				if (DIC_CheckPointValidity(tROI, x, y - step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y - step;

					if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
						int a = 0;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (distortion != NULL)
						{
							passed = false;
							CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
							CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
							MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 6; m++)
									WarpingParas[x + (y - step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[(y - step)*width1 + x] = true;
			}
			if ((x - step) >= 0 && tROI[y*width1 + x - step] && !lpROI_calculated[y*width1 + x - step])
			{
				if (DIC_CheckPointValidity(tROI, x - step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x - step;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y;
					if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
						int a = 0;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (distortion != NULL)
						{
							passed = false;
							CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
							CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
							MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 6; m++)
									WarpingParas[x - step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[y*width1 + x - step] = true;
			}
			if ((x + step) < width1 && tROI[y*width1 + x + step] && !lpROI_calculated[y*width1 + x + step])
			{
				if (DIC_CheckPointValidity(tROI, x + step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x + step;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y;
					if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
						int a = 0;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						if (distortion != NULL)
						{
							passed = false;
							CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
							CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
							MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else
							passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 6; m++)
									WarpingParas[x + step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[y*width1 + x + step] = true;
			}
		}

		if (cp > 0)
			UV_index_n++;
		total_calc_points += cp;
	}
	printf("...%d points growed\n", total_calc_points);

	printf("Expanding from seed points...\n");
	int total_calc_points1 = total_calc_points;
	for (kk = 0; kk < nSeedPoints; kk++)
	{
		firstpoint = true;
		cp = 0, M = 0;

		x = (int)(SparseCorres1[kk].x + 0.5), y = (int)(SparseCorres1[kk].y + 0.5);
		if (lpROI_calculated[y*width1 + x] || !tROI[y*width1 + x])
			continue;
		else
		{
			UV_index = UV_index_n;
			lpUV_xy[2 * UV_index] = (int)(SparseCorres1[kk].x + 0.5);
			lpUV_xy[2 * UV_index + 1] = (int)(SparseCorres1[kk].y + 0.5);

			if (DIC_Algo <= 1)
			{
				ImgPt[0] = lpUV_xy[2 * UV_index], ImgPt[1] = lpUV_xy[2 * UV_index + 1], ImgPt[2] = 1.0;
				cross_product(ImgPt, Epipole, epipline);
				//direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]), direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
				//UV_Guess[0] = 0.5*((SparseCorres2[kk].x - SparseCorres1[kk].x) / direction[0] + (SparseCorres2[kk].y - SparseCorres1[kk].y) / direction[1]);

				direction[0] = 1.0, direction[1] = 0.0; //supreeth
				UV_Guess[0] = (SparseCorres2[kk].x - SparseCorres1[kk].x);
				if (DIC_Algo == 1)
				{
					UV_Guess[1] = Scale - 1.0, UV_Guess[2] = 0.0, UV_Guess[3] = 0.0, UV_Guess[4] = Scale - 1.0;
					UV_Guess[5] = 1.0, UV_Guess[6] = 0.0;
				}
			}
			else
			{
				UV_Guess[0] = SparseCorres2[kk].x - SparseCorres1[kk].x;
				UV_Guess[1] = SparseCorres2[kk].y - SparseCorres1[kk].y;
				UV_Guess[2] = Scale - 1.0, UV_Guess[3] = 0.0, UV_Guess[4] = 0.0, UV_Guess[5] = Scale - 1.0;
				UV_Guess[6] = 1.0, UV_Guess[7] = 0.0;
			}

			for (m = 0; m < nParas; m++)
				lpUV[m*UV_length + UV_index] = UV_Guess[m];

			coeff = DIC_Calculation(UV_index_n, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold + 0.035, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
			if (WarpingParas != NULL)
				for (m = 0; m < 6; m++)
					WarpingParas[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width1 + m*UV_length] = (float)lpUV[UV_index + m*UV_length];

			if (coeff < PSSDab_thresh)
			{
				/*if (distortion != NULL) Supreeth
				{
				passed = false;
				CorresPoints[0].x = lpUV[UV_index] + x, CorresPoints[0].y = lpUV[UV_index + UV_length] + y, CorresPoints[1].x = x, CorresPoints[1].y = y;
				MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
				if (passed)
				Coeff[M] = coeff, Tindex[M] = UV_index;
				else
				M--;
				}
				else*/
				Coeff[M] = coeff, Tindex[M] = UV_index;
			}
			else
				M--;
			x = lpUV_xy[2 * UV_index];
			y = lpUV_xy[2 * UV_index + 1];
			lpROI_calculated[y*width1 + x] = true;
			firstpoint = false;
		}

		while (M >= 0)
		{
			if ((100 * (UV_index_n + 1)*step*step / total_valid_points - percent) >= 0)
			{
				cout << total_calc_points + cp << " of good points" << endl;
				double elapsed = omp_get_wtime() - start;
				cout << "%" << 100 * (UV_index_n + 1)*step*step / total_valid_points << " Time elapsed: " << setw(2) << elapsed <<
					" Time remaining: " << setw(2) << elapsed / (percent + increment)*(100.0 - percent) << endl;
				percent += increment;
			}

			UV_index = Tindex[M];
			x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
			M--; // Remove from the queque


			if ((y + step) < height1 && tROI[(y + step)*width1 + x] && !lpROI_calculated[(y + step)*width1 + x])
			{
				if (DIC_CheckPointValidity(tROI, x, y + step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y + step;

					if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
						int a = 0;

					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						/*if (distortion != NULL) Supreeth
						{
						passed = false;
						CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
						CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
						MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else*/
						passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 6; m++)
									WarpingParas[x + (y + step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[(y + step)*width1 + x] = true;
			}
			if ((y - step) >= 0 && tROI[(y - step)*width1 + x] && !lpROI_calculated[(y - step)*width1 + x])
			{
				if (DIC_CheckPointValidity(tROI, x, y - step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y - step;

					if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
						int a = 0;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						/*if (distortion != NULL) Supreeth
						{
						passed = false;
						CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
						CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
						MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else*/
						passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 6; m++)
									WarpingParas[x + (y - step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[(y - step)*width1 + x] = true;
			}
			if ((x - step) >= 0 && tROI[y*width1 + x - step] && !lpROI_calculated[y*width1 + x - step])
			{
				if (DIC_CheckPointValidity(tROI, x - step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x - step;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y;
					if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
						int a = 0;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						/*if (distortion != NULL) Supreeth
						{
						passed = false;
						CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
						CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
						MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else*/
						passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 6; m++)
									WarpingParas[x - step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[y*width1 + x - step] = true;
			}
			if ((x + step) < width1 && tROI[y*width1 + x + step] && !lpROI_calculated[y*width1 + x + step])
			{
				if (DIC_CheckPointValidity(tROI, x + step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x + step;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y;
					if (lpUV_xy[2 * (UV_index_n + 1)] == 795 && lpUV_xy[2 * (UV_index_n + 1) + 1] == 385)
						int a = 0;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						/*if (distortion != NULL) Supreeth
						{
						passed = false;
						CorresPoints[0].x = lpUV[UV_index_n + 1] + lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[0].y = lpUV[UV_index_n + 1 + UV_length] + lpUV_xy[2 * (UV_index_n + 1) + 1];
						CorresPoints[1].x = lpUV_xy[2 * (UV_index_n + 1)], CorresPoints[1].y = lpUV_xy[2 * (UV_index_n + 1) + 1];
						MultiViewGeoVerify(CorresPoints, Pmat, K, distortion, &passed, width1, height1, width2, height2, 2, 1, triThresh, tCorresPoints, ttCorresPoints, DeviceMask, tK, tdistortion, tP, A, B);
						}
						else*/
						passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 6; m++)
									WarpingParas[x + step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[y*width1 + x + step] = true;
			}
		}

		if (cp > 0)
			UV_index_n++;
		PointPerSeed[kk] = cp;
		total_calc_points += cp;
	}
	printf("...%d points growed\n", total_calc_points - total_calc_points1);
	//// DIC calculation: End

	for (ii = 0; ii < total_calc_points; ii++)
	{
		if (lpUV[ii] != lpUV[ii])
		{
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = 0.0;
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = 0.0;
			continue;
		}
		if (lpUV[UV_length + ii] != lpUV[UV_length + ii])
		{
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = 0.0;
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = 0.0;
			continue;
		}

		if (DIC_Algo <= 1)
		{
			ImgPt[0] = lpUV_xy[2 * ii], ImgPt[1] = lpUV_xy[2 * ii + 1], ImgPt[2] = 1.0;
			cross_product(ImgPt, Epipole, epipline);
			direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
			direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);

			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = lpUV[ii] * direction[0];
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = lpUV[ii] * direction[1];
		}
		else
		{
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = lpUV[ii];
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = lpUV[UV_length + ii];
		}
	}

	delete[]Tindex, delete[]Coeff, delete[]Znssd_reqd;
	delete[]Para, delete[]lpUV_xy, delete[]lpResult_UV, delete[]lpUV;
	delete[]PointPerSeed, delete[]lpImageData, delete[]iteration_check;

	return 0;
}
int GreedyMatching(char *Img1, char *Img2, CPoint2 *displacement, bool *lpROI_calculated, bool *tROI, CPoint2 *SparseCorres1, CPoint2 *SparseCorres2, int nSeedPoints, LKParameters LKArg, int nchannels, int width1, int height1, int width2, int height2, double Scale, double *Epipole, float *WarpingParas, double *Pmat, double *K, double *distortion, double triThresh)
{
	//4 supreeth
	int ii, kk, cp;
	bool debug = false, passed;
	int length1 = width1*height1, length2 = width2*height2;
	double *lpImageData = new double[nchannels*(length1 + length2)];
	double *lpResult_UV = new double[length1 + length2];
	for (ii = 0; ii < length1 + length2; ii++)
		lpResult_UV[ii] = 0.0;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, step = LKArg.step, Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, InterpAlgo = LKArg.InterpAlgo;
	int Convergence_Criteria = LKArg.Convergence_Criteria, Iter_Max = LKArg.IterMax, Analysis_Speed = LKArg.Analysis_Speed;
	double Gsigma = LKArg.Gsigma, PSSDab_thresh = LKArg.PSSDab_thresh, ZNCCThresh = LKArg.ZNCCThreshold;

	// Prepare image data
	if (Gsigma > 0.0)
		for (kk = 0; kk < nchannels; kk++)
		{
			Gaussian_smooth(Img1 + kk*length1, lpImageData + kk*length1, height1, width1, 255.0, Gsigma);
			Gaussian_smooth(Img2 + kk*length2, lpImageData + kk*length2 + nchannels*length1, height2, width2, 255.0, Gsigma);
		}
	else
	{
		for (kk = 0; kk < nchannels; kk++)
		{
			for (ii = 0; ii < length1; ii++)
				lpImageData[ii + kk*length1] = (double)((int)((unsigned char)(Img1[ii + kk*length1])));
			for (ii = 0; ii < length2; ii++)
				lpImageData[nchannels*length1 + ii] = (double)((int)((unsigned char)(Img2[ii + kk*length2])));
		}
	}

	int m, M, x, y, UV_index, UV_index_n;
	double vr_temp[] = { 0.0, 0.65, 0.8, 0.95 };
	double validity_ratio = vr_temp[Incomplete_Subset_Handling];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	double  direction[2];

	if (DIC_Algo == 0)
		conv_crit_1 /= 100.0, conv_crit_2 /= 100.0;
	else if (DIC_Algo == 1)
		conv_crit_1 /= 1000.0;

	int total_valid_points = 0, total_calc_points = 0;
	for (ii = 0; ii < length1; ii++)
		if (tROI[ii]) // 1 - Valid, 0 - Other
			total_valid_points++;
	if (total_valid_points == 0)
	{
		delete[]lpImageData;
		delete[]lpResult_UV;
		return 1;
	}

	int NN[] = { 6, 4 }, nParas = NN[DIC_Algo];
	int UV_length = length1;	// The actual value (i.e., total_calc_points, to be determined later) should be smaller.
	double *lpUV = new double[nParas*UV_length];	// U, V, Ux, Uy, Vx, Vy, Uxx, Uyy, Uxy, Vxx, Vyy, Vxy, (a) and b. or alpha, (a), and b
	int *lpUV_xy = new int[2 * UV_length];	// Coordinates of the points corresponding to lpUV

	double *Znssd_reqd = new double[6 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];

	double *Coeff = new double[UV_length];
	int *Tindex = new int[UV_length];

	double *Para = new double[(length1 + length2)*nchannels];
	for (kk = 0; kk < nchannels; kk++)
	{
		Generate_Para_Spline(lpImageData + kk*length1, Para + kk*length1, width1, height1, InterpAlgo);
		Generate_Para_Spline(lpImageData + kk*length2 + nchannels*length1, Para + kk*length2 + nchannels*length1, width2, height2, InterpAlgo);
	}

	int *iteration_check = new int[Iter_Max];
	for (m = 0; m < Iter_Max; m++)
		iteration_check[m] = 0;

	int *PointPerSeed = new int[nSeedPoints];
	for (ii = 0; ii < nSeedPoints; ii++)
		PointPerSeed[ii] = 0;

	double start = omp_get_wtime();
	int percent = 5, increment = 5;

	CPoint startF;
	CPoint2 CorresPoints[2], tCorresPoints[2], ttCorresPoints[2];

	UV_index = 0;
	UV_index_n = 0;
	bool firstpoint = true;
	double UV_Guess[8], coeff;
	printf("Expanding from seed points...\n");
	for (kk = 0; kk < nSeedPoints; kk++)
	{
		firstpoint = true;
		cp = 0, M = 0;

		x = (int)(SparseCorres1[kk].x + 0.5), y = (int)(SparseCorres1[kk].y + 0.5);
		if (lpROI_calculated[y*width1 + x] || !tROI[y*width1 + x])
			continue;
		else
		{
			UV_index = UV_index_n;
			lpUV_xy[2 * UV_index] = (int)(SparseCorres1[kk].x + 0.5);
			lpUV_xy[2 * UV_index + 1] = (int)(SparseCorres1[kk].y + 0.5);

			UV_Guess[0] = SparseCorres2[kk].x - SparseCorres1[kk].x;
			UV_Guess[1] = SparseCorres2[kk].y - SparseCorres1[kk].y;
			UV_Guess[2] = 0.0, UV_Guess[3] = 0.0;
			UV_Guess[4] = 1.0, UV_Guess[5] = 0.0;

			for (m = 0; m < nParas; m++)
				lpUV[m*UV_length + UV_index] = UV_Guess[m];

			coeff = DIC_Calculation(UV_index_n, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold + 0.035, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
			if (WarpingParas != NULL)
				for (m = 0; m < 2; m++)
					WarpingParas[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width1 + m*UV_length] = (float)lpUV[UV_index + m*UV_length];

			if (coeff < PSSDab_thresh)
				Coeff[M] = coeff, Tindex[M] = UV_index;
			else
				M--;
			x = lpUV_xy[2 * UV_index];
			y = lpUV_xy[2 * UV_index + 1];
			lpROI_calculated[y*width1 + x] = true;
			firstpoint = false;
		}

		while (M >= 0)
		{
			if ((100 * (UV_index_n + 1)*step*step / total_valid_points - percent) >= 0)
			{
				cout << total_calc_points + cp << " of good points" << endl;
				double elapsed = omp_get_wtime() - start;
				cout << "%" << 100 * (UV_index_n + 1)*step*step / total_valid_points << " Time elapsed: " << setw(2) << elapsed <<
					" Time remaining: " << setw(2) << elapsed / (percent + increment)*(100.0 - percent) << endl;
				percent += increment;
			}

			UV_index = Tindex[M];
			x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
			M--; // Remove from the queque

			if ((y + step) < height1 && tROI[(y + step)*width1 + x] && !lpROI_calculated[(y + step)*width1 + x])
			{
				if (DIC_CheckPointValidity(tROI, x, y + step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y + step;

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 2; m++)
									WarpingParas[x + (y + step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[(y + step)*width1 + x] = true;
			}
			if ((y - step) >= 0 && tROI[(y - step)*width1 + x] && !lpROI_calculated[(y - step)*width1 + x])
			{
				if (DIC_CheckPointValidity(tROI, x, y - step, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y - step;

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
							{
								for (m = 0; m < 2; m++)
									WarpingParas[x + (y - step)*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
							}
						}
					}
				}
				lpROI_calculated[(y - step)*width1 + x] = true;
			}
			if ((x - step) >= 0 && tROI[y*width1 + x - step] && !lpROI_calculated[y*width1 + x - step])
			{
				if (DIC_CheckPointValidity(tROI, x - step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x - step;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y;

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
								for (m = 0; m < 2; m++)
									WarpingParas[x - step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[y*width1 + x - step] = true;
			}
			if ((x + step) < width1 && tROI[y*width1 + x + step] && !lpROI_calculated[y*width1 + x + step])
			{
				if (DIC_CheckPointValidity(tROI, x + step, y, width1, height1, hsubset, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x + step;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y;

					coeff = DIC_Calculation(UV_index_n + 1, UV_index, lpImageData, Para, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, DIC_Algo, hsubset, step, PSSDab_thresh, LKArg.ZNCCThreshold, LKArg.ssigThresh, Iter_Max, iteration_check, conv_crit_1, conv_crit_2, InterpAlgo, Analysis_Speed, firstpoint, direction, NULL, NULL, false, LKArg.checkZNCC);
					if (coeff < PSSDab_thresh)
					{
						passed = true;
						if (passed)
						{
							cp++, M++, UV_index_n++;
							Coeff[M] = coeff;
							Tindex[M] = UV_index_n;
							DIC_AddtoQueue(Coeff, Tindex, M);
							if (WarpingParas != NULL)
								for (m = 0; m < 2; m++)
									WarpingParas[x + step + y*width1 + m*UV_length] = (float)lpUV[UV_index_n + m*UV_length];
						}
					}
				}
				lpROI_calculated[y*width1 + x + step] = true;
			}
		}

		if (cp > 0)
			UV_index_n++;
		PointPerSeed[kk] = cp;
		total_calc_points += cp;
	}
	printf("...%d points growed\n", total_calc_points);
	//// DIC calculation: End

	for (ii = 0; ii < total_calc_points; ii++)
	{
		if (lpUV[ii] != lpUV[ii])
		{
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = 0.0;
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = 0.0;
			continue;
		}
		if (lpUV[UV_length + ii] != lpUV[UV_length + ii])
		{
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = 0.0;
			displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = 0.0;
			continue;
		}

		displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].x = lpUV[ii];
		displacement[lpUV_xy[2 * ii] + lpUV_xy[2 * ii + 1] * width1].y = lpUV[UV_length + ii];
	}

	delete[]Tindex, delete[]Coeff, delete[]Znssd_reqd;
	delete[]Para, delete[]lpUV_xy, delete[]lpResult_UV, delete[]lpUV;
	delete[]PointPerSeed, delete[]lpImageData, delete[]iteration_check;

	return 0;
}
int MatchingCheck(char *Img1, char *Img2, float *WarpingParas, LKParameters LKArg, double scale, int nchannels, int width1, int height1, int width2, int height2)
{
#pragma omp critical
	printf("Matching verification process...\n");

	int ii, jj, kk;
	int length1 = width1*height1, length2 = width2*height2;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo - 2, step = LKArg.step, Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, InterpAlgo = LKArg.InterpAlgo;
	int Convergence_Criteria = LKArg.Convergence_Criteria, Iter_Max = LKArg.IterMax, Analysis_Speed = LKArg.Analysis_Speed;
	double Gsigma = LKArg.Gsigma, PSSDab_thresh = LKArg.PSSDab_thresh, ZNCCThresh = LKArg.ZNCCThreshold;

	// Prepare image data
	double *dImg1 = new double[length1*nchannels];
	double *dImg2 = new double[length2*nchannels];
	if (Gsigma > 0.0)
	{
		for (kk = 0; kk < nchannels; kk++)
		{
			Gaussian_smooth(Img1 + kk*length1, dImg1 + kk*length1, height1, width1, 255.0, Gsigma);
			Gaussian_smooth(Img2 + kk*length2, dImg2 + kk*length2, height2, width2, 255.0, Gsigma);
		}
	}
	else
	{
		for (kk = 0; kk < nchannels; kk++)
		{
			for (ii = 0; ii < length1; ii++)
				dImg1[ii + kk*length1] = (double)((int)((unsigned char)(Img1[ii + kk*length1])));
			for (ii = 0; ii < length2; ii++)
				dImg2[ii + kk*length2] = (double)((int)((unsigned char)(Img2[ii + kk*length2])));
		}
	}

	double *Timg = new double[(2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	double *T = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];

	double *Para1 = new double[length1*nchannels];
	double *Para2 = new double[length2*nchannels];
	for (kk = 0; kk < nchannels; kk++)
	{
		Generate_Para_Spline(dImg1 + kk*length1, Para1 + kk*length1, width1, height1, InterpAlgo);
		Generate_Para_Spline(dImg2 + kk*length2, Para2 + kk*length2, width2, height2, InterpAlgo);
	}

	int totalPoints = 0, verified = 0;
	for (jj = 0; jj < height1; jj++)
		for (ii = 0; ii < width1; ii++)
			if (abs(WarpingParas[ii + jj*width1]) + abs(WarpingParas[ii + jj*width1 + length1]) > 0.01)
				totalPoints++;

	double start = omp_get_wtime();
	int percent = 20, increment = 20;

	double fufv[2], ShapePara[9] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
	CPoint2 From, To;
	for (jj = 0; jj < height1; jj++)
	{
		for (ii = 0; ii < width1; ii++)
		{
			if (abs(WarpingParas[ii + jj*width1]) + abs(WarpingParas[ii + jj*width1 + length1]) > 0.01)
			{
				verified++;
#pragma omp critical
				if ((100 * verified / totalPoints - percent) > 0)
				{
					double elapsed = omp_get_wtime() - start;
					cout << "%" << 100 * verified / totalPoints << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / (percent + increment)*(100.0 - percent) << endl;
					percent += increment;
				}

				From.x = ii, From.y = jj;
				To.x = WarpingParas[ii + jj*width1] + ii, To.y = WarpingParas[ii + jj*width1 + length1] + jj;
				ShapePara[0] = scale, ShapePara[4] = scale, ShapePara[2] = From.x, ShapePara[5] = From.y;
				if (TMatching(Para2, Para1, hsubset, width2, height2, width1, height1, nchannels, To, From, DIC_Algo, Convergence_Criteria, ZNCCThresh, Iter_Max, InterpAlgo, fufv, true, ShapePara, NULL, Timg, T) > ZNCCThresh)
				{
					if (abs(fufv[0]) + abs(fufv[1]) > 2)
						WarpingParas[ii + jj*width1] = 0.0, WarpingParas[ii + jj*width1 + length1] = 0.0;
				}
				else
					WarpingParas[ii + jj*width1] = 0.0, WarpingParas[ii + jj*width1 + length1] = 0.0;
			}
		}
	}

	delete[]dImg1, delete[]dImg2;
	delete[]Timg, delete[]T;
	delete[]Para1, delete[]Para2;

	return 0;
}
int MatchingCheck2(float *WarpingParas1, float *WarpingParas2, int width1, int height1, int width2, int height2, bool AffineShape)
{
	int ii, jj, kk;
	int length1 = width1*height1, length2 = width2*height2;

	CPoint2 From, To, Back;
	for (jj = 0; jj < height1; jj++)
	{
		for (ii = 0; ii < width1; ii++)
		{
			if (abs(WarpingParas1[ii + jj*width1]) + abs(WarpingParas1[ii + jj*width1 + length1]) > 0.01)
			{
				From.x = ii, From.y = jj;
				To.x = WarpingParas1[ii + jj*width1] + ii, To.y = WarpingParas1[ii + jj*width1 + length1] + jj;
				Back.x = To.x + WarpingParas2[(int)(To.x + 0.5) + (int)(To.y + 0.5)*width2], Back.x = To.y + WarpingParas2[(int)(To.x + 0.5) + (int)(To.y + 0.5)*width2 + length2];

				if (abs(From.x - Back.x) + abs(From.y - Back.y) > 1)
				{
					if (AffineShape)
						for (kk = 0; kk < 6; kk++)
							WarpingParas1[ii + jj*width1 + kk*length1] = 0.0, WarpingParas2[(int)(To.x + 0.5) + (int)(To.y + 0.5)*width2 + kk*length2] = 0.0;
					else
					{
						WarpingParas1[ii + jj*width1] = 0.0, WarpingParas1[ii + jj*width1 + length1] = 0.0;
						WarpingParas2[(int)(To.x + 0.5) + (int)(To.y + 0.5)*width2] = 0.0, WarpingParas2[(int)(To.x + 0.5) + (int)(To.y + 0.5)*width2 + length2] = 0.0;
					}
				}
			}
		}
	}

	return 0;
}
void SDIC_AddtoQueue(float *Coeff, int *Tindex, int M)
{
	int i, j, t;
	float coeff;
	for (i = 0; i <= M - 1; i++)
	{
		if (*(Coeff + M) > *(Coeff + i))
		{
			coeff = *(Coeff + M);
			t = *(Tindex + M);
			for (j = M - 1; j >= i; j--)
			{
				*(Coeff + j + 1) = *(Coeff + j);
				*(Tindex + j + 1) = *(Tindex + j);
			}
			*(Coeff + i) = coeff;
			*(Tindex + i) = t;
			break;
		}
	}

	return;
}
bool SDIC_CheckPointValidity(bool *lpROI, double x_n, double y_n, int width, int height, int hsubset, double SR, double validity_ratio)
{


	int XN = (int)(x_n / SR), YN = (int)(y_n / SR), swidth = (int)(width / SR), sheight = (int)(height / SR);
	hsubset = (int)(hsubset / SR);

	int jump = (int)(2.0 / SR);
	int m = 0, n = 0, ii, jj, iii, jjj;
	for (jjj = -hsubset; jjj <= hsubset; jjj += jump)
	{
		for (iii = -hsubset; iii <= hsubset; iii += jump)
		{
			m++;
			jj = XN + jjj;
			ii = YN + iii;

			if (ii<0 || ii>(swidth - 1) || jj<0 || jj>(sheight - 1))
				continue;

			if (lpROI[jj*swidth + ii] == false)
				continue;

			n++;
		}
	}

	if (n < int(m*validity_ratio))
		return false;

	return true;
}
void SDIC_Initial_Guess(double *lpImageData, int width, int height, double *UV_Guess, CPoint Start_Point, int *IG_subset, int Initial_Guess_Scheme)
{
	int i;

	for (i = 0; i < 14; i++)
		UV_Guess[i] = 0.0;

	if (Initial_Guess_Scheme == 1) //Epipolar line
	{

	}
	else //Just correlation
	{
		int j, k, m, n, ii, jj, II_0, JJ_0, iii, jjj;
		double ratio = 0.2;
		double t_f, t_g, t_1, t_2, t_3, t_4, t_5, m_F, m_G, C_zncc, C_znssd_min, C_znssd_max;
		int hsubset, m_IG_subset[2];
		int length = width*height;

		m_IG_subset[0] = IG_subset[0] / 2;
		m_IG_subset[1] = IG_subset[1] / 2;

		//CProgressDlg pdlg;
		//pdlg.show_info_in_title="Initial Guess ...";
		//pdlg.Create();

		double *C_znssd = new double[length];
		char *TT = new char[length];
		double *T = new double[2 * (2 * m_IG_subset[1] + 1)*(2 * m_IG_subset[1] + 1)];

		C_znssd_min = 1e12;
		C_znssd_max = -1e12;

		for (n = 0; n < length; n++)
		{
			*(TT + n) = (char)0;
			*(C_znssd + n) = 1e2;
		}

		for (k = 0; k < 2; k++)
		{
			hsubset = m_IG_subset[k];
			for (j = hsubset; j < height - hsubset; j++)
			{
				//pdlg.SetPos(100*j/(height-hsubset-1));

				for (i = hsubset; i < width - hsubset; i++)
				{
					if (*(TT + j*width + i) == (char)1)
						continue;

					m = -1;
					t_f = 0.0;
					t_g = 0.0;
					for (jjj = -hsubset; jjj <= hsubset; jjj++)
					{
						for (iii = -hsubset; iii <= hsubset; iii++)
						{
							jj = Start_Point.y + jjj;
							ii = Start_Point.x + iii;
							JJ_0 = j + jjj;
							II_0 = i + iii;

							m_F = *(lpImageData + jj*width + ii);
							m_G = *(lpImageData + length + JJ_0*width + II_0);

							m++;
							*(T + 2 * m + 0) = m_F;
							*(T + 2 * m + 1) = m_G;
							t_f += m_F;
							t_g += m_G;
						}
					}

					t_f = t_f / (m + 1);
					t_g = t_g / (m + 1);
					t_1 = 0.0;
					t_2 = 0.0;
					t_3 = 0.0;
					for (iii = 0; iii <= m; iii++)
					{
						t_4 = *(T + 2 * iii + 0) - t_f;
						t_5 = *(T + 2 * iii + 1) - t_g;
						t_1 += (t_4*t_5);
						t_2 += (t_4*t_4);
						t_3 += (t_5*t_5);
					}
					t_2 = sqrt(t_2*t_3);
					if (t_2 < 1e-10)		// Avoid being divided by 0.
						t_2 = 1e-10;

					C_zncc = t_1 / t_2;

					// Testing shows that C_zncc may not fall into (-1, 1) range, so need the following line.
					if (C_zncc > 1.0 || C_zncc < -1.0)
						C_zncc = 0.0;	// Use 0.0 instead of 1.0 or -1.0

					*(C_znssd + j*width + i) = 2.0*(1.0 - C_zncc);

					if (*(C_znssd + j*width + i) < C_znssd_min)
					{
						C_znssd_min = *(C_znssd + j*width + i);
						UV_Guess[0] = i - Start_Point.x;
						UV_Guess[1] = j - Start_Point.y;
					}

					if (*(C_znssd + j*width + i) > C_znssd_max)
						C_znssd_max = *(C_znssd + j*width + i);	// C_znssd_max should be close to 4.0, C_znssd_min should be close to 0.0
				}
			}

			if (k == 0)
			{
				for (n = 0; n<length; n++)
				{
					if (*(C_znssd + n) >(C_znssd_min + ratio*(C_znssd_max - C_znssd_min)))
						*(TT + n) = (char)1;
				}

				C_znssd_min = 1e12;
				C_znssd_max = -1e12;
			}
		}

		delete[]T;
		delete[]TT;
		delete[]C_znssd;
	}

	return;
}
void SDIC_Initial_Guess_Refine(double x_n, double y_n, double *Img1, double *Img2, double *Znssd_reqd, bool *lpROI, double *p, int nchannels, int width1, int height1, int width2, int height2, int hsubset, double SR, int DIC_Algo, double *direction)
{

	/// Refine initial guess of u and v of the starting point with integral-pixel accuracy using ZNSSD
	// Img1 & Img2 are in pixel format

	int d_u, d_v, u0, v0, U0, V0, alpha, alpha0;
	int m, ii, jj, kk, iii, jjj, II_0, JJ_0;
	int length1 = width1*height1, length2 = width2*height2, swidth1 = (int)(width1 / SR);
	double t_1, t_2, t_3, t_4, t_5, t_F, t_G, mean_F, mean_G;
	double C_zncc, C_znssd, C_znssd_min;
	C_znssd_min = 1.0E12;

	int XN1 = (int)(x_n + 0.5), YN1 = (int)(y_n + 0.5);
	int XN2 = (int)(p[0] + 0.5), YN2 = (int)(p[1] + 0.5);

	bool printout = false;
	FILE *fp1 = 0, *fp2 = 0;

	if (DIC_Algo <= 1) //Epipoloar constraint on the flow
	{
		alpha = 0;
		for (alpha0 = -3; alpha0 <= 3; alpha0++)
		{
			u0 = (int)(direction[0] * (XN2 + 0.5*alpha0) + 0.5);
			v0 = (int)(direction[1] * (YN2 + 0.5*alpha0) + 0.5);

			m = -1;
			mean_F = 0.0;
			mean_G = 0.0;
			if (printout)
			{
				fp1 = fopen("C:/temp/src.txt", "w+");
				fp2 = fopen("C:/temp/tar.txt", "w+");
			}
			for (jjj = -hsubset; jjj <= hsubset; jjj++)
			{
				for (iii = -hsubset; iii <= hsubset; iii++)
				{
					ii = XN1 + iii, jj = YN1 + jjj;

					if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1))
						continue;

					if (lpROI[(int)(ii / SR) + (int)(jj / SR)*swidth1] == false)
						continue;

					II_0 = ii + u0 + (int)(p[1] * iii + p[2] * jjj + 0.5);
					JJ_0 = jj + v0 + (int)(p[3] * iii + p[4] * jjj + 0.5);

					if (II_0<0 || II_0>(width2 - 1) || JJ_0<0 || JJ_0>(height2 - 1))
						continue;

					for (kk = 0; kk < nchannels; kk++)
					{
						t_F = Img1[jj*width1 + ii + kk*length1];
						t_G = Img2[JJ_0*width2 + II_0 + kk*length2];

						if (printout)
						{
							fprintf(fp1, "%.2f ", t_F), fprintf(fp2, "%.2f ", t_G);
						}
						m++;
						Znssd_reqd[2 * m] = t_F;
						Znssd_reqd[2 * m + 1] = t_G;
						mean_F += t_F;
						mean_G += t_G;
					}
				}
				if (printout)
				{
					fprintf(fp1, "\n"), fprintf(fp2, "\n");
				}
			}
			if (printout)
			{
				fclose(fp1); fclose(fp2);
			}
			if (m < 10)
				continue;

			mean_F /= (m + 1);
			mean_G /= (m + 1);
			t_1 = 0.0;
			t_2 = 0.0;
			t_3 = 0.0;
			for (iii = 0; iii <= m; iii++)
			{
				t_4 = Znssd_reqd[2 * iii] - mean_F;
				t_5 = Znssd_reqd[2 * iii + 1] - mean_G;
				t_1 += (t_4*t_5);
				t_2 += (t_4*t_4);
				t_3 += (t_5*t_5);
			}

			C_zncc = t_1 / sqrt(t_2*t_3);
			C_znssd = 2.0*(1.0 - C_zncc);

			if (C_znssd < C_znssd_min)
			{
				C_znssd_min = C_znssd;
				alpha = alpha0;
			}
		}

		p[0] = p[0] + 0.5*alpha;
	}
	else //Affine shape 
	{
		U0 = 0, V0 = 0;
		for (d_u = -1; d_u <= 1; d_u++)
		{
			for (d_v = -1; d_v <= 1; d_v++)
			{
				u0 = d_u + XN2, v0 = d_v + YN2; //pixel format

				m = -1;
				mean_F = 0.0;
				mean_G = 0.0;
				if (printout)
				{
					fp1 = fopen("C:/temp/src.txt", "w+");
					fp2 = fopen("C:/temp/tar.txt", "w+");
				}
				for (jjj = -hsubset; jjj <= hsubset; jjj++)
				{
					for (iii = -hsubset; iii <= hsubset; iii++)
					{
						ii = XN1 + iii, jj = YN1 + jjj;	//pixel format				

						if (ii<0 || ii>(width1 - 1) || jj<0 || jj>(height1 - 1))
							continue;

						int uii = (int)(ii / SR), ujj = (int)(jj / SR);
						if (lpROI[uii + ujj*swidth1] == false) //ROI is in supersample format
							continue;

						II_0 = ii + u0 + (int)(p[2] * iii + p[3] * jjj);  //pixel format
						JJ_0 = jj + v0 + (int)(p[4] * iii + p[5] * jjj);

						if (II_0<0 || II_0>(width2 - 1) || JJ_0<0 || JJ_0>(height2 - 1))
							continue;

						for (kk = 0; kk < nchannels; kk++)
						{
							t_F = Img1[jj*width1 + ii + kk*length1]; // Img1 & Img2 are in pixel format
							t_G = Img2[JJ_0*width2 + II_0 + kk*length2];

							if (printout && kk == 0)
							{
								fprintf(fp1, "%.2f ", t_F), fprintf(fp2, "%.2f ", t_G);
							}
							m++;
							Znssd_reqd[6 * m] = t_F;
							Znssd_reqd[6 * m + 1] = t_G;
							mean_F += t_F;
							mean_G += t_G;
						}
					}
					if (printout)
					{
						fprintf(fp1, "\n"), fprintf(fp2, "\n");
					}
				}
				if (printout)
				{
					fclose(fp1); fclose(fp2);
				}
				if (m < 10)
					continue;

				mean_F /= (m + 1);
				mean_G /= (m + 1);
				t_1 = 0.0;
				t_2 = 0.0;
				t_3 = 0.0;
				for (iii = 0; iii <= m; iii++)
				{
					t_4 = Znssd_reqd[6 * iii] - mean_F;
					t_5 = Znssd_reqd[6 * iii + 1] - mean_G;
					t_1 += (t_4*t_5);
					t_2 += (t_4*t_4);
					t_3 += (t_5*t_5);
				}

				C_zncc = t_1 / sqrt(t_2*t_3);
				C_znssd = 2.0*(1.0 - C_zncc);

				if (C_znssd < C_znssd_min)
				{
					C_znssd_min = C_znssd;
					U0 = u0;
					V0 = v0;
				}
			}
		}

		p[0] = U0;
		p[1] = V0;
	};

	return;
}
double SDIC_Compute(int UV_index_n, int UV_index, double *Img1, double *Img2, double *Para1, double *Para2, double *lpUV, float *lpUV_xy, double *Znssd_reqd, bool *lpROI, int nchannels, int width1, int height1, int width2, int height2, int UV_length, int *flowhsubset, LKParameters LKArg, double SR, int *iteration_check, bool firsttime, double *direction)
{
	int hsubset, DIC_Algo = LKArg.DIC_Algo, step = LKArg.step, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Convergence_Criteria = LKArg.Convergence_Criteria, Iter_Max = LKArg.IterMax, Analysis_Speed = LKArg.Analysis_Speed;
	double PSSDab_thresh = LKArg.PSSDab_thresh, ZNCCThresh = LKArg.ZNCCThreshold;
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;

	double DIC_Coeff, a, b;
	int i, j, kk, iii, jjj;
	int k, m, nn, nExtraParas;
	int length1 = width1*height1, length2 = width2*height2, swidth1 = (int)(width1 / SR);
	int NN[] = { 3, 7, 4, 8 };
	double m_F, m_G, t_1, t_2, t_3, t_4, t_5, t_6;
	double S[9], p[8], ip[8], p_best[8];// U, V, Ux, Uy, Vx, Vy, (a) and b.
	double AA[64], BB[8], CC[8], gx, gy;

	double u1, v1, u2, v2;
	double x = (double)lpUV_xy[2 * UV_index], y = (double)lpUV_xy[2 * UV_index + 1], x_n = (double)lpUV_xy[2 * UV_index_n], y_n = (double)lpUV_xy[2 * UV_index_n + 1];

	hsubset = flowhsubset[(int)(x_n + 0.5) + (int)(y_n + 0.5)*width1]; //make used of precomputed scale for flow
	nn = NN[DIC_Algo];
	nExtraParas = 2;

	for (i = 0; i < nn; i++)
	{
		p[i] = lpUV[i];
		ip[i] = lpUV[i];
	}

	// The following two lines are needed for large rotation cases.
	if (DIC_Algo == 1)
		p[0] = 0.5*((p[0] * direction[0] + (p[1] * (x_n - x) + p[2] * (y_n - y))) / direction[0] + (p[0] * direction[1] + (p[3] * (x_n - x) + p[4] * (y_n - y))) / direction[1]);
	else if (DIC_Algo == 3)
	{
		p[0] += (p[2] * (x_n - x) + p[3] * (y_n - y));
		p[1] += (p[4] * (x_n - x) + p[5] * (y_n - y));
	}

	/*for(i=0; i<nn; i++)
	lpUV[i] = 0.0;
	return abs(gaussian_noise(0.0, 0.001));*/

	// Refine initial guess of u and v of the starting point with integral-pixel accuracy using ZNSSD
	//f(firsttime)
	//	SDIC_Initial_Guess_Refine(x_n, y_n, Img1, Img2, Znssd_reqd, lpROI, p, nchannels, width1, height1, width2, height2, hsubset, SR, DIC_Algo, direction);

	int TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	double *Timg = new double[nchannels*Tlength];
	for (jjj = -hsubset; jjj <= hsubset; jjj++)
	{
		for (iii = -hsubset; iii <= hsubset; iii++)
		{
			u1 = x_n + iii, v1 = y_n + jjj;
			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(Para1 + kk*length1, width1, height1, u1, v1, S, -1, Interpolation_Algorithm);
				Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength] = S[0];
			}
		}
	}

	bool printout = false; FILE *fp1 = 0, *fp2 = 0;

	int piis, pixel_increment_in_subset[] = { 1, 2, 3 };
	double DIC_Coeff_min = 9e9;
	/// Iteration: Begin
	bool Break_Flag = false;
	for (k = 0; k < Iter_Max; k++)
	{
		m = -1;
		t_1 = 0.0;
		t_2 = 0.0;
		for (iii = 0; iii < nn*nn; iii++)
			AA[iii] = 0.0;
		for (iii = 0; iii < nn; iii++)
			BB[iii] = 0.0;

		a = p[nn - 2], b = p[nn - 1];

		if (printout)
		{
			fp1 = fopen("C:/temp/src.txt", "w+");
			fp2 = fopen("C:/temp/tar.txt", "w+");
		}

		piis = pixel_increment_in_subset[Analysis_Speed];	// Depending on algorithms, Analysis_Speed may be changed during the iteration loop.
		for (jjj = -hsubset; jjj <= hsubset; jjj += piis)
		{
			for (iii = -hsubset; iii <= hsubset; iii += piis)
			{
				u1 = x_n + iii, v1 = y_n + jjj;

				if (u1<0 || u1>(width1 - 1) || v1<0 || v1>(height1 - 1))
					continue;

				if (lpROI[(int)(u1 / SR) + (int)(v1 / SR)*swidth1] == false)
					continue;

				if (DIC_Algo == 0)
				{
					u2 = u1 + p[0] * direction[0];
					v2 = v1 + p[0] * direction[1];
				}
				else if (DIC_Algo == 1)
				{
					u2 = u1 + p[0] * direction[0] + p[1] * iii + p[2] * jjj;
					v2 = v1 + p[0] * direction[1] + p[3] * iii + p[4] * jjj;
				}
				if (DIC_Algo == 2)
				{
					u2 = u1 + p[0];
					v2 = v1 + p[1];
				}
				else if (DIC_Algo == 3)
				{
					u2 = u1 + p[0] + p[2] * iii + p[3] * jjj;
					v2 = v1 + p[1] + p[4] * iii + p[5] * jjj;
				}

				if (u2<0.0 || u2>(double)(width2 - 1) - (1e-10) || v2<0.0 || v2>(double)(height2 - 1) - (1e-10))
					continue;

				for (kk = 0; kk < nchannels; kk++)
				{
					//Get_Value_Spline(Para1+kk*length1, width1, height1, u1, v1, S, -1, Interpolation_Algorithm); m_F = S[0];
					m_F = Timg[(iii + hsubset) + (jjj + hsubset)*TimgS + kk*Tlength];

					Get_Value_Spline(Para2 + kk*length2, width2, height2, u2, v2, S, 0, Interpolation_Algorithm);
					m_G = S[0], gx = S[1], gy = S[2];

					if (printout && kk == 0)
					{
						fprintf(fp1, "%.2f ", m_F);
						fprintf(fp2, "%.2f ", m_G);
					}
					m++;

					t_3 = a*m_G + b - m_F;
					t_4 = a;

					t_5 = t_4*gx, t_6 = t_4*gy;
					if (DIC_Algo == 0)
					{
						CC[0] = t_5*direction[0] + t_6*direction[1];
						CC[1] = m_G;
						CC[2] = 1.0;
					}
					else if (DIC_Algo == 1)
					{
						CC[0] = t_5*direction[0] + t_6*direction[1];
						CC[1] = t_5*iii, CC[2] = t_5*jjj, CC[3] = t_6*iii, CC[4] = t_6*jjj;
						CC[5] = m_G, CC[6] = 1.0;
					}
					else if (DIC_Algo == 2)
					{
						CC[0] = t_5, CC[1] = t_6;
						CC[2] = m_G, CC[3] = 1.0;
					}
					else if (DIC_Algo == 3)
					{
						CC[0] = t_5, CC[1] = t_6;
						CC[2] = t_5*iii, CC[3] = t_5*jjj, CC[4] = t_6*iii, CC[5] = t_6*jjj;
						CC[6] = m_G, CC[7] = 1.0;
					}

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += m_F*m_F;
				}
			}
			if (printout)
			{
				fprintf(fp1, "\n");
				fprintf(fp2, "\n");
			}
		}
		if (printout)
		{
			fclose(fp1); fclose(fp2);
		}
		DIC_Coeff = t_1 / t_2;

		if (!IsNumber(DIC_Coeff))
		{
			delete[]Timg;
			return 9e9;
		}
		if (!IsFiniteNumber(DIC_Coeff))
		{
			delete[]Timg;
			return 9e9;
		}

		QR_Solution_Double(AA, BB, nn, nn);

		for (iii = 0; iii < nn; iii++)
			p[iii] -= BB[iii];

		if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
		{
			DIC_Coeff_min = DIC_Coeff;
			for (iii = 0; iii < nn; iii++)
				p_best[iii] = p[iii];
			if (!IsNumber(p[0]) || !IsNumber(p[1]))
			{
				delete[]Timg;
				return 9e9;
			}
		}

		if (DIC_Algo <= 1)
		{
			if (abs((p[0] - ip[0])*direction[0]) > hsubset)
			{
				delete[]Timg;
				return 9e9;
			}

			if (fabs(BB[0]) < conv_crit_1)
			{
				for (iii = 1; iii < nn - nExtraParas; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn - nExtraParas)
				{
					if (Analysis_Speed == 1)	// For Analysis_Speed==1, need to run a full "normal speed" analysis
						Analysis_Speed = 0;
					else
						Break_Flag = true;
				}
			}
		}
		else
		{
			if (abs(p[0] - ip[0]) > hsubset || abs(p[1] - ip[1]) > hsubset)
			{
				delete[]Timg;
				return 9e9;
			}

			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[1]) < conv_crit_1)
			{
				for (iii = 2; iii < nn - nExtraParas; iii++)
				{
					if (fabs(BB[iii]) > conv_crit_2)
						break;
				}
				if (iii == nn - nExtraParas)
				{
					if (Analysis_Speed == 1)	// For Analysis_Speed==1, need to run a full "normal speed" analysis
						Analysis_Speed = 0;
					else
						Break_Flag = true;
				}
			}
		}

		if (Break_Flag)
			break;
	}
	if (k < 1)
		k = 1;
	iteration_check[k - 1]++;
	/// Iteration: End

	delete[]Timg;
	if (DIC_Algo <= 1)
	{
		if (abs((p[0] - ip[0])*direction[0]) > hsubset)
			return 9e9;
		else
		{
			for (i = 0; i < nn; i++)
				lpUV[i] = p_best[i];
			return DIC_Coeff_min;
		}
	}
	else
	{
		if (abs(p_best[0] - ip[0]) > hsubset || abs(p_best[1] - ip[1]) > hsubset || p_best[0] != p_best[0] || p_best[1] != p_best[1])
			return 9e9;
		else
		{
			for (i = 0; i < nn; i++)
				lpUV[i] = p_best[i];
			return DIC_Coeff_min;
		}
	}
}
double SDIC_Calculation(int UV_index_n, int UV_index, double *Img1, double *Img2, double *Para1, double *Para2, float *lpUV, float *lpUV_xy, double *Znssd_reqd, bool *lpROI, int nchannels, int width1, int height1, int width2, int height2, int UV_length, int *flowhsubset, LKParameters LKArg, double SR, int *iteration_check, bool firsttime, double *direction, double *FlowU = 0, double *FlowV = 0, bool InitFlow = 0)
{
	int DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	double PSSDab_thresh = LKArg.PSSDab_thresh;

	int i;
	int NN[] = { 3, 7, 4, 8 }, nn = NN[DIC_Algo];

	double shapepara[8];
	for (i = 0; i < nn; i++)
		shapepara[i] = lpUV[i*UV_length + UV_index];

	int x = (int)lpUV_xy[2 * UV_index_n], y = (int)lpUV_xy[2 * UV_index_n + 1];
	double mag = ComputeSSIG(Para1, x, y, 5 * SR, width1, height1, nchannels, Interpolation_Algorithm);
	if (mag < 0.1)
	{
		for (i = 0; i < nn; i++)
			*(lpUV + i*UV_length + UV_index_n) = 0.0;
		return 9e9;
	}

	double DIC_Coeff = SDIC_Compute(UV_index_n, UV_index, Img1, Img2, Para1, Para2, shapepara, lpUV_xy, Znssd_reqd, lpROI,
		nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firsttime, direction);

	if (DIC_Coeff < PSSDab_thresh)
	{
		for (i = 0; i < nn; i++)
			lpUV[i*UV_length + UV_index_n] = (float)shapepara[i];
		return DIC_Coeff;
	}
	else if (InitFlow)
	{
		for (i = 0; i < nn - 2; i++)
			shapepara[i] = 0.0;
		shapepara[nn - 2] = 1.0, shapepara[nn - 1] = 0.0;

		//Look up for flow in the low res image
		float x = lpUV_xy[2 * UV_index_n] * SR, y = lpUV_xy[2 * UV_index_n + 1] * SR;
		if (DIC_Algo <= 1)
			shapepara[0] = 0.5*(FlowU[(int)(x + 0.5) + (int)(y + 0.5)*width1] / direction[0] + FlowV[(int)(x + 0.5) + (int)(y + 0.5)*width1] / direction[1]);
		else
		{
			shapepara[0] = FlowU[(int)(x + 0.5) + (int)(y + 0.5)*width1];
			shapepara[1] = FlowV[(int)(x + 0.5) + (int)(y + 0.5)*width1];
		}

		DIC_Coeff = SDIC_Compute(UV_index_n, UV_index, Img1, Img2, Para1, Para2, shapepara, lpUV_xy, Znssd_reqd, lpROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firsttime, direction);
		if (DIC_Coeff < PSSDab_thresh)
		{
			for (i = 0; i < nn; i++)
				lpUV[i*UV_length + UV_index_n] = (float)shapepara[i];
			return DIC_Coeff;
		}
		else
		{
			for (i = 0; i < nn; i++)
				lpUV[i*UV_length + UV_index_n] = 0.0f;
			return 9e9;
		}
	}
	else
	{
		for (i = 0; i < nn; i++)
			lpUV[i*UV_length + UV_index_n] = 0.0f;
		return 9e9;
	}
}
int SGreedyMatching(char *Img1, char *Img2, float *displacement, bool *lpROI_calculated, bool *tROI, int *flowhsubset, CPoint2 *SparseCorres1, CPoint2 *SparseCorres2, int nSeedPoints, LKParameters LKArg, double SR, int nchannels, int width1, int height1, int width2, int height2, double Scale, double *Epipole)
{
	int ii, kk, cp;
	bool debug = false;
	int swidth1 = (int)(1.0*width1 / SR), swidth2 = (int)(1.0*width2 / SR), sheight1 = (int)(1.0*height1 / SR), sheight2 = (int)(1.0*height2 / SR);
	int length1 = width1*height1, length2 = width2*height2, slength1 = swidth1*sheight1, slength2 = swidth2*sheight2;

	int DIC_Algo = LKArg.DIC_Algo, Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, InterpAlgo = LKArg.InterpAlgo;
	int Convergence_Criteria = LKArg.Convergence_Criteria, Iter_Max = LKArg.IterMax, Analysis_Speed = LKArg.Analysis_Speed;
	double Gsigma = LKArg.Gsigma, PSSDab_thresh = LKArg.PSSDab_thresh, ZNCCThresh = LKArg.ZNCCThreshold, step = LKArg.step*SR;

	// Prepare image data
	double *SImg1 = new double[nchannels*length1];
	double *SImg2 = new double[nchannels*length2];
	double *Para1 = new double[nchannels*length1];
	double *Para2 = new double[nchannels*length2];
	if (Gsigma > 0.0)
	{
		for (kk = 0; kk < nchannels; kk++)
		{
			Gaussian_smooth(Img1 + kk*length1, SImg1 + kk*length1, height1, width1, 255.0, Gsigma);
			Gaussian_smooth(Img2 + kk*length2, SImg2 + kk*length2, height2, width2, 255.0, Gsigma);
		}
	}
	else
	{
		for (kk = 0; kk < nchannels; kk++)
		{
			for (ii = 0; ii < length1; ii++)
				SImg1[ii + kk*length1] = (double)((int)((unsigned char)(Img1[ii + kk*length1])));
			for (ii = 0; ii < length2; ii++)
				SImg2[ii + kk*length2] = (double)((int)((unsigned char)(Img2[ii + kk*length2])));
		}
	}

	for (kk = 0; kk < nchannels; kk++)
	{
		Generate_Para_Spline(SImg1 + kk*length1, Para1 + kk*length1, width1, height1, InterpAlgo);
		Generate_Para_Spline(SImg2 + kk*length2, Para2 + kk*length2, width2, height2, InterpAlgo);
	}

	float x, y;
	int indx, indy, m, M, UV_index, UV_index_n;
	double vr_temp[] = { 0.0, 0.65, 0.8, 0.95 };
	double validity_ratio = vr_temp[Incomplete_Subset_Handling];
	double  ImgPt[3], epipline[3], direction[2];

	int total_valid_points = 0, total_calc_points = 0;
	for (ii = 0; ii < slength1; ii++)
		if (tROI[ii]) // 1 - Valid, 0 - Other
			total_valid_points++;
	if (total_valid_points == 0)
	{
		cout << "Did not find ROI!" << endl;
		delete[]SImg1;
		delete[]SImg2;
		delete[]Para1;
		delete[]Para2;
		return 1;
	}

	int NN[] = { 3, 7, 4, 8 }, nParas = NN[DIC_Algo];
	int UV_length = slength1;	// The actual value (i.e., total_calc_points, to be determined later) should be smaller.
	float *lpUV = new float[nParas*UV_length];	// U, V, Ux, Uy, Vx, Vy, Uxx, Uyy, Uxy, Vxx, Vyy, Vxy, (a) and b. or alpha, (a), and b
	float *lpUV_xy = new float[2 * UV_length];	// Coordinates of the points corresponding to lpUV

	float *lpResult_UV = new float[slength1 + slength2];
	for (ii = 0; ii < slength1 + slength2; ii++)
		lpResult_UV[ii] = 0.0f;
	for (ii = 0; ii < nParas*UV_length; ii++)
		lpUV[ii] = 0.0f;
	for (ii = 0; ii < 2 * UV_length; ii++)
		lpUV_xy[ii] = 0.0f;

	int maxhsubset = 50;
	double *Znssd_reqd = new double[2 * (2 * maxhsubset + 1)*(2 * maxhsubset + 1)*nchannels];
	float *Coeff = new float[UV_length];
	int *Tindex = new int[UV_length];

	int *iteration_check = new int[Iter_Max];
	for (m = 0; m < Iter_Max; m++)
		iteration_check[m] = 0;

	int *PointPerSeed = new int[nSeedPoints];
	for (ii = 0; ii < nSeedPoints; ii++)
		PointPerSeed[ii] = 0;

	double start = omp_get_wtime();
	int percent = 5, increment = 5;

	UV_index = 0;
	UV_index_n = 0;
	bool firstpoint = true;
	double UV_Guess[8], coeff;
	for (kk = 0; kk < nSeedPoints; kk++)
	{
		firstpoint = true;
		cp = 0, M = 0;

		x = (float)(int)(SparseCorres1[kk].x + 0.5), y = (float)(int)(SparseCorres1[kk].y + 0.5);

		indx = (int)(x / SR + 0.5), indy = (int)(y / SR + 0.5);
		if (lpROI_calculated[indy*swidth1 + indx] || !tROI[indy*swidth1 + indx])
			continue;
		else
		{
			UV_index = UV_index_n;
			lpUV_xy[2 * UV_index] = x, lpUV_xy[2 * UV_index + 1] = y;

			if (DIC_Algo <= 1)
			{
				ImgPt[0] = lpUV_xy[2 * UV_index], ImgPt[1] = lpUV_xy[2 * UV_index + 1], ImgPt[2] = 1.0;
				cross_product(ImgPt, Epipole, epipline);

				direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
				direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
				UV_Guess[0] = 0.5*((SparseCorres2[kk].x - SparseCorres1[kk].x) / direction[0] + (SparseCorres2[kk].y - SparseCorres1[kk].y) / direction[1]);
				if (DIC_Algo == 1)
				{
					UV_Guess[1] = Scale - 1.0, UV_Guess[2] = 0.0, UV_Guess[3] = 0.0, UV_Guess[4] = Scale - 1.0;
					UV_Guess[5] = 1.0, UV_Guess[6] = 0.0;
				}
			}
			else
			{
				UV_Guess[0] = SparseCorres2[kk].x - SparseCorres1[kk].x;
				UV_Guess[1] = SparseCorres2[kk].y - SparseCorres1[kk].y;
				UV_Guess[2] = Scale - 1.0, UV_Guess[3] = 0.0, UV_Guess[4] = 0.0, UV_Guess[5] = Scale - 1.0;
				UV_Guess[6] = 1.0, UV_Guess[7] = 0.0;
			}

			for (m = 0; m < nParas; m++)
				lpUV[m*UV_length + UV_index] = (float)UV_Guess[m];

			coeff = SDIC_Calculation(UV_index_n, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
			if (coeff < PSSDab_thresh)
			{
				cp += 1;
				Coeff[M] = (float)coeff;
				Tindex[M] = UV_index;
			}
			else
				M--;

			x = lpUV_xy[2 * UV_index];
			y = lpUV_xy[2 * UV_index + 1];
			lpROI_calculated[indy*swidth1 + indx] = true;
			firstpoint = false;
		}

		while (M >= 0)
		{
			if ((100 * (UV_index_n + 1)*step*step / total_valid_points - percent) > 0)
			{
				double elapsed = omp_get_wtime() - start;
				cout << "%" << 100 * (UV_index_n + 1)*step*step / total_valid_points << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / (percent)*(100.0 - percent) << endl;
				percent += increment;
			}

			UV_index = Tindex[M];
			x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
			M--; // Remove from the queque

			indx = (int)(x / SR + 0.5), indy = (int)((step + y) / SR + 0.5);
			if (indy < sheight1 && tROI[indy*swidth1 + indx] && !lpROI_calculated[indy*swidth1 + indx])
			{
				if (SDIC_CheckPointValidity(tROI, x, y + step, width1, height1, flowhsubset[(int)(x + 0.5) + (int)(y + step + 0.5)*width1], SR, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y + (float)step;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = SDIC_Calculation(UV_index_n + 1, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
					if (coeff < PSSDab_thresh)
					{
						cp++, M++, UV_index_n++;
						Coeff[M] = (float)coeff;
						Tindex[M] = UV_index_n;
						SDIC_AddtoQueue(Coeff, Tindex, M);
					}
					else
						if (debug)
							coeff = SDIC_Calculation(UV_index_n + 1, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
				}
				lpROI_calculated[indy*swidth1 + indx] = true;
			}

			indx = (int)(x / SR + 0.5), indy = (int)((y - step) / SR + 0.5);
			if (indy >= 0 && tROI[indy*swidth1 + indx] && !lpROI_calculated[indy*swidth1 + indx])
			{
				if (SDIC_CheckPointValidity(tROI, x, y - step, width1, height1, flowhsubset[(int)(x + 0.5) + (int)(y - step + 0.5)*width1], SR, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y - (float)step;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = SDIC_Calculation(UV_index_n + 1, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
					if (coeff < PSSDab_thresh)
					{
						cp++, M++, UV_index_n++;
						Coeff[M] = (float)coeff;
						Tindex[M] = UV_index_n;
						SDIC_AddtoQueue(Coeff, Tindex, M);
					}
					else
						if (debug)
							coeff = SDIC_Calculation(UV_index_n + 1, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
				}
				lpROI_calculated[indy*swidth1 + indx] = true;
			}

			indx = (int)((x - step) / SR + 0.5), indy = (int)(y / SR + 0.5);
			if (indx >= 0 && tROI[indy*swidth1 + indx] && !lpROI_calculated[indy*swidth1 + indx])
			{
				if (SDIC_CheckPointValidity(tROI, x - step, y, width1, height1, flowhsubset[(int)(x - step + 0.5) + (int)(y + 0.5)*width1], SR, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x - (float)step;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = SDIC_Calculation(UV_index_n + 1, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
					if (coeff < PSSDab_thresh)
					{
						cp++, M++, UV_index_n++;
						Coeff[M] = (float)coeff;
						Tindex[M] = UV_index_n;
						SDIC_AddtoQueue(Coeff, Tindex, M);
					}
					else
						if (debug)
							coeff = SDIC_Calculation(UV_index_n + 1, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
				}
				lpROI_calculated[indy*swidth1 + indx] = true;
			}

			indx = (int)((x + step) / SR + 0.5), indy = (int)(y / SR + 0.5);
			if (indx < swidth1 && tROI[indy*swidth1 + indx] && !lpROI_calculated[indy*swidth1 + indx])
			{
				if (SDIC_CheckPointValidity(tROI, x + step, y, width1, height1, flowhsubset[(int)(x + step + 0.5) + (int)(y + 0.5)*width1], SR, validity_ratio))
				{
					lpUV_xy[2 * (UV_index_n + 1)] = x + (float)step;
					lpUV_xy[2 * (UV_index_n + 1) + 1] = y;
					if (DIC_Algo <= 1)
					{
						ImgPt[0] = lpUV_xy[2 * (UV_index_n + 1)], ImgPt[1] = lpUV_xy[2 * (UV_index_n + 1) + 1], ImgPt[2] = 1.0;
						cross_product(ImgPt, Epipole, epipline);
						direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
						direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
					}

					coeff = SDIC_Calculation(UV_index_n + 1, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
					if (coeff < PSSDab_thresh)
					{
						cp++, M++, UV_index_n++;
						Coeff[M] = (float)coeff;
						Tindex[M] = UV_index_n;
						SDIC_AddtoQueue(Coeff, Tindex, M);
					}
					else
						if (debug)
							coeff = SDIC_Calculation(UV_index_n + 1, UV_index, SImg1, SImg2, Para1, Para2, lpUV, lpUV_xy, Znssd_reqd, tROI, nchannels, width1, height1, width2, height2, UV_length, flowhsubset, LKArg, SR, iteration_check, firstpoint, direction);
				}
				lpROI_calculated[indy*swidth1 + indx] = true;
			}
		}

		if (cp > 0)
			UV_index_n++;

		PointPerSeed[kk] = cp;
		total_calc_points += cp;
	}
	cout << "Finsish! Total time: " << omp_get_wtime() - start << endl;
	//// DIC calculation: End

	for (ii = 0; ii < total_calc_points; ii++)
	{
		if (lpUV[ii] != lpUV[ii])
		{
			displacement[(int)(lpUV_xy[2 * ii] / SR) + (int)(lpUV_xy[2 * ii + 1] / SR)*swidth1] = 0.0f;
			displacement[(int)(lpUV_xy[2 * ii] / SR) + (int)(lpUV_xy[2 * ii + 1] / SR)*swidth1 + slength1] = 0.0f;
			continue;
		}
		if (lpUV[UV_length + ii] != lpUV[UV_length + ii])
		{
			displacement[(int)(lpUV_xy[2 * ii] / SR) + (int)(lpUV_xy[2 * ii + 1] / SR)*swidth1] = 0.0f;
			displacement[(int)(lpUV_xy[2 * ii] / SR) + (int)(lpUV_xy[2 * ii + 1] / SR)*swidth1 + slength1] = 0.0f;
			continue;
		}

		if (DIC_Algo <= 1)
		{
			ImgPt[0] = lpUV_xy[2 * ii], ImgPt[1] = lpUV_xy[2 * ii + 1], ImgPt[2] = 1.0;
			cross_product(ImgPt, Epipole, epipline);
			direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
			direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);

			displacement[(int)(lpUV_xy[2 * ii] / SR) + (int)(lpUV_xy[2 * ii + 1] / SR)*swidth1] = (float)(direction[0] * lpUV[ii]);
			displacement[(int)(lpUV_xy[2 * ii] / SR) + (int)(lpUV_xy[2 * ii + 1] / SR)*swidth1 + slength1] = (float)(direction[1] * lpUV[ii]);
		}
		else
		{
			displacement[(int)(lpUV_xy[2 * ii] / SR) + (int)(lpUV_xy[2 * ii + 1] / SR)*swidth1] = lpUV[ii];
			displacement[(int)(lpUV_xy[2 * ii] / SR) + (int)(lpUV_xy[2 * ii + 1] / SR)*swidth1 + slength1] = lpUV[UV_length + ii];
		}
	}

	delete[]Tindex;
	delete[]Coeff;
	delete[]Znssd_reqd;
	delete[]SImg1;
	delete[]SImg2;
	delete[]Para1;
	delete[]Para2;
	delete[]lpUV_xy;
	delete[]lpResult_UV;
	delete[]lpUV;
	delete[]PointPerSeed;
	delete[]iteration_check;

	return 0;
}

void cvFlowtoFloat(Mat_<Point2f> &flow, float *fx, float *fy)
{
	int  width = flow.cols, height = flow.rows;
	for (int jj = 0; jj < height; jj++)
	{
		for (int ii = 0; ii < width; ii++)
		{
			const Point2f u = flow(jj, ii);
			fx[ii + (height - 1 - jj)*width] = u.x;
			fy[ii + (height - 1 - jj)*width] = -u.y;
		}
	}

	return;
}
void cvFloattoFlow(Mat_<Point2f> &flow, float *fx, float *fy)
{
	int  width = flow.cols, height = flow.rows;
	for (int jj = 0; jj < height; jj++)
	{
		for (int ii = 0; ii < width; ii++)
		{
			const Point2f u = flow(jj, ii);
			flow(jj, ii).x = fx[ii + (height - 1 - jj)*width];
			flow(jj, ii).y = -fy[ii + (height - 1 - jj)*width];
			//fx[ii + (height - 1 - jj)*width] = u.x;
			//fy[ii + (height - 1 - jj)*width] = -u.y;
		}
	}

	return;
}
void WarpImageFlow(float *flow, unsigned char *wImg21, unsigned char *Img2, int width, int height, int nchannels, int InterpAlgo, bool removeStatic)
{
	int ii, jj, kk, length = width*height;
	double u, v, du, dv, S[3];

	double *Para = new double[length*nchannels];
	for (kk = 0; kk < nchannels; kk++)
		Generate_Para_Spline(Img2 + kk*length, Para + kk*length, width, height, InterpAlgo);

	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			du = flow[ii + jj*width], dv = flow[ii + jj*width + length];
			if (removeStatic &&abs(du) < 0.005 && abs(dv) < 0.005)
			{
				for (kk = 0; kk < nchannels; kk++)
					wImg21[ii + jj*width + kk*length] = (unsigned char)(255);
				continue;
			}

			u = du + ii, v = dv + jj;
			if (u< 1.0 || u > width - 1.0 || v<1.0 || v>height - 1)
			{
				for (kk = 0; kk < nchannels; kk++)
					wImg21[ii + jj*width + kk*length] = (unsigned char)(255);
				continue;
			}

			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(Para + kk*length, width, height, u, v, S, -1, InterpAlgo);
				if (S[0] < 0.0)
					S[0] = 0.0;
				else if (S[0] > 255.0)
					S[0] = 255.0;
				wImg21[ii + jj*width + kk*length] = (unsigned char)((int)(S[0] + 0.5));
			}
		}
	}

	return;
}
void WarpImageFlowDouble(float *flow, double *wImg21, double *Img2, int width, int height, int nchannels, int InterpAlgo, bool removeStatic)
{
	int ii, jj, kk, length = width*height;
	double u, v, du, dv, S[3];

	double *Para = new double[length*nchannels];
	for (kk = 0; kk < nchannels; kk++)
		Generate_Para_Spline(Img2 + kk*length, Para + kk*length, width, height, InterpAlgo);

	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			du = flow[ii + jj*width], dv = flow[ii + jj*width + length];
			if (removeStatic &&abs(du) < 0.005 && abs(dv) < 0.005)
			{
				for (kk = 0; kk < nchannels; kk++)
					wImg21[ii + jj*width + kk*length] = 255;
				continue;
			}

			u = du + ii, v = dv + jj;
			if (u< 1.0 || u > width - 1.0 || v<1.0 || v>height - 1)
			{
				for (kk = 0; kk < nchannels; kk++)
					wImg21[ii + jj*width + kk*length] = 255;
				continue;
			}

			for (kk = 0; kk < nchannels; kk++)
			{
				Get_Value_Spline(Para + kk*length, width, height, u, v, S, -1, InterpAlgo);
				if (S[0] < 0.0)
					S[0] = 0.0;
				else if (S[0] > 255.0)
					S[0] = 255.0;
				wImg21[ii + jj*width + kk*length] = S[0];
			}
		}
	}

	return;
}
int WarpImageFlowDriver(char *Fin, char *Fout, char *FnameX, char *FnameY, int nchannels, int Gsigma, int InterpAlgo, bool removeStatic)
{
	int ii, jj, kk;

	Mat view = imread(Fin, nchannels == 1 ? 0 : 1);
	if (view.data == NULL)
	{
		cout << "Cannot load: " << Fin << endl;
		return 1;
	}
	int width = view.cols, height = view.rows, length = width*height;

	float *Flow = new float[2 * length];
	if (!ReadFlowBinary(FnameX, FnameY, Flow, Flow + length, width, height))
		return 2;

	if (Gsigma > 0.5)
	{
		double *Image = new double[length*nchannels];
		double *wImageD = new double[length*nchannels];
		for (kk = 0; kk < nchannels; kk++)
		{
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					Image[ii + (height - 1 - jj)*width + length*kk] = (double)(int)view.data[nchannels*ii + kk + jj*nchannels*width];
			Gaussian_smooth_Double(Image + kk*length, Image + kk*length, height, width, 255.0, Gsigma);
		}

		WarpImageFlowDouble(Flow, wImageD, Image, width, height, nchannels, InterpAlgo, removeStatic);
		SaveDataToImage(Fout, wImageD, width, height);

		delete[]Image;
		delete[]wImageD;
	}
	else
	{
		unsigned char *Image = new unsigned char[length*nchannels];
		unsigned char *wImage = new unsigned char[length*nchannels];
		for (kk = 0; kk < nchannels; kk++)
			for (jj = 0; jj < height; jj++)
				for (ii = 0; ii < width; ii++)
					Image[ii + jj*width + kk*length] = (unsigned char)view.data[nchannels*ii + (height - 1 - jj)*nchannels*width + kk];

		WarpImageFlow(Flow, wImage, Image, width, height, nchannels, InterpAlgo, removeStatic);
		SaveDataToImage(Fout, wImage, width, height);

		delete[]Image;
		delete[]wImage;
	}

	delete[]Flow;

	return 0;
}
int TVL1OpticalFlowDriver(int frameID, int frameJump, int nCams, int width, int height, char *PATH, TVL1Parameters argGF, bool forward, bool backward)
{
	int jj;
	char Fname1[200], Fname2[200];

	Mat frame0, frame1;
	float *fx = new float[width*height];
	float *fy = new float[width*height];

	Mat_<Point2f> flow;
	Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();
	tvl1->set("tau", argGF.tau);
	tvl1->set("lambda", argGF.lamda);
	tvl1->set("theta", argGF.theta);
	tvl1->set("epsilon", argGF.epsilon);
	tvl1->set("iterations", argGF.iterations);
	tvl1->set("nscales", argGF.nscales);
	tvl1->set("warps", argGF.warps);
	tvl1->set("useInitialFlow", true);

	double start = omp_get_wtime();
	for (jj = 0; jj < nCams; jj++)
	{
		sprintf(Fname1, "%s/Image/RandomTextureMap/C1_%05d.png", PATH, 1);
		sprintf(Fname2, "%s/Image/RandomTextureMap/C1_%05d.png", PATH, frameID + frameJump);
		frame0 = imread(Fname1, IMREAD_GRAYSCALE); //only accept grayscale image
		frame1 = imread(Fname2, IMREAD_GRAYSCALE); //only accept grayscale image

		if (!frame0.data || !frame1.data)
		{
			cout << "Cannot load " << Fname1 << " or " << Fname2 << endl;
			delete[]fx;
			delete[]fy;
			return 1;
		}
		else
			cout << "Loaded: " << endl << Fname1 << endl << Fname2 << endl;

		//Foward flow
		if (forward)
		{
			cout << "Compute forward flow for Cam " << jj + 1 << endl;
			tvl1->calc(frame0, frame1, flow);
			cout << "...finished in " << omp_get_wtime() - start << endl;

			sprintf(Fname1, "%s/Flow/FX%d_%05d.dat", PATH, jj + 1, frameID);
			sprintf(Fname2, "%s/Flow/FY%d_%05d.dat", PATH, jj + 1, frameID);
			cvFlowtoFloat(flow, fx, fy);
			if (!WriteFlowBinary(Fname1, Fname2, fx, fy, width, height))
			{
				cout << "Cannot write " << Fname1 << " or " << Fname2 << endl;
				continue;
			}
		}

		if (backward)
		{
			//Backward flow
			sprintf(Fname1, "%s/Flow/RX%d_%05d.dat", PATH, jj + 1, frameID - 1);
			sprintf(Fname2, "%s/Flow/RY%d_%05d.dat", PATH, jj + 1, frameID - 1);
			if (ReadFlowBinary(Fname1, Fname2, fx, fy, width, height))
			{
				flow = Mat_<Point2f>(height, width);
				cvFloattoFlow(flow, fx, fy);
				tvl1->set("useInitialFlow", true);
				printf("Initial flow is available from frame %d \n", frameID - 1);
			}
			else
				tvl1->set("useInitialFlow", false);

			cout << "Compute backward flow for Cam " << jj + 1 << endl;
			tvl1->calc(frame1, frame0, flow);
			cout << "...finished in " << omp_get_wtime() - start << endl;

			sprintf(Fname1, "%s/Flow/RX%d_%05d.dat", PATH, jj + 1, frameID + frameJump);
			sprintf(Fname2, "%s/Flow/RY%d_%05d.dat", PATH, jj + 1, frameID + frameJump);
			cvFlowtoFloat(flow, fx, fy);
			if (!WriteFlowBinary(Fname1, Fname2, fx, fy, width, height))
			{
				cout << "Cannot write " << Fname1 << " or " << Fname2 << endl;
				continue;
			}
		}
	}

	delete[]fx;
	delete[]fy;
	return 0;
}
void NonMinimalSupression1D(double *src, int *MinEle, int &nMinEle, int halfWnd, int nele)
{
	int i = 0, minInd = 0, srcCnt = 0, ele;
	nMinEle = 0;
	while (i < nele)
	{
		if (minInd < i - halfWnd)
			minInd = i - halfWnd;

		ele = min(i + halfWnd, nele);
		while (minInd <= ele)
		{
			srcCnt++;
			if (src[minInd] < src[i])
				break;
			minInd++;
		}

		if (minInd > ele) // src(i) is a maxima in the search window
		{
			MinEle[nMinEle] = i, nMinEle++; // the loop above suppressed the maximum, so set it back
			minInd = i + 1;
			i += halfWnd;
		}
		i++;
	}

	return;
}
double ComputeDepth3DBased(double hypoDepth, double *rayDirect, double *CxPd, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, int *DenseScale, int DIC_Algo, int ConvCriteria, double ZNCCThresh, double *Cxcenter, FlowVect &Paraflow, CPoint2 *Cx_init, double neighRadius, int cwidth, int cheight, int InterpAlgo, bool FlowRefinement, double &depth1, double &depth3, double *TCost, bool Intersection = false, CPoint2 *Out_Cpts = NULL, int index = 0, bool reprojectionErr = false)
{
	int ii, nCams = Fimgs.nCams;

	double *aPmat = new double[12 * (nCams + 1)];
	//Pmat1[12] = {DInfo.K[0], DInfo.K[1], DInfo.K[2], 0.0, DInfo.K[3], DInfo.K[4], DInfo.K[5], 0.0, DInfo.K[6], DInfo.K[7], DInfo.K[8], 0.0};
	CPoint2 Ppts, apts[3]; CPoint3 WC;
	if (reprojectionErr)
	{
		aPmat[0] = DInfo.K[0], aPmat[1] = DInfo.K[1], aPmat[2] = DInfo.K[2], aPmat[3] = 0.0;
		aPmat[4] = DInfo.K[3], aPmat[5] = DInfo.K[4], aPmat[6] = DInfo.K[5], aPmat[7] = 0.0;
		aPmat[8] = DInfo.K[6], aPmat[9] = DInfo.K[7], aPmat[10] = DInfo.K[8], aPmat[11] = 0.0;
		for (ii = 0; ii < 12 * nCams; ii++)
			aPmat[ii + 12] = DInfo.P[ii];

		Ppts.y = (rayDirect[1] - DInfo.iK[5]) / DInfo.iK[4];
		Ppts.x = (rayDirect[0] - DInfo.iK[2] - Ppts.y*DInfo.iK[1]) / DInfo.iK[0];
	}
	int hsubset, clength = cwidth*cheight, nframes = 3;
	CPoint2 dPts[2], Ppos, C1pos1, C1pos2, C1pos3, C2pos1, C2pos2, C2pos3;

	double pix3D[3], A, B, C, E, F, G, ABC2, M, N, P, Cost1, Cost2;

	double Cxpnum1u, Cxpnum1v, Cxpdenum1, Cxnum1u, Cxnum1v, Cxdenum1, CxSdenum1, S1[3], S2[3], S3[3], S4[3];
	CPoint2 *Cxpos1 = new CPoint2[nCams];
	CPoint2 *Cxpos2 = new CPoint2[nCams];
	CPoint2 *Cxpos3 = new CPoint2[nCams];
	for (ii = 0; ii < nCams; ii++)
	{
		Cxpnum1u = hypoDepth*CxPd[3 * ii];
		Cxpnum1v = hypoDepth*CxPd[3 * ii + 1];
		Cxpdenum1 = hypoDepth*CxPd[3 * ii + 2];
		Cxnum1u = Cxpnum1u + DInfo.P[12 * ii + 3];
		Cxnum1v = Cxpnum1v + DInfo.P[12 * ii + 7];
		Cxdenum1 = Cxpdenum1 + DInfo.P[12 * ii + 11];
		CxSdenum1 = Cxdenum1*Cxdenum1;

		Cxpos2[ii].x = Cxnum1u / Cxdenum1;
		Cxpos2[ii].y = Cxnum1v / Cxdenum1;

		if (abs(Cxpos2[ii].x - Cx_init[3 * ii + 1].x) > neighRadius || abs(Cxpos2[ii].y - Cx_init[3 * ii + 1].y) > neighRadius)
		{
			TCost[0] = 9.9e9, TCost[1] = 9.9e9;
			return 9.9e9;
		}
		LensDistortion_Point(Cxpos2[ii], DInfo.K + 9 * (ii + 1), DInfo.distortion + 13 * (ii + 1));
	}

	for (ii = 0; ii < nCams; ii++)
	{
		//Get_Value_Spline_Float(Paraflow.C21x+ii*clength, cwidth, cheight, Cxpos2[ii].x, Cxpos2[ii].y, S1, -1, InterpAlgo);
		//Get_Value_Spline_Float(Paraflow.C21y+ii*clength, cwidth, cheight, Cxpos2[ii].x, Cxpos2[ii].y, S2, -1, InterpAlgo);
		//Get_Value_Spline_Float(Paraflow.C23x+ii*clength, cwidth, cheight, Cxpos2[ii].x, Cxpos2[ii].y, S3, -1, InterpAlgo);
		//Get_Value_Spline_Float(Paraflow.C23y+ii*clength, cwidth, cheight, Cxpos2[ii].x, Cxpos2[ii].y, S4, -1, InterpAlgo);

		Cxpos1[ii].x = Cxpos2[ii].x + S1[0];
		Cxpos1[ii].y = Cxpos2[ii].y + S2[0];
		Cxpos3[ii].x = Cxpos2[ii].x + S3[0];
		Cxpos3[ii].y = Cxpos2[ii].y + S4[0];

		if (abs(Cxpos1[ii].x - Cx_init[3 * ii].x) > neighRadius || abs(Cxpos1[ii].y - Cx_init[3 * ii].y) > neighRadius || abs(Cxpos3[ii].x - Cx_init[3 * ii + 2].x) > neighRadius || abs(Cxpos3[ii].y - Cx_init[3 * ii + 2].y) > neighRadius)
		{
			Cost1 = 9.9e9, Cost2 = 9.9e9;
			goto Done;
		}
	}

	if (FlowRefinement)
	{
		for (ii = 0; ii < nCams; ii++)
		{
			//2->1:
			dPts[0].x = Cxpos2[ii].x, dPts[0].y = Cxpos2[ii].y;
			dPts[1].x = Cxpos1[ii].x, dPts[1].y = Cxpos1[ii].y;
			hsubset = DenseScale[(int)(dPts[0].x + 0.5) + (int)(dPts[0].y + 0.5)*cwidth + 2 * ii*clength];
			if (hsubset == 0)
			{
				Cost1 = 9.9e9, Cost2 = 9.9e9;
				goto Done;
			}
			//if(EpipSearchLK(dPts, EpiLine+3*ii, Fimgs.Img+(3*ii+1)*clength, Fimgs.Img+3*ii*clength, Fimgs.Para+(3*ii+1)*clength, Fimgs.Para+3*ii*clength, cwidth, cheight, hsubset, ConvCriteria, DIC_Algo, ZNCCThresh, InterpAlgo)<ZNCCThresh)
			{
				Cost1 = 9.9e9, Cost2 = 9.9e9;
				goto Done;
			}
			Cxpos1[ii].x = dPts[1].x, Cxpos1[ii].y = dPts[1].y;

			//2->3:
			dPts[0].x = Cxpos2[ii].x, dPts[0].y = Cxpos2[ii].y;
			dPts[1].x = Cxpos3[ii].x, dPts[1].y = Cxpos3[ii].y;
			hsubset = DenseScale[(int)(dPts[0].x + 0.5) + (int)(dPts[0].y + 0.5)*cwidth + (2 * ii + 1)*clength];
			if (hsubset == 0)
			{
				Cost1 = 9.9e9, Cost2 = 9.9e9;
				goto Done;
			}
			//if(EpipSearchLK(dPts, EpiLine+3*ii, Fimgs.Img+(3*ii+1)*clength, Fimgs.Img+(3*ii+2)*clength, Fimgs.Para+(3*ii+1)*clength, Fimgs.Para+(3*ii+2)*clength, cwidth, cheight, hsubset, ConvCriteria, DIC_Algo, ZNCCThresh, InterpAlgo)<ZNCCThresh)
			{
				Cost1 = 9.9e9, Cost2 = 9.9e9;
				goto Done;
			}
			Cxpos3[ii].x = dPts[1].x, Cxpos3[ii].y = dPts[1].y;
		}

		if (Out_Cpts != NULL)
		{
			for (ii = 0; ii < nCams; ii++)
			{
				Out_Cpts[3 * ii].x = Cxpos1[ii].x, Out_Cpts[3 * ii].y = Cxpos1[ii].y;
				Out_Cpts[3 * ii + 1].x = Cxpos2[ii].x, Out_Cpts[3 * ii + 1].y = Cxpos2[ii].y;
				Out_Cpts[3 * ii + 2].x = Cxpos3[ii].x, Out_Cpts[3 * ii + 2].y = Cxpos3[ii].y;
			}
		}
	}

	for (ii = 0; ii < nCams; ii++)
	{
		Undo_distortion(Cxpos1[ii], DInfo.K + 9 * (ii + 1), DInfo.distortion + 13 * (ii + 1));
		Undo_distortion(Cxpos3[ii], DInfo.K + 9 * (ii + 1), DInfo.distortion + 13 * (ii + 1));
	}

	if (reprojectionErr)
	{
		Cost1 = 0.0, Cost2 = 0.0;
		CPoint2 *reapts = new CPoint2[nCams + 1];
		//2->1:
		NviewTriangulation(Cxpos1, aPmat, &WC, nCams + 1);
		//Project3DtoImg(WC, reapts, aPmat+12, DInfo.K+9, nCams);

		for (ii = 0; ii < nCams; ii++)
			Cost1 += abs(reapts[ii].x - apts[ii + 1].x) + abs(reapts[ii].y - apts[ii + 1].y);
		if (Intersection)
			depth3 = WC.z;

		//2->3
		NviewTriangulation(Cxpos3, aPmat, &WC, nCams + 1);
		//Project3DtoImg(WC, apts, aPmat+12, DInfo.K+9, nCams);

		for (ii = 0; ii < nCams; ii++)
			Cost2 += abs(reapts[ii].x - apts[ii + 1].x) + abs(reapts[ii].y - apts[ii + 1].y);
		if (Intersection)
			depth3 = WC.z;
		delete[]reapts;
	}
	else
	{
		double sumP = 0.0, sumN = 0.0, sumM = 0.0;
		double *Cxpix3d = new double[nCams * 3];

		//2->1:
		for (ii = 0; ii < nCams; ii++)
		{
			pix3D[0] = DInfo.iK[9 * (ii + 1)] * Cxpos1[ii].x + DInfo.iK[9 * (ii + 1) + 1] * Cxpos1[ii].y + DInfo.iK[9 * (ii + 1) + 2]; pix3D[1] = DInfo.iK[9 * (ii + 1) + 4] * Cxpos1[ii].y + DInfo.iK[9 * (ii + 1) + 5];
			pix3D[2] = 1.0;
			Cxpix3d[3 * ii] = DInfo.RTx1[12 * ii] * pix3D[0] + DInfo.RTx1[12 * ii + 1] * pix3D[1] + DInfo.RTx1[12 * ii + 2] + DInfo.RTx1[12 * ii + 3];
			Cxpix3d[3 * ii + 1] = DInfo.RTx1[12 * ii + 4] * pix3D[0] + DInfo.RTx1[12 * ii + 5] * pix3D[1] + DInfo.RTx1[12 * ii + 6] + DInfo.RTx1[12 * ii + 7];
			Cxpix3d[3 * ii + 2] = DInfo.RTx1[12 * ii + 8] * pix3D[0] + DInfo.RTx1[12 * ii + 9] * pix3D[1] + DInfo.RTx1[12 * ii + 10] + DInfo.RTx1[12 * ii + 11];

			A = Cxpix3d[3 * ii] - Cxcenter[3 * ii], B = Cxpix3d[3 * ii + 1] - Cxcenter[3 * ii + 1], C = Cxpix3d[3 * ii + 2] - Cxcenter[3 * ii + 2];
			E = Cxcenter[3 * ii] * Cxpix3d[3 * ii + 1] - Cxcenter[3 * ii + 1] * Cxpix3d[3 * ii], F = Cxcenter[3 * ii + 1] * Cxpix3d[3 * ii + 2] - Cxcenter[3 * ii + 2] * Cxpix3d[3 * ii + 1], G = Cxcenter[3 * ii + 2] * Cxpix3d[3 * ii] - Cxcenter[3 * ii] * Cxpix3d[3 * ii + 2];

			M = (B*B + C*C)*rayDirect[0] * rayDirect[0] + (A*A + C*C)*rayDirect[1] * rayDirect[1] + (A*A + B*B)*rayDirect[2] * rayDirect[2] - 2.0*A*C*rayDirect[0] * rayDirect[2] - 2.0*A*B*rayDirect[0] * rayDirect[1] - 2.0*B*C*rayDirect[1] * rayDirect[2];
			N = 2.0*((C*G - B*E)*rayDirect[0] + (A*E - C*F)*rayDirect[1] + (B*F - A*G)*rayDirect[2]);
			P = E*E + F*F + G*G;

			ABC2 = A*A + B*B + C*C;
			M /= ABC2, N /= ABC2; P /= ABC2;
			sumP += P, sumN += N, sumM += M;
		}
		Cost1 = sumP - sumN*sumN / sumM / 4.0;
		if (Intersection)
			depth1 = -sumN / sumM / 2.0;

		//2->3:
		sumP = 0.0, sumN = 0.0, sumM = 0.0;
		for (ii = 0; ii < nCams; ii++)
		{
			pix3D[0] = DInfo.iK[9 * (ii + 1)] * Cxpos3[ii].x + DInfo.iK[9 * (ii + 1) + 1] * Cxpos3[ii].y + DInfo.iK[9 * (ii + 1) + 2], pix3D[1] = DInfo.iK[9 * (ii + 1) + 4] * Cxpos3[ii].y + DInfo.iK[9 * (ii + 1) + 5]; pix3D[2] = 1.0;
			Cxpix3d[3 * ii] = DInfo.RTx1[12 * ii] * pix3D[0] + DInfo.RTx1[12 * ii + 1] * pix3D[1] + DInfo.RTx1[12 * ii + 2] + DInfo.RTx1[12 * ii + 3];
			Cxpix3d[3 * ii + 1] = DInfo.RTx1[12 * ii + 4] * pix3D[0] + DInfo.RTx1[12 * ii + 5] * pix3D[1] + DInfo.RTx1[12 * ii + 6] + DInfo.RTx1[12 * ii + 7];
			Cxpix3d[3 * ii + 2] = DInfo.RTx1[12 * ii + 8] * pix3D[0] + DInfo.RTx1[12 * ii + 9] * pix3D[1] + DInfo.RTx1[12 * ii + 10] + DInfo.RTx1[12 * ii + 11];

			A = Cxpix3d[3 * ii] - Cxcenter[3 * ii], B = Cxpix3d[3 * ii + 1] - Cxcenter[3 * ii + 1], C = Cxpix3d[3 * ii + 2] - Cxcenter[3 * ii + 2];
			E = Cxcenter[3 * ii] * Cxpix3d[3 * ii + 1] - Cxcenter[3 * ii + 1] * Cxpix3d[3 * ii], F = Cxcenter[3 * ii + 1] * Cxpix3d[3 * ii + 2] - Cxcenter[3 * ii + 2] * Cxpix3d[3 * ii + 1], G = Cxcenter[3 * ii + 2] * Cxpix3d[3 * ii] - Cxcenter[3 * ii] * Cxpix3d[3 * ii + 2];

			M = (B*B + C*C)*rayDirect[0] * rayDirect[0] + (A*A + C*C)*rayDirect[1] * rayDirect[1] + (A*A + B*B)*rayDirect[2] * rayDirect[2] - 2.0*A*C*rayDirect[0] * rayDirect[2] - 2.0*A*B*rayDirect[0] * rayDirect[1] - 2.0*B*C*rayDirect[1] * rayDirect[2];
			N = 2.0*((C*G - B*E)*rayDirect[0] + (A*E - C*F)*rayDirect[1] + (B*F - A*G)*rayDirect[2]);
			P = E*E + F*F + G*G;

			ABC2 = A*A + B*B + C*C;
			M /= ABC2, N /= ABC2; P /= ABC2;
			sumP += P, sumN += N, sumM += M;
		}
		Cost2 = sumP - sumN*sumN / sumM / 4.0;
		if (Intersection)
			depth3 = -sumN / sumM / 2.0;

		delete[]Cxpix3d;
	}

Done:
	TCost[0] = Cost1, TCost[1] = Cost2;
	return Cost1 + Cost2;
}
double GoldenPointDepth3DBased(IJZ3 &optimDepth, double hdepth, double ldepth, double *rayDirect, double *CxPd, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &Paraflow, int *DenseScale, int DIC_Algo, int ConvCriteria, double znccThresh, double *Cxcenter, CPoint2 *Cx_init, double neighRadius, int cwidth, int cheight, int InterpAlgo, bool FlowRefinement)
{
	double R = (sqrt(5.0) - 1.0) / 2.0;
	double depth1, depth3, TCost[2], xL, xu, x1, x2, xopt, d, f1, f2, fx, ea = 1.0, es = abs(hdepth - ldepth)*1.0e-6;

	xL = ldepth, xu = hdepth, d = R*(xu - xL);
	x1 = xL + d, x2 = xu - d;

	f1 = ComputeDepth3DBased(x1, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost);
	f2 = ComputeDepth3DBased(x2, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost);

	if (f1 < f2)
	{
		xopt = x1;
		fx = f1;
	}
	else
	{
		xopt = x2;
		fx = f2;
	}

	int iter = 0, maxIter = 30;
	while (iter < maxIter && ea > es)
	{
		iter++;
		d = R*d;
		if (f1 < f2)
		{
			xL = x2;
			x2 = x1;
			x1 = xL + d;
			f2 = f1;
			f1 = ComputeDepth3DBased(x1, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost, true);
		}
		else
		{
			xu = x1;
			x1 = x2;
			x2 = xu - d;
			f1 = f2;
			f2 = ComputeDepth3DBased(x2, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost, true);
		}

		if (f1<f2)
		{
			xopt = x1;
			fx = f1;
		}
		else
		{
			xopt = x2;
			fx = f2;
		}
		if (abs(xopt)>1.0e-9)
			ea = (1.0 - R) * abs((xu - xL) / xopt)*100.0;
	}

	optimDepth.z1 = depth1, optimDepth.z2 = xopt, optimDepth.z3 = depth3;
	return fx;
}
double BrentPointDepth3DBased(IJZ3 &optimDepth, double ldepth, double hdepth, double *rayDirect, double *CxPd, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &Paraflow, int *DenseScale, int DIC_Algo, int ConvCriteria, double znccThresh, double *Cxcenter, CPoint2 *Cx_init, double neighRadius, int cwidth, int cheight, int InterpAlgo, bool FlowRefinement, double FractError = 1.0e-6, CPoint2 *Out_Cpts = NULL, int index = 0)
{
	double tol = abs(hdepth - ldepth)*FractError, phi = (1.0 + sqrt(5.0)) / 2.0, rho = 2.0 - phi;
	double TCost[2], depth1, depth3, xL, xH, u, v, w, x, fu, fv, fw, fx, xm, d, e, r, p, q, s;

	xL = ldepth, xH = hdepth;
	u = xL + rho*(xH - xL);
	v = u; w = u; x = u;
	fu = ComputeDepth3DBased(u, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost, true, Out_Cpts, 0);
	fv = fu; fw = fu; fx = fu;
	xm = 0.5*(xL + xH);
	d = 0.0, e = 0.0;

	bool para;
	int iter = 0, maxIter = 30;
	while (iter < maxIter && abs(x - xm) > tol)
	{
		iter++;
		para = abs(e) > tol;
		if (para) //Try parabolic fit.
		{
			r = (x - w)*(fx - fv); q = (x - v)*(fx - fw);
			p = (x - v)*q - (x - w)*r; s = 2.0*(q - r);
			if (s > 0.0)
				p = -p;

			s = abs(s);
			para = ((abs(p) < abs(0.5*s*e)) & (p > s*(xL - x)) & (p < s*(xH - x)));
			if (para) //Is the parabola acceptable?
			{
				e = d; d = p / s;
			}
		}
		if (!para) //Golden-section step
		{
			if (x >= xm)
				e = xL - x;
			else
				e = xH - x;
			d = rho*e;
		}
		u = x + d;
		fu = ComputeDepth3DBased(u, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost, true, Out_Cpts, 0);

		// Update xL, xH, x, v, w, xm
		if (fu <= fx)
		{
			if (u >= x)
				xL = x;
			else
				xH = x;

			v = w; fv = fw; w = x; fw = fx; 	x = u; fx = fu;
		}
		else
		{
			if (u < x)
				xL = u;
			else
				xH = u;

			if ((fu <= fw) || (w == x))
			{
				v = w; fv = fw; 	w = u; fw = fu;
			}
			else if ((fu <= fv) || (v == x) || (v == w))
			{
				v = u; fv = fu;
			}
		}
		xm = 0.5*(xL + xH);
	}

	optimDepth.z1 = depth1, optimDepth.z2 = u, optimDepth.z3 = depth3;

	return fu;
}
double ClosetPointToLine(CPoint2 &cp, CPoint2 p, double *line)
{
	//Convert to line of the form ax+by+c = 0 to y = mx+n
	double m = -line[0] / line[1], n = -line[2] / line[1];
	cp.x = (m*p.y + p.x - m*n) / (m*m + 1.0);
	cp.y = (m*m*p.y + m*p.x + n) / (m*m + 1.0);

	return abs(p.y - m*p.x - n) / sqrt(m*m + 1.0);
}
void ProjectDepthFlowToImage(double hypoDepth, double *CxPd, DevicesInfo &DInfo, CPoint2 *Cxpos)
{
	double CxP1d, CxP2d, CxP3d;
	double Cxpnum1u, Cxpnum1v, Cxpdenum1, Cxnum1u, Cxnum1v, Cxdenum1, CxSdenum1;

	int nCams = DInfo.nCams;

	for (int ii = 0; ii < nCams; ii++)
	{
		CxP1d = CxPd[3 * ii], CxP2d = CxPd[3 * ii + 1], CxP3d = CxPd[3 * ii + 2];

		Cxpnum1u = hypoDepth*CxP1d;
		Cxpnum1v = hypoDepth*CxP2d;
		Cxpdenum1 = hypoDepth*CxP3d;
		Cxnum1u = Cxpnum1u + DInfo.P[3 + 12 * ii];
		Cxnum1v = Cxpnum1v + DInfo.P[7 + 12 * ii];
		Cxdenum1 = Cxpdenum1 + DInfo.P[11 + 12 * ii];
		CxSdenum1 = Cxdenum1*Cxdenum1;

		Cxpos[ii].x = Cxnum1u / Cxdenum1;
		Cxpos[ii].y = Cxnum1v / Cxdenum1;
	}

	return;
}
double ProjectedDepthCost(double depth, double *CxPd, DevicesInfo &DInfo, CPoint2 *FlowEndpts, double *iCov, int TwoToThree)
{
	int nCams = DInfo.nCams;
	CPoint2 *projectedPts = new CPoint2[nCams];
	//CPoint2 *fep = new CPoint2 [nCams];
	//double *iCov = new double[4*nCams];

	ProjectDepthFlowToImage(depth, CxPd, DInfo, projectedPts);

	for (int ii = 0; ii < nCams; ii++)
		LensDistortion_Point(projectedPts[ii], DInfo.K + 9 * (ii + 1), DInfo.distortion + 13 * (ii + 1));

	double	cost = 0.0;
	for (int ii = 0; ii < nCams; ii++)
		cost += pow(projectedPts[ii].x - FlowEndpts[2 * ii + TwoToThree].x, 2) *iCov[8 * ii + 4 * TwoToThree] + (iCov[8 * ii + 1 + 4 * TwoToThree] + iCov[8 * ii + 2 + 4 * TwoToThree])*(projectedPts[ii].x - FlowEndpts[2 * ii + TwoToThree].x)*(projectedPts[ii].y - FlowEndpts[2 * ii + TwoToThree].y) + iCov[8 * ii + 3 + 4 * TwoToThree] * pow(projectedPts[ii].y - FlowEndpts[2 * ii + TwoToThree].y, 2);

	delete[]projectedPts;
	return cost;
}
double GoldenDepthCost(double &depth, double ldepth, double hdepth, double *CxPd, DevicesInfo &DInfo, CPoint2 *FlowEndpts, double *CxiCov, int TwoToThree)
{
	double R = (sqrt(5.0) - 1.0) / 2.0;
	double xL, xu, x1, x2, xopt, d, f1, f2, fx, ea = 1.0, es = abs(hdepth - ldepth)*1.0e-6;

	xL = ldepth, xu = hdepth, d = R*(xu - xL);
	x1 = xL + d, x2 = xu - d;

	f1 = ProjectedDepthCost(x1, CxPd, DInfo, FlowEndpts, CxiCov, TwoToThree);
	f2 = ProjectedDepthCost(x2, CxPd, DInfo, FlowEndpts, CxiCov, TwoToThree);

	if (f1 < f2)
	{
		xopt = x1;
		fx = f1;
	}
	else
	{
		xopt = x2;
		fx = f2;
	}

	int iter = 0, maxIter = 30;
	while (iter < maxIter && ea > es)
	{
		iter++;
		d = R*d;
		if (f1 < f2)
		{
			xL = x2;
			x2 = x1;
			x1 = xL + d;
			f2 = f1;
			f1 = ProjectedDepthCost(x1, CxPd, DInfo, FlowEndpts, CxiCov, TwoToThree);
		}
		else
		{
			xu = x1;
			x1 = x2;
			x2 = xu - d;
			f1 = f2;
			f2 = ProjectedDepthCost(x2, CxPd, DInfo, FlowEndpts, CxiCov, TwoToThree);
		}

		if (f1<f2)
		{
			xopt = x1;
			fx = f1;
		}
		else
		{
			xopt = x2;
			fx = f2;
		}
		if (abs(xopt)>1.0e-9)
			ea = (1.0 - R) * abs((xu - xL) / xopt)*100.0;
	}

	if (f1 < f2)
	{
		depth = x1;
		return f1;
	}
	else
	{
		depth = x2;
		return f2;
	}
}
double BrentDepthCost(double &depth, double ldepth, double hdepth, double *CxPd, DevicesInfo &DInfo, CPoint2 *FlowEndpts, double *CxiCov, int TwoToThree)
{
	const int ITMAX = 30;
	const double CGOLD = 0.3819660;
	const double ZEPS = 1.0e-10, tol = 1.0e-6;

	double  a, b, d = 0.0, etemp, fu, fv, fw, fx;
	double  p, q, r, tol1, tol2, u, v, w, x, xm;
	double e = 0.0;

	double ax = ldepth, bx = depth, cx = hdepth;
	a = (ax < cx ? ax : cx);
	b = (ax > cx ? ax : cx);
	x = w = v = bx;
	fw = fv = fx = ProjectedDepthCost(x, CxPd, DInfo, FlowEndpts, CxiCov, TwoToThree);

	for (int iter = 0; iter<ITMAX; iter++)
	{
		xm = 0.5*(a + b);
		tol2 = 2.0*(tol1 = tol*abs(x) + ZEPS);
		if (abs(x - xm) <= (tol2 - 0.5*(b - a)))
		{
			depth = x;
			return fx;
		}

		if (abs(e) > tol1)
		{
			r = (x - w)*(fx - fv);
			q = (x - v)*(fx - fw);
			p = (x - v)*q - (x - w)*r;
			q = 2.0*(q - r);
			if (q > 0.0) p = -p;
			q = abs(q);
			etemp = e;
			e = d;
			if (abs(p) >= abs(0.5*q*etemp) || p <= q*(a - x) || p >= q*(b - x))
				d = CGOLD*(e = (x >= xm ? a - x : b - x));
			else {
				d = p / q;
				u = x + d;
				if (u - a < tol2 || b - u < tol2)
					d = SIGN(tol1, xm - x);
			}
		}
		else
		{
			d = CGOLD*(e = (x >= xm ? a - x : b - x));
		}
		u = (abs(d) >= tol1 ? x + d : x + SIGN(tol1, d));
		fu = ProjectedDepthCost(u, CxPd, DInfo, FlowEndpts, CxiCov, TwoToThree);

		if (fu <= fx)
		{
			if (u >= x) a = x; else b = x;
			v = w; w = x; x = u;
			fv = fw; fw = fx; fx = fu;
		}
		else
		{
			if (u < x) a = u; else b = u;
			if (fu <= fw || w == x)
			{
				v = w;
				w = u;
				fv = fw;
				fw = fu;
			}
			else if (fu <= fv || v == x || v == w)
			{
				v = u;
				fv = fu;
			}
		}
	}

	depth = x;
	return fx;
}
double ComputeDepth2DBasedML(IJZ3 &hypoD, double *rayDirect, double *CxPd, double *Cxcenter, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &Paraflow, int DIC_Algo, int *DenseScale, int ConvCriteria, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo, double *TCost, int searchTech)
{
	int ii, jj, hsubset, clength = cwidth*cheight, nframes = Fimgs.nframes, nCams = 2;
	double PptD[3] = { hypoD.i, hypoD.j, hypoD.z2 };

	//1: Compute flow 2->1, 2->3
	CPoint2 Ppos, Cxpos2[2], FlowEndpts[2 * 2], wFlowEndpts[2];
	double  S1[3], S2[3], DICthresh = 0.8;

	ProjectDepthFlowToImage(PptD[2], CxPd, DInfo, Cxpos2);
	if (abs(Cxpos2[0].x - C12_init[2].x) > neighRadius || abs(Cxpos2[0].y - C12_init[2].y) > neighRadius || abs(Cxpos2[1].x - C12_init[3].x) > neighRadius || abs(Cxpos2[1].y - C12_init[3].y) > neighRadius)
		return 9.9e9;

	LensDistortion_Point(Cxpos2[0], DInfo.K + 9, DInfo.distortion + 13);
	LensDistortion_Point(Cxpos2[1], DInfo.K + 18, DInfo.distortion + 26);

	for (ii = 0; ii < nCams; ii++)
	{
		//From 2 to 1
		//Get_Value_Spline(Paraflow.C21x+ii*clength, cwidth, cheight, Cxpos2[ii].x, Cxpos2[ii].y, S1, -1, InterpAlgo);
		//Get_Value_Spline(Paraflow.C21y+ii*clength, cwidth, cheight, Cxpos2[ii].x, Cxpos2[ii].y, S2, -1, InterpAlgo);
		FlowEndpts[2 * ii].x = Cxpos2[ii].x + S1[0], FlowEndpts[2 * ii].y = Cxpos2[ii].y + S2[0];

		//From 2 to 3
		//Get_Value_Spline(Paraflow.C23x+ii*clength, cwidth, cheight, Cxpos2[ii].x, Cxpos2[ii].y, S1, -1, InterpAlgo);
		//Get_Value_Spline(Paraflow.C23y+ii*clength, cwidth, cheight, Cxpos2[ii].x, Cxpos2[ii].y, S2, -1, InterpAlgo);
		FlowEndpts[2 * ii + 1].x = Cxpos2[ii].x + S1[0], FlowEndpts[2 * ii + 1].y = Cxpos2[ii].y + S2[0];
	}

	if (abs(FlowEndpts[0].x - C12_init[0].x) > neighRadius || abs(FlowEndpts[0].y - C12_init[0].y) > neighRadius || abs(FlowEndpts[2].x - C12_init[1].x) > neighRadius || abs(FlowEndpts[2].y - C12_init[1].y) > neighRadius)
	{
		TCost[0] = 4.5e9, TCost[1] = 4.5e9;
		return 9.9e9;
	}
	if (abs(FlowEndpts[1].x - C12_init[4].x) > neighRadius || abs(FlowEndpts[1].y - C12_init[4].y) > neighRadius || abs(FlowEndpts[3].x - C12_init[5].x) > neighRadius || abs(FlowEndpts[3].y - C12_init[5].y) > neighRadius)
	{
		TCost[0] = 4.5e9, TCost[1] = 4.5e9;
		return 9.9e9;
	}

	CPoint2 dPts[2];
	double *CxiCov = new double[nCams * 4 * 2];
	for (ii = 0; ii < nCams; ii++)
	{
		//From 2 to 1
		dPts[0].x = Cxpos2[ii].x, dPts[0].y = Cxpos2[ii].y, dPts[1].x = FlowEndpts[2 * ii].x, dPts[1].y = FlowEndpts[2 * ii].y;
		hsubset = DenseScale[(int)(dPts[0].x + 0.5) + (int)(dPts[0].y + 0.5)*cwidth + 2 * ii*clength];
		//EpipSearchLK(dPts, EpiLine+3*ii, Fimgs.Img+(ii*nframes+1)*clength, Fimgs.Img+(ii*nframes)*clength, Fimgs.Para+(ii*nframes+1)*clength, Fimgs.Para+(ii*nframes)*clength, cwidth, cheight, hsubset, ConvCriteria, DIC_Algo, DICthresh, InterpAlgo, CxiCov+ii*4);
		FlowEndpts[2 * ii].x = dPts[1].x, FlowEndpts[2 * ii].y = dPts[1].y;

		//From 2 to 3
		dPts[0].x = Cxpos2[ii].x, dPts[0].y = Cxpos2[ii].y, dPts[1].x = FlowEndpts[2 * ii + 1].x, dPts[1].y = FlowEndpts[2 * ii + 1].y;
		hsubset = DenseScale[(int)(dPts[0].x + 0.5) + (int)(dPts[0].y + 0.5)*cwidth + (2 * ii + 1)*clength];
		//EpipSearchLK(dPts, EpiLine+3*ii, Fimgs.Img+(ii*nframes+1)*clength, Fimgs.Img+(ii*nframes+2)*clength, Fimgs.Para+(ii*nframes+1)*clength, Fimgs.Para+(ii*nframes+2)*clength, cwidth, cheight, hsubset, ConvCriteria, DIC_Algo, DICthresh, InterpAlgo, CxiCov+ii*4+8);
		FlowEndpts[2 * ii + 1].x = dPts[1].x, FlowEndpts[2 * ii + 1].y = dPts[1].y;
	}

	if (abs(FlowEndpts[0].x - C12_init[0].x) > neighRadius || abs(FlowEndpts[0].y - C12_init[0].y) > neighRadius || abs(FlowEndpts[2].x - C12_init[1].x) > neighRadius || abs(FlowEndpts[2].y - C12_init[1].y) > neighRadius)
	{
		TCost[0] = 4.5e9, TCost[1] = 4.5e9;
		return 9.9e9;
	}
	if (abs(FlowEndpts[1].x - C12_init[4].x) > neighRadius || abs(FlowEndpts[1].y - C12_init[4].y) > neighRadius || abs(FlowEndpts[3].x - C12_init[5].x) > neighRadius || abs(FlowEndpts[3].y - C12_init[5].y) > neighRadius)
	{
		TCost[0] = 4.5e9, TCost[1] = 4.5e9;
		return 9.9e9;
	}

	//2: Run exhausive search to find the coarse estimate of d1, d3
	double tcost, d1, d3;
	int nMinEle, nbins = 50;
	int *ind = new int[nbins * 2];
	double *MinCost = new double[nbins * 2];
	double *hypoCost = new double[nbins * 2], step = 0.1*hypoD.z1 / (nbins * 2);
	//FILE *fp = fopen("C:/temp/cost1.txt", "w+");
	for (ii = -nbins; ii < nbins; ii++)
	{
		hypoCost[ii + nbins] = ProjectedDepthCost(hypoD.z1 + step*ii, CxPd, DInfo, FlowEndpts, CxiCov, 0);
		//fprintf(fp, "%.2f %.4f\n", hypoD.z1+step*ii, hypoCost[ii+nbins]);
	}
	//fclose(fp);

	NonMinimalSupression1D(hypoCost, ind, nMinEle, 3, 2 * nbins);

	for (ii = 0; ii < nMinEle; ii++)
		MinCost[ii] = hypoCost[ind[ii]];
	Quick_Sort_Double(MinCost, ind, 0, nMinEle - 1);

	int npd1 = min(nMinEle / 2 + 1, nbins / 5);
	double *pd1 = new double[npd1];
	double *hypoCostd1 = new double[npd1];
	for (ii = 0; ii < npd1; ii++)
	{
		pd1[ii] = hypoD.z1 + step*(ind[ii] - nbins);
		hypoCostd1[ii] = MinCost[ii];
	}

	//fp = fopen("C:/temp/cost2.txt", "w+");
	for (ii = -nbins; ii < nbins; ii++)
	{
		hypoCost[ii + nbins] = ProjectedDepthCost(hypoD.z3 + step*ii, CxPd, DInfo, FlowEndpts, CxiCov, 1);
		//fprintf(fp, "%.2f %.4f\n", hypoD.z3+step*ii, hypoCost[ii+nbins]);
	}
	//fclose(fp);

	NonMinimalSupression1D(hypoCost, ind, nMinEle, 3, 2 * nbins);

	for (ii = 0; ii < nMinEle; ii++)
		MinCost[ii] = hypoCost[ind[ii]];
	Quick_Sort_Double(MinCost, ind, 0, nMinEle - 1);

	int npd3 = min(nMinEle / 2 + 1, nbins / 5);
	double *pd3 = new double[npd3];
	double *hypoCostd3 = new double[npd3];
	for (ii = 0; ii < npd3; ii++)
	{
		pd3[ii] = hypoD.z3 + step*(ind[ii] - nbins);
		hypoCostd3[ii] = MinCost[ii];
	}

	double tcostmin = 9e9;
	for (jj = 0; jj < npd1; jj++)
	{
		for (ii = 0; ii < npd3; ii++)
		{
			tcost = hypoCostd1[jj] + hypoCostd3[ii];
			if (tcost < tcostmin)
			{
				tcostmin = tcost;
				d1 = pd1[jj];
				d3 = pd3[ii];
			}
		}
	}

	delete[]MinCost;
	delete[]hypoCost;
	delete[]ind;
	delete[]pd1;
	delete[]pd3;
	delete[]hypoCostd1;
	delete[]hypoCostd3;

	//3: Refine d13 using brent or golden search
	if (searchTech == -1)
	{
		TCost[0] = BrentDepthCost(d1, d1 + step, d1 - step, CxPd, DInfo, FlowEndpts, CxiCov, 0);
		TCost[1] = BrentDepthCost(d3, d3 + step, d3 - step, CxPd, DInfo, FlowEndpts, CxiCov, 1);
		hypoD.z1 = d1, hypoD.z3 = d3;
		return TCost[0] + TCost[1];
	}
	else
	{
		TCost[0] = GoldenDepthCost(d1, d1 + step, d1 - step, CxPd, DInfo, FlowEndpts, CxiCov, 0);
		TCost[1] = GoldenDepthCost(d3, d3 + step, d3 - step, CxPd, DInfo, FlowEndpts, CxiCov, 1);
		hypoD.z1 = d1, hypoD.z3 = d3;
		return TCost[0] + TCost[1];
	}
}
double GoldenPointDepth2DBasedML(IJZ3 &optimDepth, double hdepth, double ldepth, double *rayDirect, double *CxPd, double *Cxcenter, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &Paraflow, int DIC_Algo, int *DenseScale, int ConvCriteria, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo)
{
	double R = (sqrt(5.0) - 1.0) / 2.0;
	double TCost[2], xL, xu, x1, x2, xopt, d, f1, f2, fx, ea = 1.0, es = abs(hdepth - ldepth)*1.0e-6;

	xL = ldepth, xu = hdepth, d = R*(xu - xL);
	x1 = xL + d, x2 = xu - d;

	int SearchTechnique = -2;
	IJZ3 hypoDx1(optimDepth.i, optimDepth.j, optimDepth.z1, x1, optimDepth.z3), hypoDx2(optimDepth.i, optimDepth.j, optimDepth.z1, x2, optimDepth.z3);
	f1 = ComputeDepth2DBasedML(hypoDx1, rayDirect, CxPd, Cxcenter, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, C12_init, neighRadius, cwidth, cheight, InterpAlgo, TCost, SearchTechnique);
	f2 = ComputeDepth2DBasedML(hypoDx2, rayDirect, CxPd, Cxcenter, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, C12_init, neighRadius, cwidth, cheight, InterpAlgo, TCost, SearchTechnique);

	if (f1 < f2)
	{
		xopt = x1;
		fx = f1;
	}
	else
	{
		xopt = x2;
		fx = f2;
	}

	int iter = 0, maxIter = 100;
	while (iter < maxIter && ea > es)
	{
		iter++;
		d = R*d;
		if (f1 < f2)
		{
			xL = x2;
			x2 = x1;
			x1 = xL + d;
			f2 = f1;
			hypoDx1.z2 = x1;
			f1 = ComputeDepth2DBasedML(hypoDx1, rayDirect, CxPd, Cxcenter, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, C12_init, neighRadius, cwidth, cheight, InterpAlgo, TCost, SearchTechnique);
		}
		else
		{
			xu = x1;
			x1 = x2;
			x2 = xu - d;
			f1 = f2;
			hypoDx2.z2 = x2;
			f1 = ComputeDepth2DBasedML(hypoDx2, rayDirect, CxPd, Cxcenter, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, C12_init, neighRadius, cwidth, cheight, InterpAlgo, TCost, SearchTechnique);
		}

		if (f1<f2)
		{
			xopt = x1;
			fx = f1;
		}
		else
		{
			xopt = x2;
			fx = f2;
		}
		if (abs(xopt)>1.0e-9)
			ea = (1.0 - R) * abs((xu - xL) / xopt)*100.0;
	}

	optimDepth.z1 = hypoDx1.z1, optimDepth.z2 = xopt, optimDepth.z3 = hypoDx1.z3;
	return fx;
}
double BrentPointDepth2DBasedML(IJZ3 &optimDepth, double ldepth, double hdepth, double *rayDirect, double *CxPd, DevicesInfo &DInfo, double *EpiLine, IlluminationFlowImages &Fimgs, FlowVect &Paraflow, int DIC_Algo, int *DenseScale, int ConvCriteria, double *Cxcenter, CPoint2 *C12_init, double neighRadius, int cwidth, int cheight, int InterpAlgo)
{
	int SearchTechnique = -1;
	const double CGOLD = 0.3819660;
	const double ZEPS = 1.0e-10;
	double tol = 1.0e-6;

	double  a, b, d = 0.0, etemp, fu, fv, fw, fx;
	double  p, q, r, tol1, tol2, u, v, w, x, xm;
	double e = 0.0;

	double ax = ldepth, bx = optimDepth.z2, cx = hdepth;
	a = (ax < cx ? ax : cx);
	b = (ax > cx ? ax : cx);
	x = w = v = bx;
	double TCost[2];
	IJZ3 hypoDx(optimDepth.i, optimDepth.j, optimDepth.z1, x, optimDepth.z3), hypoDu(optimDepth.i, optimDepth.j, optimDepth.z1, x, optimDepth.z3);
	fw = fv = fx = ComputeDepth2DBasedML(hypoDx, rayDirect, CxPd, Cxcenter, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, C12_init, neighRadius, cwidth, cheight, InterpAlgo, TCost, SearchTechnique);

	const int iterMax = 30;
	for (int iter = 0; iter<iterMax; iter++)
	{
		xm = 0.5*(a + b);
		tol2 = 2.0*(tol1 = tol*abs(x) + ZEPS);
		if (abs(x - xm) <= (tol2 - 0.5*(b - a)))
		{
			optimDepth.z1 = hypoDx.z1, optimDepth.z2 = x, optimDepth.z3 = hypoDx.z3;
			return fx;
		}

		if (abs(e) > tol1)
		{
			r = (x - w)*(fx - fv);
			q = (x - v)*(fx - fw);
			p = (x - v)*q - (x - w)*r;
			q = 2.0*(q - r);
			if (q > 0.0) p = -p;
			q = abs(q);
			etemp = e;
			e = d;
			if (abs(p) >= abs(0.5*q*etemp) || p <= q*(a - x) || p >= q*(b - x))
				d = CGOLD*(e = (x >= xm ? a - x : b - x));
			else
			{
				d = p / q;
				u = x + d;
				if (u - a < tol2 || b - u < tol2)
					d = SIGN(tol1, xm - x);
			}
		}
		else
		{
			d = CGOLD*(e = (x >= xm ? a - x : b - x));
		}
		u = (abs(d) >= tol1 ? x + d : x + SIGN(tol1, d));
		hypoDu.z2 = u;
		fu = ComputeDepth2DBasedML(hypoDu, rayDirect, CxPd, Cxcenter, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, C12_init, neighRadius, cwidth, cheight, InterpAlgo, TCost, SearchTechnique);
		if (fu <= fx)
		{
			if (u >= x) a = x; else b = x;
			v = w; w = x; x = u;
			fv = fw; fw = fx; fx = fu;
		}
		else
		{
			if (u < x) a = u; else b = u;
			if (fu <= fw || w == x)
			{
				v = w;
				w = u;
				fv = fw;
				fw = fu;
			}
			else if (fu <= fv || v == x || v == w)
			{
				v = u;
				fv = fu;
			}
		}
	}

	optimDepth.z1 = hypoDx.z1, optimDepth.z2 = x, optimDepth.z3 = hypoDx.z3;
	return fx;
}

void PointDepthCheck(CPoint3 *XYZ, IJZ3 depth, double *EpiLine, double *rayDirect, DevicesInfo &DInfo, IlluminationFlowImages &Fimgs, int *DenseScale, int DIC_Algo, int ConvCriteria, double znccThresh, int cwidth, int cheight, int InterpAlgo)
{
	const int nchannels = 1;
	int ii, clength = cwidth*cheight, nframes = Fimgs.nframes, nCams = Fimgs.nCams;
	int hsubset = 11, maxIter = 30;

	//Project to the image:
	bool GoodforTriangulation;
	double numx, numy, denum;
	CPoint2 C1pts[3], C2Pts[3];
	CPoint2 *uv = new CPoint2[nCams];
	CPoint2 *Track1 = new CPoint2[nCams];
	CPoint2 *Track2 = new CPoint2[nCams];
	CPoint3 WCi, WCf;

	WCi.x = depth.z2*rayDirect[0], WCi.y = depth.z2*rayDirect[1], WCi.z = depth.z2*rayDirect[2];
	for (ii = 0; ii < nCams; ii++)
	{
		numx = DInfo.P[12 * ii] * WCi.x + DInfo.P[12 * ii + 1] * WCi.y + DInfo.P[12 * ii + 2] * WCi.z + DInfo.P[12 * ii + 3];
		numy = DInfo.P[12 * ii + 4] * WCi.x + DInfo.P[12 * ii + 5] * WCi.y + DInfo.P[12 * ii + 6] * WCi.z + DInfo.P[12 * ii + 7];
		denum = DInfo.P[12 * ii + 8] * WCi.x + DInfo.P[12 * ii + 9] * WCi.y + DInfo.P[12 * ii + 10] * WCi.z + DInfo.P[12 * ii + 11];
		uv[ii].x = numx / denum, uv[ii].y = numy / denum;
		LensDistortion_Point(uv[ii], DInfo.K + 9 * (ii + 1), DInfo.distortion + 13 * (ii + 1));
	}

	//Minh: may need to try diffrent hsubset and pick the one that gives most consistency
	double fufv[2];
	if (TMatching(Fimgs.Para + clength, Fimgs.Para + (nframes + 1)*clength, hsubset, cwidth, cheight, cwidth, cheight, nchannels, uv[0], uv[1], DIC_Algo, ConvCriteria, znccThresh, maxIter, InterpAlgo, fufv, true) < znccThresh)
	{
		for (ii = 0; ii < 3; ii++)
		{
			XYZ[ii].x = 0.0;
			XYZ[ii].y = 0.0;
			XYZ[ii].z = 0.0;
		}
	}
	else
	{
		uv[1].x += fufv[0], uv[1].y += fufv[1];

		//Frame 1
		GoodforTriangulation = true;
		ii = 0; //C1: 2->1
		WCf.x = depth.z1*rayDirect[0], WCf.y = depth.z1*rayDirect[1], WCf.z = depth.z1*rayDirect[2];
		numx = DInfo.P[12 * ii] * WCf.x + DInfo.P[12 * ii + 1] * WCf.y + DInfo.P[12 * ii + 2] * WCf.z + DInfo.P[12 * ii + 3];
		numy = DInfo.P[12 * ii + 4] * WCf.x + DInfo.P[12 * ii + 5] * WCf.y + DInfo.P[12 * ii + 6] * WCf.z + DInfo.P[12 * ii + 7];
		denum = DInfo.P[12 * ii + 8] * WCf.x + DInfo.P[12 * ii + 9] * WCf.y + DInfo.P[12 * ii + 10] * WCf.z + DInfo.P[12 * ii + 11];

		Track1[0].x = uv[0].x, Track1[0].y = uv[0].y; //2
		Track1[1].x = numx / denum, Track1[1].y = numy / denum; //1
		LensDistortion_Point(Track1[1], DInfo.K + 9, DInfo.distortion + 13);
		hsubset = DenseScale[(int)(Track1[0].x + 0.5) + (int)(Track1[0].y + 0.5)*cwidth];
		//zncc = EpipSearchLK(Track1, EpiLine, Fimgs.Img+clength, Fimgs.Img, Fimgs.Para+clength, Fimgs.Para, cwidth, cheight, hsubset, ConvCriteria, DIC_Algo, znccThresh, InterpAlgo);
		//if(zncc <znccThresh)
		//	GoodforTriangulation = false;

		ii = 1; //C2: 2->1
		WCf.x = depth.z1*rayDirect[0], WCf.y = depth.z1*rayDirect[1], WCf.z = depth.z1*rayDirect[2];
		numx = DInfo.P[12 * ii] * WCf.x + DInfo.P[12 * ii + 1] * WCf.y + DInfo.P[12 * ii + 2] * WCf.z + DInfo.P[12 * ii + 3];
		numy = DInfo.P[12 * ii + 4] * WCf.x + DInfo.P[12 * ii + 5] * WCf.y + DInfo.P[12 * ii + 6] * WCf.z + DInfo.P[12 * ii + 7];
		denum = DInfo.P[12 * ii + 8] * WCf.x + DInfo.P[12 * ii + 9] * WCf.y + DInfo.P[12 * ii + 10] * WCf.z + DInfo.P[12 * ii + 11];

		Track2[0].x = uv[1].x, Track2[0].y = uv[1].y; //2
		Track2[1].x = numx / denum, Track2[1].y = numy / denum; //1
		LensDistortion_Point(Track2[1], DInfo.K + 18, DInfo.distortion + 26);
		hsubset = DenseScale[(int)(Track2[0].x + 0.5) + (int)(Track2[0].y + 0.5)*cwidth + 2 * clength];
		//zncc = EpipSearchLK(Track2, EpiLine+3, Fimgs.Img+4*clength, Fimgs.Img+3*clength, Fimgs.Para+4*clength, Fimgs.Para+3*clength, cwidth, cheight, hsubset, ConvCriteria, DIC_Algo, znccThresh, InterpAlgo);
		//if(zncc <znccThresh)
		//	GoodforTriangulation = false;

		if (!GoodforTriangulation)
		{
			XYZ[0].x = 0.0;
			XYZ[0].y = 0.0;
			XYZ[0].z = 0.0;
		}
		else
		{
			//Triangulate
			Undo_distortion(Track1[1], DInfo.K + 9, DInfo.distortion + 13);
			Undo_distortion(Track2[1], DInfo.K + 18, DInfo.distortion + 26);
			Stereo_Triangulation2(&Track1[1], &Track2[1], DInfo.P, DInfo.P + 12, &XYZ[0]);
		}

		//Frame 3:
		GoodforTriangulation = true;
		ii = 0; //C1: 2->3
		WCf.x = depth.z3*rayDirect[0], WCf.y = depth.z3*rayDirect[1], WCf.z = depth.z3*rayDirect[2];
		numx = DInfo.P[12 * ii] * WCf.x + DInfo.P[12 * ii + 1] * WCf.y + DInfo.P[12 * ii + 2] * WCf.z + DInfo.P[12 * ii + 3];
		numy = DInfo.P[12 * ii + 4] * WCf.x + DInfo.P[12 * ii + 5] * WCf.y + DInfo.P[12 * ii + 6] * WCf.z + DInfo.P[12 * ii + 7];
		denum = DInfo.P[12 * ii + 8] * WCf.x + DInfo.P[12 * ii + 9] * WCf.y + DInfo.P[12 * ii + 10] * WCf.z + DInfo.P[12 * ii + 11];

		Track1[0].x = uv[0].x, Track1[0].y = uv[0].y; //2
		Track1[1].x = numx / denum, Track1[1].y = numy / denum; //3
		LensDistortion_Point(Track1[1], DInfo.K + 9, DInfo.distortion + 13);
		hsubset = DenseScale[(int)(Track1[0].x + 0.5) + (int)(Track1[0].y + 0.5)*cwidth + clength];
		//zncc = EpipSearchLK(Track1, EpiLine, Fimgs.Img+clength, Fimgs.Img+2*clength, Fimgs.Para+clength, Fimgs.Para+2*clength, cwidth, cheight, hsubset, ConvCriteria, DIC_Algo, znccThresh, InterpAlgo);
		//if(zncc <znccThresh)
		//	GoodforTriangulation = false;

		ii = 1; //C2: 2->3
		WCf.x = depth.z3*rayDirect[0], WCf.y = depth.z3*rayDirect[1], WCf.z = depth.z3*rayDirect[2];
		numx = DInfo.P[12 * ii] * WCf.x + DInfo.P[12 * ii + 1] * WCf.y + DInfo.P[12 * ii + 2] * WCf.z + DInfo.P[12 * ii + 3];
		numy = DInfo.P[12 * ii + 4] * WCf.x + DInfo.P[12 * ii + 5] * WCf.y + DInfo.P[12 * ii + 6] * WCf.z + DInfo.P[12 * ii + 7];
		denum = DInfo.P[12 * ii + 8] * WCf.x + DInfo.P[12 * ii + 9] * WCf.y + DInfo.P[12 * ii + 10] * WCf.z + DInfo.P[12 * ii + 11];

		Track2[0].x = uv[1].x, Track2[0].y = uv[1].y; //2
		Track2[1].x = numx / denum, Track2[1].y = numy / denum; //3
		LensDistortion_Point(Track2[1], DInfo.K + 18, DInfo.distortion + 26);
		hsubset = DenseScale[(int)(Track2[0].x + 0.5) + (int)(Track2[0].y + 0.5)*cwidth + 3 * clength];
		//zncc = EpipSearchLK(Track2, EpiLine+3, Fimgs.Img+4*clength, Fimgs.Img+5*clength, Fimgs.Para+4*clength, Fimgs.Para+5*clength, cwidth, cheight, hsubset, ConvCriteria, DIC_Algo, znccThresh, InterpAlgo);
		//if(zncc <znccThresh)
		//	GoodforTriangulation = false;

		if (!GoodforTriangulation)
		{
			XYZ[2].x = 0.0;
			XYZ[2].y = 0.0;
			XYZ[2].z = 0.0;
		}
		else
		{
			//Triangulate
			Undo_distortion(Track1[1], DInfo.K + 9, DInfo.distortion + 13);
			Undo_distortion(Track2[1], DInfo.K + 18, DInfo.distortion + 26);
			Stereo_Triangulation2(&Track1[1], &Track2[1], DInfo.P, DInfo.P + 12, &XYZ[2]);
		}

		//C1-C2: 2
		Undo_distortion(uv[0], DInfo.K + 9, DInfo.distortion + 13);
		Undo_distortion(uv[1], DInfo.K + 18, DInfo.distortion + 26);
		Stereo_Triangulation2(&uv[0], &uv[1], DInfo.P, DInfo.P + 12, &XYZ[1]);
	}

	delete[]uv;
	delete[]Track1;
	delete[]Track2;
	return;
}
double PointDepth1DSearchTemporalLK(CPoint3 *XYZ, IJZ3 &depth, double *rayDirect, DevicesInfo &DInfo, IlluminationFlowImages &Fimgs, int *DenseScale, int DIC_Algo, int ConvCriteria, double znccThresh, double *Cxcenter, FlowVect &Paraflow, int cwidth, int cheight, double thresh, int SearchTechnique, int InterpAlgo, bool FlowRefinement, bool CorrespondenceCheck)
{
	int ii, jj, kk, clength = cwidth*cheight, nCams = Fimgs.nCams, nframes = Paraflow.nframes;

	double ProP[3] = { depth.i, depth.j, 1.0 };
	double *CxPd = new double[3 * nCams];
	double *EpiLine = new double[3 * DInfo.nCams];

	for (ii = 0; ii < nCams; ii++)
	{
		CxPd[3 * ii] = DInfo.P[12 * ii] * rayDirect[0] + DInfo.P[12 * ii + 1] * rayDirect[1] + DInfo.P[12 * ii + 2] * rayDirect[2];
		CxPd[3 * ii + 1] = DInfo.P[12 * ii + 4] * rayDirect[0] + DInfo.P[12 * ii + 5] * rayDirect[1] + DInfo.P[12 * ii + 6] * rayDirect[2];
		CxPd[3 * ii + 2] = DInfo.P[12 * ii + 8] * rayDirect[0] + DInfo.P[12 * ii + 9] * rayDirect[1] + DInfo.P[12 * ii + 10] * rayDirect[2];
		mat_mul(DInfo.FmatPC + 9 * ii, ProP, EpiLine + 3 * ii, 3, 3, 1);
	}

	int nbins = 100;
	double DepthL = 0.98*depth.z2, DepthH = 1.02*depth.z2, step = (DepthH - DepthL) / nbins, ratio = 0.1, step2, step3, cost, mincost, hypo, depth1, depth3, dtrack, depthTrack[50];
	int *ind = new int[nbins];
	double *hypoCost = new double[nbins];

	///Guess a resonable depth projection
	double Cxpnum1u, Cxpnum1v, Cxpdenum1, Cxnum1u, Cxnum1v, Cxdenum1, S1[3], S2[3], S3[3], S4[3], neighRadius = 0.03*cwidth;
	CPoint2 *Cx_init = new CPoint2[3 * nCams];
	for (ii = 0; ii < nCams; ii++)
	{
		Cxpnum1u = depth.z2*CxPd[3 * ii];
		Cxpnum1v = depth.z2*CxPd[3 * ii + 1];
		Cxpdenum1 = depth.z2*CxPd[3 * ii + 2];
		Cxnum1u = Cxpnum1u + DInfo.P[12 * ii + 3];
		Cxnum1v = Cxpnum1v + DInfo.P[12 * ii + 7];
		Cxdenum1 = Cxpdenum1 + DInfo.P[12 * ii + 11];

		Cx_init[3 * ii + 1].x = Cxnum1u / Cxdenum1;
		Cx_init[3 * ii + 1].y = Cxnum1v / Cxdenum1;

		//Get_Value_Spline(Paraflow.C21x+ii*clength, cwidth, cheight, Cx_init[3*ii+1].x, Cx_init[3*ii+1].y, S1, -1, InterpAlgo);
		//Get_Value_Spline(Paraflow.C21y+ii*clength, cwidth, cheight, Cx_init[3*ii+1].y, Cx_init[3*ii+1].y, S2, -1, InterpAlgo);
		//Get_Value_Spline(Paraflow.C23x+ii*clength, cwidth, cheight, Cx_init[3*ii+1].x, Cx_init[3*ii+1].y, S3, -1, InterpAlgo);
		//Get_Value_Spline(Paraflow.C23y+ii*clength, cwidth, cheight, Cx_init[3*ii+1].y, Cx_init[3*ii+1].y, S4, -1, InterpAlgo);

		Cx_init[3 * ii].x = Cx_init[3 * ii + 1].x + S1[0];
		Cx_init[3 * ii].y = Cx_init[3 * ii + 1].y + S2[0];
		Cx_init[3 * ii + 2].x = Cx_init[3 * ii + 1].x + S3[0];
		Cx_init[3 * ii + 2].y = Cx_init[3 * ii + 1].y + S4[0];
	}

	/*
	{
	CPoint2 *Out_Cpts = new CPoint2[3*nCams];
	bool reprojectionerr = false;
	IJZ3 td(depth.i, depth.j, depth.z1, depth.z2, depth.z3);
	{
	int nbins = 500;
	double DepthL = 0.98*depth.z2 , DepthH = 1.02*depth.z2, step = (DepthH-DepthL)/nbins, TCost[2], minCost = 9e9;

	FILE *fp = fopen("C:/temp/cost2.txt", "w+");
	for(ii=0; ii<nbins; ii++)
	{
	ComputeDepth3DBased(DepthL + step*ii, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost, false, Out_Cpts, ii, reprojectionerr);
	fprintf(fp, "%.4f %.8f %.8f", DepthL + step*ii, TCost[0], TCost[1]);
	for(jj=0; jj<nCams; jj++)
	for(kk=0; kk<3; kk++)
	fprintf(fp, " %.4f %.4f ", Out_Cpts[3*jj+kk].x, Out_Cpts[3*jj+kk].y);
	fprintf(fp, "\n");
	}
	fclose(fp);

	//BrentPointDepth3DBased(td, DepthH, DepthL, rayDirect, C1Pd, C2Pd, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, hsubset, ConvCriteria, C1center, C2center, C12_init, neighRadius, cwidth, cheight, InterpAlgo);
	}

	delete []Out_Cpts;
	}
	*/
	/*
	{
	int nbins = 500, bestID;
	double DepthL = 0.98*depth.z2 , DepthH = 1.02*depth.z2, step = (DepthH-DepthL)/nbins, TCost[2], minCost = 9e9;
	int *ind = new int[nbins];
	double *hypoCost = new double [nbins];

	double CxPd[6] = {C1Pd[0], C1Pd[1], C1Pd[2], C2Pd[0], C2Pd[1], C2Pd[2]};
	double Cxcenter[6] = {C1center[0], C1center[1], C1center[2], C2center[0], C2center[1], C2center[2]};

	clock_t start = clock();
	FILE *fp = fopen("C:/temp/cost2.txt", "w+");
	{
	for(ii=0; ii<nbins; ii++)
	{
	IJZ3 td(depth.i, depth.j, depth.z1, DepthL + step*ii, depth.z3);
	hypoCost[ii] = ComputeDepth2DBasedML(td, rayDirect, CxPd, Cxcenter, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, C12_init, neighRadius, cwidth, cheight, InterpAlgo, TCost, SearchTechnique);
	if(minCost > hypoCost[ii])
	{
	minCost = hypoCost[ii];
	bestID = ii;
	}
	fprintf(fp, "%.3f %.8f %.8f \n", DepthL + step*ii, TCost[0], TCost[1]);
	}
	}
	fclose(fp);

	int a = 0;
	}
	delete []ind;
	delete []hypoCost;
	delete []EpiLine;

	return 0.0;
	*/

	CPoint2 Out_Cpts[9];
	double optimDepthErr;
	if (SearchTechnique < 0)
	{
		double fracErr = 1.0e-3;
		if (abs(SearchTechnique) == 1)
			optimDepthErr = BrentPointDepth3DBased(depth, DepthH, DepthL, rayDirect, CxPd, DInfo, EpiLine, Fimgs, Paraflow, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, fracErr);
		else
			optimDepthErr = GoldenPointDepth3DBased(depth, DepthH, DepthL, rayDirect, CxPd, DInfo, EpiLine, Fimgs, Paraflow, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement);

		if (SearchTechnique == -1)
			optimDepthErr = BrentPointDepth2DBasedML(depth, depth.z2 + 3.0*step, depth.z2 - 3.0*step, rayDirect, CxPd, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, Cxcenter, Cx_init, neighRadius, cwidth, cheight, InterpAlgo);
		else
			GoldenPointDepth2DBasedML(depth, depth.z2 + 3.0*step, depth.z2 - 3.0*step, rayDirect, CxPd, Cxcenter, DInfo, EpiLine, Fimgs, Paraflow, DIC_Algo, DenseScale, ConvCriteria, Cx_init, neighRadius, cwidth, cheight, InterpAlgo);

	}
	else if (SearchTechnique == 1)
		optimDepthErr = BrentPointDepth3DBased(depth, DepthH, DepthL, rayDirect, CxPd, DInfo, EpiLine, Fimgs, Paraflow, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, 9.9e-7, Out_Cpts, 0);
	else if (SearchTechnique == 2)
		optimDepthErr = GoldenPointDepth3DBased(depth, DepthH, DepthL, rayDirect, CxPd, DInfo, EpiLine, Fimgs, Paraflow, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement);
	else
	{
		double TCost[2];
		for (ii = 0; ii < nbins; ii++)
			hypoCost[ii] = ComputeDepth3DBased(DepthL + step*ii, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost);

		int nMinEle;
		double *MinCost = new double[nbins];
		NonMinimalSupression1D(hypoCost, ind, nMinEle, 3, nbins);

		for (ii = 0; ii < nMinEle; ii++)
			MinCost[ii] = hypoCost[ind[ii]];
		Quick_Sort_Double(MinCost, ind, 0, nMinEle - 1);

		nbins = nbins / 2;
		step2 = ratio*step / 2;	jj = 0, kk = 0;
		for (jj = 0; jj < min(nMinEle / 2 + 1, nbins / 10); jj++)
		{
			mincost = 9.0e9;
			hypo = DepthL + step*ind[jj];
			for (ii = -nbins / 2; ii < nbins / 2; ii++)
			{
				cost = ComputeDepth3DBased(hypo + step2*ii, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost, true);
				if (cost <mincost && abs(depth1 / depth.z1) > 0.98 && abs(depth3 / depth.z3) > 0.98)
				{
					mincost = cost;
					dtrack = hypo + step2*ii;
				}
			}

			if (mincost < 9.0e9)
			{
				ind[kk] = kk;
				hypoCost[kk] = mincost;
				depthTrack[kk] = dtrack;
				kk++;
			}
		}

		if (kk < 1)
		{
			delete[]CxPd;
			delete[]Cx_init;
			delete[]ind;
			delete[]hypoCost;
			delete[]MinCost;
			return mincost;
		}

		Quick_Sort_Double(hypoCost, ind, 0, kk - 1);

		mincost = 9.0e9;
		step3 = ratio*step2;
		for (jj = 0; jj < kk; jj++)
		{
			hypo = depthTrack[ind[jj]];
			for (ii = -nbins / 2; ii < nbins / 2; ii++)
			{
				cost = ComputeDepth3DBased(hypo + step3*ii, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth1, depth3, TCost, true);
				if (cost <mincost && abs(depth1 / depth.z1) > 0.98 && abs(depth3 / depth.z3) > 0.98)
				{
					mincost = cost;
					dtrack = hypo + step3*ii;
				}
			}
		}
		delete[]MinCost;

		depth.z2 = dtrack;
		optimDepthErr = ComputeDepth3DBased(depth.z2, rayDirect, CxPd, DInfo, EpiLine, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, Cx_init, neighRadius, cwidth, cheight, InterpAlgo, FlowRefinement, depth.z1, depth.z3, TCost, true);
	}

	//Consistency check
	IJZ3 t(depth.i, depth.j, depth.z1, depth.z2, depth.z3);
	CorrespondenceCheck = true;
	if (CorrespondenceCheck)
		PointDepthCheck(XYZ, t, EpiLine, rayDirect, DInfo, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, cwidth, cheight, InterpAlgo);
	else
	{
		XYZ[0].x = depth.z1*(DInfo.iK[0] * depth.i + DInfo.iK[1] * depth.j + DInfo.iK[2]);
		XYZ[0].y = depth.z1*(DInfo.iK[4] * depth.j + DInfo.iK[5]);
		XYZ[0].z = depth.z1;

		XYZ[1].x = depth.z2*(DInfo.iK[0] * depth.i + DInfo.iK[1] * depth.j + DInfo.iK[2]);
		XYZ[1].y = depth.z2*(DInfo.iK[4] * depth.j + DInfo.iK[5]);
		XYZ[1].z = depth.z2;

		XYZ[2].x = depth.z3*(DInfo.iK[0] * depth.i + DInfo.iK[1] * depth.j + DInfo.iK[2]);
		XYZ[2].y = depth.z3*(DInfo.iK[4] * depth.j + DInfo.iK[5]);
		XYZ[2].z = depth.z3;
	}

	delete[]CxPd;
	delete[]Cx_init;
	delete[]ind;
	delete[]hypoCost;
	delete[]EpiLine;

	return optimDepthErr;
}
void DepthFlow(CPoint3 *XYZ, IJZ3 *depth, DevicesInfo &DInfo, FlowVect &Paraflow, IlluminationFlowImages &Fimgs, int *DenseScale, int DIC_Algo, int ConvCriteria, double znccThresh, int cwidth, int cheight, int npts, int SearchTechnique, int InterpAlgo)
{
	double ProP[3]; ProP[2] = 1.0;
	double *rayDirect = new double[3 * npts];
	for (int ii = 0; ii<npts; ii++)
	{
		rayDirect[3 * ii] = DInfo.iK[0] * depth[ii].i + DInfo.iK[1] * depth[ii].j + DInfo.iK[2];
		rayDirect[3 * ii + 1] = DInfo.iK[4] * depth[ii].j + DInfo.iK[5];
		rayDirect[3 * ii + 2] = 1.0;
	}

	double Cxcenter[9] = { DInfo.RTx1[3], DInfo.RTx1[7], DInfo.RTx1[11], DInfo.RTx1[12 + 3], DInfo.RTx1[12 + 7], DInfo.RTx1[12 + 11], DInfo.RTx1[24 + 3], DInfo.RTx1[24 + 7], DInfo.RTx1[24 + 11] };
	double OneDthresh = 20.0;

	int currentWorkerThread;
	int maxThreads = omp_get_max_threads()>MAXTHREADS ? MAXTHREADS : omp_get_max_threads();
	omp_set_num_threads(maxThreads);
	double start = omp_get_wtime();
	int percent = maxThreads;

	bool FlowRefinement = true, CorrespondenceCheck = true;
#pragma omp parallel  private(currentWorkerThread) 
	{
#pragma omp for 
		for (int ii = 0; ii < npts; ii++)
		{
			currentWorkerThread = omp_get_thread_num();
			if (currentWorkerThread == 0)
			{
				if ((ii*maxThreads * 100 / npts - percent) > 0)
				{
					percent += maxThreads;
					double elapsed = omp_get_wtime() - start;
					cout << "%" << ii*maxThreads * 100 / npts << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed*(1.0*npts / ii / maxThreads - 1.0) << endl;
				}
			}
			PointDepth1DSearchTemporalLK(XYZ + 3 * ii, depth[ii], rayDirect + 3 * ii, DInfo, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, Cxcenter, Paraflow, cwidth, cheight, OneDthresh, SearchTechnique, InterpAlgo, FlowRefinement, CorrespondenceCheck);
		}
	}

	cout << "Total time: " << setw(2) << omp_get_wtime() - start << endl;

	delete[]rayDirect;
	return;
}
void FlowDepthOptimization(DevicesInfo &DInfo, CPoint2 *Ccorners, CPoint2 *Pcorners, int Cnpts, int *triangleList, int ntriangles, double step, IlluminationFlowImages &Fimgs, FlowVect &flow, FlowVect &Paraflow, int *DenseScale, int DIC_Algo, int ConvCriteria, double znccThresh, int width, int height, int pwidth, int pheight, int SearchTechnique, int InterpAlgo, int frameID, char *ResultPATH)
{
	int ii, jj, kk, ll;
	int nCams = Fimgs.nCams, nframes = Fimgs.nframes;
	char Fname[100];

	//Set up the triangulation
	CPoint2 *apts = new CPoint2[nCams + 1];
	CPoint3 WC, WC2, WC3;
	double *aPmat = new double[12 * (nCams + 1)];
	aPmat[0] = DInfo.K[0], aPmat[1] = DInfo.K[1], aPmat[2] = DInfo.K[2], aPmat[3] = 0.0;
	aPmat[4] = DInfo.K[3], aPmat[5] = DInfo.K[4], aPmat[6] = DInfo.K[5], aPmat[7] = 0.0;
	aPmat[8] = DInfo.K[6], aPmat[9] = DInfo.K[7], aPmat[10] = DInfo.K[8], aPmat[11] = 0.0;
	for (ii = 0; ii < 12 * nCams; ii++)
		aPmat[12 + ii] = DInfo.P[ii];

	/// Compute rough estimate of depth via patch matching:
	double *sDepth = new double[3 * Cnpts];
	double *dDepth = new double[5 * width*height];

	//FILE *fp = fopen("C:/temp/sdepth.txt", "w+");
	for (jj = 0; jj < nframes; jj++)
	{
		for (ii = 0; ii < Cnpts; ii++)
		{
			apts[0].x = Pcorners[ii].x, apts[0].y = Pcorners[ii].y;
			for (kk = 0; kk < nCams; kk++)
				apts[kk + 1].x = Ccorners[ii + (3 * kk + jj)*Cnpts].x, apts[kk + 1].y = Ccorners[ii + (3 * kk + jj)*Cnpts].y;

			for (kk = 0; kk < nCams + 1; kk++)
				Undo_distortion(apts[kk], DInfo.K + 9 * kk, DInfo.distortion + 13 * kk);
			//NviewTriangulation(apts, aPmat, &WC, nCams+1);
			NviewTriangulation(apts, aPmat, &WC, nCams);

			sDepth[ii + jj*Cnpts] = WC.z;
			//fprintf(fp, "%d %.3f\n", ii, WC.z);
		}
	}
	//fclose(fp);

	double AA[4 * 3], BB[4];
	int maxX, minX, maxY, minY, pointCount = 0;
	double dCoeff[9];
	CPoint2 triCoor[3];
	double *rangeX = new double[(int)(1.0*width / step + 0.5)];
	double *rangeY = new double[(int)(1.0*height / step + 0.5)];
	for (kk = 0; kk < ntriangles; kk++)
	{
		triCoor[0].x = Pcorners[triangleList[3 * kk]].x, triCoor[0].y = Pcorners[triangleList[3 * kk]].y;
		triCoor[1].x = Pcorners[triangleList[3 * kk + 1]].x, triCoor[1].y = Pcorners[triangleList[3 * kk + 1]].y;
		triCoor[2].x = Pcorners[triangleList[3 * kk + 2]].x, triCoor[2].y = Pcorners[triangleList[3 * kk + 2]].y;

		for (ll = 0; ll<nframes; ll++)
		{
			AA[0] = triCoor[0].x, AA[1] = triCoor[0].y, AA[2] = sDepth[triangleList[3 * kk] + ll*Cnpts];
			AA[3] = triCoor[1].x, AA[4] = triCoor[1].y, AA[5] = sDepth[triangleList[3 * kk + 1] + ll*Cnpts];
			AA[6] = triCoor[2].x, AA[7] = triCoor[2].y, AA[8] = sDepth[triangleList[3 * kk + 2] + ll*Cnpts];
			BB[0] = 1.0, BB[1] = 1.0, BB[2] = 1.0, BB[3] = 1.0;
			LS_Solution_Double(AA, BB, 3, 3);
			dCoeff[3 * ll] = BB[0], dCoeff[3 * ll + 1] = BB[1], dCoeff[3 * ll + 2] = BB[2];
		}

		maxX = MyFtoI(max(max(triCoor[0].x, triCoor[1].x), triCoor[2].x));
		minX = MyFtoI(min(min(triCoor[0].x, triCoor[1].x), triCoor[2].x));
		maxY = MyFtoI(max(max(triCoor[0].y, triCoor[1].y), triCoor[2].y));
		minY = MyFtoI(min(min(triCoor[0].y, triCoor[1].y), triCoor[2].y));

		maxX = maxX>pwidth - 5 ? maxX : maxX + 2;
		minX = minX < 5 ? minX : minX - 2;
		maxY = maxY > pheight - 5 ? maxY : maxY + 2;
		minY = minY < 5 ? minY : minY - 2;

		int nrangeX = (int)(1.0*(maxX - minX) / step + 0.5), nrangeY = (int)(1.0*(maxY - minY) / step + 0.5);
		double stepX = 1.0*(maxX - minX) / nrangeX, stepY = 1.0*(maxY - minY) / nrangeY;

		for (jj = 0; jj < nrangeY; jj++)
		{
			for (ii = 0; ii < nrangeX; ii++)
			{
				double proPointX = stepX*ii + minX;
				double proPointY = stepY*jj + minY;
				if (in_polygon(proPointX, proPointY, triCoor, 3))
				{
					dDepth[5 * pointCount] = (float)(1.0*proPointX);
					dDepth[5 * pointCount + 1] = (float)(1.0*proPointY);
					for (ll = 0; ll < nframes; ll++)
						dDepth[5 * pointCount + 2 + ll] = (float)((1.0 - dCoeff[3 * ll] * dDepth[5 * pointCount] - dCoeff[3 * ll + 1] * dDepth[5 * pointCount + 1]) / dCoeff[3 * ll + 2]);
					pointCount++;
				}
			}
		}
	}
	delete[]rangeX;
	delete[]rangeY;

	IJZ3 *depth = new IJZ3[pointCount];
	for (ii = 0; ii < pointCount; ii++)
	{
		depth[ii].i = dDepth[5 * ii];
		depth[ii].j = dDepth[5 * ii + 1];
		depth[ii].z1 = dDepth[5 * ii + 2];
		depth[ii].z2 = dDepth[5 * ii + 3];
		depth[ii].z3 = dDepth[5 * ii + 4];
	}

	double X, Y, Z;
	sprintf(Fname, "%s/Init_%05d.xyz", ResultPATH, frameID); FILE *fp = fopen(Fname, "w+");
	sprintf(Fname, "%s/Init_%05d.xyz", ResultPATH, frameID + 1); FILE *fp2 = fopen(Fname, "w+");
	sprintf(Fname, "%s/Init_%05d.xyz", ResultPATH, frameID + 2); FILE *fp3 = fopen(Fname, "w+");
	for (ii = 0; ii < pointCount; ii++)
	{
		X = depth[ii].z1*(DInfo.iK[0] * depth[ii].i + DInfo.iK[1] * depth[ii].j + DInfo.iK[2]);
		Y = depth[ii].z1*(DInfo.iK[4] * depth[ii].j + DInfo.iK[5]);
		Z = depth[ii].z1;
		fprintf(fp, "%.4f %.4f %4f\n", X, Y, Z);

		X = depth[ii].z2*(DInfo.iK[0] * depth[ii].i + DInfo.iK[1] * depth[ii].j + DInfo.iK[2]);
		Y = depth[ii].z2*(DInfo.iK[4] * depth[ii].j + DInfo.iK[5]);
		Z = depth[ii].z2;
		fprintf(fp2, "%.4f %.4f %4f\n", X, Y, Z);

		X = depth[ii].z3*(DInfo.iK[0] * depth[ii].i + DInfo.iK[1] * depth[ii].j + DInfo.iK[2]);
		Y = depth[ii].z3*(DInfo.iK[4] * depth[ii].j + DInfo.iK[5]);
		Z = depth[ii].z3;
		fprintf(fp3, "%.4f %.4f %4f\n", X, Y, Z);
	}
	fclose(fp), fclose(fp2), fclose(fp3);
	/*
	sprintf(Fname, "%s/depthInit_%03d.xyz", PATH, frameID);	fp = fopen(Fname, "w+");
	for(ii=0; ii<pointCount; ii++)
	fprintf(fp, "%.2f %.2f %6f %.6f %6f\n", depth[ii].i, depth[ii].j, depth[ii].z1, depth[ii].z2, depth[ii].z3);
	fclose(fp);
	*/

	CPoint3 *XYZ = new CPoint3[3 * pointCount];

	DepthFlow(XYZ, depth, DInfo, Paraflow, Fimgs, DenseScale, DIC_Algo, ConvCriteria, znccThresh, width, height, pointCount, SearchTechnique, InterpAlgo);
	/*
	sprintf(Fname, "%s/depthOptims_%03d.xyz", PATH, frameID);	fp = fopen(Fname, "w+");
	for(ii=0; ii<pointCount; ii++)
	fprintf(fp, "%.2f %.2f %6f %.6f %6f\n", depth[ii].i, depth[ii].j, depth[ii].z1, depth[ii].z2, depth[ii].z3);
	fclose(fp);
	*/
	sprintf(Fname, "%s/Final_%05d.xyz", ResultPATH, frameID);	fp = fopen(Fname, "w+");
	sprintf(Fname, "%s/Final_%05d.xyz", ResultPATH, frameID + 1); fp2 = fopen(Fname, "w+");
	sprintf(Fname, "%s/Final_%05d.xyz", ResultPATH, frameID + 2); fp3 = fopen(Fname, "w+");
	for (ii = 0; ii < pointCount; ii++)
	{
		X = depth[ii].z1*(DInfo.iK[0] * depth[ii].i + DInfo.iK[1] * depth[ii].j + DInfo.iK[2]);
		Y = depth[ii].z1*(DInfo.iK[4] * depth[ii].j + DInfo.iK[5]);
		Z = depth[ii].z1;
		fprintf(fp, "%.4f %.4f %4f\n", X, Y, Z);

		X = depth[ii].z2*(DInfo.iK[0] * depth[ii].i + DInfo.iK[1] * depth[ii].j + DInfo.iK[2]);
		Y = depth[ii].z2*(DInfo.iK[4] * depth[ii].j + DInfo.iK[5]);
		Z = depth[ii].z2;
		fprintf(fp2, "%.4f %.4f %4f\n", X, Y, Z);

		X = depth[ii].z3*(DInfo.iK[0] * depth[ii].i + DInfo.iK[1] * depth[ii].j + DInfo.iK[2]);
		Y = depth[ii].z3*(DInfo.iK[4] * depth[ii].j + DInfo.iK[5]);
		Z = depth[ii].z3;
		fprintf(fp3, "%.4f %.4f %4f\n", X, Y, Z);
	}
	fclose(fp), fclose(fp2), fclose(fp3);

	sprintf(Fname, "%s/Final2_%05d.xyz", ResultPATH, frameID);	fp = fopen(Fname, "w+");
	sprintf(Fname, "%s/Final2_%05d.xyz", ResultPATH, frameID + 1); fp2 = fopen(Fname, "w+");
	sprintf(Fname, "%s/Final2_%05d.xyz", ResultPATH, frameID + 2); fp3 = fopen(Fname, "w+");
	for (ii = 0; ii < pointCount; ii++)
	{
		X = XYZ[3 * ii].x, Y = XYZ[3 * ii].y, Z = XYZ[3 * ii].z;
		fprintf(fp, "%.4f %.4f %4f\n", X, Y, Z);

		X = XYZ[3 * ii + 1].x, Y = XYZ[3 * ii + 1].y, Z = XYZ[3 * ii + 1].z;
		fprintf(fp2, "%.4f %.4f %4f\n", X, Y, Z);

		X = XYZ[3 * ii + 2].x, Y = XYZ[3 * ii + 2].y, Z = XYZ[3 * ii + 2].z;
		fprintf(fp3, "%.4f %.4f %4f\n", X, Y, Z);
	}
	fclose(fp), fclose(fp2), fclose(fp3);

	delete[]apts;
	delete[]aPmat;
	delete[]XYZ;
	delete[]depth;
	delete[]sDepth;
	delete[]dDepth;

	return;
}
bool SetUpDevicesInfo(DevicesInfo &DInfo, char *PATH)
{
	/// Setup the camera projector parameters
	int ii, jj, kk, nCams = DInfo.nCams, nPros = DInfo.nPros;

	double *tparams = new double[12 * (nCams + nPros) + 6 * nCams + 6 * nPros];
	char Fname[100];  sprintf(Fname, "%s/%dP_%dC.txt", PATH, nPros, nCams);
	FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load: " << Fname << endl;
		return false;
	}
	else
	{
		for (ii = 0; ii < 12 * (nCams + nPros) + 6 * nCams + 6 * nPros; ii++)
			fscanf(fp, "%lf", &tparams[ii]);
		fclose(fp);
	}

	for (ii = 0; ii < nCams + nPros; ii++)
	{
		DInfo.K[9 * ii] = tparams[0 + 12 * ii], DInfo.K[1 + 9 * ii] = tparams[2 + 12 * ii], DInfo.K[2 + 9 * ii] = tparams[3 + 12 * ii], DInfo.K[3 + 9 * ii] = 0.0, DInfo.K[4 + 9 * ii] = tparams[1 + 12 * ii], DInfo.K[5 + 9 * ii] = tparams[4 + 12 * ii], DInfo.K[6 + 9 * ii] = 0.0, DInfo.K[7 + 9 * ii] = 0.0, DInfo.K[8 + 9 * ii] = 1.0;
		DInfo.distortion[0 + 13 * ii] = tparams[5 + 12 * ii], DInfo.distortion[1 + 13 * ii] = tparams[6 + 12 * ii], DInfo.distortion[2 + 13 * ii] = tparams[7 + 12 * ii], DInfo.distortion[3 + 13 * ii] = 0.0, DInfo.distortion[4 + 13 * ii] = 0.0;
		DInfo.distortion[5 + 13 * ii] = tparams[8 + 12 * ii], DInfo.distortion[6 + 13 * ii] = tparams[9 + 12 * ii], DInfo.distortion[7 + 13 * ii] = 0.0, DInfo.distortion[8 + 13 * ii] = 0.0;
		DInfo.distortion[9 + 13 * ii] = tparams[10 + 12 * ii], DInfo.distortion[10 + 13 * ii] = tparams[11 + 12 * ii], DInfo.distortion[11 + 13 * ii] = 0.0, DInfo.distortion[12 + 13 * ii] = 0.0;

		mat_invert(DInfo.K + 9 * ii, DInfo.iK + 9 * ii);
	}

	double rt1x[6], R1x[9], T1x[3], Rx1[9], Tx1[3], RT1x[12], Emat1x[9], tx1x[9], tmat[9], tmat2[9];
	for (ii = 1; ii < nPros + nCams; ii++)
	{
		rt1x[0] = tparams[12 * (nCams + nPros) + 6 * (ii - 1)], rt1x[1] = tparams[12 * (nCams + nPros) + 1 + 6 * (ii - 1)], rt1x[2] = tparams[12 * (nCams + nPros) + 2 + 6 * (ii - 1)];
		rt1x[3] = tparams[12 * (nCams + nPros) + 3 + 6 * (ii - 1)], rt1x[4] = tparams[12 * (nCams + nPros) + 4 + 6 * (ii - 1)], rt1x[5] = tparams[12 * (nCams + nPros) + 5 + 6 * (ii - 1)];

		Rodrigues_trans(rt1x, R1x, true);

		DInfo.RT1x[12 * (ii - 1)] = R1x[0], DInfo.RT1x[12 * (ii - 1) + 1] = R1x[1], DInfo.RT1x[12 * (ii - 1) + 2] = R1x[2], DInfo.RT1x[12 * (ii - 1) + 3] = rt1x[3];
		DInfo.RT1x[12 * (ii - 1) + 4] = R1x[3], DInfo.RT1x[12 * (ii - 1) + 5] = R1x[4], DInfo.RT1x[12 * (ii - 1) + 6] = R1x[5], DInfo.RT1x[12 * (ii - 1) + 7] = rt1x[4];
		DInfo.RT1x[12 * (ii - 1) + 8] = R1x[6], DInfo.RT1x[12 * (ii - 1) + 9] = R1x[7], DInfo.RT1x[12 * (ii - 1) + 10] = R1x[8], DInfo.RT1x[12 * (ii - 1) + 11] = rt1x[5];

		mat_transpose(R1x, Rx1, 3, 3);
		mat_mul(Rx1, rt1x + 3, Tx1, 3, 3, 1); Tx1[0] = -Tx1[0], Tx1[1] = -Tx1[1], Tx1[2] = -Tx1[2];
		DInfo.RTx1[12 * (ii - 1)] = Rx1[0], DInfo.RTx1[12 * (ii - 1) + 1] = Rx1[1], DInfo.RTx1[12 * (ii - 1) + 2] = Rx1[2], DInfo.RTx1[12 * (ii - 1) + 3] = Tx1[0];
		DInfo.RTx1[12 * (ii - 1) + 4] = Rx1[3], DInfo.RTx1[12 * (ii - 1) + 5] = Rx1[4], DInfo.RTx1[12 * (ii - 1) + 6] = Rx1[5], DInfo.RTx1[12 * (ii - 1) + 7] = Tx1[1];
		DInfo.RTx1[12 * (ii - 1) + 8] = Rx1[6], DInfo.RTx1[12 * (ii - 1) + 9] = Rx1[7], DInfo.RTx1[12 * (ii - 1) + 10] = Rx1[8], DInfo.RTx1[12 * (ii - 1) + 11] = Tx1[2];

		//Compute projection matrix
		for (jj = 0; jj < 3; jj++)
		{
			for (kk = 0; kk < 3; kk++)
				RT1x[kk + jj * 4] = R1x[kk + jj * 3];
			RT1x[3 + jj * 4] = rt1x[jj + 3];
		}
		mat_mul(DInfo.K + 9 * ii, RT1x, DInfo.P + 12 * (ii - 1), 3, 3, 4);
	}

	//Compute Fmatrix for each projector to cameras
	//Pro 1 to cameras
	for (ii = 0; ii < nCams; ii++)
	{
		R1x[0] = DInfo.RT1x[12 * (ii + nPros - 1)], R1x[1] = DInfo.RT1x[12 * (ii + nPros - 1) + 1], R1x[2] = DInfo.RT1x[12 * (ii + nPros - 1) + 2], T1x[0] = DInfo.RT1x[12 * (ii + nPros - 1) + 3];
		R1x[3] = DInfo.RT1x[12 * (ii + nPros - 1) + 4], R1x[4] = DInfo.RT1x[12 * (ii + nPros - 1) + 5], R1x[5] = DInfo.RT1x[12 * (ii + nPros - 1) + 6], T1x[1] = DInfo.RT1x[12 * (ii + nPros - 1) + 7];
		R1x[6] = DInfo.RT1x[12 * (ii + nPros - 1) + 8], R1x[7] = DInfo.RT1x[12 * (ii + nPros - 1) + 9], R1x[8] = DInfo.RT1x[12 * (ii + nPros - 1) + 10], T1x[2] = DInfo.RT1x[12 * (ii + nPros - 1) + 11];

		tx1x[0] = 0.0, tx1x[1] = -T1x[2], tx1x[2] = T1x[1];
		tx1x[3] = T1x[2], tx1x[4] = 0.0, tx1x[5] = -T1x[0];
		tx1x[6] = -T1x[1], tx1x[7] = T1x[0], tx1x[8] = 0.0;

		mat_mul(tx1x, R1x, Emat1x, 3, 3, 3);
		mat_transpose(DInfo.iK + 9 * (ii + nPros), tmat, 3, 3);
		mat_mul(tmat, Emat1x, tmat2, 3, 3, 3);
		mat_mul(tmat2, DInfo.iK, DInfo.FmatPC + 9 * ii, 3, 3, 3);
	}

	//Pro i to cameras
	double RT1A[16], iRT1A[16], RT1B[16], RTAB[16], RAB[9], txAB[9], EmatAB[9];
	//RT23 = RT13*inv(RT12);
	for (jj = 1; jj < nPros; jj++)
	{
		RT1A[0] = DInfo.RT1x[12 * (jj - 1)], RT1A[1] = DInfo.RT1x[12 * (jj - 1) + 1], RT1A[2] = DInfo.RT1x[12 * (jj - 1) + 2], RT1A[3] = DInfo.RT1x[12 * (jj - 1) + 3];
		RT1A[4] = DInfo.RT1x[12 * (jj - 1) + 4], RT1A[5] = DInfo.RT1x[12 * (jj - 1) + 5], RT1A[6] = DInfo.RT1x[12 * (jj - 1) + 6], RT1A[7] = DInfo.RT1x[12 * (jj - 1) + 7];
		RT1A[8] = DInfo.RT1x[12 * (jj - 1) + 8], RT1A[9] = DInfo.RT1x[12 * (jj - 1) + 9], RT1A[10] = DInfo.RT1x[12 * (jj - 1) + 10], RT1A[11] = DInfo.RT1x[12 * (jj - 1) + 11];
		RT1A[12] = 0.0, RT1A[13] = 0.0, RT1A[14] = 0.0, RT1A[15] = 1.0;
		mat_invert(RT1A, iRT1A, 4);

		for (ii = 0; ii < nCams; ii++)
		{
			RT1B[0] = DInfo.RT1x[12 * (ii + nPros - 1)], RT1B[1] = DInfo.RT1x[12 * (ii + nPros - 1) + 1], RT1B[2] = DInfo.RT1x[12 * (ii + nPros - 1) + 2], RT1B[3] = DInfo.RT1x[12 * (ii + nPros - 1) + 3];
			RT1B[4] = DInfo.RT1x[12 * (ii + nPros - 1) + 4], RT1B[5] = DInfo.RT1x[12 * (ii + nPros - 1) + 5], RT1B[6] = DInfo.RT1x[12 * (ii + nPros - 1) + 6], RT1B[7] = DInfo.RT1x[12 * (ii + nPros - 1) + 7];
			RT1B[8] = DInfo.RT1x[12 * (ii + nPros - 1) + 8], RT1B[9] = DInfo.RT1x[12 * (ii + nPros - 1) + 9], RT1B[10] = DInfo.RT1x[12 * (ii + nPros - 1) + 10], RT1B[11] = DInfo.RT1x[12 * (ii + nPros - 1) + 11];
			RT1B[12] = 0.0, RT1B[13] = 0.0, RT1B[14] = 0.0, RT1B[15] = 1.0;

			mat_mul(RT1B, iRT1A, RTAB, 4, 4, 4);

			RAB[0] = RTAB[0], RAB[1] = RTAB[1], RAB[2] = RTAB[2],
				RAB[3] = RTAB[4], RAB[4] = RTAB[5], RAB[5] = RTAB[6],
				RAB[6] = RTAB[8], RAB[7] = RTAB[9], RAB[8] = RTAB[10];

			txAB[0] = 0.0, txAB[1] = -RTAB[11], txAB[2] = RTAB[7],
				txAB[3] = RTAB[11], txAB[4] = 0.0, txAB[5] = -RTAB[3],
				txAB[6] = -RTAB[7], txAB[7] = RTAB[3], txAB[8] = 0.0;

			mat_mul(txAB, RAB, EmatAB, 3, 3, 3);
			mat_transpose(DInfo.iK + 9 * (ii + nPros), tmat, 3, 3);
			mat_mul(tmat, EmatAB, tmat2, 3, 3, 3);
			mat_mul(tmat2, DInfo.iK + 9 * jj, DInfo.FmatPC + 9 * (jj*nCams + ii), 3, 3, 3);
		}
	}

	delete[]tparams;
	return true;
}

void FlowFieldsWarpring(float *TextureFlow, float *IlluminationFlow, double *TextureImagePara, double *IlluminationImagePara, int width, int height, int InterpAlgo, char *Fname)
{
	int ii, jj, length = width*height;
	double S0[3], S1[3], intensity;

	unsigned char *Wimg = new unsigned char[length];
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			Get_Value_Spline(IlluminationImagePara, width, height, 1.0*ii + IlluminationFlow[ii + jj*width], 1.0*jj + IlluminationFlow[ii + jj*width + length], S0, -1, InterpAlgo);
			Get_Value_Spline(TextureImagePara, width, height, 1.0*ii + TextureFlow[ii + jj*width], 1.0*jj + TextureFlow[ii + jj*width + length], S1, -1, InterpAlgo);

			if (abs(IlluminationFlow[ii + jj*width]) < 0.0001 && abs(IlluminationFlow[ii + jj*width + length]) < 0.0001)
				S0[0] = 255.0;
			if (abs(TextureFlow[ii + jj*width]) < 0.0001 && abs(TextureFlow[ii + jj*width + length]) < 0.0001)
				S1[0] = 255.0;

			if (S0[0] > 255.0)
				S0[0] = 255.0;
			else if (S0[0]<0.0)
				S0[0] = 0.0;

			if (S1[0] >255.0)
				S1[0] = 255.0;
			else if (S1[0] < 0.0)
				S1[0] = 0.0;

			intensity = S0[0] / 255.0*S1[0];
			if (intensity > 255.0)
				intensity = 255.0;
			if (intensity < 0.0)
				intensity = 0.0;
			Wimg[ii + jj*width] = (unsigned char)(int)(intensity + 0.5);
		}
	}

	/*unsigned char *WT = new unsigned char [length];
	unsigned char *WI = new unsigned char [length];
	for(jj=0; jj<height; jj++)
	{
	for(ii=0; ii<width; ii++)
	{
	Get_Value_Spline(IlluminationImagePara, width, height, 1.0*ii+IlluminationFlow[ii+jj*width], 1.0*jj+IlluminationFlow[ii+jj*width+length], S0, -1, InterpAlgo);
	Get_Value_Spline(TextureImagePara, width, height, 1.0*ii+TextureFlow[ii+jj*width], 1.0*jj+TextureFlow[ii+jj*width+length], S1, -1, InterpAlgo);

	if(abs(IlluminationFlow[ii+jj*width]) < 0.0001 && abs(IlluminationFlow[ii+jj*width+length])<0.0001)
	S0[0] = 255.0;
	if(abs(TextureFlow[ii+jj*width]) < 0.0001 && abs(TextureFlow[ii+jj*width+length])<0.0001)
	S1[0] = 255.0;

	if(S0[0] >255.0)
	S0[0] = 255.0;
	else if(S0[0]<0.0)
	S0[0]  = 0.0;

	if(S1[0] >255.0)
	S1[0] = 255.0;
	else if(S1[0]<0.0)
	S1[0]  = 0.0;

	WI[ii+jj*width] = (unsigned char) (int)(S0[0]+0.5);
	WT[ii+jj*width] = (unsigned char) (int)(S1[0]+0.5);
	}
	}
	SaveDataToImage("C:/temp/WI.png", WI, width, height);
	SaveDataToImage("C:/temp/WT.png", WT, width, height);
	*/
	if (Fname != NULL)
		SaveDataToImage(Fname, Wimg, width, height);
	delete[]Wimg;

	return;
}
int DetermineTextureImageRegion(IlluminationFlowImages &Fimgs, LKParameters LKArg, float *Oflow, bool *mask)
{
	int ii, jj, kk, stepIJ = 4;
	int width = Fimgs.width, height = Fimgs.height, length = width*height, nchannels = Fimgs.nchannels, nCams = Fimgs.nCams, hsubset = LKArg.hsubset;
	CPoint2 From, To;
	double score;

	double *RefPatch = new double[(2 * LKArg.hsubset + 1)*(2 * LKArg.hsubset + 1)*nchannels];
	double *ZNCCStorage = new double[2 * (2 * LKArg.hsubset + 1)*(2 * LKArg.hsubset + 1)*nchannels];

	unsigned char *I2 = new unsigned char[length];
	unsigned char *wImg = new unsigned char[length];

	IplImage *view = 0;
	view = cvLoadImage("C:/temp/clothSim/Image/C1_00002.png", nchannels == 1 ? 0 : 1);
	if (view == NULL)
	{
		cout << "Cannot load: " << endl;
		return 2;
	}
	for (kk = 0; kk < nchannels; kk++)
		for (jj = 0; jj < height; jj++)
			for (ii = 0; ii < width; ii++)
				I2[ii + jj*width] = view->imageData[nchannels*ii + (height - 1 - jj)*nchannels*width + kk];
	cvReleaseImage(&view);

	WarpImageFlow(Oflow, wImg, I2, width, height, nchannels, LKArg.InterpAlgo, false);
	SaveDataToImage("C:/temp/w21.png", wImg, width, height);

	LKArg.DIC_Algo = 3, LKArg.IterMax = 15, LKArg.hsubset = 6;
	LKArg.ZNCCThreshold -= 0.05;

	cout << "Computing mask for texture (occluded) regions" << endl;
	int percent = 2, increP = 2;
	double start = omp_get_wtime();

	int *Imask = new int[length];
	for (kk = 0; kk < nCams; kk++)
	{
		for (jj = 0; jj < height; jj += stepIJ)
		{
			for (ii = 0; ii < width; ii += stepIJ)
			{
				if ((ii + jj*width) * 100 / length / nCams > percent)
				{
					double elapsed = omp_get_wtime() - start;
					cout << "%" << (ii + jj*width) * 100 / length / nCams << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / percent*(100.0 - percent) << endl;
					percent += increP;
				}

				Imask[ii + jj*width] = 0;
				if (abs(Oflow[ii + jj*width + 2 * kk*length]) < 0.01 && abs(Oflow[ii + jj*width + (2 * kk + 1)*length]) < 0.01)  //No flow
					Imask[ii + jj*width] = 255;
				else
				{
					From.x = ii, From.y = jj;
					To.x = Oflow[ii + jj*width + 2 * kk*length] + ii, To.y = Oflow[ii + jj*width + (2 * kk + 1)*length] + jj;
					if (From.x < hsubset || From.x > width - hsubset || From.y<hsubset || From.y >height - hsubset)
					{
						Imask[ii + jj*width] = 255;
						continue;
					}
					if (To.x < hsubset || To.x > width - hsubset || To.y<hsubset || To.y >height - hsubset)
					{
						Imask[ii + jj*width] = 255;
						continue;
					}
					score = SearchLK(From, To, Fimgs.Para + nchannels * 2 * kk*length, Fimgs.Para + nchannels*(2 * kk + 1)*length, nchannels, width, height, width, height, LKArg, RefPatch, ZNCCStorage);
					if (score < LKArg.ZNCCThreshold)
						Imask[ii + jj*width] = 255;
				}
			}
		}

		int idx, idy;
		double f00, f01, f10, f11;
		for (jj = hsubset; jj < height - hsubset; jj++)
		{
			for (ii = hsubset; ii < width - hsubset; ii++)
			{
				idx = ii / stepIJ, idy = jj / stepIJ;

				f00 = 1.0*Imask[idx*stepIJ + (idy*stepIJ)*width];
				f01 = 1.0*Imask[((idx + 1)*stepIJ) + (idy*stepIJ)*width];
				f10 = 1.0*Imask[(idx*stepIJ) + ((idy + 1)*stepIJ)*width];
				f11 = 1.0*Imask[((idx + 1)*stepIJ) + ((idy + 1)*stepIJ)*width];

				int res = MyFtoI((f01 - f00)*(ii - (idx*stepIJ)) / stepIJ + (f10 - f00)*(jj - (idy*stepIJ)) / stepIJ + (f11 - f01 - f10 + f00)*(ii - (idx*stepIJ)) / stepIJ*(jj - (idy*stepIJ)) / stepIJ + f00);
				if (res>127)
					mask[ii + jj*width + kk*length] = true;
				else
					mask[ii + jj*width + kk*length] = false;
			}
		}
	}
	delete[]Imask;
	delete[]RefPatch;
	delete[]ZNCCStorage;

	unsigned char *timg = new unsigned char[length];
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			if (mask[ii + jj*width])
				timg[ii + (height - 1 - jj)*width] = 255;
			else
				timg[ii + (height - 1 - jj)*width] = 0;
		}
	}
	SaveDataToImage("C:/temp/mask.png", timg, width, height);
	delete[]timg;

	return 0;
}
int DetermineTextureImageRegion2(double *TextureImage, bool *mask, int width, int height, double sigma, int Ithresh)
{
	int ii, jj;

	double *sTextureImage = new double[width*height];
	Gaussian_smooth_Double(TextureImage, sTextureImage, height, width, 255, sigma);

	Mat M(height, width, CV_8UC1, Scalar(0));
	Mat M2(height, width, CV_8UC1, Scalar(0));
	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			if (sTextureImage[ii + jj*width] < Ithresh)
				M.data[ii + jj*width] = 255;
			else
				M.data[ii + jj*width] = 0;
		}
	}

	//Dialate the mask
	int n = 2; //kernel size
	CvSize ksize = cvSize(2 * n + 1, 2 * n + 1);
	Point anchor; anchor.x = -n, anchor.y = n;
	Mat ele = getStructuringElement(CV_SHAPE_RECT, ksize);
	dilate(M, M2, ele);

	for (jj = 0; jj < height; jj++)
	{
		for (ii = 0; ii < width; ii++)
		{
			if ((int)M2.data[ii + jj*width]>128)
				mask[ii + jj*width] = true;
			else
				mask[ii + jj*width] = false;
		}
	}

	/*unsigned char *timg = new unsigned char[width*height];
	for(jj=0; jj<height; jj++)
	{
	for(ii=0; ii<width; ii++)
	{
	if((int)M2.data[ii+jj*width]>128)
	timg[ii+jj*width] = 255;
	else
	timg[ii+jj*width] = 0;
	}
	}
	SaveDataToImage("C:/temp/mask2.png", timg, width, height);
	delete []timg;*/

	return 0;
}
bool IsMixedFlow(IlluminationFlowImages &Fimgs, CPoint2 startP, CPoint2 &endP, double *direction, LKParameters LKArg)
{
	LKArg.DIC_Algo = 1; //Force it to be on the epipolar line

	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nchannels = Fimgs.nchannels;
	int plength = pwidth*pheight, length = width*height;

	double *iWp = 0;
	endP.x = startP.x, endP.y = startP.y;
	double score = SearchLK(startP, endP, Fimgs.Para, Fimgs.Para + nchannels*length, nchannels, width, height, width, height, LKArg, iWp, direction);

	if (score < LKArg.ZNCCThreshold)
		return true;
	else
		return false;
}

void SearchForFlowFieldsFWProp(IlluminationFlowImages &Fimgs, double *Ppatch, double *TextureImage, CPoint2 iuv, double *Epipole, int hsubset, int searchRange, int InterpAlgo, double *MagUV, double *I0patch = 0, double *IPatch = 0, double *SuperImposePatch = 0)
{
	double step = 0.5;
	int ll, kk, mm, nn, rr, ss, ii, jj;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height;
	int plength = pwidth*pheight, length = width*height, nCams = Fimgs.nCams, nchannels = Fimgs.nchannels;

	double ImgPt[3], epipline[3], direction[3];

	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	bool createdMem = false;
	if (I0patch == NULL)
	{
		createdMem = true;
		I0patch = new double[patchLength*nchannels];
		IPatch = new double[patchLength*nchannels];
		SuperImposePatch = new double[patchLength*nchannels];
	}

	double tryIllumX, tryIllumY, S[3];
	double bestMag, bestU, bestV;
	double ZNCCscore, bestZNCC = 0.0;

	//Now, start searching for texture patch that is on the epipolar line
	bestZNCC = 0.0;

	bool printout = false;

	ImgPt[0] = iuv.x, ImgPt[1] = iuv.y, ImgPt[2] = 1.0;
	cross_product(ImgPt, Epipole, epipline);
	direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
	for (ll = -searchRange; ll < searchRange; ll += 1) //for every point on the epipolar line of the illuminated image
	{
		tryIllumX = iuv.x + direction[0] * step*ll, tryIllumY = iuv.y + direction[1] * step*ll;
		for (kk = 0; kk < nchannels; kk++)
		{
			for (mm = -hsubset; mm <= hsubset; mm++)
			{
				for (nn = -hsubset; nn <= hsubset; nn++)
				{
					Get_Value_Spline(Fimgs.Para + kk*length + nchannels*length, width, height, tryIllumX + nn, tryIllumY + mm, S, -1, InterpAlgo);
					//S[0] = Fimgs.Img[kk*length+nchannels*length+ (int) (tryIllumX+nn+0.5) + ((int) ( tryIllumY+mm+0.5))*width];
					IPatch[(mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength] = S[0];
				}
			}
		}

		if (printout)
		{
			FILE *fp = fopen("C:/temp/I1.txt", "w+");
			for (mm = 0; mm < patchS; mm++)
			{
				for (nn = 0; nn < patchS; nn++)
					fprintf(fp, "%.2f ", IPatch[nn + mm*patchS]);
				fprintf(fp, "\n");
			}
			fclose(fp);
		}
		//for every point in the neighbor region of the texture image
		double localZNCC = 0.0;
		for (jj = -searchRange; jj < searchRange; jj += 1)
		{
			for (ii = -searchRange; ii < searchRange; ii += 1)
			{
				//Multiply the projected pattern with the texture image at that patch
				for (kk = 0; kk < nchannels; kk++)
				{
					for (mm = -hsubset; mm <= hsubset; mm++)
					{
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							rr = (int)(iuv.x + step*ii + nn + 0.5), ss = (int)(iuv.y + step*jj + mm + 0.5);
							SuperImposePatch[(mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength] = TextureImage[rr + ss*width + kk*length] * Ppatch[(mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength];
							//Get_Value_Spline(TextureImagePara+kk*length, width, height, iuv.x+step*ii+nn, iuv.y+step*jj + mm, S, -1, InterpAlgo);
							//SuperImposePatch[(mm+hsubset)*patchS + (nn+hsubset)+kk*patchLength] = S[0]*Ppatch[(mm+hsubset)*patchS + (nn+hsubset)+kk*patchLength];
						}
					}
				}

				if (printout)
				{
					FILE *fp = fopen("C:/temp/Text.txt", "w+");
					for (mm = -hsubset; mm <= hsubset; mm++)
					{
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							rr = (int)(iuv.x + step*ii + nn + 0.5), ss = (int)(iuv.y + step*jj + mm + 0.5);
							fprintf(fp, "%.2f ", TextureImage[rr + ss*width]);
						}
						fprintf(fp, "\n");
					}
					fclose(fp);
				}

				//compute zncc score with patch in the illuminated image vs. patch of texture * patch of projected
				ZNCCscore = ComputeZNCCPatch(IPatch, SuperImposePatch, hsubset, nchannels);
				if (ZNCCscore > bestZNCC) //retain the best score
				{
					bestMag = step*ll, bestU = iuv.x + step*ii, bestV = iuv.y + step*jj;
					bestZNCC = ZNCCscore;
				}
				if (ZNCCscore > localZNCC) //retain the best score
					localZNCC = ZNCCscore;
			}
			//if(fprintout)
			//	fprintf(fp, "\n");
		}
		//cout<<"@ll: "<<ll*step<<" (ZNCC, localZNCC): "<<bestZNCC<<" "<<localZNCC<<" (mag, u, v): "<<" "<<bestMag<<" "<<bestU<<" "<<bestV<<endl;
	}

	MagUV[0] = bestMag;
	MagUV[1] = bestU;
	MagUV[2] = bestV;

	if (createdMem)
	{
		delete[]I0patch;
		delete[]IPatch;
		delete[]SuperImposePatch;
	}

	return;
}
int FlowSeperatationFWProp(IlluminationFlowImages &Fimgs, DevicesInfo &DInfo, double *TextureImage, double *IlluminationImage, bool *mask, float *TextFlow, float *IllumFlow, double scale, LKParameters LKArg, bool GTSim, float *BTflow, float *FIflow)
{
	//Assume there is one camera
	int ii, jj, kk, mm, nn;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nchannels = Fimgs.nchannels, nCams = Fimgs.nCams, nframes = Fimgs.nframes;
	int plength = pwidth*pheight, length = width*height;
	int hsubset = LKArg.hsubset, InterpAlgo = LKArg.InterpAlgo;

	double ImgPt[3], epipline[3], direction[2];
	CPoint2  puv, startP, endP, dPts[2];

	int patchS = 2 * hsubset + 1;
	double *Ppatch = new double[patchS*patchS];
	double *RefPatch = new double[(2 * LKArg.hsubset + 1)*(2 * LKArg.hsubset + 1)*nchannels];
	double *TarPatch = new double[(2 * LKArg.hsubset + 1)*(2 * LKArg.hsubset + 1)*nchannels];
	double *ZNCCStorage = new double[2 * (2 * LKArg.hsubset + 1)*(2 * LKArg.hsubset + 1)*nchannels];

	double Epipole[3 * 10];
	for (int camID = 1; camID <= nCams; camID++)
	{
		//e't*F = 0: e' is the left null space of F
		double U[9], W[9], V[9];
		Matrix F12(3, 3); F12.Matrix_Init(&DInfo.FmatPC[(camID - 1) * 9]);
		F12.SVDcmp(3, 3, U, W, V, CV_SVD_MODIFY_A);
		//last column of U + normalize
		for (ii = 0; ii < 3; ii++)
			Epipole[3 * (camID - 1) + ii] = U[2 + 3 * ii] / U[8];
	}

	bool mixflow;
	double maguv[3];
	for (ii = 0; ii < length; ii++)
	{
		IllumFlow[ii] = 0.0, IllumFlow[ii + length] = 0.0;
		TextFlow[ii] = 0.0, TextFlow[ii + length] = 0.0;
	}

	int Allnpts = 0, npts = 0;
	CPoint ROI[2]; ROI[0].x = 600, ROI[0].y = 780, ROI[1].x = 930, ROI[1].y = 1536 - 520;
	for (jj = ROI[0].y; jj < ROI[1].y; jj += LKArg.step)
		for (ii = ROI[0].x; ii < ROI[1].x; ii += LKArg.step)
			Allnpts++;

	IJUV *vtextureFlow = new IJUV[Allnpts];
	IJUV *villumFlow = new IJUV[Allnpts];

	CPoint *proIndex = new CPoint[Allnpts];
	IJUV *GTvtextureFlow = 0, *GTvillumFlow = 0;
	if (GTSim)
	{
		GTvtextureFlow = new IJUV[Allnpts];
		GTvillumFlow = new IJUV[Allnpts];
	}

	npts = 0;
	cout << "Computing flow fields" << endl;
	int percent = 1, increP = 1;
	double start = omp_get_wtime();

	bool printout = false;
	double alpha1, alpha2, alpha, du, dv;

	FILE *fp = fopen("C:/temp/fid.txt", "w+");
	for (jj = ROI[0].y; jj < ROI[1].y; jj += LKArg.step)
	{
		for (ii = ROI[0].x; ii < ROI[1].x; ii += LKArg.step)
		{
			//ii = 194, jj = pheight-1-225;
			//ii = 140, jj = pheight-1-145;
			//ii = 835, jj= 1536-662 ;
			if (npts * 100 / Allnpts > percent)
			{
				double elapsed = omp_get_wtime() - start;
				cout << "%" << npts * 100 / Allnpts << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / percent*(100.0 - percent) << endl;
				percent += increP;
			}

			//Take the illuminated image patch
			for (mm = -hsubset; mm <= hsubset; mm++)
				for (nn = -hsubset; nn <= hsubset; nn++)
					Ppatch[(mm + hsubset)*patchS + (nn + hsubset)] = IlluminationImage[(ii + nn) + (jj + mm)*width];

			if (printout)
			{
				FILE *fp = fopen("C:/temp/ppatch.txt", "w+");
				for (mm = 0; mm < patchS; mm++)
				{
					for (nn = 0; nn < patchS; nn++)
						fprintf(fp, "%.2f ", Ppatch[nn + mm*patchS]);
					fprintf(fp, "\n");
				}
				fclose(fp);
			}

			for (kk = 0; kk < nCams; kk++)
			{
				ImgPt[0] = ii, ImgPt[1] = jj, ImgPt[2] = 1.0;
				cross_product(ImgPt, Epipole + kk * 3, epipline);
				direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
				direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);

				//Determine if the point is in texture region by: (1) optical flow, (2) stereo matching
				startP.x = ii, startP.y = jj; mixflow = false;
				if (mask[ii + jj*width])
					mixflow = true;//IsMixedFlow(Fimgs, startP, endP, direction, LKArg);

				if (GTSim)
				{
					alpha1 = FIflow[ii + jj*width] / direction[0];
					alpha2 = FIflow[ii + jj*width + length] / direction[1];
					alpha = 0.5*(alpha1 + alpha2);
					double m = FIflow[ii + jj*width] + ii, n = FIflow[ii + jj*width + length] + jj;
					du = BTflow[(int)(m + 0.5) + ((int)(n + 0.5))*width] + FIflow[ii + jj*width];
					dv = BTflow[(int)(m + 0.5) + ((int)(n + 0.5))*width + length] + FIflow[ii + jj*width + length];

					GTvillumFlow[npts].i = (float)ii, GTvillumFlow[npts].j = (float)jj;
					GTvillumFlow[npts].du = FIflow[ii + jj*width];
					GTvillumFlow[npts].dv = FIflow[ii + jj*width + length];

					if (mixflow)
					{
						GTvtextureFlow[npts].i = m - BTflow[(int)(m + 0.5) + ((int)(n + 0.5))*width];
						GTvtextureFlow[npts].j = n - BTflow[(int)(m + 0.5) + ((int)(n + 0.5))*width + length];
						GTvtextureFlow[npts].du = -BTflow[(int)(m + 0.5) + ((int)(n + 0.5))*width];
						GTvtextureFlow[npts].dv = -BTflow[(int)(m + 0.5) + ((int)(n + 0.5))*width + length];
					}
					else
					{
						GTvtextureFlow[npts].i = ii;
						GTvtextureFlow[npts].j = jj;
						GTvtextureFlow[npts].du = 0;
						GTvtextureFlow[npts].dv = 0;
					}
				}

				if (mixflow)
				{
					//Both flows are observed
					SearchForFlowFieldsFWProp(Fimgs, Ppatch, TextureImage, startP, Epipole + 3 * kk, hsubset, LKArg.searchRange, LKArg.InterpAlgo, maguv);

					villumFlow[npts].i = startP.x, villumFlow[npts].j = startP.y;
					villumFlow[npts].du = maguv[0] * direction[0];
					villumFlow[npts].dv = maguv[0] * direction[1];

					vtextureFlow[npts].i = maguv[1], vtextureFlow[npts].j = maguv[2];
					vtextureFlow[npts].du = startP.x + maguv[0] * direction[0] - maguv[1];
					vtextureFlow[npts].dv = startP.y + maguv[0] * direction[1] - maguv[2];

					proIndex[npts].x = ii, proIndex[npts].y = jj;
				}
				else
				{
					//Only illumination flow is observed
					dPts[0].x = ii, dPts[0].y = jj;
					dPts[1].x = ii; dPts[1].y = jj;
					if (EpipSearchLK(dPts, epipline, Fimgs.Img + nchannels*length*nframes*kk, Fimgs.Img + nchannels*length*(nframes*kk + 1),
						Fimgs.Para + nchannels*length*nframes*kk, Fimgs.Para + nchannels*length*(nframes*kk + 1), nchannels, width, height, width, height, LKArg, RefPatch, ZNCCStorage, TarPatch) > LKArg.ZNCCThreshold)
					{
						villumFlow[npts].i = startP.x, villumFlow[npts].j = startP.y;
						villumFlow[npts].du = dPts[1].x - startP.x, villumFlow[npts].dv = dPts[1].y - startP.y;
					}
					else
					{
						//cout<<"Illumination flow fails at "<<ii<<" "<<jj<<endl;
						fprintf(fp, "%d %d\n", ii, jj);
						villumFlow[npts].i = startP.x, villumFlow[npts].j = startP.y;
						villumFlow[npts].du = 0.0, villumFlow[npts].dv = 0.0;
					}
					//What, how can i compute it without underlying texture??? ---> set to 0s for now
					vtextureFlow[npts].i = startP.x, vtextureFlow[npts].j = startP.y;
					vtextureFlow[npts].du = 0.0, vtextureFlow[npts].dv = 0.0;

					proIndex[npts].x = ii, proIndex[npts].y = jj;
				}
				npts++;
			}
		}
	}
	fclose(fp);

	FILE *fp1 = fopen("C:/temp/clothSim/Flow/vtflow.txt", "w+");
	FILE *fp2 = fopen("C:/temp/clothSim/Flow/viflow.txt", "w+");
	for (ii = 0; ii < npts; ii++)
	{
		fprintf(fp1, "%d %d %.2f %.2f %.2f %.2f\n", proIndex[ii].x, proIndex[ii].y, vtextureFlow[ii].i, vtextureFlow[ii].j, vtextureFlow[ii].du, vtextureFlow[ii].dv);
		fprintf(fp2, "%d %d %.2f %.2f %.2f %.2f\n", proIndex[ii].x, proIndex[ii].y, villumFlow[ii].i, villumFlow[ii].j, villumFlow[ii].du, villumFlow[ii].dv);
	}
	fclose(fp1), fclose(fp2);

	fp1 = fopen("C:/temp/clothSim/Flow/_GTvtflow.txt", "w+");
	fp2 = fopen("C:/temp/clothSim/Flow/_GTviflow.txt", "w+");
	for (ii = 0; ii < npts; ii++)
	{
		fprintf(fp1, "%d %d %.2f %.2f %.2f %.2f\n", proIndex[ii].x, proIndex[ii].y, GTvtextureFlow[ii].i, GTvtextureFlow[ii].j, GTvtextureFlow[ii].du, GTvtextureFlow[ii].dv);
		fprintf(fp2, "%d %d %.2f %.2f %.2f %.2f\n", proIndex[ii].x, proIndex[ii].y, GTvillumFlow[ii].i, GTvillumFlow[ii].j, GTvillumFlow[ii].du, villumFlow[ii].dv);
	}
	fclose(fp1), fclose(fp2);

	delete[]RefPatch;
	delete[]TarPatch;
	delete[]ZNCCStorage;
	delete[]proIndex;
	delete[]vtextureFlow;
	delete[]villumFlow;
	delete[]Ppatch;
	delete[]GTvtextureFlow;
	delete[]GTvillumFlow;

	return 0;
}
void SearchForFlowFieldsBWProp(double *IPatch, double *IllimnationPatch, double *SuperImposePatch, double *TZNCC, double *IlluminationImage, double *TextureImage, CPoint2 iuv, double *direction, int hsubset, int searchRange, int width, int height, int InterpAlgo, double *MagUV)
{
	double step = 0.5;
	int ii, jj, ll, kk, mm, nn, rr, ss, rx, ry;
	int length = width*height, nchannels = 1;
	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;

	int tryIllumX, tryIllumY;
	double bestMag, bestdU, bestdV;
	double ZNCCscore, bestZNCC = 0.0;

	bool printout = false;
	//Now, start searching for texture patch that is on the epipolar line
	bestZNCC = -1.0;
	for (ll = -searchRange; ll < searchRange; ll++) //for every point on the epipolar line of the illuminated image
	{
		tryIllumX = (int)(iuv.x + direction[0] * step*ll + 0.5), tryIllumY = (int)(iuv.y + direction[1] * step*ll + 0.5);
		for (kk = 0; kk < nchannels; kk++)
		{
			for (mm = -hsubset; mm <= hsubset; mm++)
			{
				for (nn = -hsubset; nn <= hsubset; nn++)
				{
					//Get_Value_Spline(IlluminationImagePara + nchannels*length, width, height, tryIllumX+nn, tryIllumY+mm, S, -1, InterpAlgo);
					//IllimnationPatch[(mm+hsubset)*patchS + (nn+hsubset)+kk*patchLength] = S[0];
					IllimnationPatch[(mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength] = IlluminationImage[kk*length + (tryIllumX + nn) + (tryIllumY + mm)*width];
				}
			}
		}

		if (printout)
		{
			FILE *fp = fopen("C:/temp/IllumPatch.txt", "w+");
			for (mm = 0; mm < patchS; mm++)
			{
				for (nn = 0; nn < patchS; nn++)
					fprintf(fp, "%.2f ", IllimnationPatch[nn + mm*patchS]);
				fprintf(fp, "\n");
			}
			fclose(fp);
		}

		//for every point in the neighbor region of the texture image
		double localZNCC = -1.0;
		for (jj = -searchRange; jj < searchRange; jj++)
		{
			for (ii = -searchRange; ii < searchRange; ii++)
			{
				//Multiply the projected pattern with the texture image at that patch
				rx = (int)(iuv.x + step*ii + 0.5), ry = (int)(iuv.y + step*jj + 0.5);
				for (kk = 0; kk < nchannels; kk++)
				{
					for (mm = -hsubset; mm <= hsubset; mm++)
					{
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							rr = rx + nn, ss = ry + mm;
							SuperImposePatch[(mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength] = TextureImage[rr + ss*width + kk*length] * IllimnationPatch[(mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength];
							//Get_Value_Spline(TextureImagePara+kk*length, width, height, iuv.x+step*ii+nn, iuv.y+step*jj + mm, S, -1, InterpAlgo);
							//SuperImposePatch[(mm+hsubset)*patchS + (nn+hsubset)+kk*patchLength] = S[0]*Ppatch[(mm+hsubset)*patchS + (nn+hsubset)+kk*patchLength];
						}
					}
				}

				if (printout)
				{
					FILE *fp = fopen("C:/temp/Text.txt", "w+");
					for (mm = -hsubset; mm <= hsubset; mm++)
					{
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							rr = (int)(iuv.x + step*ii + nn + 0.5), ss = (int)(iuv.y + step*jj + mm + 0.5);
							fprintf(fp, "%.2f ", TextureImage[rr + ss*width]);
						}
						fprintf(fp, "\n");
					}
					fclose(fp);
				}

				//compute zncc score with patch in the illuminated image vs. patch of texture * patch of projected
				ZNCCscore = ComputeZNCCPatch(IPatch, SuperImposePatch, hsubset, nchannels, TZNCC);
				if (ZNCCscore > bestZNCC) //retain the best score
				{
					bestMag = step*ll, bestdU = step*ii, bestdV = step*jj;
					bestZNCC = ZNCCscore;
				}
				if (ZNCCscore > localZNCC) //retain the best score
					localZNCC = ZNCCscore;
			}
			//if(fprintout)
			//	fprintf(fp, "\n");
		}
		//cout<<"@ll: "<<ll*step<<" (ZNCC, localZNCC): "<<bestZNCC<<" "<<localZNCC<<" (mag, u, v): "<<" "<<bestMag<<" "<<bestdU<<" "<<bestdV<<endl;
	}

	MagUV[0] = bestMag;
	MagUV[1] = bestdU;
	MagUV[2] = bestdV;


	return;
}
double OptimizeFlowFieldsBWProp(double *IPatch, double *TZNCC, double *IlluminationImagePara, double *TextureImagePara, double *direction, CPoint2 *Target, LKParameters LKArg, int width, int height)
{
	int i, j, k, iii, jjj;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold;

	double II0, JJ0, II1, JJ1, mF, mG, mG0, mG1, g0x, g0y, g1x, g1y, DIC_Coeff, DIC_Coeff_min, a, b, t_1, t_2, t_3, t_4, t_5, t_6, S0[3], S1[3], p_best[13];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 3), conv_crit_2 = conv_crit_1*0.01;
	int jumpStep[2] = { 1, 2 }, nn = 13, nExtraParas = 2, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	double AA[13 * 13], BB[13], CC[13], p[13];
	for (i = 0; i < nn; i++)
		p[i] = (i == nn - 2 ? 1.0 : 0.0);

	int length = width*height, nchannels = 1, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	FILE *fp, *fp0, *fp1;

	bool printout = false;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (j = -hsubset; j <= hsubset; j++)
		{
			for (i = -hsubset; i <= hsubset; i++)
				fprintf(fp, "%.2f ", IPatch[(i + hsubset) + (j + hsubset)*TimgS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
			{
				fp = fopen("C:/temp/tar.txt", "w+");
				fp0 = fopen("C:/temp/tar0.txt", "w+");
				fp1 = fopen("C:/temp/tar1.txt", "w+");
			}

			a = p[nn - 2], b = p[nn - 1];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					II0 = Target[0].x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj;
					JJ0 = Target[0].y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;

					II1 = Target[1].x + iii + p[5] + p[7] * iii + p[8] * jjj;
					JJ1 = Target[1].y + jjj + p[6] + p[9] * iii + p[10] * jjj;

					if (II0<0.0 || II0>(double)(width - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(height - 1) - (1e-10))
						continue;
					if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
						continue;

					Get_Value_Spline(IlluminationImagePara, width, height, II0, JJ0, S0, 0, Interpolation_Algorithm);
					Get_Value_Spline(TextureImagePara, width, height, II1, JJ1, S1, 0, Interpolation_Algorithm);

					mF = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
					mG0 = S0[0], mG1 = S1[0], mG = mG0*mG1;
					g0x = S0[1], g0y = S0[2], g1x = S1[1], g1y = S1[2];

					t_3 = a*mG0*mG1 + b - mF;

					t_4 = a*mG1, t_5 = t_4*g0x, t_6 = t_4*g0y;
					CC[0] = t_5*direction[0] + t_6*direction[1];
					CC[1] = t_5*iii, CC[2] = t_5*jjj;
					CC[3] = t_6*iii, CC[4] = t_6*jjj;

					t_4 = a*mG0, t_5 = t_4*g1x, t_6 = t_4*g1y;
					CC[5] = t_5, CC[6] = t_6;
					CC[7] = t_5*iii, CC[8] = t_5*jjj;
					CC[9] = t_6*iii, CC[10] = t_6*jjj;

					CC[11] = mG, CC[12] = 1.0;

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += mF*mF;

					if (printout)
					{
						fprintf(fp, "%.2f ", mG);
						fprintf(fp0, "%.2f ", mG0);
						fprintf(fp1, "%.2f ", mG1);
					}
				}
				if (printout)
				{
					fprintf(fp, "\n"), fprintf(fp0, "\n"), fprintf(fp1, "\n");
				}
			}
			if (printout)
			{
				fclose(fp), fclose(fp0), fclose(fp1);
			}

			DIC_Coeff = t_1 / t_2;
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
				if (p[0] != p[0])
					return 0.0;
			}

			if (abs(p[0] * direction[0]) > hsubset || abs(p[0] * direction[1]) > hsubset || abs(p[5]) > hsubset || abs(p[6]) > hsubset)
				return 0.0;
			if (fabs(BB[0]) < conv_crit_1 && fabs(BB[5]) < conv_crit_1 && fabs(BB[6]) < conv_crit_1)
			{
				for (i = 1; i < 5; i++)
				{
					if (fabs(BB[i]) > conv_crit_2)
						break;
				}

				if (i == 5)
				{
					for (i = 7; i < nn - nExtraParas; i++)
						if (fabs(BB[i]) > conv_crit_2)
							break;
				}
				if (i == nn - nExtraParas)
					Break_Flag = true;
			}
			if (Break_Flag)
				break;
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! 
	//They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II0 = Target[0].x + iii + p[0] * direction[0] + p[1] * iii + p[2] * jjj;
				JJ0 = Target[0].y + jjj + p[0] * direction[1] + p[3] * iii + p[4] * jjj;

				II1 = Target[1].x + iii + p[5] + p[7] * iii + p[8] * jjj;
				JJ1 = Target[1].y + jjj + p[6] + p[9] * iii + p[10] * jjj;

				if (II0<0.0 || II0>(double)(width - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(height - 1) - (1e-10))
					continue;
				if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
					continue;

				Get_Value_Spline(IlluminationImagePara, width, height, II0, JJ0, S0, 0, Interpolation_Algorithm);
				Get_Value_Spline(TextureImagePara, width, height, II1, JJ1, S1, 0, Interpolation_Algorithm);

				TZNCC[2 * m] = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
				TZNCC[2 * m + 1] = S0[0] * S1[0];
				t_f += TZNCC[2 * m];
				t_g += TZNCC[2 * m + 1];

				if (printout)
					fprintf(fp, "%.2f ", TZNCC[2 * m + 1]);
				m++;
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / (m + 1);
		t_g = t_g / (m + 1);
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = TZNCC[2 * i] - t_f;
			t_5 = TZNCC[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5;
			t_2 += 1.0*t_4*t_4;
			t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;
	}

	if (abs(p[0] * direction[0]) > hsubset || abs(p[0] * direction[1]) > hsubset || abs(p[5]) > hsubset || abs(p[6]) > hsubset)
		return 0.0;

	Target[0].x += p[0] * direction[0];
	Target[0].y += p[0] * direction[1];
	Target[1].x += p[5];
	Target[1].y += p[6];

	return DIC_Coeff_min;
}
int FlowSeperatationBWProp(IlluminationFlowImages &Fimgs, DevicesInfo &DInfo, double *TextureImage, double *TextureImagePara, double *IlluminationImage, double *IlluminationImagePara, int *CamProMatch, bool *mask, float *TextFlow, float *IllumFlow, LKParameters LKArg, CPoint *ROI, bool GTSim, float *BTflow, float *BIflow)
{
	//Assume there is one camera
	int ii, jj, kk, mm, nn;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nchannels = Fimgs.nchannels, nCams = Fimgs.nCams, nframes = Fimgs.nframes;
	int plength = pwidth*pheight, length = width*height;
	int hsubset = LKArg.hsubset, InterpAlgo = LKArg.InterpAlgo;

	double ImgPt[3], epipline[3], direction[2];
	CPoint2  puv, startP, endP, dPts[2];

	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	double *IPatch = new double[patchS*patchS];
	double *TarPatch = new double[patchLength*nchannels];
	double *IllimnationPatch = new double[patchLength*nchannels];
	double *SuperImposePatch = new double[patchLength*nchannels];
	double *TZNCC = new double[2 * patchLength*nchannels];

	double Epipole[3 * 10];
	for (int camID = 1; camID <= nCams; camID++)
	{
		//e't*F = 0: e' is the left null space of F
		double U[9], W[9], V[9];
		Matrix F12(3, 3); F12.Matrix_Init(&DInfo.FmatPC[(camID - 1) * 9]);
		F12.SVDcmp(3, 3, U, W, V, CV_SVD_MODIFY_A);
		//last column of U + normalize
		for (ii = 0; ii < 3; ii++)
			Epipole[3 * (camID - 1) + ii] = U[2 + 3 * ii] / U[8];
	}

	for (ii = 0; ii < 2 * length; ii++)
		TextFlow[ii] = 0.0f, IllumFlow[ii] = 0.0f;

	bool mixflow;
	double magduv[3], score;

	int Allnpts = 0, npts = 0, nmix = 0;
	for (jj = ROI[0].y; jj < ROI[1].y; jj += LKArg.step)
		for (ii = ROI[0].x; ii < ROI[1].x; ii += LKArg.step)
			Allnpts++;

	IJUV *vtextureFlow = new IJUV[Allnpts];
	IJUV *villumFlow = new IJUV[Allnpts];


	CPoint *proIndex = 0;
	IJUV *GTvtextureFlow = 0, *GTvillumFlow = 0;
	if (GTSim)
	{
		proIndex = new CPoint[Allnpts];
		GTvtextureFlow = new IJUV[Allnpts];
		GTvillumFlow = new IJUV[Allnpts];
	}

	npts = 0;
	int percent = 1, increP = 1;
	double start;
	if (omp_get_thread_num() == 0)
	{
		cout << "Computing flow fields" << endl;
		percent = 1, increP = 1;
		start = omp_get_wtime();
	}

	double alpha1, alpha2, alpha, du, dv;

	//FILE *fp = fopen("C:/temp/fid.txt", "w+");
	for (jj = ROI[0].y; jj < ROI[1].y; jj += LKArg.step)
	{
		for (ii = ROI[0].x; ii < ROI[1].x; ii += LKArg.step)
		{
			//ii = 822, jj = 1536-646;
			//ii = 825, jj = height-620;
			//ii = 700, jj = height - 940;
			if (omp_get_thread_num() == 0 && npts * 100 / Allnpts > percent)
			{
				double elapsed = omp_get_wtime() - start;
				cout << "%" << npts * 100 / Allnpts << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / percent*(100.0 - percent) << endl;
				percent += increP;
			}

			for (kk = 0; kk < nCams; kk++)
			{
				//Take the observed 2nd image patch
				for (mm = -hsubset; mm <= hsubset; mm++)
					for (nn = -hsubset; nn <= hsubset; nn++)
						IPatch[(mm + hsubset)*patchS + (nn + hsubset)] = Fimgs.Img[(ii + nn) + (jj + mm)*width + length];

				ImgPt[0] = ii, ImgPt[1] = jj, ImgPt[2] = 1.0;
				cross_product(ImgPt, Epipole + kk * 3, epipline);
				direction[0] = -epipline[1] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);
				direction[1] = epipline[0] / sqrt(epipline[0] * epipline[0] + epipline[1] * epipline[1]);

				//Determine texture region by: (1) optical flow, (2) stereo matching --> NOT DONE YET!!!
				startP.x = ii, startP.y = jj; mixflow = false;
				//if(CamProMatch[(ii+nn)+(jj+mm)*width] == 0 && CamProMatch[(ii+nn)+(jj+mm)*width+length] == 0)
				//	mixflow = true;
				//else
				if (mask[ii + jj*width])
					mixflow = true;//IsMixedFlow(Fimgs, startP, endP, direction, LKArg);

				if (GTSim)
				{
					alpha1 = BIflow[ii + jj*width] / direction[0];
					alpha2 = BIflow[ii + jj*width + length] / direction[1];
					alpha = 0.5*(alpha1 + alpha2);
					du = BTflow[ii + jj*width];
					dv = BTflow[ii + jj*width + length];

					GTvillumFlow[npts].i = ii, GTvillumFlow[npts].j = jj;
					GTvillumFlow[npts].du = BIflow[ii + jj*width];
					GTvillumFlow[npts].dv = BIflow[ii + jj*width + length];

					if (mixflow)
					{
						GTvtextureFlow[npts].i = ii, GTvtextureFlow[npts].j = jj;
						GTvtextureFlow[npts].du = du, GTvtextureFlow[npts].dv = dv;
					}
					else
					{
						GTvtextureFlow[npts].i = ii, GTvtextureFlow[npts].j = jj;
						GTvtextureFlow[npts].du = 0, GTvtextureFlow[npts].dv = 0;
					}
				}

				if (mixflow)
				{
					//Both flows are observed
					SearchForFlowFieldsBWProp(IPatch, IllimnationPatch, SuperImposePatch, TZNCC, IlluminationImage, TextureImage, startP, direction, hsubset, LKArg.searchRange, width, height, LKArg.InterpAlgo, magduv);

					dPts[0].x = startP.x + magduv[0] * direction[0], dPts[0].y = startP.y + magduv[0] * direction[1];
					dPts[1].x = startP.x + magduv[1], dPts[1].y = startP.y + magduv[2];

					score = OptimizeFlowFieldsBWProp(IPatch, TZNCC, IlluminationImagePara, TextureImagePara, direction, dPts, LKArg, width, height);
					if (score > LKArg.ZNCCThreshold)
					{
						villumFlow[npts].i = startP.x, villumFlow[npts].j = startP.y;
						villumFlow[npts].du = dPts[0].x - startP.x, villumFlow[npts].dv = dPts[0].y - startP.y;
						IllumFlow[ii + jj*width] = dPts[0].x - startP.x, IllumFlow[ii + jj*width + length] = dPts[0].y - startP.y;

						vtextureFlow[npts].i = startP.x, vtextureFlow[npts].j = startP.y;
						vtextureFlow[npts].du = dPts[1].x - startP.x, vtextureFlow[npts].dv = dPts[1].y - startP.y;
						TextFlow[ii + jj*width] = dPts[1].x - startP.x, TextFlow[ii + jj*width + length] = dPts[1].y - startP.y;
					}
					else
					{
						villumFlow[npts].i = startP.x, villumFlow[npts].j = startP.y;
						villumFlow[npts].du = 0.0, villumFlow[npts].dv = 0.0;
						IllumFlow[ii + jj*width] = 0.0, IllumFlow[ii + jj*width + length] = 0.0;

						vtextureFlow[npts].i = startP.x, vtextureFlow[npts].j = startP.y;
						vtextureFlow[npts].du = 0.0, vtextureFlow[npts].dv = 0.0;
						TextFlow[ii + jj*width] = 0.0, TextFlow[ii + jj*width + length] = 0.0;
					}
					nmix++;
					if (GTSim)
						proIndex[npts].x = ii, proIndex[npts].y = jj;
				}
				else
				{
					//Only illumination flow is observed
					dPts[0].x = startP.x, dPts[0].y = startP.y;
					dPts[1].x = startP.x, dPts[1].y = startP.y;
					if (EpipSearchLK(dPts, epipline, Fimgs.Img + nchannels*length*(nframes*kk + 1), Fimgs.Img + nchannels*length*nframes*kk,
						Fimgs.Para + nchannels*length*(nframes*kk + 1), Fimgs.Para + nchannels*length*nframes*kk, nchannels, width, height, width, height, LKArg, IPatch, TZNCC, TarPatch) > LKArg.ZNCCThreshold)
					{
						villumFlow[npts].i = startP.x, villumFlow[npts].j = startP.y;
						villumFlow[npts].du = dPts[1].x - dPts[0].x, villumFlow[npts].dv = dPts[1].y - dPts[0].y;
						IllumFlow[ii + jj*width] = dPts[1].x - dPts[0].x, IllumFlow[ii + jj*width + length] = dPts[1].y - dPts[0].y;
					}
					else
					{
						//fprintf(fp, "%d %d\n", ii, jj);
						villumFlow[npts].i = startP.x, villumFlow[npts].j = startP.y;
						villumFlow[npts].du = 0.0, villumFlow[npts].dv = 0.0;
						IllumFlow[ii + jj*width] = 0.0, IllumFlow[ii + jj*width + length] = 0.0;
					}

					//What, how can i compute it without underlying texture??? ---> set to 0s for now
					vtextureFlow[npts].i = startP.x, vtextureFlow[npts].j = startP.y;
					vtextureFlow[npts].du = 0.0, vtextureFlow[npts].dv = 0.0;
					TextFlow[ii + jj*width] = 0.0, TextFlow[ii + jj*width + length] = 0.0;

					if (GTSim)
						proIndex[npts].x = ii, proIndex[npts].y = jj;
				}
				npts++;
			}
		}
	}
	//fclose(fp);

	char FnameX[200], FnameY[200];
	sprintf(FnameX, "C:/temp/clothSim/Flow/vtflow_%d.txt", omp_get_thread_num());
	sprintf(FnameY, "C:/temp/clothSim/Flow/viflow_%d.txt", omp_get_thread_num());
	FILE *fp1 = fopen(FnameX, "w+");
	FILE *fp2 = fopen(FnameY, "w+");
	for (ii = 0; ii < npts; ii++)
	{
		fprintf(fp1, "%d %d %.2f %.2f %.2f %.2f\n", proIndex[ii].x, proIndex[ii].y, vtextureFlow[ii].i, vtextureFlow[ii].j, vtextureFlow[ii].du, vtextureFlow[ii].dv);
		fprintf(fp2, "%d %d %.2f %.2f %.2f %.2f\n", proIndex[ii].x, proIndex[ii].y, villumFlow[ii].i, villumFlow[ii].j, villumFlow[ii].du, villumFlow[ii].dv);
	}
	fclose(fp1), fclose(fp2);


	delete[]IPatch;
	delete[]TarPatch;
	delete[]IllimnationPatch;
	delete[]SuperImposePatch;
	delete[]TZNCC;
	delete[]proIndex;
	delete[]vtextureFlow;
	delete[]villumFlow;

	return 0;
}

void ConvertProDepthToCamProMatching(DevicesInfo &DInfo, float *PDepth, int *CamProMatch, int width, int height, int pwidth, int pheight, int camID)
{
	int ii, jj, rx, ry, length = width*height, plength = pwidth*pheight;

	double Pmat[12] = { DInfo.K[0], DInfo.K[1], DInfo.K[2], 0.0,
		DInfo.K[3], DInfo.K[4], DInfo.K[5], 0.0,
		DInfo.K[6], DInfo.K[7], DInfo.K[8], 0.0 };

	double X, Y, Z, rayDirectX, rayDirectY, denum;
	CPoint2  puv, iuv;

	for (ii = 0; ii < 2 * length; ii++)
		CamProMatch[ii] = 0;

	camID = camID - 1;
	//FILE *fp = fopen("C:/temp/cpm.txt", "w+");
	for (jj = 0; jj < pheight; jj++)
	{
		for (ii = 0; ii < pwidth; ii++)
		{
			Z = PDepth[ii + jj*pwidth];
			if (abs(Z) < 0.1)
				continue;

			rayDirectX = DInfo.iK[0] * ii + DInfo.iK[1] * jj + DInfo.iK[2], rayDirectY = DInfo.iK[4] * jj + DInfo.iK[5];
			X = rayDirectX*Z;
			Y = rayDirectY*Z;

			denum = Pmat[8] * X + Pmat[9] * Y + Pmat[10] * Z + Pmat[11];
			puv.x = (Pmat[0] * X + Pmat[1] * Y + Pmat[2] * Z + Pmat[3]) / denum;
			puv.y = (Pmat[4] * X + Pmat[5] * Y + Pmat[6] * Z + Pmat[7]) / denum;
			if (puv.x <1 || puv.x >pwidth - 1 || puv.y < 1 || puv.y >pheight - 1)
				continue;

			denum = DInfo.P[12 * camID + 8] * X + DInfo.P[12 * camID + 9] * Y + DInfo.P[12 * camID + 10] * Z + DInfo.P[12 * camID + 11];
			iuv.x = (DInfo.P[12 * camID + 0] * X + DInfo.P[12 * camID + 1] * Y + DInfo.P[12 * camID + 2] * Z + DInfo.P[12 * camID + 3]) / denum;
			iuv.y = (DInfo.P[12 * camID + 4] * X + DInfo.P[12 * camID + 5] * Y + DInfo.P[12 * camID + 6] * Z + DInfo.P[12 * camID + 7]) / denum;

			if (iuv.x <1 || iuv.x > width - 1 || iuv.y <1 || iuv.y > height - 1)
				continue;

			rx = (int)(iuv.x + 0.5), ry = (int)(iuv.y + 0.5);
			CamProMatch[rx + ry*width] = ii;
			CamProMatch[rx + ry*width + length] = jj;
			//fprintf(fp, "%d %d %d %d\n", rx, ry, ii, jj);
		}
	}
	//fclose(fp);

	return;
}

void UpdateIllumTextureImages(char *PATH, bool silent, int frameID, int mode, int nPros, int proOffset, int width, int height, int pwidth, int pheight, int nchannels, int InterpAlgo, double *ParaIllumSource, float *ILWarping, double *ParaSourceTexture, float *TWarping, bool *ROI, bool color, int id)
{
	if (!silent)
		cout << "Update Images from thread " << omp_get_thread_num() << endl;

	int ii, jj, kk, length = width*height, plength = pwidth*pheight;
	double S[3], xx, yy;
	char Fname[200];

	unsigned char *LuminanceImg = new unsigned char[length*nchannels];
	if (proOffset == 0 && nPros == 2)
	{
		for (int ProID = 0; ProID < nPros; ProID++)
		{
			int offset = 6 * ProID*length;
			for (jj = 0; jj < height; jj++)
			{
				for (ii = 0; ii < width; ii++)
				{
					if (ROI != NULL &&ROI[ii + jj*width])
						for (kk = 0; kk < nchannels; kk++)
							LuminanceImg[ii + jj*width + kk*length] = (unsigned char)(0);
					else
						for (kk = 0; kk < nchannels; kk++)
							LuminanceImg[ii + jj*width + kk*length] = (unsigned char)(0);

					if (abs(ILWarping[ii + jj*width + offset]) + abs(ILWarping[ii + jj*width + length + offset]) > 0.01)
					{
						xx = ILWarping[ii + jj*width + offset] + ii, yy = ILWarping[ii + jj*width + length + offset] + jj;
						if (xx<0.0 || xx > pwidth - 1 || yy < 0.0 || yy>pheight - 1)
							continue;
						for (kk = 0; kk<nchannels; kk++)
						{
							Get_Value_Spline(ParaIllumSource + (ProID*nchannels + kk)*plength, pwidth, pheight, xx, yy, S, -1, InterpAlgo);
							if (S[0] > 255.0)
								S[0] = 255.0;
							else if (S[0] < 0.0)
								S[0] = 0.0;
							LuminanceImg[ii + jj*width + kk*length] = (unsigned char)(int)(S[0] + 0.5);
						}
					}
				}
			}
			if (id == 0)
				sprintf(Fname, "%s/wLC%dP%d_%05d.png", PATH, 1, ProID + 1, frameID);
			else
				sprintf(Fname, "%s/wLC%dP%d_%d_%05d.png", PATH, 1, ProID + 1, id, frameID);
			SaveDataToImage(Fname, LuminanceImg, width, height);
		}
	}
	else
	{
		for (jj = 0; jj < height; jj++)
		{
			for (ii = 0; ii < width; ii++)
			{
				if (ROI != NULL &&ROI[ii + jj*width])
					for (kk = 0; kk < nchannels; kk++)
						LuminanceImg[ii + jj*width + kk*length] = (unsigned char)(0);
				else
					for (kk = 0; kk < nchannels; kk++)
						LuminanceImg[ii + jj*width + kk*length] = (unsigned char)(0);

				if (abs(ILWarping[ii + jj*width]) + abs(ILWarping[ii + jj*width + length]) > 0.01)
				{
					xx = ILWarping[ii + jj*width] + ii, yy = ILWarping[ii + jj*width + length] + jj;
					if (xx<0.0 || xx > pwidth - 1 || yy < 0.0 || yy>pheight - 1)
						continue;
					for (kk = 0; kk<nchannels; kk++)
					{
						Get_Value_Spline(ParaIllumSource + kk*plength, pwidth, pheight, xx, yy, S, -1, InterpAlgo);
						if (S[0] > 255.0)
							S[0] = 255.0;
						else if (S[0] < 0.0)
							S[0] = 0.0;
						LuminanceImg[ii + jj*width + kk*length] = (unsigned char)(int)(S[0] + 0.5);
					}
				}
			}
		}
		if (id == 0)
			sprintf(Fname, "%s/wLC%dP%d_%05d.png", PATH, 1, proOffset + 1, frameID);
		else
			sprintf(Fname, "%s/wLC%dP%d_%d_%05d.png", PATH, 1, proOffset + 1, id, frameID);
		SaveDataToImage(Fname, LuminanceImg, width, height);
	}

	if (TWarping != NULL && mode < 2)
	{
		nchannels = (color == true) ? 3 : 1;
		unsigned char *TextureImg = new unsigned char[length*nchannels];
		for (jj = 0; jj < height; jj++)
		{
			for (ii = 0; ii < width; ii++)
			{
				if (ROI != NULL && ROI[ii + jj*width])
					for (kk = 0; kk < nchannels; kk++)
						TextureImg[ii + jj*width + kk*length] = (unsigned char)(0);
				else
					for (kk = 0; kk < nchannels; kk++)
						TextureImg[ii + jj*width + kk*length] = (unsigned char)(0);

				if (abs(TWarping[ii + jj*width]) + abs(TWarping[ii + jj*width + length]) > 0.01)
				{
					xx = TWarping[ii + jj*width] + ii, yy = TWarping[ii + jj*width + length] + jj;
					if (xx<0.0 || xx > width - 1 || yy < 0.0 || yy>height - 1)
						continue;
					for (int kk = 0; kk<nchannels; kk++)
					{
						Get_Value_Spline(ParaSourceTexture + kk*length, width, height, xx, yy, S, -1, InterpAlgo);
						if (S[0] > 255.0)
							S[0] = 255.0;
						else if (S[0] < 0.0)
							S[0] = 0.0;
						TextureImg[ii + jj*width + kk*length] = (unsigned char)(int)(S[0] + 0.5);
					}
				}
			}
		}
		if (id == 0)
			sprintf(Fname, "%s/wTexture_%05d.png", PATH, frameID);
		else
			sprintf(Fname, "%s/wTexture_%d_%05d.png", PATH, id, frameID);
		SaveDataToImage(Fname, TextureImg, width, height, nchannels);
	}

	return;
}
int TextImageMatchSource(char *PATH, int frameID, LKParameters LKArg, int nchannels, double scale)
{
	int ii, jj, kk;
	int Twidth, Theight, Swidth, Sheight, Tlength, Slength;

	int  step = LKArg.step, hsubset = LKArg.hsubset;
	int Incomplete_Subset_Handling = LKArg.Incomplete_Subset_Handling, Convergence_Criteria = LKArg.Convergence_Criteria, Analysis_Speed = LKArg.Analysis_Speed, DICAlgo = LKArg.DIC_Algo, InterpAlgo = LKArg.InterpAlgo;
	double PSSDab_thresh = LKArg.PSSDab_thresh, Gsigma = LKArg.Gsigma;

	char *TextImage = 0, *SourceText = 0;
	Mat img;
	char Fname[200];
	sprintf(Fname, "%s/Image/C1_%05d.png", PATH, frameID); img = imread(Fname, nchannels == 1 ? 0 : 1);
	if (img.data == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 2;
	}
	else
	{
		Twidth = img.cols, Theight = img.rows, Tlength = Twidth*Theight;
		TextImage = new char[Tlength*nchannels];
		for (kk = 0; kk < nchannels; kk++)
			for (jj = 0; jj < Theight; jj++)
				for (ii = 0; ii < Twidth; ii++)
					TextImage[ii + (Theight - 1 - jj)*Twidth + kk*Tlength] = (char)img.data[nchannels*ii + nchannels*jj*Twidth + kk];
	}

	sprintf(Fname, "%s/Image/STexture1.png", PATH, 1); img = imread(Fname, nchannels == 1 ? 0 : 1);
	if (img.data == NULL)
	{
		cout << "Cannot load " << Fname << endl;
		return 2;
	}
	else
	{
		Swidth = img.cols, Sheight = img.rows, Slength = Swidth*Sheight;
		SourceText = new char[Slength*nchannels];
		for (kk = 0; kk < nchannels; kk++)
			for (jj = 0; jj < Sheight; jj++)
				for (ii = 0; ii < Twidth; ii++)
					SourceText[ii + (Sheight - 1 - jj)*Swidth + kk*Slength] = (char)img.data[nchannels*ii + nchannels*jj*Swidth + kk];
	}

	int nSeedPoints = 0;
	CPoint2 *SparseCorres1 = 0, *SparseCorres2 = 0, tcorres;
	sprintf(Fname, "%s/Sparse/T%d_%05d.txt", PATH, 1, frameID); FILE *fp = fopen(Fname, "r");
	if (fp == NULL)
	{
		cout << "Cannot load Source texture image keypoints" << endl;
		return 3;
	}
	else
	{
		nSeedPoints = 0;
		while (fscanf(fp, "%lf %lf ", &tcorres.x, &tcorres.y) != EOF)
			nSeedPoints++;
		fclose(fp);

		SparseCorres1 = new CPoint2[nSeedPoints];
		SparseCorres2 = new CPoint2[nSeedPoints];

		sprintf(Fname, "%s/Sparse/T%d_%05d.txt", PATH, 1, frameID); fp = fopen(Fname, "r");
		for (ii = 0; ii < nSeedPoints; ii++)
			fscanf(fp, "%lf %lf ", &SparseCorres1[ii].x, &SparseCorres1[ii].y);
		fclose(fp);

		sprintf(Fname, "%s/Sparse/ST%d_%05d.txt", PATH, 1, frameID); fp = fopen(Fname, "r");
		if (fp == NULL)
		{
			cout << "Cannot load texture image keypoint" << endl;
			return 3;
		}
		else
		{
			for (ii = 0; ii < nSeedPoints; ii++)
				fscanf(fp, "%lf %lf ", &SparseCorres2[ii].x, &SparseCorres2[ii].y);
			fclose(fp);
		}
	}


	bool *cROI = new bool[Tlength];
	bool *lpROI_calculated = new bool[Tlength];
	CPoint2 *disparity = new CPoint2[Tlength];
	for (ii = 0; ii < Tlength; ii++)
	{
		cROI[ii] = false; // 1 - Valid, 0 - Other
		lpROI_calculated[ii] = false; 	// 1 - Calculated, 0 - Other

		disparity[ii].x = 0.0;
		disparity[ii].y = 0.0;
	}

	for (jj = 2 * hsubset; jj < Theight - 2 * hsubset; jj++)
		for (ii = 2 * hsubset; ii < Twidth - 2 * hsubset; ii++)
			cROI[ii + jj*Twidth] = true;

	float *WarpingParas = new float[6 * Tlength];
	for (ii = 0; ii < 6 * Tlength; ii++)
		WarpingParas[ii] = 0.0f;

	cout << "Match texture image and its source ..." << endl;
	GreedyMatching(TextImage, SourceText, disparity, lpROI_calculated, cROI, SparseCorres1, SparseCorres2, nSeedPoints, LKArg, nchannels, Twidth, Theight, Swidth, Sheight, scale, NULL, WarpingParas);

	for (ii = 0; ii < 6; ii++)
	{
		sprintf(Fname, "%s/Results/Sep/C1TS%dp%d_%05d.dat", PATH, 1, ii, frameID);
		WriteGridBinary(Fname, WarpingParas + ii*Tlength, Twidth, Theight);
	}


	//Create warped images
	double xx, yy, S[3];
	unsigned char *wImg = new unsigned char[Tlength*nchannels];
	double *DSourceText = new double[Slength*nchannels];
	double *SourcePara = new double[Slength*nchannels];

	for (int kk = 0; kk < nchannels; kk++)
	{
		Gaussian_smooth(SourceText + kk*Slength, DSourceText + kk*Slength, Sheight, Swidth, 255.0, LKArg.Gsigma);
		Generate_Para_Spline(DSourceText + kk*Slength, SourcePara + kk*Slength, Swidth, Sheight, LKArg.InterpAlgo);
	}

	for (jj = 0; jj < Theight; jj++)
	{
		for (ii = 0; ii < Twidth; ii++)
		{
			for (kk = 0; kk < nchannels; kk++)
			{
				if (abs(WarpingParas[ii + jj*Twidth]) < 0.01 && abs(WarpingParas[ii + jj*Twidth + Tlength]) < 0.01)
					wImg[ii + jj*Twidth] = 255;
				else
				{
					xx = WarpingParas[ii + jj*Twidth] + ii, yy = WarpingParas[ii + jj*Twidth + Tlength] + jj;
					Get_Value_Spline(SourcePara + kk*Slength, Swidth, Sheight, xx, yy, S, -1, LKArg.InterpAlgo);
					if (S[0] > 255.0)
						S[0] = 255.0;
					else if (S[0] < 0.0)
						S[0] = 0.0;
					wImg[ii + jj*Twidth + kk*Tlength] = (unsigned char)(int)(S[0] + 0.5);
				}
			}
		}
	}
	sprintf(Fname, "%s/Results/Sep/wST%d_%05d.png", PATH, 1, frameID);
	SaveDataToImage(Fname, wImg, Swidth, Sheight);
	delete[]DSourceText;
	delete[]SourcePara;
	delete[]wImg;


	delete[]SparseCorres1;
	delete[]SparseCorres2;
	delete[]TextImage;
	delete[]SourceText;
	delete[]cROI;
	delete[]lpROI_calculated;
	delete[]disparity;
	delete[]WarpingParas;

	return 0;
}
double TextIllumSepCoarse(int ProID, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceText, double *IPatch, CPoint2 startI, CPoint2 startP, double *direction, double *ILWp, double *TWp, double *Pcom, int hsubset, int searchRange1, int searchRange2, int InterpAlgo, double *PuvIuv, double *ProPatch = 0, double *TextPatch = 0, double *SuperImposePatch = 0, double *ZNNCStorage = 0)
{
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;

	double step = 1.0;
	int ii, jj, ll, kk, qq, mm, nn, rr;
	double II, JJ;
	int length = width*height, plength = pwidth*pheight, nchannels = 1;

	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	bool flag, createdMem = false;
	if (ProPatch == NULL)
	{
		createdMem = true;
		ProPatch = new double[patchLength*nchannels];
		TextPatch = new double[patchLength*nchannels];
		SuperImposePatch = new double[patchLength*nchannels];
		ZNNCStorage = new double[2 * patchLength*nchannels];
	}
	double tryProX, tryProY, tryTx, tryTy, ZNCCscore, bestPu = 0, bestPv = 0, bestIu = 0, bestIv = 0, bestZNCC = -1.0;
	int bestqq = 0, bestll = 0, bestjj = 0, bestii = 0;
	bool printout = false;

	//Now, start searching for projector patch that is on the band of epipolar line
	for (qq = -2; qq <= 2; qq++)
	{
		for (ll = -searchRange1; ll <= searchRange1; ll++)
		{
			tryProX = startP.x + qq + direction[0] * step*ll, tryProY = startP.y + qq + direction[1] * step*ll;
			if (tryProX < hsubset + 1 || tryProX > pwidth - hsubset - 1 || tryProY < hsubset + 1 || tryProY > pheight - hsubset - 1)
				continue;

			flag = true;
			for (jj = -hsubset; jj <= hsubset && flag; jj++)
			{
				for (ii = -hsubset; ii <= hsubset; ii++)
				{
					II = tryProX + ii + ILWp[0] * ii + ILWp[1] * jj;
					JJ = tryProY + jj + ILWp[2] * ii + ILWp[3] * jj;
					if (II<2.0 || II>pwidth - 2 || JJ<2.0 || JJ>pheight - 2)
					{
						flag = false; break;
					}
					for (kk = 0; kk < nchannels; kk++)
						ProPatch[(jj + hsubset)*patchS + (ii + hsubset) + kk*patchLength] = BilinearInterp(Fimgs.PImg + (ProID*nchannels + kk)*plength, pwidth, pheight, II, JJ);
				}
			}

			if (!flag)
				continue;

			if (printout)
			{
				FILE *fp = fopen("C:/temp/tar0.txt", "w+");
				for (jj = 0; jj < patchS; jj++)
				{
					for (ii = 0; ii < patchS; ii++)
						fprintf(fp, "%.2f ", ProPatch[ii + jj*patchS]);
					fprintf(fp, "\n");
				}
				fclose(fp);
			}

			//for every point in the neighbor region of the texture image
			double localZNCC = -1.0;
			for (jj = -searchRange2; jj <= searchRange2; jj++)
			{
				for (ii = -searchRange2; ii < searchRange2; ii++)
				{
					tryTx = startI.x + step*ii, tryTy = startI.y + step*jj;
					if (tryTx<hsubset + 1 || tryTx >width - hsubset - 1 || tryTy<hsubset + 1 || tryTy > height - hsubset - 1)
						continue;

					//Get the texture patch
					flag = true;
					for (mm = -hsubset; mm <= hsubset && flag; mm++)
					{
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							II = tryTx + nn + TWp[0] * nn + TWp[1] * mm;
							JJ = tryTy + mm + TWp[2] * nn + TWp[3] * mm;
							if (II<2.0 || II>width - 2.0 || JJ<2.0 || JJ>height - 2.0)
							{
								flag = false; break;
							}
							for (kk = 0; kk < nchannels; kk++)
								TextPatch[(mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength] = BilinearInterp(SoureTexture + kk*length, width, height, II, JJ);
						}
					}
					if (!flag)
						continue;

					if (printout)
					{
						FILE *fp = fopen("C:/temp/tar1.txt", "w+");
						for (mm = 0; mm < patchS; mm++)
						{
							for (nn = 0; nn < patchS; nn++)
								fprintf(fp, "%.2f ", TextPatch[nn + mm*patchS]);
							fprintf(fp, "\n");
						}
						fclose(fp);
					}

					//Multiply the projected pattern with the texture image at that patch
					for (kk = 0; kk < nchannels; kk++)
					{
						for (mm = -hsubset; mm <= hsubset; mm++)
						{
							for (nn = -hsubset; nn <= hsubset; nn++)
							{
								rr = (mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength;
								SuperImposePatch[rr] = Pcom[0] * ProPatch[rr] * TextPatch[rr] + Pcom[1] * ProPatch[rr];
							}
						}
					}

					//compute zncc score with patch in the illuminated image vs. patch of texture * patch of projected
					ZNCCscore = ComputeZNCCPatch(IPatch, SuperImposePatch, hsubset, nchannels, ZNNCStorage);
					if (ZNCCscore > bestZNCC) //retain the best score
					{
						bestPu = tryProX, bestPv = tryProY, bestIu = tryTx, bestIv = tryTy;
						bestZNCC = ZNCCscore;
						bestqq = qq, bestll = ll, bestjj = jj, bestii = ii;
					}
					if (ZNCCscore > localZNCC) //retain the best score
						localZNCC = ZNCCscore;
				}
			}
			if (printout)
				cout << "@ll: " << ll << " (ZNCC, localZNCC): " << bestZNCC << " " << localZNCC << " (offset, mag, u, v): " << " " << bestqq << " " << bestll << " " << bestii << " " << bestjj << endl;
		}
	}
	PuvIuv[0] = bestPu, PuvIuv[1] = bestPv, PuvIuv[2] = bestIu, PuvIuv[3] = bestIv;

	if (createdMem)
		delete[]ProPatch, delete[]TextPatch, delete[]SuperImposePatch, delete[]ZNNCStorage;
	return bestZNCC;
}
double TextTransSep(int ProID, IlluminationFlowImages &Fimgs, double *ParaSourceText, double *IPatch, double *direction, CPoint2 *Target, LKParameters LKArg, double *ILWp, double *TWp, double *Pcom, double *ZNCCStorage = 0)
{
	int i, j, k, iii, jjj;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;
	int hsubset = LKArg.hsubset, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold;

	double II0, JJ0, II1, JJ1, mF, mG, L, T, Lx, Ly, Tx, Ty, DIC_Coeff, DIC_Coeff_min, a, b, c, t_1, t_2, t_3, t_4, t_5, t_6, S0[3], S1[3], p_best[11];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int jumpStep[2] = { 1, 2 }, nn = 11, nExtraParas = 3, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	double AA[11 * 11], BB[11], CC[11], p[11];
	for (i = 0; i < nn; i++)
		p[i] = (i == nn - nExtraParas ? 1.0 / 255.0 : 0.0);

	p[2] = ILWp[0], p[3] = ILWp[1], p[4] = ILWp[2], p[5] = ILWp[3];
	p[nn - 3] = Pcom[0], p[nn - 2] = Pcom[1], p[nn - 1] = Pcom[2];

	int length = width*height, plength = pwidth*pheight, nchannels = 1, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	FILE *fp, *fp0, *fp1;

	bool printout = false;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (j = -hsubset; j <= hsubset; j++)
		{
			for (i = -hsubset; i <= hsubset; i++)
				fprintf(fp, "%.2f ", IPatch[(i + hsubset) + (j + hsubset)*TimgS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
			{
				fp = fopen("C:/temp/tar.txt", "w+");
				fp0 = fopen("C:/temp/tar0.txt", "w+");
				fp1 = fopen("C:/temp/tar1.txt", "w+");
			}

			a = p[nn - 3], b = p[nn - 2], c = p[nn - 1];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					II0 = Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj;
					JJ0 = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj;

					II1 = Target[1].x + iii + p[6] + TWp[0] * iii + TWp[1] * jjj;
					JJ1 = Target[1].y + jjj + p[7] + TWp[2] * iii + TWp[3] * jjj;

					if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
						continue;
					if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
						continue;

					Get_Value_Spline(Fimgs.PPara + ProID*nchannels*plength, pwidth, pheight, II0, JJ0, S0, 0, Interpolation_Algorithm);
					Get_Value_Spline(ParaSourceText, width, height, II1, JJ1, S1, 0, Interpolation_Algorithm);

					mF = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
					L = S0[0], T = S1[0], mG = L*T;
					Lx = S0[1], Ly = S0[2], Tx = S1[1], Ty = S1[2];

					t_3 = a*L*T + b*L + c - mF;

					t_4 = a*T + b, t_5 = t_4*Lx, t_6 = t_4*Ly;
					CC[0] = t_5, CC[1] = t_6;
					CC[2] = t_5*iii, CC[3] = t_5*jjj;
					CC[4] = t_6*iii, CC[5] = t_6*jjj;

					t_4 = a*L, t_5 = t_4*Tx, t_6 = t_4*Ty;
					CC[6] = t_5, CC[7] = t_6;

					CC[8] = mG, CC[9] = L, CC[10] = 1.0;

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += mF*mF;

					if (printout)
					{
						fprintf(fp, "%.2f ", a*L*T + b*L + c);
						fprintf(fp0, "%.2f ", L);
						fprintf(fp1, "%.2f ", T);
					}
				}
				if (printout)
					fprintf(fp, "\n"), fprintf(fp0, "\n"), fprintf(fp1, "\n");
			}
			if (printout)
				fclose(fp), fclose(fp0), fclose(fp1);

			DIC_Coeff = t_1 / t_2;
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				if (p[0] != p[0])
					return 0.0;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
			}

			if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || DIC_Coeff > 50)// || abs(p[5]) > hsubset || abs(p[6]) > hsubset)
				return 0.0;
			if (abs(p[6]) > hsubset || abs(p[7]) > hsubset) //Texture is drifted
			{
				for (i = 6; i < nn - nExtraParas; i++)
					p[i] = 0.0;
			}
			if (fabs(BB[0]) < conv_crit_1 && abs(BB[1]) < conv_crit_1 && fabs(BB[6]) < conv_crit_1 && fabs(BB[7]) < conv_crit_1)
			{
				for (i = 1; i < 6; i++)
					if (fabs(BB[i]) > conv_crit_2)
						break;
				if (i == 6)
				{
					for (i = 8; i < nn - nExtraParas; i++)
						if (fabs(BB[i]) > conv_crit_2)
							break;
				}
				if (i == nn - nExtraParas)
					Break_Flag = true;
			}
			if (Break_Flag)
			{
				if (k == 0)
				{
					Break_Flag = false;
					conv_crit_1 *= 0.1, conv_crit_2 *= 0.1;
				}
				else
					break;
			}
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}
	/// DIC Iteration: End


	bool createMem = false;
	if (ZNCCStorage == NULL)
	{
		createMem = true;
		ZNCCStorage = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	}

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	if (DIC_Coeff_min < LKArg.PSSDab_thresh)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II0 = Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ0 = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				II1 = Target[1].x + iii + p[6] + TWp[0] * iii + TWp[1] * jjj;
				JJ1 = Target[1].y + jjj + p[7] + TWp[2] * iii + TWp[3] * jjj;

				if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
					continue;
				if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
					continue;

				Get_Value_Spline(Fimgs.PPara + ProID*nchannels*plength, pwidth, pheight, II0, JJ0, S0, -1, Interpolation_Algorithm);
				Get_Value_Spline(ParaSourceText, width, height, II1, JJ1, S1, -1, Interpolation_Algorithm);

				ZNCCStorage[2 * m] = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
				ZNCCStorage[2 * m + 1] = p[8] * S0[0] * S1[0] + p[9] * S0[0] + p[10];// Not needed because of ZNCC
				t_f += ZNCCStorage[2 * m];
				t_g += ZNCCStorage[2 * m + 1];

				if (printout)
					fprintf(fp, "%.2f ", ZNCCStorage[2 * m + 1]);
				m++;
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / (m + 1), t_g = t_g / (m + 1);
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = ZNCCStorage[2 * i] - t_f;
			t_5 = ZNCCStorage[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;
	}

	if (createMem)
		delete[]ZNCCStorage;

	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || DIC_Coeff > 50)// || abs(p[5]) > hsubset || abs(p[6]) > hsubset)
		return 0.0;

	if (abs(p[6]) > hsubset || abs(p[7]) > hsubset)
	{
		for (i = 6; i < nn - nExtraParas; i++)
			p[i] = 0.0;
	}

	//Store the warping parameters
	for (i = 0; i < 4; i++)
		ILWp[i] = p[2 + i];
	Pcom[0] = p[nn - 3], Pcom[1] = p[nn - 2], Pcom[2] = p[nn - 1];

	Target[0].x += p[0], Target[0].y += p[1];
	Target[1].x += p[6], Target[1].y += p[7];

	return DIC_Coeff_min;
}
double TextIIllumAffineSep(int ProID, IlluminationFlowImages &Fimgs, double *ParaSourceText, double *IPatch, double *direction, CPoint2 *Target, LKParameters LKArg, double *ILWp, double *TWp, double *Pcom, double *ZNCCStorage = 0)
{
	int i, j, k, iii, jjj;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;
	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold;

	double II0, JJ0, II1, JJ1, mF, mG, L, T, Lx, Ly, Tx, Ty, DIC_Coeff, DIC_Coeff_min, a, b, c, t_1, t_2, t_3, t_4, t_5, t_6, S0[3], S1[3], p_best[15];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int jumpStep[2] = { 1, 2 }, nn = 15, nExtraParas = 3, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	double AA[15 * 15], BB[15], CC[15], p[15];
	for (i = 0; i < nn; i++)
		p[i] = (i == nn - nExtraParas ? 1.0 / 255.0 : 0.0);

	p[2] = ILWp[0], p[3] = ILWp[1], p[4] = ILWp[2], p[5] = ILWp[3];
	if (TWp != NULL)
		p[8] = TWp[0], p[9] = TWp[1], p[10] = TWp[2], p[11] = TWp[3], p[nn - 3] = Pcom[0], p[nn - 2] = Pcom[1], p[nn - 1] = Pcom[2];

	int length = width*height, plength = pwidth*pheight, nchannels = 1, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	FILE *fp, *fp0, *fp1;

	bool printout = false;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (j = -hsubset; j <= hsubset; j++)
		{
			for (i = -hsubset; i <= hsubset; i++)
				fprintf(fp, "%.2f ", IPatch[(i + hsubset) + (j + hsubset)*TimgS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
			{
				fp = fopen("C:/temp/tar.txt", "w+");
				fp0 = fopen("C:/temp/tar0.txt", "w+");
				fp1 = fopen("C:/temp/tar1.txt", "w+");
			}

			a = p[nn - 3], b = p[nn - 2], c = p[nn - 1];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					II0 = Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj;
					JJ0 = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj;

					II1 = Target[1].x + iii + p[6] + p[8] * iii + p[9] * jjj;
					JJ1 = Target[1].y + jjj + p[7] + p[10] * iii + p[11] * jjj;

					if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
						continue;
					if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
						continue;

					Get_Value_Spline(Fimgs.PPara + ProID*nchannels*plength, pwidth, pheight, II0, JJ0, S0, 0, Interpolation_Algorithm);
					Get_Value_Spline(ParaSourceText, width, height, II1, JJ1, S1, 0, Interpolation_Algorithm);

					mF = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
					L = S0[0], T = S1[0], mG = L*T;
					Lx = S0[1], Ly = S0[2], Tx = S1[1], Ty = S1[2];

					t_3 = a*L*T + b*L + c - mF;

					t_4 = a*T + b, t_5 = t_4*Lx, t_6 = t_4*Ly;
					CC[0] = t_5, CC[1] = t_6;
					CC[2] = t_5*iii, CC[3] = t_5*jjj;
					CC[4] = t_6*iii, CC[5] = t_6*jjj;

					t_4 = a*L + c, t_5 = t_4*Tx, t_6 = t_4*Ty;
					CC[6] = t_5, CC[7] = t_6;
					CC[8] = t_5*iii, CC[9] = t_5*jjj;
					CC[10] = t_6*iii, CC[11] = t_6*jjj;

					CC[12] = mG, CC[13] = L, CC[14] = 1.0;

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += mF*mF;

					if (printout)
					{
						fprintf(fp, "%.2f ", a*L*T + b*L + c);
						fprintf(fp0, "%.2f ", L);
						fprintf(fp1, "%.2f ", T);
					}
				}
				if (printout)
					fprintf(fp, "\n"), fprintf(fp0, "\n"), fprintf(fp1, "\n");
			}
			if (printout)
				fclose(fp), fclose(fp0), fclose(fp1);

			DIC_Coeff = t_1 / t_2;
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
			}


			if (abs((p[2] + 1) / p[3]) + abs((p[5] + 1) / p[4]) < 6 || abs((p[8] + 1) / p[9]) + abs((p[11] + 1) / p[10]) < 10) //weirdly distortted shape-->must be wrong
				return 0;
			if (p[0] != p[0] || abs(p[0]) > hsubset || abs(p[1]) > hsubset || DIC_Coeff > 50)// || abs(p[5]) > hsubset || abs(p[6]) > hsubset)
				return 0.0;
			if (abs(p[6]) > hsubset || abs(p[7]) > hsubset) //Texture is drifted
			{
				for (i = 6; i < nn - 4; i++)
					p[i] = 0.0;
			}
			if (fabs(BB[0]) < conv_crit_1 && abs(BB[1]) < conv_crit_1 && fabs(BB[6]) < conv_crit_1 && fabs(BB[7]) < conv_crit_1)
			{
				for (i = 1; i < 6; i++)
					if (fabs(BB[i]) > conv_crit_2)
						break;
				if (i == 6)
					for (i = 8; i < nn - nExtraParas; i++)
						if (fabs(BB[i]) > conv_crit_2*3.0)
							break;
				if (i == nn - nExtraParas)
					Break_Flag = true;
			}
			if (Break_Flag)
			{
				if (k <= 2)
				{
					Break_Flag = false;
					conv_crit_1 *= 0.1, conv_crit_2 *= 0.1;
				}
				else
					break;
			}
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	if (DIC_Coeff_min < LKArg.PSSDab_thresh)
	{
		bool createMem = false;
		if (ZNCCStorage == NULL)
		{
			createMem = true;
			ZNCCStorage = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
		}

		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II0 = Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ0 = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				II1 = Target[1].x + iii + p[6] + p[8] * iii + p[9] * jjj;
				JJ1 = Target[1].y + jjj + p[7] + p[10] * iii + p[11] * jjj;

				if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
					continue;
				if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
					continue;

				Get_Value_Spline(Fimgs.PPara + ProID*nchannels*plength, pwidth, pheight, II0, JJ0, S0, -1, Interpolation_Algorithm);
				Get_Value_Spline(ParaSourceText, width, height, II1, JJ1, S1, -1, Interpolation_Algorithm);

				ZNCCStorage[2 * m] = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
				ZNCCStorage[2 * m + 1] = p[12] * S0[0] * S1[0] + p[13] * S0[0] + p[14];
				t_f += ZNCCStorage[2 * m], t_g += ZNCCStorage[2 * m + 1];

				if (printout)
					fprintf(fp, "%.2f ", ZNCCStorage[2 * m + 1]);
				m++;
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / (m + 1), t_g = t_g / (m + 1);
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = ZNCCStorage[2 * i] - t_f;
			t_5 = ZNCCStorage[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;

		if (createMem)
			delete[]ZNCCStorage;
	}

	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || DIC_Coeff > 50 || DIC_Coeff > 50)// || abs(p[5]) > hsubset || abs(p[6]) > hsubset)
		return 0.0;

	if (abs(p[6]) > hsubset || abs(p[7]) > hsubset)
	{
		for (i = 6; i < nn - nExtraParas; i++)
			p[i] = 0.0;
		p[nn - 3] = 1.0 / 255, p[nn - 2] = 0.0, p[nn - 1] = 0.0;
	}

	//Store the warping parameters
	for (i = 0; i < 4; i++)
	{
		ILWp[i] = p[2 + i];
		TWp[i] = p[8 + i];
	}
	Pcom[0] = p[nn - 3], Pcom[1] = p[nn - 2], Pcom[2] = p[nn - 1];

	Target[0].x += p[0], Target[0].y += p[1];
	Target[1].x += p[6], Target[1].y += p[7];

	return DIC_Coeff_min;
}
int OptimizeTextIIllum(int ProID, int x, int y, int offx, int offy, int &cp, int &UV_index_n, int &M, double *Coeff, int *lpUV_xy, int *Tindex, double *Fx1, double *P1mat, DevicesInfo &Dinfo, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, bool *cROI, int*visitedPoints, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *SeedType, int *PrecomSearchR, LKParameters LKArg, int mode, double *IPatch, double *TarPatch, double *ProPatch, double *TextPatch, double *SuperImposePatch, double *ZNCCStorage, bool Simulation)
{
	const double intentsityFalloff = 1.0 / 255.0;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nCams = Fimgs.nCams, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int plength = pwidth*pheight, length = width*height;
	int idf, mm, nn, ll, u, v, rangeT;
	int hsubset = LKArg.hsubset, patchS = 2 * hsubset + 1, patchLength = patchS*patchS;

	double t1, t2, score, denum, puviuv[4], ImgPt[3], proEpiline[3], direction[2], ILWp[4], TWp[4], Pcom[3];
	CPoint2  puv, iuv, startP, startT, bef, aft, dPts[2];
	CPoint foundP; CPoint3 WC;
	bool flag, flag2, flag3;

	int success = 0, nx = x + offx, ny = y + offy, id = x + y*width, nid = nx + ny*width, seedtype = SeedType[id], rangeP = 1;
	if (cROI[nid] && abs(ILWarping[nid]) < 0.001 && abs(ILWarping[nid + length]) < 0.001 && visitedPoints[nid] < LKArg.npass2)
	{
		ImgPt[0] = x + offx, ImgPt[1] = y + offy, ImgPt[2] = 1;
		mat_mul(Fx1, ImgPt, proEpiline, 3, 3, 1);
		denum = pow(proEpiline[0], 2) + pow(proEpiline[1], 2);
		direction[0] = -proEpiline[1] / sqrt(denum), direction[1] = proEpiline[0] / sqrt(denum);

		//Get intial guess from computed points
		startP.x = ILWarping[id] + x, startP.y = ILWarping[id + length] + y;
		ILWp[0] = ILWarping[id + 2 * length], ILWp[1] = ILWarping[id + 3 * length], ILWp[2] = ILWarping[id + 4 * length], ILWp[3] = ILWarping[id + 5 * length];
		if (Simulation || LKArg.EpipEnforce == 1)
		{
			t1 = startP.x, t2 = startP.y;
			startP.x = (proEpiline[1] * (proEpiline[1] * t1 - proEpiline[0] * t2) - proEpiline[0] * proEpiline[2]) / denum;
			startP.y = (proEpiline[0] * (-proEpiline[1] * t1 + proEpiline[0] * t2) - proEpiline[1] * proEpiline[2]) / denum;
		}

		flag = false, flag2 = false, flag3 = false;
		if (seedtype == 1) //seeded by a pure illumination point--> try pure illumination
		{
			flag2 = true;
			dPts[0].x = ImgPt[0], dPts[0].y = ImgPt[1], dPts[1].x = startP.x, dPts[1].y = startP.y;
			score = EpipSearchLK(dPts, proEpiline, Fimgs.Img, Fimgs.PImg, Fimgs.Para, Fimgs.PPara, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp);
			if (score > LKArg.ZNCCThreshold)
				flag = true, success = 1, SeedType[nid] = 1 + ProID;

			if (flag)
				ILWarping[nid] = dPts[1].x - ImgPt[0], ILWarping[nid + length] = dPts[1].y - ImgPt[1];
		}

		if (!flag) //either seeded by the texture point or pure illumination fails
		{
			flag3 = false;
			if (seedtype != 4)
			{
				if (!IsLocalWarpAvail(TWarping, TWp, x + offx, y + offy, foundP, u, v, rangeT, width, height, PrecomSearchR[x + offx + (y + offy)*width]))
				{
					if (mode == 0)
					{
						u = x + offx, v = y + offy;
						rangeT = LKArg.searchRange;
						for (ll = 0; ll < 4; ll++)
							TWp[ll] = 0.0;
						Pcom[0] = intentsityFalloff, Pcom[1] = 0.0, Pcom[2] = 0.0;
					}
					else if (mode == 1)
						if (!IsLocalWarpAvail(previousTWarping, TWp, x + offx, y + offy, foundP, u, v, rangeT, width, height, PrecomSearchR[x + offx + (y + offy)*width]))
						{
							visitedPoints[x + offx + (y + offy)*width] += 1;
							return success; // Does not find any closed by points. 
						}
						else
						{
							idf = foundP.x + foundP.y*width;
							Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length], Pcom[2] = PhotoAdj[idf + 2 * length];
							flag3 = true;
						}
				}
				else
				{
					idf = foundP.x + foundP.y*width;
					Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length], Pcom[2] = PhotoAdj[idf + 2 * length];
				}
				startT.x = u, startT.y = v;
			}
			else
			{
				rangeT = 1;
				startT.x = TWarping[id] + x, startT.y = TWarping[id + length] + y;
				TWp[0] = TWarping[id + 2 * length], TWp[1] = TWarping[id + 3 * length], TWp[2] = TWarping[id + 4 * length], TWp[3] = TWarping[id + 5 * length];
				Pcom[0] = PhotoAdj[id], Pcom[1] = PhotoAdj[id + length], Pcom[2] = PhotoAdj[id + 2 * length];
				u = (int)(startT.x) + offx, v = (int)(startT.y) + offy;
			}

			double ssig = SSIG[u + v*width];//ComputeSSIG(ParaSourceTexture,  u,  v, hsubset, width, height, nchannels, LKArg.InterpAlgo);
			if (ssig > LKArg.ssigThresh) // texture is enough
			{
				//Take the observed image patch
				for (ll = 0; ll < nchannels; ll++)
					for (mm = -hsubset; mm <= hsubset; mm++)
						for (nn = -hsubset; nn <= hsubset; nn++)
							IPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength] = Fimgs.Img[(x + offx + nn) + (y + offy + mm)*width + ll*length];

				if (rangeT > 1 || flag3)
				{
					score = TextIllumSepCoarse(ProID, Fimgs, SoureTexture, ParaSourceTexture, IPatch, startT, startP, direction, ILWp, TWp, Pcom, hsubset, rangeP, rangeT, LKArg.InterpAlgo, puviuv, ProPatch, TextPatch, SuperImposePatch, ZNCCStorage);
					ssig = SSIG[(int)puviuv[2] + ((int)puviuv[3])*width];
				}
				else
				{
					puviuv[0] = startP.x, puviuv[1] = startP.y, puviuv[2] = startT.x, puviuv[3] = startT.y;
					score = LKArg.ZNCCThreshold;
				}

				if (ssig > LKArg.ssigThresh && score > LKArg.ZNCCThreshold - 0.35)
				{
					dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
					score = TextTransSep(ProID, Fimgs, ParaSourceTexture, IPatch, direction, dPts, LKArg, ILWp, TWp, Pcom, ZNCCStorage);
					if (score > LKArg.ZNCCThreshold - 0.06)
					{
						if (CamProGeoVerify(1.0*(x + offx), 1.0*(y + offy), dPts, WC, P1mat, Dinfo, width, height, pwidth, pheight, 0, 1.0) == 0)
						{
							score = TextIIllumAffineSep(ProID, Fimgs, ParaSourceTexture, IPatch, direction, dPts, LKArg, ILWp, TWp, Pcom, ZNCCStorage);
							if ((abs(dPts[1].x - nx) < LKArg.DisplacementThresh || abs(dPts[1].y - ny) < LKArg.DisplacementThresh) && score > LKArg.ZNCCThreshold - 0.05)
							{
								flag = true, success = 1, SeedType[nid] = 4 + ProID;
								ILWarping[nid] = dPts[0].x - nx, ILWarping[nid + length] = dPts[0].y - ny, TWarping[nid] = dPts[1].x - nx, TWarping[nid + length] = dPts[1].y - ny;
								for (ll = 0; ll<4; ll++)
									TWarping[nid + (ll + 2)*length] = TWp[ll];
								PhotoAdj[nid] = Pcom[0], PhotoAdj[nid + length] = Pcom[1], PhotoAdj[nid + 2 * length] = Pcom[2];
							}
						}
					}
				}
			}
		}

		if (!flag && !flag2) //try texture first but fail-->let's try illumination flow
		{
			dPts[0].x = ImgPt[0], dPts[0].y = ImgPt[1];
			dPts[1].x = startP.x, dPts[1].y = startP.y;
			score = EpipSearchLK(dPts, proEpiline, Fimgs.Img, Fimgs.PImg + ProID*nchannels*plength, Fimgs.Para, Fimgs.PPara + ProID*nchannels*plength, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp);
			if (score>LKArg.ZNCCThreshold - 0.01)
				flag = true, success = 1, SeedType[nid] = 1 + ProID;

			if (flag)
				ILWarping[nid] = dPts[1].x - ImgPt[0], ILWarping[nid + length] = dPts[1].y - ImgPt[1];
		}

		if (flag)
		{
			cp++, UV_index_n++, M++;
			cROI[nid] = false; Coeff[M] = 1.0 - score;
			lpUV_xy[2 * UV_index_n] = x + offx, lpUV_xy[2 * UV_index_n + 1] = y + offy;
			Tindex[M] = UV_index_n; DIC_AddtoQueue(Coeff, Tindex, M);
			for (ll = 0; ll < 4; ll++)
				ILWarping[nid + (ll + 2)*length] = ILWp[ll];
		}
		visitedPoints[nid] += 1;
	}

	return success;
}
int IllumTextureSeperation(int frameID, int ProID, char *PATH, char *TPATH, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, DevicesInfo &DInfo, float *ILWarping, float *TWarping, float *previousTWarping, float *PhotoAdj, int *SeedType, int *PrecomSearchR, LKParameters LKArg, int mode, bool *cROI, int part, bool Simulation)
{
	//Assume there is one camera
	const double intentsityFalloff = 1.0 / 255.0;
	int id, idf, seededsucces, ii, jj, kk, ll, mm, nn, x, y, u, v, rangeP, rangeT;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nCams = Fimgs.nCams, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int plength = pwidth*pheight, length = width*height;
	int hsubset = LKArg.hsubset, InterpAlgo = LKArg.InterpAlgo;
	bool flag, flag2;

	double score, denum, ImgPt[3], proEpiline[3], direction[2], ILWp[4], TWp[4], Pcom[3];
	CPoint2  puv, iuv, startI, startP, startT, bef, aft, dPts[2];
	CPoint foundP;
	CPoint3 WC;

	//Set up camera-projector calibration paras
	double P1mat[24], FCP[9];
	mat_transpose(DInfo.FmatPC + 9 * ProID, FCP, 3, 3);
	P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0,
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0,
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;
	P1mat[12 + 0] = DInfo.P[0], P1mat[12 + 1] = DInfo.P[1], P1mat[12 + 2] = DInfo.P[2], P1mat[12 + 3] = DInfo.P[3],
		P1mat[12 + 4] = DInfo.P[4], P1mat[12 + 5] = DInfo.P[5], P1mat[12 + 6] = DInfo.P[6], P1mat[12 + 7] = DInfo.P[7],
		P1mat[12 + 8] = DInfo.P[8], P1mat[12 + 9] = DInfo.P[9], P1mat[12 + 10] = DInfo.P[10], P1mat[12 + 11] = DInfo.P[11];

	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	double *IPatch = new double[patchLength*nchannels];
	double *TarPatch = new double[patchLength*nchannels];
	double *ProPatch = new double[patchLength*nchannels];
	double *TextPatch = new double[patchLength*nchannels];
	double *SuperImposePatch = new double[patchLength*nchannels];
	double *ZNCCStorage = new double[2 * patchLength*nchannels];

	double puviuv[4];
	int pointsToCompute = 0, pointsComputed = 0, cp, M, UV_index = 0, UV_index_n = 0;
	int *visitedPoints = new int[length];
	int *Tindex = new int[length];
	int *lpUV_xy = new int[2 * length];
	double *Coeff = new double[length];

	for (jj = 0; jj < height; jj += LKArg.step)
	{
		for (ii = 0; ii < width; ii += LKArg.step)
		{
			id = ii + jj*width;
			if (cROI[id])
			{
				visitedPoints[id] = 0;
				if (abs(ILWarping[id]) < 0.01 && abs(ILWarping[id + length]) < 0.01)
					pointsToCompute++;
			}
		}
	}
#pragma omp critical
	cout << "Partition #" << part << " deals with " << pointsToCompute << " pts." << endl;

	if (pointsToCompute < 500)
	{
#pragma omp critical
		cout << "Partition #" << part << " terminates because #points to compute is to small." << endl;
	}
	else
	{
		char Fname[200];
		int percent = 50, increP = 50;
		double start = omp_get_wtime();
		for (kk = 0; kk < LKArg.npass2; kk++) //do npass
		{
			for (jj = 0; jj < height; jj += LKArg.step)
			{
				for (ii = 0; ii < width; ii += LKArg.step)
				{
					cp = 0, M = -1; id = ii + jj*width;
					if (cROI[id] && abs(ILWarping[id]) < 0.001 && abs(ILWarping[id + length]) < 0.001 && visitedPoints[id] < LKArg.npass2)
					{
						M = 0; UV_index = UV_index_n;
						lpUV_xy[2 * UV_index] = ii, lpUV_xy[2 * UV_index + 1] = jj;

						//Search in the local region for computed cam-pro correspondence
						if (!IsLocalWarpAvail(ILWarping, ILWp, ii, jj, foundP, x, y, rangeP, width, height, PrecomSearchR[ii + jj*width]))
							continue; // Does not find any closed by points. 

						ImgPt[0] = ii, ImgPt[1] = jj, ImgPt[2] = 1;
						mat_mul(FCP, ImgPt, proEpiline, 3, 3, 1);
						denum = pow(proEpiline[0], 2) + pow(proEpiline[1], 2);
						direction[0] = -proEpiline[1] / sqrt(denum), direction[1] = proEpiline[0] / sqrt(denum);

						if (Simulation || LKArg.EpipEnforce == 1)//project the point to epipolar line: http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
						{
							startP.x = (proEpiline[1] * (proEpiline[1] * x - proEpiline[0] * y) - proEpiline[0] * proEpiline[2]) / denum;
							startP.y = (proEpiline[0] * (-proEpiline[1] * x + proEpiline[0] * y) - proEpiline[1] * proEpiline[2]) / denum;
						}
						else
							startP.x = x, startP.y = y;
						startI.x = ii, startI.y = jj;

						flag = false, flag2 = false;
						if (PrecomSearchR[ii + jj*width] > 4) //Not deep inside texture region
						{
							dPts[0].x = startI.x, dPts[0].y = startI.y;
							dPts[1].x = startP.x, dPts[1].y = startP.y;

							score = EpipSearchLK(dPts, proEpiline, Fimgs.Img, Fimgs.PImg + ProID*nchannels*plength, Fimgs.Para, Fimgs.PPara + ProID*nchannels*plength, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp);
							if (score > LKArg.ZNCCThreshold)
								flag = true, SeedType[id] = 1 + ProID;

							if (flag)
								ILWarping[id] = dPts[1].x - ii, ILWarping[id + length] = dPts[1].y - jj;
						}

						if (PrecomSearchR[ii + jj*width] <= 4 || !flag)
						{
							if (!IsLocalWarpAvail(TWarping, TWp, ii, jj, foundP, u, v, rangeT, width, height, PrecomSearchR[ii + jj*width]))
							{
								if (mode == 0)
								{
									u = ii, v = jj;
									rangeT = LKArg.searchRange;
									for (ll = 0; ll < 4; ll++)
										TWp[ll] = 0.0;
									Pcom[0] = intentsityFalloff, Pcom[1] = 0.0, Pcom[2] = 0.0;
								}
								else
									if (!IsLocalWarpAvail(previousTWarping, TWp, ii, jj, foundP, u, v, rangeT, width, height, PrecomSearchR[ii + jj*width]))
										continue; // Does not find any closed by points. 
									else
									{
										idf = foundP.x + foundP.y*width;
										Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length], Pcom[2] = PhotoAdj[idf + 2 * length];
									}
							}
							else
							{
								idf = foundP.x + foundP.y*width;
								Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length], Pcom[2] = PhotoAdj[idf + 2 * length];
							}

							double ssig = SSIG[u + v*width];
							if (ssig > LKArg.ssigThresh) // texture is enough
							{
								for (ll = 0; ll<nchannels; ll++)
									for (mm = -hsubset; mm <= hsubset; mm++)
										for (nn = -hsubset; nn <= hsubset; nn++)
											IPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength] = Fimgs.Img[(ii + nn) + (jj + mm)*width + ll*length];

								startT.x = u, startT.y = v;
								score = TextIllumSepCoarse(ProID, Fimgs, SoureTexture, ParaSourceTexture, IPatch, startT, startP, direction, ILWp, TWp, Pcom, hsubset, rangeP, rangeT, LKArg.InterpAlgo, puviuv, ProPatch, TextPatch, SuperImposePatch, ZNCCStorage);
								ssig = SSIG[(int)puviuv[2] + ((int)puviuv[3])*width];
								if (ssig>LKArg.ssigThresh && score > LKArg.ZNCCThreshold - 0.35)
								{
									dPts[0].x = puviuv[0], dPts[0].y = puviuv[1];
									dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
									score = TextTransSep(ProID, Fimgs, ParaSourceTexture, IPatch, direction, dPts, LKArg, ILWp, TWp, Pcom, ZNCCStorage);
									if (score > LKArg.ZNCCThreshold - 0.05)
									{
										double thresh = 1.0;
										if (CamProGeoVerify(1.0*ii, 1.0*jj, dPts, WC, P1mat, DInfo, width, height, pwidth, pheight, ProID, thresh) == 0)
										{
											score = TextIIllumAffineSep(ProID, Fimgs, ParaSourceTexture, IPatch, direction, dPts, LKArg, ILWp, TWp, Pcom, ZNCCStorage);
											if ((abs(dPts[1].x - ii) < LKArg.DisplacementThresh || abs(dPts[1].y - jj) < LKArg.DisplacementThresh) && score > LKArg.ZNCCThreshold - 0.04)
											{
												flag = true, SeedType[id] = 4 + ProID;
												ILWarping[id] = dPts[0].x - ii, ILWarping[id + length] = dPts[0].y - jj, TWarping[id] = dPts[1].x - ii, TWarping[id + length] = dPts[1].y - jj;
												for (ll = 0; ll < 4; ll++)
													TWarping[id + (ll + 2)*length] = TWp[ll];
												PhotoAdj[id] = Pcom[0], PhotoAdj[id + length] = Pcom[1], PhotoAdj[id + 2 * length] = Pcom[2];
											}
										}
									}
								}
							}
						}

						if (flag)
						{
							cp++; cROI[id] = false;
							Coeff[M] = 1.0 - score, Tindex[M] = UV_index;
							for (ll = 0; ll < 4; ll++)
								ILWarping[id + (ll + 2)*length] = ILWp[ll];
						}
						else
							M--;

						visitedPoints[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width] += 1;
					}

					//Now, PROPAGATE
					seededsucces = 0;
					while (M >= 0)
					{
						if (100 * (UV_index_n + 1) / pointsToCompute >= percent)
						{
							double elapsed = omp_get_wtime() - start;
							cout << "Partition #" << part << " ..." << 100 * (UV_index_n + 1) / pointsToCompute << "% ... #" << pointsComputed + cp << " points.TE: " << setw(2) << elapsed << " TR: " << setw(2) << elapsed / (percent + increP)*(100.0 - percent) << endl;
							percent += increP;

#pragma omp critical
							{
								for (int ll = 0; ll < 6; ll++)
								{
									sprintf(Fname, "%s/%05d_C1P%dp%d.dat", TPATH, frameID, ProID + 1, ll);
									WriteGridBinary(Fname, ILWarping + ll*length, width, height);
								}
								for (int ll = 0; ll < 6; ll++)
								{
									sprintf(Fname, "%s/%05d_C1TSp%d.dat", TPATH, frameID, ll);
									WriteGridBinary(Fname, TWarping + ll*length, width, height);
								}
								for (int ll = 0; ll < 2; ll++)
								{
									sprintf(Fname, "%s/Results/Sep/%05d_C1PA_%d.dat", TPATH, frameID, ll);
									WriteGridBinary(Fname, PhotoAdj + ll*length, width, height);
								}
								sprintf(Fname, "%s/%05d_SeedType.dat", TPATH, frameID); WriteGridBinary(Fname, SeedType, width, height);

								sprintf(Fname, "%s/Results/Sep", PATH);
								UpdateIllumTextureImages(Fname, false, frameID, mode, 1, ProID, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara + ProID*nchannels*plength, ILWarping, ParaSourceTexture, TWarping);
							}
						}

						UV_index = Tindex[M];
						x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
						M--;

						seededsucces += OptimizeTextIIllum(ProID, x, y, 0, 1, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, FCP, P1mat + 12 * ProID, DInfo, Fimgs, SoureTexture, ParaSourceTexture, SSIG, cROI, visitedPoints, ILWarping, TWarping, PhotoAdj, previousTWarping, SeedType, PrecomSearchR,
							LKArg, mode, IPatch, TarPatch, ProPatch, TextPatch, SuperImposePatch, ZNCCStorage, Simulation);

						seededsucces += OptimizeTextIIllum(ProID, x, y, 0, -1, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, FCP, P1mat + 12 * ProID, DInfo, Fimgs, SoureTexture, ParaSourceTexture, SSIG, cROI, visitedPoints, ILWarping, TWarping, PhotoAdj, previousTWarping, SeedType, PrecomSearchR,
							LKArg, mode, IPatch, TarPatch, ProPatch, TextPatch, SuperImposePatch, ZNCCStorage, Simulation);

						seededsucces += OptimizeTextIIllum(ProID, x, y, 1, 0, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, FCP, P1mat + 12 * ProID, DInfo, Fimgs, SoureTexture, ParaSourceTexture, SSIG, cROI, visitedPoints, ILWarping, TWarping, PhotoAdj, previousTWarping, SeedType, PrecomSearchR,
							LKArg, mode, IPatch, TarPatch, ProPatch, TextPatch, SuperImposePatch, ZNCCStorage, Simulation);

						seededsucces += OptimizeTextIIllum(ProID, x, y, -1, 0, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, FCP, P1mat + 12 * ProID, DInfo, Fimgs, SoureTexture, ParaSourceTexture, SSIG, cROI, visitedPoints, ILWarping, TWarping, PhotoAdj, previousTWarping, SeedType, PrecomSearchR,
							LKArg, mode, IPatch, TarPatch, ProPatch, TextPatch, SuperImposePatch, ZNCCStorage, Simulation);
					}

#pragma omp critical
					if (seededsucces > 1000)
					{
						sprintf(Fname, "%s/Results/Sep", PATH);
						UpdateIllumTextureImages(Fname, false, frameID, mode, 1, ProID, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara + ProID*nchannels*plength, ILWarping, ParaSourceTexture, TWarping);
					}

					if (cp > 0)
						UV_index_n++;
					pointsComputed += cp;
				}
			}
		}

#pragma omp critical
		{
			for (int ll = 0; ll < 6; ll++)
			{
				sprintf(Fname, "%s/%05d_C1P%dp%d.dat", TPATH, frameID, ProID + 1, ll);
				WriteGridBinary(Fname, ILWarping + ll*length, width, height);
			}
			for (int ll = 0; ll < 6; ll++)
			{
				sprintf(Fname, "%s/%05d_C1TSp%d.dat", TPATH, frameID, ll);
				WriteGridBinary(Fname, TWarping + ll*length, width, height);
			}
			for (int ll = 0; ll < 2; ll++)
			{
				sprintf(Fname, "%s/%05d_C1PA_%d.dat", TPATH, frameID, ll);
				WriteGridBinary(Fname, PhotoAdj + ll*length, width, height);
			}
			sprintf(Fname, "%s/%05d_SeedType.dat", TPATH, frameID); WriteGridBinary(Fname, SeedType, width, height);
			sprintf(Fname, "%s/Results/Sep", PATH);
			UpdateIllumTextureImages(Fname, false, frameID, mode, 1, ProID, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara, ILWarping, ParaSourceTexture, TWarping);
		}

		double elapsed = omp_get_wtime() - start;
		cout << "Partition #" << part << "... " << 100 * pointsComputed / pointsToCompute << "% (" << pointsComputed << " pts) in " << omp_get_wtime() - start << "s" << endl;

		//Kill all the bad poitns:
		if (Fimgs.nPros == 1)
			for (jj = 0; jj < height; jj++)
			{
				for (ii = 0; ii < width; ii++)
				{
					if (abs(ILWarping[ii + jj*width]) < 0.001 && abs(ILWarping[ii + jj*width + length]) < 0.001)
					{
						for (ll = 0; ll < 6; ll++)
							TWarping[ii + jj*width + ll*length] = 0.0;
						PhotoAdj[ii + jj*width] = intentsityFalloff;
						for (ll = 1; ll < 3; ll++)
							PhotoAdj[ii + jj*width + ll*length] = 0.0;
					}
				}
			}
	}

	delete[]IPatch;
	delete[]ProPatch;
	delete[]TextPatch;
	delete[]SuperImposePatch;
	delete[]ZNCCStorage;
	delete[]visitedPoints;
	delete[]Tindex;
	delete[]lpUV_xy;
	delete[]Coeff;

	return 0;
}

int MultiViewGeoVerify(CPoint2 *Pts, double *Pmat, double *K, double *distortion, bool *PassedPoints, int width, int height, int pwidth, int pheight, int nviews, int npts, double thresh, CPoint2 *apts, CPoint2 *bkapts,
	int *DeviceMask, double *tK, double *tdistortion, double *tP, double *A, double *B)
{
	int ii, jj, kk, devCount;
	CPoint3 WC;
	bool createMem = false;
	double error;

	if (apts == NULL)
	{
		createMem = true;
		apts = new CPoint2[nviews], bkapts = new CPoint2[nviews], DeviceMask = new int[nviews];
		tK = new double[9 * nviews], tdistortion = new double[13 * nviews], tP = new double[12 * nviews];
	}

	for (ii = 0; ii < npts; ii++)
	{
		devCount = 0;
		for (jj = 0; jj < nviews; jj++)
		{
			DeviceMask[jj] = 1;
			if (Pts[jj + ii*nviews].x < 1 || Pts[jj + ii*nviews].y < 1)
				DeviceMask[jj] = 0;
			else
			{
				bkapts[devCount].x = Pts[jj + ii*nviews].x, bkapts[devCount].y = Pts[jj + ii*nviews].y;
				apts[devCount].x = bkapts[devCount].x, apts[devCount].y = bkapts[devCount].y;
				Undo_distortion(apts[devCount], K + 9 * jj, distortion + 13 * jj);

				for (kk = 0; kk < 12; kk++)
					tP[12 * devCount + kk] = Pmat[12 * jj + kk];
				for (kk = 0; kk < 9; kk++)
					tK[9 * devCount + kk] = K[9 * jj + kk];
				for (kk = 0; kk < 13; kk++)
					tdistortion[13 * devCount + kk] = distortion[13 * jj + kk];
				devCount++;
			}
		}

		NviewTriangulation(apts, tP, &WC, devCount, 1, NULL, A, B);
		ProjectandDistort(WC, apts, tP, tK, tdistortion, devCount);

		error = 0.0;
		for (jj = 0; jj < devCount; jj++)
			error += pow(bkapts[jj].x - apts[jj].x, 2) + pow(bkapts[jj].y - apts[jj].y, 2);
		error = sqrt(error / devCount);
		if (error < thresh)
			PassedPoints[ii] = true;
		else
			PassedPoints[ii] = false;
	}

	if (createMem)
	{
		delete[]apts, delete[]bkapts, delete[]DeviceMask;
		delete[]tK, delete[]tdistortion, delete[]tP;
	}

	return 0;
}
int CamProGeoVerify(double cx, double cy, CPoint2 *Pxy, CPoint3 &WC, double *P1mat, DevicesInfo &DInfo, int width, int height, int pwidth, int pheight, int pview, double thresh)
{
	//pview == 2: both pros, pview == 0: pro 1, pview == 1: pro 2
	double u1, v1, u2, v2, denum, reprojectionError;
	CPoint2 Ppts, Cpts;;
	int nPros = DInfo.nPros;

	if (pview == 2)
	{
		CPoint2 pts[3];
		for (int ii = 0; ii < nPros; ii++)
		{
			pts[ii].x = Pxy[ii].x, pts[ii].y = Pxy[ii].y;
			Undo_distortion(pts[ii], DInfo.K + 9 * ii, DInfo.distortion + 13 * ii);
		}
		pts[nPros].x = cx, pts[nPros].y = cy;
		Undo_distortion(pts[nPros], DInfo.K + 9 * nPros, DInfo.distortion + 13 * nPros);

		double P[12 * 3];
		for (int ii = 0; ii < 12; ii++)
			P[ii] = P1mat[ii], P[12 + ii] = P1mat[12 + ii], P[24 + ii] = DInfo.P[12 * (nPros - 1) + ii];

		NviewTriangulation(pts, P, &WC, nPros + 1);

		//Project to images
		CPoint2 projectedPts[3];
		for (int ii = 0; ii < nPros; ii++)
		{
			denum = P1mat[12 * ii + 8] * WC.x + P1mat[12 * ii + 9] * WC.y + P1mat[12 * ii + 10] * WC.z + P1mat[12 * ii + 11];
			projectedPts[ii].x = (P1mat[12 * ii] * WC.x + P1mat[12 * ii + 1] * WC.y + P1mat[12 * ii + 2] * WC.z + P1mat[12 * ii + 3]) / denum;
			projectedPts[ii].y = (P1mat[12 * ii + 4] * WC.x + P1mat[12 * ii + 5] * WC.y + P1mat[12 * ii + 6] * WC.z + P1mat[12 * ii + 7]) / denum;
			if (projectedPts[ii].x < 0 || projectedPts[ii].x> pwidth - 1 || projectedPts[ii].y < 0 || projectedPts[ii].y > pheight - 1)
				return 1;
		}
		denum = DInfo.P[8 + 12 * (nPros - 1)] * WC.x + DInfo.P[9 + 12 * (nPros - 1)] * WC.y + DInfo.P[10 + 12 * (nPros - 1)] * WC.z + DInfo.P[11 + 12 * (nPros - 1)];
		projectedPts[2].x = (DInfo.P[0 + 12 * (nPros - 1)] * WC.x + DInfo.P[1 + 12 * (nPros - 1)] * WC.y + DInfo.P[2 + 12 * (nPros - 1)] * WC.z + DInfo.P[3 + 12 * (nPros - 1)]) / denum;
		projectedPts[2].y = (DInfo.P[4 + 12 * (nPros - 1)] * WC.x + DInfo.P[5 + 12 * (nPros - 1)] * WC.y + DInfo.P[6 + 12 * (nPros - 1)] * WC.z + DInfo.P[7 + 12 * (nPros - 1)]) / denum;
		if (projectedPts[2].x < 0 || projectedPts[2].x> width - 1 || projectedPts[2].y < 0 || projectedPts[2].y > height - 1)
			return 1;

		reprojectionError = abs(projectedPts[2].x - pts[nPros].x) + abs(projectedPts[2].y - pts[nPros].y);
		for (int ii = 0; ii < nPros; ii++)
			reprojectionError += abs(projectedPts[ii].x - pts[ii].x) + abs(projectedPts[ii].y - pts[ii].y);
		reprojectionError = reprojectionError / 2.0 / (nPros + 1);

		if (reprojectionError > thresh)
			return 1;
		else
			return 0;
	}
	else
	{
		Cpts.x = cx, Cpts.y = cy;
		Ppts.x = Pxy[0].x, Ppts.y = Pxy[0].y;

		Undo_distortion(Cpts, DInfo.K + 9 * nPros, DInfo.distortion + 13 * nPros);
		Undo_distortion(Ppts, DInfo.K + 9 * pview, DInfo.distortion + 13 * pview);
		Stereo_Triangulation2(&Ppts, &Cpts, P1mat + 12 * pview, DInfo.P + 12 * (nPros - 1), &WC);

		//Project to projector image 
		denum = P1mat[12 * pview + 8] * WC.x + P1mat[12 * pview + 9] * WC.y + P1mat[12 * pview + 10] * WC.z + P1mat[12 * pview + 11];
		u1 = (P1mat[12 * pview] * WC.x + P1mat[12 * pview + 1] * WC.y + P1mat[12 * pview + 2] * WC.z + P1mat[12 * pview + 3]) / denum;
		v1 = (P1mat[12 * pview + 4] * WC.x + P1mat[12 * pview + 5] * WC.y + P1mat[12 * pview + 6] * WC.z + P1mat[12 * pview + 7]) / denum;

		//Project to Camera image 
		denum = DInfo.P[8 + 12 * (nPros - 1)] * WC.x + DInfo.P[9 + 12 * (nPros - 1)] * WC.y + DInfo.P[10 + 12 * (nPros - 1)] * WC.z + DInfo.P[11 + 12 * (nPros - 1)];
		u2 = (DInfo.P[0 + 12 * (nPros - 1)] * WC.x + DInfo.P[1 + 12 * (nPros - 1)] * WC.y + DInfo.P[2 + 12 * (nPros - 1)] * WC.z + DInfo.P[3 + 12 * (nPros - 1)]) / denum;
		v2 = (DInfo.P[4 + 12 * (nPros - 1)] * WC.x + DInfo.P[5 + 12 * (nPros - 1)] * WC.y + DInfo.P[6 + 12 * (nPros - 1)] * WC.z + DInfo.P[7 + 12 * (nPros - 1)]) / denum;

		reprojectionError = (abs(u2 - Cpts.x) + abs(v2 - Cpts.y) + abs(u1 - Ppts.x) + abs(v1 - Ppts.y)) / 4.0;
		if (reprojectionError > thresh || u1<0 || u1>pwidth - 1 || v1<0 || v1>pheight - 1 || u2 < 0 || u2 > width - 1 || v2<0 || v2 > height - 1)
			return 1;
		else
			return 0;
	}
}
int EstimateIllumPatchAffine(int startX, int startY, int ProID1, int ProID2, double *P1mat, float *ILWarping, CPoint2 &dstXY, double *IWP, DevicesInfo &DInfo, int range, int width, int height, int pwidth, int pheight, CPoint2 *From = 0, CPoint2 *To = 0, double *A = 0, double *B = 0)
{
	int mm, nn, idx, idy, count, minDist, Dist, nPros = DInfo.nPros, length = width*height, offset = 6 * ProID1*length;
	double denum;
	CPoint2 Ppts, Cpts; CPoint3 WC;

	//Find the patch from computed ILwarping
	count = 0, minDist = 100;
	for (mm = -range; mm <= range; mm += 2)
	{
		for (nn = -range; nn <= range; nn += 2)
		{
			if (abs(ILWarping[(startX + nn) + (startY + mm)*width + offset]) + abs(ILWarping[(startX + nn) + (startY + mm)*width + length + offset]) > 0.001)
			{
				//Project to 3D
				idx = startX + nn, idy = startY + mm;
				Cpts.x = 1.0*idx, Cpts.y = 1.0*idy;
				Ppts.x = ILWarping[idx + idy*width + offset] + idx, Ppts.y = ILWarping[idx + idy*width + length + offset] + idy;

				Undo_distortion(Cpts, DInfo.K + 9 * nPros, DInfo.distortion + 13 * nPros);
				Undo_distortion(Ppts, DInfo.K + 9 * ProID1, DInfo.distortion + 13 * ProID1);
				Stereo_Triangulation2(&Ppts, &Cpts, P1mat + 12 * ProID1, DInfo.P + 12 * (nPros - 1), &WC);

				//Back project to the projector 2
				denum = P1mat[12 * ProID2 + 8] * WC.x + P1mat[12 * ProID2 + 9] * WC.y + P1mat[12 * ProID2 + 10] * WC.z + P1mat[12 * ProID2 + 11];
				To[count].x = (P1mat[12 * ProID2] * WC.x + P1mat[12 * ProID2 + 1] * WC.y + P1mat[12 * ProID2 + 2] * WC.z + P1mat[12 * ProID2 + 3]) / denum;
				To[count].y = (P1mat[12 * ProID2 + 4] * WC.x + P1mat[12 * ProID2 + 5] * WC.y + P1mat[12 * ProID2 + 6] * WC.z + P1mat[12 * ProID2 + 7]) / denum;

				From[count].x = idx, From[count].y = idy;

				Dist = nn*nn + mm*mm;
				if (Dist < minDist)
				{
					minDist = Dist;
					dstXY.x = To[count].x, dstXY.y = To[count].y;
				}
				count++;
			}
		}
	}

	if (dstXY.x < -20 || dstXY.y < -20 || dstXY.x > pwidth + 20 || dstXY.y > pheight + 20)
		return 0;

	//uv = H*xy: trying to warp camera image to projector so that the camera image can be resynthesized from the projector
	double AHomo[6];
	double error = Compute_AffineHomo(From, To, count, AHomo, A, B);
	if (error < 2.0)
	{
		IWP[0] = AHomo[0] - 1.0, IWP[1] = AHomo[1], IWP[2] = AHomo[3], IWP[3] = AHomo[4] - 1.0;
		return (int)sqrt(minDist) + 1;
	}
	else
		return 0;
}
double llumSepCoarse(IlluminationFlowImages &Fimgs, double *IPatch, CPoint2 *startP, double *direction, double *ILWp, double *Pcom, int hsubset, int searchRange1, int searchRange2, int InterpAlgo, double *PuvIuv, double *Pro1Patch = 0, double *Pro2Patch = 0, double *SuperImposePatch = 0, double *ZNNCStorage = 0)
{
	double step = 1.0;
	int ii, jj, ll, kk, qq, mm, nn, rr;
	double II, JJ;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;
	int length = width*height, plength = pwidth*pheight, nchannels = Fimgs.nchannels;

	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	bool flag, createdMem = false;
	if (Pro1Patch == NULL)
	{
		createdMem = true;
		Pro1Patch = new double[patchLength*nchannels];
		Pro2Patch = new double[patchLength*nchannels];
		SuperImposePatch = new double[patchLength*nchannels];
		ZNNCStorage = new double[2 * patchLength*nchannels];
	}
	double tryPro1X, tryPro1Y, tryPro2X, tryPro2Y, ZNCCscore, bestP1u = 0, bestP1v = 0, bestP2u = 0, bestP2v = 0, bestZNCC = -1.0;
	int bestqq = 0, bestll = 0, bestjj = 0, bestii = 0;
	bool printout = false, printout2 = false;

	//Now, start searching for projector patch that is on the band of epipolar line
	for (qq = -1; qq <= 1; qq++)
	{
		for (ll = -searchRange1; ll <= searchRange1; ll++)
		{
			tryPro1X = startP[0].x + qq + direction[0] * step*ll, tryPro1Y = startP[0].y + qq + direction[1] * step*ll;
			if (tryPro1X <= hsubset || tryPro1X >= pwidth - hsubset || tryPro1Y <= hsubset || tryPro1Y >= pheight - hsubset)
				continue;

			//Take a prewarped patch in Pro 1
			flag = true;
			for (jj = -hsubset; jj <= hsubset && flag; jj++)
			{
				for (ii = -hsubset; ii <= hsubset; ii++)
				{
					II = tryPro1X + ii + ILWp[0] * ii + ILWp[1] * jj;
					JJ = tryPro1Y + jj + ILWp[2] * ii + ILWp[3] * jj;
					if (II<0 || II>pwidth - 1 || JJ<0 || JJ>pheight - 1)
					{
						flag = false;
						break;
					}
					for (kk = 0; kk < nchannels; kk++)
						Pro1Patch[(jj + hsubset)*patchS + (ii + hsubset) + kk*patchLength] = BilinearInterp(Fimgs.PImg + kk*plength, pwidth, pheight, II, JJ);
				}
			}

			if (!flag)
				continue;

			if (printout)
			{
				FILE *fp = fopen("C:/temp/tar0.txt", "w+");
				for (jj = 0; jj < patchS; jj++)
				{
					for (ii = 0; ii < patchS; ii++)
						fprintf(fp, "%.2f ", Pro1Patch[ii + jj*patchS]);
					fprintf(fp, "\n");
				}
				fclose(fp);
			}

			//Take a prewarped patch in Pro 2
			double localZNCC = -1.0;
			for (jj = -1; jj <= 1; jj++)
			{
				for (ii = -searchRange2; ii <= searchRange2; ii++)
				{
					tryPro2X = startP[1].x + jj + direction[2] * step*ii, tryPro2Y = startP[1].y + jj + direction[3] * step*ii;
					if (tryPro2X <= hsubset - 2 || tryPro2X >= pwidth - hsubset + 2 || tryPro2Y <= hsubset - 2 || tryPro2Y >= pheight - hsubset + 2)
						continue;

					flag = true;
					for (mm = -hsubset; mm <= hsubset && flag; mm++)
					{
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							II = tryPro2X + nn + ILWp[4] * nn + ILWp[5] * mm;
							JJ = tryPro2Y + mm + ILWp[6] * nn + ILWp[7] * mm;
							if (II<0 || II>pwidth - 1 || JJ<0 || JJ>pheight - 1)
							{
								flag = false;
								break;
							}
							for (kk = 0; kk < nchannels; kk++)
								Pro2Patch[(mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength] = BilinearInterp(Fimgs.PImg + (nchannels + kk)*plength, pwidth, pheight, II, JJ);
						}
					}
					if (!flag)
						continue;

					if (printout)
					{
						FILE *fp = fopen("C:/temp/tar1.txt", "w+");
						for (mm = 0; mm < patchS; mm++)
						{
							for (nn = 0; nn < patchS; nn++)
								fprintf(fp, "%.2f ", Pro2Patch[nn + mm*patchS]);
							fprintf(fp, "\n");
						}
						fclose(fp);
					}

					//Add the projected patterns at that patch
					for (kk = 0; kk < nchannels; kk++)
					{
						for (mm = -hsubset; mm <= hsubset; mm++)
						{
							for (nn = -hsubset; nn <= hsubset; nn++)
							{
								rr = (mm + hsubset)*patchS + (nn + hsubset) + kk*patchLength;
								SuperImposePatch[rr] = Pcom[0] * Pro1Patch[rr] + Pcom[1] * Pro2Patch[rr];
							}
						}
					}

					//compute zncc score with patch in the illuminated image vs. patch of texture * patch of projected
					ZNCCscore = ComputeZNCCPatch(IPatch, SuperImposePatch, hsubset, nchannels, ZNNCStorage);
					if (ZNCCscore > bestZNCC) //retain the best score
					{
						bestP1u = tryPro1X, bestP1v = tryPro1Y, bestP2u = tryPro2X, bestP2v = tryPro2Y;
						bestZNCC = ZNCCscore;
						bestqq = qq, bestll = ll, bestjj = jj, bestii = ii;
					}
					if (ZNCCscore > localZNCC) //retain the best score
						localZNCC = ZNCCscore;
				}
			}
			if (printout2)
				cout << "@ll: " << ll << " (ZNCC, localZNCC): " << bestZNCC << " " << localZNCC << " (db1, mag1, db2, mag2): " << " " << bestqq << " " << bestll << " " << bestjj << " " << bestii << " " << endl;
		}
	}
	PuvIuv[0] = bestP1u, PuvIuv[1] = bestP1v, PuvIuv[2] = bestP2u, PuvIuv[3] = bestP2v;

	if (createdMem)
		delete[]Pro1Patch, delete[]Pro2Patch, delete[]SuperImposePatch, delete[]ZNNCStorage;

	return bestZNCC;
}
double IllumTransSep(IlluminationFlowImages &Fimgs, double *IPatch, double *direction, CPoint2 *Target, LKParameters LKArg, double *ILWp, double *Pcom, double *ZNCCStorage = 0)
{
	int i, j, k, iii, jjj;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold;

	double II0, JJ0, II1, JJ1, mF, L1, L2, L1x, L1y, L2x, L2y, DIC_Coeff, DIC_Coeff_min, a, b, c, t_1, t_2, t_3, t_4, t_5, t_6, S0[3], S1[3], p_best[16];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int jumpStep[2] = { 1, 2 }, nn = 7, nExtraParas = 3, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	double AA[7 * 7], BB[7], CC[7], p[7];

	p[0] = 0.0, p[1] = 0.0;
	p[2] = 0.0, p[3] = 0.0;
	p[4] = Pcom[0], p[5] = Pcom[1], p[6] = 0.0;

	int length = width*height, plength = pwidth*pheight, nchannels = 1, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	FILE *fp, *fp0, *fp1;

	bool printout = false;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (j = -hsubset; j <= hsubset; j++)
		{
			for (i = -hsubset; i <= hsubset; i++)
				fprintf(fp, "%.2f ", IPatch[(i + hsubset) + (j + hsubset)*TimgS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (i = 0; i < 2; i++)
		if (Target[i].x <-2 * hsubset || Target[i].y < -2 * hsubset || Target[i].x > pwidth + 2 * hsubset || Target[i].y > pheight + 2 * hsubset)
			return 0.0;

	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
			{
				fp = fopen("C:/temp/tar.txt", "w+");
				fp0 = fopen("C:/temp/tar0.txt", "w+");
				fp1 = fopen("C:/temp/tar1.txt", "w+");
			}

			a = p[4], b = p[5], c = p[6];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					II0 = Target[0].x + iii + p[0] + ILWp[0] * iii + ILWp[1] * jjj;
					JJ0 = Target[0].y + jjj + p[1] + ILWp[2] * iii + ILWp[3] * jjj;

					II1 = Target[1].x + iii + p[2] + ILWp[4] * iii + ILWp[5] * jjj;
					JJ1 = Target[1].y + jjj + p[3] + ILWp[6] * iii + ILWp[7] * jjj;

					if (II0<0.0 || II0>(double)(pwidth - 1) || JJ0<0.0 || JJ0>(double)(pheight - 1))
						continue;
					if (II1<0.0 || II1>(double)(pwidth - 1) || JJ1<0.0 || JJ1>(double)(pheight - 1))
						continue;

					Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, 0, Interpolation_Algorithm);
					Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, 0, Interpolation_Algorithm);

					mF = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
					L1 = S0[0], L2 = S1[0], L1x = S0[1], L1y = S0[2], L2x = S1[1], L2y = S1[2];

					t_3 = a*L1 + b*L2 + c - mF;

					t_4 = a, t_5 = t_4*L1x, t_6 = t_4*L1y;
					CC[0] = t_5, CC[1] = t_6;

					t_4 = b, t_5 = t_4*L2x, t_6 = t_4*L2y;
					CC[2] = t_5, CC[3] = t_6;

					CC[4] = L1, CC[5] = L2, CC[6] = 1.0;

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += mF*mF;

					if (printout)
					{
						fprintf(fp, "%.2f ", a*L1 + b*L2 + c);
						fprintf(fp0, "%.2f ", L1);
						fprintf(fp1, "%.2f ", L2);
					}
				}
				if (printout)
					fprintf(fp, "\n"), fprintf(fp0, "\n"), fprintf(fp1, "\n");
			}
			if (printout)
				fclose(fp), fclose(fp0), fclose(fp1);

			DIC_Coeff = t_1 / t_2;
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
			}

			if (p[0] != p[0] || abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[2]) > hsubset || abs(p[3]) > hsubset || DIC_Coeff > 50)
				return 0.0;
			if (fabs(BB[0]) < conv_crit_1 && abs(BB[1]) < conv_crit_1 && fabs(BB[2]) < conv_crit_1 && fabs(BB[3]) < conv_crit_1)
				Break_Flag = true;
			if (Break_Flag)
			{
				if (k <= 2)
				{
					Break_Flag = false;
					conv_crit_1 *= 0.1;
				}
				else
					break;
			}
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}
	/// DIC Iteration: End

	bool createMem = false;
	if (ZNCCStorage == NULL)
	{
		createMem = true;
		ZNCCStorage = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	}

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	if (DIC_Coeff_min < LKArg.PSSDab_thresh)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II0 = Target[0].x + iii + p[0] + ILWp[0] * iii + ILWp[1] * jjj;
				JJ0 = Target[0].y + jjj + p[1] + ILWp[2] * iii + ILWp[3] * jjj;

				II1 = Target[1].x + iii + p[2] + ILWp[4] * iii + ILWp[5] * jjj;
				JJ1 = Target[1].y + jjj + p[3] + ILWp[6] * iii + ILWp[7] * jjj;

				if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
					continue;
				if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
					continue;

				Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, -1, Interpolation_Algorithm);
				Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, -1, Interpolation_Algorithm);

				ZNCCStorage[2 * m] = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
				ZNCCStorage[2 * m + 1] = p[4] * S0[0] + p[5] * S1[0];
				t_f += ZNCCStorage[2 * m], t_g += ZNCCStorage[2 * m + 1];

				if (printout)
					fprintf(fp, "%.2f ", ZNCCStorage[2 * m + 1]);
				m++;
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / (m + 1), t_g = t_g / (m + 1);
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = ZNCCStorage[2 * i] - t_f;
			t_5 = ZNCCStorage[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;
	}

	if (createMem)
		delete[]ZNCCStorage;

	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[2]) > hsubset || abs(p[3]) > hsubset || DIC_Coeff > 50)
		return 0.0;

	Target[0].x += p[0], Target[0].y += p[1];
	Target[1].x += p[2], Target[1].y += p[3];
	Pcom[0] = p[4], Pcom[1] = p[5];
	for (i = 0; i < 2; i++)
	{
		if (Target[i].x <0.0 || Target[i].y < 0.0 || Target[i].x > pwidth - 1 || Target[i].y > pheight - 1)
			return 0.0;
	}

	return DIC_Coeff_min;
}
double IllumAffineSep(IlluminationFlowImages &Fimgs, double *IPatch, double *direction, CPoint2 *Target, LKParameters LKArg, double *ILWp, double *Pcom, double *ZNCCStorage = 0)
{
	int i, j, k, iii, jjj;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold;

	double II0, JJ0, II1, JJ1, mF, L1, L2, L1x, L1y, L2x, L2y, DIC_Coeff, DIC_Coeff_min, a, b, c, t_1, t_2, t_3, t_4, t_5, t_6, S0[3], S1[3], p_best[16];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int jumpStep[2] = { 1, 2 }, nn = 15, nExtraParas = 3, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	double AA[15 * 15], BB[15], CC[15], p[15];

	p[0] = 0.0, p[1] = 0.0, p[2] = ILWp[0], p[3] = ILWp[1], p[4] = ILWp[2], p[5] = ILWp[3];
	p[6] = 0.0, p[7] = 0.0, p[8] = ILWp[4], p[9] = ILWp[5], p[10] = ILWp[6], p[11] = ILWp[7];
	p[12] = Pcom[0], p[13] = Pcom[1], p[14] = 0.0;

	int length = width*height, plength = pwidth*pheight, nchannels = 1, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	FILE *fp, *fp0, *fp1;

	bool printout = false;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (j = -hsubset; j <= hsubset; j++)
		{
			for (i = -hsubset; i <= hsubset; i++)
				fprintf(fp, "%.2f ", IPatch[(i + hsubset) + (j + hsubset)*TimgS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (i = 0; i < 2; i++)
		if (Target[i].x <-2 * hsubset || Target[i].y < -2 * hsubset || Target[i].x > pwidth + 2 * hsubset || Target[i].y > pheight + 2 * hsubset)
			return 0.0;

	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
			{
				fp = fopen("C:/temp/tar.txt", "w+");
				fp0 = fopen("C:/temp/tar0.txt", "w+");
				fp1 = fopen("C:/temp/tar1.txt", "w+");
			}

			a = p[12], b = p[13], c = p[14];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					II0 = Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj;
					JJ0 = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj;

					II1 = Target[1].x + iii + p[6] + p[8] * iii + p[9] * jjj;
					JJ1 = Target[1].y + jjj + p[7] + p[10] * iii + p[11] * jjj;

					if (II0<0.0 || II0>(double)(pwidth - 1) || JJ0<0.0 || JJ0>(double)(pheight - 1))
						continue;
					if (II1<0.0 || II1>(double)(pwidth - 1) || JJ1<0.0 || JJ1>(double)(pheight - 1))
						continue;

					Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, 0, Interpolation_Algorithm);
					Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, 0, Interpolation_Algorithm);

					mF = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
					L1 = S0[0], L2 = S1[0], L1x = S0[1], L1y = S0[2], L2x = S1[1], L2y = S1[2];

					t_3 = a*L1 + b*L2 + c - mF;

					t_4 = a, t_5 = t_4*L1x, t_6 = t_4*L1y;
					CC[0] = t_5, CC[1] = t_6;
					CC[2] = t_5*iii, CC[3] = t_5*jjj;
					CC[4] = t_6*iii, CC[5] = t_6*jjj;

					t_4 = b, t_5 = t_4*L2x, t_6 = t_4*L2y;
					CC[6] = t_5, CC[7] = t_6;
					CC[8] = t_5*iii, CC[9] = t_5*jjj;
					CC[10] = t_6*iii, CC[11] = t_6*jjj;

					CC[12] = L1, CC[13] = L2, CC[14] = 1.0;

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += mF*mF;

					if (printout)
					{
						fprintf(fp, "%.2f ", a*L1 + b*L2 + c);
						fprintf(fp0, "%.2f ", L1);
						fprintf(fp1, "%.2f ", L2);
					}
				}
				if (printout)
					fprintf(fp, "\n"), fprintf(fp0, "\n"), fprintf(fp1, "\n");
			}
			if (printout)
				fclose(fp), fclose(fp0), fclose(fp1);

			DIC_Coeff = t_1 / t_2;
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
			}

			if (abs((p[2] + 1) / p[3]) + abs((p[5] + 1) / p[4]) < 6 || abs((p[8] + 1) / p[9]) + abs((p[11] + 1) / p[10]) < 6) //weirdly distortted shape-->must be wrong
				return 0;
			///if (abs(p[2] + 1) / abs(p[8] + 1) + abs(p[5] + 1) / abs(p[11] + 1) > 5 || abs(p[2] + 1) / abs(p[8] + 1) + abs(p[5] + 1) / abs(p[11] + 1) < 0.2)
			//return 0;
			if (p[0] != p[0] || abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[6]) > hsubset || abs(p[7]) > hsubset || DIC_Coeff > 50)
				return 0.0;
			if (fabs(BB[0]) < conv_crit_1 && abs(BB[1]) < conv_crit_1 && fabs(BB[6]) < conv_crit_1 && fabs(BB[7]) < conv_crit_1)
			{
				for (i = 2; i < 6; i++)
					if (fabs(BB[i]) > conv_crit_2)
						break;
				for (i = 8; i < 12; i++)
					if (fabs(BB[i]) > conv_crit_2)
						break;
				if (i == nn - nExtraParas)
					Break_Flag = true;
			}
			if (Break_Flag)
			{
				if (k <= 2)
				{
					Break_Flag = false;
					conv_crit_1 *= 0.1, conv_crit_2 *= 0.1;
				}
				else
					break;
			}
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}
	/// DIC Iteration: End

	bool createMem = false;
	if (ZNCCStorage == NULL)
	{
		createMem = true;
		ZNCCStorage = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	}

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	if (DIC_Coeff_min < LKArg.PSSDab_thresh)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II0 = Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ0 = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				II1 = Target[1].x + iii + p[6] + p[8] * iii + p[9] * jjj;
				JJ1 = Target[1].y + jjj + p[7] + p[10] * iii + p[11] * jjj;

				if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
					continue;
				if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
					continue;

				Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, -1, Interpolation_Algorithm);
				Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, -1, Interpolation_Algorithm);

				ZNCCStorage[2 * m] = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
				ZNCCStorage[2 * m + 1] = p[12] * S0[0] + p[13] * S1[0];
				t_f += ZNCCStorage[2 * m], t_g += ZNCCStorage[2 * m + 1];

				if (printout)
					fprintf(fp, "%.2f ", ZNCCStorage[2 * m + 1]);
				m++;
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / (m + 1), t_g = t_g / (m + 1);
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = ZNCCStorage[2 * i] - t_f;
			t_5 = ZNCCStorage[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;
	}

	if (createMem)
		delete[]ZNCCStorage;

	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[6]) > hsubset || abs(p[7]) > hsubset || DIC_Coeff > 50)
		return 0.0;

	//Store the warping parameters
	for (i = 0; i < 4; i++)
	{
		ILWp[i] = p[2 + i];
		ILWp[i + 4] = p[8 + i];
	}
	Pcom[0] = p[12], Pcom[1] = p[13];

	Target[0].x += p[0], Target[0].y += p[1];
	Target[1].x += p[6], Target[1].y += p[7];

	for (i = 0; i < 2; i++)
	{
		if (Target[i].x <0.0 || Target[i].y < 0.0 || Target[i].x > pwidth - 1 || Target[i].y > pheight - 1)
			return 0.0;
	}

	return DIC_Coeff_min;
}
double IllumHomoSep(IlluminationFlowImages &Fimgs, double *IPatch, double *direction, CPoint2 *Target, LKParameters LKArg, double *ILWp, double *Pcom, double *ZNCCStorage = 0)
{
	int i, j, k, iii, jjj;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold;

	double II0, JJ0, II1, JJ1, mF, L1, L2, L1x, L1y, L2x, L2y, DIC_Coeff, DIC_Coeff_min, a, b, c, t_1, t_2, t_3, t_4, t_5, t_6, t_7, S0[3], S1[3], p_best[19];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int jumpStep[2] = { 1, 2 }, nn = 19, nExtraParas = 3, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	double num1X, num1Y, denum1, num2X, num2Y, denum2, denum1_2, denum2_2;
	double AA[19 * 19], BB[19], CC[19], p[19];
	double affineAppro[8];

	p[0] = 0.0, p[1] = 0.0, p[2] = ILWp[0], p[3] = ILWp[1], p[4] = ILWp[2], p[5] = ILWp[3], p[6] = 0.0, p[7] = 0.0;
	p[8] = 0.0, p[9] = 0.0, p[10] = ILWp[4], p[11] = ILWp[5], p[12] = ILWp[6], p[13] = ILWp[7], p[14] = 0.0, p[15] = 0.0;
	p[16] = Pcom[0], p[17] = Pcom[1], p[18] = 0.0;

	int length = width*height, plength = pwidth*pheight, nchannels = 1, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	FILE *fp, *fp0, *fp1;

	bool printout = false;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (j = -hsubset; j <= hsubset; j++)
		{
			for (i = -hsubset; i <= hsubset; i++)
				fprintf(fp, "%.2f ", IPatch[(i + hsubset) + (j + hsubset)*TimgS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (i = 0; i < 2; i++)
		if (Target[i].x <-2 * hsubset || Target[i].y < -2 * hsubset || Target[i].x > pwidth + 2 * hsubset || Target[i].y > pheight + 2 * hsubset)
			return 0.0;

	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
			{
				fp = fopen("C:/temp/tar.txt", "w+");
				fp0 = fopen("C:/temp/tar0.txt", "w+");
				fp1 = fopen("C:/temp/tar1.txt", "w+");
			}

			a = p[16], b = p[17], c = p[18];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					num1X = (Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj), num1Y = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj, denum1 = p[6] * iii + p[7] * jjj + 1.0;
					num2X = (Target[1].x + iii + p[8] + p[10] * iii + p[11] * jjj), num2Y = (Target[1].y + jjj + p[9] + p[12] * iii + p[13] * jjj), denum2 = p[14] * iii + p[15] * jjj + 1.0;
					denum1_2 = denum1*denum1, denum2_2 = denum2*denum2;

					II0 = num1X / denum1, JJ0 = num1Y / denum1;
					II1 = num2X / denum2, JJ1 = num2Y / denum2;

					if (II0<0.0 || II0>(double)(pwidth - 1) || JJ0<0.0 || JJ0>(double)(pheight - 1))
						continue;
					if (II1<0.0 || II1>(double)(pwidth - 1) || JJ1<0.0 || JJ1>(double)(pheight - 1))
						continue;

					Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, 0, Interpolation_Algorithm);
					Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, 0, Interpolation_Algorithm);

					mF = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
					L1 = S0[0], L2 = S1[0], L1x = S0[1], L1y = S0[2], L2x = S1[1], L2y = S1[2];

					t_3 = a*L1 + b*L2 + c - mF;

					/*t_4 = a / denum1, t_5 = t_4*L1x, t_6 = t_4*L1y, t_7 = a*(L1x*num1X + L1y*num1Y) / denum1_2;
					CC[0] = a*L1x / denum1, CC[1] = a*L1y / denum1;
					CC[2] = a*L1x / denum1*iii, CC[3] = a*L1x / denum1*jjj;
					CC[4] = a*L1y / denum1*iii, CC[5] = a*L1y / denum1*jjj;
					CC[6] = -a*L1x*num1X / denum1_2*iii - a*L1y*num1Y / denum1_2*iii;
					CC[7] = -a*L1x*num1X / denum1_2*jjj - a*L1y*num1Y / denum1_2*jjj;

					t_5 = b*L2x, t_6 = b*L2y, t_7 = (L2x*num2X + L2y*num2Y) / denum2_2;
					CC[8] = b*L2x / denum2, CC[9] = b*L2y / denum2;
					CC[10] = b*L2x / denum2*iii, CC[11] = b*L2x / denum2*jjj;
					CC[12] = b*L2y / denum2*iii, CC[13] = b*L2y / denum2*jjj;
					CC[14] = -b*L2x*num2X / denum2_2*iii - b*L2y*num2Y / denum2_2*iii;
					CC[15] = -b*L2x*num2X / denum2_2*jjj - b*L2y*num2Y / denum2_2*jjj;*/

					t_4 = a / denum1, t_5 = t_4*L1x, t_6 = t_4*L1y, t_7 = a*(L1x*num1X + L1y*num1Y) / denum1_2;
					CC[0] = t_5, CC[1] = t_6;
					CC[2] = t_5*iii, CC[3] = t_5*jjj;
					CC[4] = t_6*iii, CC[5] = t_6*jjj;
					CC[6] = -t_7*iii;
					CC[7] = -t_7*jjj;

					t_4 = b / denum2, t_5 = t_4*L2x, t_6 = t_4*L2y, t_7 = b*(L2x*num2X + L2y*num2Y) / denum2_2;
					CC[8] = t_5, CC[9] = t_6;
					CC[10] = t_5*iii, CC[11] = t_5*jjj;
					CC[12] = t_6*iii, CC[13] = t_6*jjj;
					CC[14] = -t_7*iii;
					CC[15] = -t_7*jjj;

					CC[16] = L1, CC[17] = L2, CC[18] = 1.0;

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += mF*mF;

					if (printout)
					{
						fprintf(fp, "%.2f ", a*L1 + b*L2 + c);
						fprintf(fp0, "%.2f ", L1);
						fprintf(fp1, "%.2f ", L2);
					}
				}
				if (printout)
					fprintf(fp, "\n"), fprintf(fp0, "\n"), fprintf(fp1, "\n");
			}
			if (printout)
				fclose(fp), fclose(fp0), fclose(fp1);

			DIC_Coeff = t_1 / t_2;
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
			}

			//Check if the shape is weird
			//taylor expansion of the denum to approximate affine transform from homography:
			//[p2+1, p3, p0; 
			//p4, p5+1, p1; 
			//p6, p7, 1] 
			for (i = 0; i < 2; i++)
			{
				affineAppro[4 * i + 0] = p[8 * i + 2] - (p[8 * i + 0] + Target[i].x) * p[8 * i + 6], affineAppro[4 * i + 1] = p[8 * i + 3] - (p[8 * i + 0] + Target[i].x)* p[8 * i + 7];
				affineAppro[4 * i + 2] = p[8 * i + 4] - (p[8 * i + 1] + Target[i].y) * p[8 * i + 6], affineAppro[4 * i + 3] = p[8 * i + 5] - (p[8 * i + 1] + Target[i].y) * p[8 * i + 7];
			}

			if (abs((affineAppro[0] + 1) / affineAppro[1]) + abs((affineAppro[3] + 1) / affineAppro[2]) < 6 || abs((affineAppro[4] + 1) / affineAppro[5]) + abs((affineAppro[7] + 1) / affineAppro[6]) < 6) //weirdly distortted shape-->must be wrong
				return 0;
			//if (abs(p[2] + 1) / abs(p[8] + 1) + abs(p[5] + 1) / abs(p[11] + 1) > 5 || abs(p[2] + 1) / abs(p[8] + 1) + abs(p[5] + 1) / abs(p[11] + 1) < 0.2)
			//return 0;

			if (p[0] != p[0] || abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[8]) > hsubset || abs(p[9]) > hsubset || DIC_Coeff > 50)
				return 0.0;
			if (fabs(BB[0]) < conv_crit_1 && abs(BB[1]) < conv_crit_1 && fabs(BB[6]) < conv_crit_1 && fabs(BB[7]) < conv_crit_1)
			{
				for (i = 2; i < 9; i++)
					if (fabs(BB[i]) > conv_crit_2)
						break;
				for (i = 8; i < 16; i++)
					if (fabs(BB[i]) > conv_crit_2)
						break;
				if (i == nn - nExtraParas)
					Break_Flag = true;
			}
			if (Break_Flag)
			{
				if (k <= 2)
				{
					Break_Flag = false;
					conv_crit_1 *= 0.1, conv_crit_2 *= 0.1;
				}
				else
					break;
			}
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}
	/// DIC Iteration: End

	bool createMem = false;
	if (ZNCCStorage == NULL)
	{
		createMem = true;
		ZNCCStorage = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	}

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	if (DIC_Coeff_min < LKArg.PSSDab_thresh)
	{
		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				num1X = (Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj), num1Y = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj, denum1 = p[6] * iii + p[7] * jjj + 1.0;
				num2X = (Target[1].x + iii + p[8] + p[10] * iii + p[11] * jjj), num2Y = (Target[1].y + jjj + p[9] + p[12] * iii + p[13] * jjj), denum2 = p[14] * iii + p[15] * jjj + 1.0;
				denum1_2 = denum1*denum1, denum2_2 = denum2*denum2;

				II0 = num1X / denum1, JJ0 = num1Y / denum1;
				II1 = num2X / denum2, JJ1 = num2Y / denum2;

				if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
					continue;
				if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
					continue;

				Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, -1, Interpolation_Algorithm);
				Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, -1, Interpolation_Algorithm);

				ZNCCStorage[2 * m] = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
				ZNCCStorage[2 * m + 1] = p[16] * S0[0] + p[17] * S1[0];
				t_f += ZNCCStorage[2 * m], t_g += ZNCCStorage[2 * m + 1];

				if (printout)
					fprintf(fp, "%.2f ", ZNCCStorage[2 * m + 1]);
				m++;
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / (m + 1), t_g = t_g / (m + 1);
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = ZNCCStorage[2 * i] - t_f;
			t_5 = ZNCCStorage[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;
	}

	if (createMem)
		delete[]ZNCCStorage;

	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[8]) > hsubset || abs(p[9]) > hsubset || DIC_Coeff > 50)
		return 0.0;

	//Store the warping parameters
	for (i = 0; i < 6; i++)
	{
		ILWp[i] = p[2 + i];
		ILWp[i + 6] = p[10 + i];
	}
	Pcom[0] = p[16], Pcom[1] = p[17];

	Target[0].x += p[0], Target[0].y += p[1];
	Target[1].x += p[8], Target[1].y += p[9];

	for (i = 0; i < 2; i++)
	{
		if (Target[i].x <0.0 || Target[i].y < 0.0 || Target[i].x > pwidth - 1 || Target[i].y > pheight - 1)
			return 0.0;
	}

	return DIC_Coeff_min;
}

int OptimizeIIllum(int x, int y, int offx, int offy, int &cp, int &UV_index_n, int &M, double *Coeff, int *lpUV_xy, int *Tindex, DevicesInfo &DInfo, double *P1mat, double *FCP, LKParameters LKArg,
	int mode, IlluminationFlowImages &Fimgs, bool *cROI, int*visitedPoints, float *ILWarping, float *PhotoAdj, int *SeedType, float *SSIG, int *PrecomSearchR,
	double *IPatch, double *TarPatch, double *Pro1Patch, double *Pro2Patch, double *SuperImposePatch, double *ZNCCStorage, CPoint2 *HFrom, CPoint2 *HTo, double *ASolver, double *BSolver)
{
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int plength = pwidth*pheight, length = width*height;
	int id, idf, nid, seedtype, mm, nn, ll, ProID, u, v, rangeP1, rangeP2, success = 0;
	int hsubset = LKArg.hsubset, patchS = 2 * hsubset + 1, patchLength = patchS*patchS;

	double score, denum, t1, t2, puviuv[4], ImgPt[3], proEpiline[6], direction[4], ILWp[8], Pcom[2];
	CPoint2  puv, iuv, startI, startP[2], bef, aft, dPts[2];
	CPoint foundP[2]; CPoint3 WC;
	bool flag, flag2;

	success = 0, id = x + y*width, nid = x + offx + (y + offy)*width, seedtype = SeedType[id], rangeP1 = PrecomSearchR[id], rangeP2 = rangeP1;

	//Project 3D to projector image to see if interference can happen
	bool tryMix = false;
	CPoint2 Ppts, Cpts;
	if (seedtype != 3)
	{
		Cpts.x = 1.0*x, Cpts.y = 1.0*y;
		Ppts.x = ILWarping[id + 6 * (seedtype - 1)*length] + x, Ppts.y = ILWarping[id + (6 * (seedtype - 1) + 1)*length] + y;

		Undo_distortion(Ppts, DInfo.K + 9 * (seedtype - 1), DInfo.distortion + 13 * (seedtype - 1));
		Undo_distortion(Cpts, DInfo.K + 9 * nPros, DInfo.distortion + 13 * nPros);
		Stereo_Triangulation2(&Ppts, &Cpts, P1mat + 12 * (seedtype - 1), DInfo.P + 12 * (nPros - 1), &WC);

		//Project to projector image 
		if (seedtype == 1)
		{
			double denum = P1mat[8 + 12] * WC.x + P1mat[9 + 12] * WC.y + P1mat[10 + 12] * WC.z + P1mat[11 + 12];
			double u = (P1mat[0 + 12] * WC.x + P1mat[1 + 12] * WC.y + P1mat[2 + 12] * WC.z + P1mat[3 + 12]) / denum;
			double v = (P1mat[4 + 12] * WC.x + P1mat[5 + 12] * WC.y + P1mat[6 + 12] * WC.z + P1mat[7 + 12]) / denum;
			if (u > 35 && u < width - 35 && v>35 && v < height - 35)
				tryMix = true;
		}
		else if (seedtype == 2)
		{
			double denum = P1mat[8] * WC.x + P1mat[9] * WC.y + P1mat[10] * WC.z + P1mat[11];
			double u = (P1mat[0] * WC.x + P1mat[1] * WC.y + P1mat[2] * WC.z + P1mat[3]) / denum;
			double v = (P1mat[4] * WC.x + P1mat[5] * WC.y + P1mat[6] * WC.z + P1mat[7]) / denum;
			if (u>35 && u < width - 35 && v>35 && v < height - 35)
				tryMix = true;
		}
	}

	if (cROI[nid] && abs(ILWarping[nid + 7 * length]) < 0.001 && visitedPoints[nid] < LKArg.npass)
	{
		if (SSIG[nid] > LKArg.ssigThresh)
		{
			//Set up epipolar line and starting point
			ImgPt[0] = x + offx, ImgPt[1] = y + offy, ImgPt[2] = 1;
			startI.x = ImgPt[0], startI.y = ImgPt[1];
			for (ProID = 0; ProID < nPros; ProID++)
			{
				mat_mul(FCP + 9 * ProID, ImgPt, proEpiline + 3 * ProID, 3, 3, 1);
				denum = pow(proEpiline[3 * ProID], 2) + pow(proEpiline[3 * ProID + 1], 2);
				direction[2 * ProID] = -proEpiline[3 * ProID + 1] / sqrt(denum), direction[2 * ProID + 1] = proEpiline[3 * ProID] / sqrt(denum);
			}

			flag = false, flag2 = false;
			if (!tryMix && seedtype > 0 && seedtype < nPros) //not seeded by a mixtured point--> try pure illumination
			{
				flag2 = true;
				startP[seedtype - 1].x = x + ILWarping[id + 6 * (seedtype - 1)*length], startP[seedtype - 1].y = y + ILWarping[id + (1 + 6 * (seedtype - 1))*length];
				ILWp[(seedtype - 1) * 4] = ILWarping[id + (2 + 6 * (seedtype - 1))*length], ILWp[(seedtype - 1) * 4 + 1] = ILWarping[id + (3 + 6 * (seedtype - 1))*length], ILWp[(seedtype - 1) * 4 + 2] = ILWarping[id + (4 + 6 * (seedtype - 1))*length], ILWp[(seedtype - 1) * 4 + 3] = ILWarping[id + (5 + 6 * (seedtype - 1))*length];

				if (PrecomSearchR[nid] > 4) //Not deep inside mixed regions
				{
					dPts[0].x = startI.x, dPts[0].y = startI.y, dPts[1].x = startP[seedtype - 1].x, dPts[1].y = startP[seedtype - 1].y;
					score = EpipSearchLK(dPts, proEpiline + 3 * (seedtype - 1), Fimgs.Img, Fimgs.PImg + (seedtype - 1)*plength, Fimgs.Para, Fimgs.PPara + (seedtype - 1)*plength, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * (seedtype - 1));
					if (score > LKArg.ZNCCThreshold)
						flag = true, SeedType[nid] = seedtype;
					else
					{
						LKArg.hsubset -= 2; //Try a smaller patch size
						score = EpipSearchLK(dPts, proEpiline + 3 * (seedtype - 1), Fimgs.Img, Fimgs.PImg + (seedtype - 1)*plength, Fimgs.Para, Fimgs.PPara + (seedtype - 1)*plength, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * (seedtype - 1));
						if (score > LKArg.ZNCCThreshold)
							flag = true, SeedType[nid] = seedtype;
						LKArg.hsubset += 2;
					}

					if (flag)
					{
						ILWarping[nid + 6 * (seedtype - 1)*length] = dPts[1].x - ImgPt[0], ILWarping[nid + (1 + 6 * (seedtype - 1))*length] = dPts[1].y - ImgPt[1];
						for (ll = 0; ll < 4; ll++)
							ILWarping[nid + (ll + 2 + 6 * (seedtype - 1))*length] = ILWp[ll + 4 * (seedtype - 1)];
					}
				}
			}

			if (!flag) //try mixed illuminations 
			{
				//Search in the local region for computed cam-pro correspondence
				if (seedtype == 1)
				{
					if (!IsLocalWarpAvail(ILWarping + 6 * length, ILWp + 4, x + offx, y + offy, foundP[1], u, v, rangeP2, width, height, PrecomSearchR[nid]))
					{
						startP[0].x = x, startP[0].y = y; //get the second projector homography estimation
						rangeP2 = PrecomSearchR[nid];
						if (EstimateIllumPatchAffine(x + offx, y + offy, 0, 1, P1mat, ILWarping, startP[1], ILWp, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver) == 0)
							return success;
						idf = foundP[1].x + foundP[1].y*width;
						Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length];
					}
				}
				else if (seedtype == 2)
				{
					if (!IsLocalWarpAvail(ILWarping, ILWp, x + offx, y + offy, foundP[0], u, v, rangeP1, width, height, PrecomSearchR[nid]))
					{
						startP[1].x = x, startP[1].y = y; //get the first projector homography estimation
						rangeP1 = PrecomSearchR[nid];
						if (EstimateIllumPatchAffine(x + offx, y + offy, 1, 0, P1mat, ILWarping, startP[0], ILWp, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver) == 0)
							return success;
						idf = foundP[0].x + foundP[0].y*width;
						Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length];
					}
				}
				else if (seedtype == 3)
				{
					for (ProID = 0; ProID < nPros; ProID++)
					{
						startP[ProID].x = x + ILWarping[id + 6 * ProID*length], startP[ProID].y = y + ILWarping[id + (1 + 6 * ProID)*length];
						ILWp[ProID * 4] = ILWarping[id + (2 + 6 * ProID)*length], ILWp[ProID * 4 + 1] = ILWarping[id + (3 + 6 * ProID)*length], ILWp[ProID * 4 + 2] = ILWarping[id + (4 + 6 * ProID)*length], ILWp[ProID * 4 + 3] = ILWarping[id + (5 + 6 * ProID)*length];
					}
					Pcom[0] = PhotoAdj[id], Pcom[1] = PhotoAdj[id + length];
					rangeP1 = 1, rangeP2 = 1;
				}

				//Take the observed image patch
				for (ll = 0; ll < nchannels; ll++)
					for (mm = -hsubset; mm <= hsubset; mm++)
						for (nn = -hsubset; nn <= hsubset; nn++)
							IPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength] = Fimgs.Img[(x + offx + nn) + (y + offy + mm)*width + ll*length];

				if (rangeP1 > 1 || rangeP2 > 1)
				{
					if (LKArg.EpipEnforce == 1) 	//Project onto epipolar line
					{
						denum = pow(proEpiline[0], 2) + pow(proEpiline[1], 2);
						t1 = startP[0].x, t2 = startP[0].y;
						startP[0].x = (proEpiline[1] * (proEpiline[1] * t1 - proEpiline[0] * t2) - proEpiline[0] * proEpiline[2]) / denum;
						startP[0].y = (proEpiline[0] * (-proEpiline[1] * t1 + proEpiline[0] * t2) - proEpiline[1] * proEpiline[2]) / denum;

						denum = pow(proEpiline[3], 2) + pow(proEpiline[4], 2);
						t1 = startP[1].x, t2 = startP[1].y;
						startP[1].x = (proEpiline[4] * (proEpiline[4] * t1 - proEpiline[3] * t2) - proEpiline[3] * proEpiline[5]) / denum;
						startP[1].y = (proEpiline[3] * (-proEpiline[4] * t1 + proEpiline[3] * t2) - proEpiline[4] * proEpiline[5]) / denum;
					}
					score = llumSepCoarse(Fimgs, IPatch, startP, direction, ILWp, Pcom, hsubset, LKArg.searchRangeScale*rangeP1, LKArg.searchRangeScale*rangeP2, LKArg.InterpAlgo, puviuv, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage);
					dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
				}
				else //Just use previous points as inital guess is OK
				{
					dPts[0].x = startP[0].x, dPts[0].y = startP[0].y, dPts[1].x = startP[1].x, dPts[1].y = startP[1].y;
					score = LKArg.ZNCCThreshold;
				}

				if (score > LKArg.ZNCCThreshold - 0.35)
				{
					score = IllumAffineSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);
					if (score > LKArg.ZNCCThreshold - 0.04)
					{
						if (CamProGeoVerify(startI.x, startI.y, dPts, WC, P1mat, DInfo, width, height, pwidth, pheight, 2) == 0)
						{
							flag = true; SeedType[nid] = 3;
							ILWarping[nid] = dPts[0].x - startI.x, ILWarping[nid + length] = dPts[0].y - startI.y, ILWarping[nid + 6 * length] = dPts[1].x - startI.x, ILWarping[nid + 7 * length] = dPts[1].y - startI.y;
							PhotoAdj[nid] = Pcom[0], PhotoAdj[nid + length] = Pcom[1];
							for (ll = 0; ll < 4; ll++)
								ILWarping[nid + (ll + 2)*length] = ILWp[ll], ILWarping[nid + (ll + 8)*length] = ILWp[ll + 4];
						}
					}
				}
			}

			if (!flag && !flag2) //try mixed first but fail-->try illumination 
			{
				for (ProID = 0; ProID<nPros; ProID++)
				{
					dPts[0].x = startI.x, dPts[0].y = startI.y;
					dPts[1].x = startP[ProID].x, dPts[1].y = startP[ProID].y;

					score = EpipSearchLK(dPts, proEpiline + 3 * ProID, Fimgs.Img, Fimgs.PImg + ProID*plength, Fimgs.Para, Fimgs.PPara + ProID*plength, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * ProID);
					if (score>LKArg.ZNCCThreshold)
						flag = true, SeedType[nid] = ProID + 1;
					else
					{
						LKArg.hsubset -= 2; //Try a smaller patch size
						score = EpipSearchLK(dPts, proEpiline + 3 * ProID, Fimgs.Img, Fimgs.PImg, Fimgs.Para, Fimgs.PPara, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * ProID);
						if (score > LKArg.ZNCCThreshold)
							flag = true, SeedType[nid] = ProID + 1;
						LKArg.hsubset += 2;
					}

					if (flag)
					{
						ILWarping[nid] = dPts[1].x - startI.x, ILWarping[nid + length] = dPts[1].y - startI.y;
						for (ll = 0; ll < 4; ll++)
							ILWarping[nid + (ll + 2 + 6 * ProID)*length] = ILWp[ll + 4 * ProID];
					}
				}
			}

			if (flag)
			{
				cp++, UV_index_n++, M++;
				success = 1, cROI[nid] = false, Coeff[M] = 1.0 - score;
				lpUV_xy[2 * UV_index_n] = x + offx, lpUV_xy[2 * UV_index_n + 1] = y + offy;
				Tindex[M] = UV_index_n; DIC_AddtoQueue(Coeff, Tindex, M);
			}
		}
		visitedPoints[nid] += 1;
	}

	return success;
}
int IllumsReoptim(int frameID, char *PATH, char *TPATH, IlluminationFlowImages &Fimgs, DevicesInfo &DInfo, float *ILWarping, float *PhotoAdj, int *reoptim, int *SeedType, float *SSIG, LKParameters LKArg, int mode, bool *cROI, int part, bool deletePoints)
{
	//Assume there are one camera and two projectors
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int plength = pwidth*pheight, length = width*height, hsubset = LKArg.hsubset, InterpAlgo = LKArg.InterpAlgo;
	int id, idf, ProAvail[2], ii, jj, ll, mm, nn, x, y, rangeP1, rangeP2;
	bool flag;

	double score, denum, Fcp[18], ImgPt[3], proEpiline[6], direction[4], ILWp[4 * 2], Pcom[2];
	CPoint2  puv, iuv, startI, startP[2], startT, bef, aft, dPts[2];
	CPoint foundP[2]; CPoint3 WC;

	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	double *IPatch = new double[patchLength*nchannels], *TarPatch = new double[patchLength*nchannels], *Pro1Patch = new double[patchLength*nchannels], *Pro2Patch = new double[patchLength*nchannels], *SuperImposePatch = new double[patchLength*nchannels];
	double *ZNCCStorage = new double[2 * patchLength*nchannels];
	CPoint2 *HFrom = new CPoint2[patchLength], *HTo = new CPoint2[patchLength];
	double *ASolver = new double[patchLength * 3], *BSolver = new double[patchLength];

	double P1mat[12 * 2];
	P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0,
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0,
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;
	P1mat[12 + 0] = DInfo.P[0], P1mat[12 + 1] = DInfo.P[1], P1mat[12 + 2] = DInfo.P[2], P1mat[12 + 3] = DInfo.P[3],
		P1mat[12 + 4] = DInfo.P[4], P1mat[12 + 5] = DInfo.P[5], P1mat[12 + 6] = DInfo.P[6], P1mat[12 + 7] = DInfo.P[7],
		P1mat[12 + 8] = DInfo.P[8], P1mat[12 + 9] = DInfo.P[9], P1mat[12 + 10] = DInfo.P[10], P1mat[12 + 11] = DInfo.P[11];


	double t1, t2, puviuv[4];
	int pointsToCompute = 0, pointsComputed = 0, success = 0;
	for (ii = 0; ii < length; ii++)
		if (cROI[ii] && reoptim[ii] == 1)
			pointsToCompute++;

#pragma omp critical
	printf("Parition %d runs re-optim process with %d points\n", part, pointsToCompute);

	int percent = 50, increP = 50;
	double start = omp_get_wtime();

	mat_transpose(DInfo.FmatPC, Fcp, 3, 3);
	mat_transpose(DInfo.FmatPC + 9, Fcp + 9, 3, 3);
	for (jj = 0; jj < height; jj += LKArg.step)
	{
		for (ii = 0; ii < width; ii += LKArg.step)
		{
			id = ii + jj*width;
			if (cROI[id] && reoptim[id] == 1)
			{
				//printf("%d, %d\n", ii, jj);
				//if(ii == 1003 && jj == 1072)
				//	int a= 0;
				//Get the epipolar line on projectors
				ImgPt[0] = ii, ImgPt[1] = jj, ImgPt[2] = 1;
				for (int ProID = 0; ProID < nPros; ProID++)
				{
					mat_mul(Fcp + 9 * ProID, ImgPt, proEpiline + 3 * ProID, 3, 3, 1);
					denum = pow(proEpiline[3 * ProID], 2) + pow(proEpiline[1 + 3 * ProID], 2);
					direction[2 * ProID] = -proEpiline[1 + 3 * ProID] / sqrt(denum), direction[1 + 2 * ProID] = proEpiline[3 * ProID] / sqrt(denum);
				}

				//Search in the local region for computed cam-pro correspondence
				ProAvail[0] = 0, ProAvail[1] = 0;
				if (!IsLocalWarpAvail(ILWarping, ILWp, ii, jj, foundP[0], x, y, rangeP1, width, height, 1))
				{
					if (!IsLocalWarpAvail(ILWarping + 6 * length, ILWp + 4, ii, jj, foundP[1], x, y, rangeP2, width, height, 2) && deletePoints)
					{
						PhotoAdj[id] = 0.0, PhotoAdj[id + length] = 0.0;
						for (ll = 0; ll < 4; ll++)
							ILWarping[id + (ll + 2)*length] = 0.0, ILWarping[id + (ll + 8)*length] = 0.0;
						continue; // Does not find any closed by points. 
					}
					else //get the first projector homography estimation
					{
						startP[1].x = x, startP[1].y = y, ProAvail[1] = 1;
						rangeP1 = EstimateIllumPatchAffine(ii, jj, 1, 0, P1mat, ILWarping, startP[0], ILWp, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver);

						if (rangeP1 == 0)
							rangeP1 = 2; //Make it start the line search
						else if (rangeP1 > 3 && deletePoints)
						{
							PhotoAdj[id] = 0.0, PhotoAdj[id + length] = 0.0;
							for (ll = 0; ll < 6; ll++)
								ILWarping[id + ll*length] = 0.0, ILWarping[id + (ll + 6)*length] = 0.0;
							continue;
						}

						idf = foundP[1].x + foundP[1].y*width;
					}
				}
				else //get the second projector homography estimation
				{
					startP[0].x = x, startP[0].y = y, ProAvail[0] = 1;
					if (!IsLocalWarpAvail(ILWarping + 6 * length, ILWp + 4, ii, jj, foundP[1], x, y, rangeP2, width, height, 2))
					{
						startP[0].x = x, startP[0].y = y;
						rangeP2 = EstimateIllumPatchAffine(ii, jj, 0, 1, P1mat, ILWarping, startP[1], ILWp + 4, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver);

						if (rangeP2 == 0)
							rangeP2 = 2; //Make it start the line search
						else if (rangeP2 > 3 && deletePoints)
						{
							PhotoAdj[id] = 0.0, PhotoAdj[id + length] = 0.0;
							for (ll = 0; ll < 6; ll++)
								ILWarping[id + ll*length] = 0.0, ILWarping[id + (ll + 6)*length] = 0.0;
							continue;
						}

						idf = foundP[0].x + foundP[0].y*width;
					}
					else
					{
						startP[1].x = x, startP[1].y = y; ProAvail[1] = 1;
						idf = foundP[1].x + foundP[1].y*width;
					}
				}
				Pcom[0] = 0.5, Pcom[1] = 0.5;

				//Take the observed image patch
				for (ll = 0; ll < nchannels; ll++)
					for (mm = -hsubset; mm <= hsubset; mm++)
						for (nn = -hsubset; nn <= hsubset; nn++)
							IPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength] = Fimgs.Img[(ii + nn) + (jj + mm)*width + ll*length];

				if (LKArg.EpipEnforce == 1) 	//Project onto epipolar line
				{
					denum = pow(proEpiline[0], 2) + pow(proEpiline[1], 2);
					t1 = startP[0].x, t2 = startP[0].y;
					startP[0].x = (proEpiline[1] * (proEpiline[1] * t1 - proEpiline[0] * t2) - proEpiline[0] * proEpiline[2]) / denum;
					startP[0].y = (proEpiline[0] * (-proEpiline[1] * t1 + proEpiline[0] * t2) - proEpiline[1] * proEpiline[2]) / denum;

					denum = pow(proEpiline[3], 2) + pow(proEpiline[4], 2);
					t1 = startP[1].x, t2 = startP[1].y;
					startP[1].x = (proEpiline[4] * (proEpiline[4] * t1 - proEpiline[3] * t2) - proEpiline[3] * proEpiline[5]) / denum;
					startP[1].y = (proEpiline[3] * (-proEpiline[4] * t1 + proEpiline[3] * t2) - proEpiline[4] * proEpiline[5]) / denum;
				}

				flag = false;
				if (rangeP1> 1 || rangeP2 > 1) //only needed when points are far away
					score = llumSepCoarse(Fimgs, IPatch, startP, direction, ILWp, Pcom, hsubset, LKArg.searchRangeScale*rangeP1, LKArg.searchRangeScale*rangeP2, LKArg.InterpAlgo, puviuv, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage);
				else
				{
					score = LKArg.ZNCCThreshold;
					puviuv[0] = startP[0].x, puviuv[1] = startP[0].y, puviuv[2] = startP[1].x, puviuv[3] = startP[1].y;
				}

				if (score > LKArg.ZNCCThreshold - 0.35)
				{
					dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
					score = IllumAffineSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);
					if (score > LKArg.ZNCCThreshold - 0.04)
					{
						if (CamProGeoVerify(1.0*ii, 1.0*jj, dPts, WC, P1mat, DInfo, width, height, pwidth, pheight, 2) == 0)
						{
							reoptim[id] = 2, success++;
							flag = true; SeedType[id] = 3;
							ILWarping[id] = dPts[0].x - ii, ILWarping[id + length] = dPts[0].y - jj, ILWarping[id + 6 * length] = dPts[1].x - ii, ILWarping[id + 7 * length] = dPts[1].y - jj;
							PhotoAdj[id] = Pcom[0], PhotoAdj[id + length] = Pcom[1];
							for (ll = 0; ll < 4; ll++)
								ILWarping[id + (ll + 2)*length] = ILWp[ll], ILWarping[id + (ll + 8)*length] = ILWp[ll + 4];
						}
					}
				}

				if (!flag && deletePoints)
				{
					PhotoAdj[id] = 0.0, PhotoAdj[id + length] = 0.0;
					for (ll = 0; ll < 6; ll++)
						ILWarping[id + ll*length] = 0.0, ILWarping[id + (ll + 6)*length] = 0.0;
				}

				if (100 * pointsComputed / pointsToCompute >= percent)
				{
#pragma omp critical
					{
						char Fname[200];
						for (int ProID = 0; ProID < nPros; ProID++)
						{
							for (int ll = 0; ll < 6; ll++)
							{
								sprintf(Fname, "%s/%05d_C1P%dp%d.dat", TPATH, frameID, ProID + 1, ll);
								WriteGridBinary(Fname, ILWarping + (6 * ProID + ll)*length, width, height);
							}
						}
						for (ll = 0; ll < 2; ll++)
						{
							sprintf(Fname, "%s/%05d_C1PA2_%d.dat", TPATH, frameID, ll);
							WriteGridBinary(Fname, PhotoAdj + ll*length, width, height);
						}
						sprintf(Fname, "%s/%05d_SeedType.dat", TPATH, frameID);	WriteGridBinary(Fname, SeedType, width, height);
						sprintf(Fname, "%s/Results/Sep", PATH);
						UpdateIllumTextureImages(Fname, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara, ILWarping);
					}
					double elapsed = omp_get_wtime() - start;
					cout << "Partition #" << part << " ..." << 100 * pointsComputed / pointsToCompute << "% ... #" << pointsComputed << " points.TE: " << setw(2) << elapsed << " TR: " << setw(2) << elapsed / (percent + increP)*(100.0 - percent) << endl;
					percent += increP;
				}
				pointsComputed++;
			}
		}
	}
#pragma omp critical
	printf("Parition %d re-optims %d points\n", part, success);

	delete[]IPatch, delete[]Pro1Patch, delete[]Pro2Patch;
	delete[]SuperImposePatch, delete[]ZNCCStorage;
	delete[]HFrom, delete[]HTo, delete[]ASolver, delete[]BSolver;

	return 0;
}
int IllumSeperation(int frameID, char *PATH, char *TPATH, IlluminationFlowImages &Fimgs, DevicesInfo &DInfo, float *ILWarping, float *PhotoAdj, int *SeedType, float *SSIG, int *PrecomSearchR, LKParameters LKArg, int mode, bool *cROI, int part)
{
	//Assume there are one camera and two projectors
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int plength = pwidth*pheight, length = width*height, hsubset = LKArg.hsubset, InterpAlgo = LKArg.InterpAlgo;
	int id, idf, ProAvail[2], seededsucces, ii, jj, kk, ll, mm, nn, ProID, x, y, rangeP1, rangeP2;
	bool flag, flag2;

	double score, denum, Fcp[18], ImgPt[3], proEpiline[6], direction[4], ILWp[4 * 2], Pcom[2];
	CPoint2  puv, iuv, startI, startP[2], startT, bef, aft, dPts[2];
	CPoint foundP[2]; CPoint3 WC;

	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	double *IPatch = new double[patchLength*nchannels];
	double *TarPatch = new double[patchLength*nchannels];
	double *Pro1Patch = new double[patchLength*nchannels];
	double *Pro2Patch = new double[patchLength*nchannels];
	double *SuperImposePatch = new double[patchLength*nchannels];
	double *ZNCCStorage = new double[2 * patchLength*nchannels];
	CPoint2 *HFrom = new CPoint2[patchLength];
	CPoint2 *HTo = new CPoint2[patchLength];
	double *ASolver = new double[patchLength * 3];
	double *BSolver = new double[patchLength];

	double P1mat[12 * 2];
	P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0,
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0,
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;
	P1mat[12 + 0] = DInfo.P[0], P1mat[12 + 1] = DInfo.P[1], P1mat[12 + 2] = DInfo.P[2], P1mat[12 + 3] = DInfo.P[3],
		P1mat[12 + 4] = DInfo.P[4], P1mat[12 + 5] = DInfo.P[5], P1mat[12 + 6] = DInfo.P[6], P1mat[12 + 7] = DInfo.P[7],
		P1mat[12 + 8] = DInfo.P[8], P1mat[12 + 9] = DInfo.P[9], P1mat[12 + 10] = DInfo.P[10], P1mat[12 + 11] = DInfo.P[11];


	double t1, t2, puviuv[4];
	int pointsToCompute = 0, pointsComputed = 0, cp, M, UV_index = 0, UV_index_n = 0;
	int *visitedPoints = new int[length];
	int *Tindex = new int[length];
	int *lpUV_xy = new int[2 * length];
	double *Coeff = new double[length];

	for (jj = 0; jj < height; jj += LKArg.step)
	{
		for (ii = 0; ii < width; ii += LKArg.step)
		{
			id = ii + jj*width;
			if (cROI[id])
			{
				visitedPoints[id] = 0, flag = false;
				for (kk = 0; kk < nPros; kk++)
				{
					if (abs(ILWarping[id + kk * 6 * length]) + abs(ILWarping[id + (1 + kk * 6)*length]) > 0.01)
					{
						flag = true; break;
					}
				}
				if (!flag)
					pointsToCompute++;
			}
		}
	}
#pragma omp critical
	cout << "Partition #" << part << " deals with " << pointsToCompute << " pts." << endl;

	char Fname[200];
	int percent = 50, increP = 50;
	double start = omp_get_wtime();

	if (pointsToCompute < 500)
	{
#pragma omp critical
		cout << "Partition #" << part << " terminates because #points to compute is to small." << endl;
	}
	else
	{
		mat_transpose(DInfo.FmatPC, Fcp, 3, 3);
		mat_transpose(DInfo.FmatPC + 9, Fcp + 9, 3, 3);
		for (kk = 0; kk < LKArg.npass2; kk++) //do nPass
		{
			for (jj = 0; jj < height; jj += LKArg.step)
			{
				for (ii = 0; ii < width; ii += LKArg.step)
				{
					cp = 0, M = -1; id = ii + jj*width;
					if (cROI[id] && visitedPoints[id] < LKArg.npass2)
					{
						if (SSIG[id] < LKArg.ssigThresh)
						{
							visitedPoints[id] += 1;
							continue;
						}

						M = 0; UV_index = UV_index_n;
						lpUV_xy[2 * UV_index] = ii, lpUV_xy[2 * UV_index + 1] = jj;

						startI.x = ii, startI.y = jj;
						//Get the epipolar line on projectors
						ImgPt[0] = startI.x, ImgPt[1] = startI.y, ImgPt[2] = 1;
						for (ProID = 0; ProID < nPros; ProID++)
						{
							mat_mul(Fcp + 9 * ProID, ImgPt, proEpiline + 3 * ProID, 3, 3, 1);
							denum = pow(proEpiline[3 * ProID], 2) + pow(proEpiline[1 + 3 * ProID], 2);
							direction[2 * ProID] = -proEpiline[1 + 3 * ProID] / sqrt(denum), direction[1 + 2 * ProID] = proEpiline[3 * ProID] / sqrt(denum);
						}

						//Search in the local region for computed cam-pro correspondence
						ProAvail[0] = 0, ProAvail[1] = 0;
						if (!IsLocalWarpAvail(ILWarping, ILWp, ii, jj, foundP[0], x, y, rangeP1, width, height, PrecomSearchR[ii + jj*width]))
						{
							if (!IsLocalWarpAvail(ILWarping + 6 * length, ILWp + 4, ii, jj, foundP[1], x, y, rangeP2, width, height, PrecomSearchR[ii + jj*width]))
							{
								visitedPoints[id] += 1;
								continue; // Does not find any closed by points. 
							}
							else //get the first projector homography estimation
							{
								startP[1].x = x, startP[1].y = y, ProAvail[1] = 1;
								rangeP1 = min(EstimateIllumPatchAffine(ii, jj, 1, 0, P1mat, ILWarping, startP[0], ILWp, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver), PrecomSearchR[ii + jj*width]);
								if (rangeP1 == 0)
									continue;
								idf = foundP[1].x + foundP[1].y*width;
								Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length];
							}
						}
						else //get the second projector homography estimation
						{
							startP[0].x = x, startP[0].y = y, ProAvail[0] = 1;
							if (!IsLocalWarpAvail(ILWarping + 6 * length, ILWp + 4, ii, jj, foundP[1], x, y, rangeP2, width, height, PrecomSearchR[ii + jj*width]))
							{
								startP[0].x = x, startP[0].y = y;
								rangeP2 = min(EstimateIllumPatchAffine(ii, jj, 0, 1, P1mat, ILWarping, startP[1], ILWp + 4, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver), PrecomSearchR[ii + jj*width]);
								if (rangeP2 == 0)
									continue;
								idf = foundP[0].x + foundP[0].y*width;
								Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length];
							}
							else
							{
								startP[1].x = x, startP[1].y = y; ProAvail[1] = 1;
								idf = foundP[1].x + foundP[1].y*width;
								Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length];
							}
						}

						flag = false, flag2 = false;
						for (ProID = 0; ProID < nPros && !flag; ProID++)
						{
							if (ProAvail[ProID] == 0)
								continue;

							if (PrecomSearchR[ii + jj*width] > 4) //Not deep inside mixed regions
							{
								flag2 = true;
								dPts[0].x = startI.x, dPts[0].y = startI.y;
								dPts[1].x = startP[ProID].x, dPts[1].y = startP[ProID].y;

								score = EpipSearchLK(dPts, proEpiline + 3 * ProID, Fimgs.Img, Fimgs.PImg + ProID*plength, Fimgs.Para, Fimgs.PPara + ProID*plength, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * ProID);
								if (score > LKArg.ZNCCThreshold)
									flag = true, SeedType[id] = ProID + 1;
								else
								{
									LKArg.hsubset -= 2;
									score = EpipSearchLK(dPts, proEpiline + 3 * ProID, Fimgs.Img, Fimgs.PImg + ProID*plength, Fimgs.Para, Fimgs.PPara + ProID*plength, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * ProID);
									if (score > LKArg.ZNCCThreshold)
										flag = true, SeedType[id] = ProID + 1;
									LKArg.hsubset += 2;
								}

								if (flag)
								{
									ILWarping[id] = dPts[1].x - ii, ILWarping[id + length] = dPts[1].y - jj;
									for (ll = 0; ll < 4; ll++)
										ILWarping[id + (ll + 2 + ProID * 6)*length] = ILWp[ll + 4 * ProID];
								}
							}
						}

						if (!flag) //Start Illum decompostion
						{
							for (ll = 0; ll < nchannels; ll++) //Take the observed image patch
								for (mm = -hsubset; mm <= hsubset; mm++)
									for (nn = -hsubset; nn <= hsubset; nn++)
										IPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength] = Fimgs.Img[(ii + nn) + (jj + mm)*width + ll*length];

							if (LKArg.EpipEnforce == 1) 	//Project onto epipolar line
							{
								denum = pow(proEpiline[0], 2) + pow(proEpiline[1], 2);
								t1 = startP[0].x, t2 = startP[0].y;
								startP[0].x = (proEpiline[1] * (proEpiline[1] * t1 - proEpiline[0] * t2) - proEpiline[0] * proEpiline[2]) / denum;
								startP[0].y = (proEpiline[0] * (-proEpiline[1] * t1 + proEpiline[0] * t2) - proEpiline[1] * proEpiline[2]) / denum;

								denum = pow(proEpiline[3], 2) + pow(proEpiline[4], 2);
								t1 = startP[1].x, t2 = startP[1].y;
								startP[1].x = (proEpiline[4] * (proEpiline[4] * t1 - proEpiline[3] * t2) - proEpiline[3] * proEpiline[5]) / denum;
								startP[1].y = (proEpiline[3] * (-proEpiline[4] * t1 + proEpiline[3] * t2) - proEpiline[4] * proEpiline[5]) / denum;
							}

							if (rangeP1>2 || rangeP2 > 2)
								score = llumSepCoarse(Fimgs, IPatch, startP, direction, ILWp, Pcom, hsubset, LKArg.searchRangeScale*rangeP1, LKArg.searchRangeScale*rangeP2, LKArg.InterpAlgo, puviuv, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage);
							else
							{
								score = LKArg.ZNCCThreshold;
								puviuv[0] = startP[0].x, puviuv[1] = startP[0].y, puviuv[2] = startP[1].x, puviuv[3] = startP[1].y;
							}

							if (score > LKArg.ZNCCThreshold - 0.35)
							{
								dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
								score = IllumAffineSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);
								if (score > LKArg.ZNCCThreshold - 0.04)
								{
									if (CamProGeoVerify(1.0*ii, 1.0*jj, dPts, WC, P1mat, DInfo, width, height, pwidth, pheight, 2) == 0)
									{
										flag = true; SeedType[id] = 3;
										ILWarping[id] = dPts[0].x - ii, ILWarping[id + length] = dPts[0].y - jj, ILWarping[id + 6 * length] = dPts[1].x - ii, ILWarping[id + 7 * length] = dPts[1].y - jj;
										PhotoAdj[id] = Pcom[0], PhotoAdj[id + length] = Pcom[1];
										for (ll = 0; ll < 4; ll++)
											ILWarping[id + (ll + 2)*length] = ILWp[ll], ILWarping[id + (ll + 8)*length] = ILWp[ll + 4];
									}
								}
							}
						}

						if (!flag && !flag2)
						{
							for (ProID = 0; ProID<nPros && !flag; ProID++)
							{
								if (ProAvail[ProID] == 0)
									continue;

								flag2 = true;
								dPts[0].x = startI.x, dPts[0].y = startI.y;
								dPts[1].x = startP[ProID].x, dPts[1].y = startP[ProID].y;

								score = EpipSearchLK(dPts, proEpiline + 3 * ProID, Fimgs.Img, Fimgs.PImg, Fimgs.Para, Fimgs.PPara, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * ProID);
								if (score>LKArg.ZNCCThreshold)
									flag = true, SeedType[id] = ProID + 1;
								else
								{
									LKArg.hsubset -= 2;
									score = EpipSearchLK(dPts, proEpiline + 3 * ProID, Fimgs.Img, Fimgs.PImg, Fimgs.Para, Fimgs.PPara, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * ProID);
									if (score > LKArg.ZNCCThreshold)
										flag = true, SeedType[id] = ProID + 1;
									LKArg.hsubset += 2;
								}
								if (flag)
								{
									ILWarping[id] = dPts[1].x - ii, ILWarping[id + length] = dPts[1].y - jj;
									for (ll = 0; ll < 4; ll++)
										ILWarping[id + (ll + 2 + ProID * 6)*length] = ILWp[ll + 4 * ProID];
								}
							}
						}

						if (flag)
						{
							cp++; cROI[id] = false;
							Coeff[M] = 1.0 - score, Tindex[M] = UV_index;
						}
						else
							M--;

						visitedPoints[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width] += 1;
					}

					//Now, PROPAGATE
					seededsucces = 0;
					while (M >= 0)
					{
						UV_index = Tindex[M];
						x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
						M--;

						seededsucces += OptimizeIIllum(x, y, 0, 1, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, DInfo, P1mat, Fcp, LKArg, mode, Fimgs, cROI, visitedPoints, ILWarping, PhotoAdj, SeedType, SSIG, PrecomSearchR,
							IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);

						seededsucces += OptimizeIIllum(x, y, 0, -1, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, DInfo, P1mat, Fcp, LKArg, mode, Fimgs, cROI, visitedPoints, ILWarping, PhotoAdj, SeedType, SSIG, PrecomSearchR,
							IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);

						seededsucces += OptimizeIIllum(x, y, 1, 0, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, DInfo, P1mat, Fcp, LKArg, mode, Fimgs, cROI, visitedPoints, ILWarping, PhotoAdj, SeedType, SSIG, PrecomSearchR,
							IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);

						seededsucces += OptimizeIIllum(x, y, -1, 0, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, DInfo, P1mat, Fcp, LKArg, mode, Fimgs, cROI, visitedPoints, ILWarping, PhotoAdj, SeedType, SSIG, PrecomSearchR,
							IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);

						if (100 * (UV_index_n + 1) / pointsToCompute >= percent)
						{
							double elapsed = omp_get_wtime() - start;
							cout << "Partition #" << part << " ..." << 100 * (UV_index_n + 1) / pointsToCompute << "% ... #" << pointsComputed + cp << " points.TE: " << setw(2) << elapsed << " TR: " << setw(2) << elapsed / (percent + increP)*(100.0 - percent) << endl;
							percent += increP;

#pragma omp critical
							{
								for (ProID = 0; ProID < nPros; ProID++)
								{
									for (int ll = 0; ll < 6; ll++)
									{
										sprintf(Fname, "%s/%05d_C1P%dp%d.dat", TPATH, frameID, ProID + 1, ll);
										WriteGridBinary(Fname, ILWarping + (6 * ProID + ll)*length, width, height);
									}
								}
								for (ll = 0; ll < 2; ll++)
								{
									sprintf(Fname, "%s/%05d_C1PA2_%d.dat", TPATH, frameID, ll);
									WriteGridBinary(Fname, PhotoAdj + ll*length, width, height);
								}
								sprintf(Fname, "%s/%05d_SeedType.dat", TPATH, frameID);	WriteGridBinary(Fname, SeedType, width, height);
								sprintf(Fname, "%s/Results/Sep", PATH);
								UpdateIllumTextureImages(Fname, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara, ILWarping);
							}
						}
					}

					if (seededsucces > 0)
						UV_index_n++;
					pointsComputed += cp;

#pragma omp critical
					if (seededsucces > 1000)
					{
						sprintf(Fname, "%s/Results/Sep", PATH);
						UpdateIllumTextureImages(TPATH, false, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara, ILWarping);
					}
				}
			}
		}
	}

#pragma omp critical
	{
		for (ProID = 0; ProID < nPros; ProID++)
		{
			for (int ll = 0; ll < 6; ll++)
			{
				sprintf(Fname, "%s/%05d_C1P%dp%d.dat", TPATH, frameID, ProID + 1, ll);
				WriteGridBinary(Fname, ILWarping + (6 * ProID + ll)*length, width, height);
			}
		}
		for (ll = 0; ll < 2; ll++)
		{
			sprintf(Fname, "%s/%05d_C1PA2_%d.dat", TPATH, frameID, ll);
			WriteGridBinary(Fname, PhotoAdj + ll*length, width, height);
		}
		sprintf(Fname, "%s/%05d_SeedType.dat", TPATH, frameID);	WriteGridBinary(Fname, SeedType, width, height);
		sprintf(Fname, "%s/Results/Sep", PATH);
		UpdateIllumTextureImages(Fname, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara, ILWarping);
	}

	double elapsed = omp_get_wtime() - start;
	cout << "Partition #" << part << " finishes ... " << 100 * pointsComputed / pointsToCompute << "% (" << pointsComputed << " pts) in " << omp_get_wtime() - start << "s" << endl;

	//Kill all the bad poitns:
	for (jj = 0; jj < nPros; jj++)
		for (ii = 0; ii < length; ii++)
			if (abs(ILWarping[ii + 6 * jj*length]) + abs(ILWarping[ii + length + 6 * jj*length]) < 0.01)
				ILWarping[ii + (2 + 6 * jj)*length] = 0.0, ILWarping[ii + (3 + 6 * jj)*length] = 0.0, ILWarping[ii + (4 + 6 * jj)*length] = 0.0, ILWarping[ii + (5 + 6 * jj)*length] = 0.0;

	delete[]IPatch, delete[]Pro1Patch, delete[]Pro2Patch;
	delete[]SuperImposePatch, delete[]ZNCCStorage;
	delete[]visitedPoints, delete[]Tindex, delete[]lpUV_xy, delete[]Coeff;
	delete[]HFrom, delete[]HTo, delete[]ASolver, delete[]BSolver;

	return 0;
}

double TwoIllumTextSepCoarse(IlluminationFlowImages &Fimgs, double *SoureTexture, double *IPatch, CPoint2 *startP, CPoint2 &StartT, double *direction, double *ILWp, double *TWp, double *Pcom, int hsubset, int *searchRange, double *PuvIuv, double *Pro1Patch = 0, double *Pro2Patch = 0, double *TextPatch = 0, double *SuperImposePatch = 0, double *ZNNCStorage = 0)
{
	//Model: aL1T+bL2T+cL1+dL2+eT
	double step = 1.0;
	int ii, jj, kk, ll, pp, qq, mm, nn, cID, rr;
	double II, JJ;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;
	int length = width*height, plength = pwidth*pheight, nchannels = Fimgs.nchannels;

	int patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	bool flag, createdMem = false;
	if (Pro1Patch == NULL)
	{
		createdMem = true;
		Pro1Patch = new double[patchLength*nchannels];
		Pro2Patch = new double[patchLength*nchannels];
		TextPatch = new double[patchLength*nchannels];
		SuperImposePatch = new double[patchLength*nchannels];
		ZNNCStorage = new double[2 * patchLength*nchannels];
	}
	double tryPro1X, tryPro1Y, tryPro2X, tryPro2Y, tryTextX, tryTextY, ZNCCscore, bestP1X = 0, bestP1Y = 0, bestP2X = 0, bestP2Y = 0, bestTX = 0, bestTY = 0, bestZNCC = -1.0;
	int bestqq = 0, bestpp = 0, bestll = 0, bestkk = 0, bestii = 0, bestjj = 0;
	bool printout = false, printout2 = false;

	if (printout)
	{
		FILE *fp = fopen("C:/temp/src.txt", "w+");
		for (ll = 0; ll < patchS; ll++)
		{
			for (kk = 0; kk < patchS; kk++)
				fprintf(fp, "%.2f ", IPatch[kk + ll*patchS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	//Now, start searching for projector patch that is on the band of epipolar line
	for (qq = -searchRange[3]; qq <= searchRange[3]; qq++)
	{
		for (pp = -searchRange[0]; pp <= searchRange[0]; pp++)
		{
			tryPro1X = startP[0].x + qq + direction[0] * pp, tryPro1Y = startP[0].y + qq + direction[1] * pp;
			if (tryPro1X <= hsubset || tryPro1X >= pwidth - hsubset || tryPro1Y <= hsubset || tryPro1Y >= pheight - hsubset)
				continue;

			//Take a prewarped patch in Pro 1
			flag = true;
			for (ll = -hsubset; ll <= hsubset && flag; ll++)
			{
				for (kk = -hsubset; kk <= hsubset; kk++)
				{
					II = tryPro1X + kk + ILWp[0] * kk + ILWp[1] * ll;
					JJ = tryPro1Y + ll + ILWp[2] * kk + ILWp[3] * ll;
					if (II<0 || II>pwidth - 1 || JJ<0 || JJ>pheight - 1)
					{
						flag = false;
						break;
					}
					for (cID = 0; cID < nchannels; cID++)
						Pro1Patch[(ll + hsubset)*patchS + (kk + hsubset) + cID*patchLength] = BilinearInterp(Fimgs.PImg + cID*plength, pwidth, pheight, II, JJ);
				}
			}

			if (!flag)
				continue;

			if (printout)
			{
				FILE *fp = fopen("C:/temp/tar0.txt", "w+");
				for (ll = 0; ll < patchS; ll++)
				{
					for (kk = 0; kk < patchS; kk++)
						fprintf(fp, "%.2f ", Pro1Patch[kk + ll*patchS]);
					fprintf(fp, "\n");
				}
				fclose(fp);
			}

			//Take a prewarped patch in Pro 2
			double localZNCC = -1.0;
			for (ll = -searchRange[3]; ll <= searchRange[3]; ll++)
			{
				for (kk = -searchRange[1]; kk <= searchRange[1]; kk++)
				{
					tryPro2X = startP[1].x + ll + direction[2] * kk, tryPro2Y = startP[1].y + ll + direction[3] * kk;
					if (tryPro2X <= hsubset - 2 || tryPro2X >= pwidth - hsubset + 2 || tryPro2Y <= hsubset - 2 || tryPro2Y >= pheight - hsubset + 2)
						continue;

					flag = true;
					for (mm = -hsubset; mm <= hsubset && flag; mm++)
					{
						for (nn = -hsubset; nn <= hsubset; nn++)
						{
							II = tryPro2X + nn + ILWp[4] * nn + ILWp[5] * mm;
							JJ = tryPro2Y + mm + ILWp[6] * nn + ILWp[7] * mm;
							if (II<0 || II>pwidth - 1 || JJ<0 || JJ>pheight - 1)
							{
								flag = false;
								break;
							}
							for (cID = 0; cID < nchannels; cID++)
								Pro2Patch[(mm + hsubset)*patchS + (nn + hsubset) + cID*patchLength] = BilinearInterp(Fimgs.PImg + (nchannels + cID)*plength, pwidth, pheight, II, JJ);
						}
					}
					if (!flag)
						continue;

					if (printout)
					{
						FILE *fp = fopen("C:/temp/tar1.txt", "w+");
						for (mm = 0; mm < patchS; mm++)
						{
							for (nn = 0; nn < patchS; nn++)
								fprintf(fp, "%.2f ", Pro2Patch[nn + mm*patchS]);
							fprintf(fp, "\n");
						}
						fclose(fp);
					}

					//Take a prewarp Texture patch
					for (ii = -searchRange[2]; ii <= searchRange[2]; ii += step)
					{
						for (jj = -searchRange[2]; jj <= searchRange[2]; jj += step)
						{
							tryTextX = StartT.x + step*ii, tryTextY = StartT.y + step*jj;
							if (tryTextX<hsubset + 1 || tryTextX >width - hsubset - 1 || tryTextY<hsubset + 1 || tryTextY > height - hsubset - 1)
								continue;

							flag = true;
							for (mm = -hsubset; mm <= hsubset && flag; mm++)
							{
								for (nn = -hsubset; nn <= hsubset; nn++)
								{
									II = tryTextX + nn + TWp[0] * nn + TWp[1] * mm;
									JJ = tryTextY + mm + TWp[2] * nn + TWp[3] * mm;
									if (II<2.0 || II>width - 2.0 || JJ<2.0 || JJ>height - 2.0)
									{
										flag = false; break;
									}
									for (cID = 0; cID < nchannels; cID++)
										TextPatch[(mm + hsubset)*patchS + (nn + hsubset) + cID*patchLength] = BilinearInterp(SoureTexture + cID*length, width, height, II, JJ);
								}
							}
							if (!flag)
								continue;

							if (printout)
							{
								FILE *fp = fopen("C:/temp/tar2.txt", "w+");
								for (mm = 0; mm < patchS; mm++)
								{
									for (nn = 0; nn < patchS; nn++)
										fprintf(fp, "%.2f ", TextPatch[nn + mm*patchS]);
									fprintf(fp, "\n");
								}
								fclose(fp);
							}

							//Multiply the projected patterns at that patch
							for (cID = 0; cID < nchannels; cID++)
							{
								for (mm = -hsubset; mm <= hsubset; mm++)
								{
									for (nn = -hsubset; nn <= hsubset; nn++)
									{
										rr = (mm + hsubset)*patchS + (nn + hsubset) + cID*patchLength;
										SuperImposePatch[rr] = Pcom[0] * Pro1Patch[rr] * TextPatch[rr] + Pcom[1] * Pro2Patch[rr] * TextPatch[rr] + Pcom[2] * Pro1Patch[rr] + Pcom[3] * Pro2Patch[rr] + Pcom[4] * TextPatch[rr];
									}
								}
							}

							//compute zncc score with patch in the illuminated image vs. patch of texture * patch of projected
							ZNCCscore = ComputeZNCCPatch(IPatch, SuperImposePatch, hsubset, nchannels, ZNNCStorage);
							if (ZNCCscore > bestZNCC) //retain the best score
							{
								bestP1X = tryPro1X, bestP1Y = tryPro1Y, bestP2X = tryPro2X, bestP2Y = tryPro2Y, bestTX = tryTextX, bestTY = tryTextY;
								bestZNCC = ZNCCscore;
								bestqq = qq, bestpp = pp, bestll = ll, bestkk = kk, bestii = ii, bestjj = jj;
							}
							if (ZNCCscore > localZNCC) //retain the best score
								localZNCC = ZNCCscore;
						}
					}
				}
			}
			if (printout2)
				cout << "@pp: " << pp << " (ZNCC, localZNCC): " << bestZNCC << " " << localZNCC << " (db1, mag1, db2, mag2, u, v): " << " " << bestqq << " " << bestpp << " " << bestll << " " << bestkk << " " << bestii << " " << bestjj << endl;
		}
	}
	PuvIuv[0] = bestP1X, PuvIuv[1] = bestP1Y, PuvIuv[2] = bestP2X, PuvIuv[3] = bestP2Y, PuvIuv[4] = bestTX, PuvIuv[5] = bestTY;

	if (createdMem)
		delete[]Pro1Patch, delete[]Pro2Patch, delete[]TextPatch, delete[]SuperImposePatch, delete[]ZNNCStorage;

	return bestZNCC;
}
double TwoIllumTextAllTransSep(IlluminationFlowImages &Fimgs, double *ParaSourceText, double *IPatch, double *direction, CPoint2 *Target, LKParameters LKArg, double *ILWp, double *TWp, double *Pcom, double *ZNCCStorage = 0)
{
	//Model: aL1T+bL2T+cL1+dL2+eT
	int i, j, k, iii, jjj;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold;

	double II0, JJ0, II1, JJ1, II2, JJ2, mF, L1, L2, T, L1x, L1y, L2x, L2y, Tx, Ty, DIC_Coeff, DIC_Coeff_min, a, b, c, d, e, f, t_1, t_2, t_3, t_4, t_5, t_6, S0[3], S1[3], S2[3], p_best[12];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int jumpStep[2] = { 1, 2 }, nn = 12, nExtraParas = 5, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	double AA[12 * 12], BB[12], CC[12], p[12];


	p[0] = 0.0, p[1] = 0.0, p[2] = 0.0, p[3] = 0.0, p[4] = 0.0, p[5] = 0.0, p[6] = Pcom[0], p[7] = Pcom[1], p[8] = Pcom[2], p[9] = Pcom[3], p[10] = Pcom[4], p[11] = Pcom[5];

	int length = width*height, plength = pwidth*pheight, nchannels = 1, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	FILE *fp, *fp0, *fp1, *fp2;

	bool printout = false;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (j = -hsubset; j <= hsubset; j++)
		{
			for (i = -hsubset; i <= hsubset; i++)
				fprintf(fp, "%.2f ", IPatch[(i + hsubset) + (j + hsubset)*TimgS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
			{
				fp = fopen("C:/temp/tar.txt", "w+");
				fp0 = fopen("C:/temp/tar0.txt", "w+");
				fp1 = fopen("C:/temp/tar1.txt", "w+");
				fp2 = fopen("C:/temp/tar2.txt", "w+");
			}

			a = p[6], b = p[7], c = p[8], d = p[9], e = p[10], f = p[11];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					II0 = Target[0].x + iii + p[0] + ILWp[0] * iii + ILWp[1] * jjj;
					JJ0 = Target[0].y + jjj + p[1] + ILWp[2] * iii + ILWp[3] * jjj;

					II1 = Target[1].x + iii + p[2] + ILWp[4] * iii + ILWp[5] * jjj;
					JJ1 = Target[1].y + jjj + p[3] + ILWp[6] * iii + ILWp[7] * jjj;

					II2 = Target[2].x + iii + p[4] + TWp[0] * iii + TWp[1] * jjj;
					JJ2 = Target[2].y + jjj + p[5] + TWp[2] * iii + TWp[3] * jjj;

					if (II0<0.0 || II0>(double)(pwidth - 1) || JJ0<0.0 || JJ0>(double)(pheight - 1))
						continue;
					if (II1<0.0 || II1>(double)(pwidth - 1) || JJ1<0.0 || JJ1>(double)(pheight - 1))
						continue;
					if (II2<0.0 || II2>(double)(width - 1) || JJ2<0.0 || JJ2>(double)(height - 1))
						continue;

					Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, 0, Interpolation_Algorithm);
					Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, 0, Interpolation_Algorithm);
					Get_Value_Spline(ParaSourceText, width, height, II2, JJ2, S2, 0, Interpolation_Algorithm);

					mF = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
					L1 = S0[0], L1x = S0[1], L1y = S0[2], L2 = S1[0], L2x = S1[1], L2y = S1[2], T = S2[0], Tx = S2[1], Ty = S2[2];

					t_4 = a*T + c, t_5 = b*T + d, t_6 = a*L1 + b*L2 + e;
					t_3 = t_4*L1 + t_5*L2 + e*T + f - mF;

					CC[0] = t_4*L1x, CC[1] = t_4*L1y;
					CC[2] = t_5*L2x, CC[3] = t_5*L2y;
					CC[4] = t_6*Tx, CC[5] = t_6*Ty;
					CC[6] = L1*T, CC[7] = L2*T, CC[8] = L1, CC[9] = L2, CC[10] = T, CC[11] = 1.0;

					for (j = 0; j < nn; j++)
					{
						BB[j] += t_3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t_3*t_3;
					t_2 += mF*mF;

					if (printout)
					{
						fprintf(fp, "%.2f ", a*L1*T + b*L2*T + c*L1 + d*L2 + e*T + f);
						fprintf(fp0, "%.2f ", L1);
						fprintf(fp1, "%.2f ", L2);
						fprintf(fp2, "%.2f ", T);
					}
				}
				if (printout)
					fprintf(fp, "\n"), fprintf(fp0, "\n"), fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp), fclose(fp0), fclose(fp1), fclose(fp2);

			DIC_Coeff = t_1 / t_2;
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
			}

			if (p[0] != p[0] || abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[2]) > hsubset || abs(p[3]) > hsubset || abs(p[4]) > hsubset || abs(p[5]) > hsubset || DIC_Coeff > 50)
				return 0.0;
			if (fabs(BB[0]) < conv_crit_1 && abs(BB[1]) < conv_crit_1 && fabs(BB[2]) < conv_crit_1 && fabs(BB[3]) < conv_crit_1 && fabs(BB[4]) < conv_crit_1 && fabs(BB[5]) < conv_crit_1)
				Break_Flag = true;
			if (Break_Flag)
			{
				if (k <= 2)
				{
					Break_Flag = false;
					conv_crit_1 *= 0.1;
				}
				else
					break;
			}
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	if (DIC_Coeff_min < LKArg.PSSDab_thresh)
	{
		bool createMem = false;
		if (ZNCCStorage == NULL)
		{
			createMem = true;
			ZNCCStorage = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
		}

		int m = 0;
		double t_1, t_2, t_3, t_4, t_5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II0 = Target[0].x + iii + p[0] + ILWp[0] * iii + ILWp[1] * jjj;
				JJ0 = Target[0].y + jjj + p[1] + ILWp[2] * iii + ILWp[3] * jjj;

				II1 = Target[1].x + iii + p[2] + ILWp[4] * iii + ILWp[5] * jjj;
				JJ1 = Target[1].y + jjj + p[3] + ILWp[6] * iii + ILWp[7] * jjj;

				II2 = Target[2].x + iii + p[4] + TWp[0] * iii + TWp[1] * jjj;
				JJ2 = Target[2].y + jjj + p[5] + TWp[2] * iii + TWp[3] * jjj;

				if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
					continue;
				if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
					continue;
				if (II2<0.0 || II2>(double)(width - 1) || JJ2<0.0 || JJ2>(double)(height - 1))
					continue;

				Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, -1, Interpolation_Algorithm);
				Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, -1, Interpolation_Algorithm);
				Get_Value_Spline(ParaSourceText, width, height, II2, JJ2, S2, -1, Interpolation_Algorithm);

				ZNCCStorage[2 * m] = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
				ZNCCStorage[2 * m + 1] = p[6] * S0[0] * S2[0] + p[7] * S1[0] * S2[0] + p[8] * S0[0] + p[9] * S1[0] + p[10] * S2[0];
				t_f += ZNCCStorage[2 * m], t_g += ZNCCStorage[2 * m + 1];

				if (printout)
					fprintf(fp, "%.2f ", ZNCCStorage[2 * m + 1]);
				m++;
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / (m + 1), t_g = t_g / (m + 1);
		t_1 = 0.0, t_2 = 0.0, t_3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t_4 = ZNCCStorage[2 * i] - t_f;
			t_5 = ZNCCStorage[2 * i + 1] - t_g;
			t_1 += 1.0*t_4*t_5, t_2 += 1.0*t_4*t_4, t_3 += 1.0*t_5*t_5;
		}

		t_2 = sqrt(t_2*t_3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;

		if (createMem)
			delete[]ZNCCStorage;
	}

	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[2]) > hsubset || abs(p[3]) > hsubset || abs(p[4]) > hsubset || abs(p[5]) > hsubset || DIC_Coeff > 1.0)
		return 0.0;

	//Store the warping parameters
	Target[0].x += p[0], Target[0].y += p[1];
	Target[1].x += p[2], Target[1].y += p[3];
	Target[2].x += p[4], Target[2].y += p[5];
	Pcom[0] = p[6], Pcom[1] = p[7], Pcom[2] = p[8], Pcom[3] = p[9], Pcom[4] = p[10], Pcom[5] = p[11];

	return DIC_Coeff_min;
}
double TwoIllumTextAffineSep(IlluminationFlowImages &Fimgs, double *ParaSourceText, double *IPatch, double *direction, CPoint2 *Target, LKParameters LKArg, double *ILWp, double *TWp, double *Pcom, double *ZNCCStorage = 0)
{
	//Model: aL1T+bL2T+cL1+dL2+eT
	int i, j, k, iii, jjj;
	int width = Fimgs.width, height = Fimgs.height, pwidth = Fimgs.pwidth, pheight = Fimgs.pheight;

	int hsubset = LKArg.hsubset, DIC_Algo = LKArg.DIC_Algo, Interpolation_Algorithm = LKArg.InterpAlgo;
	int Iter_Max = LKArg.IterMax, Convergence_Criteria = LKArg.Convergence_Criteria, Speed = LKArg.Analysis_Speed;
	double znccThresh = LKArg.ZNCCThreshold;

	double II0, JJ0, II1, JJ1, II2, JJ2, mF, L1, L2, T, L1x, L1y, L2x, L2y, Tx, Ty, DIC_Coeff, DIC_Coeff_min, a, b, c, d, e, f, t_1, t_2, t3, t4, t5, t6, t7, S0[3], S1[3], S2[3], p_best[24];
	double conv_crit_1 = 1.0 / pow(10.0, Convergence_Criteria + 2), conv_crit_2 = conv_crit_1*0.01;
	int jumpStep[2] = { 1, 2 }, nn = 24, nExtraParas = 5, _iter = 0;
	int p_jump, p_jump_0 = jumpStep[Speed], p_jump_incr = 1;

	double AA[24 * 24], BB[24], CC[24], p[24];


	p[0] = 0.0, p[1] = 0.0, p[2] = ILWp[0], p[3] = ILWp[1], p[4] = ILWp[2], p[5] = ILWp[3];
	p[6] = 0.0, p[7] = 0.0, p[8] = ILWp[4], p[9] = ILWp[5], p[10] = ILWp[6], p[11] = ILWp[7];
	p[12] = 0.0, p[13] = 0.0, p[14] = TWp[0], p[15] = TWp[1], p[16] = TWp[2], p[17] = TWp[3];
	p[18] = Pcom[0], p[19] = Pcom[1], p[20] = Pcom[2], p[21] = Pcom[3], p[22] = Pcom[4], p[23] = Pcom[5];

	int length = width*height, plength = pwidth*pheight, nchannels = 1, TimgS = 2 * hsubset + 1, Tlength = TimgS*TimgS;
	FILE *fp, *fp0, *fp1, *fp2;

	bool printout = false;
	if (printout)
	{
		fp = fopen("C:/temp/src.txt", "w+");
		for (j = -hsubset; j <= hsubset; j++)
		{
			for (i = -hsubset; i <= hsubset; i++)
				fprintf(fp, "%.2f ", IPatch[(i + hsubset) + (j + hsubset)*TimgS]);
			fprintf(fp, "\n");
		}
		fclose(fp);
	}

	for (p_jump = p_jump_0; p_jump > 0; p_jump -= p_jump_incr)
	{
		DIC_Coeff_min = 1e10;
		bool Break_Flag = false;

		for (k = 0; k < Iter_Max; k++)
		{
			t_1 = 0.0, t_2 = 0.0;
			for (i = 0; i < nn*nn; i++)
				AA[i] = 0.0;
			for (i = 0; i < nn; i++)
				BB[i] = 0.0;

			if (printout)
			{
				fp = fopen("C:/temp/tar.txt", "w+");
				fp0 = fopen("C:/temp/tar0.txt", "w+");
				fp1 = fopen("C:/temp/tar1.txt", "w+");
				fp2 = fopen("C:/temp/tar2.txt", "w+");
			}

			a = p[18], b = p[19], c = p[20], d = p[21], e = p[22], f = p[23];
			for (jjj = -hsubset; jjj <= hsubset; jjj += p_jump)
			{
				for (iii = -hsubset; iii <= hsubset; iii += p_jump)
				{
					II0 = Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj;
					JJ0 = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj;

					II1 = Target[1].x + iii + p[6] + p[8] * iii + p[9] * jjj;
					JJ1 = Target[1].y + jjj + p[7] + p[10] * iii + p[11] * jjj;

					II2 = Target[2].x + iii + p[12] + p[14] * iii + p[15] * jjj;
					JJ2 = Target[2].y + jjj + p[13] + p[16] * iii + p[17] * jjj;

					if (II0<0.0 || II0>(double)(pwidth - 1) || JJ0<0.0 || JJ0>(double)(pheight - 1))
						continue;
					if (II1<0.0 || II1>(double)(pwidth - 1) || JJ1<0.0 || JJ1>(double)(pheight - 1))
						continue;
					if (II2<0.0 || II2>(double)(width - 1) || JJ2<0.0 || JJ2>(double)(height - 1))
						continue;

					Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, 0, Interpolation_Algorithm);
					Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, 0, Interpolation_Algorithm);
					Get_Value_Spline(ParaSourceText, width, height, II2, JJ2, S2, 0, Interpolation_Algorithm);

					mF = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
					L1 = S0[0], L2 = S1[0], L1x = S0[1], L1y = S0[2], L2x = S1[1], L2y = S1[2], T = S2[0], Tx = S2[1], Ty = S2[2];

					t4 = a*T + c, t5 = b*T + d;
					t3 = t4*L1 + t5*L2 + e*T + f - mF;

					t6 = t4*L1x, t7 = t4*L1y;
					CC[0] = t6, CC[1] = t7, CC[2] = t6*iii, CC[3] = t6*jjj, CC[4] = t7*iii, CC[5] = t7*jjj;

					t6 = t5*L2x, t7 = t5*L2y;
					CC[6] = t6, CC[7] = t7, CC[8] = t6*iii, CC[9] = t6*jjj, CC[10] = t7*iii, CC[11] = t7*jjj;

					t4 = a*L1 + b*L2 + e, t6 = t4*Tx, t7 = t4*Ty;
					CC[12] = t6, CC[13] = t7, CC[14] = t6*iii, CC[15] = t6*jjj, CC[16] = t7*iii, CC[17] = t7*jjj;

					CC[18] = L1*T, CC[19] = L2*T, CC[20] = L1, CC[21] = L2, CC[22] = T, CC[23] = 1.0;

					for (j = 0; j < nn; j++)
					{
						BB[j] += t3*CC[j];
						for (i = 0; i < nn; i++)
							AA[j*nn + i] += CC[i] * CC[j];
					}

					t_1 += t3*t3;
					t_2 += mF*mF;

					if (printout)
					{
						fprintf(fp, "%.2f ", a*L1*T + b*L2*T + c*L1 + d*L2 + e*T + f);
						fprintf(fp0, "%.2f ", L1);
						fprintf(fp1, "%.2f ", L2);
						fprintf(fp2, "%.2f ", T);
					}
				}
				if (printout)
					fprintf(fp, "\n"), fprintf(fp0, "\n"), fprintf(fp1, "\n"), fprintf(fp2, "\n");
			}
			if (printout)
				fclose(fp), fclose(fp0), fclose(fp1), fclose(fp2);

			DIC_Coeff = t_1 / t_2;
			QR_Solution_Double(AA, BB, nn, nn);
			for (i = 0; i < nn; i++)
				p[i] -= BB[i];

			if (DIC_Coeff < DIC_Coeff_min)	// If the iteration does not converge, this can be helpful
			{
				DIC_Coeff_min = DIC_Coeff;
				for (i = 0; i < nn; i++)
					p_best[i] = p[i];
			}

			//weirdly distortted shape-->must be wrong
			if (abs((p[14] + 1) / p[15]) + abs((p[17] + 1) / p[16]) < 10 || abs((p[2] + 1) / p[3]) + abs((p[5] + 1) / p[4]) < 6 || abs((p[8] + 1) / p[9]) + abs((p[11] + 1) / p[10]) < 6)
				return 0;
			if (abs(p[2] + 1) / abs(p[8] + 1) + abs(p[5] + 1) / abs(p[11] + 1) > 5 || abs(p[2] + 1) / abs(p[8] + 1) + abs(p[5] + 1) / abs(p[11] + 1) < 0.2)
				return 0;
			if (p[0] != p[0] || abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[6]) > hsubset || abs(p[7]) > hsubset || abs(p[12]) > hsubset || abs(p[3]) > hsubset || DIC_Coeff > 50)
				return 0.0;
			if (fabs(BB[0]) < conv_crit_1 && abs(BB[1]) < conv_crit_1 && fabs(BB[6]) < conv_crit_1 && fabs(BB[7]) < conv_crit_1 && fabs(BB[12]) < conv_crit_1 && fabs(BB[13]) < conv_crit_1)
			{
				for (i = 2; i < 6; i++)
					if (fabs(BB[i]) > conv_crit_2 && abs(BB[i + 6]) > conv_crit_2)
						break;
				if (i == 6)
					for (i = 14; i < 18; i++)
						if (fabs(BB[i]) > conv_crit_2*3.0)
							break;
				if (i == 18)
					Break_Flag = true;
			}
			if (Break_Flag)
			{
				if (k <= 2)
				{
					Break_Flag = false;
					conv_crit_1 *= 0.1, conv_crit_2 *= 0.1;
				}
				else
					break;
			}
		}
		_iter += k;
		// In case the iteration converges to "wrong" points, always use the data that lead to the least-square value.
		for (i = 0; i < nn; i++)
			p[i] = p_best[i];
	}
	/// DIC Iteration: End

	//Now, dont really trust the pssad error too much, compute zncc score instead! They are usually close when the convergence goes smothly, but in case of trouble, zncc is more reliable.
	if (DIC_Coeff_min < LKArg.PSSDab_thresh)
	{
		bool createMem = false;
		if (ZNCCStorage == NULL)
		{
			createMem = true;
			ZNCCStorage = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
		}

		int m = 0;
		double t_1, t_2, t3, t4, t5, t_f = 0.0, t_g = 0.0;
		if (printout)
			fp = fopen("C:/temp/tar.txt", "w+");
		for (jjj = -hsubset; jjj <= hsubset; jjj++)
		{
			for (iii = -hsubset; iii <= hsubset; iii++)
			{
				II0 = Target[0].x + iii + p[0] + p[2] * iii + p[3] * jjj;
				JJ0 = Target[0].y + jjj + p[1] + p[4] * iii + p[5] * jjj;

				II1 = Target[1].x + iii + p[6] + p[8] * iii + p[9] * jjj;
				JJ1 = Target[1].y + jjj + p[7] + p[10] * iii + p[11] * jjj;

				II2 = Target[2].x + iii + p[12] + p[14] * iii + p[15] * jjj;
				JJ2 = Target[2].y + jjj + p[13] + p[16] * iii + p[17] * jjj;

				if (II0<0.0 || II0>(double)(pwidth - 1) - (1e-10) || JJ0<0.0 || JJ0>(double)(pheight - 1) - (1e-10))
					continue;
				if (II1<0.0 || II1>(double)(width - 1) - (1e-10) || JJ1<0.0 || JJ1>(double)(height - 1) - (1e-10))
					continue;
				if (II2<0.0 || II2>(double)(width - 1) || JJ2<0.0 || JJ2>(double)(height - 1))
					continue;

				Get_Value_Spline(Fimgs.PPara, pwidth, pheight, II0, JJ0, S0, -1, Interpolation_Algorithm);
				Get_Value_Spline(Fimgs.PPara + plength, pwidth, pheight, II1, JJ1, S1, -1, Interpolation_Algorithm);
				Get_Value_Spline(ParaSourceText, width, height, II2, JJ2, S2, -1, Interpolation_Algorithm);

				ZNCCStorage[2 * m] = IPatch[(iii + hsubset) + (jjj + hsubset)*TimgS];
				ZNCCStorage[2 * m + 1] = p[18] * S0[0] * S2[0] + p[19] * S1[0] * S2[0] + p[20] * S0[0] + p[21] * S1[0] + p[22] * S2[0];
				t_f += ZNCCStorage[2 * m], t_g += ZNCCStorage[2 * m + 1];

				if (printout)
					fprintf(fp, "%.2f ", ZNCCStorage[2 * m + 1]);
				m++;
			}
			if (printout)
				fprintf(fp, "\n");
		}
		if (printout)
			fclose(fp);

		t_f = t_f / (m + 1), t_g = t_g / (m + 1);
		t_1 = 0.0, t_2 = 0.0, t3 = 0.0;
		for (i = 0; i < m; i++)
		{
			t4 = ZNCCStorage[2 * i] - t_f;
			t5 = ZNCCStorage[2 * i + 1] - t_g;
			t_1 += 1.0*t4*t5, t_2 += 1.0*t4*t4, t3 += 1.0*t5*t5;
		}

		t_2 = sqrt(t_2*t3);
		if (t_2 < 1e-10)
			t_2 = 1e-10;

		DIC_Coeff_min = t_1 / t_2; //This is the zncc score
		if (abs(DIC_Coeff_min) > 1.0)
			DIC_Coeff_min = 0.0;

		if (createMem)
			delete[]ZNCCStorage;
	}

	if (abs(p[0]) > hsubset || abs(p[1]) > hsubset || abs(p[6]) > hsubset || abs(p[7]) > hsubset || DIC_Coeff > 50)
		return 0.0;

	//Store the warping parameters
	for (i = 0; i < 4; i++)
		ILWp[i] = p[2 + i], ILWp[i + 4] = p[8 + i], TWp[i] = p[14 + i];
	Pcom[0] = p[18], Pcom[1] = p[19], Pcom[2] = p[20], Pcom[3] = p[21], Pcom[4] = p[22], Pcom[5] = p[23];

	Target[0].x += p[0], Target[0].y += p[1];
	Target[1].x += p[6], Target[1].y += p[7];
	Target[2].x += p[12], Target[2].y += p[13];

	return DIC_Coeff_min;
}

int PureIllumCase(double &score, int x, int y, int nx, int ny, int ProID, double *direction, double *proEpiline, DevicesInfo &DInfo, double *P1mat, IlluminationFlowImages &Fimgs, float *ILWarping, int *PrecomSearchR, LKParameters LKArg, double *IPatch, double *TarPatch, double *ZNCCStorage, double *iWp = 0, int *iStartP = 0)
{
	int nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, plength = pwidth*pheight, length = width*height;
	int ii, id = x + y*width, nid = nx + ny*width, nseedtype = 0;

	double denum, t1, t2, ILWp[8];
	CPoint2  dPts[2], startP[2];

	if (PrecomSearchR[nid]>4) //Don't try if deep inside mixed regions
		return nseedtype;

	bool flag = false;
	if (x == nx && y == ny) //fresh start, need special handling
	{
		startP[ProID].x = iStartP[0], startP[ProID].y = iStartP[1];
		ILWp[ProID * 4] = iWp[0], ILWp[ProID * 4 + 1] = iWp[1], ILWp[ProID * 4 + 2] = iWp[2], ILWp[ProID * 4 + 3] = iWp[3];
	}
	else
	{
		startP[ProID].x = x + ILWarping[id + 6 * ProID*length], startP[ProID].y = y + ILWarping[id + (1 + 6 * ProID)*length];
		ILWp[ProID * 4] = ILWarping[id + (2 + 6 * ProID)*length], ILWp[ProID * 4 + 1] = ILWarping[id + (3 + 6 * ProID)*length];
		ILWp[ProID * 4 + 2] = ILWarping[id + (4 + 6 * ProID)*length], ILWp[ProID * 4 + 3] = ILWarping[id + (5 + 6 * ProID)*length];
	}

	if (LKArg.EpipEnforce == 1) 	//Project onto epipolar line
	{
		denum = pow(proEpiline[3 * ProID], 2) + pow(proEpiline[3 * ProID + 1], 2);
		t1 = startP[ProID].x, t2 = startP[ProID].y;
		startP[ProID].x = (proEpiline[3 * ProID + 1] * (proEpiline[3 * ProID + 1] * t1 - proEpiline[3 * ProID] * t2) - proEpiline[3 * ProID] * proEpiline[3 * ProID + 2]) / denum;
		startP[ProID].y = (proEpiline[3 * ProID] * (-proEpiline[3 * ProID + 1] * t1 + proEpiline[3 * ProID] * t2) - proEpiline[3 * ProID + 1] * proEpiline[3 * ProID + 2]) / denum;
	}

	dPts[0].x = nx, dPts[0].y = ny, dPts[1].x = startP[ProID].x, dPts[1].y = startP[ProID].y;
	score = EpipSearchLK(dPts, proEpiline + 3 * ProID, Fimgs.Img, Fimgs.PImg + ProID*nchannels*plength, Fimgs.Para, Fimgs.PPara + ProID*nchannels*plength, nchannels, width, height, pwidth, pheight, LKArg, IPatch, ZNCCStorage, TarPatch, ILWp + 4 * ProID);
	if (score > LKArg.ZNCCThreshold)
		flag = true, nseedtype = ProID + 1;

	if (flag)
	{
		ILWarping[nid + 6 * ProID*length] = dPts[1].x - nx, ILWarping[nid + (1 + 6 * ProID)*length] = dPts[1].y - ny;
		for (ii = 0; ii < 4; ii++)
			ILWarping[nid + (ii + 2 + 6 * ProID)*length] = ILWp[ii + 4 * ProID];
	}

	return nseedtype;
}
int TwoIllumCase(double &score, int x, int y, int nx, int ny, int seedtype, double *direction, double *proEpiline, DevicesInfo &DInfo, double *P1mat, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *ILWarping, float *TWarping, float *PhotoAdj, int *PrecomSearchR, LKParameters LKArg, double *IPatch, double *TarPatch, double *Pro1Patch, double *Pro2Patch, double *SuperImposePatch, double *ZNCCStorage, CPoint2 *HFrom, CPoint2 *HTo, double *ASolver, double *BSolver, double *iWp = 0, int *iStartP = 0, int *irangeS = 0)
{
	int nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, plength = pwidth*pheight, length = width*height;
	int mm, nn, ll, u, v, id = x + y*width, nid = nx + ny*width, nseedtype = 0, ProID, OtherProID;
	int hsubset = LKArg.hsubset, patchS = 2 * hsubset + 1, patchLength = patchS*patchS, rangeS[3] = { 10, 10, 10 };

	double t1, t2, denum, puviuv[6], ILWp[8], Pcom[2];
	CPoint2  startP[2], dPts[2];
	CPoint foundP[2];
	CPoint3 WC;

	if (x == nx && y == ny) //fresh start, need special handling
	{
		for (ll = 0; ll < 8; ll++)
			ILWp[ll] = iWp[ll];
		for (ll = 0; ll < 2; ll++)
			startP[ll].x = iStartP[2 * ll], startP[ll].y = iStartP[2 * ll + 1], Pcom[ll] = 0.3, rangeS[ll] = irangeS[ll];
	}
	else
	{
		if (seedtype < 3)
		{
			ProID = seedtype - 1, OtherProID = (seedtype == 1) ? 1 : 0, rangeS[ProID] = 1;
			startP[ProID].x = x + ILWarping[id + 6 * ProID*length], startP[ProID].y = y + ILWarping[id + (1 + 6 * ProID)*length];
			for (ll = 0; ll < 4; ll++)
				ILWp[4 * ProID + ll] = ILWarping[id + (2 + ll + 6 * ProID)*length];

			if (!IsLocalWarpAvail(ILWarping + 6 * OtherProID*length, ILWp + 4 * OtherProID, nx, ny, foundP[OtherProID], u, v, rangeS[OtherProID], width, height, PrecomSearchR[nid]))
			{
				rangeS[OtherProID] = min(EstimateIllumPatchAffine(nx, ny, ProID, OtherProID, P1mat, ILWarping, startP[OtherProID], ILWp + 4 * OtherProID, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver), PrecomSearchR[id]);
				if (rangeS[OtherProID] == 0)
					return 0;
			}
			else
				startP[OtherProID].x = u, startP[OtherProID].y = v;
			Pcom[0] = 0.3, Pcom[1] = 0.3;
		}
		else if (seedtype == 3 || seedtype == 6)
		{
			rangeS[0] = 1, rangeS[1] = 1;
			startP[0].x = x + ILWarping[id], startP[0].y = y + ILWarping[id + length], startP[1].x = x + ILWarping[id + 6 * length], startP[1].y = y + ILWarping[id + 7 * length];
			for (ll = 0; ll < 4; ll++)
				ILWp[ll] = ILWarping[id + (2 + ll)*length], ILWp[4 + ll] = ILWarping[id + (8 + ll)*length];
			if (seedtype == 3)
				Pcom[0] = PhotoAdj[id], Pcom[1] = PhotoAdj[id + length];
			else
				Pcom[0] = 0.3, Pcom[1] = 0.3;
		}

		if (LKArg.EpipEnforce == 1) 	//Project onto epipolar line
		{
			denum = pow(proEpiline[0], 2) + pow(proEpiline[1], 2);
			t1 = startP[0].x, t2 = startP[0].y;
			startP[0].x = (proEpiline[1] * (proEpiline[1] * t1 - proEpiline[0] * t2) - proEpiline[0] * proEpiline[2]) / denum;
			startP[0].y = (proEpiline[0] * (-proEpiline[1] * t1 + proEpiline[0] * t2) - proEpiline[1] * proEpiline[2]) / denum;

			denum = pow(proEpiline[3], 2) + pow(proEpiline[4], 2);
			t1 = startP[1].x, t2 = startP[1].y;
			startP[1].x = (proEpiline[4] * (proEpiline[4] * t1 - proEpiline[3] * t2) - proEpiline[3] * proEpiline[5]) / denum;
			startP[1].y = (proEpiline[3] * (-proEpiline[4] * t1 + proEpiline[3] * t2) - proEpiline[4] * proEpiline[5]) / denum;
		}
	}

	//Take the observed image patch
	for (ll = 0; ll < nchannels; ll++)
		for (mm = -hsubset; mm <= hsubset; mm++)
			for (nn = -hsubset; nn <= hsubset; nn++)
				IPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength] = Fimgs.Img[(nx + nn) + (ny + mm)*width + ll*length];

	if (rangeS[0] > 2 || rangeS[1] > 2)
	{
		score = llumSepCoarse(Fimgs, IPatch, startP, direction, ILWp, Pcom, hsubset, LKArg.searchRangeScale*rangeS[0], LKArg.searchRangeScale*rangeS[1], LKArg.InterpAlgo, puviuv, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage);
		dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
	}
	else //Just use previous points as inital guess is OK
	{
		dPts[0].x = startP[0].x, dPts[0].y = startP[0].y, dPts[1].x = startP[1].x, dPts[1].y = startP[1].y;
		score = LKArg.ZNCCThreshold;
	}

	//if (score > LKArg.ZNCCThreshold - 0.35)
	//	score = IllumTransSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);

	if (score > LKArg.ZNCCThreshold - 0.35)
	{
		score = IllumAffineSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);
		if (score < LKArg.ZNCCThreshold - 0.035 && (rangeS[0] <= 2 || rangeS[1] <= 2))
		{
			score = llumSepCoarse(Fimgs, IPatch, startP, direction, ILWp, Pcom, hsubset, 2, 2, LKArg.InterpAlgo, puviuv, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage);
			dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
			if (score > LKArg.ZNCCThreshold - 0.35)
				score = IllumAffineSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);
		}
	}

	if (score < LKArg.ZNCCThreshold - 0.035 && (dPts[0].x < 40 || dPts[0].x > pwidth - 40 || dPts[1].x <40 || dPts[1].x>pwidth - 40)) //special handling at the boundary
	{
		LKArg.hsubset -= 2, hsubset -= 2, patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
		LKArg.ZNCCThreshold -= 0.05;

		for (ll = 0; ll < nchannels; ll++)
			for (mm = -hsubset; mm <= hsubset; mm++)
				for (nn = -hsubset; nn <= hsubset; nn++)
					IPatch[(mm + hsubset)*patchS + (nn + hsubset) + ll*patchLength] = Fimgs.Img[(nx + nn) + (ny + mm)*width + ll*length];

		//score = IllumAffineSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);
		score = IllumTransSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);
		if (score < LKArg.ZNCCThreshold - 0.035 && (rangeS[0] <= 2 || rangeS[1] <= 2))
		{
			score = llumSepCoarse(Fimgs, IPatch, startP, direction, ILWp, Pcom, hsubset, 2, 2, LKArg.InterpAlgo, puviuv, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage);
			dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
			if (score > LKArg.ZNCCThreshold - 0.35)
				score = IllumTransSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);//score = IllumAffineSep(Fimgs, IPatch, direction, dPts, LKArg, ILWp, Pcom, ZNCCStorage);
		}
	}

	if (score > LKArg.ZNCCThreshold - 0.035)
	{
		if (CamProGeoVerify((double)nx, (double)ny, dPts, WC, P1mat, DInfo, width, height, pwidth, pheight, 2) == 0)
		{
			nseedtype = 3;
			ILWarping[nid] = dPts[0].x - nx, ILWarping[nid + length] = dPts[0].y - ny, ILWarping[nid + 6 * length] = dPts[1].x - nx, ILWarping[nid + 7 * length] = dPts[1].y - ny;
			PhotoAdj[nid] = Pcom[0], PhotoAdj[nid + length] = Pcom[1];
			for (ll = 0; ll < 4; ll++)
				ILWarping[nid + (ll + 2)*length] = ILWp[ll], ILWarping[nid + (ll + 8)*length] = ILWp[ll + 4];
		}
	}

	return nseedtype;
}
int IllumTextCase(double &score, int x, int y, int nx, int ny, int seedtype, int mode, double *direction, double *proEpiline, DevicesInfo &DInfo, double *P1mat, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *PrecomSearchR, LKParameters LKArg, double *IPatch, double *TarPatch, double *Pro1Patch, double *TextPatch, double *SuperImposePatch, double *ZNCCStorage, double *iLWp = 0, int *iStartP = 0, double *iTWp = 0, int *iStartT = 0, int *ProjectorID = 0, int *irangeS = 0)
{
	int nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, plength = pwidth*pheight, length = width*height;
	int mm, nn, ll, u, v, id = x + y*width, nid = nx + ny*width, nseedtype = 0, ProID;
	int hsubset = LKArg.hsubset, patchS = 2 * hsubset + 1, patchLength = patchS*patchS, rangeS[3] = { 1, 1, 3 };

	double t1, t2, denum, puviuv[6], ILWp[8], TWp[4], Pcom[3];
	CPoint2  puv, iuv, startT, startP[2], dPts[2];
	CPoint foundP;
	CPoint3 WC;

	if (x == nx && y == ny) //fresh start, need special handling
	{
		ProID = ProjectorID[0];
		for (ll = 0; ll < 4; ll++)
			ILWp[ll] = iLWp[ll], ILWp[ll + 4] = iLWp[ll + 4], TWp[ll] = 0.0;

		startP[ProID].x = iStartP[0], startP[ProID].y = iStartP[1], startT.x = iStartT[0], startT.y = iStartT[1];
		Pcom[0] = 1.0 / 255.0, Pcom[1] = 0.0, Pcom[2] = 0.0, rangeS[ProID] = irangeS[ProID], rangeS[2] = irangeS[2];
	}
	else
	{
		if (seedtype < 3)
		{
			ProID = seedtype - 1;
			if (!IsLocalWarpAvail(TWarping, TWp, nx, ny, foundP, u, v, rangeS[2], width, height, PrecomSearchR[nid]))
			{
				if (mode == 0)
				{
					startT.x = nx, startT.y = ny;
					rangeS[2] = LKArg.searchRange;
					for (ll = 0; ll < 4; ll++)
						TWp[ll] = 0.0;
					Pcom[0] = 1.0 / 255.0, Pcom[1] = 0.0, Pcom[2] = 0.0;
				}
				else if (mode == 1 && !IsLocalWarpAvail(previousTWarping, TWp, nx, ny, foundP, u, v, rangeS[2], width, height, PrecomSearchR[nid]))
					return nseedtype; // Does not find any closed by points. 
			}
			else
			{
				int idf = foundP.x + foundP.y*width;
				startT.x = u, startT.y = v;
				Pcom[0] = PhotoAdj[idf], Pcom[1] = PhotoAdj[idf + length], Pcom[2] = PhotoAdj[idf + 2 * length];
			}
		}
		else if (seedtype == 4 || seedtype == 5)
		{
			ProID = seedtype - 4, rangeS[2] = 1;
			startT.x = TWarping[id] + x, startT.y = TWarping[id + length] + y;
			startP[ProID].x = ILWarping[id + 6 * ProID*length] + x, startP[ProID].y = ILWarping[id + (1 + 6 * ProID)*length] + y;
			for (ll = 0; ll < 4; ll++)
				ILWp[ProID * 4 + ll] = ILWarping[id + (2 + ll + 6 * ProID)*length], TWp[ll] = TWarping[id + (2 + ll)*length];
			Pcom[0] = PhotoAdj[id], Pcom[1] = PhotoAdj[id + length], Pcom[2] = PhotoAdj[id + 2 * length];
			u = (int)(startT.x) + (nx - x), v = (int)(startT.y) + (ny - y);
		}

		if (LKArg.EpipEnforce == 1) 	//Project onto epipolar line
		{
			denum = pow(proEpiline[3 * ProID], 2) + pow(proEpiline[3 * ProID + 1], 2);
			t1 = startP[ProID].x, t2 = startP[ProID].y;
			startP[ProID].x = (proEpiline[3 * ProID + 1] * (proEpiline[3 * ProID + 1] * t1 - proEpiline[3 * ProID] * t2) - proEpiline[3 * ProID] * proEpiline[3 * ProID + 2]) / denum;
			startP[ProID].y = (proEpiline[3 * ProID] * (-proEpiline[3 * ProID + 1] * t1 + proEpiline[3 * ProID] * t2) - proEpiline[3 * ProID + 1] * proEpiline[3 * ProID + 2]) / denum;
		}
	}

	double ssig = SSIG[nx + ny*width + length]; //current SSIG
	if (ssig < LKArg.ssigThresh) // texture is enough
		return 0;
	if (ssig < 150.0)
		LKArg.hsubset += 2, patchS = 2 * LKArg.hsubset + 1, patchLength = patchS*patchS;

	ssig = SSIG[(int)startT.x + ((int)startT.y)*width];//source SSIG
	if (ssig < LKArg.ssigThresh)
		return 0;
	if (ssig < 200.0 &&  LKArg.hsubset == hsubset)
		LKArg.hsubset += 2, patchS = 2 * LKArg.hsubset + 1, patchLength = patchS*patchS;

	for (ll = 0; ll < nchannels; ll++)
		for (mm = -LKArg.hsubset; mm <= LKArg.hsubset; mm++)
			for (nn = -LKArg.hsubset; nn <= LKArg.hsubset; nn++)
				IPatch[(mm + LKArg.hsubset)*patchS + (nn + LKArg.hsubset) + ll*patchLength] = Fimgs.Img[(nx + nn) + (ny + mm)*width + ll*length];

	bool bruteforcesreach = false;
	if (rangeS[ProID] > 2 || rangeS[2] > 2)
	{
		bruteforcesreach = true;
		score = TextIllumSepCoarse(ProID, Fimgs, SoureTexture, ParaSourceTexture, IPatch, startT, startP[ProID], direction + ProID * 2, ILWp + ProID * 4, TWp, Pcom, LKArg.hsubset, rangeS[ProID], rangeS[2], LKArg.InterpAlgo, puviuv, Pro1Patch, TextPatch, SuperImposePatch, ZNCCStorage);
		ssig = SSIG[(int)puviuv[2] + ((int)puviuv[3])*width]; //source SSIG
		if (ssig < LKArg.ssigThresh || score < LKArg.ZNCCThreshold - 0.35)
			return 0;
		dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
	}
	else
	{
		puviuv[0] = startP[ProID].x, puviuv[1] = startP[ProID].y, puviuv[2] = startT.x, puviuv[3] = startT.y;
		dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
		score = LKArg.ZNCCThreshold;
	}

	bool TransSearch = false;
	if (!(rangeS[ProID] == 1 && rangeS[2] == 1) && score > LKArg.ZNCCThreshold - 0.35)
	{
		TransSearch = true;
		score = TextTransSep(ProID, Fimgs, ParaSourceTexture, IPatch, direction + ProID * 2, dPts, LKArg, ILWp + ProID * 4, TWp, Pcom, ZNCCStorage);
		if (score < LKArg.ZNCCThreshold - 0.06)
			return 0;
	}

	score = TextIIllumAffineSep(ProID, Fimgs, ParaSourceTexture, IPatch, direction + ProID * 2, dPts, LKArg, ILWp + ProID * 4, TWp, Pcom, ZNCCStorage);
	if (score < LKArg.ZNCCThreshold - 0.03 && rangeS[ProID] <= 2 && rangeS[2] <= 2 && !bruteforcesreach)
	{
		//printf("LT: Score: %.3f\n", score);
		if (rangeS[2] == 1 && rangeS[ProID] == 1 && !TransSearch)
		{
			score = TextTransSep(ProID, Fimgs, ParaSourceTexture, IPatch, direction + ProID * 2, dPts, LKArg, ILWp + ProID * 4, TWp, Pcom, ZNCCStorage);
			if (score < LKArg.ZNCCThreshold - 0.06)
				return 0;
		}
		else if (!bruteforcesreach)
		{
			if (LKArg.EpipEnforce == 1)
				rangeS[3] = 0;
			else
				rangeS[3] = 1;
			score = TextIllumSepCoarse(ProID, Fimgs, SoureTexture, ParaSourceTexture, IPatch, startT, startP[ProID], direction + ProID * 2, ILWp + ProID * 4, TWp, Pcom, LKArg.hsubset, rangeS[ProID], rangeS[2], LKArg.InterpAlgo, puviuv, Pro1Patch, TextPatch, SuperImposePatch, ZNCCStorage);
			ssig = SSIG[(int)puviuv[2] + ((int)puviuv[3])*width]; //source SSIG
			if (ssig < LKArg.ssigThresh || score < LKArg.ZNCCThreshold - 0.35)
				return 0;
			dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3];
		}
		else
			return 0;
		score = TextIIllumAffineSep(ProID, Fimgs, ParaSourceTexture, IPatch, direction + ProID * 2, dPts, LKArg, ILWp + ProID * 4, TWp, Pcom, ZNCCStorage);
	}
	if (score > LKArg.ZNCCThreshold - 0.03)
	{
		if (CamProGeoVerify(1.0*nx, 1.0*ny, dPts, WC, P1mat, DInfo, width, height, pwidth, pheight, ProID) == 0)
		{
			nseedtype = ProID + 4;
			ILWarping[nid + 6 * ProID*length] = dPts[0].x - nx, ILWarping[nid + (1 + 6 * ProID)*length] = dPts[0].y - ny, TWarping[nid] = dPts[1].x - nx, TWarping[nid + length] = dPts[1].y - ny;
			for (ll = 0; ll < 4; ll++)
				ILWarping[nid + (2 + ll + 6 * ProID)*length] = ILWp[ll + 4 * ProID], TWarping[nid + (2 + ll)*length] = TWp[ll];
			PhotoAdj[nid] = Pcom[0], PhotoAdj[nid + length] = Pcom[1], PhotoAdj[nid + 2 * length] = Pcom[2];
		}
	}

	return nseedtype;
}
int TwoIllumTextCase(double &score, int x, int y, int nx, int ny, int seedtype, int mode, double *direction, double *proEpiline, DevicesInfo &DInfo, double *P1mat, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *PrecomSearchR, LKParameters LKArg, double *IPatch, double *TarPatch, double *Pro1Patch, double *Pro2Patch, double *TextPatch, double *SuperImposePatch, double *ZNCCStorage, CPoint2 *HFrom, CPoint2 *HTo, double *ASolver, double *BSolver, double *iLWp = 0, double *iTWp = 0, int *iStartP = 0, int *iStartT = 0, int *irangeS = 0)
{
	const double intentsityFalloff = 1.0 / 255.0;
	int nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, plength = pwidth*pheight, length = width*height;
	int mm, nn, ll, id = x + y*width, nid = nx + ny*width, nseedtype = 0;
	int rangeS[4] = { 1, 1, 3, 0 };

	double ssig, t1, t2, denum, puviuv[6], ILWp[8], TWp[4], Pcom[6];
	CPoint2 startT, startP[2], dPts[3];

	CPoint3 WC;  CPoint foundP;
	if (x == nx && y == ny)//fresh start, need special handling
	{
		for (ll = 0; ll < 4; ll++)
			ILWp[ll] = iLWp[ll], ILWp[ll + 4] = iLWp[ll + 4], TWp[ll] = iTWp[ll];
		for (ll = 0; ll < 2; ll++)
			startP[ll].x = iStartP[2 * ll], startP[ll].y = iStartP[2 * ll + 1];
		for (ll = 0; ll < 3; ll++)
			rangeS[ll] = irangeS[ll];
		startT.x = iStartT[0], startT.y = iStartT[1];
		Pcom[0] = intentsityFalloff, Pcom[1] = intentsityFalloff, Pcom[2] = 0.5, Pcom[3] = 0.5, Pcom[4] = 1.0, Pcom[5] = 0.0;

		ssig = SSIG[nx + ny*width + length];//current SSIG
		if (ssig < LKArg.ssigThresh)
			return 0;
		//if (ssig < 500.0)
		//	LKArg.hsubset += 2;
		//hsubset = LKArg.hsubset, patchS = 2 * hsubset + 1, patchLength = patchS*patchS;
	}
	else
	{
		if (!IsLocalWarpAvail(TWarping, TWp, nx, ny, foundP, mm, nn, rangeS[2], width, height, PrecomSearchR[nid]))
		{
			if (mode == 0)
			{
				startT.x = nx, startT.y = ny;
				rangeS[2] = LKArg.searchRange;
				for (ll = 0; ll < 4; ll++)
					TWp[ll] = 0.0;
			}
			else
				if (IsLocalWarpAvail(previousTWarping, TWp, nx, ny, foundP, mm, nn, rangeS[2], width, height, PrecomSearchR[nid]))
					rangeS[2] += 2;
				else
					return nseedtype;
		}
		else
			startT.x = mm, startT.y = nn;

		ssig = SSIG[nx + ny*width + length]; //current image
		if (ssig < LKArg.ssigThresh)
			return 0;

		if (seedtype == 3 || seedtype == 6)
		{
			rangeS[0] = 1, rangeS[1] = 1;
			startP[0].x = x + ILWarping[id], startP[0].y = y + ILWarping[id + length], startP[1].x = x + ILWarping[id + 6 * length], startP[1].y = y + ILWarping[id + 7 * length];
			for (ll = 0; ll < 4; ll++)
				ILWp[ll] = ILWarping[id + (ll + 2)*length], ILWp[4 + ll] = ILWarping[id + (ll + 8)*length];
			if (seedtype == 3)
				Pcom[0] = intentsityFalloff, Pcom[1] = intentsityFalloff, Pcom[2] = 0.5, Pcom[3] = 0.5, Pcom[4] = 1.0, Pcom[5] = 0.0;
			else
			{
				rangeS[2] = 1;
				for (ll = 0; ll < 6; ll++)
					Pcom[ll] = PhotoAdj[id + ll*length];
			}
		}
		else if (seedtype == 4 || seedtype == 5)
		{
			int ProID = seedtype - 4, OtherProID = (seedtype == 4) ? 1 : 0; rangeS[ProID] = 2;
			startP[ProID].x = x + ILWarping[id + 6 * ProID*length], startP[ProID].y = y + ILWarping[id + (1 + 6 * ProID)*length];
			for (ll = 0; ll < 4; ll++)
				ILWp[4 * ProID + ll] = ILWarping[id + (2 + ll + 6 * ProID)*length];

			if (!IsLocalWarpAvail(ILWarping + 6 * OtherProID*length, ILWp + 4 * OtherProID, nx, ny, foundP, mm, nn, rangeS[OtherProID], width, height, PrecomSearchR[nid]))
			{
				rangeS[OtherProID] = min(EstimateIllumPatchAffine(nx, ny, ProID, OtherProID, P1mat, ILWarping, startP[OtherProID], ILWp + 4 * OtherProID, DInfo, LKArg.hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver), PrecomSearchR[id]);
				if (rangeS[OtherProID] == 0)
					return 0;
			}
			else
				startP[OtherProID].x = mm, startP[OtherProID].y = nn;
			Pcom[0] = intentsityFalloff, Pcom[1] = intentsityFalloff, Pcom[2] = 0.5, Pcom[3] = 0.5, Pcom[4] = 1.0, Pcom[5] = 0.0;
		}

		if (LKArg.EpipEnforce == 1) 	//Project onto epipolar line
		{
			denum = pow(proEpiline[0], 2) + pow(proEpiline[1], 2);
			t1 = startP[0].x, t2 = startP[0].y;
			startP[0].x = (proEpiline[1] * (proEpiline[1] * t1 - proEpiline[0] * t2) - proEpiline[0] * proEpiline[2]) / denum;
			startP[0].y = (proEpiline[0] * (-proEpiline[1] * t1 + proEpiline[0] * t2) - proEpiline[1] * proEpiline[2]) / denum;

			denum = pow(proEpiline[3], 2) + pow(proEpiline[4], 2);
			t1 = startP[1].x, t2 = startP[1].y;
			startP[1].x = (proEpiline[4] * (proEpiline[4] * t1 - proEpiline[3] * t2) - proEpiline[3] * proEpiline[5]) / denum;
			startP[1].y = (proEpiline[3] * (-proEpiline[4] * t1 + proEpiline[3] * t2) - proEpiline[4] * proEpiline[5]) / denum;
		}
	}

	ssig = SSIG[(int)startT.x + ((int)startT.y)*width];//source SSIG
	if (ssig < LKArg.ssigThresh)
		return 0;
	if (ssig < 500.0)
		LKArg.hsubset += 2;
	int patchS = 2 * LKArg.hsubset + 1, patchLength = patchS*patchS;

	for (ll = 0; ll < nchannels; ll++) //Take the observed image patch
		for (mm = -LKArg.hsubset; mm <= LKArg.hsubset; mm++)
			for (nn = -LKArg.hsubset; nn <= LKArg.hsubset; nn++)
				IPatch[(mm + LKArg.hsubset)*patchS + (nn + LKArg.hsubset) + ll*patchLength] = Fimgs.Img[(nx + nn) + (ny + mm)*width + ll*length];

	bool bruteforcesearch = false;
	if (rangeS[2] >= 2 || rangeS[0] >= 2 || rangeS[1] >= 2)
	{
		bruteforcesearch = true;
		if (LKArg.EpipEnforce == 1)
			rangeS[3] = 0;
		else
			rangeS[3] = 1;
		score = TwoIllumTextSepCoarse(Fimgs, SoureTexture, IPatch, startP, startT, direction, ILWp, TWp, Pcom, LKArg.hsubset, rangeS, puviuv, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage);
		ssig = SSIG[(int)puviuv[4] + ((int)puviuv[5])*width];//source SSIG
		if (ssig < LKArg.ssigThresh || score < LKArg.ZNCCThreshold - 0.35)
			return 0;
		dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3], dPts[2].x = puviuv[4], dPts[2].y = puviuv[5];
	}
	else
	{
		score = LKArg.ZNCCThreshold;
		dPts[0].x = startP[0].x, dPts[0].y = startP[0].y, dPts[1].x = startP[1].x, dPts[1].y = startP[1].y, dPts[2].x = startT.x, dPts[2].y = startT.y;
	}

	bool TransSearch = false;
	if (!(rangeS[2] == 1 && rangeS[0] == 1 && rangeS[1] == 1) && score > LKArg.ZNCCThreshold - 0.35)
	{
		TransSearch = true;
		score = TwoIllumTextAllTransSep(Fimgs, ParaSourceTexture, IPatch, direction, dPts, LKArg, ILWp, TWp, Pcom, ZNCCStorage);
		if (score < LKArg.ZNCCThreshold - 0.06)
			return 0;
	}

	score = TwoIllumTextAffineSep(Fimgs, ParaSourceTexture, IPatch, direction, dPts, LKArg, ILWp, TWp, Pcom, ZNCCStorage);
	if (score < LKArg.ZNCCThreshold - 0.04 && rangeS[2] <= 2 && (rangeS[0] <= 2 || rangeS[1] <= 2))
	{
		//printf("LLT Score: %.3f\n", score);
		if (rangeS[2] == 1 && rangeS[0] == 1 && rangeS[1] == 1 && !TransSearch)
		{
			score = TwoIllumTextAllTransSep(Fimgs, ParaSourceTexture, IPatch, direction, dPts, LKArg, ILWp, TWp, Pcom, ZNCCStorage);
			if (score < LKArg.ZNCCThreshold - 0.06)
				return 0;
		}
		else if (!bruteforcesearch)
		{
			if (LKArg.EpipEnforce == 1)
				rangeS[3] = 0;
			else
				rangeS[3] = 1;
			score = TwoIllumTextSepCoarse(Fimgs, SoureTexture, IPatch, startP, startT, direction, ILWp, TWp, Pcom, LKArg.hsubset, rangeS, puviuv, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage);
			ssig = SSIG[(int)puviuv[4] + ((int)puviuv[5])*width];//source SSIG
			if (ssig < LKArg.ssigThresh || score < LKArg.ZNCCThreshold - 0.35)
				return 0;
			dPts[0].x = puviuv[0], dPts[0].y = puviuv[1], dPts[1].x = puviuv[2], dPts[1].y = puviuv[3], dPts[2].x = puviuv[4], dPts[2].y = puviuv[5];
		}
		else
			return 0;
		score = TwoIllumTextAffineSep(Fimgs, ParaSourceTexture, IPatch, direction, dPts, LKArg, ILWp, TWp, Pcom, ZNCCStorage);
	}

	if (score > LKArg.ZNCCThreshold - 0.04)
	{
		if (CamProGeoVerify((double)nx, (double)ny, dPts, WC, P1mat, DInfo, width, height, pwidth, pheight, 2) == 0)
		{
			nseedtype = 6;
			ILWarping[nid] = dPts[0].x - nx, ILWarping[nid + length] = dPts[0].y - ny, ILWarping[nid + 6 * length] = dPts[1].x - nx, ILWarping[nid + 7 * length] = dPts[1].y - ny;
			TWarping[nid] = dPts[2].x - nx, TWarping[nid + length] = dPts[2].y - ny;
			for (ll = 0; ll < 4; ll++)
				ILWarping[nid + (ll + 2)*length] = ILWp[ll], ILWarping[nid + (ll + 8)*length] = ILWp[ll + 4], TWarping[nid + (ll + 2)*length] = TWp[ll];
			for (ll = 0; ll < 6; ll++)
				PhotoAdj[nid + ll*length] = Pcom[ll];
		}
	}

	return nseedtype;
}
int IllumSeeded(double &score, int x, int y, int nx, int ny, int seedtype, int mode, double *direction, double *proEpiline, DevicesInfo &DInfo, double *P1mat, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *PrecomSearchR, LKParameters LKArg, double *IPatch, double *TarPatch, double *Pro1Patch, double *Pro2Patch, double *TextPatch, double *SuperImposePatch, double *ZNCCStorage, CPoint2 *HFrom, CPoint2 *HTo, double *ASolver, double *BSolver, int SepMode)
{
	//seedtype = 1..2
	int ProID = seedtype == 1 ? 0 : 1;

	int nseedtype = PureIllumCase(score, x, y, nx, ny, ProID, direction, proEpiline, DInfo, P1mat, Fimgs, ILWarping, PrecomSearchR, LKArg, IPatch, TarPatch, ZNCCStorage);
	if (nseedtype != 0)
		return nseedtype;

	if (SepMode == 1 || SepMode == 4)
	{
		nseedtype = TwoIllumCase(score, x, y, nx, ny, seedtype, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, ILWarping, TWarping, PhotoAdj,
			PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
		if (nseedtype != 0)
			return nseedtype;
	}

	if (SepMode == 2 || SepMode == 4)
	{
		nseedtype = IllumTextCase(score, x, y, nx, ny, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
			PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, TextPatch, SuperImposePatch, ZNCCStorage);
		if (nseedtype != 0)
			return nseedtype;
	}

	return 0;
}
int TwoIllumSeeded(double &score, int x, int y, int nx, int ny, int seedtype, int mode, double *direction, double *proEpiline, DevicesInfo &DInfo, double *P1mat, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *PrecomSearchR, LKParameters LKArg, double *IPatch, double *TarPatch, double *Pro1Patch, double *Pro2Patch, double *TextPatch, double *SuperImposePatch, double *ZNCCStorage, CPoint2 *HFrom, CPoint2 *HTo, double *ASolver, double *BSolver, int SepMode)
{
	//seedtype = 3
	if (SepMode == 1 || SepMode == 4)
	{
		int nseedtype = TwoIllumCase(score, x, y, nx, ny, seedtype, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, ILWarping, TWarping, PhotoAdj,
			PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
		if (nseedtype != 0)
			return nseedtype;
	}

	bool transitionFlag = false;
	int id = x + y*Fimgs.width, length = Fimgs.width*Fimgs.height;
	if (DInfo.nPros == 2 && (x + ILWarping[id] < 40 || x + ILWarping[id] > Fimgs.pwidth - 40 || x + ILWarping[id + 6 * length] < 40 || x + ILWarping[id + 6 * length] > Fimgs.pwidth - 40))
		transitionFlag = true;
	for (int ProID = 0; ProID < DInfo.nPros; ProID++)
	{
		int nseedtype = PureIllumCase(score, x, y, nx, ny, ProID, direction, proEpiline, DInfo, P1mat, Fimgs, ILWarping, PrecomSearchR, LKArg, IPatch, TarPatch, ZNCCStorage);
		if (nseedtype == 0 && transitionFlag)
		{
			LKArg.hsubset -= 2;
			nseedtype = PureIllumCase(score, x, y, nx, ny, ProID, direction, proEpiline, DInfo, P1mat, Fimgs, ILWarping, PrecomSearchR, LKArg, IPatch, TarPatch, ZNCCStorage);
			LKArg.hsubset += 2;
		}
		if (nseedtype != 0)
			return nseedtype;
	}

	if (SepMode == 3 || SepMode == 4)
	{
		int nseedtype = TwoIllumTextCase(score, x, y, nx, ny, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
			PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
		if (nseedtype != 0)
			return nseedtype;
	}
	return 0;
}
int IllumTextSeeded(double &score, int x, int y, int nx, int ny, int seedtype, int mode, double *direction, double *proEpiline, DevicesInfo &DInfo, double *P1mat, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *PrecomSearchR, LKParameters LKArg, double *IPatch, double *TarPatch, double *Pro1Patch, double *Pro2Patch, double *TextPatch, double *SuperImposePatch, double *ZNCCStorage, CPoint2 *HFrom, CPoint2 *HTo, double *ASolver, double *BSolver, int SepMode)
{
	//seedtype = 4..5
	if (SepMode < 5)
	{
		if (SepMode == 2 || SepMode == 4)
		{
			int nseedtype = IllumTextCase(score, x, y, nx, ny, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
				PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, TextPatch, SuperImposePatch, ZNCCStorage);
			if (nseedtype != 0)
				return nseedtype;
		}

		int ProID = seedtype == 4 ? 0 : 1;
		int nseedtype = PureIllumCase(score, x, y, nx, ny, ProID, direction, proEpiline, DInfo, P1mat, Fimgs, ILWarping, PrecomSearchR, LKArg, IPatch, TarPatch, ZNCCStorage);
		if (nseedtype != 0)
			return nseedtype;

		if (SepMode == 3 || SepMode == 4)
		{
			int nseedtype = TwoIllumTextCase(score, x, y, nx, ny, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
				PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
			if (nseedtype != 0)
				return nseedtype;
		}
	}
	else
	{
		int bestType = 0;
		double maxscore = 0.0;

		int nseedtype = IllumTextCase(score, x, y, nx, ny, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
			PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, TextPatch, SuperImposePatch, ZNCCStorage);
		if (nseedtype != 0 && score > maxscore)
		{
			bestType = nseedtype, nseedtype = 0;
			maxscore = score;
		}
		if (bestType != 0 && maxscore > 0.97)
		{
			score = maxscore;
			return bestType;
		}


		int ProID = seedtype == 4 ? 0 : 1;
		nseedtype = PureIllumCase(score, x, y, nx, ny, ProID, direction, proEpiline, DInfo, P1mat, Fimgs, ILWarping, PrecomSearchR, LKArg, IPatch, TarPatch, ZNCCStorage);
		if (nseedtype != 0 && score > maxscore)
		{
			bestType = nseedtype, nseedtype = 0;
			maxscore = score;
		}
		if (bestType != 0 && maxscore > 0.97)
		{
			score = maxscore;
			return bestType;
		}

		nseedtype = TwoIllumTextCase(score, x, y, nx, ny, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
			PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
		if (nseedtype != 0 && score > maxscore)
		{
			bestType = nseedtype, nseedtype = 0;
			maxscore = score;
		}
		if (bestType != 0)
		{
			score = maxscore;
			return bestType;
		}
	}
	return 0;
}
int TwoIllumTextSeeded(double &score, int x, int y, int nx, int ny, int seedtype, int mode, double *direction, double *proEpiline, DevicesInfo &DInfo, double *P1mat, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *PrecomSearchR, LKParameters LKArg, double *IPatch, double *TarPatch, double *Pro1Patch, double *Pro2Patch, double *TextPatch, double *SuperImposePatch, double *ZNCCStorage, CPoint2 *HFrom, CPoint2 *HTo, double *ASolver, double *BSolver, int SepMode)
{
	//seedtype = 6
	if (SepMode < 5)
	{
		if (SepMode == 3 || SepMode == 4)
		{
			int nseedtype = TwoIllumTextCase(score, x, y, nx, ny, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
				PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
			if (nseedtype != 0)
				return nseedtype;
		}

		if (SepMode == 1 || SepMode == 4)
		{
			int nseedtype = TwoIllumCase(score, x, y, nx, ny, seedtype, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, ILWarping, TWarping, PhotoAdj,
				PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
			if (nseedtype != 0)
				return nseedtype;
		}

		if (SepMode == 2 || SepMode == 4)
		{
			for (int type = 4; type < 6; type++) //quite special because L+T is never generated from 2L
			{
				int nseedtype = IllumTextCase(score, x, y, nx, ny, type, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
					PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, TextPatch, SuperImposePatch, ZNCCStorage);
				if (nseedtype != 0)
					return nseedtype;
			}
		}
	}
	else
	{
		int bestType = 0;
		double maxscore = 0.0;
		int nseedtype = TwoIllumTextCase(score, x, y, nx, ny, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
			PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
		if (nseedtype != 0 && score > maxscore)
		{
			bestType = nseedtype, nseedtype = 0;
			maxscore = score;
		}
		if (bestType != 0 && maxscore > 0.97)
		{
			score = maxscore;
			return bestType;
		}

		nseedtype = TwoIllumCase(score, x, y, nx, ny, seedtype, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, ILWarping, TWarping, PhotoAdj,
			PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver);
		if (nseedtype != 0 && score > maxscore)
		{
			bestType = nseedtype, nseedtype = 0;
			maxscore = score;
		}
		if (bestType != 0 && maxscore > 0.97)
		{
			score = maxscore;
			return bestType;
		}

		for (int type = 4; type < 6; type++) //quite special because L+T is never generated from 2L
		{
			nseedtype = IllumTextCase(score, x, y, nx, ny, type, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
				PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, TextPatch, SuperImposePatch, ZNCCStorage);
			if (nseedtype != 0 && score > maxscore)
			{
				bestType = nseedtype, nseedtype = 0;
				maxscore = score;
			}
			if (bestType != 0 && maxscore > 0.97)
			{
				score = maxscore;
				return bestType;
			}
		}

		if (bestType != 0)
		{
			score = maxscore;
			return bestType;
		}
	}

	return 0;
}

int OptimizeTwoIllumText(int x, int y, int offx, int offy, int &cp, int &UV_index_n, int &M, double *Coeff, int *lpUV_xy, int *Tindex, DevicesInfo &DInfo, double *P1mat, double *FCP, LKParameters LKArg, int mode, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, bool *cROI, int*visitedPoints, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *SeedType, int *PrecomSearchR, double *IPatch, double *TarPatch, double *Pro1Patch, double *Pro2Patch, double *TextPatch, double *SuperImposePatch, double *ZNCCStorage, CPoint2 *HFrom, CPoint2 *HTo, double *ASolver, double *BSolver, int SepMode)
{
	int nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, plength = pwidth*pheight, length = width*height;
	double score, denum, ImgPt[3], proEpiline[6], direction[4];

	int success = 0, id = x + y*width, nid = x + offx + (y + offy)*width, seedtype = SeedType[id];
	if (cROI[nid] && visitedPoints[nid] < LKArg.npass)
	{
		//Set up epipolar line and starting point
		ImgPt[0] = x + offx, ImgPt[1] = y + offy, ImgPt[2] = 1;
		for (int ProID = 0; ProID < nPros; ProID++)
		{
			mat_mul(FCP + 9 * ProID, ImgPt, proEpiline + 3 * ProID, 3, 3, 1);
			denum = pow(proEpiline[3 * ProID], 2) + pow(proEpiline[3 * ProID + 1], 2);
			direction[2 * ProID] = -proEpiline[3 * ProID + 1] / sqrt(denum), direction[2 * ProID + 1] = proEpiline[3 * ProID] / sqrt(denum);
		}

		bool flag = false;
		if (seedtype == 1 || seedtype == 2)
		{
			SeedType[nid] = IllumSeeded(score, x, y, x + offx, y + offy, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping, PhotoAdj, previousTWarping,
				PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, SepMode);
			flag = (SeedType[nid] != 0) ? true : false;
		}
		else if (seedtype == 3)
		{
			SeedType[nid] = TwoIllumSeeded(score, x, y, x + offx, y + offy, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping, PhotoAdj, previousTWarping,
				PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, SepMode);
			flag = (SeedType[nid] != 0) ? true : false;
		}
		else if (seedtype == 4 || seedtype == 5)
		{
			SeedType[nid] = IllumTextSeeded(score, x, y, x + offx, y + offy, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping, PhotoAdj, previousTWarping,
				PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, SepMode);
			flag = (SeedType[nid] != 0) ? true : false;
		}
		else if (seedtype == 6)
		{
			SeedType[nid] = TwoIllumTextSeeded(score, x, y, x + offx, y + offy, seedtype, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping, PhotoAdj, previousTWarping,
				PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, SepMode);
			flag = (SeedType[nid] != 0) ? true : false;
		}

		if (flag)
		{
			success = 1, cp++, UV_index_n++, M++;
			cROI[nid] = false, Coeff[M] = 1.0 - score;
			lpUV_xy[2 * UV_index_n] = x + offx, lpUV_xy[2 * UV_index_n + 1] = y + offy;
			Tindex[M] = UV_index_n; DIC_AddtoQueue(Coeff, Tindex, M);
		}
		visitedPoints[nid] += 1;
	}

	return success;
}
int TwoIllumTextSeperation(int frameID, char *PATH, IlluminationFlowImages &Fimgs, double *SoureTexture, double *ParaSourceTexture, float *SSIG, DevicesInfo &DInfo, float *ILWarping, float *TWarping, float *PhotoAdj, float *previousTWarping, int *SeedType, int *PrecomSearchR, LKParameters LKArg, int mode, bool *cROI, int part, int SepMode)
{
	//Assume there are one camera and two projectors
	//SepMode: 1= 2L, 2 = L+T, 3 = 2L+T
	const double intentsityFalloff = 1.0 / 255.0;
	int pwidth = Fimgs.pwidth, pheight = Fimgs.pheight, width = Fimgs.width, height = Fimgs.height, nCams = Fimgs.nCams, nPros = Fimgs.nPros, nchannels = Fimgs.nchannels, nframes = Fimgs.nframes;
	int plength = pwidth*pheight, length = width*height, hsubset = LKArg.hsubset, InterpAlgo = LKArg.InterpAlgo;
	int id, idf, seedtype, seededsucces, ii, jj, tii, tjj, kk, ll, ProID, x, y, rangeS[4], ProAvail[2], EProAvail[2], Projected[2];
	bool flag, illumtext, mixedIllums, triplemixed;

	double score, denum, Fcp[18], ImgPt[3], proEpiline[6], direction[4], ILWp[4 * 2], TWp[4];
	CPoint2  puv, iuv, startP[2], startT, dPts[3];
	CPoint foundP[2], foundT;
	CPoint3 WC;

	int patchS = 2 * (hsubset + 2) + 1, patchLength = patchS*patchS;
	double *IPatch = new double[patchLength*(nchannels + 1)], *TarPatch = new double[patchLength*(nchannels + 1)];
	double *ProXPatch = new double[patchLength*(nchannels + 1)], *Pro1Patch = new double[patchLength*(nchannels + 1)], *Pro2Patch = new double[patchLength*(nchannels + 1)], *TextPatch = new double[patchLength*(nchannels + 1)];
	double *SuperImposePatch = new double[patchLength*(nchannels + 1)], *ZNCCStorage = new double[2 * patchLength*(nchannels + 1)];
	CPoint2 *HFrom = new CPoint2[patchLength], *HTo = new CPoint2[patchLength];
	double *ASolver = new double[patchLength * 3], *BSolver = new double[patchLength];

	double P1mat[12 * 2];
	P1mat[0] = DInfo.K[0], P1mat[1] = DInfo.K[1], P1mat[2] = DInfo.K[2], P1mat[3] = 0.0,
		P1mat[4] = DInfo.K[3], P1mat[5] = DInfo.K[4], P1mat[6] = DInfo.K[5], P1mat[7] = 0.0,
		P1mat[8] = DInfo.K[6], P1mat[9] = DInfo.K[7], P1mat[10] = DInfo.K[8], P1mat[11] = 0.0;
	P1mat[12 + 0] = DInfo.P[0], P1mat[12 + 1] = DInfo.P[1], P1mat[12 + 2] = DInfo.P[2], P1mat[12 + 3] = DInfo.P[3],
		P1mat[12 + 4] = DInfo.P[4], P1mat[12 + 5] = DInfo.P[5], P1mat[12 + 6] = DInfo.P[6], P1mat[12 + 7] = DInfo.P[7],
		P1mat[12 + 8] = DInfo.P[8], P1mat[12 + 9] = DInfo.P[9], P1mat[12 + 10] = DInfo.P[10], P1mat[12 + 11] = DInfo.P[11];

	double ssig;
	int pointsToCompute = 0, pointsComputed = 0, cp, M, UV_index = 0, UV_index_n = 0, iStartP[4];
	int *visitedPoints = new int[length], *Tindex = new int[length], *lpUV_xy = new int[2 * length], *MarkBoundary = new int[length];
	double *Coeff = new double[length];
	bool *TripleMixed = new bool[length];
	for (ii = 0; ii < length; ii++)
		TripleMixed[ii] = false;

	rangeS[3] = (LKArg.EpipEnforce == 1) ? 0 : 1;
	for (jj = 0; jj < height; jj += LKArg.step)
	{
		for (ii = 0; ii < width; ii += LKArg.step)
		{
			id = ii + jj*width;
			visitedPoints[id] = 0;
			if (cROI[id])
			{
				flag = false;
				for (kk = 0; kk < nPros; kk++)
				{
					if (abs(ILWarping[id + kk * 6 * length]) + abs(ILWarping[id + (1 + kk * 6)*length]) > 0.01)
					{
						cROI[id] = true; visitedPoints[id] = 100;
						flag = true; break;
					}
				}
				if (!flag)
					pointsToCompute++;
			}
		}
	}

	char Fname[200];
	int percent = 50, increP = 50;
	double start = omp_get_wtime();
	if (pointsToCompute < 500)
#pragma omp critical
		cout << "Partition #" << part << " terminates because #points (" << pointsComputed << ") too compute is to small." << endl;
	else
	{
#pragma omp critical
		cout << "Partition #" << part << " deals with " << pointsToCompute << " pts." << endl;

		mat_transpose(DInfo.FmatPC, Fcp, 3, 3), mat_transpose(DInfo.FmatPC + 9, Fcp + 9, 3, 3);
		for (jj = 0; jj < height; jj += LKArg.step)
		{
			for (ii = 0; ii < width; ii += LKArg.step)
			{
				cp = 0, M = -1; id = ii + jj*width;
				if (cROI[id] && visitedPoints[id] < LKArg.npass2)
				{
					M = 0; UV_index = UV_index_n;
					lpUV_xy[2 * UV_index] = ii, lpUV_xy[2 * UV_index + 1] = jj;

					//Get the epipolar line on projectors
					ImgPt[0] = ii, ImgPt[1] = jj, ImgPt[2] = 1;
					for (ProID = 0; ProID < nPros; ProID++)
					{
						mat_mul(Fcp + 9 * ProID, ImgPt, proEpiline + 3 * ProID, 3, 3, 1);
						denum = pow(proEpiline[3 * ProID], 2) + pow(proEpiline[1 + 3 * ProID], 2);
						direction[2 * ProID] = -proEpiline[1 + 3 * ProID] / sqrt(denum), direction[1 + 2 * ProID] = proEpiline[3 * ProID] / sqrt(denum);
					}

					score = 0.0;
					for (ll = 0; ll < 4; ll++)
						ILWp[ll] = 0.0, ILWp[ll + 4] = 0.0, TWp[ll] = 0.0;

					//Search in the local region for computed cam-pro correspondence
					illumtext = true, mixedIllums = false, triplemixed = false;
					ProAvail[0] = 0, ProAvail[1] = 0, EProAvail[0] = 0, EProAvail[1] = 0, Projected[0] = 0, Projected[1] = 0, seedtype = SeedType[id];
					if (!IsLocalWarpAvail(ILWarping, ILWp, ii, jj, foundP[0], x, y, rangeS[0], width, height, PrecomSearchR[id]))
					{
						if (!IsLocalWarpAvail(ILWarping + 6 * length, ILWp + 4, ii, jj, foundP[1], x, y, rangeS[1], width, height, PrecomSearchR[id]))
							continue; // Does not find any closed by points. 
						else //get the first projector homography estimation
						{
							ProAvail[1] = 1, EProAvail[1] = 1;
							startP[1].x = x, startP[1].y = y;
							idf = foundP[1].x + foundP[1].y*width;

							rangeS[0] = min(EstimateIllumPatchAffine(ii, jj, 1, 0, P1mat, ILWarping, startP[0], ILWp, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver), PrecomSearchR[ii + jj*width]);
							if (rangeS[0] > 0)
								EProAvail[0] = 1;
						}
					}
					else //get the second projector homography estimation
					{
						ProAvail[0] = 1, EProAvail[0] = 1;
						startP[0].x = x, startP[0].y = y;
						idf = foundP[0].x + foundP[0].y*width;

						if (!IsLocalWarpAvail(ILWarping + 6 * length, ILWp + 4, ii, jj, foundP[1], x, y, rangeS[1], width, height, PrecomSearchR[ii + jj*width]))
						{
							rangeS[1] = min(EstimateIllumPatchAffine(ii, jj, 0, 1, P1mat, ILWarping, startP[1], ILWp + 4, DInfo, hsubset, width, height, pwidth, pheight, HFrom, HTo, ASolver, BSolver), PrecomSearchR[ii + jj*width]);
							if (rangeS[1] > 0)
								EProAvail[1] = 1;
						}
						else
						{
							startP[1].x = x, startP[1].y = y; ProAvail[1] = 1, EProAvail[1] = 1;
							idf = foundP[1].x + foundP[1].y*width;
						}
					}

					ssig = 0, flag = false;
					bool transitionFlag = false;
					if (nPros == 2 && (startP[0].x < 40 || startP[0].x > pwidth - 40 || startP[1].x < 40 || startP[1].x > pwidth - 40))
						transitionFlag = true;

					for (ProID = 0; ProID < nPros && !flag; ProID++) //Step 1+2: try pure illumination
					{
						if (ProAvail[ProID] == 1)
						{
							iStartP[0] = startP[ProID].x, iStartP[1] = startP[ProID].y;
							SeedType[id] = PureIllumCase(score, ii, jj, ii, jj, ProID, direction, proEpiline, DInfo, P1mat, Fimgs, ILWarping, PrecomSearchR, LKArg, IPatch, TarPatch, ZNCCStorage, ILWp, iStartP);

							if (SeedType[id] == 0 && transitionFlag) //special handling at the boundary
							{
								LKArg.hsubset -= 2;
								SeedType[id] = PureIllumCase(score, ii, jj, ii, jj, ProID, direction, proEpiline, DInfo, P1mat, Fimgs, ILWarping, PrecomSearchR, LKArg, IPatch, TarPatch, ZNCCStorage, ILWp, iStartP);
								LKArg.hsubset += 2;
							}

							flag = (SeedType[id] != 0) ? true : false;
						}
					}

					if ((SepMode == 1 || SepMode > 3) && !flag && EProAvail[0] == 1 && EProAvail[1] == 1) //Try 3: 2 Illums
					{
						iStartP[0] = startP[0].x, iStartP[1] = startP[0].y, iStartP[2] = startP[1].x, iStartP[3] = startP[1].y;
						SeedType[id] = TwoIllumCase(score, ii, jj, ii, jj, SeedType[id], direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, ILWarping, TWarping, PhotoAdj, PrecomSearchR, LKArg,
							IPatch, TarPatch, Pro1Patch, Pro2Patch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, ILWp, iStartP, rangeS);
						flag = (SeedType[id] != 0) ? true : false;
					}

					int iStartT[2];
					if (!flag && (SepMode == 2 || SepMode > 3))  //Try 4, 5: Illum+Text or 6: 2Illums + Text
					{
						if (!IsLocalWarpAvail(TWarping, TWp, ii, jj, foundT, tii, tjj, rangeS[2], width, height, PrecomSearchR[ii + jj*width]))
						{
							if (mode == 0)
							{
								tii = ii, tjj = jj, rangeS[2] = LKArg.searchRange;
								for (ll = 0; ll < 4; ll++)
									TWp[ll] = 0.0;
							}
							else if (!IsLocalWarpAvail(previousTWarping, TWp, ii, jj, foundT, tii, tjj, rangeS[2], width, height, PrecomSearchR[ii + jj*width]))
								continue; // Does not find any closed by points. No hope for this step 4+5+6
						}
					}

					if ((SepMode == 2 || SepMode > 3) && !flag)  //Try 4, 5: Illum+Text
					{
						for (ProID = 0; ProID < 2 && !flag; ProID++)
						{
							if (ProAvail[ProID] == 1)
							{
								iStartP[0] = startP[ProID].x, iStartP[1] = startP[ProID].y, iStartT[0] = tii, iStartT[1] = tjj;
								SeedType[id] = IllumTextCase(score, ii, jj, ii, jj, ProID, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping,
									PhotoAdj, previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, TextPatch, SuperImposePatch, ZNCCStorage, ILWp, iStartP, TWp, iStartT, &ProID, rangeS);
								flag = (SeedType[id] != 0) ? true : false;
							}
						}
					}

					if (SepMode >= 3 && !flag) ////Try 6
					{
						iStartP[0] = startP[0].x, iStartP[1] = startP[0].y, iStartP[2] = startP[1].x, iStartP[3] = startP[1].y; iStartT[0] = tii, iStartT[1] = tjj;
						SeedType[id] = TwoIllumTextCase(score, ii, jj, ii, jj, 0, mode, direction, proEpiline, DInfo, P1mat, Fimgs, SoureTexture, ParaSourceTexture, SSIG, ILWarping, TWarping, PhotoAdj,
							previousTWarping, PrecomSearchR, LKArg, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, ILWp, TWp, iStartP, iStartT, rangeS);
						flag = (SeedType[id] != 0) ? true : false;
					}

					if (flag)
					{
						cp++; cROI[id] = false;
						Coeff[M] = 1.0 - score, Tindex[M] = UV_index;
					}
					else
						M--;

					visitedPoints[lpUV_xy[2 * UV_index] + lpUV_xy[2 * UV_index + 1] * width] += 1;
				}

				//Now, PROPAGATE
				seededsucces = 0;
				while (M >= 0)
				{
					UV_index = Tindex[M];
					x = lpUV_xy[2 * UV_index], y = lpUV_xy[2 * UV_index + 1];
					M--;

					seededsucces += OptimizeTwoIllumText(x, y, 0, 1, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, DInfo, P1mat, Fcp, LKArg, mode, Fimgs, SoureTexture, ParaSourceTexture, SSIG, cROI, visitedPoints, ILWarping, TWarping,
						PhotoAdj, previousTWarping, SeedType, PrecomSearchR, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, SepMode);
					seededsucces += OptimizeTwoIllumText(x, y, 0, -1, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, DInfo, P1mat, Fcp, LKArg, mode, Fimgs, SoureTexture, ParaSourceTexture, SSIG, cROI, visitedPoints, ILWarping, TWarping,
						PhotoAdj, previousTWarping, SeedType, PrecomSearchR, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, SepMode);
					seededsucces += OptimizeTwoIllumText(x, y, 1, 0, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, DInfo, P1mat, Fcp, LKArg, mode, Fimgs, SoureTexture, ParaSourceTexture, SSIG, cROI, visitedPoints, ILWarping, TWarping,
						PhotoAdj, previousTWarping, SeedType, PrecomSearchR, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, SepMode);
					seededsucces += OptimizeTwoIllumText(x, y, -1, 0, cp, UV_index_n, M, Coeff, lpUV_xy, Tindex, DInfo, P1mat, Fcp, LKArg, mode, Fimgs, SoureTexture, ParaSourceTexture, SSIG, cROI, visitedPoints, ILWarping, TWarping,
						PhotoAdj, previousTWarping, SeedType, PrecomSearchR, IPatch, TarPatch, Pro1Patch, Pro2Patch, TextPatch, SuperImposePatch, ZNCCStorage, HFrom, HTo, ASolver, BSolver, SepMode);

					if (100 * (UV_index_n + 1) / pointsToCompute >= percent)
					{
						double elapsed = omp_get_wtime() - start;
						cout << "Partition #" << part << " ..." << 100 * (UV_index_n + 1) / pointsToCompute << "% ... #" << pointsComputed + cp << " points.TE: " << setw(2) << elapsed << " TR: " << setw(2) << elapsed / (percent + increP)*(100.0 - percent) << endl;
						percent += increP;

#pragma omp critical
						{
							for (int ProID = 0; ProID < nPros; ProID++)
							{
								for (int ll = 0; ll < 6; ll++)
								{
									sprintf(Fname, "%s/%05d_C1P%dp%d.dat", PATH, frameID, ProID + 1, ll);
									WriteGridBinary(Fname, ILWarping + (6 * ProID + ll)*length, width, height);
								}
							}
							for (int ll = 0; ll < 6 && mode < 2; ll++)
							{
								sprintf(Fname, "%s/%05d_C1TSp%d.dat", PATH, frameID, ll);
								WriteGridBinary(Fname, TWarping + ll*length, width, height);
							}
							for (int ll = 0; ll < 5; ll++)
							{
								sprintf(Fname, "%s/%05d_C1PA_%d.dat", PATH, frameID, ll);
								//WriteGridBinary(Fname, PhotoAdj+ll*length, width, height);
							}
							UpdateIllumTextureImages(PATH, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara, ILWarping, ParaSourceTexture, TWarping);
						}
					}
				}

				if (seededsucces > 0)
					UV_index_n++;
				pointsComputed += cp;

				if (seededsucces > 500)
				{
#pragma omp critical
					UpdateIllumTextureImages(PATH, false, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara, ILWarping, ParaSourceTexture, TWarping);
				}
			}
		}


#pragma omp critical
		{
			for (int ProID = 0; ProID < nPros; ProID++)
			{
				for (int ll = 0; ll < 6; ll++)
				{
					sprintf(Fname, "%s/%05d_C1P%dp%d.dat", PATH, frameID, ProID + 1, ll);
					WriteGridBinary(Fname, ILWarping + (6 * ProID + ll)*length, width, height);
				}
			}
			for (int ll = 0; ll < 6 && mode < 2; ll++)
			{
				sprintf(Fname, "%s/%05d_C1TSp%d.dat", PATH, frameID, ll);
				WriteGridBinary(Fname, TWarping + ll*length, width, height);
			}
			for (int ll = 0; ll < 5; ll++)
			{
				sprintf(Fname, "%s/%05d_C1PA_%d.dat", PATH, frameID, ll);
				//WriteGridBinary(Fname, PhotoAdj+ll*length, width, height);
			}
			sprintf(Fname, "%s/%05d_SeedType.dat", PATH, frameID);
			WriteGridBinary(Fname, SeedType, width, height);
			UpdateIllumTextureImages(PATH, true, frameID, mode, nPros, 0, width, height, pwidth, pheight, nchannels, LKArg.InterpAlgo, Fimgs.PPara, ILWarping, ParaSourceTexture, TWarping);

			double elapsed = omp_get_wtime() - start;
			cout << "Partition #" << part << " finishes ... " << 100 * pointsComputed / pointsToCompute << "% (" << pointsComputed << " pts) in " << omp_get_wtime() - start << "s" << endl;
		}
	}

	delete[]IPatch, delete[]TarPatch, delete[]ProXPatch, delete[]Pro1Patch, delete[]Pro2Patch;
	delete[]SuperImposePatch, delete[]ZNCCStorage;
	delete[]visitedPoints, delete[]MarkBoundary;
	delete[]Tindex, delete[]lpUV_xy, delete[]Coeff;
	delete[]HFrom, delete[]HTo, delete[]ASolver, delete[]BSolver;

	return 0;
}

//flow here is computed and stored in matrix of size [gw*gh]
bool IndexForMirrorBoundary(int II, int JJ, int nwidth, int nheight, bool *ROI, int *specID)
{
	//6 7 8
	//3 4 5
	//0 1 2
	// nwidth, nheight: dimension of the current depth image (could be the 0.5pixels 
	//Boundary: boundary of the current depth image
	int curID = II + nwidth*JJ;
	if (II == 0 && JJ == 0) //LL
	{
		if (ROI[curID + nwidth + 1] == false)//tr
			return false;
		if (ROI[curID + nwidth] == false)//md
			return false;
		if (ROI[curID + 1] == false)//mr
			return false;
		specID[6] = curID + nwidth + 1, specID[7] = curID + nwidth, specID[8] = curID + nwidth + 1;
		specID[3] = curID + 1, specID[4] = curID, specID[5] = curID + 1;
		specID[0] = curID + nwidth + 1, specID[1] = curID + nwidth, specID[2] = curID + nwidth + 1;
	}
	else if (II == nwidth - 1 && JJ == 0) //LR
	{
		if (ROI[curID + nwidth - 1] == false)//tl
			return false;
		if (ROI[curID + nwidth] == false)//md
			return false;
		if (ROI[curID - 1] == false)//ml
			return false;
		specID[6] = curID + nwidth - 1, specID[7] = curID + nwidth, specID[8] = curID + nwidth - 1;
		specID[3] = curID - 1, specID[4] = curID, specID[5] = curID - 1;
		specID[0] = curID + nwidth - 1, specID[1] = curID + nwidth, specID[2] = curID + nwidth - 1;
	}
	else if (II == 0 && JJ == nheight - 1) //UL
	{
		if (ROI[curID - nwidth + 1] == false)//br
			return false;
		if (ROI[curID - nwidth] == false)//mb
			return false;
		if (ROI[curID + 1] == false)//mr
			return false;
		specID[6] = curID - nwidth + 1, specID[7] = curID - nwidth, specID[8] = curID - nwidth + 1;
		specID[3] = curID + 1, specID[4] = curID, specID[5] = curID + 1;
		specID[0] = curID - nwidth + 1, specID[1] = curID - nwidth, specID[2] = curID - nwidth + 1;
	}
	else if (II == nwidth - 1 && JJ == nheight - 1) //UR
	{
		if (ROI[curID - nwidth - 1] == false)//bl
			return false;
		if (ROI[curID - nwidth] == false)//mb
			return false;
		if (ROI[curID - 1] == false)//ml
			return false;
		specID[6] = curID - nwidth - 1, specID[7] = curID - nwidth, specID[8] = curID - nwidth - 1;
		specID[3] = curID - 1, specID[4] = curID, specID[5] = curID - 1;
		specID[0] = curID - nwidth - 1, specID[1] = curID - nwidth, specID[2] = curID - nwidth - 1;
	}
	else if (II == 0) //L
	{
		if (ROI[curID - nwidth + 1] == false)//br
			return false;
		if (ROI[curID + nwidth + 1] == false)//tr
			return false;
		if (ROI[curID - nwidth] == false)//mb
			return false;
		if (ROI[curID + nwidth] == false)//md
			return false;
		if (ROI[curID + 1] == false)//mr
			return false;
		specID[6] = curID + nwidth + 1, specID[7] = curID + nwidth, specID[8] = curID + nwidth + 1;
		specID[3] = curID + 1, specID[4] = curID, specID[5] = curID + 1;
		specID[0] = curID - nwidth + 1, specID[1] = curID - nwidth, specID[2] = curID - nwidth + 1;
	}
	else if (II == nwidth - 1) //R
	{
		if (ROI[curID - nwidth - 1] == false)//bl
			return false;
		if (ROI[curID + nwidth - 1] == false)//tl
			return false;
		if (ROI[curID - nwidth] == false)//mb
			return false;
		if (ROI[curID + nwidth] == false)//md
			return false;
		if (ROI[curID - 1] == false)//ml
			return false;
		specID[6] = curID + nwidth - 1, specID[7] = curID + nwidth, specID[8] = curID + nwidth - 1;
		specID[3] = curID - 1, specID[4] = curID, specID[5] = curID - 1;
		specID[0] = curID - nwidth - 1, specID[1] = curID - nwidth, specID[2] = curID - nwidth - 1;
	}
	else if (JJ == 0) //D
	{
		if (ROI[curID + nwidth - 1] == false)//tl
			return false;
		if (ROI[curID + nwidth + 1] == false)//tr
			return false;
		if (ROI[curID + nwidth] == false)//md
			return false;
		if (ROI[curID - 1] == false)//ml
			return false;
		if (ROI[curID + 1] == false)//mr
			return false;
		specID[6] = curID + nwidth - 1, specID[7] = curID + nwidth, specID[8] = curID + nwidth + 1;
		specID[3] = curID - 1, specID[4] = curID, specID[5] = curID + 1;
		specID[0] = curID + nwidth - 1, specID[1] = curID + nwidth, specID[2] = curID + nwidth + 1;
	}
	else if (JJ == nheight - 1)  //U
	{
		if (ROI[curID - nwidth - 1] == false)//bl
			return false;
		if (ROI[curID - nwidth + 1] == false)//br
			return false;
		if (ROI[curID - nwidth] == false)//mb
			return false;
		if (ROI[curID - 1] == false)//ml
			return false;
		if (ROI[curID + 1] == false)//mr
			return false;
		specID[6] = curID - nwidth - 1, specID[7] = curID - nwidth, specID[8] = curID - nwidth + 1;
		specID[3] = curID - 1, specID[4] = curID, specID[5] = curID + 1;
		specID[0] = curID - nwidth - 1, specID[1] = curID - nwidth, specID[2] = curID - nwidth + 1;
	}
	else
	{
		/*
		if(ROI[ curID-nwidth-1]==false)//bl
		{
		if(ROI[curID+nwidth+1]==true)
		specID[0] = curID+nwidth+1;
		else if(ROI[curID-nwidth+1]==true)
		specID[0] = curID-nwidth+1;
		else if(ROI[curID+nwidth-1]==true)
		specID[0] = curID+nwidth-1;
		else
		{
		cout<<"Error handling boundary @"<<II<<","<<JJ<<endl;
		return false;
		}
		}
		if(ROI[curID+nwidth-1]==false)//tl
		{
		if(ROI[curID+nwidth+1]==true)
		specID[6] = curID+nwidth+1;
		else if(ROI[curID-nwidth+1]==true)
		specID[6] = curID-nwidth+1;
		else if(ROI[curID-nwidth-1]==true)
		specID[6] = curID-nwidth-1;
		else
		{
		cout<<"Error handling boundary @"<<II<<","<<JJ<<endl;
		return false;
		}
		}
		if(ROI[curID-nwidth+1]==false)//br
		{
		if(ROI[curID+nwidth-1]==true)
		specID[2] = curID+nwidth-1;
		else if(ROI[curID-nwidth-1]==true)
		specID[2] = curID-nwidth-1;
		else if(ROI[curID+nwidth+1]==true)
		specID[2] = curID+nwidth+1;
		else
		{
		cout<<"Error handling boundary @"<<II<<","<<JJ<<endl;
		return false;
		}
		}
		if(ROI[curID+nwidth+1]==false)//tr
		{
		if(ROI[curID-nwidth+1]==true)
		specID[8] = curID-nwidth+1;
		else if(ROI[curID-nwidth+1]==true)
		specID[8] = curID-nwidth+1;
		else if(ROI[curID+nwidth-1]==(char)255)
		specID[8] = curID+nwidth-1;
		else
		{
		cout<<"Error handling boundary @"<<II<<","<<JJ<<endl;
		return false;
		}
		}
		if(ROI[curID-nwidth]==false)//mb
		{
		if(ROI[curID+nwidth]==(char)255)
		specID[1] = curID+nwidth;
		else
		{
		cout<<"Error handling boundary @"<<II<<","<<JJ<<endl;
		return false;
		}
		}
		if(ROI[curID+nwidth]==false)//md
		{
		if(ROI[curID-nwidth]==(char)255)
		specID[7] = curID-nwidth;
		else
		{
		cout<<"Error handling boundary @"<<II<<","<<JJ<<endl;
		return false;
		}
		}
		if(ROI[curID-1]==false)//ml
		{
		if(ROI[curID+1]==(char)255)
		specID[3] = curID+1;
		else
		{
		cout<<"Error handling boundary @"<<II<<","<<JJ<<endl;
		return false;
		}
		}
		if(ROI[curID+1]==false)//mr
		{
		if(ROI[curID-1]==(char)255)
		specID[5] = curID-1;
		else
		{
		cout<<"Error handling boundary @"<<II<<","<<JJ<<endl;
		return false;
		}
		}
		*/

		if (ROI[curID - nwidth - 1] == false)//bl
			return false;
		if (ROI[curID + nwidth - 1] == false)//tl
			return false;
		if (ROI[curID - nwidth + 1] == false)//br
			return false;
		if (ROI[curID + nwidth + 1] == false)//tr
			return false;
		if (ROI[curID - nwidth] == false)//mb
			return false;
		if (ROI[curID + nwidth] == false)//md
			return false;
		if (ROI[curID - 1] == false)//ml
			return false;
		if (ROI[curID + 1] == false)//mr
			return false;

		specID[6] = curID + nwidth - 1, specID[7] = curID + nwidth, specID[8] = curID + nwidth + 1;
		specID[3] = curID - 1, specID[4] = curID, specID[5] = curID + 1;
		specID[0] = curID - nwidth - 1, specID[1] = curID - nwidth, specID[2] = curID - nwidth + 1;
	}
	return true;
}
void CleanROIforSRSV(bool *cROI, bool *ROI, int width, int height)
{
	int ii, jj;

	//left bottom
	if (!ROI[0] || !ROI[1] || !ROI[width] || !ROI[width + 1])
		cROI[0] = false;

	//right bottom
	if (!ROI[width - 1] || !ROI[width - 2] || !ROI[2 * width - 1] || !ROI[2 * width - 2])
		cROI[width - 1] = false;

	//top left
	if (!ROI[(height - 1)*width] || !ROI[(height - 1)*width + 1] || !ROI[(height - 2)*width] || !ROI[(height - 2)*width + 1])
		ROI[(height - 1)*width] = false;

	//top right
	if (!ROI[height*width - 1] || !ROI[height*width - 2] || !ROI[(height - 1)*width - 1] || !ROI[(height - 1)*width - 2])
		ROI[height*width - 1] = false;

	//bottom
	for (ii = 1; ii < width - 1; ii++)
		if (!ROI[ii] || !ROI[ii - 1] || !ROI[ii + 1] || !ROI[ii + width] || !ROI[ii + width - 1] || !ROI[ii + width + 1])
			cROI[ii] = false;

	//top
	for (ii = 1; ii < width - 1; ii++)
		if (!ROI[ii + (height - 1)*width] || !ROI[ii - 1 + (height - 1)*width] || !ROI[ii + 1 + (height - 1)*width] || !ROI[ii + (height - 2)*width] || !ROI[ii + (height - 2)*width - 1] || !ROI[ii + (height - 2)*width + 1])
			cROI[ii + (height - 1)*width] = false;

	//left
	for (jj = 1; jj < height - 1; jj++)
		if (!ROI[jj*width] || !ROI[(jj + 1)*width] || !ROI[(jj - 1)*width] || !ROI[jj*width + 1] || !ROI[(jj + 1)*width + 1] || !ROI[(jj - 1)*width + 1])
			cROI[jj*width] = false;

	//right
	for (jj = 1; jj < height - 1; jj++)
		if (!ROI[jj*width + width - 1] || !ROI[(jj + 1)*width + width - 1] || !ROI[(jj - 1)*width + width - 1] || !ROI[jj*width + width - 2] || !ROI[(jj + 1)*width + width - 2] || !ROI[(jj - 1)*width + width - 2])
			cROI[jj*width + width - 1] = false;

	//middle region
	for (jj = 1; jj < height - 1; jj++)
	{
		for (ii = 1; ii < width - 1; ii++)
		{
			if (!ROI[ii + jj*width] || !ROI[ii + 1 + jj*width] || !ROI[ii - 1 + jj*width] ||
				!ROI[ii + (jj + 1)*width] || !ROI[ii + 1 + (jj + 1)*width] || !ROI[ii - 1 + (jj + 1)*width] ||
				!ROI[ii + (jj - 1)*width] || !ROI[ii + 1 + (jj - 1)*width] || !ROI[ii - 1 + (jj - 1)*width])
				cROI[ii + jj*width] = false;
		}
	}

	return;
}
int  computeFlow4SVSR(int pointToCompute, double *flowU, double *flowV, double *rayDirect, double *PrayDirect, IlluminationFlowImages &Fimgs, FlowVect &Paraflow, float *Fp, float *Rp, int *flowScale, double *depth, CPoint2 *IJ, int *LayerVisMask, double *CamProDepth, int offset1, int offset2, bool forward, int nCams, int depthW, int depthH, DevicesInfo &DInfo, LKParameters LKArg, SVSRP srp, double *RefPatch, double *TarPatch, double *ZNCCStorage)
{
	bool FailFlow, PrecomputedFlow = srp.Precomputed;
	double u, v, iWp[4], S1[3], S2[3], denum, ProP[3], EpiLine[3];
	CPoint2 dPts[2];  CPoint foundP;

	int kk, mm, nn, range, nbad = 0;
	int hsubset = LKArg.hsubset;
	int  ImgW = Fimgs.width, ImgH = Fimgs.height, ImgLength = ImgW*ImgH, PImgW = Fimgs.pwidth, PImgH = Fimgs.pheight, Plength = PImgW*PImgH;
	int temporalW = Fimgs.nframes, nchannels = Fimgs.nchannels, ngrid = depthW*depthH, fwidth = ImgW / srp.SRF, fheight = ImgH / srp.SRF;

	for (kk = 0; kk < nCams; kk++)
	{
		if (LayerVisMask[pointToCompute + offset1 + kk*ngrid*temporalW] == 0 && nCams > 1)
		{
			denum = depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 4] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 5];
			flowU[pointToCompute + kk*ngrid] = (depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 1]) / denum;
			flowV[pointToCompute + kk*ngrid] = (depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 2] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 3]) / denum;
			nbad++;
			continue;
		}

		//(u,v) is undistorted coord.
		denum = depth[pointToCompute + offset1] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 4] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 5];
		u = (depth[pointToCompute + offset1] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 1]) / denum;
		v = (depth[pointToCompute + offset1] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 2] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 3]) / denum;

		if (u<5.0*LKArg.hsubset || u>(1.0*ImgW - 5.0*LKArg.hsubset) || v<5.0*LKArg.hsubset || v>(1.0*ImgH - 5.0*LKArg.hsubset))
		{
			denum = depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 4] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 5];
			flowU[pointToCompute + kk*ngrid] = (depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 1]) / denum;
			flowV[pointToCompute + kk*ngrid] = (depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 2] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 3]) / denum;
			nbad++;
			continue;
		}

		//Projector points are undistorted in SVSR driver
		ProP[0] = IJ[pointToCompute].x, ProP[1] = IJ[pointToCompute].y, ProP[2] = 1.0;
		mat_mul(DInfo.FmatPC + kk * 9, ProP, EpiLine, 3, 3, 1);

		FailFlow = false;
		if (forward)
		{
			dPts[0].x = u, dPts[0].y = v;
			LensDistortion_Point(dPts[0], DInfo.K + 9 * (kk + 1), DInfo.distortion + 13 * (kk + 1)); //Go to distorted image

			if (!IsLocalWarpAvail(Fp, iWp, (int)(u + 0.5), (int)(v + 0.5), foundP, mm, nn, range, ImgW, ImgH))
			{
				mm = (int)(u + 0.5), nn = (int)(v + 0.5);
				for (mm = 0; mm < 4; mm++)
					iWp[mm] = 0.0;
			}

			if (PrecomputedFlow)
			{
				//Get_Value_Spline_Float(Paraflow.C12x+kk*ImgLength, fwidth, fheight, u/srp.SRF, v/srp.SRF, S1, -1, LKArg.InterpAlgo); 
				//Get_Value_Spline_Float(Paraflow.C12y+kk*ImgLength, fwidth, fheight, u/srp.SRF, v/srp.SRF, S2, -1, LKArg.InterpAlgo); 
				flowU[pointToCompute + kk*ngrid] = u + S1[0];
				flowV[pointToCompute + kk*ngrid] = v + S2[0];
				if (abs(S1[0]) < 0.001 && abs(S2[0]) < 0.001)
					FailFlow = true;
			}
			else
			{
				dPts[1].x = mm, dPts[1].y = nn;
				LKArg.hsubset = flowScale[(int)u + (int)(v)*ImgW];
				if (LKArg.hsubset < 1)
					FailFlow = true;
				else
				{
					//Search on distorted image
					if (EpipSearchLK(dPts, EpiLine, Fimgs.Img + nchannels*ImgLength*temporalW*kk, Fimgs.Img + nchannels*ImgLength*(temporalW*kk + 1), Fimgs.Para + nchannels*ImgLength*temporalW*kk, Fimgs.Para + nchannels*ImgLength*(temporalW*kk + 1), nchannels, ImgW, ImgH, ImgW, ImgH, LKArg, RefPatch, ZNCCStorage, TarPatch, iWp) < LKArg.ZNCCThreshold)
						FailFlow = true;
					else
					{
						Undo_distortion(dPts[1], DInfo.K + 9 * (kk + 1), DInfo.distortion + 13 * (kk + 1)); //Go back to undistorted point
						flowU[pointToCompute + kk*ngrid] = dPts[1].x, flowV[pointToCompute + kk*ngrid] = dPts[1].y;
					}
				}
			}

			if (FailFlow)
			{
				nbad++;
				denum = depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 4] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 5];
				flowU[pointToCompute + kk*ngrid] = (depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 1]) / denum;
				flowV[pointToCompute + kk*ngrid] = (depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 2] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 3]) / denum;
			}
		}
		else
		{
			dPts[0].x = u, dPts[0].y = v;
			LensDistortion_Point(dPts[0], DInfo.K + 9 * (kk + 1), DInfo.distortion + 13 * (kk + 1)); //Go to distorted image

			if (!IsLocalWarpAvail(Rp, iWp, (int)(u + 0.5), (int)(v + 0.5), foundP, mm, nn, range, ImgW, ImgH))
			{
				mm = (int)(u + 0.5), nn = (int)(v + 0.5);
				for (mm = 0; mm < 4; mm++)
					iWp[mm] = 0.0;
			}

			if (PrecomputedFlow)
			{
				Get_Value_Spline(Paraflow.C21x + kk*ImgLength, fwidth, fheight, u / srp.SRF, v / srp.SRF, S1, -1, LKArg.InterpAlgo);
				Get_Value_Spline(Paraflow.C21y + kk*ImgLength, fwidth, fheight, u / srp.SRF, v / srp.SRF, S2, -1, LKArg.InterpAlgo);
				flowU[pointToCompute + kk*ngrid] = u + S1[0];
				flowV[pointToCompute + kk*ngrid] = v + S2[0];
				if (abs(S1[0]) < 0.001 && abs(S2[0]) < 0.001)
					FailFlow = true;
			}
			else
			{
				dPts[1].x = mm, dPts[1].y = nn;
				LKArg.hsubset = flowScale[(int)u + (int)(v)*ImgW];
				if (LKArg.hsubset < 1)
					FailFlow = true;
				else
				{
					//Search on distorted image
					if (EpipSearchLK(dPts, EpiLine, Fimgs.Img + nchannels*ImgLength*(temporalW*kk + 1), Fimgs.Img + nchannels*ImgLength*temporalW*kk, Fimgs.Para + nchannels*ImgLength*(temporalW*kk + 1), Fimgs.Para + nchannels*ImgLength*temporalW*kk, nchannels, ImgW, ImgH, ImgW, ImgH, LKArg, RefPatch, ZNCCStorage, TarPatch, iWp) < LKArg.ZNCCThreshold)
						FailFlow = true;
					else
					{
						Undo_distortion(dPts[1], DInfo.K + 9 * (kk + 1), DInfo.distortion + 13 * (kk + 1)); //Go back to undistorted point
						flowU[pointToCompute + kk*ngrid] = dPts[1].x;
						flowV[pointToCompute + kk*ngrid] = dPts[1].y;
					}
				}
			}

			if (FailFlow)
			{
				nbad++;
				denum = depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 4] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 5];
				flowU[pointToCompute + kk*ngrid] = (depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 1]) / denum;
				flowV[pointToCompute + kk*ngrid] = (depth[pointToCompute + offset2] * PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 2] + PrayDirect[kk * 6 * ngrid + 6 * pointToCompute + 3]) / denum;
			}
		}
	}
	return nbad;
}
int SVSR_solver(IlluminationFlowImages &Fimgs, FlowVect &Paraflow, float *Fp, float *Rp, int *flowScale, double *depth, CPoint2 *IJ, bool *LayerROI, int *LayerVisMask, double *CamProDepth, int *IndexForConstant, int *LayerCamProMask, int nConstant, int nCams, int depthW, int depthH, int ImgW, int ImgH, int PImgW, int PImgH, DevicesInfo &DInfo, LKParameters LKArg, SVSRP srp, bool forward, int iter, char *PATH)
{
	int ii, jj, kk, ll, offset1, offset2, temporalW = 2, nchannels = Fimgs.nchannels, ImgLength = ImgW*ImgH, Plength = PImgW*PImgH, ngrid = depthW*depthH, fwidth = ImgW / srp.SRF, fheight = ImgH / srp.SRF;
	double Nalpha, alpha = sqrt(srp.alpha), beta = sqrt(srp.beta), gamma = sqrt(srp.gamma), dataScale = srp.dataScale, RegularizedScale = srp.regularizationScale;
	double* flowU = new double[ngrid*nCams];
	double* flowV = new double[ngrid*nCams];
	double *rayDirect = new double[2 * ngrid];
	double *PrayDirect = new double[6 * ngrid*nCams];

	int nthreads = 4, percent = 10, increment = 10;
	int hsubset = LKArg.hsubset, patchLength = (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels;
	double *RefPatch = new double[nthreads*patchLength];
	double *TarPatch = new double[nthreads*patchLength];
	double *ZNCCStorage = new double[nthreads * 2 * patchLength];

	if (forward) //fix 1 optim 2
	{
		offset1 = 0;
		offset2 = ngrid;
	}
	else
	{
		offset1 = ngrid;
		offset2 = 0;
	}

	//Set up PrayDirect: in undistorted coord. as done in SVSR driver
	for (ii = 0; ii < ngrid; ii++)
	{
		rayDirect[2 * ii] = DInfo.iK[0] * IJ[ii].x + DInfo.iK[1] * IJ[ii].y + DInfo.iK[2], rayDirect[2 * ii + 1] = DInfo.iK[4] * IJ[ii].y + DInfo.iK[5];
		for (jj = 0; jj < nCams; jj++)
		{
			PrayDirect[jj * 6 * ngrid + 6 * ii] = DInfo.P[12 * jj] * rayDirect[2 * ii] + DInfo.P[12 * jj + 1] * rayDirect[2 * ii + 1] + DInfo.P[12 * jj + 2], PrayDirect[jj * 6 * ngrid + 6 * ii + 1] = DInfo.P[12 * jj + 3];
			PrayDirect[jj * 6 * ngrid + 6 * ii + 2] = DInfo.P[12 * jj + 4] * rayDirect[2 * ii] + DInfo.P[12 * jj + 5] * rayDirect[2 * ii + 1] + DInfo.P[12 * jj + 6], PrayDirect[jj * 6 * ngrid + 6 * ii + 3] = DInfo.P[12 * jj + 7];
			PrayDirect[jj * 6 * ngrid + 6 * ii + 4] = DInfo.P[12 * jj + 8] * rayDirect[2 * ii] + DInfo.P[12 * jj + 9] * rayDirect[2 * ii + 1] + DInfo.P[12 * jj + 10], PrayDirect[jj * 6 * ngrid + 6 * ii + 5] = DInfo.P[12 * jj + 11];
		}
	}

	cout << "Computing flow" << endl;
	/*if(offset2 == 0)
	{
	sprintf(Fname, "%s/%.2f-%.2f -%.2f-%02d_bef1.ijz", PATH, srp.alpha, srp.beta, srp.gamma, iter);
	//sprintf(Fname, "%s/aft1.ijz", PATH);
	fp = fopen(Fname, "w+");
	}
	else
	{
	sprintf(Fname, "%s/%.2f-%.2f -%.2f-%02d_bef2.ijz", PATH, srp.alpha, srp.beta, srp.gamma, iter);
	//sprintf(Fname, "%s/aft2.ijz", PATH);
	fp = fopen(Fname, "w+");
	}
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	double	rayDirectX = DInfo.iK[0]* IJ[ii+jj*depthW].x+DInfo.iK[1]* IJ[ii+jj*depthW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]* IJ[ii+jj*depthW].y+DInfo.iK[5];
	double x = rayDirectX*depth[ii + jj*depthW+offset2], y = rayDirectY*depth[ii + jj*depthW+offset2], z = depth[ii + jj*depthW+offset2];
	fprintf(fp, "%.9f ", z);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);*/
	double start = omp_get_wtime();
	int nbad[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

	int countX = 0;
	omp_set_num_threads(nthreads);
#pragma omp parallel for
	for (int ii = 0; ii < ngrid; ii++)
	{
		int threadID = omp_get_thread_num();
		if (threadID == 0 && 100 * ii*nthreads / ngrid >= percent)
		{
			double elapsed = omp_get_wtime() - start;
			cout << "%" << 100 * ii*nthreads / ngrid << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed / percent*(100.0 - percent) << endl;
			percent += increment*nthreads;
		}
		if (LayerROI[ii] == false)
		{
			countX++;
			nbad[threadID]++;
			continue;
		}

		nbad[threadID] += computeFlow4SVSR(ii, flowU, flowV, rayDirect, PrayDirect, Fimgs, Paraflow, Fp, Rp, flowScale, depth, IJ, LayerVisMask, CamProDepth, offset1, offset2,
			forward, nCams, depthW, depthH, DInfo, LKArg, srp, RefPatch + threadID*patchLength, TarPatch + threadID*patchLength, ZNCCStorage + threadID * 2 * patchLength);;
	}
	delete[]RefPatch;
	delete[]TarPatch;
	delete[]ZNCCStorage;
	cout << "Finish computing illumination flow with " << nbad[0] + nbad[1] + nbad[2] + nbad[3] << "/" << ngrid << " bad points" << " in " << omp_get_wtime() - start << "s" << endl;
	cout << "Bad points caused by ROI: " << countX << endl;

	ceres::Problem problem;
	int specID[9], CamCount;
	int cleanConstant = 0, cleanConstant2 = 0;
	int *cleanIndexforConstant = new int[nConstant];
	int *cleanIndexforConstant2 = new int[ngrid];

	for (jj = 0; jj < depthH; jj++)
	{
		for (ii = 0; ii < depthW; ii++)
		{
			if (LayerROI[ii + jj*depthW] == false)
				continue;

			for (kk = 0; kk < nConstant; kk++)
			{
				ll = ii + jj*depthW;
				if (IndexForConstant[kk] == ll)
				{
					cleanIndexforConstant[cleanConstant] = ll;
					cleanConstant++;
					break;
				}
			}

			if (LayerCamProMask[ii + jj*depthW] == 1)
			{
				cleanIndexforConstant2[cleanConstant2] = ii + jj*depthW;
				cleanConstant2++;
			}

			CamCount = 0;
			for (kk = 0; kk < nCams; kk++)
				if (LayerVisMask[ii + jj*depthW + offset2 + kk*ngrid*temporalW] > 0)
					CamCount++;

			for (kk = 0; kk < nCams; kk++)
			{
				if (LayerVisMask[ii + jj*depthW + offset2 + kk*ngrid*temporalW] > 0)
				{
					ceres::CostFunction* DataTerm = ReprojectionError::Create(ii, jj, depthW, depthH, flowU + kk*ngrid, flowV + kk*ngrid, PrayDirect + kk * 6 * ngrid);
					problem.AddResidualBlock(DataTerm, new HuberLoss(srp.dataScale), &depth[ii + jj*depthW + offset2]);
				}
			}

			if (!IndexForMirrorBoundary(ii, jj, depthW, depthH, LayerROI, specID))
				continue;

			Nalpha = (CamCount == nCams) ? alpha : alpha*3.0; //adaptively change the regularization weight
			if (ii == 0 && jj == 0) //LL
			{
				ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr1::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(ZRegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[4] + offset2], &depth[specID[5] + offset2], &depth[specID[7] + offset2], &depth[specID[8] + offset2]);
			}
			else if (ii == depthW - 1 && jj == 0) //LR
			{
				ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr2::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(ZRegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[3] + offset2], &depth[specID[4] + offset2], &depth[specID[6] + offset2], &depth[specID[7] + offset2]);
			}
			else if (ii == 0 && jj == depthH - 1) //UL
			{
				ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr3::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(ZRegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[1] + offset2], &depth[specID[2] + offset2], &depth[specID[4] + offset2], &depth[specID[5] + offset2]);
			}
			else if (ii == depthW - 1 && jj == depthH - 1) //UR
			{
				ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr4::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(ZRegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[0] + offset2], &depth[specID[1] + offset2], &depth[specID[3] + offset2], &depth[specID[4] + offset2]);
			}
			else if (ii == depthW - 1) //R
			{
				ceres::CostFunction* RegularizationTerm = ZRegularizationErr5::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(RegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[0] + offset2], &depth[specID[1] + offset2], &depth[specID[3] + offset2], &depth[specID[4] + offset2], &depth[specID[6] + offset2], &depth[specID[7] + offset2]);
			}
			else if (ii == 0) //L
			{
				ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr6::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(ZRegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[1] + offset2], &depth[specID[2] + offset2], &depth[specID[4] + offset2], &depth[specID[5] + offset2], &depth[specID[7] + offset2], &depth[specID[8] + offset2]);
			}
			else if (jj == 0) //D
			{
				ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr7::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(ZRegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[3] + offset2], &depth[specID[4] + offset2], &depth[specID[5] + offset2], &depth[specID[6] + offset2], &depth[specID[7] + offset2], &depth[specID[8] + offset2]);
			}
			else if (jj == depthH - 1)  //U
			{
				ceres::CostFunction* RegularizationTerm = ZRegularizationErr8::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(RegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[0] + offset2], &depth[specID[1] + offset2], &depth[specID[2] + offset2], &depth[specID[3] + offset2], &depth[specID[4] + offset2], &depth[specID[5] + offset2]);
			}
			else
			{
				ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr9::Create(Nalpha, ii, jj);
				problem.AddResidualBlock(ZRegularizationTerm, new HuberLoss(srp.regularizationScale), &depth[specID[0] + offset2], &depth[specID[1] + offset2], &depth[specID[2] + offset2], &depth[specID[3] + offset2], &depth[specID[4] + offset2], &depth[specID[5] + offset2], &depth[specID[6] + offset2], &depth[specID[7] + offset2], &depth[specID[8] + offset2]);
			}
		}
	}

	cout << "Adding soft constraints for checker" << endl;
	double *sparseConst = new double[cleanConstant];
	for (ii = 0; ii < cleanConstant; ii++)
	{
		ceres::CostFunction* ConstraintTerm = SoftConstraint::Create(beta, depth[cleanIndexforConstant[ii] + offset2]);
		problem.AddResidualBlock(ConstraintTerm, NULL, &depth[cleanIndexforConstant[ii] + offset2]);
		sparseConst[ii] = depth[cleanIndexforConstant[ii] + offset2];
	}
	cout << "Adding soft constraints for CamPro" << endl;
	double *denseConst = new double[cleanConstant2];
	for (ii = 0; ii < cleanConstant2; ii++)
	{
		ceres::CostFunction* ConstraintTerm = SoftConstraint::Create(gamma, CamProDepth[cleanIndexforConstant2[ii]]);
		problem.AddResidualBlock(ConstraintTerm, NULL, &depth[cleanIndexforConstant2[ii] + offset2]);
		denseConst[ii] = CamProDepth[cleanIndexforConstant2[ii]];
	}

	ceres::Solver::Options options;
	ceres::Solver::Summary summary;
	options.num_threads = nthreads;
	options.max_num_iterations = 200;

	cout << "Run the CG solver" << endl;
	options.linear_solver_type = ceres::CGNR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solve(options, &problem, &summary);
	cout << summary.FullReport() << endl;

	/*cout<<"Refine with Cholesky solver"<<endl;
	options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;
	ceres::Solve(options, &problem, &summary);
	cout << summary.FullReport() <<endl;
	//cout<<summary.BriefReport()<<endl;*/

	/*if(offset2 == 0)
	{
	sprintf(Fname, "%s/%.2f-%.2f -%.2f-%02d_aft1.ijz", PATH, srp.alpha, srp.beta, srp.gamma, iter);
	//sprintf(Fname, "%s/aft1.ijz", PATH);
	fp = fopen(Fname, "w+");
	}
	else
	{
	sprintf(Fname, "%s/%.2f-%.2f -%.2f-%02d_aft2.ijz", PATH, srp.alpha, srp.beta, srp.gamma, iter);
	//sprintf(Fname, "%s/aft2.ijz", PATH);
	fp = fopen(Fname, "w+");
	}
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	double	rayDirectX = DInfo.iK[0]* IJ[ii+jj*depthW].x+DInfo.iK[1]* IJ[ii+jj*depthW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]* IJ[ii+jj*depthW].y+DInfo.iK[5];
	double x = rayDirectX*depth[ii + jj*depthW+offset2], y = rayDirectY*depth[ii + jj*depthW+offset2], z = depth[ii + jj*depthW+offset2];
	fprintf(fp, "%.9f ", z);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);
	*/
	/*
	int *mask1 = new int[ngrid];
	int *mask2 = new int[ngrid];
	for(ii=0; ii<ngrid; ii++)
	{
	mask1[ii] = 0;
	mask2[ii] = 0;
	}
	for(ii=0; ii<cleanConstant; ii++)
	mask1[cleanIndexforConstant[ii]] = 255;
	for(ii=0; ii<cleanConstant2; ii++)
	mask2[cleanIndexforConstant2[ii]] = 255;

	FILE *fp1 = fopen("C:/temp/mask1.txt", "w+");
	FILE *fp2 = fopen("C:/temp/mask2.txt", "w+");
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	fprintf(fp1, "%d ", mask1[ii+jj*depthW]);
	fprintf(fp2, "%d ", mask2[ii+jj*depthW]);
	}
	fprintf(fp1, "\n"), fprintf(fp2, "\n");
	}
	fclose(fp1), fclose(fp2);
	delete []mask1;
	delete []mask2;
	*/
	delete[]flowU;
	delete[]flowV;
	delete[]rayDirect;
	delete[]PrayDirect;
	delete[]sparseConst;
	delete[]denseConst;

	cout << "Succesfully terminate" << endl;

	return 0;
}
int SVSR_Smother(double *depth, CPoint2 *IJ, bool *LayerROI, int *SmoothingMask, int nCams, int depthW, int depthH, DevicesInfo &DInfo, LKParameters &LKArg, SVSRP srp, double alpha, double gamma)
{
	int ii, jj, ll, ngrid = depthW*depthH;
	alpha = sqrt(alpha), gamma = sqrt(gamma);
	double dataScale = srp.dataScale, RegularizedScale = srp.regularizationScale;

	ceres::Problem problem;
	int specID[9], cleanConstant = 0;
	int *cleanIndexforConstant = new int[ngrid * 2];
	int *mask = new int[ngrid];
	double *idepth = new double[ngrid * 2];

	/*char Fname[200]; FILE *fp;
	sprintf(Fname, "C:/temp/bef.ijz");
	fp = fopen(Fname, "w+");
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	double	rayDirectX = DInfo.iK[0]* IJ[ii+jj*depthW].x+DInfo.iK[1]* IJ[ii+jj*depthW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]* IJ[ii+jj*depthW].y+DInfo.iK[5];
	double x = rayDirectX*depth[ii + jj*depthW], y = rayDirectY*depth[ii + jj*depthW], z = depth[ii + jj*depthW];
	fprintf(fp, "%.6f ", z);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	for (jj = 0; jj < depthH; jj++)
	{
		for (ii = 0; ii < depthW; ii++)
		{
			mask[ii + jj*depthW] = 0;
			for (ll = 0; ll < 2; ll++)
				idepth[ii + jj*depthW + ll*ngrid] = depth[ii + jj*depthW + ll*ngrid];

			if (LayerROI[ii + jj*depthW] == false)
				continue;

			for (ll = 0; ll < 2; ll++)
			{
				if (SmoothingMask[ii + jj*depthW + ll*ngrid] == 255)
				{
					cleanIndexforConstant[cleanConstant] = ii + jj*depthW + ll*ngrid;
					cleanConstant++;
				}
			}

			if (!IndexForMirrorBoundary(ii, jj, depthW, depthH, LayerROI, specID))
				continue;

			mask[ii + jj*depthW] = 255;
			for (ll = 0; ll < 2; ll++)
			{
				if (ii == 0 && jj == 0) //LL
				{
					ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr1::Create(alpha, ii, jj);
					problem.AddResidualBlock(ZRegularizationTerm, NULL, &depth[specID[4] + ll*ngrid], &depth[specID[5] + ll*ngrid], &depth[specID[7] + ll*ngrid], &depth[specID[8] + ll*ngrid]);
				}
				else if (ii == depthW - 1 && jj == 0) //LR
				{
					ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr2::Create(alpha, ii, jj);
					problem.AddResidualBlock(ZRegularizationTerm, NULL, &depth[specID[3] + ll*ngrid], &depth[specID[4] + ll*ngrid], &depth[specID[6] + ll*ngrid], &depth[specID[7] + ll*ngrid]);
				}
				else if (ii == 0 && jj == depthH - 1) //UL
				{
					ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr3::Create(alpha, ii, jj);
					problem.AddResidualBlock(ZRegularizationTerm, NULL, &depth[specID[1] + ll*ngrid], &depth[specID[2] + ll*ngrid], &depth[specID[4] + ll*ngrid], &depth[specID[5] + ll*ngrid]);
				}
				else if (ii == depthW - 1 && jj == depthH - 1) //UR
				{
					ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr4::Create(alpha, ii, jj);
					problem.AddResidualBlock(ZRegularizationTerm, NULL, &depth[specID[0] + ll*ngrid], &depth[specID[1] + ll*ngrid], &depth[specID[3] + ll*ngrid], &depth[specID[4] + ll*ngrid]);
				}
				else if (ii == depthW - 1) //R
				{
					ceres::CostFunction* RegularizationTerm = ZRegularizationErr5::Create(alpha, ii, jj);
					problem.AddResidualBlock(RegularizationTerm, NULL, &depth[specID[0] + ll*ngrid], &depth[specID[1] + ll*ngrid], &depth[specID[3] + ll*ngrid], &depth[specID[4] + ll*ngrid], &depth[specID[6] + ll*ngrid], &depth[specID[7] + ll*ngrid]);
				}
				else if (ii == 0) //L
				{
					ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr6::Create(alpha, ii, jj);
					problem.AddResidualBlock(ZRegularizationTerm, NULL, &depth[specID[1] + ll*ngrid], &depth[specID[2] + ll*ngrid], &depth[specID[4] + ll*ngrid], &depth[specID[5] + ll*ngrid], &depth[specID[7] + ll*ngrid], &depth[specID[8] + ll*ngrid]);
				}
				else if (jj == 0) //D
				{
					ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr7::Create(alpha, ii, jj);
					problem.AddResidualBlock(ZRegularizationTerm, NULL, &depth[specID[3] + ll*ngrid], &depth[specID[4] + ll*ngrid], &depth[specID[5] + ll*ngrid], &depth[specID[6] + ll*ngrid], &depth[specID[7] + ll*ngrid], &depth[specID[8] + ll*ngrid]);
				}
				else if (jj == depthH - 1)  //U
				{
					ceres::CostFunction* RegularizationTerm = ZRegularizationErr8::Create(alpha, ii, jj);
					problem.AddResidualBlock(RegularizationTerm, NULL, &depth[specID[0] + ll*ngrid], &depth[specID[1] + ll*ngrid], &depth[specID[2] + ll*ngrid], &depth[specID[3] + ll*ngrid], &depth[specID[4] + ll*ngrid], &depth[specID[5] + ll*ngrid]);
				}
				else
				{
					ceres::CostFunction* ZRegularizationTerm = ZRegularizationErr9::Create(alpha, ii, jj);
					problem.AddResidualBlock(ZRegularizationTerm, NULL, &depth[specID[0] + ll*ngrid], &depth[specID[1] + ll*ngrid], &depth[specID[2] + ll*ngrid], &depth[specID[3] + ll*ngrid], &depth[specID[4] + ll*ngrid], &depth[specID[5] + ll*ngrid], &depth[specID[6] + ll*ngrid], &depth[specID[7] + ll*ngrid], &depth[specID[8] + ll*ngrid]);
				}
			}
		}
	}

	//fp = fopen("C:/temp/const.txt", "w+");
	for (ii = 0; ii < cleanConstant; ii++)
	{
		ceres::CostFunction* ConstraintTerm = SoftConstraint::Create(gamma, depth[cleanIndexforConstant[ii]]);
		problem.AddResidualBlock(ConstraintTerm, NULL, &depth[cleanIndexforConstant[ii]]);
		//if(cleanIndexforConstant[ii] > ngrid)
		//	fprintf(fp, "%d %d %.4f \n", (cleanIndexforConstant[ii]-ngrid)%depthW, (int)((cleanIndexforConstant[ii]-ngrid)/depthW), depth[cleanIndexforConstant[ii]]);
		//else
		//	fprintf(fp, "%d %d %.4f \n", (cleanIndexforConstant[ii])%depthW, (int)((cleanIndexforConstant[ii])/depthW), depth[cleanIndexforConstant[ii]]);
	}
	//	fclose(fp);

	cout << "Run the solver" << endl;
	ceres::Solver::Options options;
	options.function_tolerance = 1e-3;
	options.parameter_tolerance = 1e-4;
	options.num_threads = 1;
	options.max_num_iterations = 20;
	options.linear_solver_type = ceres::CGNR;// ceres::SPARSE_NORMAL_CHOLESKY;
	options.minimizer_progress_to_stdout = true;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	cout << summary.FullReport() << endl;
	//cout<<summary.BriefReport()<<endl;

	//fp = fopen("C:/temp/mask.txt", "w+");
	for (ll = 0; ll < 2; ll++)
		for (jj = 0; jj < depthH; jj++)
		{
			for (ii = 0; ii < depthW; ii++)
			{
				if (mask[ii + jj*depthW] == 0)
					depth[ii + jj*depthW + ll*ngrid] = idepth[ii + jj*depthW + ll*ngrid];
				//fprintf(fp, "%d ", mask[ii+jj*depthW]);
			}
			//fprintf(fp, "\n");
		}
	//fclose(fp);

	/*sprintf(Fname, "C:/temp/aft.ijz");
	fp = fopen(Fname, "w+");
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	double	rayDirectX = DInfo.iK[0]* IJ[ii+jj*depthW].x+DInfo.iK[1]* IJ[ii+jj*depthW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]* IJ[ii+jj*depthW].y+DInfo.iK[5];
	double x = rayDirectX*depth[ii + jj*depthW], y = rayDirectY*depth[ii + jj*depthW], z = depth[ii + jj*depthW];
	fprintf(fp, "%.6f ", z);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);*/

	delete[]mask;
	delete[]idepth;

	cout << "Succesfully terminate" << endl;

	return 0;
}
int SVSR(IlluminationFlowImages &Fimgs, FlowVect &Paraflow, float *Fp, float *Rp, int *flowhsubset, double *depth, bool *TextureMask, CPoint2 *IJ, int *ProCamMask, double *ProCamDepth, CPoint2 *Pcorners, CPoint2 *Ccorners, int *CPindex, int *nPpts, int *triangleList, int *ntriangles, DevicesInfo &DInfo, SVSRP &srp, LKParameters LKArg, double *ProCamScale, CPoint2 *DROI, int ImgW, int ImgH, int PimgW, int PimgH, char *DataPath, int frameID)
{
	const int nCPpts = 3871, maxTriangles = 10000;
	int ii, jj, kk, ll, mm, nn, clength = Fimgs.width*Fimgs.height, plength = PimgW*PimgH, nchannels = Fimgs.nchannels, nCams = Fimgs.nCams, TemporalW = 2;
	char Fname[200];

	//Set up the triangulation
	CPoint2 *apts = new CPoint2[nCams + 1];
	CPoint3 WC, WC2, WC3;
	double *aPmat = new double[12 * (nCams + 1)];
	aPmat[0] = DInfo.K[0], aPmat[1] = DInfo.K[1], aPmat[2] = DInfo.K[2], aPmat[3] = 0.0;
	aPmat[4] = DInfo.K[3], aPmat[5] = DInfo.K[4], aPmat[6] = DInfo.K[5], aPmat[7] = 0.0;
	aPmat[8] = DInfo.K[6], aPmat[9] = DInfo.K[7], aPmat[10] = DInfo.K[8], aPmat[11] = 0.0;


	/// Compute rough estimate of depth via sparse corners
	double *sDepth = new double[TemporalW*nCPpts*nCams];
	for (kk = 0; kk < nCams; kk++)
	{
		for (ii = 0; ii < 12; ii++)
			aPmat[12 + ii] = DInfo.P[12 * kk + ii];

		for (ll = 0; ll < TemporalW; ll++)
		{
			//sprintf(Fname, "C:/temp/sd_C%d_%d.txt", kk+1, ll+1);
			//FILE *fp = fopen(Fname,"w+");
			for (jj = 0; jj < ntriangles[kk]; jj++)
			{
				for (ii = 0; ii < 3; ii++)
				{
					apts[0].x = Pcorners[triangleList[3 * jj + ii + 3 * maxTriangles*kk] + kk*nCPpts].x, apts[0].y = Pcorners[triangleList[3 * jj + ii + 3 * maxTriangles*kk] + kk*nCPpts].y;
					apts[1].x = Ccorners[triangleList[3 * jj + ii + 3 * maxTriangles*kk] + (TemporalW*kk + ll)*nCPpts].x, apts[1].y = Ccorners[triangleList[3 * jj + ii + 3 * maxTriangles*kk] + (TemporalW*kk + ll)*nCPpts].y;

					Undo_distortion(apts[0], DInfo.K, DInfo.distortion);
					Undo_distortion(apts[1], DInfo.K + 9 * (kk + 1), DInfo.distortion + 13 * (kk + 1));
					NviewTriangulation(apts, aPmat, &WC, 2);

					sDepth[triangleList[3 * jj + ii + 3 * maxTriangles*kk] + (kk*TemporalW + ll)*nCPpts] = WC.z;
					//fprintf(fp, "%.2f %.2f %.2f \n", WC.x, WC.y, WC.z);
				}
			}
			//fclose(fp);
		}
	}

	double AA[4 * 3], BB[4], dCoeff[9];
	CPoint2 triCoor[3];
	double maxX, minX, maxY, minY;
	int depthW = (int)(1.0*(DROI[1].x - DROI[0].x + 1) / srp.Rstep), depthH = (int)(1.0*(DROI[1].y - DROI[0].y + 1) / srp.Rstep), depthLength = depthW*depthH;
	double *Depth = new double[TemporalW*depthLength];
	CPoint3 *Normal = new CPoint3[TemporalW*depthLength*nCams];
	bool *ROI = new bool[depthLength];
	for (ii = 0; ii < depthLength; ii++)
		ROI[ii] = false;

	for (ii = 0; ii < TemporalW*depthLength; ii++)
	{
		Depth[ii] = 0.0;
		Normal[ii].x = 0.0, Normal[ii].y = 0.0, Normal[ii].z = 0.0;
	}

	CPoint3 ThreeD[3];
	double normal[6], vec1[3], vec2[3], Z, rayDirectX, rayDirectY;
	for (mm = 0; mm < nCams; mm++)
	{
		for (kk = 0; kk < ntriangles[mm]; kk++)
		{
			triCoor[0].x = Pcorners[triangleList[3 * kk + 3 * maxTriangles*mm] + mm*nCPpts].x, triCoor[0].y = Pcorners[triangleList[3 * kk + 3 * maxTriangles*mm] + mm*nCPpts].y;
			triCoor[1].x = Pcorners[triangleList[3 * kk + 1 + 3 * maxTriangles*mm] + mm*nCPpts].x, triCoor[1].y = Pcorners[triangleList[3 * kk + 1 + 3 * maxTriangles*mm] + mm*nCPpts].y;
			triCoor[2].x = Pcorners[triangleList[3 * kk + 2 + 3 * maxTriangles*mm] + mm*nCPpts].x, triCoor[2].y = Pcorners[triangleList[3 * kk + 2 + 3 * maxTriangles*mm] + mm*nCPpts].y;

			for (ll = 0; ll < TemporalW; ll++)
			{
				AA[0] = triCoor[0].x, AA[1] = triCoor[0].y, AA[2] = sDepth[triangleList[3 * kk + 3 * maxTriangles*mm] + (mm*TemporalW + ll)*nCPpts];
				AA[3] = triCoor[1].x, AA[4] = triCoor[1].y, AA[5] = sDepth[triangleList[3 * kk + 1 + 3 * maxTriangles*mm] + (mm*TemporalW + ll)*nCPpts];
				AA[6] = triCoor[2].x, AA[7] = triCoor[2].y, AA[8] = sDepth[triangleList[3 * kk + 2 + 3 * maxTriangles*mm] + (mm*TemporalW + ll)*nCPpts];

				BB[0] = 1.0, BB[1] = 1.0, BB[2] = 1.0, BB[3] = 1.0;
				LS_Solution_Double(AA, BB, 3, 3);
				dCoeff[3 * ll] = BB[0], dCoeff[3 * ll + 1] = BB[1], dCoeff[3 * ll + 2] = BB[2];

				//compute normal
				for (nn = 0; nn<3; nn++)
				{
					Z = sDepth[triangleList[3 * kk + nn + 3 * maxTriangles*mm] + (mm*TemporalW + ll)*nCPpts];
					rayDirectX = DInfo.iK[0] * triCoor[nn].x + DInfo.iK[1] * triCoor[nn].y + DInfo.iK[2], rayDirectY = DInfo.iK[4] * triCoor[nn].y + DInfo.iK[5];
					ThreeD[nn].x = rayDirectX*Z, ThreeD[nn].y = rayDirectY*Z, ThreeD[nn].z = Z;
				}

				vec1[0] = ThreeD[0].x - ThreeD[1].x, vec1[1] = ThreeD[0].y - ThreeD[1].y, vec1[2] = ThreeD[0].z - ThreeD[1].z;
				vec2[0] = ThreeD[0].x - ThreeD[2].x, vec2[1] = ThreeD[0].y - ThreeD[2].y, vec2[2] = ThreeD[0].z - ThreeD[2].z;
				cross_product(vec1, vec2, normal + 3 * ll);
				normalize(normal + 3 * ll, 3);
			}

			maxX = (MyFtoI(max(max(triCoor[0].x, triCoor[1].x), triCoor[2].x)) - DROI[0].x) / srp.Rstep;
			minX = (MyFtoI(min(min(triCoor[0].x, triCoor[1].x), triCoor[2].x)) - DROI[0].x) / srp.Rstep;
			maxY = (MyFtoI(max(max(triCoor[0].y, triCoor[1].y), triCoor[2].y)) - DROI[0].y) / srp.Rstep;
			minY = (MyFtoI(min(min(triCoor[0].y, triCoor[1].y), triCoor[2].y)) - DROI[0].y) / srp.Rstep;

			maxX = maxX>depthW - 5 ? maxX : maxX + 3;
			minX = minX < 5 ? minX : minX - 3;
			maxY = maxY > depthH - 5 ? maxY : maxY + 3;
			minY = minY < 5 ? minY : minY - 3;

			for (jj = minY; jj < maxY; jj++)
			{
				for (ii = minX; ii < maxX; ii++)
				{
					double proPointX = 1.0*ii*srp.Rstep + DROI[0].x, proPointY = 1.0*jj*srp.Rstep + DROI[0].y;
					if (in_polygon(proPointX, proPointY, triCoor, 3))
					{
						ROI[ii + jj*depthW] = true;
						for (ll = 0; ll < TemporalW; ll++)
						{
							Depth[ll*depthLength + ii + jj*depthW] = (float)((1.0 - dCoeff[3 * ll] * proPointX - dCoeff[3 * ll + 1] * proPointY) / dCoeff[3 * ll + 2]);
							Normal[mm*depthLength*TemporalW + ll*depthLength + ii + jj*depthW].x = normal[3 * ll];
							Normal[mm*depthLength*TemporalW + ll*depthLength + ii + jj*depthW].y = normal[3 * ll + 1];
							Normal[mm*depthLength*TemporalW + ll*depthLength + ii + jj*depthW].z = normal[3 * ll + 2];
						}
					}
				}
			}
		}
	}

	/*FILE *fp = fopen("C:/temp/ROI.txt", "w+");
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	if(ROI[ii+jj*depthW])
	fprintf(fp, "%d ", 1);
	else
	fprintf(fp, "%d ", 0);
	}
	fprintf(fp, "\n");
	}
	fclose(fp);

	{
	for(ll=0; ll<1; ll++)
	{
	sprintf(Fname, "%s/Results/SVSR/L%d/%.2f-%.2f-%.2f-%d-%d_3D_%05d.xyz", DataPath, 0, srp.alpha, srp.beta, srp.gamma, nchannels, LKArg.DIC_Algo, frameID+ll);
	FILE *fp = fopen(Fname, "w+");
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	if(abs(Depth[ii+jj*depthW+depthLength]) < 1)
	continue;
	double	rayDirectX = DInfo.iK[0]*IJ[ii+jj*depthW].x+DInfo.iK[1]*IJ[ii+jj*depthW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]*IJ[ii+jj*depthW].y+DInfo.iK[5];
	fprintf(fp, "%.3f %.3f %.3f \n", rayDirectX*Depth[depthLength*ll+ii+jj*depthW], rayDirectY*Depth[depthLength*ll+ii+jj*depthW], Depth[depthLength*ll+ii+jj*depthW]);
	}
	}
	fclose(fp);
	}
	}*/

	double *Cxcenter = new double[nCams * 3];
	for (ii = 0; ii < nCams; ii++)
		Cxcenter[3 * ii] = DInfo.RTx1[12 * ii + 3], Cxcenter[3 * ii + 1] = DInfo.RTx1[12 * ii + 7], Cxcenter[3 * ii + 2] = DInfo.RTx1[12 * ii + 11];
	int *AllVisibilityMask = new int[nCams*depthLength*TemporalW];
	for (ii = 0; ii < nCams*depthLength*TemporalW; ii++)
		AllVisibilityMask[ii] = nCams;
	/*//Determine visibility
	for(ll=0; ll<nCams; ll++)
	{
	for(kk=0; kk<TemporalW; kk++)
	{
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	if(abs(Depth[ii + jj*depthW+kk*depthLength]) >1)
	{
	if(abs(Normal[ll*depthLength*TemporalW+kk*depthLength+ii+jj*depthW].x)<1.0e-9 && abs(Normal[ll*depthLength*TemporalW+kk*depthLength+ii+jj*depthW].y)<1.0e-9 &&abs(Normal[ll*depthLength*TemporalW+kk*depthLength+ii+jj*depthW].z)<1.0e-9)
	{
	AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW] = 0;
	continue;
	}
	rayDirectX = DInfo.iK[0]*IJ[ii+jj*depthW].x+DInfo.iK[1]*IJ[ii+jj*depthW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]*IJ[ii+jj*depthW].y+DInfo.iK[5];
	ThreeD[0].x = rayDirectX*Depth[ii + jj*depthW+kk*depthLength];
	ThreeD[0].y = rayDirectY*Depth[ii + jj*depthW+kk*depthLength];
	ThreeD[0].z = Depth[ii + jj*depthW+kk*depthLength];

	vec1[0] = ThreeD[0].x - Cxcenter[3*ll];
	vec1[1] = ThreeD[0].y - Cxcenter[3*ll+1];
	vec1[2] = ThreeD[0].z - Cxcenter[3*ll+2];
	vec2[0] = Normal[ll*depthLength*TemporalW+kk*depthLength+ii+jj*depthW].x;
	vec2[1] = Normal[ll*depthLength*TemporalW+kk*depthLength+ii+jj*depthW].y;
	vec2[2] = Normal[ll*depthLength*TemporalW+kk*depthLength+ii+jj*depthW].z;

	normalize(vec1, 3);
	angle = abs(180.0/3.1415*acos(norm_dot_product(vec1, vec2)));

	if(abs(angle-180.0)<45.0 || abs(angle)<45.0)
	AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW] = 2;
	else if(abs(angle-180.0)<65.0 || abs(angle)<65.0)
	AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW] = 1;
	else
	AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW] = 0;
	}
	else
	AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW] = 0;
	}
	}
	}
	}

	//Clean all 1 with 2 if possible
	for(kk=0; kk<TemporalW; kk++)
	{
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	{
	mm = 0;
	for(ll=0; ll<nCams; ll++)
	mm += AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW];
	if(mm > nCams) //must have at least one with type 2
	{
	for(ll=0; ll<nCams; ll++) //erase one without type 2
	if(AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW]<2)
	AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW] = 0;
	}
	}
	}
	}

	for(ll=0; ll<nCams; ll++)
	{
	for(kk=0; kk<TemporalW; kk++)
	{
	sprintf(Fname, "C:/temp/CVis%d_%d.txt", ll+1, kk+1);
	FILE *fp = fopen(Fname, "w+");
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	fprintf(fp, "%d ", AllVisibilityMask[ii+jj*depthW+kk*depthLength+ll*depthLength*TemporalW]);
	fprintf(fp, "\n");
	}
	fclose(fp);
	}
	}

	for(kk=0; kk<TemporalW; kk++)
	{
	sprintf(Fname, "C:/temp/DepthMask_%d.txt", kk+1);
	FILE *fp = fopen(Fname, "w+");
	for(jj=0; jj<depthH; jj++)
	{
	for(ii=0; ii<depthW; ii++)
	fprintf(fp, "%d ", ProCamMask[ii+jj*depthW+kk*depthLength]);
	fprintf(fp, "\n");
	}
	fclose(fp);
	}*/

	//Agumented intial depth with ProCam depth & cope with texture regions
	for (kk = 0; kk < 2; kk++)
	{
		for (jj = 0; jj < depthH; jj++)
		{
			for (ii = 0; ii < depthW; ii++)
			{
				if (ProCamMask[ii + jj*depthW + kk*depthLength] == 1)
					Depth[kk*depthLength + ii + jj*depthW] = ProCamDepth[ii + jj*depthW + kk*depthLength];

				mm = DROI[0].x / srp.Rstep + ii, nn = DROI[0].y / srp.Rstep + jj;
				if (TextureMask[mm + nn*PimgW * 2 + kk*plength * 4] == false)
					ROI[ii + jj*depthW] = false;
			}
		}
	}

	//Run optimization: assume that the distance between checkerboard is 16
	CPoint2 Puv, Iuv;
	double delta;
	int numLayers = (int)log2(1.0*srp.CheckerDisk / 2.0 / srp.Rstep), nConst;
	int *IndexforConst = new int[nCPpts];
	CPoint2 *LayerIJ = new CPoint2[depthLength];
	double *InitLayerDepth = new double[TemporalW*depthLength];
	double *LayerDepth = new double[TemporalW*depthLength];
	double *LayerDepth_old = new double[TemporalW*depthLength];
	double *depthPara = new double[(depthW + 3)*(depthH + 3)];
	bool *LayerROI = new bool[depthLength];
	int *LayerVisMask = new int[TemporalW*depthLength*nCams];
	int *LayerProCamMask = new int[TemporalW*depthLength];
	int *SmoothingMask = new int[TemporalW*depthLength];
	double *LayerProCamDepth = new double[TemporalW*depthLength];

	double start = omp_get_wtime();
	int startLayer = (abs(srp.gamma) < 0.1) ? numLayers : (srp.Rstep < 1.0) ? numLayers - 1 : numLayers - 2;
	for (kk = startLayer; kk >= 0; kk--)
	{
		int gstep = pow(2, kk), gW = (int)(1.0*(depthW - 1) / gstep) + 1, gH = (int)(1.0*(depthH - 1) / gstep) + 1, ngrid = gH*gW;
		for (jj = 0; jj < gH; jj++)
		{
			for (ii = 0; ii < gW; ii++)
			{
				for (ll = 0; ll < TemporalW; ll++)
				{
					InitLayerDepth[ngrid*ll + ii + jj*gW] = Depth[depthLength*ll + ii*gstep + jj*gstep*depthW];
					LayerDepth[ngrid*ll + ii + jj*gW] = Depth[depthLength*ll + ii*gstep + jj*gstep*depthW];
					LayerProCamMask[ngrid*ll + ii + jj*gW] = ProCamMask[depthLength*ll + ii*gstep + jj*gstep*depthW];
					LayerProCamDepth[ngrid*ll + ii + jj*gW] = ProCamDepth[depthLength*ll + ii*gstep + jj*gstep*depthW];
					for (mm = 0; mm < nCams; mm++)
						LayerVisMask[mm*ngrid*TemporalW + ngrid*ll + ii + jj*gW] = AllVisibilityMask[mm*depthLength*TemporalW + depthLength*ll + ii*gstep + jj*gstep*depthW];
				}

				LayerIJ[ii + jj*gW].x = IJ[ii*gstep + jj*gstep*depthW].x;
				LayerIJ[ii + jj*gW].y = IJ[ii*gstep + jj*gstep*depthW].y;

				if (ROI[ii*gstep + jj*gstep*depthW])
					LayerROI[ii + jj*gW] = true;
				else
					LayerROI[ii + jj*gW] = false;
			}
		}

		//Setup fixed parameters
		nConst = 0;
		for (jj = 0; jj < gH; jj += srp.CheckerDisk / gstep)
		{
			for (ii = 0; ii < gW; ii += srp.CheckerDisk / gstep)
			{
				bool flag = false;
				for (ll = 0; ll < nCPpts; ll++)
				{
					if (abs(LayerIJ[ii + jj*gW].x - Pcorners[ll].x) < 0.1 && abs(LayerIJ[ii + jj*gW].y - Pcorners[ll].y) < 0.1)
					{
						flag = true;
						break;
					}
				}

				if (flag)
				{
					IndexforConst[nConst] = ii + jj*gW;
					nConst++;
				}
			}
		}

		//Alternately optimize the depth
		cout << "Process initiated for layer " << kk << " of frame " << frameID << endl;
		for (ll = 0; ll < srp.maxOuterIter; ll++)
		{
			delta = 0.0;
			for (ii = 0; ii < gW*gH; ii++)
				LayerDepth_old[ii] = LayerDepth[ii];

			cout << "Inner loop for layer " << kk << " of frame " << frameID << endl;

			SVSR_solver(Fimgs, Paraflow, Fp, Rp, flowhsubset, LayerDepth, LayerIJ, LayerROI, LayerVisMask, LayerProCamDepth + ngrid, IndexforConst, LayerProCamMask + ngrid, nConst, nCams, gW, gH, ImgW, ImgH, PimgW, PimgH, DInfo, LKArg, srp, true, ll, DataPath);  //fix d1
			SVSR_solver(Fimgs, Paraflow, Fp, Rp, flowhsubset + clength, LayerDepth, LayerIJ, LayerROI, LayerVisMask, LayerProCamDepth, IndexforConst, LayerProCamMask, nConst, nCams, gW, gH, ImgW, ImgH, PimgW, PimgH, DInfo, LKArg, srp, false, ll, DataPath); //fix d2

			for (ii = 0; ii < gW*gH; ii++)
				delta += abs(LayerDepth[ii] - LayerDepth_old[ii]);

			cout << "Diff " << delta / ngrid << endl;
			if (delta / ngrid < srp.thresh)
				break;

		}
		cout << "Process terminated for layer " << kk << " of frame " << frameID << endl;
		if (ll == srp.maxOuterIter)
			cout << "Attention: inner loop did not converge! " << delta / ngrid << endl;

		/*for(ll=0; ll<TemporalW; ll++)
		{
		sprintf(Fname, "%s/Results/SVSR/L%d/R_%.2f-%.2f-%.2f-%d-%d_3D_%05d.ijz", DataPath, kk, srp.alpha, srp.beta, srp.gamma, nchannels, LKArg.DIC_Algo, frameID+ll);
		WriteGridBinary(Fname, LayerDepth+ngrid*ll, gW, gH);
		if(!ReadGridBinary(Fname,  LayerDepth+ngrid*ll, gW, gH))
		return 1;
		sprintf(Fname, "%s/Results/SVSR/L%d/CGR_%.2f-%.2f-%.2f-%d-%d_3D_%05d.xyz", DataPath, kk, srp.alpha, srp.beta, srp.gamma, nchannels, LKArg.DIC_Algo, frameID+ll);
		fp = fopen(Fname, "w+");
		for(jj=0; jj<gH; jj++)
		{
		for(ii=0; ii<gW; ii++)
		{
		if(abs(LayerDepth[ngrid*ll+ii+jj*gW]) < 1)
		continue;
		double	rayDirectX = DInfo.iK[0]*LayerIJ[ii+jj*gW].x+DInfo.iK[1]*LayerIJ[ii+jj*gW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]*LayerIJ[ii+jj*gW].y+DInfo.iK[5];
		fprintf(fp, "%.3f %.3f %.3f \n", rayDirectX*LayerDepth[ngrid*ll+ii+jj*gW], rayDirectY*LayerDepth[ngrid*ll+ii+jj*gW], LayerDepth[ngrid*ll+ii+jj*gW]);
		}
		}
		fclose(fp);
		}*/

		//Spatial verifcation and replace with the very intial results
		/*cout<<"Depth verification ";
		int percent = 10, increment = 10;
		double start = omp_get_wtime();
		for(ll=0; ll<TemporalW; ll++)
		{
		for(jj=0; jj<gH; jj++)
		{
		for(ii=0; ii<gW; ii++)
		{
		if((100*(ii+jj*gW+ll*ngrid)/ngrid/2- percent) > 0)
		{
		double elapsed= omp_get_wtime() - start;
		cout<<"%"<<100*(ii+jj*gW+ll*ngrid)/ngrid/2<<" Time elapsed: "<< setw(2) << elapsed<<" Time remaining: "<< setw(2) <<elapsed/percent*(100.0-percent)<<endl;
		percent+=increment;
		}
		if(abs(LayerDepth[ngrid*ll+ii+jj*gW]) < 1)
		{
		SmoothingMask[ngrid*ll+ii+jj*gW] = 0;
		LayerDepth[ngrid*ll+ii+jj*gW] = 0.0;
		continue;
		}

		int count = 0;
		for(mm=0; mm<nCams; mm++)
		{
		if(LayerVisMask[ii+jj*gW+ngrid*ll+mm*ngrid*TemporalW] >0)
		{
		rayDirectX = DInfo.iK[0]*LayerIJ[ii+jj*gW].x+DInfo.iK[1]*LayerIJ[ii+jj*gW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]*LayerIJ[ii+jj*gW].y+DInfo.iK[5];
		X = rayDirectX*LayerDepth[ngrid*ll+ii+jj*gW], Y = rayDirectY*LayerDepth[ngrid*ll+ii+jj*gW], Z = LayerDepth[ngrid*ll+ii+jj*gW];
		numx = DInfo.P[12*mm]*X + DInfo.P[12*mm+1]*Y + DInfo.P[12*mm+2]*Z +DInfo.P[12*mm+3];
		numy = DInfo.P[12*mm+4]*X + DInfo.P[12*mm+5]*Y + DInfo.P[12*mm+6]*Z +DInfo.P[12*mm+7];
		denum = DInfo.P[12*mm+8]*X + DInfo.P[12*mm+9]*Y + DInfo.P[12*mm+10]*Z +DInfo.P[12*mm+11];
		Iuv.x = numx/denum, Iuv.y = numy/denum;
		Puv.x = LayerIJ[ii+jj*gW].x, Puv.y = LayerIJ[ii+jj*gW].y;

		//LensDistortion_Point(Iuv, DInfo.K+9*(mm+1), DInfo.distortion+13*(mm+1));

		double fufv[2], UV_Guess[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		UV_Guess[0] = ProCamScale[mm], UV_Guess[4] = ProCamScale[mm], UV_Guess[2] = Iuv.x, UV_Guess[5] = Iuv.y;

		if(TMatching(Fimgs.PPara, Fimgs.Para+(ll+TemporalW*mm)*clength*nchannels, 5, PimgW, PimgH, ImgW, ImgH, nchannels, Puv, Iuv, 1,  LKArg.Convergence_Criteria, LKArg.ZNCCThreshold-0.1, LKArg.IterMax, LKArg.InterpAlgo, fufv, false, UV_Guess)<LKArg.ZNCCThreshold-0.1)
		count++;
		else
		{
		SmoothingMask[ngrid*ll+ii+jj*gW] =255;
		break;
		}
		}
		else
		count++;
		}
		if(count==nCams)
		{
		SmoothingMask[ngrid*ll+ii+jj*gW] = 0;
		LayerDepth[ngrid*ll+ii+jj*gW] = 0.0;//InitLayerDepth[ngrid*ll+ii+jj*gW]; //set it back to the very intial depth
		}
		}
		}
		}
		cout<<"... finished in "<<omp_get_wtime ()-start<<endl;
		*/
		/*	sprintf(Fname, "%s/Results/SVSR/smask.txt", DataPath);
		FILE *fp = fopen(Fname, "w+");
		for(ll=0; ll<TemporalW; ll++)
		{
		for(jj=0; jj<gH; jj++)
		{
		for(ii=0; ii<gW; ii++)
		fprintf(fp, "%d ", SmoothingMask[ngrid*ll+ii+jj*gW]);
		//	fscanf(fp, "%d ", &SmoothingMask[ngrid*ll+ii+jj*gW]);
		fprintf(fp, "\n");
		}
		}
		fclose(fp);
		for(ll=0; ll<TemporalW; ll++)
		{
		//sprintf(Fname, "%s/Results/SVSR/L%d/DC_%.2f-%.2f-%.2f-%d-%d_3D_%05d.ijz", DataPath, kk, srp.alpha, srp.beta, srp.gamma, nchannels, LKArg.DIC_Algo, frameID+ll);
		//WriteGridBinary(Fname, LayerDepth+ngrid*ll, gW, gH);
		sprintf(Fname, "%s/Results/SVSR/L%d/DC_%.2f-%.2f-%.2f-%d-%d_3D_%05d.xyz", DataPath, kk, srp.alpha, srp.beta, srp.gamma, nchannels, LKArg.DIC_Algo, frameID+ll);
		FILE *fp = fopen(Fname, "w+");
		for(jj=0; jj<gH; jj++)
		{
		for(ii=0; ii<gW; ii++)
		{
		if(abs(LayerDepth[ngrid*ll+ii+jj*gW]) < 1)
		continue;
		double	rayDirectX = DInfo.iK[0]*LayerIJ[ii+jj*gW].x+DInfo.iK[1]*LayerIJ[ii+jj*gW].y+DInfo.iK[2], rayDirectY = DInfo.iK[4]*LayerIJ[ii+jj*gW].y+DInfo.iK[5];
		fprintf(fp, "%.3f %.3f %.3f \n", rayDirectX*LayerDepth[ngrid*ll+ii+jj*gW], rayDirectY*LayerDepth[ngrid*ll+ii+jj*gW], LayerDepth[ngrid*ll+ii+jj*gW]);
		}
		}
		fclose(fp);
		}*/

		//double alpha = 1.0, gamma = 10000.0;
		//cout<<"Running the smoother"<<endl;
		//SVSR_Smother(LayerDepth, LayerIJ, LayerROI, SmoothingMask, nCams, gW, gH, DInfo, LKArg, srp, alpha, gamma);

		for (ll = 0; ll < TemporalW; ll++)
		{
			sprintf(Fname, "%s/Results/SVSR/L%d/%.2f-%.2f-%.2f-%d-%d_3D_%05d.ijz", DataPath, kk, srp.alpha, srp.beta, srp.gamma, nchannels, LKArg.DIC_Algo, frameID + ll);
			if (!WriteGridBinary(Fname, LayerDepth + ngrid*ll, gW, gH))
				cout << "Cannot write " << Fname << endl;

			sprintf(Fname, "%s/Results/SVSR/L%d/%.2f-%.2f-%.2f-%d-%d_3D_%05d.xyz", DataPath, kk, srp.alpha, srp.beta, srp.gamma, nchannels, LKArg.DIC_Algo, frameID + ll);
			FILE *fp = fopen(Fname, "w+");
			for (jj = 0; jj < gH; jj++)
			{
				for (ii = 0; ii < gW; ii++)
				{
					if (abs(LayerDepth[ngrid*ll + ii + jj*gW]) < 1)
						continue;
					double	rayDirectX = DInfo.iK[0] * LayerIJ[ii + jj*gW].x + DInfo.iK[1] * LayerIJ[ii + jj*gW].y + DInfo.iK[2], rayDirectY = DInfo.iK[4] * LayerIJ[ii + jj*gW].y + DInfo.iK[5];
					fprintf(fp, "%.3f %.3f %.3f \n", rayDirectX*LayerDepth[ngrid*ll + ii + jj*gW], rayDirectY*LayerDepth[ngrid*ll + ii + jj*gW], LayerDepth[ngrid*ll + ii + jj*gW]);
				}
			}
			fclose(fp);

			if (kk != 0)
				for (jj = 0; jj < gH; jj++)
					for (ii = 0; ii < gW; ii++)
						Depth[depthLength*ll + ii*gstep + jj*gstep*depthW] = LayerDepth[ngrid*ll + ii + jj*gW];
		}
	}
	cout << "Total time: " << omp_get_wtime() - start << endl;

	delete[]AllVisibilityMask;
	delete[]aPmat;
	delete[]sDepth;
	delete[]Depth;
	delete[]depthPara;
	delete[]InitLayerDepth;
	delete[]LayerDepth;
	delete[]LayerDepth_old;
	delete[]LayerIJ;
	delete[]ROI;
	delete[]LayerROI;
	delete[]LayerVisMask;
	delete[]LayerProCamMask;
	delete[]IndexforConst;

	return 0;
}

int DepthPropagation(IlluminationFlowImages &Fimgs, float *flowX, float *flowY, double *Intdepth, double *PropDepth, DevicesInfo &DInfo, LKParameters LKArg, int ImgW, int ImgH, int PimgW, int PimgH, CPoint2 *DROI)
{
	int ImgLength = ImgW*ImgH, PLength = PimgW*PimgH, nCams = Fimgs.nCams;

	double *aPmat = new double[12 * (nCams + 1)];
	aPmat[0] = DInfo.K[0], aPmat[1] = DInfo.K[1], aPmat[2] = DInfo.K[2], aPmat[3] = 0.0;
	aPmat[4] = DInfo.K[3], aPmat[5] = DInfo.K[4], aPmat[6] = DInfo.K[5], aPmat[7] = 0.0;
	aPmat[8] = DInfo.K[6], aPmat[9] = DInfo.K[7], aPmat[10] = DInfo.K[8], aPmat[11] = 0.0;
	for (int ii = 0; ii<12 * nCams; ii++)
		aPmat[12 + ii] = DInfo.P[ii];

	int maxThreads = omp_get_max_threads()>MAXTHREADS ? MAXTHREADS : omp_get_max_threads();
	omp_set_num_threads(maxThreads);
	double start = omp_get_wtime();
	int percent = maxThreads;

	//Determine valid points so that omp can spread out its threads
	int *mask = new int[PLength], nvalid = 0;
	for (int id = 0; id < PLength; id++)
	{
		int ii = id%PimgW, jj = (int)(id / PimgW);
		if (!(ii<DROI[0].x || jj<DROI[0].y || ii>DROI[1].x || jj>DROI[1].y || abs(Intdepth[id]) < 0.1))
		{
			//Project
			double x = Intdepth[ii + jj*PimgW] * (DInfo.iK[0] * ii + DInfo.iK[1] * jj + DInfo.iK[2]);
			double y = Intdepth[ii + jj*PimgW] * (DInfo.iK[4] * jj + DInfo.iK[5]);
			double z = Intdepth[ii + jj*PimgW];

			bool breakflag = false;
			for (int kk = 0; kk < nCams; kk++)
			{
				double denum = DInfo.P[8] * x + DInfo.P[9] * y + DInfo.P[10] * z + DInfo.P[11];
				double u = (DInfo.P[0] * x + DInfo.P[1] * y + DInfo.P[2] * z + DInfo.P[3]) / denum;
				double v = (DInfo.P[4] * x + DInfo.P[5] * y + DInfo.P[6] * z + DInfo.P[7]) / denum;

				if (u<5.0*LKArg.hsubset || u>(1.0*ImgW - 5.0*LKArg.hsubset) || v<5.0*LKArg.hsubset || v>(1.0*ImgH - 5.0*LKArg.hsubset))
				{
					breakflag = true;
					break;
				}
			}

			if (!breakflag)
			{
				mask[nvalid] = id;
				nvalid++;
			}
		}
	}

	int hsubset = LKArg.hsubset, nchannels = Fimgs.nchannels;
	double *RefPatch = new double[(2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	double *TarPatch = new double[(2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];
	double *ZNCCStorage = new double[2 * (2 * hsubset + 1)*(2 * hsubset + 1)*nchannels];

	for (int ll = 0; ll < nvalid; ll++)
	{
		bool breakflag = false;
		double ProP[3], EpiLine[3];
		CPoint2 dPts[2], apts[10];
		CPoint3 WC;

		int currentWorkerThread = omp_get_thread_num();
		if (currentWorkerThread == 0)
		{
			if ((ll*maxThreads * 100 / nvalid - percent) > 0)
			{
				percent += maxThreads;
				double elapsed = omp_get_wtime() - start;
				cout << "%" << ll*maxThreads * 100 / nvalid << " Time elapsed: " << setw(2) << elapsed << " Time remaining: " << setw(2) << elapsed*(1.0*nvalid / ll / maxThreads - 1.0) << endl;
			}
		}

		int ii = mask[ll] % PimgW, jj = (int)(mask[ll] / PimgW);
		if (ii<DROI[0].x || jj<DROI[0].y || ii>DROI[1].x || jj>DROI[1].y)
		{
			PropDepth[ii + jj*PimgW] = 0.0;
			continue;
		}

		//Project
		double x = Intdepth[ii + jj*PimgW] * (DInfo.iK[0] * ii + DInfo.iK[1] * jj + DInfo.iK[2]);
		double y = Intdepth[ii + jj*PimgW] * (DInfo.iK[4] * jj + DInfo.iK[5]);
		double z = Intdepth[ii + jj*PimgW];

		for (int kk = 0; kk < nCams; kk++)
		{
			double denum = DInfo.P[8] * x + DInfo.P[9] * y + DInfo.P[10] * z + DInfo.P[11];
			double u = (DInfo.P[0] * x + DInfo.P[1] * y + DInfo.P[2] * z + DInfo.P[3]) / denum;
			double v = (DInfo.P[4] * x + DInfo.P[5] * y + DInfo.P[6] * z + DInfo.P[7]) / denum;

			if (u<5.0*LKArg.hsubset || u>(1.0*ImgW - 5.0*LKArg.hsubset) || v<5.0*LKArg.hsubset || v>(1.0*ImgH - 5.0*LKArg.hsubset))
			{
				PropDepth[ii + jj*PimgW] = 0.0;
				breakflag = true;
				break;
			}

			//Compute flow
			ProP[0] = 1.0*ii, ProP[1] = 1.0*jj, ProP[2] = 1.0;
			mat_mul(DInfo.FmatPC, ProP, EpiLine, 3, 3, 1);

			dPts[0].x = u, dPts[0].y = v;
			dPts[1].x = u + flowX[(int)(u + 0.5) + (int)(v + 0.5)*ImgW];
			dPts[1].y = v + flowY[(int)(u + 0.5) + (int)(v + 0.5)*ImgW];
			if (EpipSearchLK(dPts, EpiLine, Fimgs.Img, Fimgs.Img + ImgLength, Fimgs.Para, Fimgs.Para + ImgLength, Fimgs.nchannels, ImgW, ImgH, ImgW, ImgH, LKArg, RefPatch, ZNCCStorage, TarPatch) < LKArg.ZNCCThreshold)
			{
				PropDepth[ii + jj*PimgW] = 0.0;
				breakflag = true;
				break;
			}

			apts[kk + 1].x = dPts[1].x;
			apts[kk + 1].y = dPts[1].y;
		}
		if (breakflag)
			continue;

		//Triangulation
		apts[0].x = 1.0*ii, apts[0].y = 1.0*jj;
		for (int kk = 0; kk < nCams + 1; kk++)
			Undo_distortion(apts[kk], DInfo.K + 9 * kk, DInfo.distortion + 13 * kk);
		NviewTriangulation(apts, aPmat, &WC, nCams + 1);

		PropDepth[ii + jj*PimgW] = WC.z;
	}

	double elapsed = omp_get_wtime() - start;
	cout << "%" << 100 << " Time elapsed: " << setw(2) << elapsed << endl;

	delete[]RefPatch;
	delete[]TarPatch;
	delete[]ZNCCStorage;
	delete[]aPmat;
	delete[]mask;

	return 0;
}
