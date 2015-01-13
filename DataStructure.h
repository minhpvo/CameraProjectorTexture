#include <cstdlib>

class CPoint
{
public:
	CPoint() : x(0), y(0) {};
	CPoint(int vx, int vy): x(vx), y(vy) {};
	~CPoint(){};

	int x, y;
};

class CPoint2
{
public:
	CPoint2() : x(0.0), y(0.0) {};
	CPoint2(double vx, double vy): x(vx), y(vy) {};
	~CPoint2(){};

	double x, y;
};

class CPoint2L
{
public:
	CPoint2L() : x(0.0), y(0.0) {};
	CPoint2L(long double vx, long double vy): x(vx), y(vy) {};
	~CPoint2L(){};

	long double x, y;
};

class CPoint3
{
public:
	CPoint3() : x(0.0), y(0.0), z(0.0) {};
	CPoint3(double vx, double vy, double vz) : x(vx), y(vy), z(vz) {};
	~CPoint3(){};

	double x, y, z;
};

class CPoint5
{
public:
	CPoint5() : u(0.0), v(0.0), x(0.0), y(0.0), z(0.0) {};
	CPoint5(double vu, double vv, double vx, double vy, double vz) : u(vu), v(vv), x(vx), y(vy), z(vz) {};
	~CPoint5(){};

	double u, v, x, y, z;
};

struct IJUV
{
	float i, j, du, dv;
};


