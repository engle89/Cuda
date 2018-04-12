#ifndef CUDA_STRUCT_H
#define CUDA_STRUCT_H

#include <cuda_runtime.h>

struct curayCam {
	float3 pos;
	float3 dir;
	float  up;
};

struct curayLight {
	float  m_intensity;
	float3 m_diffuse;
	float3 m_specular;
	float3 m_pos;
};

struct curaySphere {
	float  radius;
	float3 m_pos;
	float3 m_color;
	int    matID;
};

struct curayTri {
	float3 p0, p1, p2;
	float3 m_normal;
	float3 m_color;
	float3 ABC, DEF;
	int    matID;
	//DefineABCDEF();
	//DefineNormal();
};

struct curayFog {
	float  m_color;
	float3 m_IntMinMax;
};

struct curayRay {
	float3 m_dir;
	float3 m_pos;
};

struct curayRec {
	float t;
	int   id;
};

struct curayFrameBuffer {
	int    id;
	int    ObjType;
	float  t;
	float3 normal;
	float3 point;
};

struct curayMat {
	float3 color;
	float3 spec;
	float reflectivity;
	float shininess;
};


#endif