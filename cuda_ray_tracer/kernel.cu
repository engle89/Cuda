#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "cutil_math.h" 

#define M_PI 3.14159265359f  
#define width 512  
#define height 384 
#define samples 4096

struct Ray {
	float3 origin; 
	float3 direction;  
	__device__ Ray(float3 o, float3 d) : origin(o), direction(d) {}
};

enum Material { Diffuse, Specular, Refraction }; 

struct Sphere {

	float radius;            
	float3 position, emission, color; 
	Material material;          

	__device__ float intersect_sphere(const Ray &r) const {

		float3 op = r.origin - position;
		float t, epsilon = 0.0001f;
		float b = dot(op, r.direction);

		float discriminant = b * b - dot(op, op) + radius * radius;
		if (discriminant < 0)
			return 0;
		else
			discriminant = sqrtf(discriminant);

		t = -b - discriminant;
		if (t > epsilon)
			return t;
		else
		{
			t = -b + discriminant;
			if (t > epsilon)
				return t;
			else
				return 0;
		}
	}
};

// Scene
__constant__ Sphere spheres[] = {
{ 1e5f,{ 1e5f + 1.0f, 40.8f, 81.6f },{ 0.0f, 0.0f, 0.0f },{ 0.75f, 0.25f, 0.25f }, Diffuse }, //Left 
{ 1e5f,{ -1e5f + 99.0f, 40.8f, 81.6f },{ 0.0f, 0.0f, 0.0f },{ .25f, .25f, .75f }, Diffuse }, //Rght 
{ 1e5f,{ 50.0f, 40.8f, 1e5f },{ 0.0f, 0.0f, 0.0f },{ .75f, .75f, .75f }, Diffuse }, //Back 
{ 1e5f,{ 50.0f, 40.8f, -1e5f + 600.0f },{ 0.0f, 0.0f, 0.0f },{ 1.00f, 1.00f, 1.00f }, Diffuse }, //Frnt 
{ 1e5f,{ 50.0f, 1e5f, 81.6f },{ 0.0f, 0.0f, 0.0f },{ .75f, .75f, .75f }, Diffuse }, //Botm 
{ 1e5f,{ 50.0f, -1e5f + 81.6f, 81.6f },{ 0.0f, 0.0f, 0.0f },{ .75f, .75f, .75f }, Diffuse }, //Top 
{ 16.5f,{ 27.0f, 16.5f, 47.0f },{ 0.0f, 0.0f, 0.0f },{ 1.0f, 1.0f, 1.0f }, Specular}, // small sphere 1
{ 16.5f,{ 73.0f, 16.5f, 78.0f },{ 0.0f, 0.0f, 0.0f },{ 1.0f, 1.0f, 0.5f }, Diffuse }, // small sphere 2
{ 16.5f,{ 50.0f, 50.0f, 50.0f}, { 0.0f, 0.0f, 0.0f },{ 1.0f, 1.0f, 1.0f }, Refraction }, //small sphere 3
{ 600.0f,{ 50.0f, 681.6f - .77f, 81.6f },{ 2.0f, 1.8f, 1.6f },{ 0.0f, 0.0f, 0.0f }, Diffuse }  // Light
};

__device__ inline bool intersect_scene(const Ray &r, float &t, int &id) {

	float n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;  
	for (int i = int(n); i--;) 
		if ((d = spheres[i].intersect_sphere(r)) && d < t) {  
			t = d;  
			id = i; 
		}
	return t < inf;
}

__device__ static float getrandom(unsigned int *seed0, unsigned int *seed1) {
	*seed0 = 36969 * ((*seed0) & 65535) + ((*seed0) >> 16); 
	*seed1 = 18000 * ((*seed1) & 65535) + ((*seed1) >> 16);

	unsigned int ires = ((*seed0) << 16) + (*seed1);

	union {
		float f;
		unsigned int ui;
	} res;

	res.ui = (ires & 0x007fffff) | 0x40000000;  

	return (res.f - 2.f) / 2.f;
}

__device__ float3 radiance(Ray &r, unsigned int *s1, unsigned int *s2) { 
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); 
	float3 mask = make_float3(1.0f, 1.0f, 1.0f);

	// ray bounce loop no recursionin device 
	for (int bounces = 0; bounces < 4; bounces++) { 

		float t;         
		int id = 0;        

		//miss
		if (!intersect_scene(r, t, id))
			return make_float3(0.0f, 0.0f, 0.0f); 

		const Sphere &obj = spheres[id];  
		float3 x = r.origin + r.direction*t;          
		float3 n = normalize(x - obj.position);    
		float3 nl = dot(n, r.direction) < 0 ? n : n * -1; 
		
		//emissive
		accucolor += mask * obj.emission;

		//diffuse
		if (obj.material == Diffuse)
		{
			float r1 = 2 * M_PI * getrandom(s1, s2);
			float r2 = getrandom(s1, s2);
			float r2s = sqrtf(r2);

			float3 w = nl;
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			float3 d = normalize(u*cos(r1)*r2s + v * sin(r1)*r2s + w * sqrtf(1 - r2));

			r.origin = x + nl * 0.05f; //offset for self intersection
			r.direction = d;

			mask *= obj.color;
			mask *= dot(d, nl);  // weigh light contribution using cosine of angle between incident light and normal
			mask *= 2;          // fudge factor
		}
		//specular
		else if (obj.material == Specular)
		{
			r.origin = x + nl * 0.07f;
			r.direction = r.direction - n * 2 * dot(n, r.direction);

			mask *= obj.color;
			mask *= dot(r.direction, nl);
			mask *= 2;
		}
		//refraction
		else 
		{
			r.origin = x + nl * 0.05f;
			r.direction = r.direction - n * 2 * dot(n, r.direction);

			
			bool into = (dot(n, nl) > 0);
			double nc = 1;
			double nt = 1.5; //IOR for glass is 1.5
			double nnt = into ? nc / nt : nt / nc;
			double ddn = dot(r.direction, nl);
			double cos2t;

			//total internal reflection
			if ((cos2t = 1 - nnt * nnt*(1 - ddn * ddn)) < 0)
			{
				mask *= obj.color;
				mask *= dot(r.direction, nl);
				mask *= 2;
			}
			//otherwise, choose refraction
			else
			{
				r.direction = normalize((r.direction*nnt - n * ((into ? 1 : -1)*(ddn*nnt + sqrt(cos2t)))));
				double a = nt - nc, b = nt + nc, R0 = a * a / (b*b), c = 1 - (into ? -ddn : dot(r.direction, n));
				double Re = R0 + (1 - R0)*c*c*c*c*c;
				double Tr = 1 - Re;
				double P = 0.25 + 0.5*Re;
				double RP = Re / P;
				double TP = Tr / (1 - P);
				mask *= TP;
				mask *= obj.color;
				mask *= dot(r.direction, nl);
				mask *= 2;
			}		
			
		}
	}

	return accucolor;
}

__global__ void render_kernel(float3 *output) {

	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	unsigned int i = (height - y - 1)*width + x; 

	unsigned int s1 = x; 
	unsigned int s2 = y;

	Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); 
	float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); 
	float3 cy = normalize(cross(cx, cam.direction)) * .5135;
	float3 r;      

	r = make_float3(0.0f); 

	for (int s = 0; s < samples; s++) {  

		float3 d = cam.direction + cx * ((.25 + x) / width - .5) + cy * ((.25 + y) / height - .5);

		r = r + radiance(Ray(cam.origin + d * 40, normalize(d)), &s1, &s2)*(1. / samples);
	}       

	output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }

inline int toInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); } 

int main() {

	float3* output_h = new float3[width*height]; 
	float3* output_d;    

	cudaMalloc(&output_d, width * height * sizeof(float3));

	dim3 block(8, 8, 1);
	dim3 grid(width / block.x, height / block.y, 1);

	printf("CUDA initialised.\nStart rendering...\n");

	render_kernel << < grid, block >> >(output_d);

	cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);

	cudaFree(output_d);

	printf("Done!\n");

	FILE *f = fopen("smallptcuda.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	for (int i = 0; i < width*height; i++)  
		fprintf(f, "%d %d %d ", toInt(output_h[i].x),
			toInt(output_h[i].y),
			toInt(output_h[i].z));

	printf("Saved image to 'smallptcuda.ppm'\n");

	delete[] output_h;
	system("PAUSE");
}