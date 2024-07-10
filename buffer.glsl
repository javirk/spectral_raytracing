// Ray Tracing - Primitives. Created by Reinder Nijhoff 2019
// @reindernijhoff
//
// https://www.shadertoy.com/view/tl23Rm
//
// I have combined different intersection routines in one shader (similar 
// to "Raymarching - Primitives": https://www.shadertoy.com/view/Xds3zN) and
// added a simple ray tracer to visualize a scene with all primitives.
// I should add the structs from here: https://www.shadertoy.com/view/tsKyWG

#define PATH_LENGTH 12
#iChannel0 "self"
#include "common.glsl"
#define LOWER_BOUND 450
#define UPPER_BOUND 750
#define NUM_WAVELENGTHS 10

struct Ray
{
    vec3 origin;
    vec3 direction;
};

struct Material
{
    int   materialType;
    vec3  albedo;
    float fuzz;
    float refractionIndex;
};

//
// Hash functions by Nimitz:
// https://www.shadertoy.com/view/Xt3cDn
//

uint baseHash(uvec2 p) {
    p = 1103515245U*((p >> 1U)^(p.yx));
    uint h32 = 1103515245U*((p.x)^(p.y>>3U));
    return h32^(h32 >> 16);
}

float hash1(inout float seed) {
    uint n = baseHash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    return float(n)/float(0xffffffffU);
}

vec2 hash2(inout float seed) {
    uint n = baseHash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    uvec2 rz = uvec2(n, n*48271U);
    return vec2(rz.xy & uvec2(0x7fffffffU))/float(0x7fffffff);
}

//
// Ray tracer helper functions
//

float FresnelSchlickRoughness(float cosTheta, float F0, float roughness) {
    return F0 + (max((1. - roughness), F0) - F0) * pow(abs(1. - cosTheta), 5.0);
}

vec3 cosWeightedRandomHemisphereDirection(const vec3 n, inout float seed) {
  	vec2 r = hash2(seed);
	vec3  uu = normalize(cross(n, abs(n.y) > .5 ? vec3(1.,0.,0.) : vec3(0.,1.,0.)));
	vec3  vv = cross(uu, n);
	float ra = sqrt(r.y);
	float rx = ra*cos(6.28318530718*r.x); 
	float ry = ra*sin(6.28318530718*r.x);
	float rz = sqrt(1.-r.y);
	vec3  rr = vec3(rx*uu + ry*vv + rz*n);
    return normalize(rr);
}

vec3 modifyDirectionWithRoughness(const vec3 normal, const vec3 n, const float roughness, inout float seed) {
    vec2 r = hash2(seed);
    
	vec3  uu = normalize(cross(n, abs(n.y) > .5 ? vec3(1.,0.,0.) : vec3(0.,1.,0.)));
	vec3  vv = cross(uu, n);
	
    float a = roughness*roughness;
    
	float rz = sqrt(abs((1.0-r.y) / clamp(1.+(a - 1.)*r.y,.00001,1.)));
	float ra = sqrt(abs(1.-rz*rz));
	float rx = ra*cos(6.28318530718*r.x); 
	float ry = ra*sin(6.28318530718*r.x);
	vec3  rr = vec3(rx*uu + ry*vv + rz*n);
    
    vec3 ret = normalize(rr);
    return dot(ret,normal) > 0. ? ret : n;
}

vec2 randomInUnitDisk(inout float seed) {
    vec2 h = hash2(seed) * vec2(1,6.28318530718);
    float phi = h.y;
    float r = sqrt(h.x);
	return r*vec2(sin(phi),cos(phi));
}

//
// Scene description
//

vec3 rotateY(const in vec3 p, const in float t) {
    float co = cos(t);
    float si = sin(t);
    vec2 xz = mat2(co,si,-si,co)*p.xz;
    return vec3(xz.x, p.y, xz.y);
}

vec3 opU(vec3 d, float iResult, float mat) {
	return (iResult < d.y) ? vec3(d.x, iResult, mat) : d;
}

vec3 worldhit(in Ray ray, in vec2 dist, out vec3 normal) {
    vec3 tmp0, tmp1, d = vec3(dist, 0.);
    // Last argument in opU is material.
    
    d = opU(d, iPlane(ray.origin, ray.direction, d.xy, normal, vec3(0,1,0), 0.), 1.);  // Metal
    d = opU(d, iSphere(ray.origin - vec3( 0,1, 0), ray.direction, d.xy, normal, 1.), 10.);  // Dielectric
    return d;
}

//
// Palette by Íñigo Quílez: 
// https://www.shadertoy.com/view/ll2GD3
//
vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b*cos(6.28318530718*(c*t+d));
}

float checkerBoard(vec2 p) {
   return mod(floor(p.x) + floor(p.y), 2.);
}

vec3 getSkyColor(vec3 rd) {
    vec3 col = mix(vec3(1), vec3(.5, .7, 1), .5 + .5 * rd.y);
    float sun = clamp(dot(normalize(vec3(-.4, .7, -.6)), rd), 0., 1.);
    col += vec3(1, .6, .1)*(pow(sun, 4.) + 10. * pow(sun, 32.));
    return col;
}

#define LAMBERTIAN 0.
#define METAL 1.
#define DIELECTRIC 2.

float gpuIndepentHash(float p) {
    p = fract(p * .1031);
    p *= p + 19.19;
    p *= p + p;
    return fract(p);
}

void getMaterialProperties(in vec3 pos, in float mat, out vec3 albedo, out float type, out float roughness) {
    albedo = pal(mat*.59996323+.5, vec3(.5),vec3(.5),vec3(1),vec3(0,.1,.2));

    if( mat < 1.5 ) {            
        albedo = vec3(.25 + .25*checkerBoard(pos.xz * 5.));
        roughness = .75 * albedo.x - .15;
        type = METAL;
    } else {
        type = floor(gpuIndepentHash(mat+.3) * 3.);
        roughness = (1.-type*.475) * gpuIndepentHash(mat);
    }
}

float reflectivity(float n1_over_n2, float cosTheta, float wavelenght) {
    float r0 = (n1_over_n2 - 1.) / (n1_over_n2 + 1.);
    r0 = r0*r0;
    return r0 + (1. - r0) * pow((1. - cosTheta), 5.);
}

//
// Simple ray tracer
//

float schlick(float cosine, float r0) {
    return r0 + (1. - r0) * pow(abs(1. - cosine), 5.);
}


vec3 render(in Ray ray, inout float seed) {
    vec3 albedo, normal, col = vec3(1.); 
    float roughness, type;
    
    for (int i = 0; i < PATH_LENGTH; ++i) {    
    	vec3 res = worldhit(ray, vec2(.0001, 100), normal);
		if (res.z > 0.) {
			ray.origin += ray.direction * res.y;
       		
            getMaterialProperties(ray.origin, res.z, albedo, type, roughness);
            
            if (type < LAMBERTIAN+.5) { // Added/hacked a reflection term
                float F = FresnelSchlickRoughness(max(0.,-dot(normal, ray.direction)), .04, roughness);
                if (F > hash1(seed)) {
                    ray.direction = modifyDirectionWithRoughness(normal, reflect(ray.direction,normal), roughness, seed);
                } else {
                    col *= albedo;
			        ray.direction = cosWeightedRandomHemisphereDirection(normal, seed);
                }
            } else if (type < METAL+.5) {
                col *= albedo;
                ray.direction = modifyDirectionWithRoughness(normal, reflect(ray.direction, normal), roughness, seed);            
            } else { // DIELECTRIC
                vec3 normalOut, refracted;
                float ni_over_nt, cosine, reflectProb = 1.;
                if (dot(ray.direction, normal) > 0.) {
                    normalOut = -normal;
            		ni_over_nt = 1.4;
                    cosine = dot(ray.direction, normal);
                    cosine = sqrt(1.-(1.4*1.4)-(1.4*1.4)*cosine*cosine);
                } else {
                    normalOut = normal;
                    ni_over_nt = 1./1.4;
                    cosine = -dot(ray.direction, normal);
                }
            
	            // Refract the ray.
	            refracted = refract(normalize(ray.direction), normalOut, ni_over_nt);
    	        
        	    // Handle total internal reflection.
                if(refracted != vec3(0)) {
                	float r0 = (1. - ni_over_nt)/(1. + ni_over_nt);
	        		reflectProb = FresnelSchlickRoughness(cosine, r0*r0, roughness);
                }
                
                ray.direction = hash1(seed) <= reflectProb ? reflect(ray.direction, normal) : refracted;
                ray.direction = modifyDirectionWithRoughness(-normalOut, ray.direction, roughness, seed);            
            }
        } else {
            col *= getSkyColor(ray.direction);
			return col;
        }
    }  
    return vec3(0);
}

mat3 setCamera( in vec3 ro, in vec3 ta, float cr ) {
	vec3 cw = normalize(ta-ro);
	vec3 cp = vec3(sin(cr), cos(cr),0.0);
	vec3 cu = normalize( cross(cw,cp) );
	vec3 cv = (cross(cu,cw));
  return mat3( cu, cv, cw );
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    bool reset = iFrame == 0;
        
    vec4 data = texelFetch(iChannel0, ivec2(0), 0);
    
    vec3 ro = vec3(3., 1., 2.);
    vec3 ta = vec3(0., 1., 0.);
    mat3 ca = setCamera(ro, ta, 0.);    
    vec3 normal;

    float fpd = data.x;
    if(all(equal(ivec2(fragCoord), ivec2(0)))) {
        // Calculate focus plane.
        Ray focus_ray = Ray(ro, normalize(vec3(.5,0,-.5)-ro));
        float nfpd = worldhit(focus_ray, vec2(0, 100), normal).y;
        fragColor = vec4(nfpd, iResolution.xy, iResolution.x);
    } else { 
        vec2 p = (-iResolution.xy + 2.*fragCoord - 1.)/iResolution.y;
        float seed = float(baseHash(floatBitsToUint(p - iTime)))/float(0xffffffffU);

        // AA
        p += 2.*hash2(seed)/iResolution.y;
        vec3 rd = ca * normalize(vec3(p.xy, 1.6));  

        // DOF
        vec3 fp = ro + rd * fpd;
        ro = ro + ca * vec3(randomInUnitDisk(seed), 0.)*.02;
        rd = normalize(fp - ro);

        Ray ray = Ray(ro, rd);

        vec3 col = render(ray, seed);

        if (reset) {
            fragColor = vec4(col, 1);
        } else {
            fragColor = vec4(col, 1) + texelFetch(iChannel0, ivec2(fragCoord), 0);
            //fragColor = texelFetch(iChannel0, ivec2(fragCoord), 0);
        }
    }
    
}