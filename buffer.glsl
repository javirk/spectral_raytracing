// Ray Tracing - Primitives. Created by Reinder Nijhoff 2019
// @reindernijhoff
//
// https://www.shadertoy.com/view/tl23Rm
//
// I have combined different intersection routines in one shader (similar 
// to "Raymarching - Primitives": https://www.shadertoy.com/view/Xds3zN) and
// added a simple ray tracer to visualize a scene with all primitives.
//

#define PATH_LENGTH 12
#iChannel0 "self"
#include "common.glsl"
#define LOWER_BOUND 450
#define UPPER_BOUND 750
#define NUM_WAVELENGTHS 12

// Spectrum to xyz approx function from "Simple Analytic Approximations to the CIE XYZ Color Matching Functions"
// http://jcgt.org/published/0002/02/01/paper.pdf and https://www.shadertoy.com/view/WlsXDj
//Inputs:  Wavelength in nanometers
float xFit_1931(float wave)
{
    float t1 = (wave - 442.0f)*((wave < 442.0f) ? 0.0624f : 0.0374f);
    float t2 = (wave - 599.8f)*((wave < 599.8f) ? 0.0264f : 0.0323f);
    float t3 = (wave - 501.1f)*((wave < 501.1f) ? 0.0490f : 0.0382f);
    return 0.362f*exp(-0.5f*t1*t1) + 1.056f*exp(-0.5f*t2*t2) - 0.065f*exp(-0.5f*t3*t3);
}
float yFit_1931(float wave)
{
    float t1 = (wave - 568.8f)*((wave < 568.8f) ? 0.0213f : 0.0247f);
    float t2 = (wave - 530.9f)*((wave < 530.9f) ? 0.0613f : 0.0322f);
    return 0.821f*exp(-0.5f*t1*t1) + 0.286f*exp(-0.5f*t2*t2);
}
float zFit_1931(float wave)
{
    float t1 = (wave - 437.0f)*((wave < 437.0f) ? 0.0845f : 0.0278f);
    float t2 = (wave - 459.0f)*((wave < 459.0f) ? 0.0385f : 0.0725f);

    return 1.217f*exp(-0.5f*t1*t1) + 0.681f*exp(-0.5f*t2*t2);
}

vec3 XYZtosRGB(vec3 XYZ)
{
    vec3 rgb;
    rgb.x = XYZ.x *  3.2409699f + XYZ.y * -1.5373832f + XYZ.z * -0.4986108f;
    rgb.y = XYZ.x * -0.9692436f + XYZ.y *  1.8759675f + XYZ.z *  0.0415551f;
    rgb.z = XYZ.x *  0.0556301f + XYZ.y * -0.2039770f + XYZ.z *  1.0569715f;
    
    return rgb;
}

float d65_illuminant(float wave)
{
    // float term1 = 1.35606;
    float term2 = -0.34302 * math.cos(4.88600 * wave * math.pi / 180);
    float term3 = 0.09163 * math.sin(4.88600 * wave * math.pi / 180);
    float E_D65 = 100 * (1.35606 + term2 + term3);
    return E_D65;
}

//
// Hash functions by Nimitz:
// https://www.shadertoy.com/view/Xt3cDn
//

uint baseHash( uvec2 p ) {
    p = 1103515245U*((p >> 1U)^(p.yx));
    uint h32 = 1103515245U*((p.x)^(p.y>>3U));
    return h32^(h32 >> 16);
}

float hash1( inout float seed ) {
    uint n = baseHash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    return float(n)/float(0xffffffffU);
}

vec2 hash2( inout float seed ) {
    uint n = baseHash(floatBitsToUint(vec2(seed+=.1,seed+=.1)));
    uvec2 rz = uvec2(n, n*48271U);
    return vec2(rz.xy & uvec2(0x7fffffffU))/float(0x7fffffff);
}

//
// Ray tracer helper functions
//

float FresnelSchlickRoughness( float cosTheta, float F0, float roughness ) {
    return F0 + (max((1. - roughness), F0) - F0) * pow(abs(1. - cosTheta), 5.0);
}

vec3 cosWeightedRandomHemisphereDirection( const vec3 n, inout float seed ) {
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

vec3 modifyDirectionWithRoughness( const vec3 normal, const vec3 n, const float roughness, inout float seed ) {
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

vec2 randomInUnitDisk( inout float seed ) {
    vec2 h = hash2(seed) * vec2(1,6.28318530718);
    float phi = h.y;
    float r = sqrt(h.x);
	return r*vec2(sin(phi),cos(phi));
}

//
// Scene description
//

vec3 rotateY( const in vec3 p, const in float t ) {
    float co = cos(t);
    float si = sin(t);
    vec2 xz = mat2(co,si,-si,co)*p.xz;
    return vec3(xz.x, p.y, xz.y);
}

vec3 opU( vec3 d, float iResult, float mat ) {
	return (iResult < d.y) ? vec3(d.x, iResult, mat) : d;
}

vec3 worldhit( in vec3 ro, in vec3 rd, in vec2 dist, out vec3 normal ) {
    vec3 tmp0, tmp1, d = vec3(dist, 0.);
    // Last argument in opU is material.
    d = opU(d, iPlane(ro, rd, d.xy, normal, vec3(0,1,0), 0.), 1.);  // Metal
    d = opU(d, iSphere(ro - vec3( 0,1, 0), rd, d.xy, normal, 1.), 10.);  // Dielectric
    return d;
}

//
// Palette by Íñigo Quílez: 
// https://www.shadertoy.com/view/ll2GD3
//
vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b*cos(6.28318530718*(c*t+d));
}

float checkerBoard( vec2 p ) {
   return mod(floor(p.x) + floor(p.y), 2.);
}

vec3 getSkyColor( vec3 rd ) {
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

void getMaterialProperties(in vec3 pos, in float mat, 
                           out vec3 albedo, out float type, out float roughness) {
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

float reflectivity_old(float n1, float n2, float cosTheta, float wavelenght) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0*r0;
    float exponent = 1. / (1. + 0.05 * (wavelenght - float(LOWER_BOUND)));
    return r0 + (1. - r0) * (1. - pow(cosTheta, exponent));
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

float ray_trace(in vec3 ro, in vec3 rd, in float wavelenght, inout float seed) {
    vec3 albedo, normal = vec3(1.);
    float roughness, type;
    float nt = 1.33;
    float nt_nt = nt * nt;
    vec3 col = vec3(1.);
    float cosine = 1.;
    int i_refracted = 0;
    float ray_intensity = 1.;
    for (int i=0; i < PATH_LENGTH; ++i) {
    	vec3 res = worldhit(ro, rd, vec2(.0001, 100), normal);
		if (res.z > 0.) {
			ro += rd * res.y;
       		
            getMaterialProperties(ro, res.z, albedo, type, roughness);
            
            if (type < LAMBERTIAN+.5) { // Added/hacked a reflection term
                float F = FresnelSchlickRoughness(max(0., -dot(normal, rd)), .04, roughness);
                if (F > hash1(seed)) {
                    rd = modifyDirectionWithRoughness(normal, reflect(rd,normal), roughness, seed);
                } else {
                    col *= albedo;
			        rd = cosWeightedRandomHemisphereDirection(normal, seed);
                }
            } else if (type < METAL+.5) {
                col *= albedo;
                rd = modifyDirectionWithRoughness(normal, reflect(rd,normal), roughness, seed);
                // ray_intensity *= reflectivity(1., dot(rd, normal), wavelenght);   
            } else { // DIELECTRIC
                vec3 normalOut, refracted;
                float ni_over_nt, reflectProb = 1.;
                
                if (dot(rd, normal) > 0.) {
                    // Ray is inside the object.
                    normalOut = -normal;
            		ni_over_nt = nt;
                    cosine = dot(rd, normal);
                    cosine = sqrt(1. - nt_nt - nt_nt * cosine * cosine);
                } else {
                    normalOut = normal;
                    ni_over_nt = 1./nt;
                    cosine = -dot(rd, normal);
                }
            
	            // Refract the ray.
	            refracted = refract(normalize(rd), normalOut, ni_over_nt);
    	        
        	    // Handle total internal reflection.
                if(refracted != vec3(0)) {
                	float r0 = (1. - ni_over_nt)/(1. + ni_over_nt);
	        		reflectProb = FresnelSchlickRoughness(cosine, r0*r0, roughness);
                }
                
                if (hash1(seed) <= reflectProb) {
                    rd = reflect(rd, normal);
                    ray_intensity *= reflectivity(ni_over_nt, cosine, wavelenght);
                } else {
                    rd = refracted;
                    ray_intensity *= 1. - reflectivity(ni_over_nt, dot(rd, normalOut), wavelenght);
                }
                rd = modifyDirectionWithRoughness(-normalOut, rd, roughness, seed);
            }
        } else {
            return ray_intensity;
        }
    }  
    return ray_intensity;
}

float calculate_normalization_factor() {
    float normalization_factor = 0.;
    for (int i_wave=0; i_wave < NUM_WAVELENGTHS; i_wave++) {
        float wave = float(LOWER_BOUND) + float((UPPER_BOUND-LOWER_BOUND) * i_wave) / float(NUM_WAVELENGTHS);
        normalization_factor += yFit_1931(wave);
    }
    return normalization_factor;
}

vec3 render_new( in vec3 ro, in vec3 rd, inout float seed ) {
    float color_x = 0.;
    float color_y = 0.;
    float color_z = 0.;
    float k = 100. / calculate_normalization_factor();
    
    float wave;

    // Loop over wavelengths
    for (int i_wave=0; i_wave < NUM_WAVELENGTHS; i_wave++) {
        wave = float(LOWER_BOUND) + float((UPPER_BOUND-LOWER_BOUND) * i_wave) / float(NUM_WAVELENGTHS);
        //wave = 450.;
        float intensity = ray_trace(ro, rd, wave, seed);
        // intensity = 1.;
        float illuminant = d65_illuminant(wave);

        color_x += intensity * xFit_1931(wave) * illuminant;
        color_y += intensity * yFit_1931(wave) * illuminant;
        color_z += intensity * zFit_1931(wave) * illuminant;
    }
    vec3 xyz = vec3(color_x, color_y, color_z);

    xyz *= k;
    xyz.x /= (xyz.x + xyz.y + xyz.z);
    xyz.y /= (xyz.x + xyz.y + xyz.z);
    xyz.z = 1. - xyz.x - xyz.y;

    //xyz = vec3(xFit_1931(wave), yFit_1931(wave), zFit_1931(wave));
    vec3 rgb = XYZtosRGB(xyz);
    return rgb;
    //return vec3(intensity,0,0);
}

vec3 render( in vec3 ro, in vec3 rd, inout float seed ) {
    vec3 albedo, normal, col = vec3(1.); 
    float roughness, type;
    
    for (int i=0; i<PATH_LENGTH; ++i) {    
    	vec3 res = worldhit( ro, rd, vec2(.0001, 100), normal );
		if (res.z > 0.) {
			ro += rd * res.y;
       		
            getMaterialProperties(ro, res.z, albedo, type, roughness);
            
            if (type < LAMBERTIAN+.5) { // Added/hacked a reflection term
                float F = FresnelSchlickRoughness(max(0.,-dot(normal, rd)), .04, roughness);
                if (F > hash1(seed)) {
                    rd = modifyDirectionWithRoughness(normal, reflect(rd,normal), roughness, seed);
                } else {
                    col *= albedo;
			        rd = cosWeightedRandomHemisphereDirection(normal, seed);
                }
            } else if (type < METAL+.5) {
                col *= albedo;
                rd = modifyDirectionWithRoughness(normal, reflect(rd,normal), roughness, seed);            
            } else { // DIELECTRIC
                vec3 normalOut, refracted;
                float ni_over_nt, cosine, reflectProb = 1.;
                if (dot(rd, normal) > 0.) {
                    normalOut = -normal;
            		ni_over_nt = 1.4;
                    cosine = dot(rd, normal);
                    cosine = sqrt(1.-(1.4*1.4)-(1.4*1.4)*cosine*cosine);
                } else {
                    normalOut = normal;
                    ni_over_nt = 1./1.4;
                    cosine = -dot(rd, normal);
                }
            
	            // Refract the ray.
	            refracted = refract(normalize(rd), normalOut, ni_over_nt);
    	        
        	    // Handle total internal reflection.
                if(refracted != vec3(0)) {
                	float r0 = (1.-ni_over_nt)/(1.+ni_over_nt);
	        		reflectProb = FresnelSchlickRoughness(cosine, r0*r0, roughness);
                }
                
                rd = hash1(seed) <= reflectProb ? reflect(rd,normal) : refracted;
                rd = modifyDirectionWithRoughness(-normalOut, rd, roughness, seed);            
            }
        } else {
            col *= getSkyColor(rd);
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
        float nfpd = worldhit(ro, normalize(vec3(.5,0,-.5)-ro), vec2(0, 100), normal).y;
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

        vec3 col = render(ro, rd, seed);

        if (reset) {
            fragColor = vec4(col, 1);
        } else {
            fragColor = vec4(col, 1) + texelFetch(iChannel0, ivec2(fragCoord), 0);
            //fragColor = texelFetch(iChannel0, ivec2(fragCoord), 0);
        }
    }
    
}