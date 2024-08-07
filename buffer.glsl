// Ray Tracing - Primitives. Created by Javier Gamazo 2024
// https://www.shadertoy.com/view/Mfsyz7
//
// I have combined different intersection routines in one shader (similar 
// to "Raymarching - Primitives": https://www.shadertoy.com/view/Xds3zN) and
// added a simple ray tracer to visualize a scene with all primitives.
// Functions for CIE from https://www.shadertoy.com/view/tsKyWG

#define PI 3.14159265358979323
#define PATH_LENGTH 12
#iChannel0 "self"

#define LOWER_BOUND 450
#define UPPER_BOUND 750
#define NUM_WAVELENGTHS 20
// #define MOVE_CAMERA

#define LAMBERTIAN 0
#define METAL 1
#define DIELECTRIC 2

const mat3 XYZ_2_RGB = (mat3(
     3.2404542,-0.9692660, 0.0556434,
    -1.5371385, 1.8760108,-0.2040259,
    -0.4985314, 0.0415560, 1.0572252
));

const mat3 RGB_2_XYZ = (mat3(
    0.4124564, 0.2126729, 0.0193339,
    0.3575761, 0.7151522, 0.1191920,
    0.1804375, 0.0721750, 0.9503041
));

float gaussian(float x, float mu, float sigma)
{
    return 1.0 / (sigma * sqrt(2.0 * PI)) * exp(-(x-mu)*(x-mu)/(2.*sigma*sigma));
}


// The CIE color matching functions were taken from  https://www.fourmilab.ch/documents/specrend
// The tabulated functions then were approximated with gaussians (for G and B) and with a mixture of two gaussiuns (R).
vec3 wavelength2XYZ(float l)
{
	return vec3(
    	8233.31 * gaussian(l, 593.951, 34.00) + 1891.26 * gaussian(l, 448.89, 18.785),
        10522.64 * gaussian(l, 555.38, 40.80),
        11254.78 * gaussian(l, 452.98, 21.57)
    );
}

float XYZ2WavelengthApprox(float l, vec3 color) {
    return dot(wavelength2XYZ(l), color) / 100.0;
}


struct Ray
{
    vec3 origin;
    vec3 direction;
    float wavelength;
};

struct Material
{
    int   materialType;
    vec3  albedo;
    float fuzz;
    float refractionIndex;
};

struct Hit
{
    float t;
    vec3 p;
    vec3 normal;
    Material mat;
};

struct Sphere{
    vec3 center;
    float radius;
    Material mat;
};

Sphere sceneList[] = Sphere[3](
    Sphere(
        vec3(0., 0., 0.),
        1.,
        Material(DIELECTRIC, vec3(.5, .4, .4), 1., 1.5)
    ),
    Sphere(
        vec3(1.5, 0.2, 0.2),
        0.2,
        Material(DIELECTRIC, vec3(.5, .4, .4), 1., 1.5)
    ),
    Sphere(
        vec3(0., -1001., 0.),
        1000.,
        Material(LAMBERTIAN, vec3(.5, .5, .2), .4, 0.)
    )
);


bool Sphere_hit(Sphere sphere, Ray ray, float t_min, float t_max, out Hit rec)
{
    vec3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float discriminant = b * b - a * c;

    if (discriminant > 0.0f)
    {
        float temp = (-b - sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min){
            rec.t = temp;
            rec.p = ray.origin + rec.t * ray.direction;
            rec.normal = (rec.p - sphere.center) / sphere.radius;
            rec.mat = sphere.mat;
            return true;
        }

        temp = (-b + sqrt(discriminant)) / a;

        if (temp < t_max && temp > t_min){
            rec.t = temp;
            rec.p = ray.origin + rec.t * ray.direction;
            rec.normal = (rec.p - sphere.center) / sphere.radius;
            rec.mat = sphere.mat;
            return true;
        }
    }

    return false;
}


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

bool opU(inout vec2 d, float iResult, in Material mat) {
    if (iResult < d.y) {
        d.y = iResult;
        return true;
    }
    return false;
}

//
// Palette by Íñigo Quílez: 
// https://www.shadertoy.com/view/ll2GD3
//
vec3 pal(in float t, in vec3 a, in vec3 b, in vec3 c, in vec3 d) {
    return a + b * cos(6.28318530718 * (c * t + d));
}

float checkerBoard(vec2 p) {
   return mod(floor(p.x) + floor(p.y), 2.);
}

bool worldhit(in Ray ray, in vec2 dist, out Hit rec) {
    Hit temp_rec;
    bool hit_anything = false;
    float closest_so_far = dist.y;

    for (int i = 0; i < sceneList.length(); i++) {
        if (Sphere_hit(sceneList[i], ray, dist.x, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}


vec3 getSkyColor(vec3 rd) {
    vec3 col = mix(vec3(1), vec3(.5, .7, 1), .5 + .5 * rd.y);
    col = vec3(0.);
    float sun = clamp(dot(normalize(vec3(-0.3, .7, -.6)), rd), 0., 1.);
    col += vec3(1, .6, .1) * (pow(sun, 4.) + 10. * pow(sun, 32.));
    return col;
}

float gpuIndepentHash(float p) {
    p = fract(p * .1031);
    p *= p + 19.19;
    p *= p + p;
    return fract(p);
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

vec3 refract_mine(vec3 v, vec3 n, float ni_over_nt) {
    float cos_theta = min(dot(-v, n), 1.0);
    vec3 r_out_perp = ni_over_nt * (v + cos_theta * n);
    vec3 r_out_parallel = -sqrt(abs(1. - dot(r_out_perp, r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

float skyColor(Ray ray) {
	vec3 sky = getSkyColor(ray.direction);
    sky = RGB_2_XYZ * pow(sky, vec3(2.2));
    return XYZ2WavelengthApprox(ray.wavelength, sky) * 0.5;
}

float n_wavelength(float lambda_nm) {
    float lambda_um = lambda_nm / 1000.0;
    
    // Coefficients for Cauchy's equation, adjusted to fit the range 1 < n < 2 for visible spectrum
    float A = 0.438;
    float B = 0.316;
    
    // Calculate refractive index
    float n_lambda = A + B / (lambda_um * lambda_um);
    
    return n_lambda;
}


float trace(in Ray ray, inout float seed) {
    vec3 albedo = vec3(1.); 
    float roughness, type;
    Material mat;
    Hit rec;
    float intensity = 1.;
    
    for (int i = 0; i < PATH_LENGTH; ++i) {    
    	bool didHit = worldhit(ray, vec2(.001, 100), rec);
        float res = rec.t;
        Material mat = rec.mat;
		if (didHit) {
			ray.origin += ray.direction * res;
            //ray.origin -= ray.direction * .0001;  // This should work, but it doesn't
            
            if (mat.materialType == LAMBERTIAN) { // Added/hacked a reflection term
                float F = FresnelSchlickRoughness(max(0.,-dot(rec.normal, ray.direction)), .04, mat.fuzz);
                if (F > hash1(seed)) {
                    ray.direction = modifyDirectionWithRoughness(rec.normal, reflect(ray.direction, rec.normal), mat.fuzz, seed);
                } else {
			        ray.direction = cosWeightedRandomHemisphereDirection(rec.normal, seed);
                }
                intensity *= mat.albedo.x * max(0.0, dot(rec.normal, ray.direction) / PI) * PI;  // TODO: Make this more legible. attenuation * scatterPDF / pdf
            } else if (mat.materialType == METAL) {
                ray.direction = modifyDirectionWithRoughness(rec.normal, reflect(ray.direction, rec.normal), mat.fuzz, seed);            
                intensity *= mat.albedo.x;  // TODO: Make this more legible.
            } else { // DIELECTRIC
                intensity *= 1.;
                vec3 normal, refracted;
                float ni_over_nt, cosine, reflectProb = 1.;
                float refractionIndex = mat.refractionIndex;
                refractionIndex = n_wavelength(ray.wavelength);
                // rec.normal is always pointing outwards
                if (dot(ray.direction, rec.normal) > 0.) {
                    // Ray is inside
                    normal = - rec.normal;
            		ni_over_nt = refractionIndex;
                    cosine = dot(ray.direction, normal);
                    cosine = sqrt(1. - (refractionIndex * refractionIndex) - (refractionIndex * refractionIndex) * cosine * cosine);
                } else {
                    normal = rec.normal;
                    ni_over_nt = 1. / refractionIndex;
                    cosine = - dot(ray.direction, normal);
                }
            
	            // Refract the ray.
	            refracted = refract(normalize(ray.direction), normal, ni_over_nt);
    	        
        	    // Handle total internal reflection.
                if(refracted != vec3(0)) {
                	float r0 = (1. - ni_over_nt)/(1. + ni_over_nt);
	        		reflectProb = FresnelSchlickRoughness(cosine, r0*r0, mat.fuzz);
                }
                
                ray.direction = hash1(seed) <= reflectProb ? reflect(ray.direction, normal) : refracted;
                ray.direction = modifyDirectionWithRoughness(normal, ray.direction, roughness, seed);            
            }
        } else {
            intensity *= skyColor(ray);
			return intensity;
        }
    }  
    return 0.;
}

mat3 setCamera( in vec3 ro, in vec3 ta, float cr ) {
	vec3 cw = normalize(ta - ro);
	vec3 cp = vec3(sin(cr), cos(cr), 0.0);
	vec3 cu = normalize(cross(cw, cp));
	vec3 cv = (cross(cu, cw));
    return mat3(cu, cv, cw);
}

vec3 render(in Ray ray, inout float seed) {
    vec3 col = vec3(0.);
    // Loop over the wavelengths
    for (int i = 0; i < NUM_WAVELENGTHS; i++) {
        ray.wavelength = float(LOWER_BOUND + i * (UPPER_BOUND - LOWER_BOUND) / NUM_WAVELENGTHS);
        float intensity = trace(ray, seed);
        vec3 color = wavelength2XYZ(ray.wavelength);

        col += color * intensity;
    }
    col = XYZ_2_RGB * col;
    col /= float(NUM_WAVELENGTHS);
    col /= 40.0;
	col = clamp(col, vec3(0.0), vec3(1.0));
    return col;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    bool reset = iFrame == 0;
        
    vec4 data = texelFetch(iChannel0, ivec2(0), 0);
    
    vec3 ro = vec3(3., 0., 0.);
    vec3 ta = vec3(1., 0., 0.);
    
    #ifdef MOVE_CAMERA
        vec2 mo = iMouse.xy == vec2(0) ? vec2(.125) : 
                abs(iMouse.xy)/iResolution.xy - .5;
            
        data = texelFetch(iChannel0, ivec2(0), 0);
        if (round(mo*iResolution.xy) != round(data.yz) || round(data.w) != round(iResolution.x)) {
            reset = true;
        }
        
        ro = vec3(.5+2.5*cos(1.5+6.*mo.x), 1.+2.*mo.y, -.5+2.5*sin(1.5+6.*mo.x));
        ta = vec3(.5, -.4, -.5);
    #endif

    mat3 ca = setCamera(ro, ta, 0.);
    Material mat;

    float fpd = data.x;
    if(all(equal(ivec2(fragCoord), ivec2(0)))) {
        // Calculate focus plane.
        Hit rec;
        Ray focus_ray = Ray(ro, normalize(vec3(.5,0,-.5)-ro), 0.);
        bool didHit = worldhit(focus_ray, vec2(0, 100), rec);
        fragColor = vec4(rec.t, iResolution.xy, iResolution.x);
    } else { 
        vec2 p = (-iResolution.xy + 2. * fragCoord - 1.) / iResolution.y;
        float seed = float(baseHash(floatBitsToUint(p - iTime))) / float(0xffffffffU);

        // AA
        p += 2. * hash2(seed) / iResolution.y;
        vec3 rd = ca * normalize(vec3(p.xy, 1.6));  

        // DOF
        vec3 fp = ro + rd * fpd;
        ro = ro + ca * vec3(randomInUnitDisk(seed), 0.) * .02;
        rd = normalize(fp - ro);

        Ray ray = Ray(ro, rd, 0.);

        vec3 col = render(ray, seed);

        if (reset) {
            fragColor = vec4(col, 1);
        } else {
            fragColor = vec4(col, 1) + texelFetch(iChannel0, ivec2(fragCoord), 0);
            //fragColor = texelFetch(iChannel0, ivec2(fragCoord), 0);
        }
    }
    
}