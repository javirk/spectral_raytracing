// Ray Tracing - Primitives. Created by Reinder Nijhoff 2019
// The MIT License
// @reindernijhoff
//
// https://www.shadertoy.com/view/tl23Rm
//
// I wanted to create a reference shader similar to "Raymarching - Primitives" 
// (https://www.shadertoy.com/view/Xds3zN), but with ray-primitive intersection 
// routines instead of sdf routines.
// 
// As usual, I ended up mostly just copy-pasting code from Íñigo Quílez: 
// 
// https://iquilezles.org/articles/intersectors
// 
// Please let me know if there are other routines that I should add to this shader.
// 
// Sphere:          https://www.shadertoy.com/view/4d2XWV
//

#define MAX_DIST 1e10
float dot2( in vec3 v ) { return dot(v,v); }

// Plane 
float iPlane( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
              in vec3 planeNormal, in float planeDist) {
    float a = dot(rd, planeNormal);
    float d = -(dot(ro, planeNormal)+planeDist)/a;
    if (a > 0. || d < distBound.x || d > distBound.y) {
        return MAX_DIST;
    } else {
        normal = planeNormal;
    	return d;
    }
}

// Sphere:          https://www.shadertoy.com/view/4d2XWV
float iSphere( in vec3 ro, in vec3 rd, in vec2 distBound, inout vec3 normal,
               float sphereRadius ) {
    float b = dot(ro, rd);
    float c = dot(ro, ro) - sphereRadius*sphereRadius;
    float h = b*b - c;
    if (h < 0.) {
        return MAX_DIST;
    } else {
	    h = sqrt(h);
        float d1 = -b-h;
        float d2 = -b+h;
        if (d1 >= distBound.x && d1 <= distBound.y) {
            normal = normalize(ro + rd*d1);
            return d1;
        } else if (d2 >= distBound.x && d2 <= distBound.y) { 
            normal = normalize(ro + rd*d2);            
            return d2;
        } else {
            return MAX_DIST;
        }
    }
}

