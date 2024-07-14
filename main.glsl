// Spectral Ray Tracing. Created by Javier Gamazo 2024
// The MIT License
// https://www.shadertoy.com/view/Mfsyz7
//
// https://www.shadertoy.com/view/tl23Rm
// Taking some parts from https://www.shadertoy.com/view/tl23Rm and https://www.shadertoy.com/view/tsKyWG

#iChannel0 "file://buffer.glsl"

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec4 data = texelFetch(iChannel0, ivec2(fragCoord), 0);
    vec3 col = data.rgb / data.w;
    
    // gamma correction
    col = max( vec3(0), col - 0.004);
    col = (col*(6.2*col + .5)) / (col*(6.2*col+1.7) + 0.06);
    
    // Output to screen
    fragColor = vec4(col,1.0);
}