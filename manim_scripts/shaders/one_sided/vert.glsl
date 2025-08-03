#version 330

// uniform vec3 camera_position;
uniform vec3 center;
uniform vec3 shape;
uniform float fade_shading;

in vec3 point;
in vec3 d_normal_point;
in vec4 rgba;

out vec4 v_color;

#INSERT emit_gl_Position.glsl
#INSERT get_unit_normal.glsl
#INSERT finalize_color.glsl

const float EPSILON = 1e-10;

void main(){
    vec3 unit_normal = normalize(d_normal_point - point);
    vec3 new_point = point;
    vec4 out_rgba = rgba;

    // HACK for the nice creation
    if (rgba.w < 0)
    {
        out_rgba.w = -1*rgba.w;
        new_point += (1 - out_rgba.w) * unit_normal;
    }

    if (fade_shading > 0.f)
    {
        float s = 0.f;
        float x = (point.x - center.x)/(shape.x + EPSILON);
        s += (x*x);
        float y = (point.y - center.y)/(shape.y + EPSILON);
        s += (y*y);
        float z = (point.z - center.z)/(shape.z + EPSILON);
        s += (z*z);
        s+=0.5f;
        if (s > 0.f && s < 1.f)
        {
            out_rgba.x *= s;
            out_rgba.y *= s;
            out_rgba.z *= s;
        }
    }

    emit_gl_Position(new_point);
    vec3 camera_normal = normalize(camera_position - point);
    if (dot(camera_normal, unit_normal) < 0)
    {
        v_color = vec4(0);
    }
    else
    {
        v_color = finalize_color(out_rgba, point, unit_normal);
    }
}
