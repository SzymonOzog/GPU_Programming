#version 330

// uniform vec3 camera_position;

in vec3 point;
in vec3 d_normal_point;
in vec4 rgba;

out vec4 v_color;

#INSERT emit_gl_Position.glsl
#INSERT get_unit_normal.glsl
#INSERT finalize_color.glsl

const float EPSILON = 1e-10;

void main(){
    emit_gl_Position(point);
    vec3 unit_normal = normalize(d_normal_point - point);
    vec3 camera_normal = normalize(camera_position - point);
    if (dot(camera_normal, unit_normal) < 0)
    {
        v_color = vec4(0);
    }
    else
    {
        v_color = finalize_color(rgba, point, unit_normal);
    }
}
