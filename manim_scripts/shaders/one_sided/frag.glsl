#version 330

flat in vec4 v_color;
out vec4 frag_color;

void main() {
    frag_color = v_color;
}

