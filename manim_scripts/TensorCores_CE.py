from manim import *
from manim_voiceover import VoiceoverScene
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import math

class TensorCoresGraph(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService(transcription_model="base")
            )
        normal_times = [0.0229811, 0.0724406, 0.450077, 3.40259, 27.0164, ]
        tiled_times = [0.0162278, 0.0540909, 0.378201, 2.92617, 23.4949, ]
        tensor_core_times = [0.0113766, 0.0197933, 0.0607453, 0.501476, 3.70657, ]
        
        def to_flops(times, ns):
            return [(2*n*n*n)/(t*1e6) for t, n in zip(times, ns)]

        ps = list(range(8, 9+len(tiled_times)))

        n = [2**p for p in ps] 

        normal_flops = to_flops(normal_times, n)
        tiled_flops = to_flops(tiled_times, n)
        tc_flops = to_flops(tensor_core_times, n)

        ax = Axes(
            x_range=[ps[0] , ps[-1], 2],
            y_range=[0, tc_flops[-1] + 5000, 10000],
            x_axis_config={"scaling": LogBase(2)},
            axis_config={"include_numbers": True}).shift(0.4*RIGHT)

        labels = ax.get_axis_labels(x_label="n", y_label="Throughput[GFLOPS]")

        normal_graph = ax.plot_line_graph(
            x_values=n,
            y_values=normal_flops,
            line_color=RED,
            add_vertex_dots=False
        )

        tiled_graph = ax.plot_line_graph(
            x_values=n,
            y_values=tiled_flops,
            line_color=BLUE,
            add_vertex_dots=False
        )

        tc_graph = ax.plot_line_graph(
            x_values=n,
            y_values=tc_flops,
            line_color=GREEN,
            add_vertex_dots=False
        )
        tc_label = Text("TensorCores", font_size=32, color=GREEN).next_to(labels[1], DOWN, aligned_edge=LEFT)
        tc_label.shift(0.1*DOWN).align_to(labels[0], RIGHT)
        tiled_label = Text("Tiled Matmul", font_size=32, color=BLUE).next_to(tc_label, DOWN, aligned_edge=LEFT) 
        normal_label = Text("Naive Matmul", font_size=32, color=RED).next_to(tiled_label, DOWN, aligned_edge=LEFT)
        with self.voiceover(text="""If we were to graph the throughput that we are getting with tensor cores""") as trk:
            self.play(Create(ax), Write(labels))

        with self.voiceover(text="""We can see that just by utilizing tensor cores, we are getting an algorithm
                            that's 6 times faster than our tiled kernel that we wrote a while back""") as trk:
            self.play(Create(tc_graph), Create(tiled_graph), Create(normal_graph))
            self.play(Write(tc_label), Write(tiled_label), Write(normal_label))
        self.wait(1)

        ax_t = Axes(
            x_range=[ps[0] , ps[-1], 2],
            y_range=[0, 380000, 50000],
            x_axis_config={"scaling": LogBase(2)},
            axis_config={"include_numbers": True}).shift(0.4*RIGHT)

        normal_graph_t = ax_t.plot_line_graph(
            x_values=n,
            y_values=normal_flops,
            line_color=RED,
            add_vertex_dots=False
        )

        tiled_graph_t = ax_t.plot_line_graph(
            x_values=n,
            y_values=tiled_flops,
            line_color=BLUE,
            add_vertex_dots=False
        )

        tc_graph_t = ax_t.plot_line_graph(
            x_values=n,
            y_values=tc_flops,
            line_color=GREEN,
            add_vertex_dots=False
        )

        theoretical_max_tc = ax_t.plot_line_graph(
            x_values=n,
            y_values=([330000] * len(tc_flops)),
            line_color=GOLD,
            add_vertex_dots=False
        )

        theoretical_max_tc_t = Text("Tensor Core theoretical maximum (330 TFLOPS)", color=GOLD, font_size=24).next_to(theoretical_max_tc, DOWN)

        theoretical_max = ax_t.plot_line_graph(
            x_values=n,
            y_values=([86000] * len(tc_flops)),
            line_color=YELLOW,
            add_vertex_dots=False
        )

        theoretical_max_t = Text("Cuda Cores theoretical maximum (86 TFLOPS)", color=YELLOW, font_size=24).next_to(theoretical_max, DOWN)
        
        with self.voiceover(text="""But to be completly honest with you, if we zoom out our graph we can see
                            that we are only slightly above 10% of what the hardware is capable of""") as trk:
            self.play(Transform(ax, ax_t), Transform(tc_graph, tc_graph_t), 
                      Transform(normal_graph, normal_graph_t), Transform(tiled_graph, tiled_graph_t))

            self.play(Create(theoretical_max_tc), Write(theoretical_max_tc_t))
        with self.voiceover(text="""In fact we have not even reached the theoretical maximum of what
                            the GPU can do without using tensor cores. But filling this gap will be the topic of future videos""") as trk:
            self.play(Create(theoretical_max), Write(theoretical_max_t))

        self.play(*[FadeOut(x) for x in self.mobjects])
        self.wait(2)

class TensorCoresCode2(Scene):
    def construct(self):
        code = """using layout = nvcuda::wmma::row_major;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, layout> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, layout> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> acc;"""
        code_obj = Code(code=code, tab_width=2, language="c++", style='monokai', margin=0.1, line_spacing=0.7, insert_line_no=False, font_size=12, corner_radius=0.1)
        code_obj.code = remove_invisible_chars(code_obj.code)
        hl_a = SurroundingRectangle(code_obj.code[1], buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        hl_b = SurroundingRectangle(code_obj.code[2], buff=0.03, stroke_width=2, fill_opacity=0.3, color=ORANGE)
        hl_acc = SurroundingRectangle(code_obj.code[3], buff=0.03, stroke_width=2, fill_opacity=0.3, color=GREEN)
        self.play(Create(code_obj))
        self.play(Create(hl_a), Create(hl_b), Create(hl_acc))

        self.wait(1)

        code = """constexpr int M = 16;
constexpr int K = 16;
constexpr int N = 16;
using layout = nvcuda::wmma::row_major;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, layout> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, layout> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> acc;"""
        code_obj_t = Code(code=code, tab_width=2, language="c++", style='monokai', margin=0.1, line_spacing=0.7, insert_line_no=False, font_size=12, corner_radius=0.1)
        code_obj_t.code = remove_invisible_chars(code_obj_t.code)
        hl_a_t = SurroundingRectangle(code_obj_t.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        hl_b_t = SurroundingRectangle(code_obj_t.code[5], buff=0.03, stroke_width=2, fill_opacity=0.3, color=ORANGE)
        hl_acc_t = SurroundingRectangle(code_obj_t.code[6], buff=0.03, stroke_width=2, fill_opacity=0.3, color=GREEN)
        self.play(Transform(code_obj, code_obj_t), Transform(hl_a, hl_a_t), Transform(hl_b, hl_b_t), Transform(hl_acc, hl_acc_t))

        self.wait(1)

        code = """constexpr int M = 32;
constexpr int K = 8;
constexpr int N = 16;
using layout = nvcuda::wmma::row_major;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, layout> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, layout> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> acc;"""
        code_obj_t = Code(code=code, tab_width=2, language="c++", style='monokai', margin=0.1, line_spacing=0.7, insert_line_no=False, font_size=12, corner_radius=0.1)
        code_obj_t.code = remove_invisible_chars(code_obj_t.code)
        hl_a_t = SurroundingRectangle(code_obj_t.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        hl_b_t = SurroundingRectangle(code_obj_t.code[5], buff=0.03, stroke_width=2, fill_opacity=0.3, color=ORANGE)
        hl_acc_t = SurroundingRectangle(code_obj_t.code[6], buff=0.03, stroke_width=2, fill_opacity=0.3, color=GREEN)
        self.play(Transform(code_obj, code_obj_t), Transform(hl_a, hl_a_t), Transform(hl_b, hl_b_t), Transform(hl_acc, hl_acc_t))

        self.wait(1)
    
        code = """constexpr int M = 8;
constexpr int K = 32;
constexpr int N = 16;
using layout = nvcuda::wmma::row_major;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, half, layout> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, half, layout> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, half> acc;"""
        code_obj_t = Code(code=code, tab_width=2, language="c++", style='monokai', margin=0.1, line_spacing=0.7, insert_line_no=False, font_size=12, corner_radius=0.1)
        code_obj_t.code = remove_invisible_chars(code_obj_t.code)
        self.play(Transform(code_obj, code_obj_t))


        self.wait(1)

        code = """constexpr int MKN = 16;
using layout = nvcuda::wmma::row_major;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MKN, MKN, MKN, half, layout> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MKN, MKN, MKN, half, layout> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MKN, MKN, MKN, half> acc;
"""
        code_obj_t = Code(code=code, tab_width=2, language="c++", style='monokai', margin=0.1, line_spacing=0.7, insert_line_no=False, font_size=12, corner_radius=0.1)
        code_obj_t.code = remove_invisible_chars(code_obj_t.code)
        hl_a_t = SurroundingRectangle(code_obj_t.code[2], buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        hl_b_t = SurroundingRectangle(code_obj_t.code[3], buff=0.03, stroke_width=2, fill_opacity=0.3, color=ORANGE)
        hl_acc_t = SurroundingRectangle(code_obj_t.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3, color=GREEN)
        self.play(Transform(code_obj, code_obj_t), Transform(hl_a, hl_a_t), Transform(hl_b, hl_b_t), Transform(hl_acc, hl_acc_t))

        self.wait(1)

        code = """constexpr int MKN = 16;
using layout = nvcuda::wmma::row_major;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, MKN, MKN, MKN, half, layout> a_frag;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, MKN, MKN, MKN, half, layout> b_frag;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, MKN, MKN, MKN, half> acc;

nvcuda::wmma::fill_fragment(acc, 0);
"""
        code_obj_t = Code(code=code, tab_width=2, language="c++", style='monokai', margin=0.1, line_spacing=0.7, insert_line_no=False, font_size=12, corner_radius=0.1)
        code_obj_t.code = remove_invisible_chars(code_obj_t.code)
        hl_a_t = SurroundingRectangle(code_obj_t.code[2], buff=0.03, stroke_width=2, fill_opacity=0.3, color=BLUE)
        hl_b_t = SurroundingRectangle(code_obj_t.code[3], buff=0.03, stroke_width=2, fill_opacity=0.3, color=ORANGE)
        hl_acc_t = SurroundingRectangle(code_obj_t.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3, color=GREEN)
        self.play(Transform(code_obj, code_obj_t), Transform(hl_a, hl_a_t), Transform(hl_b, hl_b_t), Transform(hl_acc, hl_acc_t))

        self.wait(1)
        

        code = """for (int32_t i = 0; i < n; i+= MKN)
{
    const int32_t matrix_a_row = warpM * MKN;
    const int32_t matrix_b_col = warpN * MKN;

    if(matrix_a_row<n && matrix_b_col<n && i<n)
    {
        nvcuda::wmma::load_matrix_sync(a_frag, a + matrix_a_row * n + i, n);
        nvcuda::wmma::load_matrix_sync(b_frag, b + i * n + matrix_b_col, n);

        nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
}
"""
        code_obj_t = Code(code=code, tab_width=2, language="c++", style='monokai', margin=0.1, line_spacing=0.7, insert_line_no=False, font_size=12, corner_radius=0.1)
        code_obj_t.code = remove_invisible_chars(code_obj_t.code)
        self.play(Transform(code_obj, code_obj_t), FadeOut(hl_a), FadeOut(hl_b), FadeOut(hl_acc))

        self.wait(1)

        hl = SurroundingRectangle(code_obj_t.code[7:9], buff=0.03, stroke_width=2, fill_opacity=0.3, color=RED)
        self.play(Create(hl))

        self.wait(1)

        hl_t = SurroundingRectangle(code_obj_t.code[10], buff=0.03, stroke_width=2, fill_opacity=0.3, color=RED)
        self.play(Transform(hl, hl_t))

        self.wait(1)

        hl_t = SurroundingRectangle(code_obj_t.code[0], buff=0.03, stroke_width=2, fill_opacity=0.3, color=RED)
        self.play(Transform(hl, hl_t))

        self.wait(1)

        hl_t = SurroundingRectangle(code_obj_t.code[7:9], buff=0.03, stroke_width=2, fill_opacity=0.3, color=RED)
        self.play(Transform(hl, hl_t))

        self.wait(1)

        hl_t = SurroundingRectangle(code_obj_t.code[10], buff=0.03, stroke_width=2, fill_opacity=0.3, color=RED)
        self.play(Transform(hl, hl_t))

        self.wait(1)

        self.play(*[FadeOut(x) for x in self.mobjects])
