from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import math

class TensorCoresCode(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            # RecorderService(transcription_model="bapse")
            GTTSService(transcription_model="base")
            )
        self.voiceovers_in_embed = True
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
            axis_config={"include_numbers": True}).shift(0.2*RIGHT)

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
        tc_label = Text("TensorCores", font_size=32, color=GREEN).next_to(labels[1], DOWN, aligned_edge=LEFT).shift(0.3*RIGHT+0.1*DOWN)
        tiled_label = Text("Tiled Matmul", font_size=32, color=BLUE).next_to(tc_label, DOWN, aligned_edge=LEFT) 
        normal_label = Text("Naive Matmul", font_size=32, color=RED).next_to(tiled_label, DOWN, aligned_edge=LEFT)
        self.play(Create(ax), Write(labels))

        self.play(Create(tc_graph), Create(tiled_graph), Create(normal_graph))
        self.play(Write(tc_label), Write(tiled_label), Write(normal_label))
        self.wait(1)

