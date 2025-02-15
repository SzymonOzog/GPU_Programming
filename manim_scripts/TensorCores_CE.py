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

        ps = list(range(8, 9+len(tiled_times)))

        n = [2**p for p in ps] 
        log_normal_t = [math.log10(t) for t in normal_times]
        log_tiled_t = [math.log10(t) for t in tiled_times]
        log_tc_t = [math.log10(t) for t in tensor_core_times]

        ax = Axes(
            x_range=[ps[0] , ps[-1], 2],
            y_range=[log_tc_t[0] , log_normal_t[-1], 1],
            x_axis_config={"scaling": LogBase(2)},
            y_axis_config={"scaling": LogBase()},
            axis_config={"include_numbers": True})

        labels = ax.get_axis_labels(x_label="n", y_label="Time[ns]")

        normal_graph = ax.plot_line_graph(
            x_values=n,
            y_values=normal_times,
            line_color=RED,
            add_vertex_dots=False
        )

        tiled_graph = ax.plot_line_graph(
            x_values=n,
            y_values=tiled_times,
            line_color=BLUE,
            add_vertex_dots=False
        )

        tc_graph = ax.plot_line_graph(
            x_values=n,
            y_values=tensor_core_times,
            line_color=GREEN,
            add_vertex_dots=False
        )
        tc_label = Text("TensorCores", font_size=32, color=GREEN).next_to(labels[1], DOWN).shift(0.3*RIGHT+0.2*DOWN)
        tiled_label = Text("Tiled Matmul", font_size=32, color=BLUE).next_to(tc_label, DOWN, aligned_edge=LEFT)
        normal_label = Text("Standard Matmul", font_size=32, color=RED).next_to(tiled_label, DOWN, aligned_edge=LEFT)
        self.play(Create(ax), Write(labels))

        self.play(Create(tc_graph), Create(tiled_graph), Create(normal_graph))
        self.play(Write(tc_label), Write(tiled_label), Write(normal_label))
        self.wait(1)

