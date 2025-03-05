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
        normal_times = [0.0108544, 0.0195504, 0.0610426, 0.502562, 3.72811, ]
        tiled_times = [0.0186675, 0.0334739, 0.0679533, 0.153344, 0.822092, ]
        cublas_times = [0.00583872, 0.0082176, 0.027305, 0.0733114, 0.576111, ]

        
 
        def to_flops(times, ns):
            return [(2*n*n*n)/(t*1e6) for t, n in zip(times, ns)]

        ps = list(range(8, 9+len(tiled_times)))

        n = [2**p for p in ps] 

        normal_flops = to_flops(normal_times, n)
        tiled_flops = to_flops(tiled_times, n)
        cublas_flops = to_flops(cublas_times, n)

        ax = Axes(
            x_range=[ps[0] , ps[-1], 2],
            y_range=[0, 380000, 50000],
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

        cublas_graph = ax.plot_line_graph(
            x_values=n,
            y_values=cublas_flops,
            line_color=GREEN,
            add_vertex_dots=False
        )
        normal_label = Text("Tensor Cores", font_size=32, color=RED).next_to(labels[1], DOWN, aligned_edge=LEFT)
        normal_label.shift(0.1*DOWN).align_to(labels[0], RIGHT)
        tiled_label = Text("Hierarchical\nTiling", font_size=32, color=BLUE).next_to(normal_label, DOWN, aligned_edge=LEFT) 
        cublas_label = Text("cuBLAS", font_size=32, color=GREEN).next_to(tiled_label, DOWN, aligned_edge=LEFT)
        with self.voiceover(text="""Last time we measured how much FLOPs we were gettin with our tensor cores kernel""") as trk:
            self.play(Create(ax), Write(labels))
            self.play(Create(normal_graph))
            self.play(Write(normal_label))


        theoretical_max_tc = ax.plot_line_graph(
            x_values=n,
            y_values=([330000] * len(cublas_flops)),
            line_color=GOLD,
            add_vertex_dots=False
        )

        theoretical_max_tc_t = Text("Tensor Core theoretical maximum (330 TFLOPS)", color=GOLD, font_size=24).next_to(theoretical_max_tc, DOWN)

        with self.voiceover(text="""and compared it to what is the theoretical maximum of our GPU""") as trk:
            self.play(Create(theoretical_max_tc), Write(theoretical_max_tc_t))

        with self.voiceover(text="""By utilizing hierarchical tiling, we can get much higher throughput, even up to 4.5x higher than 
                            we were getting with our original kernel""") as trk:
            self.play(Create(tiled_graph))
            self.play(Write(tiled_label))
        self.wait(1)

        with self.voiceover(text="""But that's still off from our theoretical maximum, and this is something that we will never achieve""") as trk:
            pass

        with self.voiceover(text="""To see what is the state of the art, we can run cublas kernels from nvidia, this will be our reference point from now on""") as trk:
            self.play(Create(cublas_graph))
            self.play(Write(cublas_label))
        self.wait(1)

        with self.voiceover(text="""And in the next episode, we'll once again move closer to this line""") as trk:
            pass

