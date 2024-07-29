from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
from NN import NeuralNetworkMobject

mnist = np.loadtxt("mnist_train.csv", delimiter=",")

class NeuralNetwork(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU Programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 5 in the series on GPU programming") as trk:
      self.play(Write(title))


    subtitle = Text("Roofline Model", font_size=48).next_to(title, DOWN)
    with self.voiceover(text="""In this episode we are going to discuss some performance characteristics, and main factors
                        that influence the performance of the GPU, we'll also introduce the roofline model for assessing
                        our code's performance with regards to the hardware possibilities""") as trk:
      self.play(Write(subtitle))

    self.play(Unwrite(title), Unwrite(subtitle))

    gpu = Rectangle(height=2, width=3, color=GREEN, fill_color=GREEN, fill_opacity=0.5).shift(2*LEFT+UP)
    gpu_t = Text("GPU", color=GREEN).move_to(gpu)

    memory = Rectangle(height=2, width=3, color=RED, fill_color=RED, fill_opacity=0.5)
    memory.next_to(gpu, DOWN, aligned_edge=LEFT, buff=0).shift(DOWN)
    mem_t = Text("HBM", color=RED).move_to(memory)

    cpu = Rectangle(height=5, width=3, color=BLUE, fill_color=BLUE, fill_opacity=0.5).next_to(gpu, RIGHT, aligned_edge=UP).shift(RIGHT)
    cpu_t = Text("CPU", color=BLUE).move_to(cpu)

    m_to_g = DoubleArrow(gpu.get_corner(DOWN), memory.get_corner(UP), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
    l1 = cpu.get_corner(LEFT)
    l1[1] = memory.get_corner(RIGHT)[1]
    c_to_m = Arrow(l1, memory.get_corner(RIGHT), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
    l1 = cpu.get_corner(LEFT)
    l1[1] = gpu.get_corner(RIGHT)[1]
    c_to_g = Arrow(l1, gpu.get_corner(RIGHT), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)

    self.play(Create(cpu), Create(memory), Create(gpu))
    self.play(Write(cpu_t), Write(mem_t), Write(gpu_t))
    self.play(Create(m_to_g), Create(c_to_m), Create(c_to_g))
