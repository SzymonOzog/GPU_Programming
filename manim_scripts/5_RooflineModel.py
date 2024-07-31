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

    with self.voiceover(text="""When we run our code, there are 3 main areas in play""") as trk:
      while trk.get_remaining_duration() > 1:
        self.wait(0.1)
      self.play(Unwrite(title), Unwrite(subtitle))

    

    gpu = Rectangle(height=2, width=3, color=GREEN, fill_color=GREEN, fill_opacity=0.5).shift(2*LEFT+UP)
    gpu_t = Text("GPU", color=GREEN).move_to(gpu)
    with self.voiceover(text="""The firs one is obviously our gpu""") as trk:
      self.play(Create(gpu))
      self.play(Write(gpu_t))

    memory = Rectangle(height=2, width=3, color=RED, fill_color=RED, fill_opacity=0.5)
    memory.next_to(gpu, DOWN, aligned_edge=LEFT, buff=0).shift(DOWN)
    mem_t = Text("HBM", color=RED).move_to(memory)
    m_to_g = DoubleArrow(gpu.get_corner(DOWN), memory.get_corner(UP), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
    with self.voiceover(text="""It is directly connected to it's High Bandwidth Memory, so our GPU's VRAM""") as trk:
      self.play(Create(memory), Create(m_to_g))
      self.play(Write(mem_t))

    cpu = Rectangle(height=5, width=3, color=BLUE, fill_color=BLUE, fill_opacity=0.5).next_to(gpu, RIGHT, aligned_edge=UP).shift(RIGHT)
    cpu_t = Text("CPU", color=BLUE).move_to(cpu)

    l1 = cpu.get_corner(LEFT)
    l1[1] = memory.get_corner(RIGHT)[1]
    c_to_m = Arrow(l1, memory.get_corner(RIGHT), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
    l1 = cpu.get_corner(LEFT)
    l1[1] = gpu.get_corner(RIGHT)[1]
    c_to_g = Arrow(l1, gpu.get_corner(RIGHT), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)

    with self.voiceover(text="""And there is also the cpu, that can schedule kernels on our gpu as well as copy data to our HBM""") as trk:
      self.play(Create(cpu), Create(c_to_m), Create(c_to_g))
      self.play(Write(cpu_t))

    with self.voiceover(text="""And performance can be a problem in all 3 of those""") as trk:
      pass
    overhead = SurroundingRectangle(cpu, color=BLUE_E).scale(1.1)
    overhead_t = Text("Overhead", color=BLUE_E, font_size=24).next_to(overhead, UP)
    with self.voiceover(text="""The time spent on the CPU is called overhead""") as trk:
      self.play(Create(overhead))
      self.play(Write(overhead_t))
    
    mem_bound = SurroundingRectangle(memory, color=RED_E).scale(1.1)
    mem_bound_t = Text("Memory Access Latency", color=RED_E, font_size=24).next_to(mem_bound, DOWN)

    with self.voiceover(text="""The time that we spend loading data from memory is our memory access latency""") as trk:
      self.play(Create(mem_bound))
      self.play(Write(mem_bound_t))

    comp_bound = SurroundingRectangle(gpu, color=GREEN_E).scale(1.1)
    comp_bound_t = Text("Compute Time", color=GREEN_E, font_size=24).next_to(comp_bound, UP)
    with self.voiceover(text="""And finally, the time on the gpu is our actuall compute time""") as trk:
      self.play(Create(comp_bound))
      self.play(Write(comp_bound_t))


