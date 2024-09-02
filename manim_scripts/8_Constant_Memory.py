from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np


class ConstantMemory(VoiceoverScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        )

    title = Text("GPU programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 8 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Constant Memory", font_size=48).next_to(title, DOWN)
    with self.voiceover(text="""This episode will be focused on constant memory, and when to use it to improve the 
                        performance of our code""") as trk:
      self.play(Write(subtitle))

    self.play(Unwrite(title), Unwrite(subtitle))

    ## some intro on how constant mem works

    with self.voiceover(text="""There is a wonderfull blogpost by Lei Mao that profiled the different usecases for constant memory
                        to show when to use it and when not to""") as trk:
      pass

    header = Text("Modifications", font_size=72).shift(UP)
    modifications = BulletedList("Change datatype to float",  "Clear L2 cache", "Multiple input sizes", font_size=48).next_to(header, DOWN)
    with self.voiceover(text="""Inspired by this I expanded the work by changing the datatype from int to float as GPU's 
                        are optimized for floating point operations""") as trk:
      self.play(Write(header))
      self.play(Write(modifications[0]))

    with self.voiceover(text="""Betweel the subsequent runs of the benchmark I'm clearing the L2 cache so that it doesn't influence
                        the outcome""") as trk:
      self.play(Write(modifications[1]))

    with self.voiceover(text="""And I'm running the benchmarks for multiple input sizes""") as trk:
      self.play(Write(modifications[2]))

    ratio_block = [0.958, 0.914, 0.979, 0.938, 1.007, 0.973, 0.941, 0.975, 0.935, 0.946, 0.949, 0.954, 0.937, 0.968, 0.954, ]
    ratio_thread = [1.355, 1.392, 1.334, 1.304, 1.239, 1.176, 1.055, 1.008, 1.001, 1.043, 1.024, 1.010, 0.967, 0.988, 0.978, ]
    ratio_random = [5.994, 4.811, 4.009, 3.992, 3.620, 2.586, 2.079, 2.199, 2.861, 3.909, 5.022, 5.748, 6.258, 6.666, 6.868, ]
    rng = list(range(10, 25))
    nums = [2**x for x in rng]

    ax = Axes(
        x_range=[rng[0], rng[-1], 2],
        y_range=[0.7, 1.5, 0.1],
        x_axis_config={"scaling": LogBase(2)},
        axis_config={"include_numbers": True}).scale(0.9)

    labels = ax.get_axis_labels(x_label="Input \\, Size", y_label="\\frac{Const}{Global}")

    block_graph = ax.plot_line_graph(
        x_values=nums,
        y_values=ratio_block,
        line_color=BLUE,
        add_vertex_dots=False
    )

    
    with self.voiceover(text="""Let's look at the results of different scenarios for using constant memory""") as trk:
      self.play(Unwrite(header), *[Unwrite(x) for x in modifications])
      self.play(Create(ax))
      self.play(Write(labels))
    
    fs = 40
    title = Text("One access per block", font_size=fs).next_to(ax, UP)
    with self.voiceover(text="""In case of one access per block <bookmark mark='1'/>
                        We can see that using constant memory provides us with an advantage of around 5-10% speed increase when compared
                        with global memory""") as trk:
      self.play(Write(title))
      self.wait_until_bookmark("1")
      self.play(Create(block_graph))


    with self.voiceover(text="""When we do one access per thread""") as trk:
      self.play(Transform(title, Text("One access per thread", font_size=fs).next_to(ax, UP)))

    thread_graph = ax.plot_line_graph(
        x_values=nums,
        y_values=ratio_thread,
        line_color=BLUE,
        add_vertex_dots=False
    )
    with self.voiceover(text="""The results are no longer as clear""") as trk:
      self.play(Transform(block_graph, thread_graph, replace_mobject_with_target_in_scene=True))


    with self.voiceover(text="""For small inputs, constant memory is giving us terrible performance, as loading values from const memory
                        is initially slower than loading them from global memory""") as trk:
      pass

    with self.voiceover(text="""But as our input lenght get's bigger and we launch more and more blocks, it starts getting simillar of even slightly
                        better performance""") as trk:
      pass

    with self.voiceover(text="""I'm not exactly sure why is this happening, my initial guess is that the first blocks executed take the values inside
                        our constant cache, and subsequent ones are just comparing reads from cache for global memory vs from constant cache
                        which starts favoring constant memory""") as trk:
      pass

    with self.voiceover(text="""But do take this with a grain of salt as it's nothing that I can swear by""") as trk:
      pass

    with self.voiceover(text="""And the last case will be random access""") as trk:
      self.play(Transform(title, Text("Random Access", font_size=fs).next_to(ax, UP)))

    ax2 = Axes(
        x_range=[rng[0], rng[-1], 2],
        y_range=[1, 8, 1],
        x_axis_config={"scaling": LogBase(2)},
        axis_config={"include_numbers": True}).scale(0.9)

    random_graph = ax2.plot_line_graph(
        x_values=nums,
        y_values=ratio_random,
        line_color=BLUE,
        add_vertex_dots=False
    )

    with self.voiceover(text="""For this one, constant memory is performing terrible, as the values are read in a random order and
                        they cannot leaverage our constant cache, resulting in a slow read from constant memory for each access""") as trk:
      self.play(Transform(thread_graph, random_graph, replace_mobject_with_target_in_scene=True),
                Transform(ax, ax2))
    self.wait(1)
