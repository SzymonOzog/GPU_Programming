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

    w_scale = 1 
    overhead_time = 2
    training_time = 9.5
    overhead_r = Rectangle(height=0.5, width=overhead_time/w_scale, color=BLUE, fill_color=BLUE, fill_opacity=0.5)
    training_r = Rectangle(height=0.5, width=training_time/w_scale, color=GREEN, fill_color=GREEN, fill_opacity=0.5)
    perf = VGroup(overhead_r, training_r).arrange(RIGHT,buff=0.02)
    with self.voiceover(text="""Let's first look into our overhead, in our case it is the time that we spend loading the dataset from disk""") as trk:
      self.play(Transform(VGroup(cpu, cpu_t, overhead, overhead_t, c_to_m, c_to_g), overhead_r, replace_mobject_with_target_in_scene=True),
                Transform(VGroup(gpu, gpu_t, memory, mem_t, mem_bound, mem_bound_t, comp_bound, comp_bound_t, m_to_g), training_r, replace_mobject_with_target_in_scene=True))
    fs = 20
    overhead_t = Text("2 ms", font_size=fs, color=BLUE).move_to(overhead_r)
    training_t = Text("9.5 ms", font_size=fs, color=GREEN).move_to(training_r)
    with self.voiceover(text="""And it takes around 2 miliseconds to do that, while our 10 epochs of training <bookmark mark='1'/>take around 9.5 miliseconds in total""") as trk:
      self.play(Write(overhead_t))
      self.wait_until_bookmark("1")
      self.play(Write(training_t))

    trans = Rectangle(height=0.5, width=1/w_scale, color=BLUE, fill_color=BLUE, fill_opacity=0.5).move_to(overhead_r, aligned_edge=RIGHT)
    with self.voiceover(text="""The first obvious thing that we can do is to optimize our cpu code""") as trk:
      pass
    self.wait(1)

    with self.voiceover(text="""I managed to get it down to 1 milisecond, by using some more optimized string parsing functions""") as trk:
      self.play(Transform(overhead_r, trans), Transform(overhead_t, Text("1 ms", font_size=fs, color=BLUE).move_to(trans)))
    self.wait(1)

    epochs = VGroup(*[Rectangle(height=0.5, width=training_time/(10*w_scale), color=GREEN, fill_color=GREEN, fill_opacity=0.5) for i in range(10)]).arrange(RIGHT, buff=0.02).move_to(training_r, aligned_edge=LEFT)
    epoch_times = VGroup(*[Text("0.95 ms", font_size=fs, color=GREEN).move_to(epochs[i]) for i in range(10)])

    with self.voiceover(text="""But we want to mitigate our overhead even further, and one thing that you can notice is that we do not need the full
                        dataset at the start of the first epoch, we only need the data that we will be working on""") as trk:
      self.play(Transform(training_r, epochs, replace_mobject_with_target_in_scene=True),
                Transform(training_t, epoch_times, replace_mobject_with_target_in_scene=True))
    self.wait(1)
    overhead_r.add(overhead_t)
    with self.voiceover(text="""So we can execute the cpu and gpu code in parallel, where the gpu already starts training our network on the first batch
                        as the cpu loads the next one in the background""") as trk:
      self.play(overhead_r.animate.next_to(epochs[0], DOWN, aligned_edge=LEFT, buff=0.02))
      group = VGroup(overhead_r, epochs[0], epoch_times[0])
      self.play(group.animate.next_to(epochs[1], LEFT, buff=0.02))
    self.wait(1)
    
    training_time=3
    e2 = Rectangle(height=0.5, width=training_time/(10*w_scale), color=GREEN, fill_color=GREEN, fill_opacity=0.5).move_to(epochs[0], aligned_edge=LEFT)
    epochs_2 = VGroup(*[Rectangle(height=0.5, width=training_time/(10*w_scale), color=GREEN, fill_color=GREEN, fill_opacity=0.5) for i in range(10)]).arrange(RIGHT, buff=0.02).next_to(group, RIGHT, buff=0.02).shift(0.3*LEFT)
    anims = []
    anims.extend([Unwrite(x) for x in epoch_times])
    anims.append(Transform(epochs[0], e2))
    for i in range(1, 10):
      anims.append(Transform(epochs[i], epochs_2[i]))

    color_grad = color_gradient([BLUE, GREEN], 2)

    with self.voiceover(text="""Now this does not eliminate our overhead, we still need to load all of our data during the first epoch, 
                        so if we optimize our <bookmark mark='1'/>epoch time, our first epoch will still be limited by the time that it takes to load
                        the data""") as trk:
      self.wait_until_bookmark("1")
      self.play(LaggedStart(*anims))
    combined = Rectangle(height=0.5, width=1.2/(w_scale), fill_opacity=0.5).next_to(epochs_2, LEFT, buff=0.02).set_color(color_grad).shift(0.3*RIGHT)
    with self.voiceover(text="""We didn't actually eliminate the overhead, we just blended it into our gpu code to hide it""") as trk:
      self.play(Transform(VGroup(overhead_r, epochs[0]), combined, replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""And the limiting factor is not going to be exactly equal to our data loading time since the cpu also
                        schedules gpu kernels and copies data into memory""") as trk:
      self.play(Write(Text("1+ ms", font_size=fs).set_color(color_grad).move_to(combined)))
    self.wait(1)
