from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import random
import math


class ConstantMemory(VoiceoverScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        )

    title = Text("GPU programming", font_size=72)
    with self.voiceover(text="Hello and welcome to another episode in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Constant Memory", font_size=48).next_to(title, DOWN)
    with self.voiceover(text="""This episode will be focused on constant memory, and when to use it to improve the 
                        performance of our code""") as trk:
      self.play(Write(subtitle))

    self.play(Unwrite(title), Unwrite(subtitle))

    def join(r1, r2, start, double=True):
      nonlocal arrows
      e_y = r2.get_y() + (1 if r2.get_y() < start[1] else -1) * r2.height/2
      end = np.array([start[0], e_y, 0])
      ret = None
      if double: 
        ret = DoubleArrow(start, end, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
      else:
        ret = Arrow(end, start, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
      arrows.append(ret)
      return ret

    shared_store = []
    shared_load = []
    register_store = []
    register_load = []
    local_store = []
    local_load = []
    global_store = []
    global_load = []
    constant_load = []

    thread_objs = []
    rects = []
    texts = []
    arrows = []
    def make_thread(idx=0):
      nonlocal thread_objs, rects, texts
      thread = Rectangle(height=0.5, width=2.2, color=BLUE)
      texts.append(Text(f"Thread {idx}", font_size=15, color=BLUE))
      rects.append(thread)
      thread.add(texts[-1])

      registers = Rectangle(height=0.5, width=1.0, color=GREEN).next_to(thread, UP, aligned_edge=LEFT, buff=0.5)
      texts.append(Text("Registers", font_size=15, color=GREEN).move_to(registers.get_center()))
      registers.add(texts[-1])
      rects.append(registers)

      local = Rectangle(height=0.5, width=1.0, color=RED_A).next_to(thread, UP, aligned_edge=RIGHT, buff=0.5)
      l = Text("Local", font_size=15, color=RED_A)
      m = Text("Memory", font_size=15, color=RED_A)
      VGroup(l, m).arrange(DOWN, buff=0.05).move_to(local.get_center())
      texts.append(l)
      texts.append(m)
      rects.append(local)
      local.add(l)
      local.add(m)

      t_group = VGroup(thread, registers, local)
      t_group.add(join(registers, thread, start=registers.get_corner(DOWN)))
      t_group.add(join(local, thread, start=local.get_corner(DOWN)))

      thread_objs.append(thread)
      return t_group

    def make_block(idx=0):
      nonlocal rects, texts
      block = Rectangle(height=3.5, width=5.0, color=PURPLE)
      rects.append(block)

      threads = VGroup(make_thread(0), make_thread(1)).arrange(RIGHT).shift(0.8*DOWN)
      block.add(threads)

      shared_mem = Rectangle(width=4.0, height=0.5, color=YELLOW).next_to(threads, UP)
      rects.append(shared_mem)
      block.add(shared_mem)

      texts.append(Text(f"Shared Memory", font_size=15, color=YELLOW).move_to(shared_mem.get_center()))
      shared_mem.add(texts[-1])
      for t in thread_objs[idx*2:]:
        block.add(join(t, shared_mem, t.get_corner(UP)))
      texts.append(Text(f"Block {idx}", color=PURPLE).next_to(shared_mem, UP))
      shared_mem.add(texts[-1])
      
      return block

    blocks = VGroup(make_block(0), make_block(1)).arrange(RIGHT).shift(UP)

    constant = Rectangle(width=blocks.width, height=1, color=RED_B).next_to(blocks, DOWN)
    texts.append(Text("Constant Memory", font_size=30, color=RED_B).move_to(constant.get_center()))
    rects.append(constant)

    gmem = Rectangle(width=blocks.width, height=1, color=RED).next_to(constant, DOWN)
    rects.append(gmem)
    texts.append(Text("Global Memory", font_size=30, color=RED).move_to(gmem.get_center()))

    subobjects = []
    queue = [blocks]
    while queue:
      o = queue.pop()
      subobjects.append(o)
      queue.extend(o.submobjects)


    for mo in subobjects:
      for so in mo.submobjects.copy():
        if any(so in x for x in [rects, texts, arrows, thread_objs]):
          mo.remove(so)

    for t in thread_objs[:2]:
      join(t, constant, t.get_corner(DOWN+LEFT)+RIGHT*0.2, False)
      join(t, gmem, t.get_corner(DOWN+LEFT))

    for t in thread_objs[2:]:
      join(t, constant, t.get_corner(DOWN+RIGHT)+LEFT*0.2, False)
      join(t, gmem, t.get_corner(DOWN+RIGHT))

    for i in [1, 3, 7, 9]:
      local_store.append(ShowPassingFlash(Arrow(start=arrows[i].get_end(), end=arrows[i].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      local_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
    for i in [0, 2, 6, 8]:
      register_store.append(ShowPassingFlash(Arrow(start=arrows[i].get_end(), end=arrows[i].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      register_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
    for i in [4, 5, 10, 11]:
      shared_store.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      shared_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_end(), end=arrows[i].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
    for i in [13, 15, 17, 19]:
      global_store.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      global_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_end(), end=arrows[i].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
    for i in [12, 14, 16, 18]:
      constant_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))

    access_anims = [shared_store, shared_load, register_store, register_load, local_store, local_load, global_store, global_load, constant_load]

    with self.voiceover(text="""To recap what we learned in the previous episodes, constant memory is a read only memory region that 
                        lives in DRAM, it's also very limited, as we can only allocate 64KB for it""") as trk:
      self.play(*[Create(r) for r in rects], *[Write(t) for t in texts], *[Create(a) for a in arrows])
      while trk.get_remaining_duration() > 1:
        self.play(*access_anims[random.randint(0, len(access_anims) -1)])

    l1 = Rectangle(height=0.5, width=7, color=GOLD_A, fill_color=GOLD_A, fill_opacity=0.5).shift(1.7*DOWN)
    l1_t = Text("128KB L1 Cache / Shared Memory", color=GOLD_A, font_size=28).move_to(l1)

    cc = Rectangle(height=0.5, width=7, color=GOLD_E, fill_color=GOLD_E, fill_opacity=0.5).next_to(l1, UP, buff=0.1)
    cc_t = Text("8KB Constant Cache", color=GOLD_E, font_size=28).move_to(cc)

    rt = Rectangle(width=2.5, height=0.7, fill_opacity=0.5).shift(2.4*DOWN + 2.25*RIGHT)
    rt_t = Text("RT Core", font_size=32).move_to(rt)

    texs = []
    tex_ts = []
    for i in range(4):
      if i == 0:
        texs.append(Rectangle(height=0.7, width=1, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.5).shift(2.4*DOWN + 3*LEFT))
      else:
        texs.append(Rectangle(height=0.7, width=1, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.5).next_to(texs[-1], RIGHT, buff=0.1))
      tex_ts.append(Text("TEX", font_size=32, color=BLUE_E).move_to(texs[-1]))

    ps = []
    p_ts = []

    for i in range(4):
      if i == 0:
        ps.append(Rectangle(height=3.6, width=1.675, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.5).shift(2.65*LEFT + 1.05*UP))
      else:
        ps.append(Rectangle(height=3.6, width=1.675, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.5).next_to(ps[-1], RIGHT, buff=0.093))
      p_ts.append(Text("Processing Block", font_size=32, color=GREEN_A).move_to(ps[-1]).rotate(PI/2))

    self.play(*[Uncreate(r) for r in rects], *[Unwrite(t) for t in texts], *[Uncreate(a) for a in arrows])
    with self.voiceover(text="""To talk about the benefits and downsides of constant memory we have to remind ourselves of the structure
                        of a streaming multiprocessor. If you didn't want the Modern GPU Architecture episode it might be good to go through it 
                        if you feel lost""") as trk:
      self.play(LaggedStart(*[Create(x) for x in texs + ps + [l1, cc, rt]], *[Write(t) for t in tex_ts + p_ts + [l1_t, cc_t, rt_t]]))


    with self.voiceover(text="""The first benefit is that it's cached in a different cache than global memory, this means that by using constant memory
                        we are actually freeing space in our L1 for other variables""") as trk:
      self.play(Indicate(VGroup(cc, cc_t)))

    with self.voiceover(text="""And the second one is a double edged sword""") as trk:
      pass

    ratios = [[0.978, 0.999, 0.993, 0.993, 0.988, 0.988, 0.997, 0.997, 0.994, 0.990, 0.980, 0.973, 0.967, 0.958],
              [1.215, 1.209, 1.475, 1.231, 1.234, 1.249, 1.219, 1.132, 1.281, 1.509, 1.850, 2.226, 2.593, 2.890],
              [1.346, 1.354, 1.585, 1.389, 1.352, 1.304, 1.288, 1.235, 1.492, 1.872, 2.408, 2.990, 3.642, 4.067],
              [1.440, 1.484, 1.222, 1.485, 1.487, 1.348, 1.241, 1.214, 1.459, 2.002, 2.652, 3.342, 4.048, 4.568],
              [1.684, 1.743, 1.428, 1.814, 1.734, 1.564, 1.442, 1.425, 1.881, 2.619, 3.607, 4.900, 5.899, 6.795],
              [1.750, 1.765, 1.439, 1.789, 1.808, 1.512, 1.361, 1.336, 1.860, 2.645, 3.728, 4.943, 6.187, 7.138],
              [1.903, 2.125, 1.906, 1.907, 1.914, 1.618, 1.438, 1.442, 2.044, 2.950, 4.378, 5.867, 7.377, 8.536],
              [2.034, 2.263, 2.029, 2.060, 2.037, 1.694, 1.536, 1.550, 2.223, 3.187, 4.760, 6.466, 8.259, 9.767],
              [2.173, 2.410, 2.173, 2.202, 2.219, 1.810, 1.650, 1.645, 2.428, 3.690, 5.483, 7.506, 9.474, 11.099],
              [2.247, 2.597, 2.339, 2.358, 2.344, 1.899, 1.756, 1.820, 2.676, 4.100, 6.053, 8.363, 10.677, 12.541],
              [2.479, 2.735, 2.508, 2.549, 2.495, 2.004, 1.776, 1.833, 2.802, 4.362, 6.602, 9.169, 11.741, 13.767],
              [2.599, 2.924, 2.660, 2.684, 2.686, 2.125, 1.862, 1.998, 3.036, 4.750, 7.153, 10.032, 12.875, 15.230],
              [2.821, 3.098, 2.867, 2.855, 2.885, 2.226, 2.043, 2.170, 3.298, 5.206, 8.010, 11.056, 14.271, 16.771],
              [2.990, 3.220, 2.988, 3.010, 3.012, 2.307, 2.137, 2.306, 3.530, 5.585, 8.622, 11.787, 15.383, 18.006],
              [3.110, 3.330, 3.154, 3.183, 3.163, 2.411, 2.227, 2.440, 3.745, 6.048, 9.305, 12.892, 16.600, 19.524],
              [3.254, 3.548, 3.326, 3.319, 3.342, 2.536, 2.334, 2.613, 4.045, 6.430, 9.934, 13.862, 17.787, 20.838],
              [3.382, 3.719, 3.471, 3.483, 3.489, 2.620, 2.409, 2.742, 4.192, 6.498, 9.718, 12.865, 15.833, 17.880]]

    rng = list(range(10, 25))
    nums = [2**x for x in rng]

    ax = Axes(
        x_range=[rng[0], rng[-1], 2],
        y_range=[0.7, 1.5, 0.1],
        x_axis_config={"scaling": LogBase(2)},
        axis_config={"include_numbers": False}).scale(0.9)


    block_graph = ax.plot_line_graph(
        x_values=nums,
        y_values=ratios[0],
        line_color=BLUE,
        add_vertex_dots=False
    )
    self.play(Create(ax))
    self.play(Create(block_graph))

    ax2 = ThreeDAxes(
        x_range=[rng[0], rng[-1], 2],
        y_range=[0,16,2],
        z_range=[0,16,2],
        x_axis_config={"scaling": LogBase(2)},
        ).scale(0.5)
    
    def interp(u, v, r=ratios):
      ul, ur = math.floor(u), math.ceil(u)
      vl, vr = math.floor(v), math.ceil(v)
      alpha = u-ul
      start = r[vl][ul] * (1-alpha) + r[vl][ur] * alpha
      end = r[vr][ul] * (1-alpha) + r[vr][ur] * alpha
      alpha = v-vl
      ret = start * (1-alpha) + end*alpha
      return ret

    points = []
    for j, ratio in enumerate(ratios):
      p = []
      for i, num in enumerate(nums[:-1]):
        p.append(ax2.c2p(num, ratio[i], j))
      points.append(p)

    block_graph2 = Surface(lambda u, v, points=points: interp(u,v,points),
                           u_range=(0,len(points[0])-1),
                           v_range=(0,len(points)-1),
                           resolution=(16,16), 
                           checkerboard_colors=False)


    self.play(Transform(ax, ax2, replace_mobject_with_target_in_scene=True),
              Transform(block_graph, block_graph2, replace_mobject_with_target_in_scene=True))
    
    one_points = [[ ax2.c2p(min(nums), 1, 16), ax2.c2p(max(nums), 1, 16)],
                   [ax2.c2p(min(nums), 1, 1), ax2.c2p(max(nums), 1, 1)]]

    one_surf = Surface(lambda u, v, points=one_points: interp(u,v,points),
                       u_range=(0, 1),
                       v_range=(0, 1),
                       resolution=(16, 16), 
                       checkerboard_colors=False,
                       fill_color=GREEN,
                       fill_opacity=0.5)

    with self.voiceover(text="""There is a wonderfull blogpost by Lei Mao that profiled the different usecases for constant memory
                        to show when to use it and when not to""") as trk:
      pass

    code_vars = """constexpr unsigned int N{64U * 1024U / sizeof(float)};
__constant__ float const_values[N];
constexpr unsigned int magic_number{1357U};"""

    code_const = """__global__ void add_constant(float* sums, float* inputs,
                             int num_sums, Access access)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int index = 0;
  switch (access)
  {
    case AccessPattern::OneAccessPerBlock:
      index = blockIdx.x % N;
      break;
    case AccessPattern::OneAccessPerThread:
      index = i % N;
      break;
    case AccessPattern::PseudoRandom:
      index = (i * magic_number) % N;
      break;
  }

  if (i < num_sums)
  {
    sums[i] = inputs[i] + const_values[index];
  }
}"""
    code_global = """__global__ void add_global(float* sums, float* inputs, float* values,
                           int num_sums, Access access)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int index{0U};
  switch (access)
  {
    case AccessPattern::OneAccessPerBlock:
      index = blockIdx.x % N;
      break;
    case AccessPattern::OneAccessPerThread:
      index = i % N;
      break;
    case AccessPattern::PseudoRandom:
      index = (i * magic_number) % N;
      break;
  }

  if (i < num_sums)
  {
    sums[i] = inputs[i] + values[index];
  }
}"""
    code_obj = Code(code=code_const, tab_width=2, language="c", font_size=12, background="rectangle", line_no_buff=0, corner_radius=0.1, margin=0.1, insert_line_no=False).shift(DOWN)
    code_obj.code = remove_invisible_chars(code_obj.code)
    code_obj2 = Code(code=code_vars, tab_width=2, language="c", font_size=12, background="rectangle", line_no_buff=0, corner_radius=0.1, margin=0.1, insert_line_no=False).next_to(code_obj, UP)
    code_obj2.code = remove_invisible_chars(code_obj2.code)
    with self.voiceover(text="""It checks the timings for three different access patterns""") as trk:
      self.play(Create(code_obj))
      self.play(Create(code_obj2))

    hl = SurroundingRectangle(code_obj.code[7:10], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""One where each block acesses only one memory address""") as trk:
      self.play(Create(hl))

    hl_t = SurroundingRectangle(code_obj.code[10:13], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""One where each thread in a block accessess a consecutive memory address""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[13:16], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And the last one where each thread accesses a random memory address""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[20], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Finally, we add the number to some input vector residing in global memory""") as trk:
      self.play(Transform(hl, hl_t))

    with self.voiceover(text="""We run the experiment twice """) as trk:
      self.play(Uncreate(hl), code_obj.animate.to_edge(LEFT))
      code_obj3 = Code(code=code_global, tab_width=2, language="c", font_size=12, background="rectangle", line_no_buff=0, corner_radius=0.1, margin=0.1, insert_line_no=False).shift(DOWN).to_edge(RIGHT)
      code_obj3.code = remove_invisible_chars(code_obj3.code)
      self.play(Create(code_obj3))

    hl = SurroundingRectangle(code_obj2.code[1], BLUE, fill_color=BLUE, buff=0.03, stroke_width=2, fill_opacity=0.3)
    hl2 = SurroundingRectangle(code_obj.code[20][-20:], BLUE, fill_color=BLUE, buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Once when the second vector resides in constant memory""") as trk:
      self.play(Create(hl), Create(hl2))

    hl3 = SurroundingRectangle(code_obj3.code[20][-14:], buff=0.03, stroke_width=2, fill_opacity=0.3)
    hl4 = SurroundingRectangle(code_obj3.code[0][-13:-1], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And once when it resides in the global memory""") as trk:
      self.play(Create(hl3), Create(hl4))

    self.wait(1)

    header = Text("Modifications", font_size=72).shift(UP)
    modifications = BulletedList("Change datatype to float",  "Clear L2 cache", "Multiple input sizes", font_size=48).next_to(header, DOWN)
    with self.voiceover(text="""Inspired by this I expanded the work by changing the datatype from int to float as GPU's 
                        are optimized for floating point operations""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
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
