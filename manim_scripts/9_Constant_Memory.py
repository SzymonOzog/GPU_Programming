from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import random
import math
from math import radians


class ConstantMemory(VoiceoverScene, ZoomedScene):
  def construct(self):
    self.set_speech_service(
        RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model="base")
        )

    title = Text("GPU programming", font_size=72)
    with self.voiceover(text="Hello and welcome to another episode in the series on GPU programming") as trk:
      self.play(Write(title))
    self.play(Unwrite(title))

    subtitle = Text("Constant Memory", font_size=48).next_to(title, DOWN)



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
    with self.voiceover(text="""This episode will be focused on constant memory, and when to use it to improve the 
                        performance of our code""") as trk:
      self.play(*[Create(r) for r in rects], *[Write(t) for t in texts], *[Create(a) for a in arrows])
      while trk.get_remaining_duration() > 1:
        self.play(*access_anims[random.randint(0, len(access_anims) -1)])

    with self.voiceover(text="""To recap what we learned in the previous episodes, constant memory is a read only memory region that 
                        lives in DRAM, it's also very limited, as we can only allocate 64KB for it""") as trk:
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
      self.wait(2)
      self.play(Indicate(VGroup(cc, cc_t), run_time=3))

    with self.voiceover(text="""And the second one is not really a benefit - more of a tradeoff""") as trk:
      pass
    fpcs = []
    side = 0.8
    buff = 0.08
    for i in range(4):
      for j in range(8):
        fpc = Rectangle(width=side, height=side, color=GREEN_C, fill_color=GREEN_C, fill_opacity=0.5, stroke_width=1)
        if j == 0:
          if i == 0:
            fpc.move_to(ps[0], aligned_edge=UL)
          else:
            fpc.next_to(fpcs[(i-1)*8], DOWN, aligned_edge=LEFT, buff=buff)
        else:
          fpc.next_to(fpcs[-1], RIGHT, buff=buff)
        fpcs.append(fpc)
    fpc_t = Text("Cores", font_size=72, color=GREEN_C).set_z_index(1).move_to(VGroup(*fpcs))

    cache = Rectangle(width=7, height=1, color=GOLD, fill_color=GOLD, fill_opacity=0.5).next_to(VGroup(*fpcs), DOWN)
    cache_t = Text("Cache", color=GOLD, font_size=40).move_to(cache)

    dram = Rectangle(width=7, height=1, color=RED, fill_color=RED, fill_opacity=0.5).next_to(cache, DOWN)
    dram_t = Text("DRAM", color=RED, font_size=40).move_to(dram)
    with self.voiceover(text="""To explain it we have to look at our simplified architecture again""") as trk:
      self.wait(1)
      self.play(Transform(VGroup(cc, l1), cache, replace_mobject_with_target_in_scene=True),
                Transform(VGroup(l1_t, cc_t), cache_t, replace_mobject_with_target_in_scene=True),
                Transform(VGroup(*ps), VGroup(*fpcs), replace_mobject_with_target_in_scene=True),
                Transform(VGroup(*p_ts), fpc_t, replace_mobject_with_target_in_scene=True),
                Transform(VGroup(*texs, rt), dram, replace_mobject_with_target_in_scene=True),
                Transform(VGroup(*tex_ts, rt_t), dram_t, replace_mobject_with_target_in_scene=True))

    mem = Rectangle(width=side, height=side, color=RED, fill_color=RED, fill_opacity=0.5).move_to(cache, aligned_edge=UL).shift(0.1*DR)
    mem_broadcast = [mem.copy().move_to(x) for x in fpcs]
    with self.voiceover(text="""When we load a value from constant memory <bookmark mark='1'/> we broadcast the <bookmark mark='2'/>value across all threads""") as trk:
      self.wait_until_bookmark("1")
      self.play(FadeIn(mem, target_position=dram))
      self.wait_until_bookmark("2")
      self.play(*[FadeIn(m, target_position=mem) for m in mem_broadcast])
    self.wait(1)

    with self.voiceover(text="""This works to our advantage when all the threads will use the value""") as trk:
      pass

    with self.voiceover(text="""But if even one thread needs to access a different value from our memory <bookmark mark='1'/>we have to pay the broadcasting cost again""") as trk:
      self.wait_until_bookmark("1")
      mem2_broadcast = [mem.copy().move_to(x) for x in fpcs]
      self.play(*[FadeIn(m, target_position=mem) for m in mem2_broadcast])
       
    with self.voiceover(text="""When that memory is adjacent to the memory access that we did in other threads it's probably not that bad
                        as values in the constant cache are loaded by blocks so we don't have to pay the DRAM access cost""") as trk:
      pass

    with self.voiceover(text="""But if we were to load a memory address from a different memory region the cost starts growing rapidly""") as trk:
      mem2 = Rectangle(width=side, height=side, color=RED, fill_color=RED, fill_opacity=0.5).next_to(mem, RIGHT, buff=0.1)
      self.play(FadeIn(mem2, target_position=dram))
      self.wait(2)
      mem2_broadcast = [mem2.copy().move_to(x) for x in fpcs]
      self.play(*[FadeIn(m, target_position=mem2) for m in mem2_broadcast])

    with self.voiceover(text="""Let's look at and benchmark an example of this""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
    code_vars = """#define CONST_SIZE 16384
#define ACCESSES 10
__constant__ float c_mem[CONST_SIZE];
"""

    code_const = """__global__ void add_const
    (int n , float* a, float* c, int accesses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.x % accesses;
    if (i < n-ACCESSES)
    {
      for(int x = 0; x<ACCESSES; x++)
      {
        c[i] = a[i] + c_mem[j+x];
        }
      }
    } """

    code_global = """__global__ void add
    (int n , float* a, float* b, float* c, int accesses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = threadIdx.x % accesses;
    if (i < n-ACCESSES)
    {
      for(int x = 0; x<ACCESSES; x++)
      {
        c[i] = a[i] + b[j+x];
        }
      }
    }"""

    code_obj = Code(code=code_const, tab_width=2, language="c", font_size=14, background="rectangle", line_no_buff=1, corner_radius=0.1, margin=0.1, insert_line_no=False).shift(DOWN).scale(1.2)
    code_obj.code = remove_invisible_chars(code_obj.code)
    code_obj2 = Code(code=code_vars, tab_width=2, language="c", font_size=14, background="rectangle", line_no_buff=0, corner_radius=0.1, margin=0.1, insert_line_no=False).scale(1.2).next_to(code_obj, UP)
    code_obj2.code[:2].set_color(GREEN_E)
    code_obj3 = Code(code=code_global, tab_width=2, language="c", font_size=14, background="rectangle", line_no_buff=0, corner_radius=0.1, margin=0.1, insert_line_no=False).shift(DOWN).scale(1.2).to_edge(RIGHT)
    code_obj3.code = remove_invisible_chars(code_obj3.code)
    with self.voiceover(text="""We are going to use a code that uses the maximum amount of constant memory that we can use""") as trk:
      self.play(Create(code_obj2))

    with self.voiceover(text="""And the kernel just adds 2 numbers together, one in global and one in constant memory - and it does that 10 times.
                        The reason for that is that I've found that one access is too short and tends to give very noisy results""") as trk:
      self.play(Create(code_obj))

    
    hl = SurroundingRectangle(code_obj.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""We are also passing in as a parameter the number of accesses per warp - so when it's one all threads in a warp
                        use the same memory, if it's two half the threads access the same memory etc...""") as trk:
      self.play(Create(hl))

    with self.voiceover(text="""And we are going to do the same thing the second time, but this time the second vector is also in global memory""") as trk:
      self.play(Uncreate(hl))
      self.play(code_obj.animate.to_edge(LEFT))
      self.play(Create(code_obj3))

    self.play(*[FadeOut(x) for x in self.mobjects])

    ratios_by_access = [[0.899, 0.870, 0.856, 0.883, 0.843, 0.834, 0.846, 0.803, 0.828, 0.833, ],
                       [0.956, 0.952, 0.901, 0.862, 0.896, 0.842, 0.835, 0.826, 0.823, 0.835, ],
                       [0.943, 0.902, 0.892, 0.895, 0.861, 0.870, 0.782, 0.863, 0.844, 0.854, ],
                       [0.932, 0.974, 0.947, 0.937, 0.942, 0.987, 0.914, 0.954, 0.952, 0.933, ],
                       [1.009, 1.048, 1.005, 1.059, 1.064, 1.139, 1.022, 1.081, 1.069, 1.059, ],
                       [1.083, 1.129, 1.174, 1.187, 1.193, 1.261, 1.164, 1.215, 1.189, 1.174, ],
                       [1.160, 1.213, 1.272, 1.323, 1.344, 1.433, 1.318, 1.372, 1.324, 1.308, ],
                       [1.186, 1.312, 1.366, 1.462, 1.449, 1.579, 1.457, 1.473, 1.446, 1.424, ],
                       [1.254, 1.362, 1.500, 1.585, 1.537, 1.707, 1.600, 1.602, 1.551, 1.542, ],
                       [1.291, 1.311, 1.553, 1.692, 1.773, 1.775, 1.749, 1.737, 1.711, 1.705, ]]
    rng = list(range(15, 25))
    nums = [2**x for x in rng]

    ax = Axes(
        x_range=[rng[0], rng[-1], 2],
        y_range=[0.7, 1.5, 0.1],
        x_axis_config={"scaling": LogBase(2)},
        axis_config={"include_numbers": True}).scale(0.9)


    one_access_graph = ax.plot_line_graph(
        x_values=nums,
        y_values=ratios_by_access[0],
        line_color=BLUE,
        add_vertex_dots=False
    )
    with self.voiceover(text="""Let's firs look at the result of doing one access across all threads""") as trk:
      self.play(Create(ax))

    x_l = ax.get_x_axis_label("N")
    y_l = ax.get_y_axis_label("\\frac{Const}{Global}")
    with self.voiceover(text="""As with previous benchmarks, we are checking for different input sizes""") as trk:
      self.play(Write(x_l), Write(y_l))

    with self.voiceover(text="""And we can see that when all threads in the warp access the same location we are getting about a 15% speed increase over global memory""") as trk:
      self.play(Create(one_access_graph))
    ax2 = ThreeDAxes(
        x_range=[rng[0], rng[-1], 2],
        y_range=[0.7, 2, 0.2],
        z_range=[-16, 0, 2],
        x_axis_config={"scaling": LogBase(2)},
        z_axis_config={"include_numbers": False},
        axis_config={"include_numbers": True, "include_tip": False})
    ax2.scale(0.6)
    ax2.rotate(radians(-25), axis=UP)
    ax2.rotate(radians(15), axis=RIGHT)
    
    def interp(u, v, r=ratios_by_access):
      ul, ur = math.floor(u), math.ceil(u)
      vl, vr = math.floor(v), math.ceil(v)
      alpha = u-ul
      start = r[vl][ul] * (1-alpha) + r[vl][ur] * alpha
      end = r[vr][ul] * (1-alpha) + r[vr][ur] * alpha
      alpha = v-vl
      ret = start * (1-alpha) + end*alpha
      return ret

    points = []
    for j, ratio in enumerate(ratios_by_access):
      p = []
      for i, num in enumerate(nums[:-1]):
        p.append(ax2.c2p(num, ratio[i], -j))
      points.append(p)

    multiple_access_graph = Surface(lambda u, v, points=points: interp(u,v,points),
                           u_range=(0,len(points[0])-1),
                           v_range=(0,len(points)-1),
                           fill_color=BLUE,
                           stroke_color=BLUE,
                           stroke_width=4,
                           fill_opacity=0.5,
                           stroke_opacity=0,
                           resolution=(16,16), 
                           checkerboard_colors=None)

    with self.voiceover(text="""We can now add another dimension to our graph that represents our amount of accesses per warp""") as trk:
      self.play(Transform(VGroup(ax, x_l, y_l), ax2, replace_mobject_with_target_in_scene=True),
                Transform(one_access_graph, multiple_access_graph, replace_mobject_with_target_in_scene=True))
    
    one_points = [[ax2.c2p(min(nums), 1, -16), ax2.c2p(max(nums), 1, -16)],
                   [ax2.c2p(min(nums), 1, 0), ax2.c2p(max(nums), 1, 0)]]

    one_surf = Surface(lambda u, v, points=one_points: interp(u,v,points),
                       u_range=(0, 1),
                       v_range=(0, 1),
                       resolution=(16, 16), 
                       checkerboard_colors=False,
                       fill_color=GREEN,
                       fill_opacity=0.5)
    with self.voiceover(text="""With the green surface showing a 1 to 1 ratio of const to global usage""") as trk:
      self.play(Create(one_surf))

    with self.voiceover(text="""This graph is getting a bit hard to read and reason about""") as trk:
      pass

    mean_r = np.mean(ratios_by_access, axis=1)
    mean_graph = ax2.plot_line_graph(np.ones(mean_r.shape) * nums[0], mean_r, list([-x for x in range(1, 16)]), line_color=BLUE, add_vertex_dots=False)
    one_graph = ax2.plot_line_graph([nums[0], nums[0]], [1, 1], [0, -16], line_color=GREEN, add_vertex_dots=False)
    ax3 = Axes(
        x_range=[1, 11, 1],
        y_range=[0.7, 2, 0.2],
        axis_config={"include_numbers": True}).scale(0.8)

    mean_graph2 = ax3.plot_line_graph(list(range(1, 11)), mean_r, line_color=BLUE, add_vertex_dots=False)
    one_graph2 = ax3.plot_line_graph([1, 10], [1, 1], line_color=GREEN, add_vertex_dots=False)

    x_l = ax3.get_x_axis_label("\\frac{Accesses}{Warp}")
    y_l = ax3.get_y_axis_label("\\frac{Const}{Global}")
    with self.voiceover(text="""So what I'm going to do is average the ratios <bookmark mark='1'/> for different input sizes""") as trk:
      self.wait_until_bookmark("1")
      self.play(Transform(multiple_access_graph, mean_graph, replace_mobject_with_target_in_scene=True),
                Transform(one_surf, one_graph, replace_mobject_with_target_in_scene=True),
              ax2.get_x_axis().animate.scale(0.00001).move_to(ax2.c2p(nums[0], 0.7, 0)))

    with self.voiceover(text="""We can now see that it's around 4 accesses per warp when we are getting no improvements over global, and 
                        our performance worsening beyond that point""") as trk:
      self.play(Transform(ax2.get_y_axis(), ax3.get_y_axis(), replace_mobject_with_target_in_scene=True),
                Transform(ax2.get_z_axis(), ax3.get_x_axis(), replace_mobject_with_target_in_scene=True),
                Transform(mean_graph, mean_graph2, replace_mobject_with_target_in_scene=True),
                Transform(one_graph, one_graph2, replace_mobject_with_target_in_scene=True))
      self.play(Write(x_l), Write(y_l))

    self.wait(2)
    self.play(*[FadeOut(x) for x in self.mobjects])

    code_const = """__global__ void add_const
    (int n , float* a, float* c, int distance)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = (threadIdx.x * dist) % CONST_SIZE 
  if (i < n-ACCESSES)
  {
    for(int x = 0; x<ACCESSES; x++)
    {
      c[i] = a[i] + c_mem[j+x];
    }
  }
} """
    code_global = """__global__ void add
    (int n , float* a, float* b, float* c, int distance)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = (threadIdx.x * dist) % CONST_SIZE 
  if (i < n-ACCESSES)
  {
    for(int x = 0; x<ACCESSES; x++)
    {
      c[i] = a[i] + b[j+x];
    }
  }
}"""
    with self.voiceover(text="""Let's run a second benchmark""") as trk:
      self.play(*[FadeIn(x) for x in [code_obj, code_obj2, code_obj3]])
    code_obj_t = Code(code=code_const, tab_width=2, language="c", font_size=14, background="rectangle", line_no_buff=1, corner_radius=0.1, margin=0.1, insert_line_no=False).shift(DOWN).scale(1.2).to_edge(LEFT)
    code_obj_t.code = remove_invisible_chars(code_obj.code)
    code_obj3_t = Code(code=code_global, tab_width=2, language="c", font_size=14, background="rectangle", line_no_buff=0, corner_radius=0.1, margin=0.1, insert_line_no=False).shift(DOWN).scale(1.2).to_edge(RIGHT)
    code_obj3_t.code = remove_invisible_chars(code_obj3.code)

    with self.voiceover(text="""This time, instead of passing the amout of accesses per warp, each thread in a warp will access a different memory region
                        and we are controlling how far away from each other those accesses are""") as trk:
      self.play(Transform(code_obj, code_obj_t), Transform(code_obj3, code_obj3_t))
      hl = SurroundingRectangle(code_obj.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3)
      hl2 = SurroundingRectangle(code_obj.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3)
      self.play(Create(hl), Create(hl2))

    self.play(*[FadeOut(x) for x in self.mobjects])

    ratios = [[3.333, 3.611, 4.627, 5.473, 5.889, 6.397, 6.574, 6.616, 6.523, 6.388, ],
              [3.351, 4.353, 5.549, 6.166, 6.698, 7.109, 7.175, 7.093, 7.046, 6.915, ],
              [3.460, 4.449, 5.961, 6.765, 7.545, 8.242, 8.165, 8.057, 7.978, 7.797, ],
              [4.192, 5.361, 6.408, 7.108, 7.608, 7.783, 7.776, 7.772, 7.650, 7.573, ],
              [3.905, 5.834, 7.850, 9.803, 10.905, 11.477, 11.717, 11.632, 11.559, 11.283, ],
              [5.540, 7.461, 9.423, 10.744, 11.842, 12.479, 12.287, 12.128, 12.012, 11.778, ],
              [6.628, 9.090, 11.780, 13.643, 14.672, 15.559, 15.566, 15.501, 15.355, 15.191, ],
              [6.517, 7.950, 9.043, 10.237, 11.030, 11.233, 11.195, 11.129, 10.981, 10.953, ],
              [7.763, 10.835, 12.008, 16.066, 18.098, 18.928, 19.117, 19.047, 18.984, 18.778, ],
              [8.520, 11.465, 14.235, 17.422, 19.241, 20.031, 20.148, 20.113, 20.001, 19.810, ],
              [8.580, 11.444, 14.832, 18.860, 21.047, 22.345, 22.554, 22.449, 22.281, 22.042, ],
              [10.232, 12.840, 15.117, 17.817, 19.403, 19.898, 19.939, 19.901, 19.584, 19.631, ],
              [9.356, 13.018, 15.495, 18.419, 20.506, 20.758, 20.724, 20.428, 20.113, 20.052, ],
              [10.885, 13.043, 15.410, 17.506, 19.317, 19.537, 19.680, 19.817, 19.424, 19.467, ],
              [11.166, 13.586, 14.967, 16.140, 18.034, 18.246, 18.420, 18.412, 18.508, 18.566, ],
              [10.541, 12.836, 14.552, 15.821, 16.914, 17.017, 17.081, 16.773, 16.730, 16.680, ]]
    rng = list(range(15, 25))
    nums = [2**x for x in rng]

    ax = Axes(
        x_range=[rng[0], rng[-1], 2],
        y_range=[1, 7, 1],
        x_axis_config={"scaling": LogBase(2)},
        axis_config={"include_numbers": True}).scale(0.9)

    x_l = ax.get_x_axis_label("N")
    y_l = ax.get_y_axis_label("\\frac{Const}{Global}")


    one_access_graph = ax.plot_line_graph(
        x_values=nums,
        y_values=ratios[0],
        line_color=BLUE,
        add_vertex_dots=False
    )

    with self.voiceover(text="""If we look at the graph, we are getting a 3-6x worse performance even when all of the threads access a consecutive memory address""") as trk:
      self.play(Create(ax))
      self.play(Write(x_l), Write(y_l))
      self.play(Create(one_access_graph))

    ax2 = ThreeDAxes(
        x_range=[rng[0], rng[-1], 2],
        y_range=[1, 22, 2],
        z_range=[-16, 0, 2],
        x_axis_config={"scaling": LogBase(2), "include_numbers": True},
        z_axis_config={"include_numbers": False},
        axis_config={"include_numbers": True, "include_tip": False})
    ax2.scale(0.6)
    ax2.rotate(radians(-25), axis=UP)
    ax2.rotate(radians(15), axis=RIGHT)
    
    points = []
    for j, ratio in enumerate(ratios):
      p = []
      for i, num in enumerate(nums[:-1]):
        p.append(ax2.c2p(num, ratio[i], -j))
      points.append(p)

    multiple_access_graph = Surface(lambda u, v, points=points: interp(u,v,points),
                           u_range=(0,len(points[0])-1),
                           v_range=(0,len(points)-1),
                           fill_color=BLUE,
                           fill_opacity=1,
                           resolution=(16,16), 
                           checkerboard_colors=None)

    with self.voiceover(text="""If we look at the performance as we make our threads access locations further away from each other""") as trk:
      self.play(Transform(VGroup(ax, x_l, y_l), ax2, replace_mobject_with_target_in_scene=True),
                Transform(one_access_graph, multiple_access_graph, replace_mobject_with_target_in_scene=True))
    
    mean_r = np.mean(ratios, axis=1)
    mean_graph = ax2.plot_line_graph(np.ones(mean_r.shape) * nums[0], mean_r, list([-x for x in range(1, 16)]), line_color=BLUE, add_vertex_dots=False)

    ax3 = Axes(
        x_range=[1, 17, 1],
        y_range=[1, 22, 2],
        axis_config={"include_numbers": True}).scale(0.8)

    with self.voiceover(text="""And we average them out for readability""") as trk:
      self.play(Transform(multiple_access_graph, mean_graph, replace_mobject_with_target_in_scene=True),
                ax2.get_x_axis().animate.scale(0.00001).move_to(ax2.c2p(nums[0], 0.7, 0)))
      mean_graph2 = ax3.plot_line_graph(list(range(1, 17)), mean_r, line_color=BLUE, add_vertex_dots=False)
      self.play(Transform(ax2.get_y_axis(), ax3.get_y_axis(), replace_mobject_with_target_in_scene=True),
                Transform(ax2.get_z_axis(), ax3.get_x_axis(), replace_mobject_with_target_in_scene=True),
                Transform(mean_graph, mean_graph2, replace_mobject_with_target_in_scene=True))

      x_l = ax3.get_x_axis_label("Distance")
      y_l = ax3.get_y_axis_label("\\frac{Const}{Global}")
      self.play(Write(x_l), Write(y_l))

    with self.voiceover(text="""We can see that we can get up to 18 times worse performance!""") as trk:
      pass

    with self.voiceover(text="""This is where we finish the episode on constant memory. As you can see it's not an easy one to use so be sure 
                        that you understand it's underlying principles and the why and when to use it""") as trk:
      pass
    
    bmac = Text("https://buymeacoffee.com/simonoz", font_size=48, color=YELLOW)
    alex = Text("Alex", font_size=60).next_to(bmac, DOWN)
    unknown = Text("Anonymous x2", font_size=60).next_to(alex, DOWN)
    with self.voiceover(text="""I'm hosting a buy me a coffe for those that want to support this channel. A shoutout to Alex and two anonymous donors that supported so far""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
      self.camera.auto_zoom(VGroup(bmac, alex, unknown), margin=4, animate=False)
      self.play(Write(bmac))
      self.play(Write(alex))
      self.play(Write(unknown))

    with self.voiceover(text="""But you can always support me for free by subscribing, leaving a like, commenting and sharing this video with your friends""") as trk:
      pass

    with self.voiceover(text="""I'll see you in the next episode, bye""") as trk:
      pass

    self.play(*[FadeOut(x) for x in self.mobjects])
    self.wait(2)
