from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np


class MemoryHierarchy(VoiceoverScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 6 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Memory Hierarchy", font_size=48).next_to(title, DOWN)
    with self.voiceover(text="""In this episode, we are going to go over the memory hierarchy of our gpu as understanding
                        it will be crucial to getting the best performence our of our hardware""") as trk:
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

      local = Rectangle(height=0.5, width=1.0, color=GREEN_A).next_to(thread, UP, aligned_edge=RIGHT, buff=0.5)
      l = Text("Local", font_size=15, color=GREEN_A)
      m = Text("Memory", font_size=15, color=GREEN_A)
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

    constant = Rectangle(width=blocks.width, height=1, color=YELLOW).next_to(blocks, DOWN)
    texts.append(Text("Constant Memory", font_size=30, color=YELLOW).move_to(constant.get_center()))
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


    with self.voiceover(text="""When speaking about memory hierarchy we have to take each unit into consideration, that is <bookmark mark='1'/>our blocks
                        <bookmark mark='2'/> and the threads that run inside our blocks.""") as trk:
      self.wait_until_bookmark("1")
      self.play(*[Create(r) for r in rects if r.color == PURPLE], *[Write(t) for t in texts if t.color == PURPLE])
      self.wait_until_bookmark("2")
      self.play(*[Create(r) for r in rects if r.color == BLUE], *[Write(t) for t in texts if t.color == BLUE])

    with self.voiceover(text="""The first type of memory that we should be familliar with is global memory""") as trk:
      self.play(*[Create(r) for r in rects if r.color == RED], *[Write(t) for t in texts if t.color == RED])

    with self.voiceover(text="""Each thread can read and write to global memory""") as trk:
      self.play(*[Create(arrows[i]) for i in [13, 15, 17, 19]])

    
    with self.voiceover(text="""Global memory is our largest but also slowest memory space - it is the VRAM of our GPU.""") as trk:
      pass

    malloc = Code(code="cudaMalloc((void**) &pointer, size);", tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    global_var = Code(code="__device__ int GlobalVariable = 0;", tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1).next_to(malloc, DOWN)

    with self.voiceover(text="""Every time that we call <bookmark mark='1'/>a malloc function or create a<bookmark mark='2'/>
                        global variable, it gets stored inside global memory""") as trk:
      self.wait_until_bookmark("1")
      self.play(Create(malloc))
      self.wait_until_bookmark("2")
      self.play(Create(global_var))

    target = VGroup(*[r.copy() for r in rects if r.color == RED])
    self.play(Transform(VGroup(malloc, global_var), target, replace_mobject_with_target_in_scene=True))
    self.remove(target)

    with self.voiceover(text="""The next type of memory that we have been using so far are registers""") as trk:
      self.play(*[Create(r) for r in rects if r.color == GREEN], *[Write(t) for t in texts if t.color == GREEN])

    with self.voiceover(text="""They are local to each thread, and extremely fast""") as trk:
      pass

    reg = Code(code="float reg = pointer[i];", tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    with self.voiceover(text="""Every time that we create a local variable inside our kernel, it gets stored inside our registers""") as trk:
      self.play(Create(reg))
