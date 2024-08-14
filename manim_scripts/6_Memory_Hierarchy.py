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
    with self.voiceover(text="""In this episode, we are going to briefly go over the memory hierarchy of our gpu as understanding
                        it will be crucial to getting the best performence our of our hardware""") as trk:
      self.play(Write(subtitle))

    with self.voiceover(text="""The purpose of this episode is to give you a quick overview of how memory in cuda works
                        befor we dive deeper into each kind of memory in future episodes""") as trk:
      pass


    pcb = ImageMobject("./PCB.jpg").scale(0.2)

    with self.voiceover(text="""When talking about memory we will ofter refer to some particular kind of memory as being on or off chip""") as trk:
      self.play(Unwrite(title), Unwrite(subtitle))

    with self.voiceover(text="""It might be confusing if you are not familliar with how the gpu internals look like""") as trk:
      self.play(FadeIn(pcb))

    with self.voiceover(text="""When you open up your gpu, you can see that it's actually like a small computer inside your computer""") as trk:
      pass

    chip = Rectangle(width=2, height=2, fill_color=GREEN, fill_opacity=0.5, color=GREEN).shift(0.5*UP)
    chip_text = Text("Chip", color=GREEN, font_size=36).next_to(chip, DOWN)
    with self.voiceover(text="""There is a chip that does the actuall computation""") as trk:
      self.play(Create(chip))
      self.play(Write(chip_text))

    offchip1 = Rectangle(width=0.7, height=2, fill_color=RED, fill_opacity=0.5, color=RED).shift(1.5*LEFT+0.5*UP)
    offchip2 = Rectangle(width=0.7, height=2, fill_color=RED, fill_opacity=0.5, color=RED).shift(1.3*RIGHT+0.5*UP)
    offchip3 = Rectangle(width=1.45, height=0.65, fill_color=RED, fill_opacity=0.5, color=RED).shift(1.95*UP)
    offchip_text = Text("Memory", color=RED, font_size=36).next_to(offchip3, UP)
    with self.voiceover(text="""And it's connected to VRAM that resides on the PCB""") as trk:
      self.play(Create(offchip1), Create(offchip2), Create(offchip3))
      self.play(Write(offchip_text))

    with self.voiceover(text="""but some of the memory resides in the actuall chip making it much faster to access""") as trk:
      pass

    self.play(*[Uncreate(x) for x in [chip, offchip1, offchip2, offchip3]], Unwrite(chip_text), Unwrite(offchip_text))

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


    with self.voiceover(text="""When speaking about memory hierarchy we have to take each unit into consideration, that is <bookmark mark='1'/>our blocks
                        <bookmark mark='2'/> and the threads that run inside our blocks.""") as trk:
      self.play(FadeOut(pcb))
      self.wait_until_bookmark("1")
      self.play(*[Create(r) for r in rects if r.color == PURPLE], *[Write(t) for t in texts if t.color == PURPLE])
      self.wait_until_bookmark("2")
      self.play(*[Create(r) for r in rects if r.color == BLUE], *[Write(t) for t in texts if t.color == BLUE])

    with self.voiceover(text="""The first type of memory that we should be familliar with is global memory""") as trk:
      self.play(*[Create(r) for r in rects if r.color == RED], *[Write(t) for t in texts if t.color == RED])

    with self.voiceover(text="""Each thread can read and write to global memory""") as trk:
      self.play(*[Create(arrows[i]) for i in [13, 15, 17, 19]])

    
    with self.voiceover(text="""Global memory is our largest but also slowest memory space - it is the VRAM of our GPU and it resides off chip.""") as trk:
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
      self.play(*[Create(arrows[i]) for i in [0, 2, 6, 8]])

    with self.voiceover(text="""They are local to each thread, and extremely fast as they reside on chip""") as trk:
      pass

    reg = Code(code="float reg = pointer[i];", tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    with self.voiceover(text="""Every time that we create a local variable inside our kernel, it gets stored inside our registers""") as trk:
      self.play(Create(reg))

    target = VGroup(*[r.copy() for r in rects if r.color == GREEN])
    self.play(Transform(reg, target, replace_mobject_with_target_in_scene=True))
    self.remove(target)

    with self.voiceover(text="""We can check how much registers we are using by adding a compilation flag 
                        for increased verbosity in ptxas""") as trk:
      pass


    with self.voiceover(text="""We can also use cuobjdump to check how are our registers accessed in PTX, and SASS assembly""") as trk:
      pass

    with self.voiceover(text="""Don't worry if those look like black magic - we will go over what PTX and SASS are in later episodes""") as trk:
      pass

    with self.voiceover(text="""There are some performance considerations when using our registers""") as trk:
      pass

    with self.voiceover(text="""First would be that using too much registers can cause reduced occupancy. We will go over occupancy in later
                        episodes as it deserves some more explanation, but for now just think about it as not having enough resources to run new thread
                        groups""") as trk:
      pass

    with self.voiceover(text="""The second one occurs when we use too much registers and the compiler determines that there is no more register
                        space to hold our variables""") as trk:
      pass

    with self.voiceover(text="""in this case, our variables get spilled into another kind of memory, which is local memory""") as trk:
      self.play(*[Create(r) for r in rects if r.color == RED_A], *[Write(t) for t in texts if t.color == RED_A])
      self.play(*[Create(arrows[i]) for i in [1, 3, 7, 9]])

    with self.voiceover(text="""And the name might be a bit confusing - it's called local not because of it's physical location but because it's local to a thread""") as trk:
      pass

    with self.voiceover(text="""it lives off chip - therefore accessing it is very slow and we want to avoid doing it""") as trk:
      pass

    with self.voiceover(text="""When compiling with increased verbosity we can also look into how much of our memory access is to local memory""") as trk:
      pass

    with self.voiceover(text="""I've made a kernel using a lot of variables and as you can see, after using 255 registers they started spilling into local memory,
                        resulting in 2040 bytes read and written to local memory""") as trk:
      pass

    self.wait(1)

    with self.voiceover(text="""Another kind of memory that we can use is Constant Memory""") as trk:
      self.play(*[Create(r) for r in rects if r.color == RED_B], *[Write(t) for t in texts if t.color == RED_B])
      self.play(*[Create(arrows[i]) for i in [12, 14, 16, 18]])

    with self.voiceover(text="""It is a special kind of memory, it resides off chip as global and local memory but it's cached and read-only""") as trk:
      pass

    with self.voiceover(text="""It is limited to only 64KB""") as trk:
      pass

    with self.voiceover(text="""and accesses to different addresses by threads within a warp are serialized - that means that if we access the same memory
                        address by multiple threads we can get better performance than when using global memory""") as trk:
      pass

    const_mem = """__constant__ float const_mem[size];
cudaMemcpyToSymbol(const_mem, const_mem_h, size*sizeof(float));"""
    const_mem_code = Code(code=const_mem, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)

    with self.voiceover(text="""To use constnt memory we have to use the __constant__ derivative when declaring our array,
                        we then have to use cudaMemcpyToSymbol to move our data from the cpu to const memory""") as trk:
      self.play(Create(const_mem_code))

    self.play(Uncreate(const_mem_code))
    with self.voiceover(text="""The final type of memory is shared memory""") as trk:
      self.play(*[Create(r) for r in rects if r.color == YELLOW], *[Write(t) for t in texts if t.color == YELLOW])
      self.play(*[Create(arrows[i]) for i in [4, 5, 10, 11]])

    with self.voiceover(text="""As the name suggests, it's shared between the threads in a block""") as trk:
      pass

    with self.voiceover(text="""And what that means is that if one thread in a <bookmark mark='1'/>block writes to shared memory, all the other threads in a block <bookmark mark='2'/>can read
                        the value written by that thread""") as trk:
      self.wait_until_bookmark("1")
      self.play(ShowPassingFlash(Arrow(start=arrows[4].get_start(), end=arrows[4].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      self.wait_until_bookmark("2")
      self.play(ShowPassingFlash(Arrow(start=arrows[5].get_end(), end=arrows[5].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))

    with self.voiceover(text="""moreover, shared memory lives on chip - meaning that accessing it is much faster than accessing global memory. 
                        That is why it is very often used in order to increase performence when multiple threads access the same memory address""") as trk:
      pass

    shared_mem = "__shared__ float shared_mem[size];"
    shared_mem_code = Code(code=shared_mem, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)

    with self.voiceover(text="""To allocate an array in shared memory we just have to add a __shared__ keyword when declaring our variable""") as trk:
      self.play(Create(shared_mem_code))

    self.play(Uncreate(shared_mem_code))

    with self.voiceover(text="""So to recap everything we've learned so far""") as trk:
      self.play(*[Uncreate(r) for r in rects + arrows], *[Unwrite(t) for t in texts])

    summary = Table([
      ["On", "R/W", "Thread", "Thread"],
      ["On", "R/W", "Block", "Block"],
      ["Off", "R/W", "Thread", "Thread"],
      ["Off", "R/W", "Global", "Host Controlled"],
      ["Off", "R", "Global", "Host Controlled"]],
                    row_labels=[Text(t) for t in ["Registers", "Shared", "Local", "Global", "Constant"]],
                    col_labels=[Text(t) for t in ["On/Off Chip", "Access", "Scope", "Lifetime"]]).scale(0.5)
    
    with self.voiceover(text="""We have five kinds of memory that we can use in our CUDA code""") as trk:
      self.play(*[Create(x) for x in summary.get_vertical_lines()])
      self.play(*[Write(x) for x in summary.get_col_labels()])

    def create_row(i):
      nonlocal summary
      self.play(Create(summary.get_horizontal_lines()[i]))
      self.play(LaggedStart(Write(summary.get_row_labels()[i]), *[Write(x) for x in summary.get_entries_without_labels()[i*4:(i+1)*4]]))


    with self.voiceover(text="""Register memory that lives on chip, can be read and written to and has a scope and a lifetime of our thread""") as trk:
      create_row(0)

    with self.voiceover(text="""Shared memory that also lives on chip, can be read and written to and has a scope and a lifetime of one block""") as trk:
      create_row(1)

    with self.voiceover(text="""Local memory that resides off chip, can be read and written to and has a scope and a lifetime of a thread""") as trk:
      create_row(2)

    with self.voiceover(text="""Global memory that resides off chip, can be read and written that can be accessed anywhere in our code and it's lifetime is controlled
                        by the host that decides when to deallocate it""") as trk:
      create_row(3)

    with self.voiceover(text="""And constant memory that also resides off chip, is read only, globally accessed and it's lifetime is also controlled by the host""") as trk:
      create_row(4)

    self.wait(1)

    with self.voiceover(text="""This will be it for our introduction to memory in CUDA, in the upcoming episodes we will dive deeper into
                        how we can use each kind of memory to improve the performance of our code""") as trk:
      pass

    
    with self.voiceover(text="""Subscribe not to miss it, leave a like, comment your feedback and do anything that helps the algorithm.
                        And I'll see you in the next episode - bye.""") as trk:
      pass

    anims = [] 
    for obj in self.mobjects:
      anims.append(FadeOut(obj))
    self.play(*anims)
    self.wait(3)


