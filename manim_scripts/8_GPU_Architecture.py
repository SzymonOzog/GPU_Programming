from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import types
import random


class GPUArchitecture(VoiceoverScene, MovingCameraScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        )

    title = Text("GPU programming", font_size=72)
    with self.voiceover(text="Hello and welcome to the next episode in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Modern GPU Architecture", font_size=48).next_to(title, DOWN)

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

    self.play(Unwrite(title))
    with self.voiceover(text="""I started this out as an episode on constant memory, in the middle of it I realized that explaining 
                        the concept of a warp and a Streaming Multiprocessor will make things easier, and while trying to get to the 
                        Streaming Multiprocessor I ended up going through the full architecture of a modern GPU""") as trk:
      self.play(*[Create(r) for r in rects], *[Write(t) for t in texts], *[Create(a) for a in arrows])
      while trk.get_remaining_duration() > 1:
        self.play(*access_anims[random.randint(0, len(access_anims) -1)])

    with self.voiceover(text="""And I do feel like that historian that wanted to tell you about Hiroshima so he starts with Julius Ceasar
                        to give you some good context on the events leading up to it""") as trk:
      while trk.get_remaining_duration() > 1:
        self.play(*access_anims[random.randint(0, len(access_anims) -1)])

    with self.voiceover(text="""Anyway this episode will be focused on the Architecture of the Modern GPU""") as trk:
      self.play(*[Uncreate(r) for r in rects], *[Unwrite(t) for t in texts], *[Uncreate(a) for a in arrows])
      self.play(Write(subtitle))
    self.play(Unwrite(subtitle))


    gpu_rects = []
    gpu_texts = []
    gpu_alu_list = []
    sm_list = []
    
    for i in range(5):
      cc = VGroup(Rectangle(width=0.5, height=0.2, color=PURPLE, fill_opacity=0.5), 
                  Rectangle(width=0.5, height=0.2, color=RED, fill_opacity=0.5)).arrange(DOWN, buff=0.1)
      gpu_rects.append(cc[0])
      gpu_rects.append(cc[1])
      alus = [Rectangle(width=0.5, height=0.5, color=BLUE, fill_opacity=0.5) for _ in range(8)]
      gpu_rects.extend(alus)
      gpu_alu_list.append(VGroup(cc, *alus).arrange(RIGHT, buff=0.1))
      sm_list.append([cc[0], cc[1], *alus])

    gpu_alus = VGroup(*gpu_alu_list).scale(0.8).arrange(DOWN, buff=0.16)


    l2 = Rectangle(width=4.25, height=0.4, color=RED, fill_opacity=0.5).match_width(gpu_alus).next_to(gpu_alus, DOWN, buff=0.1)
    gpu_rects.append(l2)
    l2_text = Text("L2 Cache", font_size=14).move_to(l2)
    gpu_texts.append(l2_text)

    dram_gpu = Rectangle(width=4.25, height=0.5, color=GREEN, fill_opacity=0.5).match_width(gpu_alus).next_to(l2, DOWN, buff=0.1)
    gpu_rects.append(dram_gpu)
    dram_gpu_text = Text("DRAM", font_size=14).move_to(dram_gpu.get_center())
    gpu_texts.append(dram_gpu_text)


    gpu = VGroup(gpu_alus, l2, l2_text, dram_gpu, dram_gpu_text)
    gpu_title = Text("GPU").scale(0.8).next_to(gpu, UP)
    gpu_texts.append(gpu_title)

    subobjects = []
    queue = [gpu]
    while queue:
      o = queue.pop()
      subobjects.append(o)
      queue.extend(o.submobjects)

    for mo in subobjects:
      for so in mo.submobjects.copy():
        if any(so in x for x in [gpu_rects, gpu_texts]):
          mo.remove(so)

    with self.voiceover(text="""We touched on this subject before in the first episode where we've shown a highly simplified architecture of a GPU""") as trk:
      self.play(LaggedStart(*[Create(x) for x in gpu_rects]))
      self.play(LaggedStart(*[Write(x) for x in gpu_texts]))
    
    with self.voiceover(text="""We outlined that there are thread groups that share control units,
                        and those are the streaming multiprocessors - the core components of our GPU""") as trk:
      for sm in sm_list:
        self.play(Indicate(VGroup(*sm)))

    gpu = Rectangle(height=6, width=12, color=GREEN)
    gpu_t = Text("GPU", color=GREEN, font_size=24).next_to(gpu, UP, buff=0.1, aligned_edge=LEFT)
    die = ImageMobject("./die_shot.jpg")
    blog_post = ImageMobject("./blogpost.png").rotate(-PI/7).scale(1.5)
    whitepaper = ImageMobject("./ada_wp.png").rotate(PI/8).scale(1.3)
    forum = ImageMobject("./forum.png").rotate(-PI/9)
    microbench = ImageMobject("./microbench.png").rotate(PI/7).scale(1.2)

    with self.voiceover(text="""But as you might imagine, the actuall gpu architecture is much more complicated, and do take everything with a grain of salt. 
                        A lot of the details are not released to the public, and the information is very often stiched together out <bookmark mark='1'/>of multiple blog posts, 
                        <bookmark mark='2'/>official architectural whitepapers, <bookmark mark='3'/>nvidia forum discussions and <bookmark mark='4'/>third party microbenchmarks. """) as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
      self.play(FadeIn(die))
      self.wait_until_bookmark("1")
      self.add(blog_post)
      self.wait_until_bookmark("2")
      self.add(whitepaper)
      self.wait_until_bookmark("3")
      self.add(forum)
      self.wait_until_bookmark("4")
      self.add(microbench)

    with self.voiceover(text="""Most of it is true, and we do have high quality die shots like the one on the screen to confirm it, but small details might be missing or wrong""") as trk:
      self.play(*[FadeOut(x) for x in [blog_post, whitepaper, forum, microbench]])

    dram = Rectangle(height=5, width=1, color=RED, fill_color=RED, fill_opacity=0.5).shift(5.25*LEFT)
    dram_t = Text("DRAM", color=RED, font_size=52).move_to(dram).rotate(PI/2)

    chip = Rectangle(height=5, width=10, color=YELLOW).shift(0.5*RIGHT)
    chip_t = Text("CHIP", color=YELLOW, font_size=24).next_to(chip, UP, buff=0.1, aligned_edge=LEFT)

    with self.voiceover(text="""As I've shown in the episode on memory, on the actual PCB we have our <bookmark mark='1'/>DRAM memory
                        and a chip <bookmark mark='2'/>  that does the calculations""") as trk:
      self.play(FadeOut(die))
      self.play(Create(gpu), Write(gpu_t))
      self.wait_until_bookmark("1")
      self.play(Create(dram), Write(dram_t))
      self.wait_until_bookmark("2")
      self.play(Create(chip), Write(chip_t))

    with self.voiceover(text="""The PCB obviously contains much more components, like power circuts, voltage
                        controllers and all that beautifull stuff that I'm going to omit. I bet there are a lot
                        of electronics nerds here that would love to hear all about this stuff but I'm sure that
                        they would also be outraged by my lack of knowledge on the topic""") as trk:
      pass

    l2 = Rectangle(height=1, width=9, color=BLUE, fill_color=BLUE, fill_opacity=0.5).shift(0.5*RIGHT)
    l2_t = Text("L2 Cache", color=BLUE, font_size=52).move_to(l2)
    with self.voiceover(text="""I'm going to take the AD102 chip as a reference for this video. This is the chip 
                        that is in the 4090 and some other cards from the Ada architecture""") as trk:
      pass

    with self.voiceover(text="""The core logic is the same but there are details changing from architecture,
                        to architecture so keep that in mind""") as trk:
      pass

    with self.voiceover(text="""So our chip contains the L2 cache that is shared between all cores, if you are not familliar with caches,
                        they are a type of fast SRAM memory, much smaller than our main DRAM, when we access DRAM memory some of it gets stored in the fast
                        cache so that when we want to access it later we don't have to wait for it again""") as trk:
      self.play(Create(l2), Write(l2_t))

    with self.voiceover(text="""I realize that this was a brief desctiption, but cache is so complex that it would need an episode of it's own
                        to explain in detail. If you are interested in the topic let me know in the comments and I'll might make one in the future""") as trk:
      pass
    mc = Rectangle(height=4.5, width=0.25).shift(4.25*LEFT)
    mc_t = Text("6 x Memory Controller", font_size=14).move_to(mc).rotate(PI/2)

    mc2 = Rectangle(height=4.5, width=0.25).shift(5.25*RIGHT)
    mc_t2 = Text("6 x Memory Controller", font_size=14).move_to(mc2).rotate(-PI/2)
    with self.voiceover(text="""It also contains 12 memory controllers that handle data transfer between layers of memory""") as trk:
      self.play(Create(mc), Write(mc_t), Create(mc2), Write(mc_t2))
    
    gpcs = []
    gpc_ts = []

    for i in range(6):
      if i == 0:
        gpcs.append(Rectangle(height=1.5, width=1.25, color=PURPLE, fill_color=PURPLE, fill_opacity=0.5, stroke_width=2).next_to(l2, UP, aligned_edge=LEFT, buff=0.25).shift(0.125*RIGHT))
      else:
        gpcs.append(Rectangle(height=1.5, width=1.25, color=PURPLE, fill_color=PURPLE, fill_opacity=0.5, stroke_width=2).next_to(gpcs[-1], RIGHT))
      gpc_ts.append(Text("GPC", font_size=32, color=PURPLE).move_to(gpcs[-1]))

    for i in range(6):
      if i == 0:
        gpcs.append(Rectangle(height=1.5, width=1.25, color=PURPLE, fill_color=PURPLE, fill_opacity=0.5, stroke_width=2).next_to(l2, DOWN, aligned_edge=LEFT, buff=0.25).shift(0.125*RIGHT))
      else:
        gpcs.append(Rectangle(height=1.5, width=1.25, color=PURPLE, fill_color=PURPLE, fill_opacity=0.5, stroke_width=2).next_to(gpcs[-1], RIGHT))
      gpc_ts.append(Text("GPC", font_size=32, color=PURPLE).move_to(gpcs[-1]))

    with self.voiceover(text="""Another part of our chip are the Graphic Processing Clusters, GPC's for short, in the case of 
                        the AD102 there are 12 of them on the chip""") as trk:
      self.play(LaggedStart(*[Create(gpc) for gpc in gpcs], *[Write(t) for t in gpc_ts]))

    re = Rectangle(height=0.1, width=1.15, stroke_width=1).move_to(gpcs[0], UP).shift(0.03*DOWN)
    re_t = Text("Raster Engine", font_size=18).move_to(re).scale(0.3)

    rop = Rectangle(height=0.1, width=0.55, stroke_width=1).move_to(gpcs[0], LEFT+DOWN).shift(0.04*(UP+RIGHT))
    rop_t = Text("8x ROP", font_size=18).move_to(rop).scale(0.3)
    rop2 = Rectangle(height=0.1, width=0.55, stroke_width=1).move_to(gpcs[0], RIGHT+DOWN).shift(0.04*(UP+LEFT))
    rop_t2 = Text("8x ROP", font_size=18).move_to(rop2).scale(0.3)
    
    tpcs = []
    tpc_ts = []

    for i in range(3):
      if i == 0:
        tpcs.append(Rectangle(height=0.55, width=0.33, color=ORANGE, fill_color=ORANGE, fill_opacity=0.5, stroke_width=1).move_to(gpcs[0], LEFT+UP).shift(0.07*RIGHT + 0.17*DOWN))
      else:
        tpcs.append(Rectangle(height=0.55, width=0.33, color=ORANGE, fill_color=ORANGE, fill_opacity=0.5, stroke_width=1).next_to(tpcs[-1], RIGHT, buff=0.07))
      tpc_ts.append(Text("TPC", font_size=12, color=ORANGE).move_to(tpcs[-1]).rotate(PI/2))

    for i in range(3):
      if i == 0:
        tpcs.append(Rectangle(height=0.55, width=0.33, color=ORANGE, fill_color=ORANGE, fill_opacity=0.5, stroke_width=1).move_to(gpcs[0], LEFT+DOWN).shift(0.07*RIGHT+ 0.17*UP))
      else:
        tpcs.append(Rectangle(height=0.55, width=0.33, color=ORANGE, fill_color=ORANGE, fill_opacity=0.5, stroke_width=1).next_to(tpcs[-1], RIGHT, buff=0.07))
      tpc_ts.append(Text("TPC", font_size=12, color=ORANGE).move_to(tpcs[-1]).rotate(PI/2))

    with self.voiceover(text="""And each one contains, 6 Texture Processing Clusters, as well as some components for rasterization
                         namely the <bookmark mark='1'/>Raster Engine that generates pixel information from triangles
                         and 16 Render Output Units <bookmark mark='2'/>divided into two Raster Operations Partitions""") as trk:
      self.play(FadeOut(gpc_ts[0]), Transform(gpcs[0], Rectangle(height=1.5, width=1.25, color=PURPLE).next_to(l2, UP, aligned_edge=LEFT, buff=0.25).shift(0.125*RIGHT)))
      self.play(self.camera.auto_zoom(gpcs[0]))
      self.play(LaggedStart(*[Create(tpc) for tpc in tpcs], *[Write(t) for t in tpc_ts]))
      self.wait_until_bookmark("1")
      self.play(Create(re), Write(re_t))
      self.wait_until_bookmark("2")
      self.play(Create(rop), Write(rop_t), Create(rop2), Write(rop_t2))

    with self.voiceover(text="""Going further down a level into our Texture Processing Cluster""") as trk:
      self.play(FadeOut(tpc_ts[0]), Transform(tpcs[0], Rectangle(height=0.55, width=0.33, color=ORANGE, stroke_width=2).move_to(tpcs[0])))
      all = VGroup(*[x for x in self.mobjects if isinstance(x, Rectangle) or isinstance(x, Text)])
      for x in self.mobjects:
        if isinstance(x, Rectangle):
          x.stroke_width*=10
      all.scale(10)
      self.camera.auto_zoom(gpcs[0], animate=False)
      self.play(self.camera.auto_zoom(tpcs[0]))

    pm = Rectangle(width=2.5, height=0.5, fill_opacity=0.5).move_to(tpcs[0]).shift(2.2*UP)
    pm_t = Text("PolyMorph Engine", font_size=40).scale(0.5).move_to(pm)
    with self.voiceover(text="""We can see that it's composed of a PolyMorph Engine - another component used for computer graphics.
                        It handles things like Vertex Fetch, Tessellation, Viewport Transform, Attribute Setup, and Stream Output""") as trk:
      self.play(Create(pm), Write(pm_t))

    sm = Rectangle(width=2.5, height=2, color=MAROON, fill_color=MAROON, fill_opacity=0.5).move_to(tpcs[0]).shift(0.7*UP)
    sm_t = Text("    Streaming\nMultiprocessor", font_size=24, color=MAROON).move_to(sm)
    
    sm2 = Rectangle(width=2.5, height=2, color=MAROON, fill_color=MAROON, fill_opacity=0.5).move_to(tpcs[0]).shift(1.5*DOWN)
    sm2_t = Text("    Streaming\nMultiprocessor", font_size=24, color=MAROON).move_to(sm2)
    with self.voiceover(text="""But more importantly to our use cases, it contains 2 streaming multiprocessors""") as trk:
      self.play(Create(sm), Create(sm2), Write(sm_t), Write(sm2_t))

    with self.voiceover(text="""Let's zoom in on our SM's""") as trk:
      self.play(FadeOut(sm_t), Transform(sm, Rectangle(width=2.5, height=2, color=MAROON).move_to(sm)))
      all = VGroup(*[x for x in self.mobjects if isinstance(x, Rectangle) or isinstance(x, Text)])
      for x in self.mobjects:
        if isinstance(x, Rectangle):
          x.stroke_width*=3
      all.scale(3)
      self.camera.auto_zoom(tpcs[0], animate=False)
      self.play(self.camera.auto_zoom(sm))

    rt = Rectangle(width=2.5, height=0.7, fill_opacity=0.5).move_to(sm).shift(2.4*DOWN + 2.25*RIGHT)
    rt_t = Text("RT Core", font_size=32).move_to(rt)
    with self.voiceover(text="""It contains a Ray Tracing Core - a dedicated hardware unit for ray tracing operations""") as trk:
      self.play(Create(rt), Write(rt_t))

    texs = []
    tex_ts = []
    for i in range(4):
      if i == 0:
        texs.append(Rectangle(height=0.7, width=1, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.5).move_to(sm).shift(2.4*DOWN + 3*LEFT))
      else:
        texs.append(Rectangle(height=0.7, width=1, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.5).next_to(texs[-1], RIGHT, buff=0.1))
      tex_ts.append(Text("TEX", font_size=32, color=BLUE_E).move_to(texs[-1]))

    with self.voiceover(text="""We also have 4 Texture Units that perform operations on Textures""") as trk:
      self.play(LaggedStart(*[Create(tex) for tex in texs], *[Write(t) for t in tex_ts]))

    l1 = Rectangle(height=0.5, width=7, color=GOLD_A, fill_color=GOLD_A, fill_opacity=0.5).move_to(sm).shift(1.7*DOWN)
    l1_t = Text("128KB L1 Cache / Shared Memory", color=GOLD_A, font_size=28).move_to(l1)

    with self.voiceover(text="""128KB of memory divided into L1 cache and shared memory""") as trk:
      self.play(Create(l1), Write(l1_t))

    with self.voiceover(text="""The fact that this memory is shared is very important for us, it tells us that the more shared memory we use
                        the less L1 cache we have available""") as trk:
      pass

    cc = Rectangle(height=0.5, width=7, color=GOLD_E, fill_color=GOLD_E, fill_opacity=0.5).next_to(l1, UP, buff=0.1)
    cc_t = Text("8KB Constant Cache", color=GOLD_E, font_size=28).move_to(cc)
    with self.voiceover(text="""8KB of special cache for accesses to constant memory""") as trk:
      self.play(Create(cc), Write(cc_t))


    cc_i = ImageMobject("./constant_cache1.png").move_to(sm)
    with self.voiceover(text="""It's worth mentioning that one group that microbenchmarked the volta architecture gpus also mentioned a 1.5 level
                        constant cache but I'm unable to find any confirmations for it in the official sources""") as trk:
      self.play(FadeIn(cc_i))
    self.play(FadeOut(cc_i))

    ps = []
    p_ts = []

    for i in range(4):
      if i == 0:
        ps.append(Rectangle(height=3.6, width=1.675, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.5).move_to(sm).shift(2.65*LEFT + 1.05*UP))
      else:
        ps.append(Rectangle(height=3.6, width=1.675, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.5).next_to(ps[-1], RIGHT, buff=0.093))
      p_ts.append(Text("Processing Block", font_size=32, color=GREEN_A).move_to(ps[-1]).rotate(PI/2))

    with self.voiceover(text="""And it also contains 4 Processing Blocks""") as trk:
      self.play(LaggedStart(*[Create(p) for p in ps], *[Write(t) for t in p_ts]))

    with self.voiceover(text="""Inside our Processing blocks we can indentify the smallest components, that execute our instructions""") as trk:
      self.play(FadeOut(p_ts[0]), Transform(ps[0], Rectangle(height=3.6, width=1.675, color=GREEN_A).move_to(ps[0])))
      all = VGroup(*[x for x in self.mobjects if isinstance(x, Rectangle) or isinstance(x, Text)])
      for x in self.mobjects:
        if isinstance(x, Rectangle):
          x.stroke_width*=2
      all.scale(2)
      self.play(self.camera.auto_zoom(ps[0]))

    ws = Rectangle(width=3, height=0.5, color=YELLOW_A, fill_color=YELLOW_A, fill_opacity=0.5).move_to(ps[0]).shift(3.15*UP)
    ws_t = Text("Warp Scheduler", color=YELLOW_A, font_size=32).scale(0.6).move_to(ws)

    du = Rectangle(width=3, height=0.5, color=YELLOW_B, fill_color=YELLOW_B, fill_opacity=0.5).next_to(ws, DOWN, buff=0.2)
    du_t = Text("Dispatch Unit", color=YELLOW_B, font_size=32).scale(0.6).move_to(du)

    ic = Rectangle(width=3, height=0.5, color=YELLOW_C, fill_color=YELLOW_C, fill_opacity=0.5).next_to(du, DOWN, buff=0.2)
    ic_t = Text("L0 Instruction Cache", color=YELLOW_C, font_size=32).scale(0.6).move_to(ic)

    rf = Rectangle(width=3, height=0.66, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.5).next_to(ic, DOWN, buff=0.2)
    rf_t = Text("64KB Register File", color=BLUE_A, font_size=32).scale(0.7).move_to(rf)

    tc = Rectangle(width=3, height=0.5, color=GREEN_B, fill_color=GREEN_B, fill_opacity=0.5).next_to(rf, DOWN, buff=0.2)
    tc_t = Text("Tensor Core", color=GREEN_B, font_size=32).scale(0.7).move_to(tc)
    
    fpcs = []
    for i in range(2):
      for j in range(8):
        fpc = Rectangle(width=0.33, height=0.33, color=GREEN_C, fill_color=GREEN_C, fill_opacity=0.5, stroke_width=1)
        if j == 0:
          if i == 0:
            fpc.next_to(tc, DOWN, aligned_edge=LEFT, buff=0.2)
          else:
            fpc.next_to(fpcs[0], DOWN, aligned_edge=LEFT, buff=0.05)
        else:
          fpc.next_to(fpcs[-1], RIGHT, buff=0.05)
        fpcs.append(fpc)
    fpc_t = Text("FP32", font_size=32, color=GREEN_C).move_to(VGroup(*fpcs))


    fpcis = []
    for i in range(2):
      for j in range(8):
        fpci = Rectangle(width=0.33, height=0.33, color=GREEN_E, fill_color=GREEN_E, fill_opacity=0.5, stroke_width=1)
        if j == 0:
          if i == 0:
            fpci.next_to(fpcs[8], DOWN, aligned_edge=LEFT, buff=0.2)
          else:
            fpci.next_to(fpcis[0], DOWN, aligned_edge=LEFT, buff=0.05)
        else:
          fpci.next_to(fpcis[-1], RIGHT, buff=0.05)
        fpcis.append(fpci)
    fpci_t = Text("FP32/I32", font_size=32, color=GREEN_E).move_to(VGroup(*fpcis))


    lsus = []
    lsu_ts = []
    for i in range(4):
      lsu = Rectangle(width=0.68, height=0.55, color=RED_A, fill_color=RED_A, fill_opacity=0.5, stroke_width=2)
      if i == 0:
        lsu.next_to(fpcis[8], DOWN, aligned_edge=LEFT, buff=0.2)
      else:
        lsu.next_to(lsus[-1], RIGHT, buff=0.1)
      lsus.append(lsu)
      lsu_ts.append(Text("LD/ST", font_size=32, color=RED_A).scale(0.5).move_to(lsus[-1]))


    sfus = []
    sfu_ts = []
    for i in range(4):
      sfu = Rectangle(width=0.68, height=0.55, color=RED_C, fill_color=RED_C, fill_opacity=0.5, stroke_width=2)
      if i == 0:
        sfu.next_to(lsus[0], DOWN, aligned_edge=LEFT, buff=0.2)
      else:
        sfu.next_to(sfus[-1], RIGHT, buff=0.1)
      sfus.append(sfu)
      sfu_ts.append(Text("SFU", font_size=32, color=RED_C).scale(0.5).move_to(sfus[-1]))

    with self.voiceover(text="""We have 16 cuda cores per processing blocks that are capable of running FP32 operations""") as trk:
      self.play(LaggedStart(*[Create(x) for x in fpcs]), Write(fpc_t))
    with self.voiceover(text="""Another 16 cores that can execute either FP32 or INT32 instructions""") as trk:
      self.play(LaggedStart(*[Create(x) for x in fpcis]), Write(fpci_t))
    with self.voiceover(text="""It also contains a Tensor Core - this is a specialized unit for performing matrix multiplication and acummulation""") as trk:
      self.play(Create(tc), Write(tc_t))
    with self.voiceover(text="""CUDA Programming Guide as well as some architectural whitepapers also mention that there are 2 FP64 cores per SM 
                        I'm not outlining them here because it's not really clear where they reside. I doubt that they are outside Processing Blocks.
                        But there are more Processing Blocks than there are FP64 Cores mentioned. Are not all Processing Blocks the same size? Do some have FP64
                        Cores disabled? It's reallly impossible to tell without doing some photon screening of the chip""") as trk:
      pass

    with self.voiceover(text="""So we have 32 CUDA cores available for computation, this brings us to an idea of a warp""") as trk:
      self.play(Indicate(VGroup(*fpcs, *fpcis), run_time=2))
      self.play(*[FadeOut(x) for x in ps], [FadeOut(x) for x in p_ts], FadeOut(sm), FadeOut(tpcs[0]))
    
    block = Rectangle(width=4.3, height=6, color=BLUE).next_to(ps[0], RIGHT)
    block_t = Text("Block", color=BLUE, font_size=40).move_to(block, aligned_edge=UP).shift(0.2*DOWN)

    threads = [Rectangle(width=0.33, height=0.33, color=GREEN, fill_color=GREEN, fill_opacity=0.5) for _ in range(72)]
    tg = VGroup(*threads).arrange_in_grid(rows=9, buff=(0.05, 0.15)).move_to(block).shift(0.1*UP)
    tmp = VGroup(*threads[:32])
    w1_t = Text("Warp 0", font_size=24, color=PURPLE).rotate(PI/2).next_to(tmp, LEFT, buff=0.1)
    w1 = SurroundingRectangle(VGroup(tmp, w1_t, w1_t.copy().next_to(tmp, RIGHT, buff=0.1)), buff=0.05, color=PURPLE)
    w1.stretch_to_fit_width(w1.width*1.05)

    tmp = VGroup(*threads[32:64])
    w2_t = Text("Warp 1", font_size=24, color=PURPLE).rotate(PI/2).next_to(tmp, LEFT, buff=0.1)
    w2 = SurroundingRectangle(VGroup(tmp, w2_t, w2_t.copy().next_to(tmp, RIGHT, buff=0.1)), buff=0.05, color=PURPLE)
    w2.stretch_to_fit_width(w2.width*1.05)

    tmp = VGroup(*threads[64:])
    w3_t = Text("Warp 2", font_size=24, color=PURPLE).rotate(PI/2).next_to(tmp, LEFT, aligned_edge=UP, buff=0.1)
    w3 = SurroundingRectangle(VGroup(tmp, w3_t, w3_t.copy().next_to(tmp, RIGHT, aligned_edge=UP, buff=0.1)), buff=0.05, color=PURPLE)
    w3.stretch_to_fit_width(w3.width*1.05)

    with self.voiceover(text="""When we actually launch our kernel grid, all of the blocks inside are further divided into collections of 32
                        threads that our Processing Blocks can execute, this thread collections are called warps""") as trk:
      self.play(Create(block), Write(block_t))
      self.play(LaggedStart(*[Create(x) for x in threads]))
      self.play(Write(w1_t), Create(w1))
      self.play(Write(w2_t), Create(w2))
      self.play(Write(w3_t), Create(w3))

    with self.voiceover(text="""You might have heard a mantra to always have your blocks be a multiple of 32 threads""") as trk:
      pass

    with self.voiceover(text="""This is the reason, since the blocks are divided into warps we might have a situation like this<bookmark mark='1'/>
                        where our last warp only has 8 threads executing, leaving 24 threads idle while they could be doing some work""") as trk:
      self.wait_until_bookmark("1")
      self.play(Indicate(VGroup(w3, w3_t, *threads[64:])))

    with self.voiceover(text="""Warps take us to 2 control components that is the <bookmark mark='1'/>Warp Scheduler, and a <bookmark mark='2'/> Dispatch Unit""") as trk:
      self.wait_until_bookmark("1")
      self.play(Create(ws), Write(ws_t))
      self.wait_until_bookmark("2")
      self.play(Create(du), Write(du_t))

    with self.voiceover(text="""The division of work between the two is one of those stichted together from different sources type of information so I might be wrong on some of the details
                        All of this happens but it's not exactly clear which component does what part of the work.
                        As I read the literature the <bookmark mark='1'/>Warp Scheduler assigns and manages warps that are executed by a Processing Block""") as trk:
      anims = []
      for t, c in zip(threads[:32], fpcs+fpcis):
        t.save_state()
        anims.append(t.animate.move_to(c))
      self.wait_until_bookmark("1")
      self.play(LaggedStart(*anims))

    with self.voiceover(text="""So for example when one warp is waiting for a data fetch from global memory, the warp scheduler might perform a context switch <bookmark mark='1'/>
                        and transfer control to another warp untill the first one is ready to resume execution. This further hides latency and speeds up execution.""") as trk:
      pass
      self.wait_until_bookmark("1")
      self.play(LaggedStart(Restore(x) for x in threads[:32]))
      anims = []
      for t, c in zip(threads[32:64], fpcs+fpcis):
        t.save_state()
        anims.append(t.animate.move_to(c))
      self.play(LaggedStart(*anims))
      
    with self.voiceover(text="""And the Dispatch Unit dispatches the instructions that our warps will execute""") as trk:
      pass

    with self.voiceover(text="""Another control component inside our Processing Block is an instruction cache, that works similarly to our data cache but it caches the next instructions to be executed""") as trk:
      self.play(Create(ic), Write(ic_t))
      self.play(FadeOut(block), FadeOut(block_t), *[FadeOut(x) for x in threads], FadeOut(w1_t), FadeOut(w1), FadeOut(w2_t), FadeOut(w2), FadeOut(w3_t), FadeOut(w3))
      self.play(*[FadeIn(x) for x in ps], [FadeIn(x) for x in p_ts], FadeIn(sm), FadeIn(tpcs[0]))

    with self.voiceover(text="""A processing block also contains a 64 KB register file to hold our data inside our registers""") as trk:
      self.play(Create(rf), Write(rf_t))

    with self.voiceover(text="""4 Load/Store units that control our memory access instructions""") as trk:
      self.play(LaggedStart(*[Create(x) for x in lsus]), LaggedStart(*[Write(x) for x in lsu_ts]))

    with self.voiceover(text="""And I'm going to go with 4 Special Function Units - they perform functions for graphics interpolation as well as trigonometric and transcendental operations""") as trk:
      self.play(LaggedStart(*[Create(x) for x in sfus]), LaggedStart(*[Write(x) for x in sfu_ts]))

    log = Tex("$\\log(x)$", font_size=48).next_to(rf, LEFT).shift(0.8*LEFT)
    sin = Tex("$\\sin(x)$", font_size=48).next_to(log, DOWN)
    cos = Tex("$\\cos(x)$", font_size=48).next_to(sin, DOWN)
    with self.voiceover(text="""So functions like for example a Logarithm Function, sine, cosine etc..""") as trk:
      self.play(Write(log))
      self.play(Write(sin))
      self.play(Write(cos))
    whitepaper_t = ImageMobject("./Whitepaper_T.png").move_to(ps[0]).scale(1.7)
    whitepaper_f = ImageMobject("./Whitepaper_F.png").move_to(ps[0]).scale(1.5)
    pg = ImageMobject("./ProgrammingGuide.png").move_to(ps[0]).scale(1.3)
    with self.voiceover(text="""You might have also notice that I've said I'm going to go with 4 SFUs, that's because there
                        is a bit of an ambiguity in this area""") as trk:
      self.play(LaggedStart(Unwrite(cos), Unwrite(sin), Unwrite(log), lag_ratio=0.15))

    with self.voiceover(text="""Because if you look at the Programming Guide from NVIDIA, they mention 16 SFU's per SM so 4 per Processing Block""") as trk:
      self.play(FadeIn(pg))

    with self.voiceover(text="""But the official Ada architecture whitepaper claims one""") as trk:
      self.play(FadeOut(pg))
      self.play(FadeIn(whitepaper_t))

    with self.voiceover(text="""And if you look at the figure they draw it as one SFU made out of 4 elements.
                        Again a lot of architectural stuff and barely mentioned so we have to take all of this with a grain of salt""") as trk:
      self.play(FadeOut(whitepaper_t))
      self.play(FadeIn(whitepaper_f))
    self.play(FadeOut(whitepaper_f))

    def monkey_patch(
        self,
        mobjects: list[Mobject],
        margin: float = 0,
        only_mobjects_in_frame: bool = False,
        animate: bool = True,
        run_time=1,
    ):
        scene_critical_x_left = None
        scene_critical_x_right = None
        scene_critical_y_up = None
        scene_critical_y_down = None

        for m in mobjects:
            if (m == self.frame) or (
                only_mobjects_in_frame and not self.is_in_frame(m)
            ):
                # detected camera frame, should not be used to calculate final position of camera
                continue

            # initialize scene critical points with first mobjects critical points
            if scene_critical_x_left is None:
                scene_critical_x_left = m.get_critical_point(LEFT)[0]
                scene_critical_x_right = m.get_critical_point(RIGHT)[0]
                scene_critical_y_up = m.get_critical_point(UP)[1]
                scene_critical_y_down = m.get_critical_point(DOWN)[1]

            else:
                if m.get_critical_point(LEFT)[0] < scene_critical_x_left:
                    scene_critical_x_left = m.get_critical_point(LEFT)[0]

                if m.get_critical_point(RIGHT)[0] > scene_critical_x_right:
                    scene_critical_x_right = m.get_critical_point(RIGHT)[0]

                if m.get_critical_point(UP)[1] > scene_critical_y_up:
                    scene_critical_y_up = m.get_critical_point(UP)[1]

                if m.get_critical_point(DOWN)[1] < scene_critical_y_down:
                    scene_critical_y_down = m.get_critical_point(DOWN)[1]

        # calculate center x and y
        x = (scene_critical_x_left + scene_critical_x_right) / 2
        y = (scene_critical_y_up + scene_critical_y_down) / 2

        # calculate proposed width and height of zoomed scene
        new_width = abs(scene_critical_x_left - scene_critical_x_right)
        new_height = abs(scene_critical_y_up - scene_critical_y_down)

        m_target = self.frame.animate(run_time=run_time) if animate else self.frame
        # zoom to fit all mobjects along the side that has the largest size
        if new_width / self.frame.width > new_height / self.frame.height:
            return m_target.set_x(x).set_y(y).set(width=new_width + margin)
        else:
            return m_target.set_x(x).set_y(y).set(height=new_height + margin)

    self.camera.auto_zoom = types.MethodType(monkey_patch, self.camera)
    with self.voiceover(text="""This is it, we coverd all the components that are mentioned, throughout the architectural whitepapers
                        and some that are omitted.I'll repeat my self once more but, as I've said in the beginning due to competetive advantages and many other factors, not everything
                        is described in detail and a lot of details are omitted or obfuscated so do treat it like a history lesson not as a gospel. 
                        The key and the most important ideas are definetly there but some details get conflicting evidence""") as trk:
      print(trk.duration, trk.get_remaining_duration())
      self.play(self.camera.auto_zoom(sm, run_time=(trk.duration-12)/4))
      self.wait(3)
      self.play(self.camera.auto_zoom(tpcs[0], run_time=(trk.duration-12)/4))
      self.wait(3)
      self.play(self.camera.auto_zoom(gpcs[0], run_time=(trk.duration-12)/4))
      self.wait(3)
      self.play(self.camera.auto_zoom(gpu, margin=2, run_time=(trk.duration-12)/4))


    bmac = Text("https://buymeacoffee.com/simonoz", font_size=48, color=YELLOW)
    alex = Text("Alex", font_size=60).next_to(bmac, DOWN)
    unknown = Text("Anonymous x2", font_size=60).next_to(alex, DOWN)
    with self.voiceover(text="""I've recently started a buy me a coffe page for those that want to support this channel. A shoutout to Alex and two anonymous donors that supported so far""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
      self.camera.auto_zoom(VGroup(bmac, alex, unknown), margin=4, animate=False)
      self.play(Write(bmac))
      self.play(Write(alex))
      self.play(Write(unknown))

    with self.voiceover(text="""But you can always support me without spending a buck by subscribing, leaving a like, commenting and sharing this video with your friends""") as trk:
      pass

    with self.voiceover(text="""And as always, I'll see you in the next episode, bye""") as trk:
      pass

    self.play(*[FadeOut(x) for x in self.mobjects])
    self.wait(2)
