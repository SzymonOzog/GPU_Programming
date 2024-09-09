from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np


class StreamingMultiprocessor(VoiceoverScene, MovingCameraScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        )

    title = Text("GPU programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 8 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Streaming Multiprocessor", font_size=48).next_to(title, DOWN)
    with self.voiceover(text="""I started this out as an episode on constant memory, but we have to take a short
                        detour from our memory journey to get more into the architecture of the GPU""") as trk:
      pass

    with self.voiceover(text="""So today we will be gettin into what are Streaming Multiprocessors or SM's for short""") as trk:
      self.play(Write(subtitle))

    self.play(Unwrite(title), Unwrite(subtitle))

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
    with self.voiceover(text="""But as you might imagine, the actuall gpu architecture is much more complicated""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
      self.play(Create(gpu), Write(gpu_t))



    dram = Rectangle(height=5, width=1, color=RED, fill_color=RED, fill_opacity=0.5).shift(5.25*LEFT)
    dram_t = Text("DRAM", color=RED, font_size=52).move_to(dram).rotate(PI/2)

    chip = Rectangle(height=5, width=10, color=YELLOW).shift(0.5*RIGHT)
    chip_t = Text("CHIP", color=YELLOW, font_size=24).next_to(chip, UP, buff=0.1, aligned_edge=LEFT)

    with self.voiceover(text="""As I've shown in the episode on memory, on the actual PCB we have our <bookmark mark='1'/>DRAM memory
                        and a chip <bookmark mark='2'/>  that does the calculations""") as trk:
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

    with self.voiceover(text="""So our chip contains the L2 cache that is shared between all cores""") as trk:
      self.play(Create(l2), Write(l2_t))

    mc = Rectangle(height=4.5, width=0.25).shift(4.25*LEFT)
    mc_t = Text("6 x Memory Controller", font_size=14).move_to(mc).rotate(PI/2)

    mc2 = Rectangle(height=4.5, width=0.25).shift(5.25*RIGHT)
    mc_t2 = Text("6 x Memory Controller", font_size=14).move_to(mc2).rotate(-PI/2)
    with self.voiceover(text="""It also contains 12 memory controllers""") as trk:
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

    with self.voiceover(text="""It also contains Graphic Processing Clusters, GPC's for short, in the case of 
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
                         for rasterization, namely the <bookmark mark='1'/>Raster Engine and 16 Raster Operations units <bookmark mark='2'/>divided into two partitions""") as trk:
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
    sm_t = Text("SM", font_size=48, color=MAROON).move_to(sm)
    
    sm2 = Rectangle(width=2.5, height=2, color=MAROON, fill_color=MAROON, fill_opacity=0.5).move_to(tpcs[0]).shift(1.5*DOWN)
    sm2_t = Text("SM", font_size=48, color=MAROON).move_to(sm2)
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

    with self.voiceover(text="""#TODO""") as trk:
      self.play(LaggedStart(*[Create(tex) for tex in texs], *[Write(t) for t in tex_ts]))

    l1 = Rectangle(height=0.5, width=7, color=GOLD_A, fill_color=GOLD_A, fill_opacity=0.5).move_to(sm).shift(1.7*DOWN)
    l1_t = Text("128KB L1 Cache / Shared Memory", color=GOLD_A, font_size=28).move_to(l1)

    with self.voiceover(text="""128KB of memory divided into L1 cache and shared memory""") as trk:
      self.play(Create(l1), Write(l1_t))

    cc = Rectangle(height=0.5, width=7, color=GOLD_E, fill_color=GOLD_E, fill_opacity=0.5).next_to(l1, UP, buff=0.1)
    cc_t = Text("8KB Constant Cache", color=GOLD_E, font_size=28).move_to(cc)
    with self.voiceover(text="""8KB of special cache for accesses to constant memory""") as trk:
      self.play(Create(cc), Write(cc_t))

    with self.voiceover(text="""The fact that this memory is shared is very important for us, it tells us that the more shared memory we use
                        the less L1 cache we have available""") as trk:
      pass
    ps = []
    p_ts = []


    for i in range(4):
      if i == 0:
        ps.append(Rectangle(height=4, width=1.7, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.5).move_to(sm).shift(2.7*LEFT + 0.7*UP))
      else:
        ps.append(Rectangle(height=4, width=1.7, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.5).next_to(ps[-1], RIGHT, buff=0.1))
      p_ts.append(Text("Processing Block", font_size=32, color=GREEN_A).move_to(ps[-1]).rotate(PI/2))

    with self.voiceover(text="""And it also contains 4 Processing Blocks""") as trk:
      self.play(LaggedStart(*[Create(p) for p in ps], *[Write(t) for t in p_ts]))
