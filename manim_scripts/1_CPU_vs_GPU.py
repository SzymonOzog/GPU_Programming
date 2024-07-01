from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService

class Introduction(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(
        GTTSService()
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU programming", font_size=72).shift(2*UP)
    with self.voiceover(text="Hello and welcome to episode 1 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("GPU vs CPU", font_size=48).next_to(title, DOWN)
    desc = BulletedList("Architectural Differences", "Latency and Throughput", "When is it beneficial to use a GPU", font_size=32).next_to(subtitle, DOWN)

    with self.voiceover(text="In this episode we are going to discuss the key differences between the gpu and the cpu") as trk:
      self.play(Write(subtitle))

    with self.voiceover(text="How the architecture of the two differs") as trk:
      self.play(Write(desc[0]))

    with self.voiceover(text="What is this latency and throughput stuff that is always mentioned when talking about those things") as trk:
      self.play(Write(desc[1]))

    with self.voiceover(text="and when to use one over the other") as trk:
      self.play(Write(desc[2]))

    with self.voiceover(text="And finally, we are going to crack open the editor and write some code") as trk:
      for i in range(3):
        self.play(Unwrite(desc[2-i]), run_time=trk.duration/5)
      self.play(Unwrite(subtitle), run_time=trk.duration/5)
      self.play(Unwrite(title), run_time=trk.duration/5)

    cpu_rects = []
    cpu_texts = []
    def create_alu():
      alu = Rectangle(width=1, height=1, color=BLUE, fill_opacity=0.5)
      sw = 0.04
      cpu_rects.append(alu)
      alu_text = Text("ALU", font_size=14).move_to(alu.get_center())
      cpu_texts.append(alu_text)

      cache = Rectangle(width=1, height=0.25, fill_opacity=0.5, color=RED).next_to(alu, DOWN, aligned_edge=LEFT, buff=sw)
      cpu_rects.append(cache)
      cache_text = Text("L1 Cache", font_size=14).move_to(cache.get_center())
      cpu_texts.append(cache_text)
      
      control = Rectangle(width=0.5, height=1.25+sw, color=PURPLE, fill_opacity=0.5).next_to(alu, RIGHT, aligned_edge=UP, buff=sw)
      cpu_rects.append(control)
      control_text1 = Text("Con", font_size=14)
      control_text2 = Text("trol", font_size=14)
      ct = VGroup(control_text1, control_text2).arrange(DOWN, buff=0.05).move_to(control)
      cpu_texts.append(control_text1)
      cpu_texts.append(control_text2)
      return VGroup(alu, alu_text, cache, cache_text, control, ct)


    alu1 = create_alu().shift(4*LEFT+UP)
    alu2 = create_alu().next_to(alu1, RIGHT, buff=0.1)
    alu3 = create_alu().next_to(alu1, DOWN, aligned_edge=LEFT, buff=0.1)
    alu4 = create_alu().next_to(alu3, RIGHT, buff=0.1)
    
    cache1 = Rectangle(width=alu1.width, height=0.4, color=RED, fill_opacity=0.5).next_to(alu3, DOWN, aligned_edge=LEFT, buff=0.1)
    cache2 = Rectangle(width=alu1.width, height=0.4, color=RED, fill_opacity=0.5).next_to(alu4, DOWN, aligned_edge=LEFT, buff=0.1)
    cpu_rects.append(cache1)
    cpu_rects.append(cache2)
    cpu_texts.append(Text("L2 Cache", font_size=14).move_to(cache1))
    cpu_texts.append(Text("L2 Cache", font_size=14).move_to(cache2))

    cache3 = Rectangle(width=cache1.width*2 + 0.1, height=0.5, color=RED, fill_opacity=0.5).next_to(cache1, DOWN, aligned_edge=LEFT, buff=0.1)
    cpu_rects.append(cache3)
    cpu_texts.append(Text("L3 Cache", font_size=14).move_to(cache3))


    dram_cpu = Rectangle(width=cache3.width, height=0.7, color=GREEN, fill_opacity=0.5).next_to(cache3, DOWN, buff=0.1).align_to(alu3, LEFT)
    dram_cpu_text = Text("DRAM", font_size=24).move_to(dram_cpu.get_center())
    cpu_rects.append(dram_cpu)
    cpu_texts.append(dram_cpu_text)

    gpu_rects = []
    gpu_texts = []
    gpu_alu_list = []
    for _ in range(5):
      cc = VGroup(Rectangle(width=0.5, height=0.2, color=PURPLE, fill_opacity=0.5), 
                  Rectangle(width=0.5, height=0.2, color=RED, fill_opacity=0.5)).arrange(DOWN, buff=0.1)
      gpu_rects.append(cc[0])
      gpu_rects.append(cc[1])
      alus = [Rectangle(width=0.5, height=0.5, color=BLUE, fill_opacity=0.5) for _ in range(8)]
      gpu_rects.extend(alus)
      gpu_alu_list.append(VGroup(cc, *alus).arrange(RIGHT, buff=0.1))
    gpu_alus = VGroup(*gpu_alu_list).scale(0.8).arrange(DOWN, buff=0.16).shift(RIGHT * 4)


    l2 = Rectangle(width=4.25, height=0.4, color=RED, fill_opacity=0.5).match_width(gpu_alus).next_to(gpu_alus, DOWN, buff=0.1)
    gpu_rects.append(l2)
    l2_text = Text("L2 Cache", font_size=14).move_to(l2)
    gpu_texts.append(l2_text)

    dram_gpu = Rectangle(width=4.25, height=0.5, color=GREEN, fill_opacity=0.5).match_width(gpu_alus).next_to(l2, DOWN, buff=0.1)
    gpu_rects.append(dram_gpu)
    dram_gpu_text = Text("DRAM", font_size=14).move_to(dram_gpu.get_center())
    gpu_texts.append(dram_gpu_text)

    cpu = VGroup(*cpu_rects, *cpu_texts, dram_cpu, dram_cpu_text)

    gpu = VGroup(gpu_alus, l2, l2_text, dram_gpu, dram_gpu_text).match_height(cpu).align_to(cpu, UP)

    cpu_title = Text("CPU").scale(0.8).next_to(cpu, UP)
    cpu_texts.append(cpu_title)
    gpu_title = Text("GPU").scale(0.8).next_to(gpu, UP)
    gpu_texts.append(gpu_title)

    subobjects = []
    queue = [cpu, gpu]
    while queue:
      o = queue.pop()
      subobjects.append(o)
      queue.extend(o.submobjects)

    for mo in subobjects:
      for so in mo.submobjects.copy():
        if any(so in x for x in [cpu_rects, cpu_texts, gpu_rects, gpu_texts]):
          mo.remove(so)

    with self.voiceover(text="What you are seeing right now on the screen is a highly simplified comparison of a cpu and gpu architectures") as trk:
      self.play(*[Create(x) for x in cpu_rects])
      self.play(*[Write(x) for x in cpu_texts])
      self.wait(1)

      self.play(*[Create(x) for x in gpu_rects])
      self.play(*[Write(x) for x in gpu_texts])

    with self.voiceover(text="""The real architecture is obvoiusly much more complicated, but this simplification will 
                        help us understand the key differences """) as trk:
      pass

    with self.voiceover(text="""First of all, we can see that the GPU consists of a much greated number of cores
                        but that comes at a cost, as the CPU cores are much more capable""") as trk:
      self.wait(1)
      self.play(*[Indicate(x, run_time=3) for x in cpu_rects + gpu_rects if x.color == BLUE])

    with self.voiceover(text="""Secondly, the CPU has a much deeper memory hierarchy, allowing for much lower memory access latency""") as trk:
      self.wait(1)
      self.play(*[Indicate(x, run_time=3) for x in cpu_rects if x.color == RED])

    with self.voiceover(text="""Also, we can see that there are thread groups in the GPU that share the control units,
                        that can point us to the fact that they all must execute exatly the same instruction at the same time""") as trk:
      self.wait(2)
      self.play(*[Indicate(x, run_time=3) for x in gpu_rects if x.color == PURPLE])
