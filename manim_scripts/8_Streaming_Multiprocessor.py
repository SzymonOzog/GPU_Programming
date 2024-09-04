from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np


class StreamingMultiprocessor(VoiceoverScene):
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
