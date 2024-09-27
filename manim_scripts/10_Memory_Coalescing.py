from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import random
import math
from math import radians

class Transistor(VGroup):
  def __init__(self, **kwargs):
    self.base = Line(UP, DOWN)
    self.l1 = Line().next_to(self.base, DOWN, buff=0)
    self.l2 = Line().next_to(self.l1, DOWN, buff=0.2)
    self.collector = Line(0.1*DOWN, UP).next_to(self.l2, LEFT, buff=0, aligned_edge=UP)
    self.emitter = Line(UP, 0.1*DOWN).next_to(self.l2, RIGHT, buff=0, aligned_edge=UP)
    super().__init__(self.base, self.l1, self.l2, self.collector, self.emitter, **kwargs)

def set_line(line, enabled, scene, run_time_scale=0.3):
  color = GREEN if enabled else WHITE
  cp = line.copy()
  scene.add(cp)
  scene.remove(line)
  line.set_color(color)
  scene.play(Create(line, lag_ratio=0, run_time=run_time_scale*line.get_length()))
  scene.remove(cp)

class Coalescing(VoiceoverScene, ZoomedScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        )
    t = Transistor()
    self.play(Create(t))
    b_t = Text("Base", font_size=24).next_to(t.base, UP)
    c_t = Text("Collector", font_size=24).next_to(t.collector, DOWN)
    e_t = Text("Emitter", font_size=24).next_to(t.emitter, DOWN)
    self.play(Write(c_t), Write(b_t), Write(e_t))

    
    input = Line().next_to(t.collector, LEFT, aligned_edge=DOWN, buff=0)
    output = Line().next_to(t.emitter, RIGHT, aligned_edge=DOWN, buff=0)

    self.play(Create(input))
    self.play(Create(output))

    set_line(input, True, self)
    self.wait(1)

    set_line(t.base, True, self)
    set_line(t.collector, True, self)
    set_line(t.l2, True, self)
    set_line(t.emitter, True, self)
    set_line(output, True, self)

    self.wait(1)

    set_line(t.base, False, self)
    set_line(t.collector, False, self)
    set_line(t.l2, False, self)
    set_line(t.emitter, False, self)
    set_line(output, False, self)

    self.wait(1)
