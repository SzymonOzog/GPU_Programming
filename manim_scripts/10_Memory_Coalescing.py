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
    self.drain = Line(0.2*DOWN, 0.2*UP).next_to(self.l2, LEFT, buff=0, aligned_edge=UP)
    self.source = Line(0.2*UP, 0.2*DOWN).next_to(self.l2, RIGHT, buff=0, aligned_edge=UP)
    super().__init__(self.base, self.l1, self.l2, self.drain, self.source, **kwargs)

class Capacitor(VGroup):
  def __init__(self, **kwargs):
    self.out = Line(0.5*DOWN, 0.5*UP)
    self.l1 = Line().next_to(self.out, DOWN, buff=0)
    self.l2 = Line().next_to(self.l1, DOWN, buff=0.2)
    self.inp = Line(0.5*DOWN, 0.5*UP).next_to(self.l2, DOWN, buff=0, aligned_edge=UP)
    self.cap = VGroup(self.out, self.l1, self.l2, self.inp, **kwargs)
    self.g1 = Line(0.8*LEFT, 0.8*RIGHT).next_to(self.inp, DOWN, buff=0)
    self.g2 = Line(0.5*LEFT, 0.5*RIGHT).next_to(self.g1, DOWN, buff=0.2)
    self.g3 = Line(0.2*LEFT, 0.2*RIGHT).next_to(self.g2, DOWN, buff=0.2)
    self.gnd = VGroup(self.g1, self.g2, self.g3)
    super().__init__(self.cap, self.gnd, **kwargs)

def set_line(line, enabled, scene, run_time_scale=0.3, backward=False):
  color = GREEN if enabled else WHITE
  cp = line.copy()
  scene.add(cp)
  scene.remove(line)
  line.set_color(color)
  line.scale(-1 if backward else 1)
  scene.play(Create(line, lag_ratio=0, run_time=run_time_scale*line.get_length()))
  line.scale(1)
  scene.remove(cp)

class Coalescing(VoiceoverScene, ZoomedScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        )
    t = Transistor()
    self.play(Create(t))
    b_t = Text("Base", font_size=24).next_to(t.base, UP)
    c_t = Text("Drain", font_size=24).next_to(t.drain, DOWN)
    e_t = Text("Source", font_size=24).next_to(t.source, DOWN)
    self.play(Write(c_t), Write(b_t), Write(e_t))

    
    input = Line().next_to(t.drain, LEFT, aligned_edge=DOWN, buff=0)
    output = Line().next_to(t.source, RIGHT, aligned_edge=DOWN, buff=0)

    self.play(Create(input))
    self.play(Create(output))

    set_line(input, True, self)
    self.wait(1)

    set_line(t.base, True, self)
    self.play(t.base.animate(run_time=0.2).set_color(GREEN))
    set_line(t.drain, True, self)
    set_line(t.l2, True, self)
    set_line(t.source, True, self)
    set_line(output, True, self)

    self.wait(1)

    set_line(t.base, False, self)
    set_line(t.drain, False, self)
    set_line(t.l2, False, self)
    set_line(t.source, False, self)
    set_line(output, False, self)

    self.wait(1)
    
    mem = VGroup(t, input, output)
    self.play(Unwrite(b_t), Unwrite(c_t), Unwrite(e_t), FadeOut(mem))
    mem.shift(UR)
    input.set_color(WHITE)
    c = Capacitor()
    self.play(Create(c.cap))
    self.wait(1)
    self.play(Create(c.gnd))
    self.wait(1)

    voltage = Line(RIGHT, LEFT, color=GREEN).next_to(c.out, UP, buff=0, aligned_edge=LEFT)

    self.play(Create(voltage))
    set_line(c.out, True, self, backward=True)
    self.play(c.l1.animate.set_color(GREEN), c.l2.animate.set_color(GREEN))
    self.play(Uncreate(voltage))
    
    out = Line().next_to(input, LEFT, buff=0, aligned_edge=UP)
    self.play(Create(out))
    self.play(c.animate.next_to(input, LEFT, buff=0, aligned_edge=UP, submobject_to_align=c.out))
    set_line(out, True, self, backward=True)
    self.play(c.animate.set_color(WHITE), out.animate.set_color(WHITE))
    self.play(Uncreate(out))
    self.play(FadeIn(mem))
    self.wait(1)


