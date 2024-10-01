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

class MemoryUnit(VGroup):
  def __init__(self, c=None, t=None, input=None, output=None, **kwargs):
    self.c = Capacitor() if c is None else c
    self.t = Transistor() if t is None else t
    self.out = Line().next_to(self.t.drain, LEFT, aligned_edge=DOWN, buff=0) if output is None else output
    self.inp = Line().next_to(self.t.source, RIGHT, aligned_edge=DOWN, buff=0) if input is None else input
    self.c.next_to(self.inp, LEFT, buff=0, aligned_edge=UP, submobject_to_align=self.c.out)
    self.charged = False
    super().__init__(self.c, self.t, self.out, self.inp, **kwargs)

  def write(self, scene, one=True, run_time_scale=0.3):
    set_line(self.t.base, one, scene, run_time_scale)
    set_line(self.out, one, scene, run_time_scale, backward=True)
    set_line(self.t.source, one, scene, run_time_scale, backward=True)
    set_line(self.t.l2, one, scene, run_time_scale, backward=True)
    set_line(self.t.drain, one, scene, run_time_scale, backward=True)
    set_line(self.inp, one, scene, run_time_scale, backward=True)
    set_line(self.c.out, one, scene, run_time_scale, backward=True)
    scene.play(self.c.l1.animate.set_color(GREEN if one else WHITE), 
              self.c.l2.animate.set_color(GREEN if one else WHITE))
    self.charged=one

  def disable_line(self, scene, run_time_scale=0.3):
    set_line(self.t.base, False, scene, run_time_scale)
    set_line(self.out, False, scene, run_time_scale, backward=False)
    set_line(self.t.source, False, scene, run_time_scale, backward=False)
    set_line(self.t.l2, False, scene, run_time_scale, backward=False)
    set_line(self.t.drain, False, scene, run_time_scale, backward=False)

  def read(self, scene, run_time_scale=0.3):
    set_line(self.t.base, self.charged, scene, run_time_scale)
    set_line(self.t.drain, self.charged, scene, run_time_scale, backward=True)
    set_line(self.t.l2, self.charged, scene, run_time_scale, backward=True)
    set_line(self.t.source, self.charged, scene, run_time_scale, backward=True)
    set_line(self.out, self.charged, scene, run_time_scale, backward=True)
    scene.play(*[x.animate.set_color(WHITE) for x in [self.t.drain, self.t.l2, self.t.source, self.out, self.c, self.inp]])
    self.charged=False

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
    # self.play(t.base.animate(run_time=0.2).set_color(GREEN))
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

    mem = MemoryUnit(c, t, input, output)

    mem.write(self, True)
    self.wait(1)
    
    mem.disable_line(self)
    self.wait(1)

    mem.read(self)
    self.wait(1)

