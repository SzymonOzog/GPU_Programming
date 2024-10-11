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
    self.l2 = Line().next_to(self.l1, DOWN, buff=0.3)
    self.drain = Line(0.2*DOWN, 0.2*UP).next_to(self.l2, LEFT, buff=0, aligned_edge=UP)
    self.source = Line(0.2*UP, 0.2*DOWN).next_to(self.l2, RIGHT, buff=0, aligned_edge=UP)
    super().__init__(self.base, self.l1, self.l2, self.drain, self.source, **kwargs)

class Capacitor(VGroup):
  def __init__(self, charge=0, **kwargs):
    color = WHITE.interpolate(GREEN, charge)
    self.out = Line(0.5*DOWN, 0.5*UP, color=color)
    self.l1 = Line(color=color).next_to(self.out, DOWN, buff=0)
    self.l2 = Line(color=color).next_to(self.l1, DOWN, buff=0.3)
    self.inp = Line(0.5*DOWN, 0.5*UP).next_to(self.l2, DOWN, buff=0, aligned_edge=UP)
    self.cap = VGroup(self.out, self.l1, self.l2, self.inp, **kwargs)
    self.g1 = Line(0.8*LEFT, 0.8*RIGHT).next_to(self.inp, DOWN, buff=0)
    self.g2 = Line(0.5*LEFT, 0.5*RIGHT).next_to(self.g1, DOWN, buff=0.2)
    self.g3 = Line(0.2*LEFT, 0.2*RIGHT).next_to(self.g2, DOWN, buff=0.2)
    self.gnd = VGroup(self.g1, self.g2, self.g3)
    super().__init__(self.cap, self.gnd, **kwargs)

class MemoryUnit(VGroup):
  def __init__(self, c=None, t=None, input=None, output=None, charge=0, **kwargs):
    color = WHITE.interpolate(GREEN, charge)
    self.c = Capacitor(charge=charge) if c is None else c
    self.t = Transistor() if t is None else t
    self.inp = Line(color=color).next_to(self.t.drain, LEFT, aligned_edge=DOWN, buff=0) if input is None else input
    self.out = Line().next_to(self.t.source, RIGHT, aligned_edge=DOWN, buff=0) if output is None else output
    self.c.next_to(self.inp, LEFT, buff=0, aligned_edge=UP, submobject_to_align=self.c.out)
    self.charged = charge
    super().__init__(self.c, self.t, self.out, self.inp, **kwargs)

  def write(self, scene, alpha=1, run_time_scale=0.3):
    anims = []
    anims.append(set_line(self.t.base, alpha, scene, run_time_scale))
    anims.append(set_line(self.out, alpha, scene, run_time_scale, backward=True))
    anims.append(set_line(self.t.source, alpha, scene, run_time_scale, backward=True))
    anims.append(set_line(self.t.l2, alpha, scene, run_time_scale, backward=True))
    anims.append(set_line(self.t.drain, alpha, scene, run_time_scale, backward=True))
    anims.append(set_line(self.inp, alpha, scene, run_time_scale, backward=True))
    anims.append(set_line(self.c.out, alpha, scene, run_time_scale, backward=True))
    color = WHITE.interpolate(GREEN, alpha) 
    anims.append([self.c.l1.animate.set_color(color), 
              self.c.l2.animate.set_color(color)])
    self.charged=alpha
    return anims

  def disable_line(self, scene, alpha=0, run_time_scale=0.3):
    anims = []
    anims.append(set_line(self.t.base, False, scene, run_time_scale))
    anims.append(set_line(self.out, False, scene, run_time_scale, backward=False))
    anims.append(set_line(self.t.source, False, scene, run_time_scale, backward=False))
    anims.append(set_line(self.t.l2, False, scene, run_time_scale, backward=False))
    anims.append(set_line(self.t.drain, False, scene, run_time_scale, backward=False))
    return anims

  def read(self, scene, alpha=0, run_time_scale=0.3):
    color = WHITE.interpolate(GREEN, alpha) 
    anims = []
    anims.append(set_line(self.t.base, self.charged, scene, run_time_scale))
    anims.append(set_line(self.t.drain, self.charged, scene, run_time_scale, backward=True))
    anims.append(set_line(self.t.l2, self.charged, scene, run_time_scale, backward=True))
    anims.append(set_line(self.t.source, self.charged, scene, run_time_scale, backward=True))
    anims.append(set_line(self.out, self.charged, scene, run_time_scale, backward=True))
    anims.append([x.animate.set_color(color) for x in [self.t.drain, self.t.l2, self.t.source, self.out, self.c, self.inp]])
    self.charged=alpha
    return anims

def set_line(line, alpha, scene, run_time_scale=0.3, backward=False):
  color = WHITE.interpolate(GREEN, alpha) 
  cp = line.copy()
  scene.add(cp)
  scene.remove(line)
  line.set_color(color)
  line.scale(-1 if backward else 1)
  def on_finish(scene):
    line.scale(1)
    scene.remove(cp)
  return [Create(line, lag_ratio=0, _on_finish=on_finish, run_time=run_time_scale*line.get_length())]

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

    self.play(*set_line(input, 1, self))
    self.wait(1)

    self.play(*set_line(t.base, 1, self))
    self.play(*set_line(t.drain, 1, self))
    self.play(*set_line(t.l2, 1, self))
    self.play(*set_line(t.source, 1, self))
    self.play(*set_line(output, 1, self))

    self.wait(1)

    self.play(*set_line(t.base, 0, self))
    self.play(*set_line(t.drain, 0, self))
    self.play(*set_line(t.l2, 0, self))
    self.play(*set_line(t.source, 0, self))
    self.play(*set_line(output, 0, self))

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
    self.play(*set_line(c.out, True, self, backward=True))
    self.play(c.l1.animate.set_color(GREEN), c.l2.animate.set_color(GREEN))
    self.play(Uncreate(voltage))
    
    out = Line().next_to(input, LEFT, buff=0, aligned_edge=UP)
    self.play(Create(out))
    self.play(c.animate.next_to(input, LEFT, buff=0, aligned_edge=UP, submobject_to_align=c.out))
    self.play(*set_line(out, True, self, backward=True))
    self.play(c.animate.set_color(WHITE), out.animate.set_color(WHITE))
    self.play(Uncreate(out))
    self.play(FadeIn(mem))
    self.wait(1)

    mem = MemoryUnit(c, t, input, output)

    for a in mem.write(self, 1):
      self.play(*a)
    self.wait(1)
    
    for a in mem.disable_line(self):
      self.play(*a)
    self.wait(1)

    for a in mem.read(self):
      self.play(*a)

    self.wait(1)

    for a in mem.disable_line(self):
      self.play(*a)

    mem_s = 0.2
    mem.scale(mem_s).shift(2*UR).to_edge(UL)
    mems = [mem]
    charges = [   1, 0, 0,
               1, 0, 0, 1,
               0, 1, 1, 1, 
               1, 0, 1, 1]

    for i in range(15):
      mems.append(MemoryUnit(charge = charges[i]).scale(mem_s))
    VGroup(*mems).arrange_in_grid(4,4)
    self.play(*[Create(x) for x in mems])

    sz_w = 3.5
    sz_b = 2.4
    bit_lines = [Line(sz_b*UP, sz_b*DOWN).next_to(mems[i].out, RIGHT, aligned_edge=UP, buff=0) for i in range(4)]
    word_lines = [Line(sz_w*LEFT, sz_w*RIGHT).next_to(mems[i*4].t.base, UP, aligned_edge=LEFT, buff=0) for i in range(4)]
    self.play(*[Create(x) for x in bit_lines])
    self.play(*[Create(x) for x in word_lines])

    rd = Rectangle(height=5, width=0.5).next_to(VGroup(*word_lines), RIGHT, buff=0)
    sa = Rectangle(height=0.5, width=6).next_to(VGroup(*bit_lines), DOWN, buff=0)
    rd_t = Text("Row Decoder", font_size=32).rotate(PI/2).move_to(rd)
    sa_t = Text("Sense Amplifiers", font_size=32, z_index=1).move_to(sa)

    self.play(Create(rd), Write(rd_t))
    self.play(Create(sa), Write(sa_t))
    
    sz=0.2
    decoder_lines = [Line(sz*UP, sz*DOWN).next_to(sa, DOWN, buff=0).set_x(bit_lines[i].get_x()) for i in range(4)]

    self.play(*[Create(x) for x in decoder_lines])

    cd = Rectangle(height=0.5, width=6).next_to(VGroup(*decoder_lines), DOWN, buff=0)
    cd_t = Text("Column Decoder", font_size=32).move_to(cd)
    self.play(Create(cd), Write(cd_t))

    data_out = Line().next_to(sa, RIGHT, buff=0)
    self.play(Create(data_out))

    st = rd.get_top() + DOWN + 3*RIGHT
    e1 = st.copy()
    e1[1] = rd.get_y()
    e2 = st.copy()
    e2[1] = cd.get_y()
    addres_lines = [Line(st+0.5*LEFT, e1+0.5*LEFT-0.1*DOWN),
                    Line(st+0.25*LEFT, e1+0.25*LEFT+0.1*DOWN),
                    Line(st, e2-0.1*DOWN),
                    Line(st+0.25*RIGHT, e2+0.25*RIGHT+0.1*DOWN)]

    addres_lines2 = []
    for i in range(4):
      end_obj = rd if i < 2 else cd
      st = addres_lines[i].get_end()
      end = st.copy()
      end[0] = end_obj.get_right()[0]
      addres_lines2.append(Line(st, end))

    self.play(*[Create(x) for x in addres_lines])
    self.play(*[Create(x) for x in addres_lines2])

    pre_charge_value = 0.7
    chrg_hi = 0.8
    chrg_lo = 0.5
    anims = []
    for i in range(4):
      anims.extend(set_line(bit_lines[i], pre_charge_value, self))

    self.play(*anims)
    address = [Text(x, font_size=24).next_to(addres_lines[i], UP, buff=1) for i, x in enumerate("0100")]
    self.play(*[Write(x) for x in address])
    self.play(address[0].animate.next_to(addres_lines[0], UP),
              address[1].animate.next_to(addres_lines[1], UP))
    self.play(*set_line(addres_lines[1], 1, self))
    self.play(*set_line(addres_lines2[1], 1, self))
    self.play(*set_line(word_lines[1], 1, self, backward=True))
    spotlight = Exclusion(Rectangle(width=100, height=100), SurroundingRectangle(mems[4], buff=0.1), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)
    self.play(FadeIn(spotlight))

    for a in mems[4].read(self, (mems[4].charged + pre_charge_value)/2):
      self.play(*a)
    self.play(*set_line(bit_lines[0], (chrg_hi if mems[4].charged > pre_charge_value else chrg_lo), self))

    
    self.play(Transform(spotlight,
                        Exclusion(Rectangle(width=100, height=100), SurroundingRectangle(mems[5], buff=0.1), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)))
    for a in mems[5].read(self, (chrg_hi if mems[5].charged > pre_charge_value else chrg_lo)):
      self.play(*a)
    self.play(*set_line(bit_lines[1], (chrg_hi if mems[5].charged > pre_charge_value else chrg_lo), self))

    self.play(Transform(spotlight,
                        Exclusion(Rectangle(width=100, height=100), SurroundingRectangle(VGroup(mems[6], mems[7]), buff=0.1), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)))

    for a1, a2 in zip(mems[6].read(self, (chrg_hi if mems[6].charged > pre_charge_value else chrg_lo)), mems[7].read(self, (chrg_hi if mems[7].charged > pre_charge_value else chrg_lo))):
      self.play(*a1, *a2)
    self.play(*set_line(bit_lines[2], (chrg_hi if mems[6].charged > pre_charge_value else chrg_lo), self),
              *set_line(bit_lines[3], (chrg_hi if mems[7].charged > pre_charge_value else chrg_lo), self))

    self.play(Transform(spotlight,
                        Exclusion(Rectangle(width=100, height=100), SurroundingRectangle(sa, buff=0.1), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)))

    vals = []
    for i, b in enumerate(bit_lines):
      vals.append(Rectangle(width=0.5, height=0.5, color=GREEN if mems[4+i].charged > pre_charge_value else WHITE, fill_opacity=0.5).next_to(b, DOWN, buff=0))
    self.play(*[Create(x) for x in vals])
    self.play(FadeOut(spotlight))

    self.play(address[2].animate.next_to(addres_lines[2], UP),
              address[3].animate.next_to(addres_lines[3], UP))
    self.play(*set_line(decoder_lines[0], 1, self))
    self.play(sa.animate.set_color(vals[0].color))
    self.play(*set_line(data_out, 1 if vals[0].color == GREEN else 0, self))

    self.play(address[2].animate.next_to(addres_lines[2], UP, buff=1),
              address[3].animate.next_to(addres_lines[3], UP, buff=1))

    self.play(*set_line(decoder_lines[0], 0, self))
    self.play(sa.animate.set_color(WHITE))
    self.play(*set_line(data_out, 0, self))

    anims = []
    for i in range(4):
      anims.append(FadeOut(vals[i]))
      anims.extend(set_line(bit_lines[i], 1 if vals[i].color == GREEN else 0, self, backward=True))
    self.play(*anims)

    for a1, a2, a3, a4 in zip(mems[4].write(self, 1 if vals[0].color == GREEN else 0), 
                              mems[5].write(self, 1 if vals[1].color == GREEN else 0), 
                              mems[6].write(self, 1 if vals[2].color == GREEN else 0), 
                              mems[7].write(self, 1 if vals[3].color == GREEN else 0)):
      self.play(*a1, *a2, *a3, *a4)

    self.play(address[0].animate.next_to(addres_lines[0], UP, buff=1),
              address[1].animate.next_to(addres_lines[1], UP, buff=1))

    self.play(*set_line(addres_lines[1], 0, self))
    self.play(*set_line(addres_lines2[1], 0, self))
    self.play(*set_line(word_lines[1], 0, self, backward=True))

    for a1, a2, a3, a4 in zip(mems[4].disable_line(self, 0), 
                              mems[5].disable_line(self, 0), 
                              mems[6].disable_line(self, 0), 
                              mems[7].disable_line(self, 0)):
      self.play(*a1, *a2, *a3, *a4)

    anims = []
    for i in range(4):
      anims.extend(set_line(bit_lines[i], 0, self))
    self.play(*anims)

    self.play(Transform(address[0], Text("1", font_size=24).next_to(addres_lines[0], UP, buff=1)),
              Transform(address[1], Text("0", font_size=24).next_to(addres_lines[1], UP, buff=1)))

    anims = []
    for i in range(4):
      anims.extend(set_line(bit_lines[i], pre_charge_value, self))

    self.play(*anims)

    self.play(address[0].animate.next_to(addres_lines[0], UP),
              address[1].animate.next_to(addres_lines[1], UP))
    self.play(*set_line(addres_lines[0], 1, self))
    self.play(*set_line(addres_lines2[0], 1, self))
    self.play(*set_line(word_lines[2], 1, self, backward=True))

    for a1, a2, a3, a4 in zip(mems[8].read(self, (chrg_hi if mems[8].charged > pre_charge_value else chrg_lo)), 
                              mems[9].read(self, (chrg_hi if mems[9].charged > pre_charge_value else chrg_lo)), 
                              mems[10].read(self, (chrg_hi if mems[10].charged > pre_charge_value else chrg_lo)), 
                              mems[11].read(self, (chrg_hi if mems[11].charged > pre_charge_value else chrg_lo))):
      self.play(*a1, *a2, *a3, *a4)

    self.play(*set_line(bit_lines[0], (chrg_hi if mems[8].charged > pre_charge_value else chrg_lo), self),
              *set_line(bit_lines[1], (chrg_hi if mems[9].charged > pre_charge_value else chrg_lo), self),
              *set_line(bit_lines[2], (chrg_hi if mems[10].charged > pre_charge_value else chrg_lo), self),
              *set_line(bit_lines[3], (chrg_hi if mems[11].charged > pre_charge_value else chrg_lo), self))

    vals = []
    for i, b in enumerate(bit_lines):
      vals.append(Rectangle(width=0.5, height=0.5, color=GREEN if mems[8+i].charged > pre_charge_value else WHITE, fill_opacity=0.5).next_to(b, DOWN, buff=0))
    self.play(*[Create(x) for x in vals])

    self.play(address[2].animate.next_to(addres_lines[2], UP),
              address[3].animate.next_to(addres_lines[3], UP))
    self.play(*set_line(decoder_lines[0], 1, self))
    self.play(sa.animate.set_color(vals[0].color))
    self.play(*set_line(data_out, 1 if vals[0].color == GREEN else 0, self))

    for i, v in enumerate(["01", "10", "11"]):
      self.play(address[2].animate.next_to(addres_lines[2], UP, buff=1),
                address[3].animate.next_to(addres_lines[3], UP, buff=1))
      self.play(*set_line(addres_lines[2], 0, self), *set_line(addres_lines[3], 0, self))
      self.play(*set_line(addres_lines2[2], 0, self), *set_line(addres_lines2[3], 0, self))

      self.play(*set_line(decoder_lines[i], 0, self))
      self.play(sa.animate.set_color(WHITE))
      self.play(*set_line(data_out, 0, self))

      self.play(Transform(address[2], Text(v[0], font_size=24).next_to(addres_lines[2], UP, buff=1)),
                Transform(address[3], Text(v[1], font_size=24).next_to(addres_lines[3], UP, buff=1)))
      self.play(address[2].animate.next_to(addres_lines[2], UP),
                address[3].animate.next_to(addres_lines[3], UP))
      self.play(*set_line(addres_lines[2], int(v[0]), self), *set_line(addres_lines[3], int(v[1]), self))
      self.play(*set_line(addres_lines2[2], int(v[0]), self), *set_line(addres_lines2[3], int(v[1]), self))
      self.play(*set_line(decoder_lines[i+1], 1, self))
      self.play(sa.animate.set_color(vals[i+1].color))
      self.play(*set_line(data_out, 1 if vals[i+1].color == GREEN else 0, self))


    self.play(*[FadeOut(x) for x in self.mobjects])
    strided_cp = """__global__ void copy(int n , float* in, float* out, int stride)
    {
      int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n)
      {
        out[i] = in[i * stride];
      }
    }"""

    code_obj = Code(code=strided_cp, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1)
    self.play(Create(code_obj))

    self.wait(1)
    self.play(Uncreate(code_obj))

    timings = [0.003619, 0.004091, 0.004631, 0.005690, 0.005933, 0.006505, 0.006846, 0.007276, 0.009585, 0.009903, 0.010967, 0.011174, 0.011700, 0.011849, 0.012079, ]
    timings = [t*1e6 for t in timings]
    stride = list(range(1, 15))
    # stride = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    ax = Axes(x_range=[1, stride[-1]+1, 1],
              y_range=[3000, 13000, 1000],
              x_axis_config={"scaling": LogBase(2)},
              axis_config={"include_numbers": True}).scale(0.8)

    graph = ax.plot_line_graph(x_values=[2**x for x in stride], y_values=timings, line_color=BLUE, add_vertex_dots=False) 

    x_label = ax.get_x_axis_label("Stride")
    y_label = ax.get_y_axis_label("Time [\\mu s]")

    self.play(Create(ax))
    self.play(Create(graph))
    self.play(Write(x_label), Write(y_label))


