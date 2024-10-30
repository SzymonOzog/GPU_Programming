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
    anims.append(set_line(self.out, False, scene, run_time_scale, backward=True))
    anims.append(set_line(self.t.source, False, scene, run_time_scale, backward=True))
    anims.append(set_line(self.t.l2, False, scene, run_time_scale, backward=True))
    anims.append(set_line(self.t.drain, False, scene, run_time_scale, backward=True))
    return anims

  def read(self, scene, alpha=0, run_time_scale=0.3):
    color = WHITE.interpolate(GREEN, alpha) 
    anims = []
    end = [x.animate.set_color(color) for x in [self.t.drain, self.t.l2, self.t.source, self.out, self.c.out, self.c.l1, self.c.l2, self.inp]]
    anims.append(set_line(self.t.base, 1, scene, run_time_scale))
    anims.append(set_line(self.t.drain, self.charged, scene, run_time_scale, backward=False))
    anims.append(set_line(self.t.l2, self.charged, scene, run_time_scale, backward=False))
    anims.append(set_line(self.t.source, self.charged, scene, run_time_scale, backward=False))
    anims.append(set_line(self.out, self.charged, scene, run_time_scale, backward=False))
    anims.append(end)
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
    line.scale(-1 if backward else 1)
    scene.remove(cp)
  rt = max(0.2, run_time_scale*(line.get_length()*4/line.get_stroke_width()))
  return [Create(line, lag_ratio=0, _on_finish=on_finish, run_time=rt)]

class Coalescing(VoiceoverScene, ZoomedScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        )
    Rectangle.set_default(stroke_width=10)
    Line.set_default(stroke_width=15)
    t = Transistor()
    input = Line().next_to(t.drain, LEFT, aligned_edge=DOWN, buff=0)
    output = Line().next_to(t.source, RIGHT, aligned_edge=DOWN, buff=0)
    mem = VGroup(t, input, output)
    mem.shift(UR)
    c = Capacitor()
    out = Line().next_to(input, LEFT, buff=0, aligned_edge=UP)

    mem = MemoryUnit(c, t, input, output)
    mem_s = 1
    mem.scale(mem_s).shift(2*UR).to_edge(UL)
    mems = [mem]
    charges = [   1, 0, 0,
               1, 0, 0, 1,
               0, 1, 1, 1, 
               1, 0, 1, 1]

    for i in range(15):
      mems.append(MemoryUnit(charge = charges[i]).scale(mem_s))
    VGroup(*mems).arrange_in_grid(4,4)

    sz_w = 3.5*5
    sz_b = 2.4*5
    bit_lines = [Line(sz_b*UP, sz_b*DOWN).next_to(mems[i].out, RIGHT, aligned_edge=UP, buff=0) for i in range(4)]
    word_lines = [Line(sz_w*LEFT, sz_w*RIGHT).next_to(mems[i*4].t.base, UP, aligned_edge=LEFT, buff=0) for i in range(4)]

    rd = Rectangle(height=5*5, width=0.5*5).next_to(VGroup(*word_lines), RIGHT, buff=0)
    sa = Rectangle(height=0.5*5, width=6*5).next_to(VGroup(*bit_lines), DOWN, buff=0)
    rd_t = Text("Row Decoder", font_size=32*5).rotate(PI/2).move_to(rd)
    sa_t = Text("Sense Amplifiers", font_size=32*4, z_index=1).move_to(sa)

    
    sz=0.2*5
    decoder_lines = [Line(sz*UP, sz*DOWN).next_to(sa, DOWN, buff=0).set_x(bit_lines[i].get_x()) for i in range(4)]


    cd = Rectangle(height=0.5*5, width=6*5).next_to(VGroup(*decoder_lines), DOWN, buff=0)
    cd_t = Text("Column Decoder", font_size=32*4).move_to(cd)

    data_out = Line().next_to(sa, RIGHT, buff=0)

    all = VGroup(*mems, *bit_lines, *word_lines, rd, sa, *decoder_lines, cd, data_out)
    self.camera.frame.save_state()
    self.camera.auto_zoom(all, margin=4, animate=False)

    with self.voiceover(text="""Hello and welcome to the next episode in our GPU series, this episode will focus on how memory, and specifically
                        DRAM memory works in modern GPUs and why should we care about such low level detais as GPU programmers. Without further ado let's jump in""") as trk:
      self.play(*[Create(x) for x in mems])
      self.play(*[Create(x) for x in bit_lines])
      self.play(*[Create(x) for x in word_lines])
      self.play(Create(rd), Write(rd_t))
      self.play(Create(sa), Write(sa_t))
      self.play(*[Create(x) for x in decoder_lines])
      self.play(Create(cd), Write(cd_t))
      self.play(Create(data_out))

    self.play([FadeOut(x) for x in self.mobjects])

    self.camera.frame.restore()
    Rectangle.set_default()
    Line.set_default()

    t = Transistor()

    with self.voiceover(text="""To understand how memory works, we have to go into electronic component level of detais, and we'll start with the transistor""") as trk:
      self.play(Create(t))
    b_t = Text("Base", font_size=24).next_to(t.base, UP)
    c_t = Text("Drain", font_size=24).next_to(t.drain, DOWN)
    e_t = Text("Source", font_size=24).next_to(t.source, DOWN)
    with self.voiceover(text="""The transistor has 3 entry points, the <bookmark mark='1'/>Base, the<bookmark mark='2'/> Drain  and the <bookmark mark='3'/> source""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(b_t))
      self.wait_until_bookmark("2")
      self.play(Write(c_t))
      self.wait_until_bookmark("3")
      self.play(Write(e_t))
    
    input = Line().next_to(t.drain, LEFT, aligned_edge=DOWN, buff=0)
    output = Line().next_to(t.source, RIGHT, aligned_edge=DOWN, buff=0)

    with self.voiceover(text="""We can now plug in some input line and output line to our transistor""") as trk:
      self.play(Create(input))
      self.play(Create(output))

    with self.voiceover(text="""In this state, if we put the voltage through the transistor, it won't pass through
                        and the output won't have voltage""") as trk:
      self.play(*set_line(input, 1, self))

    with self.voiceover(text="""But if we also put the voltage on the base the transistor starts being open, and voltage
                        passes through to the output""") as trk:
      self.play(*set_line(t.base, 1, self))
      self.play(*set_line(t.drain, 1, self))
      self.play(*set_line(t.l2, 1, self))
      self.play(*set_line(t.source, 1, self))
      self.play(*set_line(output, 1, self))

    self.wait(1)

    with self.voiceover(text="""If we turn off the voltage from the base, the transistor turns off and once again, no voltage can pass through it""") as trk:
      self.play(*set_line(t.base, 0, self))
      self.play(*set_line(t.drain, 0, self))
      self.play(*set_line(t.l2, 0, self))
      self.play(*set_line(t.source, 0, self))
      self.play(*set_line(output, 0, self))

    
    mem = VGroup(t, input, output)
    self.play(Unwrite(b_t), Unwrite(c_t), Unwrite(e_t), FadeOut(mem))
    mem.shift(UR)
    input.set_color(WHITE)
    c = Capacitor()
    with self.voiceover(text="""The second component that we need to know about is the capacitor and for simplicity, we'll 
                        assume that it's always<bookmark mark='1'/> connected to the ground""") as trk:
      self.play(Create(c.cap))
      self.wait_until_bookmark("1")
      self.play(Create(c.gnd))

    voltage = Line(RIGHT, LEFT, color=GREEN).next_to(c.out, UP, buff=0, aligned_edge=LEFT)

    with self.voiceover(text="""We can use the capacitor to store charge""") as trk:
      pass
    
    with self.voiceover(text="""This means that when we put voltage on it's imput<bookmark mark='1'/> it starts accumulating charge""") as trk:
      self.wait_until_bookmark("1")
      self.play(Create(voltage))
      self.play(*set_line(c.out, True, self, backward=True))
      self.play(c.l1.animate.set_color(GREEN), c.l2.animate.set_color(GREEN))
      self.play(Uncreate(voltage))
    
    out = Line().next_to(input, LEFT, buff=0, aligned_edge=UP)
    with self.voiceover(text="""And when we plug it in again, it slowly discharges""") as trk:
      self.play(Create(out))
      self.play(c.animate.next_to(input, LEFT, buff=0, aligned_edge=UP, submobject_to_align=c.out))
      self.play(*set_line(out, True, self, backward=True))
      self.play(c.animate.set_color(WHITE), out.animate.set_color(WHITE))


    with self.voiceover(text="""With that in mind, we can combine those two elements and create a simple memory cell""") as trk:
      self.play(Uncreate(out))
      self.play(FadeIn(mem))

    mem = MemoryUnit(c, t, input, output)

    with self.voiceover(text="""We can open out transistor, and put voltage on the source to charge our capacitor and store a 1""") as trk:
      for a in mem.write(self, 1):
        self.play(*a)
    
    with self.voiceover(text="""If we then disable the transistor, the capacitor stores our value""") as trk:
      for a in mem.disable_line(self):
        self.play(*a)

    with self.voiceover(text="""And if we want to read it, we just open our transistor again""") as trk:
      for a in mem.read(self):
        self.play(*a)

    for a in mem.disable_line(self):
      self.play(*a)

    self.play(FadeOut(mem))

    self.camera.frame.save_state()
    self.camera.auto_zoom(all, margin=4, animate=False)
    Rectangle.set_default(stroke_width=10)
    Line.set_default(stroke_width=15)
    with self.voiceover(text="""Those memory cells are arranged in a rectangular grid, called a memory array""") as trk:
      self.play(*[Create(x) for x in mems])

    with self.voiceover(text="""The bases of each transistors are connected using word lines""") as trk:
      self.play(*[Create(x) for x in word_lines])

    with self.voiceover(text="""the outputs are connected to the bit lines""") as trk:
      self.play(*[Create(x) for x in bit_lines])

    with self.voiceover(text="""The word lines are coming out of a row decoder""") as trk:
      self.play(Create(rd), Write(rd_t))

    with self.voiceover(text="""the bit lines are connected to sense amplifiers""") as trk:
      self.play(Create(sa), Write(sa_t))
    
    with self.voiceover(text="""That are controlled by a column decoder""") as trk:
      self.play(*[Create(x) for x in decoder_lines])
      self.play(Create(cd), Write(cd_t))

    with self.voiceover(text="""And finally everything is connected to the data input and output line""") as trk:
      self.play(Create(data_out))

    st = rd.get_top() + DOWN + 6*RIGHT
    e1 = st.copy()
    e1[1] = rd.get_y()
    e2 = st.copy()
    e2[1] = cd.get_y()
    addres_lines = [Line(st+5*0.5*LEFT, e1+5*0.5*LEFT-5*0.1*DOWN),
                    Line(st+5*0.25*LEFT, e1+5*0.25*LEFT+5*0.1*DOWN),
                    Line(st, e2-5*0.1*DOWN),
                    Line(st+5*0.25*RIGHT, e2+5*0.25*RIGHT+5*0.1*DOWN)]

    addres_lines2 = []
    for i in range(4):
      end_obj = rd if i < 2 else cd
      st = addres_lines[i].get_end()
      end = st.copy()
      end[0] = end_obj.get_right()[0]
      addres_lines2.append(Line(st, end))

    with self.voiceover(text="""Now that we have our memory array constructed, we need a way to control it""") as trk:
      pass

    with self.voiceover(text="""And we do that through address lines, some that represent the row that we want to access, and some that represent
                        the columns""") as trk:
      self.play(*[Create(x) for x in addres_lines])
      self.play(*[Create(x) for x in addres_lines2])

    pre_charge_value = 0.7
    chrg_hi = 0.8
    chrg_lo = 0.5
    anims = []
    for i in range(4):
      anims.extend(set_line(bit_lines[i], pre_charge_value, self))


    with self.voiceover(text="""With all of the elements in place, we can now see how to read a value from our memory array""") as trk:
      pass

    def animate_access(mem):
      if mem.charged > pre_charge_value:
        return mem.read(self, chrg_hi)
      return mem.write(self, chrg_lo)
    with self.voiceover(text="""The first step is the precharging step, where we put some voltage on our bitlines, that is non zero but lower than the 
                        voltage stored in our transistors""") as trk:
      self.play(*anims)
    address = [Text(x, font_size=5*24).next_to(addres_lines[i], UP, buff=1) for i, x in enumerate("0100")]
    with self.voiceover(text="""Then we need to provide the address of the memory that we want to access""") as trk:
      self.play(*[Write(x) for x in address])
    with self.voiceover(text="""The high part of the address goes to the row decoder""") as trk:
      self.play(address[0].animate.next_to(addres_lines[0], UP),
                address[1].animate.next_to(addres_lines[1], UP))
      self.play(*set_line(addres_lines[1], 1, self))
      self.play(*set_line(addres_lines2[1], 1, self))

    with self.voiceover(text="""That decodes it and activates a whole row of transistors""") as trk:
      self.play(*set_line(word_lines[1], 1, self, backward=True))
    spotlight = Exclusion(Rectangle(width=100, height=100), SurroundingRectangle(mems[4], buff=0.5), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)
    with self.voiceover(text="""If a memory cell was storing a value of 1, it starts discharging into the bitline increasing it's voltage""") as trk:
      self.play(FadeIn(spotlight))
      for a in mems[4].read(self, (mems[4].charged + pre_charge_value)/2):
        self.play(*a)
      self.play(*set_line(bit_lines[0], (chrg_hi if mems[4].charged > pre_charge_value else chrg_lo), self))

    
    with self.voiceover(text="""Likewhise, if the memory cell stored a 0, it starts draining from the bitline decreasing it's voltage""") as trk:
      self.play(Transform(spotlight,
                          Exclusion(Rectangle(width=100, height=100), SurroundingRectangle(mems[5], buff=0.5), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)))
      for a in mems[5].read(self, (chrg_hi if mems[5].charged > pre_charge_value else chrg_lo)):
        self.play(*a)
      self.play(*set_line(bit_lines[1], (chrg_hi if mems[5].charged > pre_charge_value else chrg_lo), self))

    with self.voiceover(text="""this happens for all of the transistors in a row""") as trk:
      self.play(Transform(spotlight,
                          Exclusion(Rectangle(width=100, height=100), SurroundingRectangle(VGroup(mems[6], mems[7]), buff=0.5), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)))

      for a1, a2 in zip(animate_access(mems[6]), animate_access(mems[7])):
        self.play(*a1, *a2)
      self.play(*set_line(bit_lines[2], (chrg_hi if mems[6].charged > pre_charge_value else chrg_lo), self),
                *set_line(bit_lines[3], (chrg_hi if mems[7].charged > pre_charge_value else chrg_lo), self))

    with self.voiceover(text="""Now, sense amplifiers come into play, their job is to detect the voltage on the bitlines, if it's higher than expected
                        they load the values as 1 and if it's lower they load a 0""") as trk:
      self.play(Transform(spotlight,
                            Exclusion(Rectangle(width=100, height=100), SurroundingRectangle(sa, buff=0.5), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)))
      vals = []
      for i, b in enumerate(bit_lines):
        vals.append(Rectangle(width=2.5, height=2.5, color=GREEN if mems[4+i].charged > pre_charge_value else WHITE, fill_opacity=0.5).next_to(b, DOWN, buff=0))
      self.play(*[Create(x) for x in vals])

    with self.voiceover(text="""The amount of data that was loaded to our sense amplifiers is called a page size""") as trk:
      self.play(FadeOut(spotlight))

    with self.voiceover(text="""Now that the values are loaded into the sense amplifiers we can put the remaining half of the address onto our
                        column decoder""") as trk:
      self.play(address[2].animate.next_to(addres_lines[2], UP),
                address[3].animate.next_to(addres_lines[3], UP))
    with self.voiceover(text="""It decodes the address of the value that we want to access, and passsess it onto the data output line""") as trk:
      self.play(*set_line(decoder_lines[0], 1, self))
      self.play(sa.animate.set_color(vals[0].color))
      self.play(*set_line(data_out, 1 if vals[0].color == GREEN else 0, self))

    with self.voiceover(text="""You might have noticed that it's a lot of work, but it starts getting even worse. 
                        Because after reading the values our memory cells either lost some of their charge or gained an unnecessary charge""") as trk:
      self.play(address[2].animate.next_to(addres_lines[2], UP, buff=1),
                address[3].animate.next_to(addres_lines[3], UP, buff=1))

      self.play(*set_line(decoder_lines[0], 0, self))
      self.play(sa.animate.set_color(WHITE))
      self.play(*set_line(data_out, 0, self))

    anims = []
    for i in range(4):
      anims.extend(set_line(bit_lines[i], 1 if vals[i].color == GREEN else 0, self, backward=True))

    with self.voiceover(text="""So now we have to write the row values from our sense amplifiers back to the memory cells""") as trk:
      self.play(*anims)

      for a1, a2, a3, a4 in zip(mems[4].write(self, 1 if vals[0].color == GREEN else 0), 
                                mems[5].write(self, 1 if vals[1].color == GREEN else 0), 
                                mems[6].write(self, 1 if vals[2].color == GREEN else 0), 
                                mems[7].write(self, 1 if vals[3].color == GREEN else 0)):
        self.play(*a1, *a2, *a3, *a4)

    
    with self.voiceover(text="""Now if we want to access a value that's in a different row than our previous access, we come to a very problematic situation called a row miss
                        """) as trk:
      self.play(address[0].animate.next_to(addres_lines[0], UP, buff=1),
                address[1].animate.next_to(addres_lines[1], UP, buff=1))
      self.play(Transform(address[0], Text("1", font_size=5*24).next_to(addres_lines[0], UP, buff=1)),
                Transform(address[1], Text("0", font_size=5*24).next_to(addres_lines[1], UP, buff=1)))

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

    anims = []
    for i in range(4):
      anims.extend(set_line(bit_lines[i], pre_charge_value, self))
    with self.voiceover(text="""We have to go to the whole process again, prechargint the bitlines, decoding the row address
                        reading into sense amplifiers and decoding the column address.""") as trk:

      self.play(*anims)

      self.play(address[0].animate.next_to(addres_lines[0], UP),
                address[1].animate.next_to(addres_lines[1], UP))
      self.play(*set_line(addres_lines[0], 1, self))
      self.play(*set_line(addres_lines2[0], 1, self))
      self.play(*set_line(word_lines[2], 1, self, backward=True))

      for a1, a2, a3, a4 in zip(animate_access(mems[8]), 
                                animate_access(mems[9]), 
                                animate_access(mems[10]), 
                                animate_access(mems[11])):
        self.play(*a1, *a2, *a3, *a4)

      self.play(*set_line(bit_lines[0], (chrg_hi if mems[8].charged > pre_charge_value else chrg_lo), self),
                *set_line(bit_lines[1], (chrg_hi if mems[9].charged > pre_charge_value else chrg_lo), self),
                *set_line(bit_lines[2], (chrg_hi if mems[10].charged > pre_charge_value else chrg_lo), self),
                *set_line(bit_lines[3], (chrg_hi if mems[11].charged > pre_charge_value else chrg_lo), self))

      vals_t = []
      for i, b in enumerate(bit_lines):
        vals_t.append(Rectangle(width=2.5, height=2.5, color=GREEN if mems[8+i].charged > pre_charge_value else WHITE, fill_opacity=0.5).next_to(b, DOWN, buff=0))
      self.play(*[Transform(v1, v2) for v1, v2 in zip(vals, vals_t)])

      self.play(address[2].animate.next_to(addres_lines[2], UP),
                address[3].animate.next_to(addres_lines[3], UP))
      self.play(*set_line(decoder_lines[0], 1, self))
      self.play(sa.animate.set_color(vals[0].color))
      self.play(*set_line(data_out, 1 if vals[0].color == GREEN else 0, self))

    with self.voiceover(text="""Lucily for us, we can also have a row hit situation where we try to access a value in a raw that is already
                        loaded into memory. We can now skip all of the steps that were required when we needed to change the row stored in the 
                        sense amplifiers, and just decode the column address and pass it to our data output line.
                        And the memory kinda does that for you, during the time of writing the rows back to memory, it just outputs a few next values from the row.
                        This is called a DRAM burst and the amout of bytes that get returned from the burst is a burst length or a burst size""") as trk:
      for i, v in enumerate(["01", "10", "11"]):
        self.play(address[2].animate.next_to(addres_lines[2], UP, buff=1),
                  address[3].animate.next_to(addres_lines[3], UP, buff=1))
        self.play(*set_line(addres_lines[2], 0, self), *set_line(addres_lines[3], 0, self))
        self.play(*set_line(addres_lines2[2], 0, self), *set_line(addres_lines2[3], 0, self))

        self.play(*set_line(decoder_lines[i], 0, self))
        self.play(sa.animate.set_color(WHITE))
        self.play(*set_line(data_out, 0, self))

        self.play(Transform(address[2], Text(v[0], font_size=5*24).next_to(addres_lines[2], UP, buff=1)),
                  Transform(address[3], Text(v[1], font_size=5*24).next_to(addres_lines[3], UP, buff=1)))
        self.play(address[2].animate.next_to(addres_lines[2], UP),
                  address[3].animate.next_to(addres_lines[3], UP))
        self.play(*set_line(addres_lines[2], int(v[0]), self), *set_line(addres_lines[3], int(v[1]), self))
        self.play(*set_line(addres_lines2[2], int(v[0]), self), *set_line(addres_lines2[3], int(v[1]), self))
        self.play(*set_line(decoder_lines[i+1], 1, self))
        self.play(sa.animate.set_color(vals[i+1].color))
        self.play(*set_line(data_out, 1 if vals[i+1].color == GREEN else 0, self))


    self.play(*[FadeOut(x) for x in self.mobjects])
    self.play(Restore(self.camera.frame))

    bytes = [Square(side_length=0.2, color=BLUE, fill_color=BLUE, fill_opacity=0.5, stroke_width=1) for _ in range(32)]
    VGroup(*bytes).arrange(RIGHT, buff=0.05).move_to(self.camera.frame)
    brace = Brace(VGroup(*bytes), direction=UP)
    bytes_t = Text("32 Bytes", font_size=32).next_to(brace, UP)
    segments = [Rectangle(height=0.3, width=1, color=BLUE, fill_color=BLUE, fill_opacity=0.5, stroke_width=2) for _ in range(8)]
    VGroup(*segments).arrange(RIGHT, buff=0.05).move_to(self.camera.frame)
    brace_transform = Brace(segments[0], direction=UP)
    
    with self.voiceover(text="""Because of that, memory is accessed in segments, where<bookmark mark='1'/> one segment consists of 32 Bytes""") as trk:
      self.play(*[Create(x) for x in bytes], Create(brace), Write(bytes_t))
      self.wait_until_bookmark("1")
      self.play(Transform(VGroup(*bytes), segments[0], replace_mobject_with_target_in_scene=True),
                Transform(brace, brace_transform),
                Transform(bytes_t, Text("32B Segment", font_size=32).next_to(brace_transform, UP)))
      self.play(*[Create(x) for x in segments[1:]])

    brace64 = Brace(VGroup(*segments[:2]), direction=UP)
    brace128 = Brace(VGroup(*segments[:4]), direction=UP)

    with self.voiceover(text="""When we make a memory access, a whole warp generates a memory transaction that can be <bookmark mark='1'/>either a 32B transaction,
                        <bookmark mark='2'/> a 64 byte transaction <bookmark mark='3'/>or a 128 byte transaction""") as trk:
      self.wait_until_bookmark("1")
      self.play(*[s.animate.set_color(GREEN) for s in segments[:1]],
                Transform(bytes_t, Text("32B Transaction", font_size=32).next_to(brace, UP)))
      self.wait_until_bookmark("2")
      self.play(*[s.animate.set_color(GREEN) for s in segments[:2]],
                Transform(brace, brace64),
                Transform(bytes_t, Text("64B Transaction", font_size=32).next_to(brace64, UP)))
      self.wait_until_bookmark("3")
      self.play(*[s.animate.set_color(GREEN) for s in segments[:4]],
                Transform(brace, brace128),
                Transform(bytes_t, Text("128B Transaction", font_size=32).next_to(brace128, UP)))

    with self.voiceover(text="""The transaction size depends on the alignment of our data, the type of memory we are accessing,
                        and the architecture we are using, for example requests to the L1 use 128B segments, while requests from L1 to L2
                        are made in 32B segments""") as trk:
      self.play(*[s.animate.set_color(BLUE) for s in segments[:4]],
                FadeOut(brace), FadeOut(bytes_t))

    
    brace = Brace(segments[0], direction=UP).scale(0.5).next_to(VGroup(*segments[:2]), UP, buff=0.1)
    transaction_t = Text("2B Transaction", font_size=24).next_to(brace, UP, buff=0.1)
    with self.voiceover(text="""Also our accesses need to be aligned to those segments, for example - <bookmark mark='1'/>if we want to access even just 2 bytes that are in 2
                        different segments<bookmark mark='2'/> we still need to load both of those segments into memory""") as trk:
      self.wait_until_bookmark("1")
      self.play(Create(brace))
      self.play(Write(transaction_t))
      self.wait_until_bookmark("2")
      self.play(*[s.animate.set_color(GREEN) for s in segments[:2]])

    self.play(*[FadeOut(x) for x in self.mobjects])
    Rectangle.set_default()
    Line.set_default()

    offset_cp = """__global__ void copy(int n , float* in, float* out, int offset)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i + offset] = in[i + offset];
  }
}"""
    code_obj = Code(code=offset_cp, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1)
    with self.voiceover(text="""This brings us to the crux of the problem, what does that mean to us as programmers. To anwser this
                        we can examine a simple offset copy kernel""") as trk:
      self.play(Create(code_obj))
    
    VGroup(*segments).next_to(code_obj, 2*DOWN)
    brace = Brace(VGroup(*segments[:4]), direction=DOWN)
    text = Text("Offset = 0", font_size=32).next_to(brace, DOWN)
    for s in segments:
      s.set_color(BLUE)
    with self.voiceover(text="""In this kernel, if we look at what the whole warp is doing, it's accessing 128 bytes of memory with an offset""") as trk:
      self.play(*[Create(s) for s in segments], Create(brace), Write(text))
      self.wait(1)
      self.play(brace.animate.shift(0.25*RIGHT))
      self.play(Transform(text, Text("Offset = 2", font_size=32).next_to(brace, DOWN)))
      self.play(brace.animate.shift(0.25*RIGHT))
      self.play(Transform(text, Text("Offset = 4", font_size=32).next_to(brace, DOWN)))

    self.play(*[FadeOut(x) for x in self.mobjects])

    timings = [0.019367, 0.019925, 0.019936, 0.019934, 0.020016, 0.019917, 0.019904, 0.019895, 0.019908, 0.019940, 0.019978, 0.020023, 0.019927, 0.019989, 0.019981, 0.020036, 0.019957, 0.019928, 0.019966, 0.019954, 0.019941, 0.019984, 0.019906, 0.019981, 0.019988, 0.019848, 0.019775, 0.019766, 0.019844, 0.019798, 0.019781, 0.019863, 0.018953, 0.019896, 0.019926, 0.019977, 0.019960, 0.019939, 0.019905, 0.019913, 0.019926, 0.019980, 0.019938, 0.019950, 0.020023, 0.019957, 0.020110, 0.020120, 0.019976, 0.019959, 0.019931, 0.019998, 0.019952, 0.019945, 0.020011, 0.019952, 0.020014, 0.019774, 0.019777, 0.019763, 0.019877, 0.019785, 0.019781, 0.019846, 0.018926, 0.019941, 0.019955, 0.019994, 0.019931, 0.019984, 0.019984, 0.019937, 0.019971, 0.019923, 0.019978, 0.019955, 0.020009, 0.019993, 0.019995, 0.019985, 0.019997, 0.019918, 0.019966, 0.019996, 0.020016, 0.019929, 0.020023, 0.019961, 0.020032, 0.019795, 0.019772, 0.019782, 0.019834, 0.019786, 0.019889, 0.019814, 0.018984, 0.019938, 0.019944, 0.019935, 0.020103, 0.019914, 0.019990, 0.019909, 0.020012, 0.020028, 0.020015, 0.020021, 0.019926, 0.020081, 0.019974, 0.020045, 0.020032, 0.020067, 0.019973, 0.019943, 0.019926, 0.020061, 0.019954, 0.019965, 0.020057, 0.019855, 0.019780, 0.019792, 0.019804, 0.019813, 0.019769, 0.019795, 0.018979, ]
    stride = list(range(130))
    timings = [t*1e3 for t in timings]
    stride = [s*4 for s in stride]

    ax = Axes(x_range=[0, stride[-1]+1, 32],
              y_range=[18, 22, 1],
              x_length=16,
              y_length=8,
              axis_config={"include_numbers": True}).scale(0.8)

    graph = ax.plot_line_graph(x_values=stride, y_values=timings, line_color=GREEN, add_vertex_dots=False) 

    x_label = ax.get_x_axis_label("Offset[bytes]")
    y_label = ax.get_y_axis_label("Time [ms]")

    with self.voiceover(text="""We run this kernel and measure the timing for different values of the offset variable""") as trk:
      self.play(Create(ax))
      self.play(Write(x_label), Write(y_label))

    with self.voiceover(text="""Note that on the graph, the offset is given in bytes""") as trk:
      self.play(Indicate(x_label))

    with self.voiceover(text="""And if we examine the results, we can see that the kernel performs the best when the offset is a multiple of 128 bytes,
                        and this is the memory transaction size used when accessing data through the L1 cache on my architecture""") as trk:
      self.play(Create(graph))

    self.play(*[FadeOut(x) for x in self.mobjects])

    strided_cp = """__global__ void copy(int n , float* in, float* out, int stride)
{
  unsigned long i = (blockIdx.x*blockDim.x + threadIdx.x)*stride;
  if (i < n)
  {
    out[i] = in[i];
  }
}"""
    code_obj = Code(code=strided_cp, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1)
    with self.voiceover(text="""Another kernel that we can examine is the strided copy kernal""") as trk:
      self.play(Create(code_obj))

    with self.voiceover(text="""All it does is just copying from one buffer to another, but we can specify the stride between successive
                        memory accesses""") as trk:
      pass
    buffers = [Rectangle(width=0.25, height=0.25, color=WHITE, fill_color=WHITE, fill_opacity=0.25, stroke_width=2) for x in range(16)]
    VGroup(*buffers).arrange(RIGHT, buff=0.05).next_to(code_obj, DOWN)
    def strided(rng, stride):
      accessed = [stride * i for i in range(rng)]
      buffers = [Rectangle(width=0.25, height=0.25, color=GREEN if x in accessed else WHITE, fill_color=GREEN if x in accessed else WHITE, fill_opacity=0.25, stroke_width=2) for x in range(16)]
      VGroup(*buffers).arrange(RIGHT, buff=0.05).next_to(code_obj, DOWN)
      return buffers
    with self.voiceover(text="""So when we are using a stride of 1<bookmark mark='1'/> we access memory regions that are next to each other,
                        with a stride of 2 <bookmark mark='2'/> they are separated by one value, and with a stride of 4<bookmark mark='3'/>
                        each access is 4 values away of the previous one """) as trk:
      self.play(*[Create(x) for x in buffers])
      self.wait_until_bookmark("1")
      self.play(*[Transform(b1, b2) for b1, b2 in zip(buffers, strided(4, 1))])
      self.wait_until_bookmark("2")
      self.play(*[Transform(b1, b2) for b1, b2 in zip(buffers, strided(4, 2))])
      self.wait_until_bookmark("3")
      self.play(*[Transform(b1, b2) for b1, b2 in zip(buffers, strided(4, 4))])

    self.play(Uncreate(code_obj))
    self.play(*[Uncreate(x) for x in buffers])

    timings = [0.007830, 0.008743, 0.009828, 0.011690, 0.014539, 0.018326, 0.018364, 0.018616, 0.021799, 0.022405, 0.022495, 0.022226, 0.023013, 0.022790, 0.023171, ]

    timings = [t*1e3 for t in timings]
    stride = list(range(15))
    stride = [s+2 for s in stride]

    ax = Axes(x_range=[2, stride[-1]+1, 1],
              y_range=[8, 24, 2],
              x_length=12,
              x_axis_config={"scaling": LogBase(2, custom_labels=True)},
              axis_config={"include_numbers": True}).scale(0.8)

    graph = ax.plot_line_graph(x_values=[2**x for x in stride], y_values=timings, line_color=BLUE, add_vertex_dots=False) 
    segment_size = ax.plot_line_graph(x_values=[128, 128], y_values=[8, 24], line_color=ORANGE, add_vertex_dots=False)
    segment_t = Text("       128B\nSegment Size", font_size=24, color=ORANGE).next_to(segment_size, LEFT).shift(UP)
    page_size = ax.plot_line_graph(x_values=[1024, 1024], y_values=[8, 24], line_color=RED, add_vertex_dots=False)
    page_t = Text("1KB Page Size", font_size=24, color=RED).next_to(page_size, RIGHT)

    x_label = ax.get_x_axis_label("Stride[Bytes]")
    y_label = ax.get_y_axis_label("Time [ms]")

    with self.voiceover(text="""We can run this plot for diffrent stride values too see how this affects the speed of our algorithm""") as trk:
      self.play(Create(ax))
      self.play(Write(x_label), Write(y_label))

    with self.voiceover(text="""As expected, the bigger our stride the longer it takes to access our memory""") as trk:
      self.play(Create(graph))
    with self.voiceover(text="""But there are two points that are standing out, where we have big increases followed by plateus""") as trk:
      self.play(Create(segment_size), Create(page_size))

    with self.voiceover(text="""One is 128 bytes, which is the segment size of one memory request going through L1 cache""") as trk:
      self.play(Write(segment_t))

    with self.voiceover(text="""And the next one is at 1KB, and this is the page size, so the amount of values that get loaded 
                        into sense amplifiers when we select a row""") as trk:
      self.play(Write(page_t))

    with self.voiceover(text="""We've come to the end of another episode, related to memory - by now you should see how imortant
                        efficient memory managment is to performance of our kernels""") as trk:
      pass

    self.play(*[FadeOut(x) for x in self.mobjects])
    
    bmac = Text("https://buymeacoffee.com/simonoz", font_size=48, color=YELLOW)
    alex = Text("Alex", font_size=60).next_to(bmac, DOWN)
    udit = Text("Udit Ransaria", font_size=60).next_to(alex, DOWN)
    unknown = Text("Anonymous x3", font_size=60).next_to(udit, DOWN)
    with self.voiceover(text="""I'm hosting a buy me a coffe for those that want to support this channel. A shoutout to Alex, Udit Ransaria and three anonymous donors that supported so far""") as trk:
      self.camera.auto_zoom(VGroup(bmac, alex, unknown), margin=4, animate=False)
      self.play(Write(bmac))
      self.play(Write(alex))
      self.play(Write(udit))
      self.play(Write(unknown))

    subscribe = SVGMobject("icons/subscribe.svg")
    like = SVGMobject("icons/like.svg")
    share = SVGMobject("icons/share.svg")
    VGroup(subscribe, like, share).arrange(RIGHT).next_to(unknown, DOWN).scale(0.7)

    with self.voiceover(text="""But you can always support me for fre by <bookmark mark='1'/>subscribing, <bookmark mark='2'/>leaving a like, <bookmark mark='3'/>commenting and sharing this video with your friends""") as trk:
      self.play(Create(like), Create(subscribe), Create(share))
      self.wait_until_bookmark("1")
      self.play(subscribe.animate.set_color(RED))
      self.wait_until_bookmark("2")
      self.play(like.animate.set_color(RED))
      self.wait_until_bookmark("3")
      self.play(share.animate.set_color(RED))


    return
    with self.voiceover(text="""I'll see you in the next episode, bye""") as trk:
      pass

    self.play(*[FadeOut(x) for x in self.mobjects])
    self.wait(2)
