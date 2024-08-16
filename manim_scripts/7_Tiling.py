from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import math


class Tiling(VoiceoverScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU Programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 7 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Tiling With Shared Memory", font_size=48).next_to(title, DOWN)
    with self.voiceover(text="""In this episode, we are going to go over how we can use shared memory to implement
                        a tilied matrix multiplication algorithm to improve it's performance""") as trk:
      self.play(Write(subtitle))

    self.play(Unwrite(subtitle), Unwrite(title))

    m3 = Matrix([[f"c_{{{j},{i}}}" for i in range(4)] for j in range(4)]).shift(2.8*RIGHT + 1.6*DOWN)
    m1 = Matrix([[f"a_{{{j},{i}}}" for i in range(4)] for j in range(4)]).next_to(m3, LEFT)
    m2 = Matrix([[f"b_{{{j},{i}}}" for i in range(4)] for j in range(4)]).next_to(m3, UP)

    def create_matrix(m):
      return LaggedStart(Create(m.get_brackets()[0]), Create(m.get_brackets()[1]), *[Write(e) for e in m.get_entries()])

    matmul = """__global__ void matmul_elem
    (int n, float* a, float* b, float* c)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < n && column < n)
  {
    float dot_prod = 0.f;
    for(int i = 0; i < n; i++)
    {
      dot_prod += a[row*n + i] * b[i*n + column];
    }
    c[row*n+column] = dot_prod;
  }
}"""

    matmul_obj = Code(code=matmul, tab_width=2, language="c", font_size=18, background="rectangle", line_no_buff=0.1, corner_radius=0.1)
    with self.voiceover(text="""As a reminder, this is what our matrix multiplication code from episode 2 looked like""") as trk:
      self.play(Create(matmul_obj))
    self.wait(1)
    self.play(Uncreate(matmul_obj))
    
    with self.voiceover(text="""Let's look at how memory is accessed in our kernel""") as trk:
      self.play(create_matrix(m3), create_matrix(m1), create_matrix(m2))

    i1 = SurroundingRectangle(m1.get_entries()[:4], color=BLUE)
    i2 = SurroundingRectangle(VGroup(*[m2.get_entries()[i*4] for i in range(4)]), color=BLUE)
    i3 = SurroundingRectangle(m3.get_entries()[0])
    pos = m2.get_center().copy()
    pos[0] = m1.get_center()[0]
    registers = Rectangle(height=1, width=3, color=GREEN, fill_color=GREEN, fill_opacity=0.5).move_to(pos)
    registers_text = Text("Registers", font_size=32, color=GREEN).move_to(registers)

    
    with self.voiceover(text="""The first thread takes the first row of matrix A, an the first column of matrix B <bookmark mark='1'/>
                        and loads them into the registers""") as trk:
      self.play(Create(i1), Create(i2), Create(i3))
      self.wait_until_bookmark("1")
      self.play(Create(registers), Write(registers_text))
      self.play(LaggedStart(*[FadeOut(m1.get_entries()[i].copy(), target_position=registers) for i in range(4)],
                            *[FadeOut(m2.get_entries()[i*4].copy(), target_position=registers) for i in range(4)]))

    fs = 36
    c1 = [1, 0, 0, 0]
    c2 = [1, 0, 0, 0]
    count1 = [Tex(str(c1[i]), font_size=fs, color=BLUE).next_to(m1.get_entries()[i*4], LEFT, buff=1) for i in range(4)]
    count2 = [Tex(str(c2[i]), font_size=fs, color=BLUE).next_to(m2.get_entries()[i], UP) for i in range(4)]


    def do_calc(row, col, run_time=1):
      nonlocal i1, i2, i3, c1, c2, count1, count2, fs, registers

      self.play(Transform(i1, SurroundingRectangle(m1.get_entries()[row*4:row*4 + 4], color=BLUE), run_time=run_time/3),
                Transform(i2, SurroundingRectangle(VGroup(*[m2.get_entries()[i*4 + col] for i in range(4)]), color=BLUE), run_time=run_time/3),
                Transform(i3, SurroundingRectangle(m3.get_entries()[row*4 + col]), run_time=run_time/3))

      self.play(LaggedStart(*[FadeOut(m1.get_entries()[row*4 + i].copy(), target_position=registers, run_time=run_time/3) for i in range(4)],
                            *[FadeOut(m2.get_entries()[i*4 + col].copy(), target_position=registers, run_time=run_time/3) for i in range(4)]))
      c1[row]+=1
      c2[col]+=1
      self.play(Transform(count1[row], Tex(str(c1[row]), font_size=fs, color=BLUE).next_to(m1.get_entries()[row*4], LEFT, buff=1), run_time=run_time/3),
                Transform(count2[col], Tex(str(c2[col]), font_size=fs, color=BLUE).next_to(m2.get_entries()[col], UP)), run_time=run_time/3)


    self.wait(1)
    with self.voiceover(text="""If we were to start keeping track of how many times was each row and column accessed
                        we can see that we are actually reading them multiple times, and more precisely the number of accesses
                        is equal to the length of the side of our matrix""") as trk:
      self.play(LaggedStart(*[Write(c) for c in count1+count2]))
      for row in range(4):
        for col in range(4):
          if row == 0 and col == 0: continue
          do_calc(row, col, run_time=math.log(trk.duration)/(row*4+col))
    self.wait(1)
