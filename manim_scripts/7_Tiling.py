from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import math


class Tiling(VoiceoverScene, MovingCameraScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base") # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU Programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 7 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Tiling With Shared Memory", font_size=48).next_to(title, DOWN)
    with self.voiceover(text="""In this episode, we are going to go over how we can use shared memory to implement
                        a tilied matrix multiplication algorithm to improve it's performance""") as trk:
      self.play(Write(subtitle))

    self.play(Unwrite(subtitle), Unwrite(title))

    N = 4
    tile_size=2
    m3 = Matrix([[f"c_{{{j},{i}}}" for i in range(N)] for j in range(N)]).shift(2.8*RIGHT + 1.6*DOWN)
    m1 = Matrix([[f"a_{{{j},{i}}}" for i in range(N)] for j in range(N)]).next_to(m3, LEFT)
    m2 = Matrix([[f"b_{{{j},{i}}}" for i in range(N)] for j in range(N)]).next_to(m3, UP)

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

    code_obj = Code(code=matmul, tab_width=2, language="c", font_size=18, background="rectangle", line_no_buff=0.1, corner_radius=0.1)
    with self.voiceover(text="""As a reminder, this is what our matrix multiplication code from episode 2 looked like""") as trk:
      self.play(Create(code_obj))
    self.wait(1)
    self.play(Uncreate(code_obj))
    
    with self.voiceover(text="""Let's look at how memory is accessed in our kernel""") as trk:
      self.play(create_matrix(m3), create_matrix(m1), create_matrix(m2))

    i1 = SurroundingRectangle(m1.get_entries()[:N], color=BLUE)
    i2 = SurroundingRectangle(VGroup(*[m2.get_entries()[i*N] for i in range(N)]), color=BLUE)
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
      registers.add(registers_text)
      self.play(LaggedStart(*[FadeOut(m1.get_entries()[i].copy(), target_position=registers) for i in range(N)],
                            *[FadeOut(m2.get_entries()[i*N].copy(), target_position=registers) for i in range(N)]))

    fs = 36
    c1 = [1, 0, 0, 0]
    c2 = [1, 0, 0, 0]
    count1 = [Tex(str(c1[i]), font_size=fs, color=BLUE).next_to(m1.get_entries()[i*N], LEFT, buff=1) for i in range(N)]
    count2 = [Tex(str(c2[i]), font_size=fs, color=BLUE).next_to(m2.get_entries()[i], UP) for i in range(N)]


    def do_calc(row, col, run_time=1):
      nonlocal i1, i2, i3, c1, c2, count1, count2, fs, registers

      self.play(Transform(i1, SurroundingRectangle(m1.get_entries()[row*N:row*N + N], color=BLUE), run_time=run_time/3),
                Transform(i2, SurroundingRectangle(VGroup(*[m2.get_entries()[i*N + col] for i in range(N)]), color=BLUE), run_time=run_time/3),
                Transform(i3, SurroundingRectangle(m3.get_entries()[row*N + col]), run_time=run_time/3))

      self.play(LaggedStart(*[FadeOut(m1.get_entries()[row*N + i].copy(), target_position=registers, run_time=run_time/3) for i in range(N)],
                            *[FadeOut(m2.get_entries()[i*N + col].copy(), target_position=registers, run_time=run_time/3) for i in range(N)]))
      c1[row]+=1
      c2[col]+=1
      self.play(Transform(count1[row], Tex(str(c1[row]), font_size=fs, color=BLUE).next_to(m1.get_entries()[row*N], LEFT, buff=1), run_time=run_time/3),
                Transform(count2[col], Tex(str(c2[col]), font_size=fs, color=BLUE).next_to(m2.get_entries()[col], UP)), run_time=run_time/3)


    self.wait(1)
    with self.voiceover(text="""If we were to start keeping track of how many times was each row and column accessed
                        we can see that we are actually reading them multiple times, and more precisely the number of accesses
                        is equal to the length of the side of our matrix""") as trk:
      self.play(LaggedStart(*[Write(c) for c in count1+count2]))
      for row in range(N):
        for col in range(N):
          if row == 0 and col == 0: continue
          do_calc(row, col, run_time=math.log(trk.duration)/(row*N+col))
    self.wait(1)


    shared = Rectangle(height=1, width=N, color=YELLOW, fill_color=YELLOW, fill_opacity=0.5)
    shared_text = Text("Shared memory", font_size=32, color=YELLOW)
    fs = 36
    c1 = [0, 0, 0, 0]
    c2 = [0, 0, 0, 0]
    count_s1 = [Tex(str(c1[i]), font_size=fs, color=BLUE).next_to(m1.get_entries()[i*N], LEFT, buff=1) for i in range(N)]
    count_s2 = [Tex(str(c2[i]), font_size=fs, color=BLUE).next_to(m2.get_entries()[i], UP) for i in range(N)]
    with self.voiceover(text="""And to reduce our global memory workload we can utilize shared memory""") as trk:
      self.play(Uncreate(i1), Uncreate(i2), Uncreate(i3))
      self.play(registers.animate.shift(0.6*UP))
      shared.next_to(registers, DOWN)
      shared_text.move_to(shared)
      self.play(LaggedStart(*[Transform(c, cs, replace_mobject_with_target_in_scene=True) for (c,cs) in zip(count1+count2, count_s1+count_s2)]))
      self.play(Create(shared), Write(shared_text))


    shared_args = {"buff": MED_SMALL_BUFF, "color": PURPLE}
    i1s = SurroundingRectangle(VGroup(*[m1.get_entries()[i*N + j] for i in range(tile_size) for j in range(tile_size)]), **shared_args)
    i2s = SurroundingRectangle(VGroup(*[m2.get_entries()[i*N + j] for i in range(tile_size) for j in range(tile_size)]), **shared_args)
    i3s = SurroundingRectangle(VGroup(*[m3.get_entries()[i*N + j] for i in range(tile_size) for j in range(tile_size)]), **shared_args)

    with self.voiceover(text="""This is a next big step in our cuda journey as this time, we have to start thinking about what our whole
                        blocks(indicated with a purple color here) will be doing instead of just individual threads""") as trk:
      self.play( Create(i3s))


    with self.voiceover(text="""We start by splitting our input matrices into tiles that are the same shape as our blocks""") as trk:
      self.play(Create(i1s), Create(i2s))

    rt = 0.5 
    anims = []
    row=0
    col=0
    with self.voiceover(text="""Each thread in the block then loads the corresponding value in each input matrix into shared memory""") as trk:
      step=0
      e1 = [m1.get_entries()[(i + row*tile_size)*N + j + step*tile_size].copy() for i in range(tile_size) for j in range(tile_size)]
      e2 = [m2.get_entries()[(i + step*tile_size)*N + j + col*tile_size].copy() for i in range(tile_size) for j in range(tile_size)]
      e3 = [m3.get_entries()[(i + row*tile_size)*N + j + col*tile_size].copy()for i in range(tile_size) for j in range(tile_size)]
      self.play(Transform(i1s, SurroundingRectangle(VGroup(*e1), **shared_args), run_time=rt),
                Transform(i2s, SurroundingRectangle(VGroup(*e2), **shared_args), run_time=rt),
                Transform(i3s, SurroundingRectangle(VGroup(*e3), **shared_args), run_time=rt), 
                *anims)
      self.play(LaggedStart(*[FadeOut(e.copy(), target_position=shared, run_time=rt) for e in e1],
                            *[FadeOut(e.copy(), target_position=shared, run_time=rt) for e in e2]))

    with self.voiceover(text="""And proceeds with calculating a partial dot product, but instead of reading the values
                        from slow global memory, we read from fast shared memory instead""") as trk:
      for x in range(tile_size):
        for y in range(tile_size):
          if x == 0 and y == 0:
            self.play(Create(i1:=SurroundingRectangle(VGroup(*e1[y*tile_size:y*tile_size + tile_size]), color=BLUE), run_time=rt),
                      Create(i2:=SurroundingRectangle(VGroup(*[e2[i*tile_size + x] for i in range(tile_size)]), color=BLUE), run_time=rt),
                      Create(i3:=SurroundingRectangle(e3[y*tile_size + x]), run_time=rt))
          else:
            self.play(Transform(i1, SurroundingRectangle(VGroup(*e1[y*tile_size:y*tile_size + tile_size]), color=BLUE), run_time=rt),
                      Transform(i2, SurroundingRectangle(VGroup(*[e2[i*tile_size + x] for i in range(tile_size)]), color=BLUE), run_time=rt),
                      Transform(i3, SurroundingRectangle(e3[y*tile_size + x]), run_time=rt))
          self.play(*[FadeIn(e.set_color(YELLOW), run_time=rt, target_position=shared) for e in e1], *[FadeIn(e.set_color(YELLOW), run_time=rt, target_position=shared) for e in e2])
          self.wait(rt)
          self.play(*[FadeOut(e, run_time=rt, target_position=registers) for e in e1], *[FadeOut(e, run_time=rt, target_position=registers) for e in e2])
    anims = [Uncreate(i1, run_time=rt), Uncreate(i2, run_time=rt), Uncreate(i3, run_time=rt)]

    with self.voiceover(text="""When we are done, we move our tiles and do the same thing to finalize calculating the dot product corresponding
                        to each entry in the output matrix""") as trk:
      step=1
      e1 = [m1.get_entries()[(i + row*tile_size)*N + j + step*tile_size].copy() for i in range(tile_size) for j in range(tile_size)]
      e2 = [m2.get_entries()[(i + step*tile_size)*N + j + col*tile_size].copy() for i in range(tile_size) for j in range(tile_size)]
      e3 = [m3.get_entries()[(i + row*tile_size)*N + j + col*tile_size].copy()for i in range(tile_size) for j in range(tile_size)]
      self.play(Transform(i1s, SurroundingRectangle(VGroup(*e1), **shared_args), run_time=rt),
                Transform(i2s, SurroundingRectangle(VGroup(*e2), **shared_args), run_time=rt),
                Transform(i3s, SurroundingRectangle(VGroup(*e3), **shared_args), run_time=rt), 
                *anims)
      self.play(LaggedStart(*[FadeOut(e.copy(), target_position=shared, run_time=rt) for e in e1],
                            *[FadeOut(e.copy(), target_position=shared, run_time=rt) for e in e2]))
      for x in range(tile_size):
        for y in range(tile_size):
          if x == 0 and y == 0:
            self.play(Create(i1:=SurroundingRectangle(VGroup(*e1[y*tile_size:y*tile_size + tile_size]), color=BLUE), run_time=rt),
                      Create(i2:=SurroundingRectangle(VGroup(*[e2[i*tile_size + x] for i in range(tile_size)]), color=BLUE), run_time=rt),
                      Create(i3:=SurroundingRectangle(e3[y*tile_size + x]), run_time=rt))
          else:
            self.play(Transform(i1, SurroundingRectangle(VGroup(*e1[y*tile_size:y*tile_size + tile_size]), color=BLUE), run_time=rt),
                      Transform(i2, SurroundingRectangle(VGroup(*[e2[i*tile_size + x] for i in range(tile_size)]), color=BLUE), run_time=rt),
                      Transform(i3, SurroundingRectangle(e3[y*tile_size + x]), run_time=rt))
          self.play(*[FadeIn(e.set_color(YELLOW), run_time=rt, target_position=shared) for e in e1], *[FadeIn(e.set_color(YELLOW), run_time=rt, target_position=shared) for e in e2])
          self.wait(rt)
          self.play(*[FadeOut(e, run_time=rt, target_position=registers) for e in e1], *[FadeOut(e, run_time=rt, target_position=registers) for e in e2])
      anims = [Uncreate(i1, run_time=rt), Uncreate(i2, run_time=rt), Uncreate(i3, run_time=rt)]
      for i in range(tile_size):
        c1[row*tile_size+i]+=1
        c2[col*tile_size+i]+=1
        anims.append(Transform(count_s1[row*tile_size+i], Tex(str(c1[row*tile_size+i]), font_size=fs, color=BLUE).next_to(m1.get_entries()[(row*tile_size+i)*N], LEFT, buff=1), run_time=rt))
        anims.append(Transform(count_s2[col*tile_size+i], Tex(str(c2[col*tile_size+i]), font_size=fs, color=BLUE).next_to(m2.get_entries()[col*tile_size+i], UP), run_time=rt))
      self.play(*anims)

    with self.voiceover(text="""And this runs for each part of the output matrix""") as trk:
      for row in range(N//tile_size):
        for col in range(N//tile_size):
          if row == 0 and col == 0: continue
          rt = 0.5/(1 + row*N//tile_size + col)
          anims = []
          for step in range(N//tile_size):
            e1 = [m1.get_entries()[(i + row*tile_size)*N + j + step*tile_size].copy() for i in range(tile_size) for j in range(tile_size)]
            e2 = [m2.get_entries()[(i + step*tile_size)*N + j + col*tile_size].copy() for i in range(tile_size) for j in range(tile_size)]
            e3 = [m3.get_entries()[(i + row*tile_size)*N + j + col*tile_size].copy()for i in range(tile_size) for j in range(tile_size)]
            self.play(Transform(i1s, SurroundingRectangle(VGroup(*e1), **shared_args), run_time=rt),
                      Transform(i2s, SurroundingRectangle(VGroup(*e2), **shared_args), run_time=rt),
                      Transform(i3s, SurroundingRectangle(VGroup(*e3), **shared_args), run_time=rt), 
                      *anims)
            self.play(LaggedStart(*[FadeOut(e.copy(), target_position=shared, run_time=rt) for e in e1],
                                  *[FadeOut(e.copy(), target_position=shared, run_time=rt) for e in e2]))
            for x in range(tile_size):
              for y in range(tile_size):
                if x == 0 and y == 0:
                  self.play(Create(i1:=SurroundingRectangle(VGroup(*e1[y*tile_size:y*tile_size + tile_size]), color=BLUE), run_time=rt),
                            Create(i2:=SurroundingRectangle(VGroup(*[e2[i*tile_size + x] for i in range(tile_size)]), color=BLUE), run_time=rt),
                            Create(i3:=SurroundingRectangle(e3[y*tile_size + x]), run_time=rt))
                else:
                  self.play(Transform(i1, SurroundingRectangle(VGroup(*e1[y*tile_size:y*tile_size + tile_size]), color=BLUE), run_time=rt),
                            Transform(i2, SurroundingRectangle(VGroup(*[e2[i*tile_size + x] for i in range(tile_size)]), color=BLUE), run_time=rt),
                            Transform(i3, SurroundingRectangle(e3[y*tile_size + x]), run_time=rt))
                self.play(*[FadeIn(e.set_color(YELLOW), run_time=rt, target_position=shared) for e in e1], *[FadeIn(e.set_color(YELLOW), run_time=rt, target_position=shared) for e in e2])
                self.play(*[FadeOut(e, run_time=rt, target_position=registers) for e in e1], *[FadeOut(e, run_time=rt, target_position=registers) for e in e2])
            anims = [Uncreate(i1, run_time=rt), Uncreate(i2, run_time=rt), Uncreate(i3, run_time=rt)]

          for i in range(tile_size):
            c1[row*tile_size+i]+=1
            c2[col*tile_size+i]+=1
            anims.append(Transform(count_s1[row*tile_size+i], Tex(str(c1[row*tile_size+i]), font_size=fs, color=BLUE).next_to(m1.get_entries()[(row*tile_size+i)*N], LEFT, buff=1), run_time=rt))
            anims.append(Transform(count_s2[col*tile_size+i], Tex(str(c2[col*tile_size+i]), font_size=fs, color=BLUE).next_to(m2.get_entries()[col*tile_size+i], UP), run_time=rt))
          self.play(*anims)

    tiled_mm = """__shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
__shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

int column = blockIdx.x*TILE_WIDTH + threadIdx.x;
int row = blockIdx.y*TILE_WIDTH + threadIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;
float dot_prod = 0.f;

for (int tile_offset = 0; tile_offset<n; tile_offset+=TILE_WIDTH)
{
  int a_chk = tile_offset+tx < n && row < n;
  a_tile[ty][tx] = a_chk ? a[row*n + tile_offset+tx] : 0.f;

  int b_chk = (tile_offset+ty) < n && column < n;
  b_tile[ty][tx] = b_chk ? b[(tile_offset+ty)*n + column] : 0.f;

  __syncthreads();
  for(int i = 0; i < TILE_WIDTH; i++)
  {
    dot_prod += a_tile[ty][i] * b_tile[i][tx];
  }
  __syncthreads();
}

if (row < n && column < n)
{
  c[row*n+column] = dot_prod;
}"""
    code_obj = Code(code=tiled_mm, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1).scale(0.8).next_to(m1, UP, aligned_edge=RIGHT)
    code_obj.code = remove_invisible_chars(code_obj.code)
    
    with self.voiceover(text="""The code that we will use for achieving this introduces some new concepts""") as trk:
      self.play(self.camera.frame.animate.scale(1.3).shift(UP+LEFT))
      self.play(Transform(VGroup(*count_s1, *count_s2, i1s, i2s, i3s, shared, shared_text, registers),
                          code_obj, replace_mobject_with_target_in_scene=True))
