from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
from math import radians, degrees
import random

class Thread(VGroup):
    def __init__(
        self,
        side_length: float = 2,
        fill_opacity: float = 0.75,
        fill_color: ParsableManimColor = BLUE,
        stroke_width: float = 0,
        thread_idx: tuple[int, int, int] = (0,1,2),
        font_size = 12,
        **kwargs,
    ) -> None:
        self.side_length = side_length
        self.thread_idx = thread_idx
        self.visible = False
        super().__init__(
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            stroke_width=stroke_width,
            **kwargs,
        )

    def generate_points(self) -> None:
        for vect in reversed([IN, OUT, LEFT, RIGHT, UP, DOWN]):
            face = Square(
                side_length=self.side_length,
                shade_in_3d=True,
            )
            face.flip()
            face.shift(self.side_length * OUT / 2.0)
            face.apply_matrix(z_to_vector(vect))

            self.add(face)
    init_points = generate_points

texts = []

class Block:
  def __init__(self, n, center, show_tid=True):
      self.threads = [[[None] * n for j in range(n)] for i in range(n)]
      current_pos = center.copy()
      self.n = n
      for x in range(n):
        current_pos[1] = center[1]
        current_pos += 0.25 * RIGHT
        for y in range(n):
          current_pos[2] = center[2]
          current_pos += 0.25 * DOWN
          for z in range(n):
            current_pos += 0.25 * IN
            t = Thread(side_length=0.25, stroke_width=0.5,fill_opacity=1)
            if show_tid:
              if z == 0:
                texts.append(Text(str(x), font_size=15).move_to(t.get_corner(OUT)))
                t.add(texts[-1])
              if y == 0:
                texts.append(Text(str(z), font_size=15).move_to(t.get_corner(UP)).rotate(radians(90),LEFT))
                t.add(texts[-1])
              if x == 0:
                texts.append(Text(str(y), font_size=15).move_to(t.get_corner(LEFT)).rotate(radians(90),DOWN))
                t.add(texts[-1])
            self.threads[x][y][z] = t.move_to(current_pos)

  def create(self, x_range=1, y_range=1, z_range=1, z_index=0):
    anims = []
    for x in range(x_range):
      for y in range(y_range):
        for z in range(z_range):
          if self.threads[x][y][z] is not None and not self.threads[x][y][z].visible:
            self.threads[x][y][z].visible=True
            self.threads[x][y][z].z_index=z_index
            for so in self.threads[x][y][z].submobjects:
              so.z_index=z_index
            anims.append(Create(self.threads[x][y][z]))
    return anims

class Block2:
  def __init__(self, x_r, y_r, z_r, center, show_tid=True):
      self.threads = [[[None] * x_r for j in range(y_r)] for i in range(z_r)]
      current_pos = center.copy()
      self.x_r = x_r
      self.y_r = y_r
      self.z_r = z_r
      for x in range(x_r):
        current_pos[1] = center[1]
        current_pos += 0.25 * RIGHT
        for y in range(y_r):
          current_pos[2] = center[2]
          current_pos += 0.25 * DOWN
          for z in range(z_r):
            current_pos += 0.25 * IN
            t = Thread(side_length=0.25, stroke_width=0.5,fill_opacity=1)
            self.threads[x][y][z] = t.move_to(current_pos)

  def create(self, x_range=1, y_range=1, z_range=1, z_index=0):
    anims = []
    for x in range(x_range):
      for y in range(y_range):
        for z in range(z_range):
          if self.threads[x][y][z] is not None and not self.threads[x][y][z].visible:
            self.threads[x][y][z].visible=True
            self.threads[x][y][z].z_index=z_index
            for so in self.threads[x][y][z].submobjects:
              so.z_index=z_index
            anims.append(Create(self.threads[x][y][z]))
    return anims

  def rotate(self):
    anims = []
    for x in range(self.x_r):
      for y in range(self.y_r):
        for z in range(self.z_r):
          anims.append(Rotate(self.threads[x][y][z], angle = PI/4, axis = UP+RIGHT+OUT, about_point=ORIGIN))
    return anims

  def get_entries(self):
    entries = []
    for z in range(self.z_r):
      for y in range(self.y_r):
        for x in range(self.x_r):
          entries.append(self.threads[x][y][z])
    return entries

class KernelGrid(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(
        # GTTSService()
        RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU programming", font_size=72).shift(2*UP)
    with self.voiceover(text="Hello and welcome to episode 2 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Kernel Grid", font_size=48).next_to(title, DOWN)

    with self.voiceover(text="In this episode, we are going to talk about the kernel grid") as trk:
      self.play(Write(subtitle))

    self.play(Unwrite(title), Unwrite(subtitle))

    n = 6 
    v2 = Matrix([*[[f"b_{i}"] for i in range(n-2)], ["\\vdots"], ["b_n"]], element_alignment_corner=ORIGIN).shift(DOWN)
    plus = Tex("+").next_to(v2, LEFT)
    v1 = Matrix([*[[f"a_{i}"] for i in range(n-2)], ["\\vdots"], ["a_n"]], element_alignment_corner=ORIGIN).next_to(plus, LEFT)
    eq = Tex("=").next_to(v2, RIGHT)
    v3 = Matrix([*[["?"] for i in range(n-2)], ["\\vdots"], ["?"]], element_alignment_corner=ORIGIN).next_to(eq, RIGHT)
    
    fs = 32
    block1 = SurroundingRectangle(VGroup(*v1.get_entries()[:2])).shift(1.5*LEFT)
    t1 = []
    t1.append(Tex("$T_0$", font_size=fs).move_to(block1.get_corner(UP)+DOWN*0.2, aligned_edge=UP))
    t1.append(Tex("$T_1$", font_size=fs).move_to(block1.get_corner(DOWN)-DOWN*0.2, aligned_edge=DOWN))
    t1.append(Tex("$B_0$", font_size=fs).next_to(block1, LEFT))
    block2 = SurroundingRectangle(VGroup(*v1.get_entries()[2:4])).shift(1.5*LEFT)
    t2 = []
    t2.append(Tex("$T_0$", font_size=fs).move_to(block2.get_corner(UP)+DOWN*0.2, aligned_edge=UP))
    t2.append(Tex("$T_1$", font_size=fs).move_to(block2.get_corner(DOWN)-DOWN*0.2, aligned_edge=DOWN))
    t2.append(Tex("$B_1$", font_size=fs).next_to(block2, LEFT))

    block3 = SurroundingRectangle(VGroup(*v1.get_entries()[4:])).shift(1.5*LEFT)
    t3 = []
    t3.append(Tex("$T_0$", font_size=fs).move_to(block3.get_corner(UP)+DOWN*0.2, aligned_edge=UP))
    t3.append(Tex("$T_1$", font_size=fs).move_to(block3.get_corner(DOWN)-DOWN*0.2, aligned_edge=DOWN))
    t3.append(Tex("$B_{\\frac{n}{2}}$", font_size=fs).next_to(block3, LEFT))

    with self.voiceover(text="""During the last episode we've presented a vector addition kernel, where we launched blocks of 2 threads""") as trk:
      self.play(*[Create(x) for x in [v1, v2, v3, plus, eq]])
      self.play(Create(block1), *[Write(t) for t in t1])
      self.play(Create(block2), *[Write(t) for t in t2])
      self.play(Create(block3), *[Write(t) for t in t3])

    m = 3
    n = 3
    blocks = [[[None] * m for j in range(m)] for i in range(m)]
    start_pos = ORIGIN.copy() + 2*(LEFT + UP + OUT)
    current_pos = ORIGIN.copy() + 2*(LEFT + UP + OUT)
    for x in range(m):
      current_pos[1] = start_pos[1]
      for y in range(m):
        current_pos[2] = start_pos[2]
        for z in range(m):
          blocks[x][y][z] = Block(n, current_pos, show_tid=z==0 and y>0)
          current_pos += 2 * IN
        current_pos += 2 * DOWN
      current_pos += 2 * RIGHT
    
    self.play(*[Uncreate(x) for x in [v1, v2, v3, plus, eq]])
    self.play(Uncreate(block1), *[Unwrite(t) for t in t1], Uncreate(block2), *[Unwrite(t) for t in t2], Uncreate(block3), *[Unwrite(t) for t in t3])

    code = """int N=6;
int BLOCK_SIZE=2;
add<<<ceil(N/(float)BLOCK_SIZE), BLOCK_SIZE>>>(N, a_d, b_d, c_d); """

    code_obj = Code(code=code, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).shift(2*UP)

    gpu_code = """__global__ void add(int n , float* a, float* b, float* c)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}"""
    gpu_code_obj = Code(code=gpu_code, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).shift(2*DOWN)
    def transform_code(tidx, bidx):
      new = f"""__global__ void add(int n , float* a, float* b, float* c)
{{
  //int i = blockIdx.x * blockDim.x + threadIdx.x;
  int i = {bidx} * 2 + {tidx};
  if (i < n)
  {{
    c[i] = a[i] + b[i];
  }}
}}"""
      c = Code(code=new, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).shift(2*DOWN)
      c.code = remove_invisible_chars(c.code)
      c.code[2].set_color(GREEN_E)
      return c


    with self.voiceover(text="""If we launched the kernel with just 6 elements and 2 threads per block""") as trk:
      self.play(Create(gpu_code_obj))
      self.play(Create(code_obj))

    with self.voiceover(text="""the resulting kernel grid would look like this""") as trk:
      self.play(LaggedStart(blocks[0][1][0].create(x_range=2, z_index=1)))
      self.play(LaggedStart(blocks[1][1][0].create(x_range=2, z_index=1)))
      self.play(LaggedStart(blocks[2][1][0].create(x_range=2, z_index=1)))

    l1 = Line(blocks[0][1][0].threads[0][0][0].get_corner(UP+LEFT), blocks[0][1][0].threads[1][0][0].get_corner(UP+RIGHT))
    b1 = Brace(l1, direction=UP)
    t1 = b1.get_text("threadIdx.x").scale(0.6)

    l2 = Line(blocks[0][1][0].threads[0][0][0].get_corner(UP+LEFT), blocks[2][1][0].threads[1][0][0].get_corner(UP+RIGHT))
    b2 = Brace(l2, direction=UP, buff=0.4)
    t2 = b2.get_text("blockIdx.x").scale(0.6)

    with self.voiceover(text="""Where the code gets assigned a thread index""") as trk:
      self.play(Create(b1), Write(t1))

    with self.voiceover(text="""And a block index""") as trk:
      self.play(Create(b2), Write(t2))

    with self.voiceover(text="""When running each thread in our block, it will run a copy of our code with 
                        values of blockIdx and threadIdx set to match the curerntly executed thread.
                        The blockDim variable represents our block dimension and is constant across all threads
                        in our case - we set the block size to 2""") as trk:
      for b in range(3):
        for t in range(2):
          blocks[b][1][0].threads[t][0][0].save_state()
          self.play(blocks[b][1][0].threads[t][0][0].animate.set_color(GREEN), Transform(gpu_code_obj, transform_code(t, b)))
          self.wait(0.5)
          self.play(Restore(blocks[b][1][0].threads[t][0][0]))
          self.wait(0.5)

    with self.voiceover(text="""Some of the more alert viewers might have noticed that we keep using threadIdx and blockIdx x values,
                        and that might imply that there are more dimensions""") as trk:
      self.play(Uncreate(b2), Unwrite(t2), Uncreate(b1), Unwrite(t1))
      self.wait(4)
      self.play(Uncreate(gpu_code_obj))

    def transform_run(dim_grid, dim_block):
      code = f"""dim3 dimGrid({','.join(map(str,dim_grid))});
dim3 dimBlock({','.join(map(str,dim_block))});
add<<<dimGrid, dimBlock>>>(N, a_d, b_d, c_d); """
      return Code(code=code, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).shift(2*UP)

    with self.voiceover(text="""And that is indeed true, we can run up to 3 dimensions by passing in a dim3 variable 
                        as our kernel parameters""") as trk:
      self.wait(2)
      self.play(Transform(code_obj, transform_run([3,1,1], [2,1,1])))



    with self.voiceover(text="""so a 2 dimensional kernel grid would look like this""") as trk:
      self.play(Transform(code_obj, transform_run([3,2,1], [2,2,1])), 
                LaggedStart(blocks[0][1][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[1][1][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[2][1][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[0][2][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[1][2][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[2][2][0].create(x_range=2, y_range=2, z_index=1)))

    self.wait(1)
    creations = []
    for x in range(m):
      for y in range(m):
        for z in range(m):
          creations.extend(blocks[x][y][z].create(x_range=n, y_range=n, z_range=n, z_index=0))

    self.add_fixed_in_frame_mobjects(code_obj)
    self.add_fixed_orientation_mobjects(code_obj)

    with self.voiceover(text="""While a 3 dimensional grid might look like this""") as trk:
      self.move_camera(theta=-radians(25), gamma=radians(85), phi=-radians(45),
                       added_anims=[LaggedStart(*creations, lag_ratio=0.001), Transform(code_obj, transform_run([m,m,m], [n,n,n]))])

    self.wait(2)
    self.begin_ambient_camera_rotation(-0.1, about="phi")
    with self.voiceover(text="You might wonder what is the purpose of multiple dimensions") as trk:
      self.play(*[Unwrite(x) for x in texts])

    self.wait(0.5)
    with self.voiceover(text="""and it's mostly just syntactic sugar - some algorithms operate on multidimensional data
                        and checking boundary conditions for them might be easier in those""") as trk:
      pass

    self.wait(0.5)
    with self.voiceover(text="""also, they might be more readable when you express them in a row/column form""") as trk:
      pass

    self.wait(0.5)
    with self.voiceover(text="""as a side note - there might be some edge cases where using a multidimensional grid instead of a single dimensional grid
                        results in a bit smaller register usage but that is rarely of big importance""") as trk:
      pass

    self.wait(0.5)
    with self.voiceover(text="""As an example we can look into a square matrix multiplication kernel""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
      self.stop_ambient_camera_rotation("phi")
      self.move_camera(theta=-radians(90), gamma=radians(0), phi=radians(0))


    with self.voiceover(text="""As a remainder, matrix multiplication is a function that takes 2 matrices as the input
                        and returns another matrix whose entries are dot products beteen rows of the first matrix and columns of the second one""") as trk:
      mul = Tex("$\\cdot$").shift(2*LEFT + UP)
      m1 = Matrix([[f"a_{{0,0}}", f"a_{{0,1}}"], [f"a_{{1,0}}", f"a_{{1,1}}"]]).next_to(mul, LEFT)
      m2 = Matrix([[f"b_{{0,0}}", f"b_{{0,1}}"], [f"b_{{1,0}}", f"b_{{1,1}}"]]).next_to(mul, RIGHT)
      eq = Tex("$=$").next_to(m2, RIGHT)
      m3 = Matrix([[f"c_{{0,0}}", f"c_{{0,1}}"], [f"c_{{1,0}}", f"c_{{1,1}}"]]).next_to(eq, RIGHT)
      m = [[f"$c_{{{j},{i}}} = a_{{{j},0}}*b_{{0,{i}}}+a_{{{j},0}}*b_{{1,{i}}}$" for i in range(2)] for j in range(2)]
      fs = 48
      t1 = Tex(m[0][0], font_size = fs).next_to(m1, DOWN, aligned_edge=LEFT)
      t2 = Tex(m[0][1], font_size = fs).next_to(t1, DOWN, aligned_edge=LEFT)
      t3 = Tex(m[1][0], font_size = fs).next_to(t2, DOWN, aligned_edge=LEFT)
      t4 = Tex(m[1][1], font_size = fs).next_to(t3, DOWN, aligned_edge=LEFT)
      

      self.add(m1)
      self.add(mul)
      self.add(m2)
      self.add(eq)
      self.add(m3)

      i1 = SurroundingRectangle(m1.get_entries()[:2], color=BLUE)
      i2 = SurroundingRectangle(VGroup(m2.get_entries()[0], m2.get_entries()[2]), color=BLUE)
      g1 = VGroup(i1.copy(), i2.copy(),
                  m1.get_entries()[0].copy(), m1.get_entries()[1].copy(), 
                  m2.get_entries()[0].copy(), m2.get_entries()[2].copy()) 
      self.play(Create(i1), Create(i2))
      self.play(Transform(g1, t1, replace_mobject_with_target_in_scene=True))
      self.wait(1)

      dd = m1.get_entries()[0].get_y() - m1.get_entries()[2].get_y()
      dr = m2.get_entries()[0].get_x() - m2.get_entries()[1].get_x()
      self.play(i2.animate.shift(LEFT * dr))
      g1 = VGroup(i1.copy(), i2.copy(),
                  m1.get_entries()[0].copy(), m1.get_entries()[1].copy(), 
                  m2.get_entries()[1].copy(), m2.get_entries()[3].copy()) 
      self.play(Transform(g1, t2, replace_mobject_with_target_in_scene=True))
      self.wait(1)
      self.play(i1.animate.shift(DOWN * dd), i2.animate.shift(RIGHT * dr))
      g1 = VGroup(i1.copy(), i2.copy(),
                  m1.get_entries()[2].copy(), m1.get_entries()[3].copy(), 
                  m2.get_entries()[0].copy(), m2.get_entries()[2].copy()) 
      self.play(Transform(g1, t3, replace_mobject_with_target_in_scene=True))
      self.wait(1)
      self.play(i2.animate.shift(LEFT * dr))
      g1 = VGroup(i1.copy(), i2.copy(),
                  m1.get_entries()[2].copy(), m1.get_entries()[3].copy(), 
                  m2.get_entries()[1].copy(), m2.get_entries()[3].copy()) 
      self.play(Transform(g1, t4, replace_mobject_with_target_in_scene=True))
      self.wait(1)

    with self.voiceover(text="""I do realize that the description was very brief so I'm going to leave some more links in the description
                        for those that are unfamilliar with the operation""") as trk:
      pass

    m4 = Matrix([[f"a_{{{j},{i}}}" for i in range(3)] for j in range(3)])

    with self.voiceover(text="""Before we jump into the code, there is one thing that you have to know about memory layout""") as trk:
      self.play(*[Uncreate(x) for x in [m2, m3, i1, i2]], 
                *[Unwrite(x) for x in [t1, t2, t3, t4, eq, mul]])

    with self.voiceover(text="""When we create a 2D array in our code, the computer still stores it in 1 Dimension - the 2D access is just an 
                        abstraction that is easier for us to read""") as trk:
      self.play(Transform(m1, m4))

      t1 = Tex("Row * Width + Column").set_color(BLUE).next_to(m1, DOWN)
      v = Matrix([[f"a_{i}" for i in range(9)]]).set_color(GREEN).next_to(t1, DOWN)
      self.play(Create(v.get_brackets()[0]))
      for i in range(3):
        self.play(Transform(VGroup(m1.get_entries()[i*3:(i+1)*3]).copy(), VGroup(v.get_entries()[i*3:(i+1)*3]), replace_mobject_with_target_in_scene=True))
      self.play(Create(v.get_brackets()[1]))


    with self.voiceover(text="""In cuda we get access to the raw pointer so we actually have to index into it ourselves - when we have our row and column index""") as trk:
      pass
    with self.voiceover(text="""we can do that by multiplying the row by our matrix width
                        and adding the column index into it""") as trk:
      self.play(Write(t1))
    self.wait(1)


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
    matmul_sd="""__global__ void matmul_elem_onedim
    (int n, float* a, float* b, float* c)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int row = idx/n;
  int column = idx%n;
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

    matmul_obj = Code(code=matmul, tab_width=2, language="c", font_size=14, background="rectangle", line_no_buff=0.1, corner_radius=0.1).shift(3*LEFT)
    matmul_sd_obj = Code(code=matmul_sd, tab_width=2, language="c", font_size=14, background="rectangle", line_no_buff=0.1, corner_radius=0.1).next_to(matmul_obj, RIGHT)

    matmul_obj.code = remove_invisible_chars(matmul_obj.code)
    matmul_sd_obj.code = remove_invisible_chars(matmul_sd_obj.code)


    with self.voiceover(text="""To run out matrix multiplication kernel, we can assign each thread to 1 element in our output array""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
      self.play(Create(matmul_obj))

    hl = SurroundingRectangle(matmul_obj.code[3:5], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""We first calculate our row and column indices based on the current thread and block""") as trk:
      self.play(Create(hl))

    hl_t = SurroundingRectangle(matmul_obj.code[5], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Then we do the boundary check not to read and write outside our matrices""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(matmul_obj.code[7], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Then we create an intermediate variable that will store our dot product""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(matmul_obj.code[8:12], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And we iterate over the row vector of the first matrix, and the column vector of the second matrix
                        calculating the dot product""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(matmul_obj.code[12], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Finally, we save our result in the output matrix""") as trk:
      self.play(Transform(hl, hl_t))

    self.wait(1)
    self.play(Uncreate(hl))
    self.wait(1)
    hl = SurroundingRectangle(matmul_sd_obj.code[4:6], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And just as I mentioned before, we could also do the same thing with a single dimensional grid.""") as trk:
      self.play(Create(matmul_sd_obj))
    self.wait(1)
    with self.voiceover(text="""We just have to parse the rows and columns from our x dimension, this adds a bit of an overhead but it's negligable 
                        compared to the rest of the work done by the kernel""") as trk:
      self.play(Create(hl))


    self.wait(2)
    x_r, y_r, z_r = 3,2,3
    block = Block2(x_r, y_r, z_r, ORIGIN+UP, False)
    with self.voiceover(text="""And the simillar memory pattern happens when we extend out data to the third dimension""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
      self.play(*block.create(x_r, y_r, z_r))

      self.play(*block.rotate())
      self.wait(1)

    v = Matrix([[f"a_{{{i}}}" for i in range(x_r*y_r*z_r)]]).scale(0.5)


    self.play(Create(v.get_brackets()[0]))
    with self.voiceover(text="""It just simply gets flattened out, across each data dimension we add""") as trk:
      for z in range(z_r):
        for y in range(y_r):
          i = z * x_r * y_r + y * x_r
          self.play(Transform(VGroup(*block.get_entries()[i:i+3]), VGroup(v.get_entries()[i:i+3]), replace_mobject_with_target_in_scene=True))
    self.play(Create(v.get_brackets()[1]))

    with self.voiceover(text="""Can you come up with the formula for our 1 dimensional index
                        when we know our x, y and z coordinates?""") as trk:
      pass

    self.wait(2)
    t1 = Tex("Z * Width * Height + Y * Width + X", font_size=36).set_color(BLUE).next_to(v, UP)
    with self.voiceover(text="""If you guessed the following - you were right!""") as trk:
      self.play(Write(t1))

    self.wait(2)
    with self.voiceover(text="""Now that we have the theory behind us, I'm going to leave an excercise for those that want to practice
                        running a multidimensional kernel grid""") as trk:
      pass
    self.wait(1)


    with self.voiceover(text="""And the excercise looks like this: take in 3 arrays as the input""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])

    a = Tex("$a \\in \\mathbb{R}^{x \\times y \\times z}$").shift(UP)
    b = Tex("$b \\in \\mathbb{R}^{x \\times y}$").next_to(a, DOWN, aligned_edge=LEFT)
    c = Tex("$c \\in \\mathbb{R}^{x}$").next_to(b, DOWN, aligned_edge=LEFT)
    out = Tex("$out[x][y][z] = a[x][y][z] + b[x][y] + c[x]$").next_to(c, DOWN)
    with self.voiceover(text="""A 3 dimensional array a""") as trk:
      self.play(Write(a))

    with self.voiceover(text="""A 2 dimensional array b""") as trk:
      self.play(Write(b))

    with self.voiceover(text="""A 1 dimensional array c""") as trk:
      self.play(Write(c))

    with self.voiceover(text="""And produce the output that is a 3 dimensional array being a sum of 3 input arrays
                        broadcasted to 3 dimensions""") as trk:
      self.play(Write(out))

    with self.voiceover(text="""Please, share and discuss your code in the comments. Also if you liked the video, 
                        subscribe to stay up to date, leave a thumbs up and share it with your friends""") as trk:
      pass

    with self.voiceover(text="""See you in the next episode - bye""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])

    self.wait(2)
