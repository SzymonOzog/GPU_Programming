from manim import *
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
                t.add(Text(str(x), font_size=15).move_to(t.get_corner(OUT)))
              if y == 0:
                t.add(Text(str(z), font_size=15).move_to(t.get_corner(UP)).rotate(radians(90),LEFT))
              if x == 0:
                t.add(Text(str(y), font_size=15).move_to(t.get_corner(LEFT)).rotate(radians(90),DOWN))
            if x == 0 or y == 0 or z == 0:
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

class KernelGrid(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(
        GTTSService()
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
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

    with self.voiceover(text="""In the last episode we've presented a vector addition kernel, where we launched blocks 2 threads""") as trk:
      self.play(*[Create(x) for x in [v1, v2, v3, plus, eq]])
      self.play(Create(block1), *[Write(t) for t in t1])
      self.play(Create(block2), *[Write(t) for t in t2])
      self.play(Create(block3), *[Write(t) for t in t3])

    m = 4
    n = 4
    blocks = [[[None] * m for j in range(m)] for i in range(m)]
    current_pos = ORIGIN.copy() + 2*(LEFT + UP)
    for x in range(m):
      current_pos[1] = ORIGIN[1]
      for y in range(m):
        current_pos[2] = ORIGIN[2]
        for z in range(m):
          blocks[x][y][z] = Block(n, current_pos, show_tid=z==0)
          current_pos += 2 * IN
        current_pos += 2 * DOWN
      current_pos += 2 * RIGHT
    
    self.play(*[Uncreate(x) for x in [v1, v2, v3, plus, eq]])
    self.play(Uncreate(block1), *[Unwrite(t) for t in t1])
    self.play(Uncreate(block2), *[Unwrite(t) for t in t2])
    self.play(Uncreate(block3), *[Unwrite(t) for t in t3])

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
  int i = {bidx} * 2 + {tidx};
  if (i < n)
  {{
    c[i] = a[i] + b[i];
  }}
}}"""
      return Code(code=new, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).shift(2*DOWN)


    with self.voiceover(text="""If we launched the kernel with just 6 elements and 2 threads per block""") as trk:
      self.play(Create(gpu_code_obj))
      self.play(Create(code_obj))

    with self.voiceover(text="""the resulting kernel grid would look like this""") as trk:
      self.play(LaggedStart(blocks[0][0][0].create(x_range=2, z_index=1)))
      self.play(LaggedStart(blocks[1][0][0].create(x_range=2, z_index=1)))
      self.play(LaggedStart(blocks[2][0][0].create(x_range=2, z_index=1)))

    l1 = Line(blocks[0][0][0].threads[0][0][0].get_corner(UP+LEFT), blocks[0][0][0].threads[1][0][0].get_corner(UP+RIGHT))
    b1 = Brace(l1, direction=UP)
    t1 = b1.get_text("threadIdx.x").scale(0.6)

    l2 = Line(blocks[0][0][0].threads[0][0][0].get_corner(UP+LEFT), blocks[2][0][0].threads[1][0][0].get_corner(UP+RIGHT))
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
          blocks[b][0][0].threads[t][0][0].save_state()
          gpu_code_obj.save_state()
          self.play(blocks[b][0][0].threads[t][0][0].animate.set_color(GREEN), Transform(gpu_code_obj, transform_code(b, t)))
          self.wait(0.5)
          self.play(Restore(blocks[b][0][0].threads[t][0][0]), Restore(gpu_code_obj))
          self.wait(0.5)

    with self.voiceover(text="""Some of the more alert viewers might have noticed that we keep using threadIdx and blockIdx x values,
                        and that might imply that there are more dimensions""") as trk:
      self.play(Uncreate(b2), Unwrite(t2), Uncreate(b1), Unwrite(t1))
      self.play(Uncreate(gpu_code_obj))

    def transform_run(dim_grid, dim_block):
      code = f"""dim3 dimGrid({','.join(map(str,dim_grid))});
dim3 dimBlock({','.join(map(str,dim_block))});
add<<<dimGrid, dimBlock>>>(N, a_d, b_d, c_d); """
      return Code(code=code, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).shift(2*UP)

    with self.voiceover(text="""And that is indeed true, we can run up to 3 dimensions by passing in a dim3 variable 
                        as our kernel parameters""") as trk:
      self.play(Transform(code_obj, transform_run([3,1,1], [2,1,1])))



    with self.voiceover(text="""so a 2 dimensional kernel grid would look like this""") as trk:
      self.play(Transform(code_obj, transform_run([3,2,1], [2,2,1])), 
                LaggedStart(blocks[0][0][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[1][0][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[2][0][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[0][1][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[1][1][0].create(x_range=2, y_range=2, z_index=1)),
                LaggedStart(blocks[2][1][0].create(x_range=2, y_range=2, z_index=1)))

    self.wait(1)
    creations = []
    for x in range(m):
      for y in range(m):
        for z in range(m):
          if x == 0 or y == 0 or z == 0:
            creations.extend(blocks[x][y][z].create(x_range=n, y_range=n, z_range=n, z_index=0))

    self.add_fixed_in_frame_mobjects(code_obj)
    self.add_fixed_orientation_mobjects(code_obj)

    with self.voiceover(text="""While a 3 dimensional grid might look like this""") as trk:
      self.move_camera(theta=-radians(65), gamma=radians(45), phi=-radians(45),
                       added_anims=[LaggedStart(*creations, lag_ratio=0.001), Transform(code_obj, transform_run([m,m,m], [n,n,n]))])

    self.wait(2)
