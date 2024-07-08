from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
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


class Introduction(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None))
    
    title = Text("GPU programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 0 in the series on GPU programming") as trk:
      self.play(Write(title))

    self.play(Unwrite(title))

    cpu_rects = []
    cpu_texts = []
    def create_alu():
      alu = Rectangle(width=1, height=1, color=BLUE, fill_opacity=0.5)
      sw = 0.04
      cpu_rects.append(alu)
      alu_text = Text("Core", font_size=14).move_to(alu.get_center())
      cpu_texts.append(alu_text)

      cache = Rectangle(width=1, height=0.25, fill_opacity=0.5, color=RED).next_to(alu, DOWN, aligned_edge=LEFT, buff=sw)
      cpu_rects.append(cache)
      cache_text = Text("L1 Cache", font_size=14).move_to(cache.get_center())
      cpu_texts.append(cache_text)
      
      control = Rectangle(width=0.5, height=1.25+sw, color=PURPLE, fill_opacity=0.5).next_to(alu, RIGHT, aligned_edge=UP, buff=sw)
      cpu_rects.append(control)
      control_text1 = Text("Con", font_size=14)
      control_text2 = Text("trol", font_size=14)
      ct = VGroup(control_text1, control_text2).arrange(DOWN, buff=0.05).move_to(control)
      cpu_texts.append(control_text1)
      cpu_texts.append(control_text2)
      return VGroup(alu, alu_text, cache, cache_text, control, ct)


    alu1 = create_alu().shift(4*LEFT+UP)
    alu2 = create_alu().next_to(alu1, RIGHT, buff=0.1)
    alu3 = create_alu().next_to(alu1, DOWN, aligned_edge=LEFT, buff=0.1)
    alu4 = create_alu().next_to(alu3, RIGHT, buff=0.1)
    
    cache1 = Rectangle(width=alu1.width, height=0.4, color=RED, fill_opacity=0.5).next_to(alu3, DOWN, aligned_edge=LEFT, buff=0.1)
    cache2 = Rectangle(width=alu1.width, height=0.4, color=RED, fill_opacity=0.5).next_to(alu4, DOWN, aligned_edge=LEFT, buff=0.1)
    cpu_rects.append(cache1)
    cpu_rects.append(cache2)
    cpu_texts.append(Text("L2 Cache", font_size=14).move_to(cache1))
    cpu_texts.append(Text("L2 Cache", font_size=14).move_to(cache2))

    cache3 = Rectangle(width=cache1.width*2 + 0.1, height=0.5, color=RED, fill_opacity=0.5).next_to(cache1, DOWN, aligned_edge=LEFT, buff=0.1)
    cpu_rects.append(cache3)
    cpu_texts.append(Text("L3 Cache", font_size=14).move_to(cache3))


    dram_cpu = Rectangle(width=cache3.width, height=0.7, color=GREEN, fill_opacity=0.5).next_to(cache3, DOWN, buff=0.1).align_to(alu3, LEFT)
    dram_cpu_text = Text("DRAM", font_size=24).move_to(dram_cpu.get_center())
    cpu_rects.append(dram_cpu)
    cpu_texts.append(dram_cpu_text)

    gpu_rects = []
    gpu_texts = []
    gpu_alu_list = []
    for _ in range(5):
      cc = VGroup(Rectangle(width=0.5, height=0.2, color=PURPLE, fill_opacity=0.5), 
                  Rectangle(width=0.5, height=0.2, color=RED, fill_opacity=0.5)).arrange(DOWN, buff=0.1)
      gpu_rects.append(cc[0])
      gpu_rects.append(cc[1])
      alus = [Rectangle(width=0.5, height=0.5, color=BLUE, fill_opacity=0.5) for _ in range(8)]
      gpu_rects.extend(alus)
      gpu_alu_list.append(VGroup(cc, *alus).arrange(RIGHT, buff=0.1))
    gpu_alus = VGroup(*gpu_alu_list).scale(0.8).arrange(DOWN, buff=0.16).shift(RIGHT * 4)


    l2 = Rectangle(width=4.25, height=0.4, color=RED, fill_opacity=0.5).match_width(gpu_alus).next_to(gpu_alus, DOWN, buff=0.1)
    gpu_rects.append(l2)
    l2_text = Text("L2 Cache", font_size=14).move_to(l2)
    gpu_texts.append(l2_text)

    dram_gpu = Rectangle(width=4.25, height=0.5, color=GREEN, fill_opacity=0.5).match_width(gpu_alus).next_to(l2, DOWN, buff=0.1)
    gpu_rects.append(dram_gpu)
    dram_gpu_text = Text("DRAM", font_size=14).move_to(dram_gpu.get_center())
    gpu_texts.append(dram_gpu_text)

    cpu = VGroup(*cpu_rects, *cpu_texts, dram_cpu, dram_cpu_text)

    gpu = VGroup(gpu_alus, l2, l2_text, dram_gpu, dram_gpu_text).match_height(cpu).align_to(cpu, UP)

    cpu_title = Text("CPU").scale(0.8).next_to(cpu, UP)
    gpu_title = Text("GPU").scale(0.8).next_to(gpu, UP)

    subobjects = []
    queue = [cpu, gpu]
    while queue:
      o = queue.pop()
      subobjects.append(o)
      queue.extend(o.submobjects)

    for mo in subobjects:
      for so in mo.submobjects.copy():
        if any(so in x for x in [cpu_rects, cpu_texts, gpu_rects, gpu_texts]):
          mo.remove(so)

    font_size = 36 
    cpu_details = BulletedList("Low latency",  "Low throughput", "Optimized for serial operations", font_size=font_size).next_to(cpu_title, DOWN)
    gpu_details = BulletedList("High latency", "High throughput", "Optimized for parallel operations", font_size=font_size).next_to(gpu_title, DOWN)
    with self.voiceover(text="""It's supposed to give you a quickstart on how, and when to run code on a GPU
                        as opposed to the CPU and what are the key differences between them""") as trk:
      self.play(Write(cpu_title), Write(gpu_title))
      self.wait(1.6)
      for i in range(3):
        self.play(Write(cpu_details[i]), Write(gpu_details[i]))
        self.wait(1.5)

    with self.voiceover(text="""This time I'm recording the series as the episodes are released so there won't be a 
                        direct table of contents, but there is a rough outline of topics that I want to cover""") as trk:
      while trk.get_remaining_duration() > 4:
        self.wait(0.3)
      for i in range(3):
        self.play(Unwrite(cpu_details[2-i]), Unwrite(gpu_details[2-i]))



    cpu_code = """void add(int n , float* a, float* b, float* c)
{
  for (int i = 0; i<n; i++)
  {
    c[i] = a[i] + b[i];
  }
}"""
    cpu_code_obj = Code(code=cpu_code, tab_width=2, language="c", font_size=11, background="rectangle", line_no_buff=0.1, corner_radius=0.1).next_to(cpu_title, DOWN)
    gpu_code = """__global__ void add(int n , float* a, float* b, float* c)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}"""
    gpu_code_obj = Code(code=gpu_code, tab_width=2, language="c", font_size=11, background="rectangle", line_no_buff=0.1, corner_radius=0.1).next_to(gpu_title, DOWN).shift(0.2*LEFT)

    with self.voiceover(text="""First we will go over the overall structure of a GPU, how it's architecture differs
                        from the CPU, and we will write a few simple kernels and see how the code compares to what we are used to with classical programming""") as trk:
      self.play(*[Create(x) for x in cpu_rects])
      self.play(*[Write(x) for x in cpu_texts])

      self.play(*[Create(x) for x in gpu_rects])
      self.play(*[Write(x) for x in gpu_texts])

      self.wait(3)
      self.play(*[Unwrite(x) for x in cpu_texts], *[Uncreate(x) for x in cpu_rects])

      self.play(*[Unwrite(x) for x in gpu_texts],*[Uncreate(x) for x in gpu_rects])
      self.play(Write(cpu_code_obj), Write(gpu_code_obj))

    self.play(Unwrite(cpu_code_obj), Unwrite(gpu_code_obj), Unwrite(gpu_title), Unwrite(cpu_title))

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

    anims = []
    code_obj = Code(code=f"<<<dim3(1, 1, 1), dim3({n}, 1, 1)>>>", tab_width=2, language="c").shift(2*UP)

    z_max = 2
    with self.voiceover(text="""We'll then go into more detail on how is the kernel grid organized, and how to leverage multiple dimensions,
                        and again, we will write a few kernels to get some more understanding of the topic""") as trk:
      self.play(Create(code_obj))
      self.add_fixed_in_frame_mobjects(code_obj)
      self.add_fixed_orientation_mobjects(code_obj)
      self.play(LaggedStart(blocks[0][0][0].create(x_range=n, z_index=z_max)))
      self.wait(0.5)

      self.play(code_obj.animate.become(Code(code=f"<<<dim3(1, 1, 1), dim3({n}, {n}, 1)>>>", tab_width=2, language="c").shift(2*UP)), LaggedStart(blocks[0][0][0].create(x_range=n, y_range=n, z_index=z_max)))
      self.wait(0.5)

      self.play(code_obj.animate.become(Code(code=f"<<<dim3(2, 1, 1), dim3({n}, {n}, 1)>>>", tab_width=2, language="c").shift(2*UP)), LaggedStart(blocks[1][0][0].create(x_range=n, y_range=n, z_index=z_max)))
      self.wait(0.5)

      self.play(code_obj.animate.become(Code(code=f"<<<dim3(2, 2, 1), dim3({n}, {n}, 1)>>>", tab_width=2, language="c").shift(2*UP)),
                LaggedStart(blocks[0][1][0].create(x_range=n, y_range=n, z_index=z_max)), LaggedStart(blocks[1][1][0].create(x_range=n, y_range=n, z_index=z_max)))

      self.play(code_obj.animate.become(Code(code=f"<<<dim3({m}, {m}, {m}), dim3({n}, {n}, {n})>>>", tab_width=2, language="c").shift(2*UP+9*OUT).set_z_index(z_max+1)))
      creations = []
      for x in range(m):
        for y in range(m):
          for z in range(m):
            if x == 0 or y == 0 or z == 0:
              creations.extend(blocks[x][y][z].create(x_range=n, y_range=n, z_range=n, z_index=z_max-z))
      self.move_camera(theta=-radians(65), gamma=radians(45), phi=-radians(45), added_anims=[LaggedStart(*creations, lag_ratio=0.001)])
    anims = [] 
    for obj in self.mobjects:
      anims.append(FadeOut(obj))
    self.play(*anims)
    self.move_camera(theta=-radians(90), gamma=radians(0), phi=radians(0))

    gpu_code = """
__global__ void update_layer(int w, int h, int batch_size, float lr,
                             float* weights, float* biases, float* activations, float* d_l)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float dw = 0.f;
    float db = 0.f;
    for(int i = 0; i < batch_size; i++)
    {
      float act = activations[i*h + row];
      float dl = d_l[i*w + column];
      dw += act*dl;
      db += dl;
    }
    weights[row*w + column] -= lr * dw / batch_size;
    biases[column] -= lr * db / batch_size;
  }
} """
    gpu_code_obj = Code(code=gpu_code, tab_width=2, language="c", font_size=12, background="rectangle")

    with self.voiceover(text="""After that, we will go through a bigger project, where we'll create a simple neural network using just CUDA 
                        and no external libraries""") as trk:
      self.play(Create(gpu_code_obj, run_time=5))

    self.play(Uncreate(gpu_code_obj))
    axes = Axes(x_range=[0, 10, 1], y_range=[0, 7, 1], x_length=7, y_length=5, axis_config={"include_tip": False})
    
    x_text = VGroup(VGroup(MathTex("Computational"), MathTex("Intensity")).arrange(DOWN), MathTex("[\\frac{FLOP}{B}]")).arrange(RIGHT)
    y_text = VGroup(VGroup(MathTex("Attainable"), MathTex("Performance")).arrange(DOWN), MathTex("[FLOPS]")).arrange(RIGHT)
    x_label = axes.get_x_axis_label(x_text, edge=UP, direction=DOWN, buff=MED_SMALL_BUFF)
    y_label = axes.get_y_axis_label(y_text.rotate(PI/2), edge=LEFT, direction=LEFT)

    bandwidth_line = Line(start=axes.c2p(0, 0), end=axes.c2p(7, 7), color=BLUE).set_opacity(0.7)
    bandwidth_label = Tex("Bandwitdth limit", color=BLUE, font_size=40).rotate(PI/4).move_to(bandwidth_line.get_center() + LEFT + 0.5*DOWN)

    throughput_line = Line(start=axes.c2p(0, 5), end=axes.c2p(10, 5), color=YELLOW).set_opacity(0.7)
    throughput_label = Tex("Throughput limit", color=YELLOW, font_size=40).next_to(throughput_line, UP, buff=0.1)

    bandwidth_bound = Polygon(
        axes.c2p(0, 0), axes.c2p(5, 5), axes.c2p(5, 0),
        color=RED, fill_opacity=0.5, stroke_width=0
    )
    
    compute_bound = Polygon(
        axes.c2p(5, 0), axes.c2p(5, 5), axes.c2p(10, 5), axes.c2p(10, 0),
        color=GREEN, fill_opacity=0.5, stroke_width=0
    )
    mem_text = VGroup(Tex("Memory"), Tex("bound")).arrange(DOWN).move_to(bandwidth_bound).shift(0.7*(DOWN+RIGHT))
    compute_text = VGroup(Tex("Compute"), Tex("bound")).arrange(DOWN).move_to(compute_bound)

    graph = VGroup(axes, x_label, y_label, bandwidth_line, bandwidth_label,
                   throughput_line, throughput_label, bandwidth_bound, compute_bound, mem_text, compute_text)
    with self.voiceover(text="""With the fundamentals covered we'll dive into performance characteristics, We'll discuss the most common
                        bottlenecks and some ways to mitgate them. We will use some of that information immediately to optimize our neural network code""") as trk:
      self.play(Create(axes), Write(x_label), Write(y_label))
      self.play(Create(bandwidth_line), Write(bandwidth_label), Create(throughput_line), Write(throughput_label))
      self.play(FadeIn(bandwidth_bound), FadeIn(compute_bound))
      self.play(Create(mem_text), Create(compute_text))
      while trk.get_remaining_duration() > 3:
        self.wait(0.3)
      self.play(Uncreate(mem_text), Uncreate(compute_text))
      self.play(FadeOut(bandwidth_bound), FadeOut(compute_bound))
      self.play(Uncreate(bandwidth_line), Unwrite(bandwidth_label), Uncreate(throughput_line), Unwrite(throughput_label))

    def join(r1, r2, start, double=True):
      nonlocal arrows
      e_y = r2.get_y() + (1 if r2.get_y() < start[1] else -1) * r2.height/2
      end = np.array([start[0], e_y, 0])
      ret = None
      if double: 
        ret = DoubleArrow(start, end, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
      else:
        ret = Arrow(end, start, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
      arrows.append(ret)
      return ret

    shared_store = []
    shared_load = []
    register_store = []
    register_load = []
    local_store = []
    local_load = []
    global_store = []
    global_load = []
    constant_load = []

    thread_objs = []
    rects = []
    texts = []
    arrows = []
    def make_thread(idx=0):
      nonlocal thread_objs, rects, texts
      thread = Rectangle(height=0.5, width=2.2, color=BLUE)
      texts.append(Text(f"Thread {idx}", font_size=15, color=BLUE))
      rects.append(thread)
      thread.add(texts[-1])

      registers = Rectangle(height=0.5, width=1.0, color=GREEN).next_to(thread, UP, aligned_edge=LEFT, buff=0.5)
      texts.append(Text("Registers", font_size=15, color=GREEN).move_to(registers.get_center()))
      registers.add(texts[-1])
      rects.append(registers)

      local = Rectangle(height=0.5, width=1.0, color=GREEN).next_to(thread, UP, aligned_edge=RIGHT, buff=0.5)
      l = Text("Local", font_size=15, color=GREEN)
      m = Text("Memory", font_size=15, color=GREEN)
      VGroup(l, m).arrange(DOWN, buff=0.05).move_to(local.get_center())
      texts.append(l)
      texts.append(m)
      rects.append(local)
      local.add(l)
      local.add(m)

      t_group = VGroup(thread, registers, local)
      t_group.add(join(registers, thread, start=registers.get_corner(DOWN)))
      t_group.add(join(local, thread, start=local.get_corner(DOWN)))

      thread_objs.append(thread)
      return t_group

    def make_block(idx=0):
      nonlocal rects, texts
      block = Rectangle(height=3.5, width=5.0, color=PURPLE)
      rects.append(block)

      threads = VGroup(make_thread(0), make_thread(1)).arrange(RIGHT).shift(0.8*DOWN)
      block.add(threads)

      shared_mem = Rectangle(width=4.0, height=0.5, color=YELLOW).next_to(threads, UP)
      rects.append(shared_mem)
      block.add(shared_mem)

      texts.append(Text(f"Shared Memory", font_size=15, color=YELLOW).move_to(shared_mem.get_center()))
      shared_mem.add(texts[-1])
      for t in thread_objs[idx*2:]:
        block.add(join(t, shared_mem, t.get_corner(UP)))
      texts.append(Text(f"Block {idx}", color=PURPLE).next_to(shared_mem, UP))
      shared_mem.add(texts[-1])
      
      return block

    blocks = VGroup(make_block(0), make_block(1)).arrange(RIGHT).shift(UP)

    constant = Rectangle(width=blocks.width, height=1, color=YELLOW).next_to(blocks, DOWN)
    texts.append(Text("Constant Memory", font_size=30, color=YELLOW).move_to(constant.get_center()))
    rects.append(constant)

    gmem = Rectangle(width=blocks.width, height=1, color=RED).next_to(constant, DOWN)
    rects.append(gmem)
    texts.append(Text("Global Memory", font_size=30, color=RED).move_to(gmem.get_center()))

    subobjects = []
    queue = [blocks]
    while queue:
      o = queue.pop()
      subobjects.append(o)
      queue.extend(o.submobjects)


    for mo in subobjects:
      for so in mo.submobjects.copy():
        if any(so in x for x in [rects, texts, arrows, thread_objs]):
          mo.remove(so)

    for t in thread_objs[:2]:
      join(t, constant, t.get_corner(DOWN+LEFT)+RIGHT*0.2, False)
      join(t, gmem, t.get_corner(DOWN+LEFT))

    for t in thread_objs[2:]:
      join(t, constant, t.get_corner(DOWN+RIGHT)+LEFT*0.2, False)
      join(t, gmem, t.get_corner(DOWN+RIGHT))

    print(arrows)
    for i in [1, 3, 7, 9]:
      local_store.append(ShowPassingFlash(Arrow(start=arrows[i].get_end(), end=arrows[i].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      local_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
    for i in [0, 2, 6, 8]:
      register_store.append(ShowPassingFlash(Arrow(start=arrows[i].get_end(), end=arrows[i].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      register_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
    for i in [4, 5, 10, 11]:
      shared_store.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      shared_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_end(), end=arrows[i].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
    for i in [13, 15, 17, 19]:
      global_store.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
      global_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_end(), end=arrows[i].get_start(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))
    for i in [12, 14, 16, 18]:
      constant_load.append(ShowPassingFlash(Arrow(start=arrows[i].get_start(), end=arrows[i].get_end(), color=BLUE, buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1).set_z_index(1), time_width=1))

    access_anims = [shared_store, shared_load, register_store, register_load, local_store, local_load, global_store, global_load, constant_load]

    with self.voiceover(text="""Afterwards we will study memory architecture in gpus. This will be one of the most important
                        part to understand""") as trk:
      self.play(Uncreate(axes), Unwrite(x_label), Unwrite(y_label))
      self.play(*[Create(r) for r in rects])
      self.wait(1)
      self.play(*[Write(t) for t in texts])
      self.wait(1)
      self.play(*[Create(a) for a in arrows])
      while trk.get_remaining_duration() > 0:
        self.play(*access_anims[random.randint(0, len(access_anims) -1)])

    with self.voiceover(text="""And with our new knowledge we will dive deeper into more advanced memory level optimizations that
                        our neural network example won't be able to cover because of it's low memory requirements""") as trk:
      while trk.get_remaining_duration() > 1:
        self.play(*access_anims[random.randint(0, len(access_anims) -1)])
      anims = [] 
      for obj in self.mobjects:
        anims.append(FadeOut(obj))
      self.play(*anims)
    self.wait(1)


    with self.voiceover(text="""Of course, in order to optimize, you need to know where your bottlenecks are, so a few tools
                        for profiling will naturally get discussed""") as trk:
      pass
    self.wait(1)

    with self.voiceover(text="""That should mark the end of the fundamental part - after that, we'll see where this series goes""") as trk:
      pass
    self.wait(1)

    with self.voiceover(text="""Remember, that real learning happens when you actually do the work yourself, make errors and struggle with fixing them
                        so I highly encourage you to code along and experiment in your own environment""") as trk:
      pass
    self.wait(1)

    with self.voiceover(text="""And that will be your first excercise - set up your workplace. I'll leave some links in the description on how
                        to install and run CUDA on your device.""") as trk:
      pass
    self.wait(1)

    with self.voiceover(text="""Also, for those that don't have a CUDA capable GPU, there is a wonderfull lecture
                        from Jeremy Howard where he shows how to run CUDA code in notebooks such as google Colab where you can get some free compute""") as trk:
      pass
    self.wait(1)

    with self.voiceover(text="""I'll leave the link to the notebook and the video and I highly recommend that you check it out
                        as well as all other lectures from the CUDA Mode group - they are truly wonderful resources for both beginners and advanced users""") as trk:
      pass
    self.wait(1)

    ppmp = ImageMobject("./PPMP.jpg")
    with self.voiceover(text="""For those that want to dive deeper into GPU programming, I highly recomend a book \"programming massively parallel processors\", 
                        It has an amazing overview of GPU's and is one of the best if not the best resource on the topic""") as trk:
      self.play(FadeIn(ppmp))
    self.play(FadeOut(ppmp))
    self.wait(1)

    tbob = ImageMobject("./3b1b.png")
    with self.voiceover(text="""In this series we will also build a simple neural network, I will explain it as we go but explaining neural networks
                        in depth is beyond the scope of this series, so if you want a refresher on how they work I recommend watching episodes 1-4 from 
                        3 Blue 1 Brown on the subjects - again link in the description""") as trk:
      self.play(FadeIn(tbob))
    self.wait(1)
    self.play(FadeOut(tbob))

    gh = ImageMobject("./gh_logo.png", scale_to_resolution=1080)
    with self.voiceover(text="""The code for all of the examples as well as the animations will be 
                        available on github. So go check it out.""") as trk:
      self.play(FadeIn(gh))
    self.wait(1)
    self.play(FadeOut(gh))
    self.wait(3)
