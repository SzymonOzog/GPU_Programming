import math
from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
from manim.mobject.text.text_mobject import remove_invisible_chars

class GPUvsCPU(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(
        # GTTSService()
        RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU programming", font_size=72).shift(2*UP)
    with self.voiceover(text="Hello and welcome to episode 1 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("GPU vs CPU", font_size=48).next_to(title, DOWN)
    desc = BulletedList("Architectural Differences", "When is it beneficial to use a GPU", "How to write code for the GPU", font_size=48).next_to(subtitle, DOWN).align_to(title, LEFT)

    with self.voiceover(text="In this episode we are going to discuss the key differences between the gpu and the cpu") as trk:
      self.play(Write(subtitle))

    with self.voiceover(text="How the architecture of the two differs") as trk:
      self.play(Write(desc[0]))

    with self.voiceover(text="When to use one over the other") as trk:
      self.play(Write(desc[1]))

    with self.voiceover(text="And finally, we are going to crack open the editor and write some code") as trk:
      self.play(Write(desc[2]))

    self.play(*[Unwrite(desc[2-i]) for i in range(3)],Unwrite(subtitle), Unwrite(title))

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
    cpu_texts.append(cpu_title)
    gpu_title = Text("GPU").scale(0.8).next_to(gpu, UP)
    gpu_texts.append(gpu_title)

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

    with self.voiceover(text="What you are seeing right now on the screen is a highly simplified comparison of a cpu and gpu architectures") as trk:
      self.play(*[Create(x) for x in cpu_rects])
      self.play(*[Write(x) for x in cpu_texts])
      self.wait(1)

      self.play(*[Create(x) for x in gpu_rects])
      self.play(*[Write(x) for x in gpu_texts])

    with self.voiceover(text="""The real architecture is obvoiusly much more complicated, but this simplification will 
                        help us understand the key differences """) as trk:
      pass

    with self.voiceover(text="""First of all, we can see that the GPU consists of a much greated number of cores
                        but that comes at a cost, as the CPU cores are much more capable""") as trk:
      self.wait(1)
      self.play(*[Indicate(x, run_time=3) for x in cpu_rects + gpu_rects if str(x.color).upper() == BLUE])

    with self.voiceover(text="""Secondly, the CPU has a much deeper memory hierarchy, allowing for much lower memory access latency""") as trk:
      self.wait(1)
      self.play(*[Indicate(x, run_time=3) for x in cpu_rects if str(x.color).upper() == RED])

    with self.voiceover(text="""Also, we can see that there are thread groups in the GPU that share the control units,
                        that can point us to the fact that they all must execute exatly the same instruction at the same time""") as trk:
      self.wait(2)
      self.play(*[Indicate(x, run_time=3) for x in gpu_rects if str(x.color).upper() == PURPLE])

    
    with self.voiceover(text="""Let's put this comparison to actual numbers and compare my CPU which is a 
                        AMD Ryzen 7 3800X and my GPU - GTX 3090Ti""") as trk:
      self.wait(3)
      self.play(*[Unwrite(x) for x in cpu_texts if x is not cpu_title], *[Uncreate(x) for x in cpu_rects], *[Unwrite(x) for x in gpu_texts if x is not gpu_title],*[Uncreate(x) for x in gpu_rects])

    font_size = 32
    cpu_details = BulletedList("8 Cores",  "32 MB L3 Cache", "4MB L2 Cache", "64 KB/Core L1 Cache", font_size=font_size).next_to(cpu_title, DOWN)
    gpu_details = BulletedList("10752 Cores", "No L3 Cache", "6MB L2 Cache", "128 KB/SM L1 Cache", font_size=font_size).next_to(gpu_title, DOWN)


    with self.voiceover(text="""My cpu has just 8 cores - compared to 10752 cores in my GPU""") as trk:
      self.play(Write(cpu_details[0]), Write(gpu_details[0]))
    with self.voiceover(text="""It also has 32 MB of L3 cache that is not present in my GPU""") as trk:
      self.play(Write(cpu_details[1]), Write(gpu_details[1]))
    with self.voiceover(text="""It has 4 MB of L2 cache whereas my GPU has 6MB""") as trk:
      self.play(Write(cpu_details[2]), Write(gpu_details[2]))

    with self.voiceover(text="""But to be fair, each core has it's own L2 Cache giving us 512 KB per core""") as trk:
      font_size = 20
      part = Text("512KB/Core L2 Cache", font_size=font_size)
      dot = MathTex("\\cdot").scale(2)
      dot.next_to(part[0], LEFT, SMALL_BUFF)
      part.add_to_back(dot)
      part.move_to(cpu_details[2], aligned_edge=LEFT)
      self.play(Transform(cpu_details[2], part))
    with self.voiceover(text="""The GPU L2 Cache is more simillar in the structure to the L3 in the CPU because it's shared between 
                        the cores""") as trk:
      pass
    with self.voiceover(text="""But if we were to compare it to the CPU L2 it would give around 558B per core""") as trk:
      part = Text("558B/Core L2 Cache", font_size=font_size)
      dot = MathTex("\\cdot").scale(2)
      dot.next_to(part[0], LEFT, SMALL_BUFF)
      part.add_to_back(dot)
      part.move_to(gpu_details[2], aligned_edge=LEFT)
      self.play(Transform(gpu_details[2], part))


    with self.voiceover(text="""And finally the CPU has 64KB of L1 Cache per Core and the GPU has 128 KB of L1 Cache per SM""") as trk:
      self.play(Write(cpu_details[3]), Write(gpu_details[3]))

    gpu_rects = []
    cc = VGroup(Rectangle(width=0.5, height=0.2, color=PURPLE, fill_opacity=0.5), 
                Rectangle(width=0.5, height=0.2, color=RED, fill_opacity=0.5)).arrange(DOWN, buff=0.1)
    gpu_rects.append(cc[0])
    gpu_rects.append(cc[1])
    alus = [Rectangle(width=0.5, height=0.5, color=BLUE, fill_opacity=0.5) for _ in range(8)]
    gpu_rects.extend(alus)
    sm = VGroup(cc, *alus).arrange(RIGHT, buff = 0.1).set_z_index(1).shift(3*UP) 
    with self.voiceover(text="""You might wonder what this SM thing is - we will discuss the detais of it later, but for now
                        think of it as one row of connected cores in our diagram""") as trk:
      self.wait(5)
      self.play(*[Create(x) for x in gpu_rects])

    with self.voiceover(text="""In my gpu a SM consists of 128 cores so that gives us 1KB of L1 Cache per core""") as trk:
      self.play(*[Uncreate(x) for x in gpu_rects])
      part = Text("1KB/Core L1 Cache", font_size=font_size)
      dot = MathTex("\\cdot").scale(2)
      dot.next_to(part[0], LEFT, SMALL_BUFF)
      part.add_to_back(dot)
      part.move_to(gpu_details[3], aligned_edge=LEFT)
      self.play(Transform(gpu_details[3], part))

    self.wait(1)

    with self.voiceover(text="""And both architectures have their advantages, and disadvantages - they are just ment to solve completly different problems""") as trk:
      for i in range(4):
        self.play(Unwrite(cpu_details[3-i], run_time=trk.duration/4), Unwrite(gpu_details[3-i], run_time=trk.duration/4))

        
    with self.voiceover(text="""I like to give an analogy that the CPU is like having 8 Albert Einsteins and the GPU is like having 10752 average high school students""") as trk:
      pass

    font_size = 36 
    cpu_details = BulletedList("Low latency",  "Low throughput", "Optimized for serial operations", font_size=font_size).next_to(cpu_title, DOWN)
    gpu_details = BulletedList("High latency", "High throughput", "Optimized for parallel operations", font_size=font_size).next_to(gpu_title, DOWN)

    with self.voiceover(text="""The cpu is optimized for running code that runs sequentially, minimizing the latency of the operations
                        where as the gpu is optimized for code that runs in parallel - maximizing the throughput""") as trk:
      for i in range(3):
        self.play(Write(cpu_details[i]))
      self.wait(3)
      for i in range(3):
        self.play(Write(gpu_details[i]))

    self.play(*[Unwrite(cpu_details[i]) for i in range(3)], *[Unwrite(gpu_details[i]) for i in range(3)], Unwrite(cpu_title), Unwrite(gpu_title))

    n = 6
    v2 = Matrix([*[[f"b_{i}"] for i in range(n-2)], ["\\vdots"], ["b_n"]], element_alignment_corner=ORIGIN).shift(DOWN)
    plus = Tex("+").next_to(v2, LEFT)
    v1 = Matrix([*[[f"a_{i}"] for i in range(n-2)], ["\\vdots"], ["a_n"]], element_alignment_corner=ORIGIN).next_to(plus, LEFT)
    eq = Tex("=").next_to(v2, RIGHT)
    v3 = Matrix([*[["?"] for i in range(n-2)], ["\\vdots"], ["?"]], element_alignment_corner=ORIGIN).next_to(eq, RIGHT)

    with self.voiceover(text="""One such example of a highly parralelizable algorithm is vector addition""") as trk:
      self.play(*[Create(x) for x in [v1, v2, v3, plus, eq]])

    def create_m(m):
      ret = []
      for i in range(n):
        if i == n - 2:
          ret.append(["\\vdots"])
        elif m == n - 2:
          ret.append(["?"])
        elif i <= m:
          if i == n-1:
            ret.append([f"a_n + b_n"])
          else:
            ret.append([f"a_{i} + b_{i}"])
        else:
          ret.append(["?"])
      return ret
    self.wait(1)
    cpu_code = """void add(int n , float* a, float* b, float* c)
{
  for (int i = 0; i<n; i++)
  {
    c[i] = a[i] + b[i];
  }
}"""
    cpu_code_obj = Code(code=cpu_code, tab_width=2, language="c", font_size=14, background="rectangle", line_no_buff=0.1, corner_radius=0.1).next_to(v2, UP)
    with self.voiceover(text="""The way that you would typically do this on a CPU is to iterate over all of the values in our input vecor,
                        add them together and store them in the third vector""") as trk:
      self.play(Create(cpu_code_obj))
      for i in range(n):
        if i == n-2: continue
        self.play(FadeOut(v1.get_entries()[i].copy(), target_position=v3.get_entries()[i]), 
                  FadeOut(v2.get_entries()[i].copy(), target_position=v3.get_entries()[i]), 
                  Transform(v3, Matrix(create_m(i), element_alignment_corner=ORIGIN).next_to(eq, RIGHT)))

    with self.voiceover(text="""But we can clearly see that the output does not depend on any of the previous iterations""") as trk:
      self.play(Transform(v3, Matrix(create_m(n-2), element_alignment_corner=ORIGIN).next_to(eq, RIGHT)))

    with self.voiceover(text="""So what we want to do, is just run everything in parallel""") as trk:
      self.wait(2)
      self.play(*[FadeOut(v1.get_entries()[i].copy(), target_position=v3.get_entries()[i]) for i in range(n) if i != n-2], 
                *[FadeOut(v2.get_entries()[i].copy(), target_position=v3.get_entries()[i]) for i in range(n) if i != n-2],
                Transform(v3, Matrix(create_m(n-1), element_alignment_corner=ORIGIN).next_to(eq, RIGHT)))

    gpu_code = """__global__ void add(int n , float* a, float* b, float* c)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}"""
    gpu_code_obj = Code(code=gpu_code, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).next_to(v2, UP)
    gpu_code_obj.code = remove_invisible_chars(gpu_code_obj.code)
    with self.voiceover(text="""Here is the GPU code that let's us achieve this""") as trk:
      self.play(Transform(cpu_code_obj, gpu_code_obj))
    self.wait(1)
    with self.voiceover(text="""First difference that we can notice is the global keyword - this is just an information for the
                        compiler that this function is to be run on the GPU""") as trk:
      sw = SurroundingRectangle(gpu_code_obj.code[0][0:10], buff=0.03, stroke_width=2, fill_opacity=0.3)
      self.play(Create(sw))
    self.wait(1)
    with self.voiceover(text="""Next difference is the index calculation, instead of a for loop. Since the code will run in parallel there will be multiple 
                        instances of it created, and every one needs to be able to calculate which element in our vectors it is supposed to work on""") as trk:
      self.play(Transform(sw, SurroundingRectangle(gpu_code_obj.code[2], buff=0.03, stroke_width=2, fill_opacity=0.3)))

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
    with self.voiceover(text="""The way CUDA achieves this is by using a concept of thread blocks. There will be a whole episode directed towards understanding them
                        but for now you need to understand a few basic concepts""") as trk:
      pass
    self.wait(0.5)

    with self.voiceover(text="""When we launch our Kernel, we need to specify how many blocks we will launch, and how many threads will each block contain""") as trk:
      pass

    with self.voiceover(text="""For exaple, when we launch n/2 blocks, each containing 2 threads, here's how the structure of our blocks will look like""") as trk:
      self.play(Create(block1), *[Write(t) for t in t1])
      self.play(Create(block2), *[Write(t) for t in t2])
      self.play(Create(block3), *[Write(t) for t in t3])
    with self.voiceover(text="""Each thread runs it's own version of the code with the appropriate block index and thread index. 
                        Block dimensions are shared between the threads""") as trk:
      pass
    with self.voiceover(text="""It's left as an excercise to the watcher to verivy that each thread will calculate a different index""") as trk:
      pass
    self.wait(1)

    with self.voiceover(text="""The last difference is checking if our index is inside our vector""") as trk:
      self.play(Transform(sw, SurroundingRectangle(gpu_code_obj.code[3], buff=0.03, stroke_width=2, fill_opacity=0.3)))

    with self.voiceover(text="""For example, if we had an uneaven number of elements, the alignment of blocks might look like this,
                        where the last thread in the last block would read and write out of bouds, the check ensures that we don't do that""") as trk:
      self.wait(2)
      self.play(*[x.animate.shift(0.7*DOWN) for x in [block3]+t3])
      self.play(t3[1].animate.set_color(RED))

    self.wait(1)

    with self.voiceover(text="""Now that we have our kernel written, we actually need to run it""") as trk:
      anims = [] 
      for obj in self.mobjects:
        anims.append(FadeOut(obj))
      self.play(*anims)
      
    steps = BulletedList("Allocate the memory on the GPU",
                         "Copy the data to the GPU",
                         "Run the kernel",
                         "Copy the data back to the CPU",
                         "Free the memory",
                         font_size=24).shift(3*LEFT)

    with self.voiceover(text="""To do that, we need to first allocate the memory for our inputs, and outputs on the GPU""") as trk:
      self.play(Create(steps[0]))

    with self.voiceover(text="""And since our GPU cannot access the data in our RAM or on our Hard Drive - we need to copy our data from the 
                        CPU to the GPU""") as trk:
      self.play(Create(steps[1]))

    with self.voiceover(text="""We can then run our kernel""") as trk:
      self.play(Create(steps[2]))

    with self.voiceover(text="""Copy the results back to the CPU to be able to read them""") as trk:
      self.play(Create(steps[3]))

    with self.voiceover(text="""And free the memory on our GPU""") as trk:
      self.play(Create(steps[4]))

    c0 = """float* a_d;
float* b_d;
float* c_d;
int N = 4096;
cudaMalloc((void**) &a_d, N*sizeof(float));
cudaMalloc((void**) &b_d, N*sizeof(float));
cudaMalloc((void**) &c_d, N*sizeof(float)); """

    c1 = """cudaMemcpy(a_d, a, N*sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(b_d, b, N*sizeof(float), cudaMemcpyHostToDevice); """

    c2 = """int BLOCK_SIZE=2048;
add<<<ceil(N/(float)BLOCK_SIZE), BLOCK_SIZE>>>(N, a_d, b_d, c_d); """

    c3 = "cudaMemcpy(c, c_d, N*sizeof(float), cudaMemcpyDeviceToHost);"

    c4 = """cudaFree(a_d);
cudaFree(b_d);
cudaFree(c_d);"""

    device = Text("Device = GPU", font_size=48).move_to(cpu_title).shift(0.5*UP)
    host = Text("Host = CPU", font_size=48).move_to(gpu_title).shift(0.5*UP)
    with self.voiceover(text="""Just a quick jargon checkup before showing you the code - CUDA refers to GPU as the Device,
                        and the CPU as the host""") as trk:
      self.play(Write(device))
      self.play(Write(host))
    
    code_obj = Code(code=c0, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).next_to(steps,RIGHT)
    with self.voiceover(text="""In order to allocate our memory, we need to first create a pointer, just like in regular C,
                        the _d suffix is a common naming convention for marking pointers that are living on the device - so in our case the GPU
                        and then, again simillarly to C we call the cudaMalloc function, passing in our pointer, and the size
                        that we want to allocate""") as trk:
      self.play(steps[0].animate.set_color(YELLOW))
      self.play(Transform(steps[0].copy(), code_obj, replace_mobject_with_target_in_scene=True))
    self.play(FadeOut(code_obj))

    code_obj = Code(code=c1, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).next_to(steps,RIGHT)
    with self.voiceover(text="""Then we have to move the data from the cpu to the gpu, assuming that we already have our CPU pointers allocated,
                        we just have to call the cudaMemcpy function. As arguments we pass in the source pointer, the destination poiner,
                        the size of the memory we want to copy, and a marker indicating the type of the copy. In our case we are copying from Host memory
                        to Device memory, so from the CPU to the GPU""") as trk:
      self.play(steps[0].animate.set_color(GREEN))
      self.play(steps[1].animate.set_color(YELLOW))
      self.play(Transform(steps[1].copy(), code_obj, replace_mobject_with_target_in_scene=True))
    self.play(FadeOut(code_obj))


    code_obj = Code(code=c2, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).next_to(steps,RIGHT)
    with self.voiceover(text="""In order to run our kernel, we need to specify how many blocks we want to run, and how many threads are there in each block.
                        CUDA lets us configure this with this strange looking triple angle brackets.""") as trk:
      self.play(steps[1].animate.set_color(GREEN))
      self.play(steps[2].animate.set_color(YELLOW))
      self.play(Transform(steps[2].copy(), code_obj, replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""To cover the full span of our vector, we need to launch the minimum of N threads, so if each block consist of BLOCK_SIZE threads.
                        we have to run the ceiling division of N divided by block size """) as trk:
      pass
    self.play(FadeOut(code_obj))

    code_obj = Code(code=c3, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).next_to(steps,RIGHT)
    with self.voiceover(text="""Finally we copy the result back to the CPU using the same function as before, but this time the destination pointer is,
                        the cpu pointer, the source pointer is the GPU pointer and the direction is from Device to Host""") as trk:
      self.play(steps[2].animate.set_color(GREEN))
      self.play(steps[3].animate.set_color(YELLOW))
      self.play(Transform(steps[3].copy(), code_obj, replace_mobject_with_target_in_scene=True))
    self.play(FadeOut(code_obj))

    code_obj = Code(code=c4, tab_width=2, language="c", font_size=14, line_no_buff=0.1, corner_radius=0.1).next_to(steps,RIGHT)
    with self.voiceover(text="""What is left is to clean up our memory - we do that by calling the cuda free funciton""") as trk:
      self.play(steps[3].animate.set_color(GREEN))
      self.play(steps[4].animate.set_color(YELLOW))
      self.play(Transform(steps[4].copy(), code_obj, replace_mobject_with_target_in_scene=True))
    self.play(steps[4].animate.set_color(GREEN))
    self.wait(1)

    with self.voiceover(text="""Now let's check what are the actual time differences when running our vector addition code on a GPU vs the CPU""") as trk:
      pass

    anims = [] 
    for obj in self.mobjects:
      anims.append(FadeOut(obj))
    self.play(*anims)

    ps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    cpu_t = [8610,  9760,  11060,  16320,  27310,  55421,  95662,  181362,  357233,  706197,  1.40494e+06,  2.84067e+06,  5.66813e+06,  1.13283e+07,  2.27314e+07,  4.65509e+07,  1.38753e+08,  3.43102e+08,  4.8381e+08,  1.00493e+09,  1.76705e+09,  3.9403e+09,  8.13902e+09,  1.7344e+10]
    gpu_t = [2.28495e+06,2.30132e+06,2.26564e+06,2.27229e+06,2.26786e+06,2.26626e+06,2.28136e+06,2.29248e+06,2.29704e+06,2.33805e+06,2.33565e+06,2.87786e+06,2.3503e+06,2.34703e+06,2.36478e+06,2.47582e+06,2.72161e+06,3.02623e+06,4.00075e+06,8.21823e+06,1.37786e+07,3.02901e+07,4.73732e+07,9.24e+07]

    n = [2**p for p in ps] 
    log_cpu_t = [math.log10(t) for t in cpu_t]
    log_gpu_t = [math.log10(t) for t in gpu_t]

    ax = Axes(
        x_range=[ps[0] , ps[-1], 2],
        y_range=[log_cpu_t[0] , log_cpu_t[-1], 1],
        x_axis_config={"scaling": LogBase(2)},
        y_axis_config={"scaling": LogBase()},
        axis_config={"include_numbers": True})

    labels = ax.get_axis_labels(x_label="n", y_label="Time[ns]")

    cpu_graph = ax.plot_line_graph(
        x_values=n,
        y_values=cpu_t,
        line_color=BLUE,
        add_vertex_dots=False
    )

    gpu_graph = ax.plot_line_graph(
        x_values=n,
        y_values=gpu_t,
        line_color=GREEN,
        add_vertex_dots=False
    )
    gpu_label = Text("GPU times", font_size=32, color=GREEN).next_to(labels[1], DOWN).shift(0.2*RIGHT+0.2*DOWN)
    cpu_label = Text("CPU times", font_size=32, color=BLUE).next_to(gpu_label, DOWN)

    with self.voiceover(text="""Here is a graph of the results for different input sizes""") as trk:
      self.play(Create(ax), Write(labels))
      self.play(Create(cpu_graph), Create(gpu_graph))
      self.play(Write(gpu_label), Write(cpu_label))

    with self.voiceover(text="""It's worth noting that the scale is logarithmic so the ratios are much higher than they seem at first glance""") as trk:
      pass


    ratio = [c/g for c,g in zip(cpu_t, gpu_t)]
    ax2 = Axes(
        x_range=[ps[0] , ps[-1], 2],
        y_range=[ratio[0] , ratio[-1], 20],
        x_axis_config={"scaling": LogBase(2)},
        y_axis_config={},
        axis_config={"include_numbers": True})

    labels2 = ax.get_axis_labels(x_label="n", y_label="\\frac{CPU}{GPU}")
    
    ratio_graph = ax2.plot_line_graph(
        x_values=n,
        y_values=ratio,
        line_color=BLUE,
        add_vertex_dots=False
    )
    with self.voiceover(text="""If we were to graph the ratios, it turns out that our GPU code runs up to 180 times faster for big input sizes""") as trk:
      self.play(Transform(ax, ax2), Transform(labels, labels2), Transform(VGroup(gpu_label, cpu_label, gpu_graph, cpu_graph), ratio_graph))
    self.wait(1)

    with self.voiceover(text="""And to be real with you, this kernel is just the simplest one to present.
                        Since it does 3 memory accesses per one operation - it's not actually that much faster. 
                        We'll get into some kernels that get crazy improvements on the GPU - and if you want to see those - subscribe and stay tuned for the next episode""") as trk:
      pass
    self.wait(1)
    anims = [] 
    for obj in self.mobjects:
      anims.append(FadeOut(obj))
    self.play(*anims)
    self.wait(3)
