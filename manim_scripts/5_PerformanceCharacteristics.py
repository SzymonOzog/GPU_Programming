from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
from NN import NeuralNetworkMobject

mnist = np.loadtxt("mnist_train.csv", delimiter=",")

class PerformanceCharacteristics(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU Programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 5 in the series on GPU programming") as trk:
      self.play(Write(title))


    subtitle = Text("Performance Characteristics", font_size=48).next_to(title, DOWN)
    with self.voiceover(text="""In this episode we are going to discuss some performance characteristics, and main factors
                        that influence the performance of the GPU, we'll also introduce the roofline model for assessing
                        our code's performance with regards to the hardware possibilities""") as trk:
      self.play(Write(subtitle))

    with self.voiceover(text="""When we run our code, there are 3 main areas in play""") as trk:
      while trk.get_remaining_duration() > 1:
        self.wait(0.1)
      self.play(Unwrite(title), Unwrite(subtitle))

    

    gpu = Rectangle(height=2, width=3, color=GREEN, fill_color=GREEN, fill_opacity=0.5).shift(2*LEFT+UP)
    gpu_t = Text("GPU", color=GREEN).move_to(gpu)
    with self.voiceover(text="""The firs one is obviously our gpu""") as trk:
      self.play(Create(gpu))
      self.play(Write(gpu_t))

    memory = Rectangle(height=2, width=3, color=RED, fill_color=RED, fill_opacity=0.5)
    memory.next_to(gpu, DOWN, aligned_edge=LEFT, buff=0).shift(DOWN)
    mem_t = Text("HBM", color=RED).move_to(memory)
    m_to_g = DoubleArrow(gpu.get_corner(DOWN), memory.get_corner(UP), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
    with self.voiceover(text="""It is directly connected to it's High Bandwidth Memory, so our GPU's VRAM""") as trk:
      self.play(Create(memory), Create(m_to_g))
      self.play(Write(mem_t))

    cpu = Rectangle(height=5, width=3, color=BLUE, fill_color=BLUE, fill_opacity=0.5).next_to(gpu, RIGHT, aligned_edge=UP).shift(RIGHT)
    cpu_t = Text("CPU", color=BLUE).move_to(cpu)

    l1 = cpu.get_corner(LEFT)
    l1[1] = memory.get_corner(RIGHT)[1]
    c_to_m = Arrow(l1, memory.get_corner(RIGHT), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
    l1 = cpu.get_corner(LEFT)
    l1[1] = gpu.get_corner(RIGHT)[1]
    c_to_g = Arrow(l1, gpu.get_corner(RIGHT), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)

    with self.voiceover(text="""And there is also the cpu, that can schedule kernels on our gpu as well as copy data to our HBM""") as trk:
      self.play(Create(cpu), Create(c_to_m), Create(c_to_g))
      self.play(Write(cpu_t))

    with self.voiceover(text="""And performance can be a problem in all 3 of those""") as trk:
      pass
    overhead = SurroundingRectangle(cpu, color=BLUE_E).scale(1.1)
    overhead_t = Text("Overhead", color=BLUE_E, font_size=24).next_to(overhead, UP)
    with self.voiceover(text="""The time spent on the CPU is called overhead""") as trk:
      self.play(Create(overhead))
      self.play(Write(overhead_t))
    
    mem_bound = SurroundingRectangle(memory, color=RED_E).scale(1.1)
    mem_bound_t = Text("Memory Access Latency", color=RED_E, font_size=24).next_to(mem_bound, DOWN)

    with self.voiceover(text="""The time that we spend loading data from memory is our memory access latency""") as trk:
      self.play(Create(mem_bound))
      self.play(Write(mem_bound_t))

    comp_bound = SurroundingRectangle(gpu, color=GREEN_E).scale(1.1)
    comp_bound_t = Text("Compute Time", color=GREEN_E, font_size=24).next_to(comp_bound, UP)
    with self.voiceover(text="""And finally, the time on the gpu is our actuall compute time""") as trk:
      self.play(Create(comp_bound))
      self.play(Write(comp_bound_t))

    w_scale = 1 
    overhead_time = 2
    training_time = 9.5
    overhead_r = Rectangle(height=0.5, width=overhead_time/w_scale, color=BLUE, fill_color=BLUE, fill_opacity=0.5)
    training_r = Rectangle(height=0.5, width=training_time/w_scale, color=GREEN, fill_color=GREEN, fill_opacity=0.5)
    throughput = VGroup(overhead_r, training_r).arrange(RIGHT,buff=0.02)
    with self.voiceover(text="""Let's first look into our overhead, in our case it is the time that we spend loading the dataset from disk""") as trk:
      self.play(Transform(VGroup(cpu, cpu_t, overhead, overhead_t, c_to_m, c_to_g), overhead_r, replace_mobject_with_target_in_scene=True),
                Transform(VGroup(gpu, gpu_t, memory, mem_t, mem_bound, mem_bound_t, comp_bound, comp_bound_t, m_to_g), training_r, replace_mobject_with_target_in_scene=True))
    fs = 20
    overhead_t = Text("2 s", font_size=fs, color=BLUE).move_to(overhead_r)
    training_t = Text("9.5 s", font_size=fs, color=GREEN).move_to(training_r)
    with self.voiceover(text="""And it takes around 2 seconds to do that, while our 10 epochs of training <bookmark mark='1'/>take around 9.5 seconds in total""") as trk:
      self.play(Write(overhead_t))
      self.wait_until_bookmark("1")
      self.play(Write(training_t))

    trans = Rectangle(height=0.5, width=1/w_scale, color=BLUE, fill_color=BLUE, fill_opacity=0.5).move_to(overhead_r, aligned_edge=RIGHT)
    with self.voiceover(text="""The first obvious thing that we can do is to optimize our cpu code""") as trk:
      pass
    self.wait(1)

    with self.voiceover(text="""I managed to get it down to 1 second, by using some more optimized string parsing functions""") as trk:
      self.play(Transform(overhead_r, trans), Transform(overhead_t, Text("1 s", font_size=fs, color=BLUE).move_to(trans)))
    self.wait(1)

    epochs = VGroup(*[Rectangle(height=0.5, width=training_time/(10*w_scale), color=GREEN, fill_color=GREEN, fill_opacity=0.5) for i in range(10)]).arrange(RIGHT, buff=0.02).move_to(training_r, aligned_edge=LEFT)
    epoch_times = VGroup(*[Text("0.95 s", font_size=fs, color=GREEN).move_to(epochs[i]) for i in range(10)])

    with self.voiceover(text="""But we want to mitigate our overhead even further, and one thing that you can notice is that we do not need the full
                        dataset at the start of the first epoch, we only need the data that we will be working on""") as trk:
      self.play(Transform(training_r, epochs, replace_mobject_with_target_in_scene=True),
                Transform(training_t, epoch_times, replace_mobject_with_target_in_scene=True))
    self.wait(1)
    overhead_r.add(overhead_t)
    with self.voiceover(text="""So we can execute the cpu and gpu code in parallel, where the gpu already starts training our network on the first batch
                        as the cpu loads the next one in the background""") as trk:
      self.play(overhead_r.animate.next_to(epochs[0], DOWN, aligned_edge=RIGHT, buff=0.02))
      group = VGroup(overhead_r, epochs[0], epoch_times[0])
      self.play(group.animate.next_to(epochs[1], LEFT, buff=0.02))
    self.wait(1)
    
    training_time=3
    epochs_2 = VGroup(*[Rectangle(height=0.5, width=training_time/(10*w_scale), color=GREEN, fill_color=GREEN, fill_opacity=0.5) for i in range(10)]).arrange(RIGHT, buff=0.02).next_to(group, RIGHT, buff=0.02).shift(0.3*LEFT)
    anims = []
    anims.extend([Unwrite(x) for x in epoch_times])
    for i in range(1, 10):
      anims.append(Transform(epochs[i], epochs_2[i]))

    color_grad = color_gradient([BLUE, GREEN], 2)

    with self.voiceover(text="""Now this does not eliminate our overhead, we still need to load all of our data during the first epoch, 
                        so if we optimize our <bookmark mark='1'/>epoch time, our first epoch will still be limited by the time that it takes to load
                        the data""") as trk:
      self.wait_until_bookmark("1")
      self.play(LaggedStart(*anims))
    combined = Rectangle(height=0.5, width=1.2/(w_scale), fill_opacity=0.5).next_to(epochs_2, LEFT, buff=0.02).set_color(color_grad).shift(0.3*RIGHT)
    with self.voiceover(text="""We didn't actually eliminate the overhead, we just blended it into our gpu code to hide it""") as trk:
      self.play(Transform(VGroup(overhead_r, epochs[0]), combined, replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""And the limiting factor is not going to be exactly equal to our data loading time since the cpu also
                        schedules gpu kernels and copies data into memory""") as trk:
      self.play(Write(Text("1+ ms", font_size=fs).set_color(color_grad).move_to(combined)))
    self.wait(1)
    training_code = """load_full_dataset();
for(int epoch = 0; epoch<EPOCHS; epoch++)
{
  for(int batch = 0; batch<train_length/BATCH_SIZE; batch++)
  {
    forward<<<dimGrid, dimBlock>>>(...);
    relu<<<dimGrid, dimBlock>>>(...);
    forward<<<dimGrid, dimBlock>>>(...);
    relu<<<dimGrid, dimBlock>>>(...);
    forward<<<dimGrid, dimBlock>>>(...);
    softmax<<<dimGrid, dimBlock>>>(...);
    cross_entropy<<<dimGrid, dimBlock>>>(...);

    cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);
    check_accuracy(...);

    cross_entropy_backwards<<<dimGrid, dimBlock>>>(...);
    backward<<<dimGrid, dimBlock>>>(...);
    relu_backwards<<<dimGrid, dimBlock>>>(...);
    backward<<<dimGrid, dimBlock>>>(...);
    relu_backwards<<<dimGrid, dimBlock>>>(...);

    update_layer<<<dimGrid, dimBlock>>>(...);
    update_layer<<<dimGrid, dimBlock>>>(...);
    update_layer<<<dimGrid, dimBlock>>>(...);
  }
}
"""
    code_obj = Code(code=training_code, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj.code = remove_invisible_chars(code_obj.code)

    with self.voiceover(text="""Right now our training code looks somewhat like this, I've omitted some parts for better readablity""") as trk:
      self.play(Transform(VGroup(*[x for x in self.mobjects if isinstance(x, VMobject)]), code_obj, replace_mobject_with_target_in_scene=True))
    self.wait(1)

    hl = SurroundingRectangle(code_obj.code[0], buff=0.03, stroke_width=2, fill_opacity=0.3)

    with self.voiceover(text="""We load the full dataset before stepping int the training loop""") as trk:
      self.play(Create(hl))

    hl_t = SurroundingRectangle(code_obj.code[1:4], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Then in each epoch, we go through our dataset by batches""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[5:12], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""We run our forward pass""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[13], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Copy the results to the cpu""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[14], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Check our accuracy""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[16:21], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""We then run our backward pass""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[22:25], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And update our weights and biases""") as trk:
      self.play(Transform(hl, hl_t))
    self.wait(1)

    with self.voiceover(text="""And we don't actually need to do much to get our dataset to load in parallel""") as trk:
      self.play(Uncreate(hl))

    hl = SurroundingRectangle(code_obj.code[5], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""By design, cuda does not run synchronously,<bookmark mark='1'/> so when we run our kernel""") as trk:
      self.wait_until_bookmark("1")
      self.play(Create(hl))

    hl_t = SurroundingRectangle(code_obj.code[5:12], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""The cpu will not wait for the kernel to finish, <bookmark mark='1'/>and calling other kernels will 
                        add it to the queue which the driver will manage for us""") as trk:
      self.wait_until_bookmark("1")
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[13], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And when we call some synchronous function,<bookmark mark='1'/> in our case cudaMemcpy - it will wait for the kernels
                        to finish executing""") as trk:
      self.wait_until_bookmark("1")
      self.play(Transform(hl, hl_t))

    training_code = """load_full_dataset();
for(int epoch = 0; epoch<EPOCHS; epoch++)
{
  for(int batch = 0; batch<train_length/BATCH_SIZE; batch++)
  {
    forward<<<dimGrid, dimBlock>>>(...);
    relu<<<dimGrid, dimBlock>>>(...);
    forward<<<dimGrid, dimBlock>>>(...);
    relu<<<dimGrid, dimBlock>>>(...);
    forward<<<dimGrid, dimBlock>>>(...);
    softmax<<<dimGrid, dimBlock>>>(...);
    cross_entropy<<<dimGrid, dimBlock>>>(...);

    cross_entropy_backwards<<<dimGrid, dimBlock>>>(...);
    backward<<<dimGrid, dimBlock>>>(...);
    relu_backwards<<<dimGrid, dimBlock>>>(...);
    backward<<<dimGrid, dimBlock>>>(...);
    relu_backwards<<<dimGrid, dimBlock>>>(...);

    update_layer<<<dimGrid, dimBlock>>>(...);
    update_layer<<<dimGrid, dimBlock>>>(...);
    update_layer<<<dimGrid, dimBlock>>>(...);

    cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);
    check_accuracy(...);
  }
}
"""
    code_obj_t = Code(code=training_code, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj_t.code = remove_invisible_chars(code_obj_t.code)

    with self.voiceover(text="""The first thing to do is to schedule all of our kernels before running our synchronous call,<bookmark mark='1'/>
                        that way we can keep the gpu busy while loading our batch""") as trk:
      self.play(Uncreate(hl))
      self.wait_until_bookmark("1")
      self.play(Transform(code_obj, code_obj_t))

    training_code = """load_next_batch();
for(int epoch = 0; epoch<EPOCHS; epoch++)
{
  for(int batch = 0; batch<train_length/BATCH_SIZE; batch++)
  {
    forward<<<dimGrid, dimBlock>>>(...);
    relu<<<dimGrid, dimBlock>>>(...);
    forward<<<dimGrid, dimBlock>>>(...);
    relu<<<dimGrid, dimBlock>>>(...);
    forward<<<dimGrid, dimBlock>>>(...);
    softmax<<<dimGrid, dimBlock>>>(...);
    cross_entropy<<<dimGrid, dimBlock>>>(...);

    cross_entropy_backwards<<<dimGrid, dimBlock>>>(...);
    backward<<<dimGrid, dimBlock>>>(...);
    relu_backwards<<<dimGrid, dimBlock>>>(...);
    backward<<<dimGrid, dimBlock>>>(...);
    relu_backwards<<<dimGrid, dimBlock>>>(...);

    update_layer<<<dimGrid, dimBlock>>>(...);
    update_layer<<<dimGrid, dimBlock>>>(...);
    update_layer<<<dimGrid, dimBlock>>>(...);

    if (epoch == 0)
    {
      load_next_batch();
    }

    cudaMemcpy(out_h, out_d, out_size, cudaMemcpyDeviceToHost);
    check_accuracy(...);
  }
}
"""
    code_obj_t = Code(code=training_code, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj_t.code = remove_invisible_chars(code_obj_t.code)

    with self.voiceover(text="""And now we can just create a function that loads one batch at a time,<bookmark mark='1'/>
                        and place it just before our synchronization point""") as trk:
      self.wait_until_bookmark("1")
      self.play(Transform(code_obj, code_obj_t))
    self.wait(3)

    gpu = Rectangle(height=2, width=3, color=GREEN, fill_color=GREEN, fill_opacity=0.5).shift(2*LEFT+UP)
    gpu_t = Text("GPU", color=GREEN).move_to(gpu)
    memory = Rectangle(height=2, width=3, color=RED, fill_color=RED, fill_opacity=0.5)
    memory.next_to(gpu, DOWN, aligned_edge=LEFT, buff=0).shift(DOWN)
    mem_t = Text("HBM", color=RED).move_to(memory)
    m_to_g = DoubleArrow(gpu.get_corner(DOWN), memory.get_corner(UP), buff=0, stroke_width=4, tip_length=0.12, max_stroke_width_to_length_ratio=90, max_tip_length_to_length_ratio=1)
    with self.voiceover(text="""Let's now get into the real juice which is evaluating and improving the performance of the code that runs on the gpu""") as trk:
      self.play(Transform(code_obj, VGroup(gpu, gpu_t, memory, mem_t, m_to_g), replace_mobject_with_target_in_scene=True))
    
    mem_bound = SurroundingRectangle(memory, color=RED_E).scale(1.1)
    mem_bound_t = Text("Memory Access Latency", color=RED_E, font_size=24).next_to(mem_bound, DOWN)
    comp_bound = SurroundingRectangle(gpu, color=GREEN_E).scale(1.1)
    comp_bound_t = Text("Compute Time", color=GREEN_E, font_size=24).next_to(comp_bound, UP)

    with self.voiceover(text="""And as I'me mentioned in the beginning, we can have 2 kinds of performance issues with our gpu,
                        <bookmark mark='1'/> time spend doing the compute, <bookmark mark='2'/>and time spent accessing memory""") as trk:
      self.wait_until_bookmark("1")
      self.play(Create(comp_bound), Write(comp_bound_t))
      self.wait_until_bookmark("2")
      self.play(Create(mem_bound), Write(mem_bound_t))

    throughput = Text("82.58 TFLOPS", font_size=24, color=GREEN).next_to(comp_bound, RIGHT)
    with self.voiceover(text="""our compute time is bound directly by the throughput of the datatype that we are using on our gpu, for example
                        a 4090 <bookmark mark='1'/>, has 82.58 TFLOPS throughput for 32 bit floating point numbers""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(throughput))

    flops = Text("FLOPS = Floating point operations / s", font_size=24).next_to(throughput, UP, aligned_edge=LEFT)
    with self.voiceover(text="""FLOPS is just how many floating point operations we can do in a second""") as trk:
      self.play(Write(flops))

    bandwidth = Text("1.01 TB/s", font_size=24, color=RED).next_to(mem_bound, RIGHT)
    with self.voiceover(text="""And our memory is bound by bandwidth, which for a 4090 is <bookmark mark='1'/> 1.01 TB/s""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(bandwidth))


    with self.voiceover(text="""As you can see, we can load our data from memory much slower than we can actually do the computation""") as trk:
      pass

    comp_int = Text("Computational Intensity = FLOP/B", color=YELLOW, font_size=24).next_to(m_to_g).align_to(bandwidth, LEFT)
    with self.voiceover(text="""this introduces another very important metric for our kernels which is <bookmark mark='1'/>
                        how many floating point operations we are doing per each byte of memory access""") as trk:
      self.play(Write(comp_int))

    axes = Axes(x_range=[0, 10, 1], y_range=[0, 7, 1], x_length=7, y_length=5, axis_config={"include_tip": False})
    
    x_text = VGroup(VGroup(MathTex("Computational"), MathTex("Intensity")).arrange(DOWN), MathTex("[\\frac{FLOP}{B}]")).arrange(RIGHT)
    y_text = VGroup(VGroup(MathTex("Attainable"), MathTex("Performance")).arrange(DOWN), MathTex("[FLOPS]")).arrange(RIGHT)
    x_label = axes.get_x_axis_label(x_text, edge=UP, direction=DOWN, buff=MED_SMALL_BUFF)
    y_label = axes.get_y_axis_label(y_text.rotate(PI/2), edge=LEFT, direction=LEFT)

    bandwidth_line = Line(start=axes.c2p(0, 0), end=axes.c2p(7, 7), color=RED_E).set_opacity(0.7)
    bandwidth_label = Tex("Bandwitdth limit", color=RED_E, font_size=40).rotate(PI/4).move_to(bandwidth_line.get_center() + LEFT + 0.5*DOWN)

    throughput_line = Line(start=axes.c2p(0, 5), end=axes.c2p(10, 5), color=GREEN_E).set_opacity(0.7)
    throughput_label = Tex("Throughput limit", color=GREEN_E, font_size=40).next_to(throughput_line, UP, buff=0.1)

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

    title = Text("Roofline Model", font_size=60).next_to(axes, UP)
    with self.voiceover(text="""And our computational intensity metric brings us straight into the <bookmark mark='1'/>roofline model,
                        which is a great way of visualising our performance bottleneck""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(title))
      self.play(Transform(VGroup(mem_bound, mem_bound_t, comp_bound, comp_bound_t, gpu, gpu_t, memory, mem_t, m_to_g),
                          axes, replace_mobject_with_target_in_scene=True))
      self.play(Transform(comp_int, x_label, replace_mobject_with_target_in_scene=True),
                Transform(flops, y_label, replace_mobject_with_target_in_scene=True))

    self.wait(1)


    with self.voiceover(text="""We can put our maximal througput and bandwith on a graph to visualise, what is the maximum performance
                        that we can get based on our computational intensity""") as trk:
      self.play(Transform(throughput, throughput_line, replace_mobject_with_target_in_scene=True),
                Transform(bandwidth, bandwidth_line, replace_mobject_with_target_in_scene=True))
      self.play(Write(bandwidth_label), Write(throughput_label))

    with self.voiceover(text="""If we are on the right side of the graph, that means that we are compute bound - the GPU is working at full power
                        and not waiting for anything. And that is usually the place where we want to be""") as trk:
      self.play(Create(compute_bound))

    with self.voiceover(text="""Although there are a few things that we can do, to name a few examles, we can change our datatype to a more performant one that allows more FLOPS""") as trk:
      pass

    with self.voiceover(text="""We can try to improve our algorithms to require less computation""") as trk:
      pass

    with self.voiceover(text="""Or we can use more specialized hardware such as Tensor Cores that allow higher throughput""") as trk:
      pass

    with self.voiceover(text="""Finally, we can make nvidia shareholders happy and just buy more GPU's - if they aren't sold out yet""") as trk:
      pass

    self.wait(1)

    with self.voiceover(text="""And if we are on the left side on this graph we are in a bad spot, our gpu is bored and is not doing any work
                        because it's waiting for the data to arrive from memory""") as trk:
      self.play(Create(bandwidth_bound))

    with self.voiceover(text="""The good news is, that there are a lot of things we can do to get ourselves out of this situation""") as trk:
      pass

    with self.voiceover(text="""We can use differend kinds of memory - but that will be the topic for the next video""") as trk:
      pass

    with self.voiceover(text="""Or we can do something called kernel fusion""") as trk:
      pass


    forward = """__global__ void forward(int batch_size, int n, int out_w,
                        float* input, float* weights, float* biases, float* output)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    output[row*out_w+column] = biases[column];
    for(int i = 0; i < n; i++)
    {
      output[row*out_w+column] += weights[i*out_w + column] * input[row*n + i];
    }
  }
}"""

    relu = """__global__ void relu(int w, int h, float* a, float* b)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float activation = a[row*w+column];
    b[row*w+column] = activation > 0.f ? activation : 0.f;
  }
}"""

    code_obj = Code(code=forward, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj.code = remove_invisible_chars(code_obj.code)

    code_obj2 = Code(code=relu, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj2.code = remove_invisible_chars(code_obj2.code)
    
    with self.voiceover(text="""To show an example of kernel fusion, let's look at our biggest bottleneck - ReLU""") as trk:
      self.play(Transform(VGroup(*[x for x in self.mobjects if isinstance(x, VMobject)]), code_obj2, replace_mobject_with_target_in_scene=True))

    hl = SurroundingRectangle(code_obj2.code[6][16:], buff=0.03, stroke_width=2, fill_opacity=0.3, color=RED)
    hl_t = SurroundingRectangle(code_obj2.code[7][:15], buff=0.03, stroke_width=2, fill_opacity=0.3, color=RED)
    with self.voiceover(text="""We are doing 2 memory accesses, <bookmark mark='1'/>one read, and one <bookmark mark='2'/>write,
                        both of size 4 bytes so that's 8 bytes of memory loaded""") as trk:
      self.wait_until_bookmark("1")
      self.play(Create(hl))
      self.wait_until_bookmark("2")
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj2.code[7][16:30], buff=0.03, stroke_width=2, fill_opacity=0.3, color=GREEN)
    hl_t2 = SurroundingRectangle(code_obj2.code[7][15:], buff=0.03, stroke_width=2, fill_opacity=0.3, color=GREEN)
    with self.voiceover(text="""And we are only doing 2 operations, <bookmark mark='1'/>one comparison, and one <bookmark mark='2'/>assignment""") as trk:
      self.wait_until_bookmark("1")
      self.play(Transform(hl, hl_t))
      self.wait_until_bookmark("2")
      self.play(Transform(hl, hl_t2))

    self.wait(1)
    comp_intensity = MathTex("0.25 \\frac{FLOP}{B}").next_to(code_obj2, DOWN)
    with self.voiceover(text="""This gives us just 0.25 floating point operations per one byte of memory""") as trk:
      self.play(Uncreate(hl))
      self.play(Write(comp_intensity))

    with self.voiceover(text="""So what we can do is to take our forward pass kernel""") as trk:
      self.play(Unwrite(comp_intensity))
      self.play(Create(code_obj2.to_edge(DOWN)))
      self.play(Create(code_obj.to_edge(UP)))

    forward_relu="""__global__ void forward_relu(int batch_size, int n, int out_w,
                             float* input, float* weights, float* biases, float* output)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    float out = biases[column];
    for(int i = 0; i < n; i++)
    {
      out += weights[i*out_w + column] * input[row*n + i];
    }
    output[row*out_w+column] = out > 0.f ? out : 0.f;
  }
}"""

    code_obj3 = Code(code=forward_relu, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj3.code = remove_invisible_chars(code_obj3.code)
    with self.voiceover(text="""And fuse them together, completly eliminating the 2 memory accessed that relu kernel introduced""") as trk:
      self.play(Transform(VGroup(code_obj, code_obj2), code_obj3, replace_mobject_with_target_in_scene=True))

    relu_backwards = """__global__ void relu_backwards(int w, int h, float* a, float* d_l, float* b)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float activation = a[row*w+column];
    b[row*w+column] = activation > 0.f ? d_l[row*w+column] : 0.f;
  }
}"""

    backwards = """__global__ void backward(int batch_size, int n, int out_w,
                         float* weights, float* biases, float* d_l, float* out_d_l)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    float dl = 0.f;
    for(int i = 0; i < n; i++)
    {
      float w = weights[i*out_w + column];
      dl += w*d_l[row*n + i];
    }
    out_d_l[row*out_w + column] = dl;
  }
}"""


    code_obj = Code(code=backwards, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1).to_edge(UP)
    code_obj.code = remove_invisible_chars(code_obj.code)

    code_obj2 = Code(code=relu_backwards, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1).to_edge(DOWN)
    code_obj2.code = remove_invisible_chars(code_obj2.code)

    back_group = VGroup(code_obj2, code_obj)
    self.wait(2)
    with self.voiceover(text="""And if we look at the backward pass we have the exact same situation""") as trk:
      self.play(Transform(code_obj3, back_group, replace_mobject_with_target_in_scene=True))


    backwards_relu="""__global__ void backward(int batch_size, int n, int out_w, float* weights,
                         float* biases, float* d_l, float* out_d_l, float* activations)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    float dl = 0.f;
    for(int i = 0; i < n; i++)
    {
      float w = weights[i*out_w + column];
      dl += w*d_l[row*n + i];
    }
    float activation = activations[row*out_w+column];
    out_d_l[row*out_w + column] = activation > 0.f ? dl : 0.f;
  }
}"""

    code_obj3 = Code(code=backwards_relu, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj3.code = remove_invisible_chars(code_obj3.code)
    with self.voiceover(text="""So there is nothing holding us back from fusing those kernels too""") as trk:
      self.play(Transform(back_group, code_obj3, replace_mobject_with_target_in_scene=True))
      
    self.wait(1)

    training_time=9.5
    overhead_r = Rectangle(height=0.5, width=overhead_time/w_scale, color=BLUE, fill_color=BLUE, fill_opacity=0.5)
    training_r = Rectangle(height=0.5, width=training_time/w_scale, color=GREEN, fill_color=GREEN, fill_opacity=0.5)
    throughput = VGroup(overhead_r, training_r).arrange(RIGHT,buff=0.02)
    overhead_t = Text("2 s", font_size=fs, color=BLUE).move_to(overhead_r)
    training_t = Text("9.5 s", font_size=fs, color=GREEN).move_to(training_r)
    epochs = VGroup(*[Rectangle(height=0.5, width=training_time/(10*w_scale), color=GREEN, fill_color=GREEN, fill_opacity=0.5) for i in range(10)]).arrange(RIGHT, buff=0.02).move_to(training_r, aligned_edge=LEFT)
    epoch_times = VGroup(*[Text("0.95 s", font_size=fs, color=GREEN).move_to(epochs[i]) for i in range(10)])

    with self.voiceover(text="""And with those changes we managed to get from our initial timings, of 2 seconds od data loading, and 0.95 seconds per epoch totalling in 11.5 seconds for full training""") as trk:
      self.play(Transform(code_obj3, VGroup(overhead_r, overhead_t, epochs, epoch_times), replace_mobject_with_target_in_scene=True))
    self.wait(1)

    training_time=4
    epochs_2 = VGroup(*[Rectangle(height=0.5, width=training_time/(10*w_scale), color=GREEN, fill_color=GREEN, fill_opacity=0.5) for i in range(10)]).arrange(RIGHT, buff=0.02).next_to(group, RIGHT, buff=0.02).shift(0.3*LEFT)
    anims = []
    anims.extend([Unwrite(x) for x in epoch_times])
    anims.append(Unwrite(overhead_t))
    for i in range(1, 10):
      anims.append(Transform(epochs[i], epochs_2[i]))

    color_grad = color_gradient([BLUE, GREEN], 2)

    combined = Rectangle(height=0.5, width=1.2/(w_scale), fill_opacity=0.5).next_to(epochs_2, LEFT, buff=0.02).set_color(color_grad).shift(0.4*RIGHT)
    anims.append(Transform(VGroup(overhead_r, epochs[0]), combined, replace_mobject_with_target_in_scene=True))
    anims.append(Write(Text("1.1 s", font_size=fs).set_color(color_grad).move_to(combined)))
    b = Brace(epochs_2[-1], color=GREEN)
    time = Text("400 ms", color=GREEN, font_size=fs).next_to(b, DOWN)
    anims.append(Create(b))
    anims.append(Write(time))
    with self.voiceover(text="""To a first epoch taking 1.1 seconds of work and dataloading, and the rest taking 400 miliseconds, giving us a total training time of around 5 seconds,
                        reducing our total training time by more than a half""") as trk:
      self.play(LaggedStart(*anims))
    self.wait(1)

    with self.voiceover(text="""But wait, there's more""") as trk:
      pass

    with self.voiceover(text="""Since our data and network is very small it's currently highly underutilized, because the kernels are not even filling all of the possible threads""") as trk:
      pass

    with self.voiceover(text="""This is a bit like cheating, but if we increase our batch size from 16 to 64 we can parallelize our calculations even further""") as trk:
      pass
    training_time=1.2
    epochs_3 = VGroup(*[Rectangle(height=0.5, width=training_time/(10*w_scale), color=GREEN, fill_color=GREEN, fill_opacity=0.5) for i in range(10)]).arrange(RIGHT, buff=0.02).next_to(combined, RIGHT, buff=0.02).shift(0.12*LEFT)
    anims = []
    for i in range(1, 10):
      anims.append(Transform(epochs[i], epochs_3[i]))

    b2 = Brace(epochs_3[-1], color=GREEN)
    time2 = Text("120 ms", color=GREEN, font_size=fs).next_to(b2, DOWN)
    anims.append(Transform(b, b2))
    anims.append(Transform(time, time2))
    with self.voiceover(text="""And get to just 120 miliseconds per epoch, leaving us with a final improvement of 5x over the original code""") as trk:
      self.play(LaggedStart(*anims))
    self.wait(1)

    with self.voiceover(text="""With this it's time to end the episode, with no magic at all, just simple kernel fusion and parallelization we managed to 
                        improve our performance significantly""") as trk:
      pass

    with self.voiceover(text="""This was just a toy example - operating on very small data. In the next episodes we are going to look more in depth into 
                        memory organization of CUDA and how to leavrage it to get maximum performance""") as trk:
      pass
    
    with self.voiceover(text="""Subscribe if you don't want to miss it. Also leave your feedback in the comments, like the video and share it with your friends""") as trk:
      pass

    with self.voiceover(text="""As always, I'll link the code in the description so that you can play around with it.
                        I bet that you could squeeze out even more performance out of it, and if you do - share your changes in the comments""") as trk:
      pass

    with self.voiceover(text="""Thank you for your support and see you in the next episode, bye""") as trk:
      pass
    
    self.wait(1)
    anims = [] 
    for obj in self.mobjects:
      anims.append(FadeOut(obj))
    self.play(*anims)
    self.wait(3)
