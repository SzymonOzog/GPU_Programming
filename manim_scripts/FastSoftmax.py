from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import random
import math
from math import radians

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=0)

class FastSoftmax (VoiceoverScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        ) 
    example = [-1, 2, 0.1, 0.6, 0.2, 0.1, 1.1, -1.2, 0, -1]
    conf = {"font_size": 36}

    
    func = Tex("$softmax($", "$x$", "$)$")
    chart = BarChart(values = example, bar_names = [f"{i}" for i in range(10)], x_axis_config=conf).scale(0.5).next_to(func, LEFT)
    chart2 = BarChart(values = softmax(np.array(example)), bar_names = [f"p({i})" for i in range(10)], x_axis_config=conf).scale(0.5).next_to(func, RIGHT)

    with self.voiceover(text="""Softmax is one of the most important functions in deep learning as for today""") as trk:
      self.play(Create(chart))
      self.wait(1)
      self.play(Write(func))

    with self.voiceover(text="""It takes in vector of real numbers""") as trk:
      self.play(Transform(chart.copy(), func[1].copy().set_opacity(0), replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""And return a probability distribution, that we can reason about""") as trk:
      self.play(Transform(func.copy().copy().set_opacity(0), chart2, replace_mobject_with_target_in_scene=True))
    self.wait(1)


    vec = [[3], [5], [2]]
    formulas = [["\\frac{e^3}{e^3+e^5+e^2}"], ["\\frac{e^5}{e^3+e^5+e^2}"], ["\\frac{e^2}{e^3+e^5+e^2}"]]
    results = np.expand_dims(softmax(np.array(vec).flatten()), axis=1).tolist()
    results = [[f"{r[0]:.3f}"] for r in results]
    vbuff=1.5
    v0 = Matrix(vec, v_buff=vbuff)
    a1 = Arrow(start=LEFT, end=RIGHT)
    v1 = Matrix(formulas, v_buff=vbuff)
    eq = Tex("=")
    v2 = Matrix(results, v_buff=vbuff)
    calculations = VGroup(v0, a1, v1, eq, v2).arrange(RIGHT).move_to(ORIGIN)
    softmax_t = Tex("softmax").next_to(a1, UP, buff=0.1)
    formula = MathTex(
        r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}"
    ).next_to(calculations, UP)

    with self.voiceover(text="""The usual way of calculating it is by replacing each element with an exponent, raised to the power of said element
                        divided by the sum of exponents of all elements in our vector""") as trk:
      self.play(FadeOut(chart), FadeOut(chart2))
      self.play(Create(v0), Create(a1), Write(softmax_t), Create(v1), Write(eq), Create(v2))
      self.play(Transform(func, formula, replace_mobject_with_target_in_scene=True))

    formula2 = MathTex(
        r"\text{softmax}(x_i) = \frac{e^{x_i - max(x)}}{\sum_{j=1}^{K} e^{x_j - max(x)}}"
    ).move_to(formula)
    vec = [[3025], [3020], [3000]]
    formulas = [["\\frac{e^{3025}}{e^{3025}+e^{3020}+e^{3000}}"], ["\\frac{e^{3020}}{e^{3025}+e^{3020}+e^{3000}}"], ["\\frac{e^{3000}}{e^{3025}+e^{3020}+e^{3000}}"]]
    results = np.expand_dims(softmax(np.array(vec).flatten()), axis=1).tolist()
    vbuff=1.5
    v0_t = Matrix(vec, v_buff=vbuff)
    a1_t = Arrow(start=LEFT, end=RIGHT)
    v1_t = Matrix(formulas, v_buff=vbuff)
    v2_t = Matrix(results, v_buff=vbuff)    
    eq_t = Tex("=")
    calculations = VGroup(v0_t, a1_t, v1_t, eq_t, v2_t).arrange(RIGHT).move_to(ORIGIN)
    softmax_t_t = Tex("softmax").next_to(a1_t, UP, buff=0.1)
    with self.voiceover(text="""Although there is one caveat, since it uses an exponential function, that grows - well exponentially
                        if our input vector will contain multiple<bookmark mark='1'/> positive values, it can overflow as we will add a lot of big numbers together
                        in our divisorr""") as trk:
      self.wait_until_bookmark("1")
      self.play(Transform(v0, v0_t),
                Transform(v1, v1_t),
                Transform(v2, v2_t),
                Transform(a1, a1_t),
                Transform(eq, eq_t),
                Transform(softmax_t, softmax_t_t))
      
    vec = [[3025], [3020], [3000]]
    formulas = [["\\frac{e^{3025-3025}}{e^{3025-3025}+e^{3020-3025}+e^{3000-3025}}"], ["\\frac{e^{3020-3025}}{e^{3025-3025}+e^{3020-3025}+e^{3000-3025}}"], ["\\frac{e^{3000-3025}}{e^{3025-3025}+e^{3020-3025}+e^{3000-3025}}"]]
    results = np.expand_dims(softmax(np.array(vec).flatten() - 3025), axis=1).tolist()
    results = [[f"{r[0]:.3f}"] for r in results]
    vbuff=1.5
    v0_t = Matrix(vec, v_buff=vbuff)
    a1_t = Arrow(start=LEFT, end=RIGHT)
    v1_t = Matrix(formulas, v_buff=vbuff)
    v2_t = Matrix(results, v_buff=vbuff)    
    eq_t = Tex("=")
    calculations = VGroup(v0_t, a1_t, v1_t, eq_t, v2_t).arrange(RIGHT).move_to(ORIGIN)
    softmax_t_t = Tex("softmax").next_to(a1_t, UP, buff=0.1)

    with self.voiceover(text="""We can mitigate this by subtracting the maximum of our vector from the exponent. 
                        That way - the powers will always be negative, and our values will remain in range of 0 to 1""") as trk:
      self.play(Transform(formula, formula2))
      self.play(Transform(v0, v0_t),
                Transform(v1, v1_t),
                Transform(v2, v2_t),
                Transform(a1, a1_t),
                Transform(eq, eq_t),
                Transform(softmax_t, softmax_t_t))
    self.wait(1)


    m = Tex("$m = max(x)$", color=YELLOW)
    x = Tex("$x = x - m$", color=BLUE)
    e = Tex("$exp = e^x$", color=RED)
    s = Tex("$s = sum(exp)$", color=GREEN)
    out = Tex("$out = \\frac{exp_i}{s}$", color=ORANGE)
    VGroup(m,x,e,s,out).arrange(DOWN).next_to(formula2, DOWN)
    flops = Tex("FLOPS", "=", "N", "+", "N", "+", "N", "+", "N", "+", "N", font_size=48).next_to(out, DOWN).align_to(formula2, LEFT).shift(LEFT)
    bytes_loaded = Tex("Bytes", "=", "2*N", "*4", font_size=48).next_to(out, DOWN).align_to(formula2, RIGHT).shift(RIGHT)

    with self.voiceover(text="""To have an estimate for how fast our kenel can theoretically be we need to calculate how <bookmark mark='1'/>much floating 
                        point operations are we calculating, <bookmark mark='2'/>and how much memory are we accessing""") as trk:
        self.play(Uncreate(v0), Uncreate(a1), Uncreate(softmax_t), Uncreate(v1), Uncreate(v2), Uncreate(eq))
        self.wait_until_bookmark("1")
        self.play(Write(flops[:2]))
        self.wait_until_bookmark("2")
        self.play(Write(bytes_loaded[:2]))

    with self.voiceover(text="""For bytes loaded it's quite simple, we load the whole vector once, and save it once so we get <bookmark mark='1'/> 2 times our vector size memory accesses of floating point values
                        that are <bookmark mark='2'/>4 bytes each""") as trk:
        self.wait_until_bookmark("1")
        self.play(Write(bytes_loaded[2]))
        self.wait_until_bookmark("2")
        self.play(Write(bytes_loaded[3]))
        self.play(Transform(bytes_loaded, 
                              Tex("Bytes", "=", "8*N", font_size=48).next_to(out, DOWN).align_to(formula2, RIGHT).shift(RIGHT)))

    with self.voiceover(text="""For the flops we have to split our function into suboperations""") as trk:
        pass

    with self.voiceover(text="""First we calculate <bookmark mark='1'/>the maximum, giving us <bookmark mark='2'/>N operations""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(m))
      self.wait_until_bookmark("2")
      self.play(Write(flops[2].set_color(m.color)))

    with self.voiceover(text="""We then subtract <bookmark mark='1'/>the maximum from our vector giving us <bookmark mark='2'/> another N operations""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(x))
      self.wait_until_bookmark("2")
      self.play(Write(flops[3]))
      self.play(Write(flops[4].set_color(x.color)))

    with self.voiceover(text="""This is followed by an<bookmark mark='1'/> exponent <bookmark mark='2'/> for the next N operations""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(e))
      self.wait_until_bookmark("2")
      self.play(Write(flops[5]))
      self.play(Write(flops[6].set_color(e.color)))

    with self.voiceover(text="""The next operation is sum<bookmark mark='1'/> across all elements <bookmark mark='2'/> so the next N operations""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(s))
      self.wait_until_bookmark("2")
      self.play(Write(flops[7]))
      self.play(Write(flops[8].set_color(s.color)))

    with self.voiceover(text="""And for the final output<bookmark mark='1'/> each element needs to be divied by the sum<bookmark mark='2'/> giving us the next N operations""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(out))
      self.wait_until_bookmark("2")
      self.play(Write(flops[9]))
      self.play(Write(flops[10].set_color(out.color)))

    with self.voiceover(text="""This leaves us with 5 N FLOPS per 8 Bytes loaded""") as trk:
        self.play(Transform(flops, 
                            Tex("FLOPS", "=", "5*N", font_size=48).next_to(out, DOWN).align_to(formula2, LEFT).shift(LEFT)))

    x_range = list(range(10, 18))
    ns = [2**x for x in x_range]
    axes = Axes(
            x_range=[10, 17, 1],
            y_range=[0, 700, 100],
            x_length=7,
            y_length=5,
            axis_config={"include_tip": False, "include_numbers": True},
            x_axis_config={"scaling": LogBase(2)},
            ).shift(LEFT+UP)
    x_text = MathTex("N")
    y_text = MathTex("Performance[GFLOPS]")
    x_label = axes.get_x_axis_label(x_text, edge=DR, direction=DR)
    y_label = axes.get_y_axis_label(y_text.rotate(PI/2), edge=LEFT, direction=LEFT)

    theoretical_performance = Line(start=axes.c2p(2**10, 625), end=axes.c2p(2**17, 625), color=GREEN)
    theoretical_text = Text("Theoretical Maximum 625 GFLOPS", color=GREEN, font_size=18).next_to(theoretical_performance, RIGHT, buff=0.1)

    graph = VGroup(axes, x_label, y_label, theoretical_performance)

    theoretical = Tex("$TheoreticalMaximum =$", "$\\frac{5}{8}$", "$*1\\frac{TB}{s}$", "$= 625\\,GFLOPs$").next_to(graph, DOWN)

    with self.voiceover(text="""With this info we can calculate<bookmark mark='1'/> a theoretical maximum of a performance
                        that we can get out of this kernel. With <bookmark mark='2'/> 5 floating point operations per 8
                        loaded bytes we are bottlenecked by memory bandwith which is <bookmark mark='3'/> 1 TB/s
                        on my gpu, and that gives us a theoretical maximum <bookmark mark='4'/>of 625 GFLOPS""") as trk:
        self.wait_until_bookmark("1")
        self.play(Write(theoretical[0]))
        self.wait_until_bookmark("2")
        self.play(Transform(VGroup(flops, bytes_loaded), theoretical[1], replace_mobject_with_target_in_scene=True))
        self.wait_until_bookmark("3")
        self.play(Write(theoretical[2]))
        self.wait_until_bookmark("4")
        self.play(Write(theoretical[3]))

    with self.voiceover(text="""We can now compare the speed of different implementations, for different widths of the input.
                        The height or batch size in case of neural networks will be fixed at 128""") as trk:
        self.play(Transform(VGroup(formula,m,x,e,s,out),
                            axes,
                            replace_mobject_with_target_in_scene=True))
        self.play(Transform(theoretical, theoretical_performance, replace_mobject_with_target_in_scene=True))
        self.play(Write(theoretical_text))
        self.play(Write(x_text), Write(y_text))

    times_torch = [4.768, 4.352, 5.888, 9.088, 16.64, 31.808, 64.64, 175.04]
    times_triton = [3.072, 3.68, 5.344, 8.672, 15.616, 28.32, 70.272, 630.24]
    flops_torch = [(128*n*5)/(t*1e3) for (t,n) in zip(times_torch, ns)]
    flops_triton = [(128*n*5)/(t*1e3) for (t,n) in zip(times_triton, ns)]
    graph_torch = axes.plot_line_graph(ns, flops_torch, line_color=ORANGE, add_vertex_dots=False)
    graph_triton = axes.plot_line_graph(ns, flops_triton, line_color=BLUE, add_vertex_dots=False)
    text_torch=Text("Torch", color=ORANGE, font_size=18).move_to(axes.c2p(2**17, flops_torch[-1])+0.1*RIGHT, LEFT)
    text_triton=Text("Triton", color=BLUE, font_size=18).move_to(axes.c2p(2**17, flops_triton[-1])+0.1*RIGHT, LEFT)
    with self.voiceover(text="""And as a reference point, I took the <bookmark mark='1'/>pytorch kernel, as well as a triton kernel <bookmark mark='2'/>that was available 
                        in their documentation as an example""") as trk:
        self.wait_until_bookmark("1")
        self.play(Create(graph_torch), Write(text_torch))
        self.wait_until_bookmark("2")
        self.play(Create(graph_triton), Write(text_triton))

    with self.voiceover(text="""You are probably screaming right now, looking at the result""") as trk:
        pass

    mem_chart = ImageMobject("image.png").scale(1.75)
    # c = NumberPlane().add_coordinates()
    # self.play(Write(c))
    with self.voiceover(text="""And I pulled my hair on it for quite a while, until I took out the profiler and did some reaserch on how 
                        are cuda stores done""") as trk:
        self.play(FadeIn(mem_chart))
    
    w1 = Rectangle(width=2, height=1, color=RED).shift(0.8*UP + RIGHT)
    w2 = Rectangle(width=2, height=1, color=BLUE).shift(4.2*RIGHT + 0.3*UP)
    with self.voiceover(text="""Nvidia GPUs use something called write back cache, this essentially means that we are writing to <bookmark mark='1'/> L2 
                        cache only during kernel execution and the global<bookmark mark='2'/> memory recieves the data later, when we discard the cache block""") as trk:
        self.wait_until_bookmark("1")
        self.play(Create(w1))
        self.wait_until_bookmark("2")
        self.play(Create(w2))

    axes_t = Axes(
            x_range=[10, 17, 1],
            y_range=[0, 1400, 200],
            x_length=7,
            y_length=5,
            axis_config={"include_tip": False, "include_numbers": True},
            x_axis_config={"scaling": LogBase(2)},
            ).shift(LEFT+UP)
    theoretical_performance_t = Line(start=axes_t.c2p(2**10, 2*625), end=axes_t.c2p(2**17, 2*625), color=GREEN)
    theoretical_text_t = Text("Theoretical Maximum 1250 GFLOPS", color=GREEN, font_size=18).next_to(theoretical_performance, RIGHT, buff=0.1)
    graph_torch_t = axes_t.plot_line_graph(ns, flops_torch, line_color=ORANGE, add_vertex_dots=False)
    graph_triton_t = axes_t.plot_line_graph(ns, flops_triton, line_color=BLUE, add_vertex_dots=False)
    with self.voiceover(text="""And since our L2 write speed is much higher than our global memory read spead, the only bottleneck is reads from global memory,
                        so our theoretical maximum increases by 2x""") as trk:
        self.play(Uncreate(w1), Uncreate(w2))
        self.play(FadeOut(mem_chart))
        self.play(Transform(axes, axes_t),
                  Transform(theoretical_performance, theoretical_performance_t),
                  Transform(theoretical_text, theoretical_text_t),
                  Transform(graph_torch, graph_torch_t),
                  Transform(graph_triton, graph_triton_t),
                  text_torch.animate.move_to(axes_t.c2p(2**17, flops_torch[-1])+0.1*RIGHT, LEFT),
                  text_triton.animate.move_to(axes_t.c2p(2**17, flops_triton[-1])+0.1*RIGHT, LEFT),
                  y_label.animate.shift(0.2*LEFT))

    axes = axes_t

    softmax_code="""__global__ void softmax(int w, int h, float* input, float* output)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    float maxval = input[row*w];
    for (int i = 1; i<w; i++)
    {
      maxval = max(maxval, input[row*w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i<w; i++)
    {
      divisor += exp(input[row*w + i] - maxval);
    }
    output[row*w + col] = exp(input[row*w + col]-maxval)/(divisor);
  }
}"""


    code_obj = Code(code=softmax_code, tab_width=2, language="c", font_size=12, line_no_buff=0.1, corner_radius=0.1)
    code_obj.code = remove_invisible_chars(code_obj.code)

    with self.voiceover(text="""If you watched the episode on neural networks, we wrote a very simple softmax kernel there""") as trk:
        self.play(*[FadeOut(x) for x in self.mobjects])
        self.play(Create(code_obj))

    with self.voiceover(text="""In it each thread calculates one output element of a softmax, 
                        so we create as many threads as there are elements in our input matrix""") as trk:
        self.play(code_obj.animate.to_edge(UP))
        m2 = Matrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,w}"],
                     ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,w}"],
                     ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                     ["x_{h,0}", "x_{h,1}", "\\cdots", "x_{h,w}"]], element_alignment_corner=ORIGIN).scale(0.8).next_to(code_obj, DOWN)
        self.play(Create(m2))

    hl = SurroundingRectangle(code_obj.code[7:11], buff=0.03, stroke_width=2, fill_opacity=0.3)
    hl2 = SurroundingRectangle(code_obj.code[12:16], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""There is one major bottleneck in this kernel, each thread in the row recalculates the maxval and the divisor""") as trk:
        self.play(Create(hl))
        self.play(Create(hl2))
        
    with self.voiceover(text="""While this wasn't really a big problem in our MNIST solver, 
                        where the height of the input was much bigger than the width""") as trk:
        self.play(Transform(m2, 
                            Matrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,w}"],
                                    ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,w}"],
                                    ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                    ["x_{h,0}", "x_{h,1}", "\\cdots", "x_{h,w}"]], v_buff=1.2, element_alignment_corner=ORIGIN).scale(0.8).next_to(code_obj, DOWN)
                            )
                  )

    with self.voiceover(text="""But in recent trends, the amount of classes that we are predicting
                        is much bigger than the batch size we are feeding the model""") as trk:
        self.play(Transform(m2, 
                            Matrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,w}"],
                                    ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,w}"],
                                    ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                                    ["x_{h,0}", "x_{h,1}", "\\cdots", "x_{h,w}"]], h_buff=2.8, element_alignment_corner=ORIGIN).scale(0.8).next_to(code_obj, DOWN)
                            )
                  )
    results = Text("8.9 GFLOPS", color=RED).next_to(m2, DOWN)
    with self.voiceover(text="""I'm not patient enough to run it for all of the shapes, but for just 1024 elements, it achieves a 
                        magnificent<bookmark mark='1'/> 8.9 GFLOPS""") as trk:
        self.play(Write(results))

    objs = [Square(side_length=0.5) for _ in range(16)]
    VGroup(*objs).arrange(RIGHT,buff=0.06).move_to(ORIGIN+3*UP)

    with self.voiceover(text="""The key to making a fast softmax algorithm is understanding how to perform a fast reduction
                        algorithm""") as trk:
        self.play(*[FadeOut(x) for x in self.mobjects])
        self.play(LaggedStart(*[Create(x) for x in objs]))
    self.camera.background_color = GREY

    def find_end(o1, o2):
        y = min(o1.get_y(), o2.get_y()) - 0.5
        end = (o1.get_center() + o2.get_center())/2
        end[1] = y
        end = o1.get_bottom() + 0.3*DOWN + 0.25 * RIGHT
        return end

    start = objs[0]
    anims = []
    uncreate_anims = []
    for obj in objs[1:]:
        end = find_end(start, obj)
        op = Text("Op", font_size=14).move_to(end)
        l1 = Line(start.get_corner(DOWN), op.get_corner(UL))
        l2 = Line(obj.get_corner(DOWN), op.get_corner(UR))
        start = op
        anims.extend([Create(l1), Create(l2), Write(op)])
        uncreate_anims.extend([Uncreate(l1), Uncreate(l2), Unwrite(op)])

    with self.voiceover(text="""A reduction algorithm is a type of algorithm where we need to perform
                        an operation on every input element where the input to the operation is a result of the previous input""") as trk:
        self.play(LaggedStart(*anims))


    def find_end(o1, o2):
        end = o1.get_bottom() + 0.3*DOWN + 0.25 * LEFT
        return end

    with self.voiceover(text="""In order for this to parallelize nicely, the operator need to be associative""") as trk:
        self.play(LaggedStart(*uncreate_anims))
    anims = []
    uncreate_anims = []
    start = objs[-1]
    for obj in reversed(objs[:-1]):
        end = find_end(start, obj)
        op = Text("Op", font_size=14).move_to(end)
        l1 = Line(start.get_corner(DOWN), op.get_corner(UR))
        l2 = Line(obj.get_corner(DOWN), op.get_corner(UL))
        start = op
        anims.extend([Create(l1), Create(l2), Write(op)])
        uncreate_anims.extend([Uncreate(l1), Uncreate(l2), Unwrite(op)])

    with self.voiceover(text="""That means that no matter the order of the operations, the result will be the same""") as trk:
        self.play(LaggedStart(*anims))
    self.wait(1)
    with self.voiceover(text="""This also gives us a wonderful property""") as trk:
        self.play(LaggedStart(*uncreate_anims))

    def find_end(o1, o2):
        y = min(o1.get_y(), o2.get_y()) - 1
        end = (o1.get_center() + o2.get_center())/2
        end[1] = y
        return end

    anims = []
    uncreate_anims = []
    step = objs
    ops = []
    while len(step) > 1:
        next_step = []
        for i in range(0, len(step), 2):
            o1 = step[i]
            o2 = step[i+1]
            end = find_end(o1, o2)
            op = Text("Op", font_size=20).move_to(end)
            l1 = Line(o1.get_corner(DOWN), op.get_corner(UL))
            l2 = Line(o2.get_corner(DOWN), op.get_corner(UR))
            anims.extend([Create(l1), Create(l2), Write(op)])
            uncreate_anims.extend([Uncreate(l1), Uncreate(l2), Unwrite(op)])
            ops.append(op)
            next_step.append(op)
        step = next_step

    with self.voiceover(text="""Where we don't need to calculate sequentially, but we can do it in a tree like manner""") as trk:
        self.play(LaggedStart(*anims))
    self.wait(1)

    with self.voiceover(text="""In the case of our softmax we perform 2 associative reductions""") as trk:
        pass

    with self.voiceover(text="""One is finding a maximum""") as trk:
        self.play(*[Transform(op,
                              Text("Max", font_size=16).move_to(op))
                    for op in ops])

    with self.voiceover(text="""And the second one is summing all elements to calculate our divisor""") as trk:
        self.play(*[Transform(op,
                              Text("+", font_size=24).move_to(op))
                    for op in ops])

    with self.voiceover(text="""Now with this prerequisite taken of, we need to look at speeding up our algorithm""") as trk:
        self.play(LaggedStart(*uncreate_anims))

    ln1 = Line(UP, 10*DOWN).next_to(objs[1], DOWN).shift(DOWN)
    ln2 = Line(UP, 10*DOWN).next_to((objs[7].get_bottom() + objs[8].get_bottom())/2, DOWN).shift(DOWN)
    ln3 = Line(UP, 10*DOWN).next_to(objs[-2], DOWN).shift(DOWN)

    # b1 = Text("Block 1", color=BLUE, font_size=36).next_to(l1, UP)
    # b2 = Text("Block 2", color=BLUE, font_size=36).next_to(l3, UP)

    t_fs = 24
    dir = (ln3.get_center() - ln1.get_center()) / 4
    start = ln1.get_corner(UP) + 0.5*DOWN
    t1 = Text("Thread 1", color=YELLOW, font_size=t_fs).move_to(start - dir)
    t2 = Text("Thread 2", color=GREEN, font_size=t_fs).move_to(start + dir)
    t3 = Text("Thread 3", color=ORANGE, font_size=t_fs).move_to(start + 3*dir)
    t4 = Text("Thread 4", color=TEAL, font_size=t_fs).move_to(start  + 5*dir)
    with self.voiceover(text="""And to do that, we need to think more deeply about how do we behave on a thread 
                        and block level""") as trk:
        self.play(Create(ln1), Create(ln2), Create(ln3))
        # self.play(Write(b1), Write(b2))
        self.play(Write(t1), Write(t2), Write(t3), Write(t4))

    w1 = VGroup(*objs).copy()
    w2 = VGroup(*objs).copy()
    w3 = VGroup(*objs).copy()
    w4 = VGroup(*objs).copy()

    with self.voiceover(text="""In our naive kernel, each thread operates on the entirety of the data""") as trk:
        self.play(w1.animate.scale(0.33).next_to(t1, DOWN),
                  w2.animate.scale(0.33).next_to(t2, DOWN),
                  w3.animate.scale(0.33).next_to(t3, DOWN),
                  w4.animate.scale(0.33).next_to(t4, DOWN))

    def find_end(o1, o2):
        y = min(o1.get_y(), o2.get_y()) - 0.5
        end = (o1.get_center() + o2.get_center())/2
        end[1] = y
        return end
    anims = []
    uncreate_anims = []
    for x in [w1, w2, w3, w4]:
        step = x
        while len(step) > 1:
            next_step = []
            for i in range(0, len(step), 2):
                o1 = step[i]
                o2 = step[i+1]
                end = find_end(o1, o2)
                op = Circle(radius=0.05, color=BLUE).move_to(end)
                l1 = Line(o1.get_corner(DOWN), op.get_corner(UL))
                l2 = Line(o2.get_corner(DOWN), op.get_corner(UR))
                anims.extend([Create(l1), Create(l2), Write(op)])
                uncreate_anims.extend([Uncreate(l1), Uncreate(l2), Unwrite(op)])
                next_step.append(op)
            step = next_step

    with self.voiceover(text="""And independently performs a reduction""") as trk:
        self.play(LaggedStart(*anims))

    with self.voiceover(text="""it doesn't take much effort to notice that this is a lot of repeated work""") as trk:
        pass

    with self.voiceover(text="""We're going to need to be able to distribute the work between our threads""") as trk:
        self.play(LaggedStart(*uncreate_anims, Uncreate(w1), Uncreate(w2), Uncreate(w3), Uncreate(w4)))

    ws=[]
    ogs=[]
    color_anims=[]
    for i in range(4):
        ogs.append(VGroup(*objs[i*4:(i+1)*4]))
        ws.append(VGroup(*objs[i*4:(i+1)*4]).copy())

    ts = [t1, t2, t3, t4]
    with self.voiceover(text="""The first thing is to distribute the input equally between the threads""") as trk:
        for w, t, og in zip(ws, ts, ogs):
            self.play(w.animate.set_color(t.color).next_to(t, DOWN),
                      og.animate.set_color(t.color))

    last = []
    thread_level_reduction = []
    for w, t in zip(ws, ts):
        step = w
        while len(step) > 1:
            next_step = []
            for i in range(0, len(step), 2):
                o1 = step[i]
                o2 = step[i+1]
                end = find_end(o1, o2)
                op = Circle(radius=0.1, color=t.color).move_to(end)
                l1 = Line(o1.get_corner(DOWN), op.get_corner(UL))
                l2 = Line(o2.get_corner(DOWN), op.get_corner(UR))
                anims.extend([Create(l1), Create(l2), Write(op)])
                uncreate_anims.extend([Uncreate(l1), Uncreate(l2), Unwrite(op)])
                next_step.append(op)
                thread_level_reduction.extend([l1, l2, op])
            step = next_step
        last.extend(step)

    with self.voiceover(text="""then we can perform the reduction on those inputs like we did before""") as trk:
        self.play(LaggedStart(*anims))

    def find_end(o1, o2):
        y = min(o1.get_y(), o2.get_y()) - 1 
        end = (o1.get_center())
        end[1] = y
        return end

    step = last
    anims = []
    shared_mem_reduction = []
    while len(step) > 1:
        next_step = []
        for i in range(0, len(step), 2):
            o1 = step[i]
            o2 = step[i+1]
            end = find_end(o1, o2)
            op = Circle(radius=0.1, color=o1.color).move_to(end)
            l1 = Line(o1.get_corner(DOWN), op.get_corner(UP))
            l2 = Line(o2.get_corner(DOWN), op.get_corner(UR))
            anims.extend([Create(l1), Create(l2), Write(op)])
            uncreate_anims.extend([Uncreate(l1), Uncreate(l2), Unwrite(op)])
            next_step.append(op)
            shared_mem_reduction.extend([l1, l2, op])
        step = next_step

    with self.voiceover(text="""Afterwards, we transmit the data between the threads and finalize the reduction""") as trk:
        self.play(LaggedStart(*anims))


    end = op.get_center() + DOWN
    transmitted = [Circle(radius=0.1, color=ts[i].color).move_to(end + dir*i*2) for i in range(1, 4)]
    lines = [Line(op.get_corner(DR), t) for t in transmitted]
    with self.voiceover(text="""And in the case of softmax, we need to finalize by transmitting the data to all other threads""") as trk:
        self.play(LaggedStart(*[Create(x) for x in lines + transmitted]))
    all = VGroup(*(ws+ts+thread_level_reduction+shared_mem_reduction+transmitted+lines), ln1, ln2, ln3)
    all.save_state()
    self.play(all.animate.scale(0.5).to_edge(LEFT).shift(3*UP))

    reduction_code = """__shared__ float reduction[BLOCK_DIM_Y]; 
float maxval = 0;
for (int i = ty*BLOCK_DIM_Y; i<min(w, (ty+1)*BLOCK_DIM_Y); i+=1)
{
  maxval = fmaxf(maxval, a[row*w + i]);
}

reduction[ty] = maxval;
for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
{
  __syncthreads();
  if (ty < stride)
  {
    reduction[ty] = fmaxf(reduction[ty], reduction[ty+stride]);
  }
}

__syncthreads();
maxval = reduction[0];
"""
    code_obj = Code(code=reduction_code, tab_width=2, language="c", font_size=12, line_no_buff=0.1, corner_radius=0.1, insert_line_no=False).to_edge(RIGHT, buff=0.25)
    code_obj.code = remove_invisible_chars(code_obj.code)
    with self.voiceover(text="""This is how the code for this reduction looks like""") as trk:
        self.play(Create(code_obj))

    hl1 = Rectangle(width=6.5, height=VGroup(*ws, *thread_level_reduction).height+0.05, color=RED_A, fill_color=RED_A, fill_opacity=0.25, stroke_width=2).move_to(VGroup(*ws, *thread_level_reduction)).shift(0.05*UP)
    hl2 = SurroundingRectangle(code_obj.code[2:6], color=RED_A, fill_color=RED_A, fill_opacity=0.25, buff=0.03, stroke_width=2)
    with self.voiceover(text="""For the first step we perform the reduction on a thread level""") as trk:
        self.play(Create(hl1), Create(hl2))

    hl3 = Rectangle(width=6.5, height=VGroup(*shared_mem_reduction).height+0.05, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.25, stroke_width=2).next_to(hl1, DOWN, buff=0)
    hl4 = SurroundingRectangle(code_obj.code[7:16], color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.25, buff=0.03, stroke_width=2)
    with self.voiceover(text="""We then exchange the data between the threads in shared memory and perform the reduction on a block level""") as trk:
        self.play(Create(hl3), Create(hl4))

    hl5 = Rectangle(width=6.5, height=VGroup(*(transmitted+lines)).height+0.05, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.25, stroke_width=2).next_to(hl3, DOWN, buff=0)
    hl6 = SurroundingRectangle(code_obj.code[17:], color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.25, buff=0.03, stroke_width=2)
    with self.voiceover(text="""And finally, we broadcast the data to all other threads""") as trk:
        self.play(Create(hl5), Create(hl6))

    reduction_code = """__shared__ float reduction[BLOCK_DIM_Y]; 
float divisor = 0.f;
for (int i = ty*BLOCK_DIM_Y; i<min(w, (ty+1)*BLOCK_DIM_Y); i+=1)
{
  divisor += __expf(a[row*w + i] - maxval);
}

reduction[ty] = divisor;
for(int stride = BLOCK_DIM_Y/2; stride>=1; stride/=2)
{
  __syncthreads();
  if (ty < stride)
  {
    reduction[ty] = reduction[ty] + reduction[ty+stride];
  }
}

__syncthreads();
divisor = reduction[0];
"""

    code_obj_t = Code(code=reduction_code, tab_width=2, language="c", font_size=12, line_no_buff=0.1, corner_radius=0.1, insert_line_no=False).to_edge(RIGHT, buff=0.25)
    code_obj_t.code = remove_invisible_chars(code_obj_t.code)
    with self.voiceover(text="""And in the same way that we found our maximum, we can fix calculating the divisor""") as trk:
        self.play(Transform(code_obj, code_obj_t),
                  Transform(hl2, SurroundingRectangle(code_obj.code[2:6], color=RED_A, fill_color=RED_A, fill_opacity=0.25, buff=0.03, stroke_width=2)),
                  Transform(hl4, SurroundingRectangle(code_obj.code[7:16], color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.25, buff=0.03, stroke_width=2)),
                  Transform(hl6, SurroundingRectangle(code_obj.code[17:], color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.25, buff=0.03, stroke_width=2)))

    graph = VGroup(axes, theoretical_performance, theoretical_text, graph_torch, graph_triton, text_torch, text_triton, y_text, x_text).shift(DOWN)
    self.play(*[FadeOut(x) for x in self.mobjects])
    times_cuda = [9.12, 10.976, 14.976, 18.944, 34.176, 65.568, 120.288, 310.688]
    flops_cuda = [(128*n*5)/(t*1e3) for (t,n) in zip(times_cuda, ns)]
    print(flops_cuda)
    print(flops_torch)
    graph_cuda = axes.plot_line_graph(ns, flops_cuda, line_color=RED, add_vertex_dots=False)
    text_cuda = Text("Fast Recuction", color=RED, font_size=18).move_to(axes.c2p(2**17, flops_cuda[-1])+0.1*RIGHT, LEFT)
    with self.voiceover(text="""We can now check the speed of our kernel""") as trk:
        self.play(FadeIn(graph))

    with self.voiceover(text="""And even though we are much better than the initial 8 GFLOPs, we're still off compared to real world implementations""") as trk:
        self.play(Create(graph_cuda))
        self.play(Write(text_cuda))

    graph.add(graph_cuda)
    graph.add(text_cuda)
    with self.voiceover(text="""Another thing that we can do to improve the speed of our kernel is to investigate our memory access pattern""") as trk:
        self.play(FadeOut(graph), FadeOut(graph_cuda), FadeOut(text_cuda))
        all.restore()
        self.play(FadeIn(all), *[FadeIn(x) for x in objs])

    anims = []
    with self.voiceover(text="""And we can see that we are accessing our data with a stride of BLOCK_SIZE, if you watched
                        the video on DRAM and memory coalescing you know that this is a very bad access pattern""") as trk:
        for i in range(4):
            self.play(*[objs[i + j*4].animate.set_fill(objs[i + j*4].color, 0.5) for j in range(4)], *anims)
            self.wait(1)
            anims.extend([objs[i + j*4].animate.set_fill(WHITE, 0) for j in range(4)])
