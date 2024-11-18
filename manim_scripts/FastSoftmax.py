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

    formula2.to_edge(UP)
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
        self.play(FadeOut(m2))
        self.play(Uncreate(code_obj), Uncreate(hl), Uncreate(hl2))
        self.play(Create(formula2))
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
