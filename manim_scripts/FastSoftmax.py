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

