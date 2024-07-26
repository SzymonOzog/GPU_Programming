from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
from NN import NeuralNetworkMobject

mnist = np.loadtxt("mnist_train.csv", delimiter=",")

def create_mnist(index, shape=(28,28)):
  example = mnist[index]
  label = example[0]
  pixels = example[1:]
  pixels = pixels.reshape((28,28))/255.0
  img = VGroup()
  sq = []
  for x in range(shape[0]):
    row = VGroup()
    for y in range(shape[1]):
      sq.append(Square(side_length=0.1, stroke_width=0.5, fill_opacity=pixels[x][y], fill_color=BLUE, color=WHITE))
      row.add(sq[-1])
    img.add(row.arrange(RIGHT, buff=0.01))
  img.arrange(DOWN,0.01)
  return sq, label, img

class NeuralNetwork(VoiceoverScene, ThreeDScene):
  def construct(self):
    self.set_speech_service(
        GTTSService(transcription_model="base")
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU Programming", font_size=72)
    with self.voiceover(text="Hello and welcome to episode 4 in the series on GPU programming") as trk:
      self.play(Write(title))

    self.play(Unwrite(title))
    nn = NeuralNetworkMobject([784, 300, 100, 10]).to_edge(DOWN, buff=0.05)
    idx = 0
    
    with self.voiceover(text="""In this episode, we will continue training our neural network from the previous episode,
                        so we will have to write a backward pass for our neural network, what the backward pass does is it 
                        changes each weight and bias in our neural network to minimize the loss function - improving it's performance""") as trk:
      anims = []
      for x in nn.neurons:
        anims.extend([Create(n) for n in x])
      for x in nn.edges:
        anims.extend([Create(n) for n in x])
      self.play(*anims)
      while trk.get_remaining_duration() > 0:
        idx += 1
        img, label, group = create_mnist(idx+1)
        group.scale(0.7).next_to(nn.neurons[0][10], LEFT)
        self.play(*[Create(x) for x in img])
        anims = []
        j = 0
        for i, x in enumerate(img):
          if i < 10 or i > 773:
            anims.append(FadeOut(x, target_position=nn.neurons[0][j]))
            j+=1
          else:
            j=11
            anims.append(FadeOut(x, target_position=nn.neurons[0][10]))
        self.play(*anims)
        for i in range(len(nn.neurons)-1):
          self.play(*[ShowPassingFlash(e.copy().set_color(BLUE)) for e in nn.edges[i]])
        self.play(Indicate(nn.neurons[-1][int(label) - 1], color=RED))
        for i in range(len(nn.edges)-1, -1, -1):
          self.play(*[ShowPassingFlash(e.copy().set_color(RED), reverse_rate_function=True) for e in nn.edges[i]])


    with self.voiceover(text="""But we cannot just change our weights and biases randomly untill we get some good performence,
                        not if we want to find the optimum in a reasonable time anyway""") as trk:
      idx += 1
      img, label, group = create_mnist(idx+1)
      group.scale(0.7).next_to(nn.neurons[0][10], LEFT)
      self.play(*[Create(x) for x in img])
      anims = []
      j = 0
      for i, x in enumerate(img):
        if i < 10 or i > 773:
          anims.append(FadeOut(x, target_position=nn.neurons[0][j]))
          j+=1
        else:
          j=11
          anims.append(FadeOut(x, target_position=nn.neurons[0][10]))
      self.play(*anims)
      for i in range(len(nn.neurons)-1):
        self.play(*[ShowPassingFlash(e.copy().set_color(BLUE)) for e in nn.edges[i]])
      self.play(Indicate(nn.neurons[-1][int(label) - 1], color=RED))
      for i in range(len(nn.edges)-1, -1, -1):
        self.play(*[ShowPassingFlash(e.copy().set_color(RED), reverse_rate_function=True) for e in nn.edges[i]])


    loss = Text("L", color=BLUE, font_size=24).next_to(nn, RIGHT, buff=LARGE_BUFF)
    rect = SurroundingRectangle(loss,color=BLUE)

    lines = []
    for output_neuron in nn.neurons[-1]:
      lines.append(Line(output_neuron, rect.get_corner(LEFT), stroke_width=nn.CONFIG["edge_stroke_width"]))

    with self.voiceover(text="""And to solve the problem we need to think about what we want to achieve, <bookmark mark='1'/>
                        we have our loss function""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(loss), Create(rect))
      self.play(LaggedStart(*[Create(l) for l in lines]))

    
    eq = Tex("$w$", "$=w-\\delta_w$", font_size=32).next_to(loss, UP, buff=LARGE_BUFF)
    eq_b = Tex("$b$", "$=b-\\delta_b$", font_size=32).next_to(eq, UP)
    with self.voiceover(text="""And we want to take<bookmark mark='1'/> the weights and biases of our network
                        and subtract the <bookmark mark='2'/> errors that make the loss higher""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(eq[0]), Write(eq_b[0]))
      self.wait_until_bookmark("2")
      self.play(Write(eq[1]), Write(eq_b[1]))
      
    eq2 = Tex("$\\delta_w$", "$=\\frac{\\partial L}{\\partial w}$", font_size=32).next_to(eq, RIGHT)
    eq2_b = Tex("$\\delta_b$", "$=\\frac{\\partial L}{\\partial b}$", font_size=32).next_to(eq_b, RIGHT)
    with self.voiceover(text="""And the error can be specified as the <bookmark mark='1'/> derivative of the loss with respect to our weight""") as trk:
      self.play(Write(eq2[0]), Write(eq2_b[0]))
      self.wait_until_bookmark("1")
      self.play(Write(eq2[1]), Write(eq2_b[1]))

    with self.voiceover(text="""Remember that the derivative tells us how changing our weight will influence our loss. Meaning that if our derivative is slightly positive, 
                        increasing our weight, will make the loss go up, and decreasing it will make the loss go down""") as trk:
      pass

    with self.voiceover(text="""Now there is a problem of how to calculate the derivative with respect to each weigh""") as trk:
      pass

    chain_rule = Tex("$\\frac{\\partial L}{\\partial w}$", "$=\\frac{\\partial L}{\\partial a}$", "$*\\frac{\\partial a}{\\partial w}$").move_to(eq, aligned_edge=LEFT)
    with self.voiceover(text="""And it turns out that it's not as hard as you might expect""") as trk:
      self.play(Unwrite(eq), Unwrite(eq2), Unwrite(eq_b), Unwrite(eq2_b))

    with self.voiceover(text="""There is an amazing mathematical tool that allows us to do this with simple functions, it's called the chain rule""") as trk:
      self.play(Write(chain_rule))

    with self.voiceover(text="""It essentially means that when we know the error in the last layer""") as trk:
      self.play(chain_rule[1].animate.set_color(BLUE))

    with self.voiceover(text="""We can calculate the error in the weights of the layer before it""") as trk:
      self.play(chain_rule[0].animate.set_color(RED))

    with self.voiceover(text="""By multiplying it's influence on the activations by that error""") as trk:
      self.play(chain_rule[2].animate.set_color(GREEN))

    
    d1 = Tex("$\\frac{\\partial L}{\\partial a^n}$").next_to(rect, UP, aligned_edge=LEFT)
    d2 = Tex("$\\frac{\\partial L}{\\partial x^n}$").next_to(nn.neurons[-1][0], UP)
    deltas = []
    for i in range(len(nn.neurons)):
      deltas.append(Tex(f"$\\frac{{\\partial L}}{{\\partial x^{i}}}$").next_to(nn.neurons[i][0], UP))


    with self.voiceover(text="""That is why we call this backpropagation, we start by calculating the error at the end, and we go backwards propagating the error through our network""") as trk:
      self.play(Unwrite(chain_rule))
      self.play(Write(d1[0]))
      self.play(*[ShowPassingFlash(e.copy().set_color(RED), reverse_rate_function=True) for e in lines])
      self.play(Write(d2))
      for i in range(len(nn.edges)-1, -1, -1):
        self.play(*[ShowPassingFlash(e.copy().set_color(RED), reverse_rate_function=True) for e in nn.edges[i]])
        self.play(Write(deltas[i]))
    self.wait(1)

    eq1 = MathTex("\\frac{\\partial L}{\\partial x^n}", "=\\frac{\\partial L}{\\partial a^n}", "\\frac{\\partial a^n}{\\partial x^n}", font_size=32).to_edge(UP, buff=0.05)

    with self.voiceover(text="""To formalize, our first backpropagation equation would be that the<bookmark mark='1'/> error in the last layer
                        is equal to the <bookmark mark='2'/> derivative of the loss with respect to the activations of the last layer, times the derivative of the activation of the last layer
                        <bookmark mark='3'/> with respect to the inputs of that layer""") as trk:
      self.wait_until_bookmark("1")
      self.play(Transform(d2, eq1[0], replace_mobject_with_object_in_scene=True))
      self.wait_until_bookmark("2")
      self.play(Transform(d1, eq1[1], replace_mobject_with_object_in_scene=True))
      self.wait_until_bookmark("3")
      self.play(Write(eq1[2]))

    anims = []
    for x in nn.neurons:
      anims.extend([Uncreate(n) for n in x])
    for x in nn.edges:
      anims.extend([Uncreate(n) for n in x])
    anims.extend([Uncreate(n) for n in lines])
    for d in deltas:
      anims.append(Unwrite(d))
    anims.append(Unwrite(loss))
    anims.append(Uncreate(rect))
    with self.voiceover(text="""Let's star by calculating the error in the last layer of our neural network""") as trk:
      self.play(*anims)

    with self.voiceover(text="""And I do have to warn you - this will be the hardest part of this episode, but bear with me
                        as it simplifies beutifully in the end""") as trk:
      pass

    softmax = MathTex(r"s = \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=0}^{K} e^{x_j}}", font_size=34).next_to(eq1.get_center(), DOWN, aligned_edge=RIGHT).shift(0.5*DOWN)

    cross_entropy = Tex("$H(p, s) = $","$-\\sum\\limits_{i=0}^{K}$", "$\\,p_i$", "$\\,\\log$", "$\\,s_i$", font_size=34).next_to(softmax, RIGHT)

    with self.voiceover(text="""As a remainder, our final activation layer was using the softmax function""") as trk:
      self.play(Write(softmax))

    with self.voiceover(text="""And our loss function was the cross entropy loss""") as trk:
      self.play(Write(cross_entropy))

    with self.voiceover(text="""Let's start by calculating the derivative of the softmax with respect to the inputs""") as trk:
      pass

    s_d = Matrix([["\\frac{\\partial s_1}{\\partial x_1}", "\\frac{\\partial s_1}{\\partial x_2}", "\\cdots", "\\frac{\\partial s_1}{\\partial x_K}"],
                 ["\\frac{\\partial s_2}{\\partial x_1}", "\\frac{\\partial s_2}{\\partial x_2}", "\\cdots", "\\frac{\\partial s_2}{\\partial x_K}"],
                 ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                 ["\\frac{\\partial s_K}{\\partial x_1}", "\\frac{\\partial s_K}{\\partial x_2}", "\\cdots", "\\frac{\\partial s_K}{\\partial x_K}"]],
                 element_alignment_corner=ORIGIN, element_to_mobject_config={"font_size": 24}, h_buff=1).next_to(softmax, DOWN)

    with self.voiceover(text="""Since each element of the softmax function depends on all the other elements in the input, we have to expect
                        the derivative to take the form of a matrix""") as trk:
      self.play(Create(s_d))

    d_eq = MathTex("\\frac{\\partial \\log(s_i)}{\\partial x_i}", "=", "\\frac{1}{s_i}", "\\frac{\\partial s_i}{\\partial x_i}", font_size=24).next_to(s_d, DOWN)


    with self.voiceover(text="""To simplify calculating the derivative, we'll consult our new best friend - the chain rule once again""") as trk:
      pass

    with self.voiceover(text="""Using something called the log derivative trick""") as trk:
      self.play(Write(d_eq))
    self.wait(1)

    d_eq2 = MathTex("s_i", "\\cdot", "\\frac{\\partial \\log(s_i)}{\\partial x_i}", "=", "\\frac{\\partial s_i}{\\partial x_i}", font_size=24).next_to(s_d, DOWN)
    with self.voiceover(text="""By rearranging the equation a little bit""") as trk:
      self.play(TransformMatchingTex(d_eq, d_eq2))
    self.wait(1)

    d_eq3 = MathTex("\\frac{\\partial s_i}{\\partial x_i}", "=", "s_i", "\\cdot", "\\frac{\\partial \\log(s_i)}{\\partial x_i}", font_size=24).next_to(s_d, DOWN)
    with self.voiceover(text="""We can transform our derivative of a softmax element into that element multiplied by a derivative of it's logarithm""") as trk:
      self.play(TransformMatchingTex(d_eq2, d_eq3))
    self.wait(1)

    fs = 24
    ls = MathTex("\\log(s_i) = ", "log(\\frac{e^{x_i}}{\sum_{j=0}^{K} e^{x_j}})", font_size=fs).next_to(s_d, aligned_edge=UP)
    ls2 = MathTex("\\log(s_i) = ", "log(e^{x_i})", " - log(\sum_{j=0}^{K} e^{x_j})", font_size=fs).next_to(s_d, aligned_edge=UP)
    ls3 = MathTex("\\log(s_i) = ", "x_i", " - log(\sum_{j=0}^{K} e^{x_j})", font_size=fs).next_to(s_d, aligned_edge=UP)

    with self.voiceover(text="""This is very helpful when calculating the derivative of softmax""") as trk:
      self.play(Write(ls))

    with self.voiceover(text="""as it helps us remove the division operator""") as trk:
      self.play(Transform(ls, ls2))
    self.wait(1)

    with self.voiceover(text="""as well as the exponent""") as trk:
      self.play(Transform(ls, ls3))
    self.wait(1)

    d_ls = MathTex("\\frac{\\partial \\log(s_i)} {\\partial x_k} = ", "\\frac{\\partial x_i}{\\partial x_k}", " - ", "\\frac{\\partial log(\sum_{j=0}^{K} e^{x_j})}{\\partial x_k}", font_size=fs).next_to(s_d, aligned_edge=UP)

    with self.voiceover(text="""If we were to take a derivative of this with respect to an arbitary input""") as trk:
      self.play(Transform(ls, d_ls, replace_mobject_with_target_in_scene=True))
    self.wait(1)

    t1 = MathTex(r"\begin{cases} 1 & \text{if } k = i \\ 0 & \text{if } k \neq i \end{cases}", font_size=fs, color=BLUE).next_to(d_ls, DOWN, aligned_edge=LEFT)
    with self.voiceover(text="""The left hand side would simplify to 1 if our element index matches the derivative index and 0 otherwise""") as trk:
      self.play(Write(t1), d_ls[1].animate.set_color(BLUE))

    self.wait(1)

    t2 = MathTex("\\frac{1}{\sum_{j=0}^{K} e^{x_j}}", " \\frac{\\partial \sum_{j=0}^{K} e^{x_j}}{\\partial x_k}", color=PURPLE, font_size=fs).next_to(t1, RIGHT)
    with self.voiceover(text="""as for the right hand side, we can go for the chain rule once again!""") as trk:
      self.play(Write(t2), d_ls[3].animate.set_color(PURPLE))
    
    t3 = MathTex("\\frac{1}{\sum_{j=0}^{K} e^{x_j}}", " \sum_{j=0}^{K} \\frac{\\partial e^{x_j}}{\\partial x_k}", color=PURPLE, font_size=fs).next_to(t1, RIGHT)
    with self.voiceover(text="""We can take the sum outside of our derivative""") as trk:
      self.play(Transform(t2, t3, replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""And notice that the right hand side of this, depends on x only for one element""") as trk:
      pass

    t4 = MathTex("\\frac{1}{\sum_{j=0}^{K} e^{x_j}}", " \\cdot e^{x_k}", color=PURPLE, font_size=fs).next_to(t1, RIGHT)
    with self.voiceover(text="""And the derivative of e^x over x is e^x""") as trk:
      self.play(Transform(t3, t4, replace_mobject_with_target_in_scene=True))

    t5 = MathTex("\\frac{e^{x_k}}{\sum_{j=0}^{K} e^{x_j}}", color=PURPLE, font_size=fs).next_to(t1, RIGHT)
    with self.voiceover(text="""If we move it into our fraction""") as trk:
      self.play(Transform(t4, t5, replace_mobject_with_target_in_scene=True))

    t6 = MathTex("\\frac{e^{x_k}}{\sum_{j=0}^{K} e^{x_j}}", "= s_k", color=PURPLE, font_size=fs).next_to(t1, RIGHT)
    with self.voiceover(text="""We can see that we arrived back into the softmax equation""") as trk:
      self.play(Transform(t5, t6, replace_mobject_with_target_in_scene=True))

    def monkeypatch(self, matrix):
      ret = []
      colormap = self.element_to_mobject_config["colormap"]
      del self.element_to_mobject_config["colormap"]
      for row in matrix:
        r = []
        for item in row:
          r.append(self.element_to_mobject(*item, **self.element_to_mobject_config))
          if len(r[-1].submobjects) > 2:
            for i, c in enumerate(colormap):
              r[-1][i].set_color(c)
        ret.append(r)
      return ret
    mtmm = Matrix._matrix_to_mob_matrix
    Matrix._matrix_to_mob_matrix = monkeypatch
    c = [GREEN, WHITE, BLUE, WHITE, PURPLE, WHITE]
    mat = []
    for i in range(4):
      row = []
      for j in range(4):
        if i == 2 and j == 2:
          elem = ["\\ddots"]
        elif i == 2:
          elem = ["\\vdots"]
        elif j == 2:
          elem = ["\\cdots"]
        else:
          elem = [f"s_{'n' if i == 3 else i+1}", "(", f"{1 if i == j else 0}", "-", f"s_{'n' if j == 3 else j+1}", ")"]
        row.append(elem)
      mat.append(row)

    s_d2 = Matrix(mat, element_to_mobject_config={"colormap":c, "font_size": 24}, element_alignment_corner=ORIGIN).move_to(s_d)
    with self.voiceover(text="""We can plug in all of the values and arrive with this matrix of softmax derivatives""") as trk:
      self.play(Transform(s_d, s_d2.move_to(s_d, aligned_edge=RIGHT), replace_mobject_with_target_in_scene=True), d_eq3[2].animate.set_color(GREEN))

    with self.voiceover(text="""I know that this has been a lot so pause the video and look over all the components
                        one more time to make sure that you understand where they come from""") as trk:
      pass

    Matrix._matrix_to_mob_matrix = mtmm
    d_L = Matrix([["\\frac{\\partial L}{\\partial s_1}", "\\frac{\\partial L}{\\partial s_2}", "\\vdots", "\\frac{\\partial L}{\\partial s_n}"]], element_to_mobject_config={"font_size":24}, element_alignment_corner=ORIGIN).next_to(s_d2, RIGHT)
    with self.voiceover(text="""To finish with our calculations we have to calculate the derivative of our loss with respect to the activations""") as trk:
      self.play(Unwrite(t6), Unwrite(d_ls), Unwrite(t1), Unwrite(d_eq3))
      self.play(Create(d_L))

    d_L2 = Matrix([["-\\frac{p_1}{s_1}"], ["-\\frac{p_2}{s_2}"], ["\\vdots"], ["-\\frac{p_n}{s_n}"]], element_to_mobject_config={"font_size":24}, element_alignment_corner=ORIGIN).next_to(s_d2, RIGHT)

    with self.voiceover(text="""And it's the derivative of log of our softmax output times our true probability""") as trk:
      self.play(Transform(d_L, d_L2, replace_mobject_with_target_in_scene=True))
    eq = MathTex("=").next_to(d_L2, RIGHT)
    self.play(Write(eq))

    with self.voiceover(text="""We can do a vector matrix multiplication to get the derivative of our loss with respect to the inputs""") as trk:
      d_LX = Matrix([["p_1 - s_1*(p_1+p_2+\\cdots+p_n)"], ["p_2 - s_2*(p_1+p_2+\\cdots+p_n)"], ["\\vdots"], ["p_n - s_n*(p_1+p_2+\\cdots+p_n)"]], element_to_mobject_config={"font_size":24}, element_alignment_corner=ORIGIN).next_to(eq, RIGHT)
    self.play(Create(d_LX))
    self.wait(1)
    with self.voiceover(text="""Now remember, p is a probability distribution""") as trk:
      pass

    d_LX2 = Matrix([["p_1 - s_1*(1)"], ["p_2 - s_2*(1)"], ["\\vdots"], ["p_n - s_n*(1)"]], element_to_mobject_config={"font_size":24}, element_alignment_corner=ORIGIN).next_to(eq, RIGHT)
    with self.voiceover(text="""So it's sum will always be equal to 1""") as trk:
      self.play(Transform(d_LX, d_LX2))
    self.wait(1)
    d_LX2 = Matrix([["p_1 - s_1"], ["p_2 - s_2"], ["\\vdots"], ["p_n - s_n"]], element_to_mobject_config={"font_size":24}, element_alignment_corner=ORIGIN).next_to(eq, RIGHT)
    with self.voiceover(text="""Simplifying even further, it truns out that the derivative of our loss with respect to the inputs
                        is simply just a vector subtraction of the true distribution and the softmaxed outputs""") as trk:
      self.play(Transform(d_LX, d_LX2))
    self.wait(1)

