from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
from math import radians, degrees
import random
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
        GTTSService()
        # RecorderService(trim_buffer_end=50, trim_silence_threshold=-80, transcription_model=None)
        )

    title = Text("GPU programming", font_size=72).shift(2*UP)
    with self.voiceover(text="Hello and welcome to episode 3 in the series on GPU programming") as trk:
      self.play(Write(title))

    subtitle = Text("Simple Neural Network", font_size=48).next_to(title, DOWN)

    with self.voiceover(text="""In this episode, we are going to create a simple neural network for recognizing
                        hand written digits using just CUDA and the knowledge that we gained from previous episodes in the series""") as trk:
      self.play(Write(subtitle))

    nn = NeuralNetworkMobject([784, 300, 100, 10]).shift(0.1*DOWN)

    img, _, _ = create_mnist(0)

    with self.voiceover(text="""The data that we will be using for training our Neural Network will be
                        the mnist dataset - it contains grayscaled images of handwrittend digits of size 28 by 28""") as trk:
      self.play(Unwrite(title), Unwrite(subtitle))
      self.play(*[Create(x) for x in img])

    anims = []
    j = 0
    rest = VGroup()
    for i, x in enumerate(img):
      if i < 10 or i > 773:
        anims.append(Transform(x, nn.neurons[0][j], replace_mobject_with_target_in_scene=True))
        j+=1
      else:
        j=11
        rest.add(x)
    anims.insert(10, Transform(rest, nn.neurons[0][10], replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""And the neural network architecture that we'll use will be a multilayer
                        perceptron, where the first layer will be the input layer getting 784 pixel values of our image""") as trk:
      self.wait(3)
      self.play(*anims)
      b, l = nn.braces[0]
      b.next_to(nn.neurons[0][0], UP, buff=0.04)
      l.next_to(b, UP, buff=0.04)
      self.play(Create(b), Write(l))
    self.wait(1)
    with self.voiceover(text="""It will have 2 hidden layers - one of size 300 and one of size 100""") as trk:
      for i in range(2):
        self.play(*[Create(n) for n in nn.neurons[1+i]], *[Create(n) for n in nn.edges[i]])
        b, l = nn.braces[1+i]
        b.next_to(nn.neurons[1+i][0], UP, buff=0.04)
        l.next_to(b, UP, buff=0.04)
        self.play(Create(b), Write(l))
        self.wait(0.5)

    with self.voiceover(text="""And an output layer with 10 output neurons corresponding to digits 0 to 9""") as trk:
      self.play(*[Create(n) for n in nn.neurons[-1]], *[Create(n) for n in nn.edges[-1]])
      b, l = nn.braces[-1]
      b.next_to(nn.neurons[-1][0], UP, buff=0.04)
      l.next_to(b, UP, buff=0.04)
      self.play(Create(b), Write(l))


    with self.voiceover(text="""As I've mentioned in the introduction to the series, I'm going to do some explanations on how neural networks work
                        but it will not be a very detailed one - so for those that are unfamilliar with Neural Networks at all
                        I'll leave a link in the desctiption for the series from 3 blue one brown where he goes in depth on this exact problem""") as trk:
      idx = 0
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
        self.play(Indicate(nn.neurons[-1][int(label)], color=BLUE))

    with self.voiceover(text="""I also have to mention that you should be able to code all of the functionality yourself - and I highly encourage you to try to 
                        pause the video and try to come up with a solution on your own""") as trk:
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
        self.play(Indicate(nn.neurons[-1][int(label)], color=BLUE))

    self.wait(1)

    anims = []
    for x in nn.neurons[1:]:
      anims.extend([Uncreate(n) for n in x if n != nn.neurons[1][0]])
    for x in nn.edges:
      anims.extend([Uncreate(n) for n in x if n not in nn.neuron_to_input[nn.neurons[1][0]]])
    for b,l in nn.braces:
      anims.append(Unwrite(l))
      anims.append(Uncreate(b))
    with self.voiceover(text="""The first kernel that we will be writing, is the forward pass for our neural network layer""") as trk:
      self.wait(2)
      self.play(anims)

    t1 = Tex("$x^1_0$=", "$x^0_0\\cdot$", "$w^0_{0,0}$+", "$x^0_1\\cdot$", "$w^0_{0,1}$+", "$x^0_2\\cdot$", "$w^0_{0,2}$+",
             "$...+$", "$x^0_n\\cdot$", "$w^0_{0,n}$+", "$b^0_0$", font_size=32).next_to(nn.neurons[1][0], RIGHT)
    with self.voiceover(text="""To calculate the output of a single neuron""") as trk:
      self.play(Transform(nn.neurons[1][0].copy() , t1[0], replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""We need to take the first input""") as trk:
      self.play(Transform(nn.neurons[0][0].copy(), t1[1], replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""And multiply it with a corresponding weight""") as trk:
      self.play(Transform(nn.neuron_to_input[nn.neurons[1][0]][0].copy(), t1[2], replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""We then do that for every other input and sum the results together""") as trk:
      for i in range(1, 3):
        self.play(Transform(nn.neurons[0][i].copy(), t1[(i*2)+1], replace_mobject_with_target_in_scene=True))
        self.play(Transform(nn.neuron_to_input[nn.neurons[1][0]][i].copy(), t1[(i*2)+2], replace_mobject_with_target_in_scene=True))
      rest = VGroup()
      for i in range(3, 20):
        rest.add(nn.neurons[0][i].copy())
        if i < 19:
          rest.add(nn.neuron_to_input[nn.neurons[1][0]][i].copy())
      self.play(Transform(rest, t1[7], replace_mobject_with_target_in_scene=True)) 
      self.play(Transform(nn.neurons[0][20].copy(), t1[8], replace_mobject_with_target_in_scene=True))
      self.play(Transform(nn.neuron_to_input[nn.neurons[1][0]][19].copy(), t1[9], replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""Adding a single bias term at the end""") as trk:
      self.play(Write(t1[-1]))

    self.wait(2)

    s = 0.8
    v1 = Matrix([["x_0", "x_1", "...", "x_n"]], element_alignment_corner=ORIGIN).scale(s).shift(3*LEFT)
    v2 = Matrix([["w_0"], ["w_1"], ["\\vdots"], ["w_n"]], element_alignment_corner=ORIGIN).scale(s).next_to(v1, RIGHT)
    plus = Tex("+").next_to(v2, RIGHT).scale(s)
    bias = Tex("b_0").next_to(plus, RIGHT).scale(s)
    with self.voiceover(text="""You might have noticed that the operation is just a dot product between the input and the weights so 
                        we can express it like this""") as trk:
      self.play(Unwrite(t1), Uncreate(nn.neurons[1][0]))
      self.play(Transform(VGroup(*[x for x in nn.neurons[0]]), v1, replace_mobject_with_target_in_scene=True),
                Transform(VGroup(*[x for x in nn.neuron_to_input[nn.neurons[1][0]]]), v2, replace_mobject_with_target_in_scene=True),
                Write(plus), Write(bias))


    m2 = Matrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,n}"],
                 ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,n}"],
                 ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                 ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,n}"]], element_alignment_corner=ORIGIN).scale(s).next_to(v1, RIGHT)
    plus2 = Tex("+").next_to(m2, RIGHT)
    b = Matrix([["b_0"], ["b_1"], ["\\vdots"], ["b_n"]], element_alignment_corner=ORIGIN).scale(s).next_to(plus2, RIGHT)
    with self.voiceover(text="""Going even further, we can stack the weights into a matrix, and biases to a vector to get an equation, for the output
                        of every neuron - not just one""") as trk:
      self.wait(2)
      self.play(Transform(v2, m2),
                Transform(plus, plus2),
                Transform(bias, b))

    m1 = Matrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,m}"],
                 ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,m}"],
                 ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                 ["x_{b,0}", "x_{b,1}", "\\cdots", "x_{b,m}"]], element_alignment_corner=ORIGIN).scale(s).move_to(v1)

    with self.voiceover(text="""And usually we calculate our output for multiple inputs, so we can stack our inputs in a matrix as well""") as trk:
      self.wait(3)
      self.play(Transform(v1, m1))


