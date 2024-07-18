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
