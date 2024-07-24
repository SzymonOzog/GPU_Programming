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
    nn = NeuralNetworkMobject([784, 300, 100, 10]).shift(0.1*DOWN)
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

    
    eq = Tex("$w_{0,0}$", "$-=\\delta$", font_size=32).next_to(loss, UP, buff=LARGE_BUFF)
    with self.voiceover(text="""And we want to take<bookmark mark='1'/> a weight of our network
                        and subtract the <bookmark mark='2'/> error in this particular weight""") as trk:
      self.wait_until_bookmark("1")
      self.play(Write(eq[0]))
      self.wait_until_bookmark("2")
      self.play(Write(eq[1]))
      
    eq2 = Tex("$\\delta$", "$=\\frac{\\partial L}{\\partial w_{0,0}}$", font_size=32).next_to(eq, UP)
    with self.voiceover(text="""And the error can be specified as the <bookmark mark='1'/> derivative of the loss with respect to our weight""") as trk:
      self.play(Write(eq2[0]))
      self.wait_until_bookmark("1")
      self.play(Write(eq2[1]))

    with self.voiceover(text="""Remember that the derivative tells us how changing our weight will influence our loss. Meaning that if our derivative is slightly positive, 
                        increasing our weight, will make the loss go up, and decreasing it will make the loss go down""") as trk:
      pass

