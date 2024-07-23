from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
from math import radians, degrees
import random
import numpy as np
from NN import NeuralNetworkMobject

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

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

    texts = [Tex(i, font_size=20).next_to(nn.neurons[-1][i], RIGHT, buff=SMALL_BUFF) for i in range(10)]

    with self.voiceover(text="""And an output layer with 10 output neurons corresponding to digits 0 to 9""") as trk:
      self.play(*[Create(n) for n in nn.neurons[-1]], *[Create(n) for n in nn.edges[-1]])
      b, l = nn.braces[-1]
      b.next_to(nn.neurons[-1][0], UP, buff=0.04)
      l.next_to(b, UP, buff=0.04)
      self.play(Create(b), Write(l))
      self.play(LaggedStart(*[Write(x) for x in texts]))


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
                        pause the video and try to come up with a solution on your own. Another note would be that this episode will not introduce any new concepts
                        in the realm of GPU programming - it's ment to reinforce what we've learned so far""") as trk:
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
    anims.extend([Unwrite(x) for x in texts])
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
    bias = Tex("$b_0$").next_to(plus, RIGHT).scale(s)
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

    with self.voiceover(text="""And usually we calculate our output for a batch of inputs, so we can stack our inputs in a matrix as well""") as trk:
      self.wait(3)
      self.play(Transform(v1, m1))

    self.wait(2)
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
    code_obj = Code(code=forward, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj.code = remove_invisible_chars(code_obj.code)

    with self.voiceover(text="""And the code that we'll use for this will be simillar to our matmul code from the
                        previous episode""") as trk:
      self.wait(1)
      self.play(Transform(VGroup(v1, v2, plus, bias), code_obj, replace_mobject_with_target_in_scene=True))

    hl = SurroundingRectangle(code_obj.code[0][22:], buff=0.03, stroke_width=2, fill_opacity=0.3)

    with self.voiceover(text="""This time, the matrix will not be square so we need to take 3 parameters for the shapes of our matrices""") as trk:
      self.play(Create(hl))

    hl_t = SurroundingRectangle(code_obj.code[3:5], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Like in the matmul code, each thread produces one output element""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[5], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""We check the bounary conditions before doing any reads or writes""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[7], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Then we initialize our output to the coresponding bias""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[8:12], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And we iterate over the weights and inputs and calculate the dot product""") as trk:
      self.play(Transform(hl, hl_t))
    self.wait(1)

    with self.voiceover(text="""In order for our neural network to work, our layers require an activation function,
                        otherwise they would just simply collaps to a one big linear layer""") as trk:
      pass

    axes = Axes(
        x_range=[-5, 5, 1],
        y_range=[-1, 5, 1],
    ).scale(0.8)
    relu_graph = axes.plot(lambda x: max(x, 0), color=BLUE)

    title = Text("ReLU Activation Function", font_size=40)
    title.to_edge(UP)

    formula = MathTex(r"\text{ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}", font_size=32).shift(2*LEFT+UP)

    self.wait(1)
    with self.voiceover(text="""And the activation that we will be using is the relu function""") as trk:
      self.play(*[FadeOut(x) for x in self.mobjects])
      self.play(Write(title))

    with self.voiceover(text="""It is a very simple concept - it returns 0 if x is lesser than 0, and x otherwise""") as trk:
      self.play(Write(formula))

    with self.voiceover(text="""This is what it looks like on a graph""") as trk:
      self.play(Create(axes))
      self.play(Create(relu_graph))

    self.wait(2)

    relu = """__global__ void relu(int w, int h, float* input, float* output)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float activation = input[row*w+column];
    output[row*w+column] =  activation > 0.f ? activation : 0.f;
  }
}"""

    code_obj = Code(code=relu, tab_width=2, language="c", font_size=24, line_no_buff=0.1, corner_radius=0.1)
    code_obj.code = remove_invisible_chars(code_obj.code)

    with self.voiceover(text="""Time to write the code!""") as trk:
      self.play(Unwrite(title), Unwrite(formula), Uncreate(axes), Uncreate(relu_graph))
      self.play(Create(code_obj))

    hl = SurroundingRectangle(code_obj.code[2:4], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Again, each thread calculates one element in the output matrix""") as trk:
      self.play(Create(hl))

    hl_t = SurroundingRectangle(code_obj.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""We need to perform a boundary check""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[6:8], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And we set our output to the activation if it was greater than 0 and 0 otherwise""") as trk:
      self.play(Transform(hl, hl_t))
    self.wait(1)

    with self.voiceover(text="""We will use the relu activation function for every layer in our network except for the last one""") as trk:
      pass

    formula = MathTex(
        r"\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{K} e^{x_j}}"
    )
    example = [-1, 2, 0.1, 0.6, 0.2, 0.1, 1.1, -1.2, 0, -1]
    conf = {"font_size": 36}

    
    func = Tex("$f($", "$x$", "$)$")
    chart = BarChart(values = example, bar_names = [f"{i}" for i in range(10)], x_axis_config=conf).scale(0.5).next_to(func, LEFT)

    
    with self.voiceover(text="""What we want our last layer to do, is to output the probability, of an image representing
                        a number from 0 to 9""") as trk:
      pass

    with self.voiceover(text="""Right now, our layer would output just numbers, that we would have to interpret ourselves""") as trk:
      self.play(Uncreate(hl), Uncreate(code_obj))
      self.play(Create(chart))

    self.wait(1)
    with self.voiceover(text="""We need to have some function""") as trk:
      self.play(Write(func))
    chart2 = BarChart(values = softmax(np.array(example)), bar_names = [f"p({i})" for i in range(10)], x_axis_config=conf).scale(0.5).next_to(func, RIGHT)

    with self.voiceover(text="""That will take in our final layer's output""") as trk:
      self.play(Transform(chart.copy(), func[1].copy().set_opacity(0), replace_mobject_with_target_in_scene=True))

    with self.voiceover(text="""And return a probability distribution""") as trk:
      self.play(Transform(func.copy().copy().set_opacity(0), chart2, replace_mobject_with_target_in_scene=True))
    self.wait(1)

    with self.voiceover(text="""We can achieve this result, by using a softmax function as our final activation""") as trk:
      self.play(FadeOut(chart), FadeOut(chart2))
      self.play(Transform(func, formula, replace_mobject_with_target_in_scene=True))

    formula2 = MathTex(
        r"\text{softmax}(x_i) = \frac{e^{x_i - max(x)}}{\sum_{j=1}^{K} e^{x_j - max(x)}}"
    )
    with self.voiceover(text="""Although there is one caveat, since it uses an exponential function, that grows - well exponentially
                        if our input vector will contain multiple positive values, it can overflow as we will add a lot of big numbers together
                        in our divisor""") as trk:
      pass
      

    with self.voiceover(text="""We can mitigate this by subtracting the maximum of our vector from the exponent. 
                        That way - the powers will always be negative, and our values will remain in range of 0 to 1""") as trk:
      self.play(Transform(formula, formula2))
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


    code_obj = Code(code=softmax_code, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj.code = remove_invisible_chars(code_obj.code)

    with self.voiceover(text="""This is the code for the kernel that will implement our softmax function""") as trk:
      self.wait(1)
      self.play(Transform(formula, code_obj, replace_mobject_with_target_in_scene=True))

    hl = SurroundingRectangle(code_obj.code[2:4], buff=0.03, stroke_width=2, fill_opacity=0.3)

    with self.voiceover(text="""We can see a simillar pattern for calculating one output element in a thread""") as trk:
      self.play(Create(hl))

    hl_t = SurroundingRectangle(code_obj.code[4], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And performing a boundary check""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[6], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""We then initialize our max value in the input vector to the first element of that vector""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[7:11], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And iterate over the rest of the values to find the maximum""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[11:16], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""Then we iterate over the values again, calculating the divisor""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[16], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And finally, we calculate the probability for our element and store it in the output array""") as trk:
      self.play(Transform(hl, hl_t))
    self.wait(1)

    
    title = Text("Loss").to_edge(UP)
    with self.voiceover(text="""The final component of our neural network will be the loss function""") as trk:
      self.play(Uncreate(code_obj), Uncreate(hl))
      self.wait(1)
      self.play(Write(title))

    with self.voiceover(text="""you can think of the loss function as the measure of how bad is our neural network -
                        the higher the loss function, the worse is the result of our outputs""") as trk:
      pass

    eq = Tex("$H(p, q) = $","$-\\sum\\limits_{x=0}^{n}$", "$\\,p(x)$", "$\\,\\log$", "$\\,q(x)$")
    with self.voiceover(text="""We will use a cross entropy loss function for our neural network""") as trk:
      self.play(Write(eq))
    real = Tex("$real \\; probability$", color=GREEN).next_to(eq[1], UP).shift(0.5*UP+RIGHT)
    l1 = real.get_center()
    l1[0] = eq[2].get_center()[0]
    a1 = Arrow(l1, eq[2].get_center(), stroke_width=3, max_tip_length_to_length_ratio=0.15, color=GREEN)

    with self.voiceover(text="""It takes 2 vectors, as the input""") as trk:
      pass

    with self.voiceover(text="""The vector of real probabilities for each class""") as trk:
      self.play(Create(real), Create(a1), eq[2].animate.set_color(GREEN))

    predicted = Tex("$predicted \\; probability$", color=BLUE).next_to(eq[3], DOWN).shift(0.5 * DOWN + LEFT)
    l2 = predicted.get_center()
    l2[0] = eq[4].get_center()[0]
    a2 = Arrow(l2, eq[4].get_center(), stroke_width=3, max_tip_length_to_length_ratio=0.15, color=BLUE)
    with self.voiceover(text="""And the vector of predictions of our neural network""") as trk:
      self.play(Create(predicted), Create(a2), eq[4].animate.set_color(BLUE))

    with self.voiceover(text="""It then iterates over all labels - so in our case the numbers from 0 to 9
                        and calculates the cross entropy between the truth and our prediction""") as trk:
      self.play(eq[1].animate.set_color(PURPLE))

    with self.voiceover(text="""Note that our real probabilities will have 1 entry that is equal to 1, indicating the real number
                        and the rest of the vector will be all zeros""") as trk:
      self.wait(5)
      self.play(VGroup(eq, real, a1, predicted, a2).animate.to_edge(LEFT))
    axes = Axes(
        x_range=[0, 1, 0.2],
        y_range=[0, 5, 1],
        axis_config={"include_tip": False, "include_numbers": True},
       ).scale(0.5).to_edge(RIGHT)

    x_label = axes.get_x_axis_label("q(x_{true})", direction=DOWN, edge=DOWN)
    y_text = Tex("$H=-\\log q(x_{true})$")
    y_label = axes.get_y_axis_label(y_text.rotate(PI/2), direction=LEFT, edge=LEFT)

    plot = axes.plot(
        lambda x: -np.log(x),
        color=BLUE,
        x_range=[0.01, 1]
        )

    with self.voiceover(text="""So our loss will collapse to just being the negative logarithm of the predicted probability for our true label""") as trk:
      self.play(Create(axes), Write(x_label), Write(y_label))

    self.wait(1)

    with self.voiceover(text="""When we graph it, we can see that it's getting higher the lower our predicted probability is 
                        for the number in the image, and it's 0 when the networks has correctly guessed the label""") as trk:
      self.play(Create(plot))
    self.wait(1)

    with self.voiceover(text="""If you want some more intuition on where this loss function comes from,
                        I invite you to watch my series on Information Theory""") as trk:
      pass

    self.wait(3)


    cross_entropy_code = """__global__ void cross_entropy(int w, int h, float* preds, float* real, float* output)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < h)
  {
    float loss = 0.f;
    for (int i = 0; i<w; i++)
    {
      loss -= real[idx*w + i] * log(max(1e-6, preds[idx*w + i]));
    }
    output[idx] = loss;
  }
}"""
    code_obj = Code(code=cross_entropy_code, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj.code = remove_invisible_chars(code_obj.code)

    with self.voiceover(text="""Without further ado, lets write the kernel for our loss""") as trk:
      self.play(Transform(VGroup(axes, x_label, y_label, plot, a2, a1, predicted, real, eq, title), code_obj, replace_mobject_with_target_in_scene=True))


    hl = SurroundingRectangle(code_obj.code[2], buff=0.03, stroke_width=2, fill_opacity=0.3)

    with self.voiceover(text="""This time we will reduce our input size to 1 dimension, so we only operate on 
                        the X axis in our kernel grid""") as trk:
      self.play(Create(hl))

    hl_t = SurroundingRectangle(code_obj.code[3], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""We do our standard boundary check""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[5:10], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""And we iterate over each number and calculate it's contribution to our loss""") as trk:
      self.play(Transform(hl, hl_t))

    with self.voiceover(text="""This part could have been parallelized even further, but that would require introducing atomic operations,
                        and I intend to keep my promise of not introducing anything new in this episode""") as trk:
      pass

    hl_t = SurroundingRectangle(code_obj.code[8][20:-1], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""One thing to note is the max function in the logarithm, we do the max of a very small number,
                        and our prediction for numerical stability - because the log function is undefined for 0""") as trk:
      self.play(Transform(hl, hl_t))

    hl_t = SurroundingRectangle(code_obj.code[10], buff=0.03, stroke_width=2, fill_opacity=0.3)
    with self.voiceover(text="""In the end, we save our result in the output vector""") as trk:
      self.play(Transform(hl, hl_t))

    self.wait(2)

    init_rand = """__global__ void init_rand(int w, int h, float* mat)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    curandState state;
    curand_init(42, row*w+column, 0, &state);
    mat[row*w + column] = curand_normal(&state)*sqrtf(2.f/h);
  }
}"""

    code_obj2 = Code(code=init_rand, tab_width=2, language="c", font_size=16, line_no_buff=0.1, corner_radius=0.1)
    code_obj2.code = remove_invisible_chars(code_obj2.code)

    with self.voiceover(text="""There is one last kernel that I want to mention""") as trk:
      self.play(Uncreate(hl))

    with self.voiceover(text="""Our weights have to actually be initialized to some value, we can also do that on the gpu
                        using the code on the screen""") as trk:
      self.play(Transform(code_obj, code_obj2))


    idx = 0
    with self.voiceover(text="""It's pretty straightforward, so I won't go into it in detail, one thing that might be
                        confusing is the sqrt function, this is just He initialization - I won't go over it as it's outside
                        of the scope for the series, but I'll leave a link in the description for those that want to read more
                        on the subject""") as trk:
      pass

    self.play(Uncreate(code_obj))

    with self.voiceover(text="""This will be the end for this episode, in the next one - we are going to implement the backward
                        pass and backpropagate our error to actually teach our neural network to predict the correct digit,
                        subscribe if you don't want to miss it - also please leave you feedback in the comments. It helps
                        me shape the future direction of the series. See you in the next episode - bye""") as trk: 
      nn = NeuralNetworkMobject([784, 300, 100, 10]).shift(0.1*DOWN)
      anims = []
      for x in nn.neurons:
        anims.extend([Create(n) for n in x])
      for x in nn.edges:
        anims.extend([Create(n) for n in x])
      self.play(*anims)
      while idx < 3:
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
      anims = []
      for x in nn.neurons:
        anims.extend([Uncreate(n) for n in x])
      for x in nn.edges:
        anims.extend([Uncreate(n) for n in x])
      self.play(*anims)
      self.wait(3)
