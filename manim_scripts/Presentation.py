import os
from manimlib import *
from math import radians
import cv2

class Speed(Scene):
    def construct(self):

        title = Text("4090").scale(2).shift(3*UP)

        self.play(Write(title))
        self.wait()

        #write out specs
        opsp = Text("Operation speed").next_to(title, DOWN).shift(2*RIGHT)
        mem = Text("Memory Speed").next_to(title, DOWN).shift(2*LEFT)
        flops = Text("82.6 TFLOPS").next_to(opsp, DOWN)
        bandwidth = Text("1.01 TB/s").next_to(mem, DOWN)
        self.play(Write(opsp), Write(mem),
                Write(flops), Write(bandwidth))
        self.wait()

        #flops byte ratio
        breakdown = Tex("328\\frac{operations}{float}").next_to(title, DOWN).shift(2*DOWN)
        self.play(Write(breakdown))
        self.wait()

        #timings
        clock = Text("Clock speed = 1-1.8 GHz").next_to(title, DOWN)
        self.play(*[FadeOut(x) for x in [opsp, mem, flops, bandwidth, breakdown]])
        self.play(Write(clock))
        self.wait()

        timing = Tex("1\\,clock\\,cycle = \\frac{2}{3} ns").next_to(clock, DOWN)
        self.play(Write(timing))
        self.wait()

        distance = Tex("distance=100 \\,mm").next_to(timing, DOWN)
        self.play(Write(distance))
        self.wait()

