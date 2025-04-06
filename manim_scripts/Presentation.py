import os
from manimlib import *
from math import radians
import cv2

class Speed(Scene):
    def construct(self):
        def play_vid(path, fps = 30):
            vid = cv2.VideoCapture(path)
            flag = True
            while flag:
                flag, frame = vid.read()
                if flag:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_img = ImageMobject(frame)
                    self.add(frame_img)
                    self.wait(1/fps)
                    self.remove(frame_img)
            cap.release()

        title = Text("H100").scale(2).shift(3*UP)

        self.play(Write(title))

        #write out specs
        op = Text("Operation speed").next_to(title, DOWN).shift(2*RIGHT)
        mem = Text("Memory Speed").next_to(title, DOWN).shift(2*LEFT)
        flops = Text("51.2 TFLOPS").next_to(op, DOWN)
        bandwidth = Text("2.04 TB/s").next_to(mem, DOWN)
        self.play(Write(op), Write(mem),
                Write(flops), Write(bandwidth))

        #flops byte ratio
        breakdown = Tex("100\\frac{operations}{float}").next_to(title, DOWN).shift(2*DOWN)
        self.play(Write(breakdown))
