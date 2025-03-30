import os
from manimlib import *
from math import radians

class Quantization(Scene):
    def construct(self):
        self.play(*[FadeOut(x) for x in self.mobjects])
        #start
        ax = Axes(x_range=(-70000, 70000, 10000),
                  y_range=(-0.1, 1.1, 0.1),
                  height=6, 
                  width=10
                  )
        self.play(ShowCreation(ax))

        #create rect
        w = (ax.c2p(65504, 0) - ax.c2p(-65504, 0))[0]
        h = (ax.c2p(0, 1.1) - ax.c2p(0, 0))[1]
        fp16_box = Rectangle(w, h, color=GREEN)
        print(w,h)
        fp16_text = Text("FP16 range (-65504, 65504)").next_to(fp16_box, UP)
        self.play(ShowCreation(fp16_box), Write(fp16_text))

        #create q4
        w = (ax.c2p(65504, 0) - ax.c2p(-65504, 0))[0]
        h = (ax.c2p(0, 0.6) - ax.c2p(0, 0))[1]
        q4_box = Rectangle(w, h, color=RED).shift(DOWN)
        lines = []
        start_point = q4_box.get_corner(UL)
        end_point = q4_box.get_corner(DL)
        for i in range(15):
            start_point[0] += w/16
            end_point[0] += w/16
            lines.append(Line(start_point, end_point, color=RED, stroke_width=2))

        q4_text = Text("Q4 representation").next_to(q4_box, UP)
        self.play(ShowCreation(q4_box), Write(q4_text))
        self.play(LaggedStart(*[ShowCreation(x) for x in lines]))
        q4 = VGroup(q4_box, *lines)

        #create values
        values = []
        rng = 20000
        start_point = ax.c2p(-rng, 0.3) 
        end_point = ax.c2p(-rng, 0) 
        w = (ax.c2p(rng, 0) - ax.c2p(-rng, 0))[0]
        for i in range(24):
            start_point[0] += w/16
            end_point[0] += w/16
            values.append(Line(start_point, end_point, color=ORANGE, stroke_width=4, z_index=5))
        self.play(*[ShowCreation(x) for x in values[:16]])

        #show scale
        self.play(q4.animate.stretch_to_fit_width(w))

        # not centared around 0
        self.play(*[Uncreate(x) for x in values[:8]])

        #more values
        self.play(*[ShowCreation(x) for x in values[16:]])

        #show zero point
        self.play(q4.animate.shift(w/2 * RIGHT))
