import os
from manimlib import *
from math import radians

class Quantization(Scene):
    def construct(self):
        t = Text("QUANTIZATION").scale(3)
        self.play(Write(t))
        self.wait()

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
        self.wait()

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
        self.wait()

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
        self.wait()

        #show scale
        self.play(q4.animate.stretch_to_fit_width(w))
        self.wait()

        # not centared around 0
        self.play(*[Uncreate(x) for x in values[:8]])
        self.wait()

        #more values
        self.play(*[ShowCreation(x) for x in values[16:]])
        self.wait()

        #show zero point
        self.play(q4.animate.shift(w/2 * RIGHT))
        self.wait()

        #Show quantized block
        self.play(*[FadeOut(x) for x in self.mobjects])
        block = Rectangle(width=8, height=2, color=BLUE)
        self.play(ShowCreation(block))
        scale_global = Text("FP16 scale & FP16 shift", color=BLUE).next_to(block, UP)
        self.play(Write(scale_global))
        self.wait()

        start_point = block.get_corner(UL)
        end_point = block.get_corner(DL)
        w = 8
        lines=[]
        scale_blocks = [] 
        for i in range(8):
            start_point[0] += w/8
            end_point[0] += w/8
            lines.append(Line(start_point, end_point, color=BLUE))
            scale_blocks.append(Text("6Bit scale\n6Bit shift").scale(0.35).move_to(end_point + 0.5*LEFT + 0.5*DOWN))

        self.play(LaggedStart(*[ShowCreation(x) for x in lines]))
        self.wait()

        self.play(LaggedStart(*[Write(x) for x in scale_blocks]))
        self.wait()

        #summarize
        x = Text("32 bits").next_to(scale_global).shift(2*RIGHT)
        self.play(Write(x))
        self.wait()
        y = Text("256x4bits").next_to(lines[-1], RIGHT)
        center_x = x.get_center()[0]
        y_loc = y.get_center().copy()
        y_loc[0] = center_x
        y.move_to(y_loc)
        self.play(Write(y))
        self.wait()
        z = Text("8x12bits").next_to(scale_blocks[-1], RIGHT)
        z_loc = z.get_center().copy()
        z_loc[0] = center_x
        z.move_to(z_loc)
        self.play(Write(z))
        self.wait()
        equation = [Text("+").next_to(x, DOWN).shift(0.2*DOWN),
                    Text("+").next_to(y, DOWN).shift(0.2*DOWN),
                    Text("=").next_to(z, DOWN).shift(0.2*DOWN)
                    ]
        self.play(*[Write(t) for t in equation])

        result = Text(f"{32+(4*256)+(12*8)} bits").next_to(equation[-1], DOWN)
        self.play(Write(result))
        self.wait()
