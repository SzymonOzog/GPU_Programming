import os
from manimlib import *
from math import radians

class MoE(Scene):
    def construct(self):

        vector2 = TexMatrix([["x_1"], ["x_2"], ["x_3"], ["x_4"], ["\\vdots"], ["x_n"]]).to_edge(LEFT)
        
        dot = Tex("\\cdot").next_to(vector2)
        
        mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,n}"],
                 ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,n}"],
                 ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                 ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,n}"]]).next_to(dot)
        eq = Tex("=").next_to(mat)
        vector3 = TexMatrix([["x_1"], ["x_2"], ["x_3"], ["x_4"], ["\\vdots"], ["x_m"]]).next_to(eq)

        self.play(ShowCreation(vector2))
        self.play(ShowCreation(mat), ShowCreation(dot))
        self.play(ShowCreation(vector3), ShowCreation(eq))

        #fade out signs
        self.play(FadeOut(eq), FadeOut(dot))


        #experts
        mats = [mat.copy().shift(2*x*IN + 16*OUT) for x in range(16)]
        bgs = []
        for i, m in enumerate(list(reversed(mats))):
            m.set_z_index(i*2 + 1)
            bg = Rectangle(m.get_width(), m.get_height(), color=BLACK, fill_color=BLACK, fill_opacity=1).move_to(m).shift(0.1*IN).set_z_index(i*2)
            bgs.append(bg)

        self.play(*[ShowCreation(m) for m in mats + bgs])

        #asdj
        self.play(self.frame.animate.move_to([-0.56036127, 0.8495176, 1.0232906]).set_euler_angles(-3.14159265,  0.26179939,  3.11379317).set_shape(53.023605, 29.802404))
        self.play(vector2.animate.shift(4*LEFT), FadeOut(vector3))
        vector3.shift(4*RIGHT)
        
        #show mapping
        active = [0, 4, 8, 15]
        lines1 = []
        lines2 = []
        outputs = []
        anims = []
        for a in active:
            lines1.append(Line(vector2.get_corner(RIGHT), mats[a].get_corner(LEFT)))
            outputs.append(vector3.copy().set_z(mats[a].get_z()))
            lines2.append(Line(mats[a].get_corner(RIGHT), outputs[-1].get_corner(LEFT)))
            anims.append(mats[a].animate.set_color(YELLOW))
        self.play(*[ShowCreation(x) for x in lines1], *anims)
        self.play(*[ShowCreation(x) for x in lines2 + outputs])

        

