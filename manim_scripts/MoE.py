import os
from manimlib import *
from math import radians
#TODO why do I have to do this
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene

class MoE(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService(transcription_model="base")
            )

        vector2 = TexMatrix([["x_1"], ["x_2"], ["x_3"], ["x_4"], ["\\vdots"], ["x_n"]]).to_edge(LEFT)
        
        dot = Tex("\\cdot").next_to(vector2)
        
        mat = TexMatrix([["w_{0,0}", "w_{0,1}", "\\cdots", "w_{0,n}"],
                 ["w_{1,0}", "w_{1,1}", "\\cdots", "w_{1,n}"],
                 ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                 ["w_{m,0}", "w_{m,1}", "\\cdots", "w_{m,n}"]]).next_to(dot)
        eq = Tex("=").next_to(mat)
        vector3 = TexMatrix([["x_1"], ["x_2"], ["x_3"], ["x_4"], ["\\vdots"], ["x_m"]]).next_to(eq)
        
        with self.voiceover(text="""While in a classic tensor parallelism, we had each input tensor multiplied by a matrix""") as trk:
            self.play(ShowCreation(vector2))
            self.play(ShowCreation(mat), ShowCreation(dot))
            self.play(ShowCreation(vector3), ShowCreation(eq))

        #fade out signs
        with self.voiceover(text="""Scaling our models out started having a lot of issues, as this matrix would be big, which would
                            yield a very slow calculation""") as trk:
            self.play(FadeOut(eq), FadeOut(dot))


        #experts
        mats = [mat.copy().shift(2*x*IN + 16*OUT) for x in range(16)]
        bgs = []
        for i, m in enumerate(list(reversed(mats))):
            m.set_z_index(i*2 + 1)
            bg = Rectangle(m.get_width(), m.get_height(), color=BLACK, fill_color=BLACK, fill_opacity=1).move_to(m).shift(0.1*IN).set_z_index(i*2)
            bgs.append(bg)

        with self.voiceover(text="""One solution that caught across the industry became mixture of experts models, with them 
                           <bookmark mark='1'/> instead of having bigger matrices, we would just have more matrices""") as trk:
            self.wait_until_bookmark("1")
            self.play(self.frame.animate.move_to([-0.56036127, 0.8495176, 1.0232906]).set_euler_angles(-3.14159265,  0.26179939,  3.11379317).set_shape(53.023605, 29.802404))
            self.play(vector2.animate.shift(4*LEFT), FadeOut(vector3))
            self.play(*[ShowCreation(m) for m in mats + bgs])

        vector3.shift(4*RIGHT)
        
        #show mapping
        active = [0, 4, 8, 15]
        lines1 = []
        lines2 = []
        outputs = []
        anims = []
        for a in active:
            lines1.append(Line3D(vector2.get_corner(RIGHT), mats[a].get_corner(LEFT), width=0.1))
            outputs.append(vector3.copy().set_z(mats[a].get_z()))
            lines2.append(Line3D(mats[a].get_corner(RIGHT), outputs[-1].get_corner(LEFT), width=0.1))
            anims.append(mats[a].animate.set_color(YELLOW))
        with self.voiceover(text="""Here based on a result of a routing function, we would assign our input vector to a set of experts""") as trk:
            self.play(*[ShowCreation(x) for x in lines1], *anims)
            self.play(*[ShowCreation(x) for x in lines2 + outputs])
        self.wait()

        #combine
        lines3 = []
        final = vector3.copy().next_to(Group(*outputs), RIGHT).shift(4*RIGHT)
        for x in outputs:
            lines3.append(Line3D(x.get_corner(RIGHT), final.get_corner(LEFT), width=0.1))
        
        with self.voiceover(text="""After the matrix multiplication the results are combined together""") as trk:
            self.play(*[ShowCreation(x) for x in lines3])
            self.play(ShowCreation(final))

        self.play(*[FadeOut(x) for x in self.mobjects])

        return
        # more tokens
        toks = [vector2.copy().shift(2*x*IN + 16*OUT) for x in range(16)]
        bgs2 = []
        for i, m in enumerate(list(reversed(toks))):
            m.set_z_index(i*2 + 1)
            bg = Rectangle(m.get_width(), m.get_height(), color=BLACK, fill_color=BLACK, fill_opacity=1).move_to(m).shift(0.1*IN).set_z_index(i*2)
            bgs2.append(bg)

        self.play(*[ShowCreation(x) for x in toks + bgs2])
        self.wait()

        #create more mappings
        lines3 = []
        for t in toks:
            for _ in range(4):
                a = random.randint(0, 15)
                lines3.append(Line(t.get_corner(RIGHT), mats[a].get_corner(LEFT), z_index=100))
        self.play(*[ShowCreation(x) for x in lines3])
        self.wait()


        # show isolated kernel
        self.play(*[FadeOut(x) for x in lines3 + lines1 + lines2 + outputs])

        self.play(mats[0].animate.shift(10*UP))
        mat2 = TexMatrix([["x_{0,0}", "x_{0,1}", "\\cdots", "x_{0,n}"],
                 ["x_{1,0}", "x_{1,1}", "\\cdots", "x_{1,n}"],
                 ["\\vdots", "\\vdots", "\\ddots", "\\vdots"],
                 ["x_{m,0}", "x_{m,1}", "\\cdots", "x_{m,n}"]]).next_to(mats[0], LEFT).shift(LEFT)
        lines4 = []
        for tok in toks[:4]:
            lines4.append(Line(mat2.get_corner(DOWN), tok.get_corner(UP)))
        self.play(*[ReplacementTransform(t.copy(), x) for x, t in zip(lines4, toks[:4])])
        self.play(ShowCreation(mat2))

