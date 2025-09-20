import os
from manimlib import *
from math import radians

from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene
import moderngl

class Penny(VoiceoverScene):
    samples=4
    def construct(self):
        self.set_speech_service(
            RecorderService(transcription_model="base")
            # GTTSService()
            )
        self.frame.set_euler_angles(0., 0., 0.).set_shape(15.881941, 8.926617).move_to([ 0.07937918, -0.01939364,  0.        ])
        gpus = []
        chunks = []
        chunks_cp = []
        dirs = [UP, RIGHT, DOWN, LEFT]

        for i in range(4):
            s = Square(color=GREEN)
            t = Text(f"GPU {i}").move_to(s, UP).scale(0.8).shift(0.1*DOWN)
            gpu = Group(s, t).shift(3*dirs[i])
            gpus.append(gpu)
            c = []
            c2 = []
            for j in range(4):
                c2.append(Rectangle(height=0.25, width=1.5, color=GREY, stroke_width=1))
                s2 = Rectangle(height=0.25, width=1.5, depth_test=True)
                t2 = Text(f"{j*4 + i}").scale(0.35).move_to(s2)

                c.append(Group(s2, t2))
            chunks_cp.append(Group(*c2).arrange(DOWN, buff=0.03).move_to(gpu).shift(0.2*DOWN))
            chunks.append(Group(*c).arrange(DOWN, buff=0.03).move_to(gpu).shift(0.2*DOWN))




        gpus = Group(*gpus)
        arcs = []

        for i in range(4):
            arcs.append(
                    ArcBetweenPoints(
                        gpus[i-1].get_corner(dirs[i]),
                        gpus[i].get_corner(dirs[i-1]),
                        angle=-TAU/4
                        )
                    )

            
        for i in range(4):
            self.add(gpus[i])
            self.add(arcs[i])
            self.add(chunks[i])
            self.add(chunks_cp[i])
        # REDUCE 
        for j in range(3):
            anims = []
            for i in range(4):
                c = (i-j)%4
                n = (i+1)%4
                anims.append(chunks[i][c].animate.move_to(gpus[i].get_corner(dirs[n])))
            self.play(*anims)
            #arc
            anims = []
            for i in range(4):
                c = (i-j)%4
                n = (i+1)%4
                anims.append(MoveAlongPath(chunks[i][c], arcs[n]))
            self.play(*anims)
            #transform
            anims = []
            for i in range(4):
                c = (i-j)%4
                n = (i+1)%4
                new_text = f"{chunks[i][c].submobjects[1].text} + {chunks[n][c].submobjects[1].text}"
                anims.append(
                        AnimationGroup(
                            Transform(chunks[n][c].submobjects[1],
                                      Text(new_text).scale(0.35).move_to(chunks[n][c].submobjects[1])
                                      ),
                            Transform(chunks[i][c],
                                      Group(Text(new_text).scale(0.35).move_to(chunks[n][c].submobjects[1])),
                                      remover=True
                                      ),
                            )
                        )
                # somehow this doesn't get updated
                chunks[n][c].submobjects[1].text = new_text
            self.play(*anims)


        # BROADCAST
        for j in range(3):
            anims = []
            cif = []
            for i in range(4):
                c = (i+1-j)%4
                n = (i+1)%4
                cif.append(chunks[i][c].copy())
            for i in range(4):
                n = (i+1)%4
                anims.append(cif[i].animate.move_to(gpus[i].get_corner(dirs[n])))
            self.play(*anims)
            #arc
            anims = []
            for i in range(4):
                n = (i+1)%4
                anims.append(MoveAlongPath(cif[i], arcs[n]))
            self.play(*anims)
            #transform
            anims = []
            for i in range(4):
                c = (i+1-j)%4
                n = (i+1)%4
                anims.append(cif[i].animate.move_to(chunks_cp[n][c]))
            self.play(*anims)
            for i in range(4):
                c = (i+1-j)%4
                n = (i+1)%4
                chunks[n].submobjects[c] = cif[i]
            self.play(*anims)
