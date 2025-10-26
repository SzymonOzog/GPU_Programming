import os
from manimlib import *
from math import radians

from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene
import moderngl

class Penny(Scene):
    samples=4
    def construct(self):
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


class PennySimple(Scene):
    samples=4
    def construct(self):
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
                    angle=-TAU/4,
                    color=GREY
                )
            )

        for i in range(4):
            self.add(gpus[i])
            self.add(arcs[i])
            self.add(chunks[i])
            self.add(chunks_cp[i])

        # Move chunk
        for i in range(3):
            n = i+1
            self.play(chunks[i].animate.move_to(gpus[i].get_corner(dirs[n])))
            self.play(MoveAlongPath(chunks[i], arcs[n]))
            anims = []
            for c in range(4):
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

        self.wait()

        # Broadcast
        for i in range(3):
            n = i
            i = i-1
            cif = chunks[i].copy()
            self.play(cif.animate.move_to(gpus[i].get_corner(dirs[n])))
            self.play(MoveAlongPath(cif, arcs[n]))
            anims = []
            self.play(cif.animate.move_to(chunks_cp[n]))
            chunks[n] = cif
            self.play(*anims)

class PennyOneShot(Scene):
    samples=4
    def construct(self):
        self.frame.set_euler_angles(0., 0., 0.).set_shape(15.881941, 8.926617).move_to([ 0.07937918, -0.01939364,  0.        ])

        gpus = []
        chunks = []
        output = []
        symmem = []
        chunks_cp = []
        dirs = [UP, RIGHT, DOWN, LEFT]
        legend = Group(
                Text("Input"),
                Text("Output", fill_color = BLUE),
                Text("Symmetric Memory", fill_color=RED),
                )

        self.add(legend.arrange(DOWN).to_corner(UR))

        for i in range(4):
            s = Square(color=GREEN)
            t = Text(f"GPU {i}").move_to(s, UP).scale(0.8).shift(0.1*DOWN)
            gpu = Group(s, t).shift(3*dirs[i])
            gpus.append(gpu)

            c = []
            c2 = []
            s = []
            o = []
            for j in range(4):
                c2.append(Rectangle(height=0.25, width=0.25, color=WHITE, stroke_width=2))
                o.append(Rectangle(height=0.25, width=0.25, color=BLUE, stroke_width=2))
                s.append(Rectangle(height=0.25, width=1.25, color=RED, stroke_width=2))
                t2 = Text(f"{j*4 + i}").scale(0.35).move_to(c2[-1])
                c.append(Group(c2[-1], t2))
            # chunks_cp.append(Group(*c2).arrange(DOWN, buff=0.03).move_to(gpu).shift(0.2*DOWN))
            chunks.append(Group(*c).arrange(DOWN, buff=0.03))
            output.append(Group(*o).arrange(DOWN, buff=0.03))
            symmem.append(Group(*s).arrange(DOWN, buff=0.03))
            Group(chunks[-1], output[-1], symmem[-1]).arrange(RIGHT, buff=0.03).move_to(gpu).shift(0.2*DOWN)

        gpus = Group(*gpus)
        paths = []
        for i in range(4):
            paths.append([])
            for j in range(4):
                paths[i].append(None)
                if i == j:
                    continue
                paths[i][j] = Line(gpus[i], gpus[j])
                # paths.append(
                    # ArcBetweenPoints(
                    #     gpus[i-1].get_corner(dirs[i]),
                    #     gpus[i].get_corner(dirs[i-1]),
                    #     angle=-TAU/4,
                    #     color=GREY
                    # )
                # )

        for i in range(4):
            self.add(gpus[i])
            for j in range(4):
                if paths[i][j] != None:
                    self.add(paths[i][j])
            self.add(chunks[i])
            self.add(output[i])
            self.add(symmem[i])

        # Get from all other PEs
        all_chunks_cp = []
        anims = []
        for g in range(4):
            chunks_cp = []
            for i in range(3):
                n = i+int(i >= g)
                c = chunks[n].copy()
                chunks_cp.append(c)
                anims.append(c.animate.move_to(paths[n][g].get_start()))
            all_chunks_cp.append(chunks_cp)
        self.play(*anims)

        # stage 2
        anims = []
        for g in range(4):
            chunks_cp = all_chunks_cp[g]
            for i in range(3):
                n = i+int(i >= g)
                c = chunks_cp[i]
                anims.append(MoveAlongPath(c, paths[n][g]))
        self.play(*anims)

        # state 3
        anims = []
        for g in range(4):
            chunks_cp = all_chunks_cp[g]
            for i in range(3):
                n = i+int(i >= g)
                c = chunks_cp[i]
                anims.append(c.animate.move_to(symmem[g].get_left() + (i+1)*0.33*RIGHT))
        self.play(*anims)

        # sum
        anims = []
        for i in reversed(range(1,3)):
            anims = []
            for g in range(4):
                chunks_cp = all_chunks_cp[g]
                n = i-1
                for c in range(4):
                    new_text = str(int(chunks_cp[i][c].submobjects[1].text) + int(chunks_cp[n][c].submobjects[1].text))
                    anims.append(
                            AnimationGroup(
                                Transform(chunks_cp[n][c].submobjects[1],
                                          Text(new_text).scale(0.35).move_to(chunks_cp[n][c].submobjects[1])
                                          ),
                                Transform(chunks_cp[i][c],
                                          Group(Text(new_text).scale(0.35).move_to(chunks_cp[n][c].submobjects[1])),
                                          remover=True
                                          ),
                                )
                            )
                    # somehow this doesn't get updated
                    chunks_cp[n][c].submobjects[1].text = new_text
            self.play(*anims)

        #final merge
        anims = []
        for g in range(4):
            chunks_cp = all_chunks_cp[g]
            out_texts = []
            for c in range(4):
                new_text = str(int(chunks_cp[0][c].submobjects[1].text) + int(chunks[g][c].submobjects[1].text))
                print(new_text)
                out_texts.append(Text(new_text).scale(0.35).move_to(output[g][c]))
                anims.append(
                        AnimationGroup(
                            Transform(chunks_cp[0][c].submobjects[1].copy(),
                                      out_texts[-1]
                                      ),
                            Transform(chunks[g][c].submobjects[1].copy(),
                                      out_texts[-1].copy()
                                      ),
                            Transform(chunks_cp[0][c],
                                      Group(output[g][c].copy()),
                                      remover=True
                                      ),
                            )
                        )
        self.play(*anims)
