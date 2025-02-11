import os
from manimlib import *
from math import radians
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
from pathlib import Path
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene

class Talk(Scene):
    def construct(self):
        title = Text("HOW TO KEEP YOUR GPU HAPPY")
        self.play(Write(title))
        self.wait()
        self.play(*[FadeOut(x) for x in self.mobjects])
        
        cpu = Square(color=YELLOW).shift(2*LEFT)
        cpu_t = Text("CPU", color=GREEN).move_to(cpu)

        gpu = Square(color=GREEN).shift(2*RIGHT)
        gpu_t = Text("GPU", color=GREEN).move_to(gpu)
        connection = Line(cpu.get_corner(RIGHT), gpu.get_corner(LEFT), color=BLUE)
        
        #show creation
        self.play(ShowCreation(cpu), Write(cpu_t))
        self.play(ShowCreation(gpu), Write(gpu_t))
        self.play(ShowCreation(connection))
        self.wait()

        #show analogy
        self.play(*[x.animate.shift(2*UP) for x in [cpu, cpu_t, gpu, gpu_t, connection]])

        manager = Rectangle(color=YELLOW).shift(3*LEFT+2*DOWN)
        manager_t = Text("Manager", color=GREEN).move_to(manager)

        programmer = Rectangle(color=GREEN).shift(3*RIGHT+2*DOWN)
        programmer_t = Text("Programmer", color=GREEN).move_to(programmer)
        connection2 = Line(manager.get_corner(RIGHT), programmer.get_corner(LEFT), color=BLUE)

        self.play(ReplacementTransform(cpu.copy(), manager),
                  ReplacementTransform(cpu_t.copy(), manager_t),
                  ReplacementTransform(gpu.copy(), programmer),
                  ReplacementTransform(gpu_t.copy(), programmer_t),
                  ReplacementTransform(connection.copy(), connection2))
        self.wait()

        code = """
def iterate1(x):
\tfor elem in x:
\t\tdo_stuff()


def iterate2(x):
\t\tfor elem in x.tolist():
\t\tdo_stuff()
"""
        #show code
        code_obj = Code(code)
        self.play(*[FadeOut(x) for x in self.mobjects])
        self.play(ShowCreation(code_obj))
        time1 = 58.657
        time2 = 0.1449
        self.wait()

        #write results
        t1 = Text(f"{time1} ms").next_to(code_obj, RIGHT).shift(UP)
        t2 = Text(f"{time2} ms").next_to(code_obj, RIGHT).shift(DOWN)
        self.play(Write(t1))
        self.play(Write(t2))


        self.wait()
        #graph caching
        self.play(*[FadeOut(x) for x in self.mobjects])
        cpu = Rectangle(color=GOLD, width=16).shift(2*UP)
        cpu_t = Text("CPU Thread", color=GOLD).move_to(cpu)
        k1 = Rectangle(color=RED).shift(4*LEFT+DOWN)
        k2 = Rectangle(color=BLUE, width=2).next_to(k1, RIGHT, buff=1.5)
        k3 = Rectangle(color=GREEN, width=3).next_to(k2, RIGHT, buff=2)
        cpu_bot = cpu.get_corner(DOWN)

        l1 = Line(k1.get_corner(UL), cpu_bot+6.5*LEFT, color=RED)
        l2 = Line(k1.get_corner(UR), cpu_bot+1.55*LEFT, color=RED)

        l3 = Line(k2.get_corner(UL), cpu_bot+0.80*LEFT, color=BLUE)
        l4 = Line(k2.get_corner(UR), cpu_bot+1.85*RIGHT, color=BLUE)

        l5 = Line(k3.get_corner(UL), cpu_bot+3*RIGHT, color=GREEN)
        l6 = Line(k3.get_corner(UR), cpu_bot+7*RIGHT, color=GREEN)

        self.play(ShowCreation(cpu), Write(cpu_t))

        self.play(ShowCreation(k1), ShowCreation(k2), ShowCreation(k3),
                  ShowCreation(l1), ShowCreation(l2),
                  ShowCreation(l3), ShowCreation(l4),
                  ShowCreation(l5), ShowCreation(l6))

        self.wait()

        #show graph caching
        self.play(*[FadeOut(l) for l in [l1, l2, l3, l4, l5, l6]])
        k4 = Rectangle(color=RED).shift(3*LEFT+DOWN)
        k5 = Rectangle(color=BLUE, width=2).next_to(k4, RIGHT, buff=0.05)
        k6 = Rectangle(color=GREEN, width=3).next_to(k5, RIGHT, buff=0.05)
        self.play(Transform(k1, k4), Transform(k2, k5), Transform(k3, k6))
        graph_t = Text("CUDA graph").next_to(k2, DOWN)
        self.play(Write(graph_t))
        l1 = Line(k1.get_corner(UL), cpu_bot+5.3*LEFT, color=RED)
        l2 = Line(k3.get_corner(UR), cpu_bot+4.6*RIGHT, color=GREEN)
        self.play(ShowCreation(l1), ShowCreation(l2))

        self.wait()

        #show results
        self.play(*[FadeOut(x) for x in self.mobjects])
        gc_1 = Text("Graph caching", color=GREEN).shift(2*LEFT)
        gc_2 = Text("1534 tok/s", color=GREEN).next_to(gc_1, DOWN)

        nc_1 = Text("No graph caching", color=RED).shift(2*RIGHT)
        nc_2 = Text("677 tok/s", color=RED).next_to(nc_1, DOWN)

        self.play(*[Write(t) for t in [gc_1, gc_2, nc_1, nc_2]])

        self.wait()

        #shill myself
        self.play(*[FadeOut(x) for x in self.mobjects])
        yt_logo = SVGMobject("/Users/szymon.ozog/Downloads/youtube.svg").shift(UP+2*LEFT).set_color(RED)
        x_logo = SVGMobject("/Users/szymon.ozog/Downloads/X_logo_2023_original.svg").next_to(yt_logo, DOWN).set_color(WHITE)

        yt_user = Text("Simon Oz").next_to(yt_logo, RIGHT)
        x_user = Text("@SzymonOz").next_to(x_logo, RIGHT)
        self.play(ShowCreation(yt_logo), ShowCreation(x_logo), Write(yt_user), Write(x_user))







