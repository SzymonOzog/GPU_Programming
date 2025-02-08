import os
from manimlib import *
from math import radians
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene

class Talk(Scene):
    def construct(self):
        
        cpu = Square(color=YELLOW).shift(2*LEFT)
        cpu_t = Text("CPU", color=GREEN).move_to(cpu)

        gpu = Square(color=GREEN).shift(2*RIGHT)
        gpu_t = Text("GPU", color=GREEN).move_to(gpu)
        connection = Line(cpu.get_corner(RIGHT), gpu.get_corner(LEFT), color=BLUE)
        
        #show creation
        self.play(ShowCreation(cpu), Write(cpu_t))
        self.play(ShowCreation(gpu), Write(gpu_t))
        self.play(ShowCreation(connection))

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

        code = """
def iterate1(x):
\tfor elem in x:
\t\tif elem > 999:
\t\t\t\tprint("never called")

def iterate2(x):
\t\tfor elem in x.tolist():
\t\tif elem > 999:
\t\t\t\tprint("never called")
"""
        #show code
        code_obj = Code(code)
        self.play(*[FadeOut(x) for x in self.mobjects])
        self.play(ShowCreation(code_obj))
        time1 = 30.9233
        time2 = 0.0777

        #write results
        t1 = Text(f"{time1} ms").next_to(code_obj, RIGHT).shift(UP)
        t2 = Text(f"{time2} ms").next_to(code_obj, RIGHT).shift(DOWN)
        self.play(Write(t1))
        self.play(Write(t2))



