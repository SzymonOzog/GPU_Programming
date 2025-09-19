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
        gpus = []
        dirs = [UP, RIGHT, DOWN, LEFT]

        for i in range(4):
            s = Square(color=GREEN)
            t = Text(f"GPU {i}").next_to(s, UP)
            gpu = Group(s, t).shift(3*dirs[i])
            gpus.append(gpu)


        gpus = Group(*gpus)
        arcs = []

        for i in range(4):
            arcs.append(
                    ArcBetweenPoints(
                        gpus[i].get_corner(dirs[i-1]),
                        gpus[i-1].get_corner(dirs[i])
                        )
                    )

            
        for i in range(4):
            self.add(gpus[i])
            self.add(arcs[i])
        self.wait()


