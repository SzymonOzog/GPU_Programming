import os
from manimlib import *
from math import radians

from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene
import moderngl

class Parallelism(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService(transcription_model="base")
            )
        shader_dir = os.path.dirname(os.path.abspath(__file__)) + "/shaders/one_sided"
        Square3D.shader_folder = shader_dir

        class FBlock(Group):
            def __init__(self, formula=None, *args, **kwargs):
                super().__init__()
                self.block = Prism(*args, **kwargs)
                
                self.add(self.block)
                if formula is not None:
                    self.t = Tex(formula).move_to(self.block.get_corner(OUT))
                    self.add(self.t)

            def create(self):
                return LaggedStart(ShowCreation(self.block), Write(self.t))

        attn = FBlock("softmax(\\frac{QK^T}{\\sqrt{d_k}})V", width = 8)
        norm = Cube().next_to(attn, UP)
        conn = Line(norm, attn)
        self.play(attn.create(), ShowCreation(norm), ShowCreation(conn))

