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
                else:
                    self.t = None

            def create(self):
                anims = [ShowCreation(self.block)]
                if self.t:
                    anims.append(Write(self.t))
                return LaggedStart(*anims)

        class TransformerBlock(Group):
            def __init__(self):
                super().__init__()
                self.attn = FBlock("softmax(\\frac{QK^T}{\\sqrt{d_k}})V", width=8, opacity=0.5)
                self.add(self.attn)

                self.rope = FBlock("RoPE(x)", width=8)
                self.add(self.rope)

                self.arrange(UP, buff=1)

                lines = []
                for x1, x2 in zip(self.submobjects, self.submobjects[1:]):
                    l = Line(x1, x2)
                    lines.append(l)
                i = len(self.submobjects) - 1
                for l in reversed(lines):
                    self.insert_submobject(i, l)
                    i -= 2

                self.high_level = FBlock("Transformer Block", width = self.get_width() * 1.05, height = self.get_height() * 1.05)

            def extend_at(self, obj, dist=2):
                idx = self.submobjects.index(obj)
                anims = []
                for i, smo in enumerate(self):
                    if i <= idx:
                        anims.append(smo.animate.shift(dist*DOWN))
                    elif i == idx + 1:
                        assert isinstance(smo, Line)
                        anims.append(Transform(smo, Line(smo.get_bottom() + dist*DOWN, smo.get_top() + dist*UP)))
                    else:
                        anims.append(smo.animate.shift(dist*UP))
                return AnimationGroup(*anims)


            def create(self, high_level=True):
                self.is_hl = high_level
                if high_level:
                    anims = []
                    for obj in self:
                        if isinstance(obj, FBlock):
                            anims.append(obj.create())
                        else:
                            anims.append(ShowCreation(obj))
                    return LaggedStart(*anims)
                else:
                    return self.high_level.create()

            def transform(self):
                if self.is_hl:
                    ret = AnimationGroup(FadeOut(self), self.high_level.create())
                else:
                    ret = AnimationGroup(FadeOut(self.high_level), self.create())
                self.is_hl = not self.is_hl
                return ret

        t = TransformerBlock()
        self.play(t.create())
        self.play(t.transform())
        self.play(t.transform())
        self.play(t.extend_at(t.attn))
