from manim import *
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService

class EndScreen (VoiceoverScene, ZoomedScene):
    def construct(self):
        self.set_speech_service(
                GTTSService(transcription_model="base")
                )
        bmac = Text("https://buymeacoffee.com/simonoz", font_size=48, color=YELLOW)
        donors = [Text("Alex", font_size=50),
                  Text("Udit Ransaria", font_size=50),
                  Text("stuartmcvicar.bsky.social", font_size=50),
                  Text("Ilgwon Ha", font_size=50),
                  Text("maneesh29s", font_size=50),
                  Text("Gaussian Pombo", font_size=50),
                  Text("Marc Uecker", font_size=50),
                  Text("drunkyoda", font_size=50),
                  Text("danikhan632", font_size=50),
                  Text("SowmithK", font_size=50),
                  Text("Anonymous x5", font_size=50)]
        VGroup(*donors).arrange(DOWN).next_to(bmac, DOWN)

        subscribe = SVGMobject("icons/subscribe.svg")
        like = SVGMobject("icons/like.svg")
        share = SVGMobject("icons/share.svg")
        VGroup(subscribe, like, share).arrange(RIGHT).next_to(VGroup(*donors), DOWN).scale(0.7)

        self.camera.auto_zoom(VGroup(bmac, share, like, subscribe), margin=4, animate=False)
        with self.voiceover(text="""This channel is ad free and it's because I get ocasional donations from people who enjoy this work
                            that are sufficient enough for me to buy all the essential gear for making those videos""") as trk:
            self.play(Write(bmac))
            for donor in donors:
                self.play(Write(donor))

        with self.voiceover(text="""Huge thanks for them for doing so. If you want do become one of them - you can visit my buymeacoffe page""") as trk:
            pass

        with self.voiceover(text="""And you can always support me for fre by <bookmark mark='1'/>subscribing, <bookmark mark='2'/>leaving a like, <bookmark mark='3'/>commenting and sharing this video with your friends""") as trk:
            self.play(Create(like), Create(subscribe), Create(share))
            self.wait_until_bookmark("1")
            self.play(subscribe.animate.set_color(RED))
            self.wait_until_bookmark("2")
            self.play(like.animate.set_color(RED))
            self.wait_until_bookmark("3")
            self.play(share.animate.set_color(RED))

        with self.voiceover(text="""I'll see you in the next episode, bye""") as trk:
            pass

        self.play(*[FadeOut(x) for x in self.mobjects])
        self.wait(2)
