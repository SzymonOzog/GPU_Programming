from manim import *
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover import VoiceoverScene
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import numpy as np
import random
import math
from math import radians

class Occupancy(VoiceoverScene, ZoomedScene):
    def construct(self):
        self.set_speech_service(
                GTTSService(transcription_model="base")
                )
        cores = [Square(color=GREEN, fill_color=GREEN, fill_opacity=0.5) for _ in range(2048)]
        VGroup(*cores).arrange_in_grid(32, 64).move_to(ORIGIN)
        self.camera.auto_zoom(VGroup(*cores), animate=False)
        # it's late when I'm writing this, there must be a smarter way
        corner = cores[0].get_corner(UL)
        radius = 1
        nicely_animated = []
        while len(nicely_animated) != len(cores):
            for c in cores:
                center = c.get_center()
                if c not in nicely_animated and np.sqrt(np.sum((center-corner)**2)) < radius:
                    nicely_animated.append(c)
            radius+=2


        self.play(LaggedStart(*[Create(x) for x in nicely_animated], lag_ratio=0.001))
        all = SurroundingRectangle(VGroup(*cores), color=GREEN, fill_color=GREEN, fill_opacity=0.5)
        self.play(FadeIn(all), FadeOut(VGroup(*cores)))
        self.wait(1)
        return

        gpu = Rectangle(height=6, width=12, color=GREEN)
        gpu_t = Text("GPU", color=GREEN, font_size=24).next_to(gpu, UP, buff=0.1, aligned_edge=LEFT)

        dram = Rectangle(height=5, width=1, color=RED, fill_color=RED, fill_opacity=0.5).shift(5.25*LEFT)
        dram_t = Text("DRAM", color=RED, font_size=52).move_to(dram).rotate(PI/2)

        chip = Rectangle(height=5, width=10, color=YELLOW).shift(0.5*RIGHT)
        chip_t = Text("CHIP", color=YELLOW, font_size=24).next_to(chip, UP, buff=0.1, aligned_edge=LEFT)

        l2 = Rectangle(height=1, width=9, color=BLUE, fill_color=BLUE, fill_opacity=0.5).shift(0.5*RIGHT)
        l2_t = Text("L2 Cache", color=BLUE, font_size=52).move_to(l2)

        self.play(Create(gpu), Write(gpu_t), Create(dram), Write(dram_t), Create(chip), Write(chip_t), Create(l2), Write(l2_t))

        mc = Rectangle(height=4.5, width=0.25).shift(4.25*LEFT)
        mc_t = Text("6 x Memory Controller", font_size=14).move_to(mc).rotate(PI/2)

        mc2 = Rectangle(height=4.5, width=0.25).shift(5.25*RIGHT)
        mc_t2 = Text("6 x Memory Controller", font_size=14).move_to(mc2).rotate(-PI/2)

        gpcs = []
        gpc_ts = []

        for i in range(6):
            if i == 0:
                gpcs.append(Rectangle(height=1.5, width=1.25, color=PURPLE, fill_color=PURPLE, fill_opacity=0.5, stroke_width=2).next_to(l2, UP, aligned_edge=LEFT, buff=0.25).shift(0.125*RIGHT))
            else:
              gpcs.append(Rectangle(height=1.5, width=1.25, color=PURPLE, fill_color=PURPLE, fill_opacity=0.5, stroke_width=2).next_to(gpcs[-1], RIGHT))
            gpc_ts.append(Text("GPC", font_size=32, color=PURPLE).move_to(gpcs[-1]))

        for i in range(6):
            if i == 0:
                gpcs.append(Rectangle(height=1.5, width=1.25, color=PURPLE, fill_color=PURPLE, fill_opacity=0.5, stroke_width=2).next_to(l2, DOWN, aligned_edge=LEFT, buff=0.25).shift(0.125*RIGHT))
            else:
                gpcs.append(Rectangle(height=1.5, width=1.25, color=PURPLE, fill_color=PURPLE, fill_opacity=0.5, stroke_width=2).next_to(gpcs[-1], RIGHT))
            gpc_ts.append(Text("GPC", font_size=32, color=PURPLE).move_to(gpcs[-1]))

        self.play(Create(mc), Write(mc_t), Create(mc2), Write(mc_t2), LaggedStart(*[Create(gpc) for gpc in gpcs], *[Write(t) for t in gpc_ts]))

        re = Rectangle(height=0.1, width=1.15, stroke_width=1).move_to(gpcs[0], UP).shift(0.03*DOWN)
        re_t = Text("Raster Engine", font_size=18).move_to(re).scale(0.3)

        rop = Rectangle(height=0.1, width=0.55, stroke_width=1).move_to(gpcs[0], LEFT+DOWN).shift(0.04*(UP+RIGHT))
        rop_t = Text("8x ROP", font_size=18).move_to(rop).scale(0.3)
        rop2 = Rectangle(height=0.1, width=0.55, stroke_width=1).move_to(gpcs[0], RIGHT+DOWN).shift(0.04*(UP+LEFT))
        rop_t2 = Text("8x ROP", font_size=18).move_to(rop2).scale(0.3)

        tpcs = []
        tpc_ts = []

        for i in range(3):
            if i == 0:
                tpcs.append(Rectangle(height=0.55, width=0.33, color=ORANGE, fill_color=ORANGE, fill_opacity=0.5, stroke_width=1).move_to(gpcs[0], LEFT+UP).shift(0.07*RIGHT + 0.17*DOWN))
            else:
                tpcs.append(Rectangle(height=0.55, width=0.33, color=ORANGE, fill_color=ORANGE, fill_opacity=0.5, stroke_width=1).next_to(tpcs[-1], RIGHT, buff=0.07))
            tpc_ts.append(Text("TPC", font_size=12, color=ORANGE).move_to(tpcs[-1]).rotate(PI/2))

        for i in range(3):
            if i == 0:
                tpcs.append(Rectangle(height=0.55, width=0.33, color=ORANGE, fill_color=ORANGE, fill_opacity=0.5, stroke_width=1).move_to(gpcs[0], LEFT+DOWN).shift(0.07*RIGHT+ 0.17*UP))
            else:
                tpcs.append(Rectangle(height=0.55, width=0.33, color=ORANGE, fill_color=ORANGE, fill_opacity=0.5, stroke_width=1).next_to(tpcs[-1], RIGHT, buff=0.07))
            tpc_ts.append(Text("TPC", font_size=12, color=ORANGE).move_to(tpcs[-1]).rotate(PI/2))

        self.play(FadeOut(gpc_ts[0]), Transform(gpcs[0], Rectangle(height=1.5, width=1.25, color=PURPLE).next_to(l2, UP, aligned_edge=LEFT, buff=0.25).shift(0.125*RIGHT)))
        self.play(self.camera.auto_zoom(gpcs[0]))
        self.play(LaggedStart(*[Create(tpc) for tpc in tpcs], *[Write(t) for t in tpc_ts]), Create(re), Write(re_t), Create(rop), Write(rop_t), Create(rop2), Write(rop_t2))

        self.play(FadeOut(tpc_ts[0]), Transform(tpcs[0], Rectangle(height=0.55, width=0.33, color=ORANGE, stroke_width=2).move_to(tpcs[0])))
        all = VGroup(*[x for x in self.mobjects if isinstance(x, Rectangle) or isinstance(x, Text)])
        for x in self.mobjects:
            if isinstance(x, Rectangle):
                x.stroke_width*=10
        all.scale(10)
        self.camera.auto_zoom(gpcs[0], animate=False)
        self.play(self.camera.auto_zoom(tpcs[0]))

        pm = Rectangle(width=2.5, height=0.5, fill_opacity=0.5).move_to(tpcs[0]).shift(2.2*UP)
        pm_t = Text("PolyMorph Engine", font_size=40).scale(0.5).move_to(pm)

        sm = Rectangle(width=2.5, height=2, color=MAROON, fill_color=MAROON, fill_opacity=0.5).move_to(tpcs[0]).shift(0.7*UP)
        sm_t = Text("    Streaming\nMultiprocessor", font_size=24, color=MAROON).move_to(sm)

        sm2 = Rectangle(width=2.5, height=2, color=MAROON, fill_color=MAROON, fill_opacity=0.5).move_to(tpcs[0]).shift(1.5*DOWN)
        sm2_t = Text("    Streaming\nMultiprocessor", font_size=24, color=MAROON).move_to(sm2)
        self.play(Create(pm), Write(pm_t), Create(sm), Create(sm2), Write(sm_t), Write(sm2_t))

        self.play(FadeOut(sm_t), Transform(sm, Rectangle(width=2.5, height=2, color=MAROON).move_to(sm)))
        all = VGroup(*[x for x in self.mobjects if isinstance(x, Rectangle) or isinstance(x, Text)])
        for x in self.mobjects:
            if isinstance(x, Rectangle):
                x.stroke_width*=3
        all.scale(3)
        self.camera.auto_zoom(tpcs[0], animate=False)
        self.play(self.camera.auto_zoom(sm))

        rt = Rectangle(width=2.5, height=0.7, fill_opacity=0.5).move_to(sm).shift(2.4*DOWN + 2.25*RIGHT)
        rt_t = Text("RT Core", font_size=32).move_to(rt)

        texs = []
        tex_ts = []
        for i in range(4):
            if i == 0:
                texs.append(Rectangle(height=0.7, width=1, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.5).move_to(sm).shift(2.4*DOWN + 3*LEFT))
            else:
                texs.append(Rectangle(height=0.7, width=1, color=BLUE_E, fill_color=BLUE_E, fill_opacity=0.5).next_to(texs[-1], RIGHT, buff=0.1))
            tex_ts.append(Text("TEX", font_size=32, color=BLUE_E).move_to(texs[-1]))

        l1 = Rectangle(height=0.5, width=7, color=GOLD_A, fill_color=GOLD_A, fill_opacity=0.5).move_to(sm).shift(1.7*DOWN)
        l1_t = Text("128KB L1 Cache / Shared Memory", color=GOLD_A, font_size=28).move_to(l1)

        cc = Rectangle(height=0.5, width=7, color=GOLD_E, fill_color=GOLD_E, fill_opacity=0.5).next_to(l1, UP, buff=0.1)
        cc_t = Text("8KB Constant Cache", color=GOLD_E, font_size=28).move_to(cc)

        ps = []
        p_ts = []

        for i in range(4):
            if i == 0:
                ps.append(Rectangle(height=3.6, width=1.675, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.5).move_to(sm).shift(2.65*LEFT + 1.05*UP))
            else:
                ps.append(Rectangle(height=3.6, width=1.675, color=GREEN_A, fill_color=GREEN_A, fill_opacity=0.5).next_to(ps[-1], RIGHT, buff=0.093))
            p_ts.append(Text("Processing Block", font_size=32, color=GREEN_A).move_to(ps[-1]).rotate(PI/2))

        self.play(Create(rt), Write(rt_t), LaggedStart(*[Create(tex) for tex in texs], *[Write(t) for t in tex_ts]), Create(l1), Write(l1_t), Create(cc), Write(cc_t), LaggedStart(*[Create(p) for p in ps], *[Write(t) for t in p_ts]))

        self.play(FadeOut(p_ts[0]), Transform(ps[0], Rectangle(height=3.6, width=1.675, color=GREEN_A).move_to(ps[0])))
        all = VGroup(*[x for x in self.mobjects if isinstance(x, Rectangle) or isinstance(x, Text)])
        for x in self.mobjects:
            if isinstance(x, Rectangle):
                x.stroke_width*=2
        all.scale(2)
        self.camera.auto_zoom(ps[0], animate=False)

        ws = Rectangle(width=3, height=0.5, color=YELLOW_A, fill_color=YELLOW_A, fill_opacity=0.5).move_to(ps[0]).shift(3.15*UP)
        ws_t = Text("Warp Scheduler", color=YELLOW_A, font_size=32).scale(0.6).move_to(ws)

        du = Rectangle(width=3, height=0.5, color=YELLOW_B, fill_color=YELLOW_B, fill_opacity=0.5).next_to(ws, DOWN, buff=0.2)
        du_t = Text("Dispatch Unit", color=YELLOW_B, font_size=32).scale(0.6).move_to(du)

        ic = Rectangle(width=3, height=0.5, color=YELLOW_C, fill_color=YELLOW_C, fill_opacity=0.5).next_to(du, DOWN, buff=0.2)
        ic_t = Text("L0 Instruction Cache", color=YELLOW_C, font_size=32).scale(0.6).move_to(ic)

        rf = Rectangle(width=3, height=0.66, color=BLUE_A, fill_color=BLUE_A, fill_opacity=0.5).next_to(ic, DOWN, buff=0.2)
        rf_t = Text("64KB Register File", color=BLUE_A, font_size=32).scale(0.7).move_to(rf)

        tc = Rectangle(width=3, height=0.5, color=GREEN_B, fill_color=GREEN_B, fill_opacity=0.5).next_to(rf, DOWN, buff=0.2)
        tc_t = Text("Tensor Core", color=GREEN_B, font_size=32).scale(0.7).move_to(tc)

        fpcs = []
        for i in range(2):
            for j in range(8):
                fpc = Rectangle(width=0.33, height=0.33, color=GREEN_C, fill_color=GREEN_C, fill_opacity=0.5, stroke_width=1)
                if j == 0:
                    if i == 0:
                        fpc.next_to(tc, DOWN, aligned_edge=LEFT, buff=0.2)
                    else:
                        fpc.next_to(fpcs[0], DOWN, aligned_edge=LEFT, buff=0.05)
                else:
                    fpc.next_to(fpcs[-1], RIGHT, buff=0.05)
                fpcs.append(fpc)
        fpc_t = Text("FP32", font_size=32, color=GREEN_C).move_to(VGroup(*fpcs))


        fpcis = []
        for i in range(2):
            for j in range(8):
                fpci = Rectangle(width=0.33, height=0.33, color=GREEN_E, fill_color=GREEN_E, fill_opacity=0.5, stroke_width=1)
                if j == 0:
                    if i == 0:
                        fpci.next_to(fpcs[8], DOWN, aligned_edge=LEFT, buff=0.2)
                    else:
                      fpci.next_to(fpcis[0], DOWN, aligned_edge=LEFT, buff=0.05)
                else:
                    fpci.next_to(fpcis[-1], RIGHT, buff=0.05)
                fpcis.append(fpci)
        fpci_t = Text("FP32/I32", font_size=32, color=GREEN_E).move_to(VGroup(*fpcis))


        lsus = []
        lsu_ts = []
        for i in range(4):
            lsu = Rectangle(width=0.68, height=0.55, color=RED_A, fill_color=RED_A, fill_opacity=0.5, stroke_width=2)
            if i == 0:
              lsu.next_to(fpcis[8], DOWN, aligned_edge=LEFT, buff=0.2)
            else:
              lsu.next_to(lsus[-1], RIGHT, buff=0.1)
            lsus.append(lsu)
            lsu_ts.append(Text("LD/ST", font_size=32, color=RED_A).scale(0.5).move_to(lsus[-1]))


        sfus = []
        sfu_ts = []
        for i in range(4):
          sfu = Rectangle(width=0.68, height=0.55, color=RED_C, fill_color=RED_C, fill_opacity=0.5, stroke_width=2)
          if i == 0:
              sfu.next_to(lsus[0], DOWN, aligned_edge=LEFT, buff=0.2)
          else:
              sfu.next_to(sfus[-1], RIGHT, buff=0.1)
          sfus.append(sfu)
          sfu_ts.append(Text("SFU", font_size=32, color=RED_C).scale(0.5).move_to(sfus[-1]))

        self.camera.auto_zoom(sm, animate=False)
        self.play(LaggedStart(*[Create(x) for x in fpcs]), Write(fpc_t), LaggedStart(*[Create(x) for x in fpcis]), Write(fpci_t), Create(tc), Write(tc_t), Create(ws), Write(ws_t), Create(du), Write(du_t), Create(ic), Write(ic_t), Create(rf), Write(rf_t), LaggedStart(*[Create(x) for x in lsus]), LaggedStart(*[Write(x) for x in lsu_ts]), LaggedStart(*[Create(x) for x in sfus]), LaggedStart(*[Write(x) for x in sfu_ts]))
        self.wait(1)
