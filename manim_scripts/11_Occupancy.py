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

        self.camera.frame.save_state()
        gpu = Rectangle(height=6, width=12, color=GREEN)
        gpu_t = Text("GPU", color=GREEN, font_size=24).next_to(gpu, UP, buff=0.1, aligned_edge=LEFT)

        dram = Rectangle(height=5, width=1, color=RED, fill_color=RED, fill_opacity=0.5).shift(5.25*LEFT)
        dram_t = Text("DRAM", color=RED, font_size=52).move_to(dram).rotate(PI/2)

        chip = Rectangle(height=5, width=10, color=YELLOW).shift(0.5*RIGHT)
        chip_t = Text("CHIP", color=YELLOW, font_size=24).next_to(chip, UP, buff=0.1, aligned_edge=LEFT)

        l2 = Rectangle(height=1, width=9, color=BLUE, fill_color=BLUE, fill_opacity=0.5).shift(0.5*RIGHT)
        l2_t = Text("L2 Cache", color=BLUE, font_size=52).move_to(l2)


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


        with self.voiceover(text="""If you watched my episode on the GPU architecture, you might remember
                            thet when we go to the lowest component level, we arrived at the unit that 
                            handles the execution of our code, that is refered to as a processing block""") as trk:
            self.play(Create(gpu), Write(gpu_t), Create(dram), Write(dram_t), Create(chip), Write(chip_t), Create(l2), Write(l2_t), Create(mc), Write(mc_t), Create(mc2), Write(mc_t2), LaggedStart(*[Create(gpc) for gpc in gpcs], *[Write(t) for t in gpc_ts]))
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
            self.play(self.camera.auto_zoom(ps[0]))

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

            # self.camera.auto_zoom(sm, animate=False)
            self.play(LaggedStart(*[Create(x) for x in fpcs]), Write(fpc_t), LaggedStart(*[Create(x) for x in fpcis]), Write(fpci_t), Create(tc), Write(tc_t), Create(ws), Write(ws_t), Create(du), Write(du_t), Create(ic), Write(ic_t), Create(rf), Write(rf_t), LaggedStart(*[Create(x) for x in lsus]), LaggedStart(*[Write(x) for x in lsu_ts]), LaggedStart(*[Create(x) for x in sfus]), LaggedStart(*[Write(x) for x in sfu_ts]))

        scene = Group(*[x for x in self.mobjects])

        spotlight = Exclusion(Rectangle(width=1000, height=1000), SurroundingRectangle(ws, buff=0.3), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)

        with self.voiceover(text="""Inside it there is a component named warp scheduler""") as trk:
            self.play(FadeIn(spotlight))
        self.play(FadeOut(spotlight))


        block = Rectangle(width=4.3, height=7, color=BLUE).next_to(ps[0], RIGHT)
        block_t = Text("Block", color=BLUE, font_size=40).move_to(block, aligned_edge=UP).shift(0.2*DOWN)

        threads = [Rectangle(width=0.33, height=0.33, color=GREEN, fill_color=GREEN, fill_opacity=0.5) for _ in range(96)]
        tg = VGroup(*threads).arrange_in_grid(rows=12, buff=(0.05, 0.15)).move_to(block)
        tmp = VGroup(*threads[:32])
        w1_t = Text("Warp 0", font_size=24, color=PURPLE).rotate(PI/2).next_to(tmp, LEFT, buff=0.1)
        w1 = SurroundingRectangle(VGroup(tmp, w1_t, w1_t.copy().next_to(tmp, RIGHT, buff=0.1)), buff=0.05, color=PURPLE)
        w1.stretch_to_fit_width(w1.width*1.05)

        tmp = VGroup(*threads[32:64])
        w2_t = Text("Warp 1", font_size=24, color=PURPLE).rotate(PI/2).next_to(tmp, LEFT, buff=0.1)
        w2 = SurroundingRectangle(VGroup(tmp, w2_t, w2_t.copy().next_to(tmp, RIGHT, buff=0.1)), buff=0.05, color=PURPLE)
        w2.stretch_to_fit_width(w2.width*1.05)

        tmp = VGroup(*threads[64:])
        w3_t = Text("Warp 2", font_size=24, color=PURPLE).rotate(PI/2).next_to(tmp, LEFT, buff=0.1)
        w3 = SurroundingRectangle(VGroup(tmp, w3_t, w3_t.copy().next_to(tmp, RIGHT, aligned_edge=UP, buff=0.1)), buff=0.05, color=PURPLE)
        w3.stretch_to_fit_width(w3.width*1.05)

        with self.voiceover(text="""It's job is to manage active warps in a way that hides latency of memory operations""") as trk:
            self.play(*[FadeOut(x) for x in ps], [FadeOut(x) for x in p_ts], FadeOut(sm), FadeOut(tpcs[0]))
            self.play(Create(block), Write(block_t) ,LaggedStart(*[Create(x) for x in threads], lag_ratio=0.03) ,Write(w1_t), Create(w1) ,Write(w2_t), Create(w2) ,Write(w3_t), Create(w3))

        with self.voiceover(text="""So when one warps requests memory, it swaps it out with another active warp
                            so that all cores are executing instructions whenever possible""") as trk:
            anims = []
            for t, c in zip(threads[:32], fpcs+fpcis):
                t.save_state()
                anims.append(t.animate.move_to(c))
            self.play(LaggedStart(*anims))

            anims = []
            for t, c in zip(threads[:32], fpcs+fpcis):
                anims.append(Restore(t))
            for i, (t, c) in enumerate(zip(threads[32:64], fpcs+fpcis)):
                t.save_state()
                anims.append(t.animate.move_to(c))
            self.play(LaggedStart(*anims))

            anims = []
            for t, c in zip(threads[32:64], fpcs+fpcis):
                anims.append(Restore(t))
            for i, (t, c) in enumerate(zip(threads[64:], fpcs+fpcis)):
                t.save_state()
                anims.append(t.animate.move_to(c))
            self.play(LaggedStart(*anims))

        
        with self.voiceover(text="""This leads us to the intuitive assumtion that the more warps we launch
                            the better the warp scheduler can be at hiding latency""") as trk:
            pass
        self.play(*[FadeOut(x) for x in self.mobjects])

        cores = [Square(color=GREEN, fill_color=GREEN, fill_opacity=0.75) for _ in range(64)]
        VGroup(*cores).arrange_in_grid(8, 8).move_to(ORIGIN)
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


        all = VGroup(*cores)
        cores_count = Text("64 Warps", color=WHITE, font_size = 150, z_index=1).scale(2).move_to(all)

        with self.voiceover(text="""One SM can run a maximum of 64 warps on a datacenter grade GPU like the A100 or H100""") as trk:
            self.play(LaggedStart(*[Create(x) for x in nicely_animated], lag_ratio=0.001))
            self.wait(1)
            self.play(Write(cores_count))

        rows = 8 
        with self.voiceover(text="""But not all are active when I launch my kernel""") as trk:
            for occupancy in [0.25, 0.5,  0.75]:
                anims = []
                for i, core in enumerate(cores):
                    active = i / len(cores) >= occupancy
                    color = GREEN if active else GREEN_E
                    opacity = 0.75 if active else 0.25
                    anims.append(core.animate.set_fill(color, opacity=opacity))
                groups = []
                for i in range(rows):
                    current = anims[i*8:(i+1)*8]
                    if i == rows//2:
                        current.append(Transform(cores_count, Text(f"{int((1-occupancy)*100)} %", color=WHITE, font_size=150, z_index=1).scale(2).move_to(all)))
                    groups.append(AnimationGroup(*current))
                self.play(LaggedStart(*groups, lag_ratio=0.1))
                self.wait(1)
                
        self.play(*[FadeOut(x) for x in self.mobjects])
        occupancy_t = Text("Occupancy")
        occupancy_t2 = Tex("$\\frac{active\\;warps}{total\\;warps}$").next_to(occupancy_t, DOWN)
        self.camera.auto_zoom(VGroup(occupancy_t, occupancy_t2), animate=False, margin=5)
        with self.voiceover(text="""This is what we call occupancy, the ratio of active warps
                            to the maximum active warps on my device""") as trk:
            self.play(Write(occupancy_t))
            self.wait(1)
            self.play(Write(occupancy_t2))

        self.play(*[FadeOut(x) for x in self.mobjects])
        self.camera.auto_zoom(VGroup(*cores), animate=False)
        with self.voiceover(text="""In this episode we'll look into all of the factors that we need to consider
                            to ensure that our GPU's run at maximum occupancy, and spoiler alert. It's 
                            not just how much warps we tell the GPU to run""") as trk:
            self.play(FadeIn(all))
            for occupancy in [0.5,  0.25, 0.125, 0]:
                anims = []
                for i, core in enumerate(cores):
                    active = i / len(cores) >= occupancy
                    color = GREEN if active else GREEN_E
                    opacity = 0.75 if active else 0.25
                    anims.append(core.animate.set_fill(color, opacity=opacity))
                groups = []
                for i in range(rows):
                    current = anims[i*8:(i+1)*8]
                    if i == rows//2:
                        current.append(Transform(cores_count, Text(f"{int((1-occupancy)*100)} %", color=WHITE, font_size=150, z_index=1).scale(2).move_to(all)))
                    groups.append(AnimationGroup(*current))
                groups = list(reversed(groups))

                self.play(LaggedStart(*groups, lag_ratio=0.1))
                self.wait(1)

        with self.voiceover(text="""It is mostly dependant on the resources that we have available for our warps""") as trk:
            self.play(*[FadeOut(x) for x in self.mobjects])
            self.camera.auto_zoom(ps[0], animate=False)
            self.play(FadeIn(scene))

        spotlight = Exclusion(Rectangle(width=1000, height=1000), SurroundingRectangle(rf, buff=0.3), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)

        with self.voiceover(text="""First of all, on the procesisng block level we have a register file, on my architecture
                            it's 64 KB <bookmark mark='1'/> and there are 4 of those in our streaming multiprocessor""") as trk:
            self.play(FadeIn(spotlight))
            self.wait_until_bookmark("1")
            self.play(self.camera.auto_zoom(sm))
            self.play(FadeOut(spotlight))

        self.play(FadeOut(scene))
        
        with self.voiceover(text="""This gives us a limit of 64 thousand 4 byte registers per block, <bookmark mark='1'/>to launch 
                            64 warps we have a limit of 1 thousand registers per warp, that consists of 32 threads.<bookmark mark='2'/>
                            So to achieve full occupancy we can use at most 32 registers per threadd""") as trk:
            large_rect = Rectangle(height=4, width=6, color=BLUE)
            self.camera.auto_zoom(large_rect, margin=4, animate=False)
            label_block = MathTex(r"65{,}536 \text{ registers per block}").next_to(large_rect, UP).shift(UP)
            self.play(Create(large_rect), Write(label_block))
            
            warps = VGroup()
            for _ in range(64):
                warp = Rectangle(height=0.5, width=1.5, color=ORANGE)
                warps.add(warp)
            warps.arrange_in_grid(rows=8, cols=8, buff=0.1)
            labels_warp = MathTex(r"1{,}024 \text{ registers}").scale(0.4)
            for warp in warps:
                warp.add(labels_warp.copy().move_to(warp))
            self.wait_until_bookmark("1")
            self.play(Transform(large_rect, warps, replace_mobject_with_target_in_scene=True))
            
            selected_warp = warps[0]
            threads = VGroup()
            for _ in range(32):
                thread = Rectangle(height=0.4, width=1, color=GREEN)
                threads.add(thread)
            threads.arrange_in_grid(rows=4, cols=8, buff=0.05)
            label_thread = MathTex(r"\text{thread}").scale(0.5)
            for thread in threads:
                thread.add(label_thread.copy().move_to(thread))
            self.wait_until_bookmark("2")
            self.play(selected_warp.animate.scale(3).move_to(ORIGIN-DOWN), *[FadeOut(x) for x in warps[1:]])
            self.wait(0.3)
            self.play(Transform(selected_warp, threads, replace_mobject_with_target_in_scene=True))
            
            calculation = MathTex(r"\frac{1{,}024 \text{ registers}}{32 \text{ threads}} = 32").to_edge(UP)
            conclusion = MathTex(r"\text{at most } 32 \text{ registers per thread for full occupancy}").next_to(calculation, DOWN)
            self.play(Transform(label_block, calculation))
            self.play(Write(conclusion))

        self.play(*[FadeOut(x) for x in self.mobjects])
        self.camera.auto_zoom(sm, animate=False)
        self.play(FadeIn(scene))

        spotlight = Exclusion(Rectangle(width=1000, height=1000), SurroundingRectangle(l1, buff=0.3), color=BLACK, fill_opacity=0.7, stroke_width=0, z_index=2)
        with self.voiceover(text="""We also have a limit imposed by our shared <bookmark mark='1'/> memory""") as trk:
            self.wait_until_bookmark("1")
            self.play(FadeIn(spotlight))

        with self.voiceover(text="""So in here the limit will be dictated by the number of active blocks""") as trk:
            pass

        self.play(FadeOut(scene), FadeOut(spotlight))

        self.camera.frame.restore()

        block = Rectangle(width=4, height=5, color=BLUE)
        warps = [Rectangle(width=0.33, height=0.33, color=GREEN, fill_color=GREEN, fill_opacity=0.5) for _ in range(64)]
        x = VGroup(*warps).arrange_in_grid(8,8, buff=0.1).move_to(block)
        block_t = Text("128KB", color=BLUE).move_to(block, aligned_edge=DOWN).shift(0.1*UP)

        with self.voiceover(text="""For example, if we were to fit all of our warps inside one block, we coud utilise all of our shared 
                            memory""") as trk:
            self.play(Create(block))
            self.play(LaggedStart(*[Create(w) for w in warps], lag_ratio=0.02))
            self.play(Write(block_t))


        with self.voiceover(text="""If we were to distribute it between 2 blocks we could use half of it etc...""") as trk:
            self.play(block.animate.shift(3*LEFT), x.animate.shift(3*LEFT), block_t.animate.shift(3*LEFT))
            
            block2 = Rectangle(width=4, height=5, color=BLUE).shift(3*RIGHT)
            block2_t = Text("64KB", color=BLUE).move_to(block2, aligned_edge=DOWN).shift(0.1*UP)

            self.play(Create(block2))
            self.play(VGroup(*warps[32:]).animate.arrange_in_grid(4,8, buff=(0.1, 0.5)).move_to(block2), 
                      VGroup(*warps[:32]).animate.arrange_in_grid(4,8, buff=(0.1, 0.5)).move_to(block))

            self.play(Transform(block_t, Text("64KB", color=BLUE).move_to(block, aligned_edge=DOWN).shift(0.1*UP)),
                      Write(block2_t))

        self.play(*[FadeOut(x) for x in self.mobjects])
        self.camera.auto_zoom(VGroup(*ps), margin=4, animate=False)
        self.camera.frame.shift(UP)

        with self.voiceover(text="""But that is just a theoretical occupancy, and it can differ from what we achieve""") as trk:
            self.play(*[FadeIn(x) for x in fpcs], FadeIn(fpc_t), *[FadeIn(x) for x in fpcis], FadeIn(fpci_t), FadeIn(tc), FadeIn(tc_t), FadeIn(ws), FadeIn(ws_t), FadeIn(du), FadeIn(du_t), FadeIn(ic), FadeIn(ic_t), FadeIn(rf), FadeIn(rf_t), *[FadeIn(x) for x in lsus], *[FadeIn(x) for x in lsu_ts], *[FadeIn(x) for x in sfus], *[FadeIn(x) for x in sfu_ts],
                      *[FadeIn(x) for x in ps], *[FadeIn(x) for x in p_ts[1:]])

        with self.voiceover(text="""One situation where the achieved occupancy might be worse than the theoretical one is a tail effect""") as trk:
            pass

        warps = [Rectangle(width=0.33, height=0.33, color=GREEN, fill_color=GREEN, fill_opacity=0.5) for _ in range(64)]
        
        warps1 = VGroup(*warps[:16]).arrange_in_grid(4,4, buff=0.1).next_to(ps[0], UP)
        warps1 = VGroup(*warps[16:32]).arrange_in_grid(4,4, buff=0.1).next_to(ps[1], UP)
        warps1 = VGroup(*warps[32:48]).arrange_in_grid(4,4, buff=0.1).next_to(ps[2], UP)
        warps1 = VGroup(*warps[48:]).arrange_in_grid(4,4, buff=0.1).next_to(ps[3], UP)

        with self.voiceover(text="""Let's say that we are launching a perfect 64 warps across our processing blocks""") as trk:
            self.play(LaggedStart(*[Create(w) for w in warps[:16]]),
                      LaggedStart(*[Create(w) for w in warps[16:32]]),
                      LaggedStart(*[Create(w) for w in warps[32:48]]),
                      LaggedStart(*[Create(w) for w in warps[48:]]),
                      )

        anims = [[] for _ in range(4)]
        for i in range(4):
            anims[0].append(FadeOut(VGroup(*warps[i*4 : (i+1)*4])))
            anims[1].append(FadeOut(VGroup(*warps[16+i*4 : 16+(i+1)*4])))
            anims[2].append(FadeOut(VGroup(*warps[32+i*4 : 32+(i+1)*4])))
            anims[3].append(FadeOut(VGroup(*warps[48+i*4 : 48+(i+1)*4])))

        with self.voiceover(text="""If some of those warps take a longer time to finish than the other ones, we get a situation where the active blocks start decreasing because we need to
                            wait for our block to finish before issuing a new one""") as trk:
            self.play(LaggedStart(*anims[0], lag_ratio=6), LaggedStart(*anims[1], lag_ratio=2), LaggedStart(*anims[2], lag_ratio=2), LaggedStart(*anims[3], lag_ratio=2))

        with self.voiceover(text="""For this, the only solution would be to balance the workload more efficiently between the SM's""") as trk:
            pass

        with self.voiceover(text="""This can also accur not only on a thread level but also on a block level""") as trk:
            self.play(FadeIn(scene))
            self.play(self.camera.auto_zoom(scene, margin=165))
            self.play(self.camera.frame.animate.shift(30*UP))
        blocks = [Rectangle(width=30, height=30, color=BLUE, fill_color=BLUE, fill_opacity=0.5).shift(400*UP) for _ in range(117)]
        anim_groups = []
        j = 0
        for i, gpc in enumerate(gpcs):
            anims = []
            for k in range(7 if i == 3 else 10):
                if k == 0:
                    rt = 5 if i == 3 else 2
                else:
                    rt = 2 if i == 3 else 1
                cp = gpc.copy().set_color(PURPLE_A)
                anims.append(Transform(blocks[j], cp))
                anims.append(FadeOut(blocks[j], run_time=rt))
                j+=1
            anim_groups.append(anims)

        with self.voiceover(text="""One block might take significantly more time to finish than the other ones, reducing our achieved occupancy""") as trk:
            self.play(*[Succession(a.pop(0), a.pop(0), lag_ratio=1.5, suspend_mobject_updating=False) for a in anim_groups])

        with self.voiceover(text=""" The solution to this is actually very simple, we can just increase the number of blocks, and the 
                            time of the longer ones just gets hidden away.
                            This is usially refered to as an unbalanced workload across blocs as opposed to the previous situation
                            where the worload was unbalanced within blocks""") as trk:
            self.play(*[Succession(*a, lag_ratio=1.5, suspend_mobject_updating=False) for a in anim_groups])


        blocks = [Rectangle(width=30, height=30, color=BLUE, fill_color=BLUE, fill_opacity=0.5).shift(400*UP) for _ in range(8)]
        anims = []
        for i, block in enumerate(blocks):
            gpc = gpcs[i%len(gpcs)]
            rt = 3 
            cp = gpc.copy().set_color(PURPLE_A)
            anims.append(Succession(Transform(block, cp), FadeOut(block, run_time=rt), lag_ratio=2.5))

        with self.voiceover(text="""Another factor that might impact our theoretical occupancy is the obvious one of just not launching enough blocks 
                            to fill all of our SM's with work""") as trk:
            self.play(*anims)

        blocks = [Rectangle(width=30, height=30, color=BLUE, fill_color=BLUE, fill_opacity=0.5).shift(400*UP) for _ in range(40)]
        anim_groups = [[] for _ in range(len(gpcs))]
        for i, block in enumerate(blocks):
            gpc = gpcs[i%len(gpcs)]
            rt = 3.5 if i > 36 else 2
            cp = gpc.copy().set_color(PURPLE_A)
            anim_groups[i%len(gpcs)].append(Transform(block, cp))
            anim_groups[i%len(gpcs)].append(FadeOut(block, run_time=rt))

        with self.voiceover(text="""And lastly, we can also get an effect called a partial last wave, where the ending wave of scheduled blocks
                            is not complete enough to fill our whole GPU, but simillarly to the unbalanced workload across blocks, the efects diminish 
                            as we launch more and more blocks""") as trk:
            self.play(*[Succession(*a, lag_ratio=1.5, suspend_mobject_updating=False) for a in anim_groups])

        self.play(*[FadeOut(x) for x in self.mobjects])
        
        bmac = Text("https://buymeacoffee.com/simonoz", font_size=48, color=YELLOW)
        donors = [Text("Alex", font_size=50),
                  Text("Udit Ransaria", font_size=50),
                  Text("stuartmcvicar.bsky.social", font_size=50),
                  Text("Ilgwon Ha", font_size=50),
                  Text("maneesh29s", font_size=50),
                  Text("Gaussian Pombo", font_size=50),
                  Text("Marc Uecker", font_size=50),
                  Text("drunkyoda", font_size=50),
                  Text("Anonymous x5", font_size=50)]
        VGroup(*donors).arrange(DOWN).next_to(bmac, DOWN)

        subscribe = SVGMobject("icons/subscribe.svg")
        like = SVGMobject("icons/like.svg")
        share = SVGMobject("icons/share.svg")
        VGroup(subscribe, like, share).arrange(RIGHT).next_to(VGroup(*donors), DOWN).scale(0.7)

        self.camera.auto_zoom(VGroup(bmac, share, like, subscribe), margin=4, animate=False)
        with self.voiceover(text="""If you can head much better audio quality in this video, this is because I've just gotten a new microphone
                            sponsored entirely by the supporters of this channel""") as trk:
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
