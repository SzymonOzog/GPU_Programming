import os
from manimlib import *
from math import radians
from manim_voiceover.services.gtts import GTTSService
from manim_voiceover.services.recorder import RecorderService
#TODO why do I have to do this
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from voicover_gl import VoiceoverScene

class TensorCores(VoiceoverScene):
    def construct(self):
        # init scene
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService(transcription_model="base")
            )
        self.voiceovers_in_embed = True

        total_n = 32
        tile_n = 8
        self.play(*[FadeOut(x) for x in self.mobjects])
        mat1_f = [Square(stroke_width=0, fill_color=GREEN, fill_opacity=0.5) for _ in range(total_n*total_n)]
        mat2_f = [Square(stroke_width=0, fill_color=BLUE, fill_opacity=0.5) for _ in range(total_n*total_n)]
        mat3_f = [Square(stroke_width=0, fill_color=ORANGE, fill_opacity=0.5) for _ in range(total_n*total_n)]


        g1 = Group(*mat1_f).arrange_in_grid(total_n, total_n, buff=4).move_to(ORIGIN, aligned_edge=UL).shift(25*UP + 15*LEFT)
        g2 = Group(*mat2_f).arrange_in_grid(total_n, total_n, buff=4).next_to(g1, UP, buff = 2)
        g3 = Group(*mat3_f).arrange_in_grid(total_n, total_n, buff=4).next_to(g1, LEFT, buff = 2)

        self.frame.set_shape(183, 103)
        self.frame.move_to([-7, 24, 0])

        mat1 = []
        mat2 = []
        mat3 = []
        for i in range(tile_n):
            for j in range(tile_n):
                r = i
                c = j
                mat1.append(mat1_f[r*total_n + c])
                r = total_n - tile_n + i
                c = j
                mat2.append(mat2_f[r*total_n + c])
                r = i
                c = total_n - tile_n + j
                mat3.append(mat3_f[r*total_n + c])

        with self.voiceover(text="""When discissing GPU programming, the algorithm that always pops up is matrix multiplication""") as trk:
            self.play(*[ShowCreation(x) for x in mat2], run_time=2)
            self.play(*[ShowCreation(x) for x in mat3], run_time=2)

        #highlight vectors
        v1 = [mat2[i*8] for i in range(tile_n)]
        v2 = [mat3[i] for i in range(tile_n)]
        with self.voiceover(text="""During this operation we take the row vector of one input matrix, and take it's dot product with 
                            a column vector of the second input matrix, to produce one element of our output matrix""") as trk:

            self.play(*[v.animate.set_opacity(1) for v in v1])
            self.play(*[v.animate.set_opacity(1) for v in v2])

            #interm vector
            v3 = []
            for i in range(tile_n):
                p1 = v2[i].get_center()
                p2 = v1[i].get_center()
                pos = [p1[0], p2[1], p2[2]] 
                v3.append(Square(stroke_width=0, fill_color=YELLOW, fill_opacity=1).move_to(pos))

            for i in range(tile_n):
                vv1 = v1[i].copy()
                vv2 = v2[i].copy()
                self.play(vv1.animate.move_to(v3[i].get_center()),
                          vv2.animate.move_to(v3[i].get_center()), run_time=0.4)
                self.play(FadeOut(vv1), FadeOut(vv2), FadeIn(v3[i]), run_time=0.4)

            #accumulate
            def to_green(step, total = 6):
                percentage_red = 1 - step/total
                red = int(255*percentage_red + 131 * (1-percentage_red)) 
                return f"#{red:02x}C167"

                
            for i in range(tile_n-1):
                self.play(v3[i].animate.move_to(v3[i+1]),
                          v3[i+1].animate.set_color(to_green(i)), run_time=0.4)
                self.remove(v3[i])

            self.play(v3[-1].animate.move_to(mat1[0].get_center()), run_time=0.3)

        #move to 3d
        mat1_3d_f = [VCube(fill_color=GREY, fill_opacity=0.1).move_to(x.get_center()) for x in mat1_f]
        mat2_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()).shift(2*IN) for x in mat2_f]
        mat3_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()).shift(2*IN) for x in mat3_f]

        mat1_3d = []
        mat2_3d = []
        mat3_3d = []

        for i in range(tile_n):
            for j in range(tile_n):
                r = i
                c = j
                mat1_3d.append(mat1_3d_f[r*total_n + c])
                r = total_n - tile_n + i
                c = j
                mat2_3d.append(mat2_3d_f[r*total_n + c])
                r = i
                c = total_n - tile_n + j
                mat3_3d.append(mat3_3d_f[r*total_n + c])

        with self.voiceover(text="""This has a very good three dimensional representation""") as trk:

            self.play(*[FadeOut(x)for x in [mat1[0]] + mat2 + mat3], 
                      *[FadeIn(x)for x in mat1_3d + mat2_3d + mat3_3d],
                      FadeOut(v3[-1]))

            #rotate matrices
            self.play(self.frame.animate.set_euler_angles(-2.24045432,  1.17009916,  1.86961547))

            mat1_3d_f_g = VGroup(*[x for x in mat1_3d_f if x not in mat1_3d])
            mat2_3d_f_g = VGroup(*[x for x in mat2_3d_f if x not in mat2_3d])
            mat3_3d_f_g = VGroup(*[x for x in mat3_3d_f if x not in mat3_3d])


            mat2_3d_f_g.rotate(radians(90), axis=LEFT, about_edge=DOWN)
            mat3_3d_f_g.rotate(radians(90), axis=DOWN, about_edge=RIGHT)


            mat2_3d_g = VGroup(*mat2_3d)
            mat3_3d_g = VGroup(*mat3_3d)

            self.play(mat2_3d_g.animate.rotate(radians(90), axis=LEFT, about_edge=DOWN), mat3_3d_g.animate.rotate(radians(90), axis=DOWN, about_edge=RIGHT))
            self.play(self.frame.animate.set_shape(137, 77).move_to([-8.3, 4.62, -16.36]))

            #show index calculation
            mat2_3d_f_g.shift(4*IN).shift(4*UP)
            mat3_3d_f_g.shift(4*IN).shift(4*LEFT)
            self.play(mat2_3d_g.animate.shift(4*IN).shift(4*UP), mat3_3d_g.animate.shift(4*IN).shift(4*LEFT))

        frame_start = self.frame.copy()
        #highlight vectors
        frame_end = self.frame.copy().set_euler_angles(0, 0, 0)
        frame_end.saved_alpha = 0
        def updater(m, dt):
            frame_end.saved_alpha = min(1, frame_end.saved_alpha+dt/30)
            m.interpolate(frame_start, frame_end, frame_end.saved_alpha)
            
        #show matmul
        thread_numbers = [Text(str(i)).scale(3).move_to(x.get_center()) for i, x in enumerate(mat1_3d)]
        with self.voiceover(text="""Here we can clearly see what vectors of each input matrices correspont to each entry
                            in the outpup matrix and how we can calculate it's dot product. The naive way to to do this on a GPU, that we discussed
                            in one of the first episodes of the GPU programming series had each output element calculated by one thread.
                            So thread 0 <bookmark mark='1'/>would calculate the element at row 0 column 0, thread 1 would calculate <bookmark mark='2'/>the element at row 1 colum one etc.""") as trk:
            for j in range(tile_n):
                for k in range(tile_n):
                    if j + k == 0:
                        run_time = 1
                    elif j > 0:
                        run_time = 0.1
                    else:
                        run_time = 0.25

                    v1 = [mat2_3d[i*8 + k] for i in range(8)]
                    v2 = [mat3_3d[j*8 + i] for i in range(8)]

                    disable_highlight = [
                            m.animate.set_opacity(0.3) for m in mat3_3d+mat2_3d if m not in v1+v2
                            ]
                    
                    dot_prod = []
                    for i in range(8):
                        pos = mat1_3d[j*8 + k].get_center().copy()
                        pos[2] = v1[i].get_center()[2] - (43) + i * 6
                        dot_prod.append(VCube(fill_color=YELLOW).move_to(pos))

                    anims = []
                    for i in range(8):
                        c1 = v1[i].copy()
                        c2 = v2[i].copy()
                        dp = dot_prod[i]
                        anims.extend([ReplacementTransform(c1, dp),ReplacementTransform(c2, dp)])

                    self.play(*[v.animate.set_opacity(1) for v in v1], *[v.animate.set_opacity(1) for v in v2], *disable_highlight, *anims, run_time=run_time)


                    #sum dot products
                    if j + k == 0:
                        run_time=0.5 if j + k == 0 else 0.03
                        for i in range(tile_n-1):
                            tmp = dot_prod[i+1].copy().set_color(to_green(i, total_n))
                            self.play(Transform(dot_prod[i], tmp, run_time=run_time, rate_func=linear), Transform(dot_prod[i+1], tmp, run_time=run_time, rate_func=linear))
                            self.remove(dot_prod[i])

                        tmp = dot_prod[-1].copy().set_color(to_green(7, total_n)).move_to(mat1_3d[j*8+k].get_center()).deactivate_depth_test()
                        self.play(Transform(dot_prod[-1], tmp, remover=True), Transform(mat1_3d[j*8+k], tmp), run_time=run_time)
                        self.frame.add_updater(updater)
                    else:
                        run_time = 0.2 if j < 2 else 0.1
                        dot_prod[-1].deactivate_depth_test()
                        tmp = dot_prod[-1].copy().set_color(to_green(7, total_n)).move_to(mat1_3d[j*8+k].get_center()).deactivate_depth_test()
                        self.play(*[Transform(x, tmp, remover=True) for x in dot_prod], Transform(mat1_3d[j*8+k], tmp), run_time=run_time)

                    self.play(Write(thread_numbers[j*tile_n + k]), run_time=0.05)
            self.wait_until_bookmark("1")
            self.play(mat1_3d_f[0].animate.set_color(GREEN))
            self.wait_until_bookmark("2")
            self.play(mat1_3d_f[0].animate.set_color(to_green(7, total_n)), mat1_3d_f[1].animate.set_color(GREEN))
            self.wait(1)
            self.play(mat1_3d_f[1].animate.set_color(to_green(7, total_n)))
            while trk.get_remaining_duration() > 0.01:
                self.wait(0.01)
            self.frame.remove_updater(updater)

            self.play(*[m.animate.set_opacity(0.3) for m in mat3_3d+mat2_3d])

        #Show full matrix
        with self.voiceover(text="""Our input matrices tend to be very big""") as trk:
            self.play(FadeIn(mat1_3d_f_g), FadeIn(mat2_3d_f_g), FadeIn(mat3_3d_f_g),
                      self.frame.animate.set_euler_angles(-2.24045432,  1.17009916,  1.86961547).set_shape(329, 187).move_to([-28.3, 8.62, -25.36]),
                      *[FadeOut(t) for t in thread_numbers]
                      )


        n_tiles = total_n // tile_n

        mat1_tiles = []
        mat2_tiles = []
        mat3_tiles = []

        for tile_x in range(0, total_n, tile_n):
            outer_tile1 = []
            outer_tile2 = []
            outer_tile3 = []
            for tile_y in range(0, total_n, tile_n):
                tile1 = []
                tile2 = []
                tile3 = []
                for x in range(tile_n):
                    for y in range(tile_n):
                        r = tile_x + x
                        c = tile_y + y
                        tile1.append(mat1_3d_f[r*total_n + c])

                        r = total_n - tile_n - tile_x + x
                        c = tile_y + y
                        tile2.append(mat2_3d_f[r*total_n + c])

                        r = tile_x + x
                        c = total_n - tile_n - tile_y + y
                        tile3.append(mat3_3d_f[r*total_n + c])
                outer_tile1.append(tile1)
                outer_tile2.append(tile2)
                outer_tile3.append(tile3)
            mat1_tiles.append(outer_tile1)
            mat2_tiles.append(outer_tile2)
            mat3_tiles.append(outer_tile3)

        anims = []
        for x in range(n_tiles):
            for y in range(n_tiles):
                anims.append(VGroup(*mat1_tiles[x][y]).animate.shift(y*4*RIGHT + x*4*DOWN))
                anims.append(VGroup(*mat2_tiles[x][y]).animate.shift(y*4*RIGHT + x*4*IN))
                anims.append(VGroup(*mat3_tiles[x][y]).animate.shift(y*4*IN + x*4*DOWN))


        # Show tiling
        with self.voiceover(text="""To optimize it we split our matrices into tiles""") as trk:
            self.play(self.frame.animate.set_shape(329, 187).move_to([-28.3, 8.62, -25.36]))
            self.play(self.frame.animate.set_euler_angles(-2.24045432,  1.17009916,  1.86961547))

            self.play(*anims)

        #play next tile
        with self.voiceover(text="""We talked about the motivation for doing this in an episode on matix tiling so if you want a 
                            refresher on this you can go back to that video, but
                            long story short it helps us move the data to faster memory and reuse it across the threads.
                            The pattern of memory access is used across all highly performant matrix multiplication algorithms""") as trk:
            for tile in range(1, n_tiles):
                mat2_3d = mat2_tiles[tile][0]
                mat3_3d = mat3_tiles[0][tile]
                self.play(VGroup(*mat2_3d).animate.set_opacity(0.5), VGroup(*mat3_3d).animate.set_opacity(0.5), run_time=0.2)
                anims = []
                for j in range(tile_n):
                    for k in range(tile_n):
                        run_time = 0.09

                        v1 = [mat2_3d[i*8 + k] for i in range(8)]
                        v2 = [mat3_3d[j*8 + i] for i in range(8)]

                        disable_highlight = [
                                m.animate.set_opacity(0.5) for m in mat3_3d+mat2_3d if m not in v1+v2
                                ]
                        
                        dot_prod = []
                        for i in range(8):
                            pos = mat1_3d[j*8 + k].get_center().copy()
                            pos[2] = v1[i].get_center()[2] # (43) + i * 6
                            dot_prod.append(VCube(fill_color=YELLOW).move_to(pos))

                        anims = []
                        for i in range(8):
                            c1 = v1[i].copy()
                            c2 = v2[i].copy()
                            anims.extend([ReplacementTransform(c1, dot_prod[i]),ReplacementTransform(c2, dot_prod[i])])

                        self.play(*[v.animate.set_opacity(1) for v in v1], *[v.animate.set_opacity(1) for v in v2], *disable_highlight, *anims, run_time=run_time)

                        run_time = 0.05
                        dot_prod[-1].deactivate_depth_test()
                        tmp = dot_prod[-1].copy().set_color(to_green(tile*8 + 7, total_n)).move_to(mat1_3d[j*8+k].get_center()).deactivate_depth_test()

                        self.play(*[Transform(x, tmp, remover=True) for x in dot_prod], Transform(mat1_3d[j*8+k], tmp), run_time=run_time)
                self.play(*[m.animate.set_opacity(0.3) for m in mat3_3d+mat2_3d], run_time=0.2)

        # reset matrix
        self.play(VGroup(*mat1_3d).animate.set_color(GREY).set_opacity(0.1))

        #show tensor cores
        with self.voiceover(text="""This is what tensor cores are designed to do, they are given 3 matrices as input, <bookmark mark='1'/>2 matrices that we want to multiply
                            and an accumulator containing the result of previous tiled matmul. And they perform <bookmark mark='2'/>a tiled matrix multiplication operation _""") as trk:
            tile = 0
            mat2_3d = mat2_tiles[tile][0]
            mat3_3d = mat3_tiles[0][tile]
            self.wait_until_bookmark("1")
            self.play(VGroup(*mat2_3d).animate.set_opacity(1), VGroup(*mat3_3d).animate.set_opacity(1))
            anims = [] 
            dot_prods = []
            cs = []
            for j in range(tile_n):
                for k in range(tile_n):
                    run_time = 1

                    v1 = [mat2_3d[i*8 + k] for i in range(8)]
                    v2 = [mat3_3d[j*8 + i] for i in range(8)]

                    dot_prod = []
                    for i in range(8):
                        pos = mat1_3d[j*8 + k].get_center().copy()
                        pos[2] = v1[i].get_center()[2] - (43) + i * 6
                        dot_prod.append(VCube(fill_color=YELLOW, side_length=1).move_to(pos))
                    dot_prods.append(dot_prod)

                    for i in range(8):
                        c1 = v1[i].copy()
                        c2 = v2[i].copy()
                        cs.extend([c1, c2])
                        anims.extend([Transform(cs[-2], dot_prod[i], remover=True),ReplacementTransform(cs[-1], dot_prod[i])])

            self.wait_until_bookmark("2")
            self.play(*anims, run_time=run_time)

            #visualize accumulate
            for i in range(7):
                anims = []
                for dot_prod in dot_prods:
                    run_time = 0.5
                    tmp = dot_prod[i+1].copy().set_color(to_green(i, total_n))
                    anims.extend([Transform(dot_prod[i], tmp, run_time=run_time, rate_func=linear, remover=True), 
                                  Transform(dot_prod[i+1], tmp, run_time=run_time, rate_func=linear)])
                self.play(*anims)

            anims = []
            for i, dot_prod in enumerate(dot_prods):
                tmp = mat1_3d[i].copy().set_opacity(1).set_color(to_green(7, total_n)).deactivate_depth_test()
                anims.extend([Transform(dot_prod[-1], tmp, remover=True), Transform(mat1_3d[i], tmp)])
            self.play(*anims)

            self.play(VGroup(*mat2_3d).animate.set_opacity(0.3), VGroup(*mat3_3d).animate.set_opacity(0.3))


        #show tensor cores
        with self.voiceover(text="""Previously each thread was calculating one element of the output matrix, in here the whole warp
                            is working in synchronization to produce one output tile. """) as trk:
            for tile in range(1, n_tiles):
                mat2_3d = mat2_tiles[tile][0]
                mat3_3d = mat3_tiles[0][tile]
                self.play(VGroup(*mat2_3d).animate.set_opacity(1), VGroup(*mat3_3d).animate.set_opacity(1))
                anims = [] 
                dot_prods = []
                cs = []
                for j in range(tile_n):
                    for k in range(tile_n):
                        run_time = 1

                        v1 = [mat2_3d[i*8 + k] for i in range(8)]
                        v2 = [mat3_3d[j*8 + i] for i in range(8)]

                        dot_prod = []
                        for i in range(8):
                            pos = mat1_3d[j*8 + k].get_center().copy()
                            pos[2] = v1[i].get_center()[2]
                            dot_prod.append(VCube(fill_color=YELLOW, side_length=1).move_to(pos))
                        dot_prods.append(dot_prod)

                        for i in range(8):
                            c1 = v1[i].copy()
                            c2 = v2[i].copy()
                            cs.extend([c1, c2])
                            anims.extend([Transform(cs[-2], dot_prod[i], remover=True),ReplacementTransform(cs[-1], dot_prod[i])])

                self.play(*anims, run_time=run_time)

                #visualize accumulate
                anims = []
                for i, dot_prod in enumerate(dot_prods):
                    run_time = 0.5
                    tmp = mat1_3d[i].copy().set_color(to_green(tile*8 + 7, total_n))
                    anims.extend([Transform(x, tmp, remover=True) for x in dot_prod] + [Transform(mat1_3d[i], tmp)])
                self.play(*anims)
                self.play(VGroup(*mat2_3d).animate.set_opacity(0.3), VGroup(*mat3_3d).animate.set_opacity(0.3))

        self.wait()

        self.play(VGroup(*mat1_3d).animate.set_color(GREY).set_opacity(0.1))

        #play full matmul
        with self.voiceover(text="""We can than launch as many warps as there are output tiles in the output matrix to perform a matrix multiplication
                            on our whole matrix using tensor cores.""") as trk:
            for tile_o in range(n_tiles):
                anims1 = []
                anims2 = [] 
                anims3 = []
                anims4 = [] 
                dot_prods = []
                cs = []
                for tile_i in range(n_tiles):
                    mat2_3d = mat2_tiles[tile_o][tile_i]
                    for tile_j in range(n_tiles):
                        mat3_3d = mat3_tiles[tile_j][tile_o]
                        mat1_3d = mat1_tiles[tile_j][tile_i]
                        anims1.extend([VGroup(*mat2_3d).animate.set_opacity(1), VGroup(*mat3_3d).animate.set_opacity(1)])
                        anims3.extend([VGroup(*mat2_3d).animate.set_opacity(0.3), VGroup(*mat3_3d).animate.set_opacity(0.3)])
                        dot_prod = []
                        for j in range(8):
                            for k in range(8):
                                v1 = [mat2_3d[i*8 + k] for i in range(8)]
                                v2 = [mat3_3d[j*8 + i] for i in range(8)]

                                for i in range(8):
                                    pos = mat1_3d[j*8 + k].get_center().copy()
                                    pos[2] = v1[i].get_center()[2] - (43) + i * 6 if tile_o == 0 and tile_i == 0 else v1[i].get_center()[2]
                                    dot_prod.append(VCube(fill_color=YELLOW, side_length=1).move_to(pos))

                        acc = VGroup(*dot_prod)

                        anims2.extend([Transform(VGroup(*mat2_3d).copy(), acc, remover=True),ReplacementTransform(VGroup(*mat3_3d).copy(), acc)])

                        #visualize accumulate
                        run_time = 0.5
                        mat_group = VGroup(*mat1_3d)
                        tmp = mat_group.copy().set_color(to_green(tile_o*8 + 7, total_n)).set_opacity(1)
                        anims4.extend([Transform(acc, tmp, remover=True), Transform(mat_group, tmp)])

                self.play(*anims1, run_time=run_time)
                self.play(*anims2, run_time=run_time)

                self.play(*anims4)
                self.play(*anims3)
        self.wait(1)
        self.play(*[FadeOut(x) for x in self.mobjects])

class TensorCoresCode(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService(transcription_model="base")
            )
        self.voiceovers_in_embed = True
        # init scene
        total_n = 32
        tile_n = 8
        self.play(*[FadeOut(x) for x in self.mobjects])
        mat1_f = [Square(stroke_width=0, fill_color=GREEN, fill_opacity=0.5) for _ in range(total_n*total_n)]
        mat2_f = [Square(stroke_width=0, fill_color=BLUE, fill_opacity=0.5) for _ in range(total_n*total_n)]
        mat3_f = [Square(stroke_width=0, fill_color=ORANGE, fill_opacity=0.5) for _ in range(total_n*total_n)]


        g1 = Group(*mat1_f).arrange_in_grid(total_n, total_n, buff=4).move_to(ORIGIN, aligned_edge=UL).shift(25*UP + 15*LEFT)
        g2 = Group(*mat2_f).arrange_in_grid(total_n, total_n, buff=4).next_to(g1, UP, buff = 2)
        g3 = Group(*mat3_f).arrange_in_grid(total_n, total_n, buff=4).next_to(g1, LEFT, buff = 2)

        self.frame.set_shape(183, 103)
        self.frame.move_to([-7, 24, 0])

        #accumulate
        def to_green(step, total = 6):
            percentage_red = 1 - step/total
            red = int(255*percentage_red + 131 * (1-percentage_red)) 
            return f"#{red:02x}C167"

            
        #move to 3d
        mat1_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()) for x in mat1_f]
        mat2_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()).shift(2*IN) for x in mat2_f]
        mat3_3d_f = [VCube(fill_color=x.get_color(), fill_opacity=0.3).move_to(x.get_center()).shift(2*IN) for x in mat3_f]


        #rotate matrices
        self.play(self.frame.animate.set_euler_angles(-2.24045432,  1.17009916,  1.86961547))

        mat1_3d_f_g = VGroup(*mat1_3d_f)
        mat2_3d_f_g = VGroup(*mat2_3d_f)
        mat3_3d_f_g = VGroup(*mat3_3d_f)


        mat2_3d_f_g.rotate(radians(90), axis=LEFT, about_edge=DOWN)
        mat3_3d_f_g.rotate(radians(90), axis=DOWN, about_edge=RIGHT)

        n_tiles = total_n // tile_n

        mat1_tiles = []
        mat2_tiles = []
        mat3_tiles = []

        for tile_x in range(0, total_n, tile_n):
            outer_tile1 = []
            outer_tile2 = []
            outer_tile3 = []
            for tile_y in range(0, total_n, tile_n):
                tile1 = []
                tile2 = []
                tile3 = []
                for x in range(tile_n):
                    for y in range(tile_n):
                        r = tile_x + x
                        c = tile_y + y
                        tile1.append(mat1_3d_f[r*total_n + c])

                        r = total_n - tile_n - tile_x + x
                        c = tile_y + y
                        tile2.append(mat2_3d_f[r*total_n + c])

                        r = tile_x + x
                        c = total_n - tile_n - tile_y + y
                        tile3.append(mat3_3d_f[r*total_n + c])
                outer_tile1.append(tile1)
                outer_tile2.append(tile2)
                outer_tile3.append(tile3)
            mat1_tiles.append(outer_tile1)
            mat2_tiles.append(outer_tile2)
            mat3_tiles.append(outer_tile3)

        a_tile1_1 = mat1_tiles[0][0].copy()
        a_tile1_2 = mat1_tiles[0][1].copy()
        a_tile1_3 = mat1_tiles[1][0].copy()
        a_tile1_4 = mat1_tiles[1][1].copy()

        a_tile2_1 = mat2_tiles[0][0].copy()
        a_tile2_2 = mat2_tiles[0][1].copy()
        a_tile2_3 = mat2_tiles[1][0].copy()
        a_tile2_4 = mat2_tiles[1][1].copy()

        a_tile3_1 = mat3_tiles[0][0].copy()
        a_tile3_2 = mat3_tiles[0][1].copy()
        a_tile3_3 = mat3_tiles[1][0].copy()
        a_tile3_4 = mat3_tiles[1][1].copy()

        def nicely_animate(mobjects, start_dir):
            def dist(mobj, point):
                return np.sqrt(np.sum((mobj.get_center() - point)**2))
            nicely_animated = []
            radius = 1
            while sum([len(x) for x in nicely_animated]) != len(mobjects):
                batch = []
                for mobj in mobjects:
                    if dist(mobj, start_dir) < radius and not any(mobj in x for x in nicely_animated): 
                        batch.append(mobj)
                nicely_animated.append(batch)
                radius += 5

            return nicely_animated

        def lagged_fade(mobjects, start_dir, fade_in, **kwargs):
            nicely_animated = nicely_animate(mobjects, start_dir)
            anim = FadeIn if fade_in else FadeOut
            anims = [AnimationGroup(*[anim(y) for y in x]) for x in nicely_animated if len(x) > 0]
            return LaggedStart(*anims, **kwargs)

        def lagged_select(mobjects, start_dir, select, **kwargs):
            nicely_animated = nicely_animate(mobjects, start_dir)
            anims = [AnimationGroup(*[y.animate.set_opacity(0.6 if select else 0.3) for y in x]) for x in nicely_animated if len(x) > 0]
            return LaggedStart(*anims, **kwargs)

        crossing = (mat1_3d_f_g.get_corner(UL) + mat2_3d_f_g.get_corner(OUT+LEFT) + mat3_3d_f_g.get_corner(OUT+UP))/3

        def create_braces(m, k, n, buff=1):
            tiles_in_m = m // tile_n
            tmp = []
            for i in range(tiles_in_m):
                tmp += mat2_tiles[0][i]
            tmp = VGroup(*tmp)
            
            b1 = Brace(tmp, stroke_width=5, font_size=1500, direction=UP).next_to(crossing, RIGHT, buff=buff, aligned_edge=DOWN)
            t1 = Tex("M", font_size=1000).next_to(b1, UP, buff=2)

            tiles_in_k = k // tile_n
            tmp = []
            for i in range(tiles_in_k):
                tmp += mat2_tiles[0][i]
            tmp = VGroup(*tmp)

            b2 = Brace(tmp, stroke_width=5, font_size=1500, direction=UP).rotate(-radians(90), axis=UP).next_to(crossing, IN, buff=buff, aligned_edge=DOWN)
            t2 = Tex("K", font_size=1000).rotate(-radians(90), axis=UP).next_to(b2, UP, buff=2)

            tiles_in_n = n // tile_n
            tmp = []
            for i in range(tiles_in_n):
                tmp += mat1_tiles[i][0]
            tmp = VGroup(*tmp)
            print(tiles_in_m, tiles_in_n, tiles_in_k)
            b3 = Brace(tmp, stroke_width=5, font_size=1500, direction=LEFT).next_to(crossing, DOWN, buff=buff, aligned_edge=RIGHT)
            t3 = Tex("N", font_size=1000).next_to(b3, LEFT, buff=2)
            return (b1, b2, b3), (t1, t2, t3)


        #create first shape
        # a = m x k
        # b = k x n
        # acc = m x n
        self.frame.set_shape(183, 103)
        self.frame.move_to([-7, 24, 0])
        with self.voiceover(text="""Whan programming tensor cores, we first need to specify the shapes of our input and output matrices.
                            Those are reffered to as M N and K
                            """) as trk:
            self.play(lagged_fade(mat1_3d_f + mat2_3d_f + mat3_3d_f,
                                  crossing, True, lag_ratio=0.02))
            self.play(mat2_3d_f_g.animate.shift(4*IN).shift(4*UP), mat3_3d_f_g.animate.shift(4*IN).shift(4*LEFT))

        crossing = (mat1_3d_f_g.get_corner(UL) + mat2_3d_f_g.get_corner(OUT+LEFT) + mat3_3d_f_g.get_corner(OUT+UP))/3
        braces, texts = create_braces(32, 32, 32)

        #Show M and N
        with self.voiceover(text="""M is the common dimension between matrix A and our accumulator<bookmark mark='1'/>
                            and N is shared between matrix 2 and the accumulator""") as trk:
            self.play(self.frame.animate.set_shape(427, 240)\
                    .move_to([ 8.3777702e+01, -6.0713127e+01, -1.1432872e-18])\
                    .set_euler_angles(-3.88572500e-03,  1.35525272e-20,  0.00000000e+00))
            self.play(ShowCreation(braces[0]), Write(texts[0]))
            self.wait_until_bookmark("1")
            self.play(ShowCreation(braces[2]), Write(texts[2]))

        #Show K
        with self.voiceover(text="""And K is the dimension that is the same between our input matrices""") as trk:
            self.play(self.frame.animate.set_shape(283, 159)\
                    .move_to([-7, 24, 0])\
                    .set_euler_angles(-2.24045432,  1.17009916,  1.86961547))
            self.play(ShowCreation(braces[1]), Write(texts[1]))

            
        with self.voiceover(text="""Our tensor cores can operate on three different shapes assuming that we're using half precision""") as trk:
            pass

        def get_transforms(braces, texts, new_braces, new_texts):
            anims = []
            for x, y in zip(braces + texts, new_braces + new_texts):
                anims.append(Transform(x, y))
            return anims

        #16by16by16
        with self.voiceover(text="""16 by 16 by 16""") as trk:

            self.play(lagged_select(mat1_tiles[0][0] + mat1_tiles[0][1] +
                                  mat1_tiles[1][0] + mat1_tiles[1][1] +

                                  mat2_tiles[0][0] + mat2_tiles[0][1] +
                                  mat2_tiles[1][0] + mat2_tiles[1][1] +

                                  mat3_tiles[0][0] + mat3_tiles[0][1] +
                                  mat3_tiles[1][0] + mat3_tiles[1][1],
                                  crossing, True, lag_ratio=0.02),
                      *get_transforms(braces, texts, *create_braces(16,16,16)))

        
        #create next shape
        with self.voiceover(text="""32 by 8 by 16""") as trk:
            self.play(lagged_select(mat1_tiles[0][2] + mat1_tiles[0][3] +
                                  mat1_tiles[1][2] + mat1_tiles[1][3] +
                                  mat2_tiles[0][2] + mat2_tiles[0][3], 
                                  crossing, True),
                      lagged_select(mat2_tiles[1][0] + mat2_tiles[1][1],
                                  mat2_3d_f_g.get_corner(LEFT+IN), False),

                      lagged_select(mat3_tiles[0][1] + mat3_tiles[1][1],
                                    mat2_3d_f_g.get_corner(LEFT+IN), False),
                      *get_transforms(braces, texts, *create_braces(32,8,16)))

        #create third shape
        with self.voiceover(text="""or 8 by 32 by 16""") as trk:
            self.play(lagged_select(mat1_tiles[0][3] + mat1_tiles[0][1] + mat1_tiles[0][2] +
                                  mat1_tiles[1][3] + mat1_tiles[1][1] + mat1_tiles[1][2],
                                  mat1_3d_f_g.get_corner(RIGHT+UP), False),
                      lagged_select(mat2_tiles[1][0] + mat2_tiles[2][0] + mat2_tiles[3][0],
                                  mat2_3d_f_g.get_corner(OUT+LEFT), True),
                      lagged_select(mat2_tiles[0][1] + mat2_tiles[0][2] + mat2_tiles[0][3],
                                  mat2_3d_f_g.get_corner(RIGHT), False),
                      lagged_select(mat3_tiles[0][1] + mat3_tiles[0][2] + mat3_tiles[0][3] +
                                  mat3_tiles[1][1] + mat3_tiles[1][2] + mat3_tiles[1][3],
                                  crossing, True),
                      *get_transforms(braces, texts, *create_braces(8,32,16)))
        
        
        # show accumulation 
        with self.voiceover(text="""For this example let's work with 16 by 16 by 16 matrices""") as trk:
            self.play(*[FadeOut(x) for x in braces + texts],
                        lagged_select(mat1_tiles[0][0] + mat1_tiles[0][1] +
                                      mat1_tiles[1][0] + mat1_tiles[1][1] +

                                      mat2_tiles[0][0] + mat2_tiles[0][1] +
                                      mat2_tiles[1][0] + mat2_tiles[1][1] +

                                      mat3_tiles[0][0] + mat3_tiles[0][1] +
                                      mat3_tiles[1][0] + mat3_tiles[1][1],
                                      crossing, True, lag_ratio=0.02),
                        lagged_select(mat2_tiles[3][0] + mat2_tiles[2][0] +

                                      mat3_tiles[0][3] + mat3_tiles[1][3] +
                                      mat3_tiles[0][2] + mat3_tiles[1][2],
                                      mat2_3d_f_g.get_corner(IN+LEFT), False, lag_ratio=0.02))

        # initialize acc
        with self.voiceover(text="""We first initialize our accumulator to zeros""") as trk:
            self.play(*[x.animate.set_color(GREY) for x in 
                        mat1_tiles[0][0] + mat1_tiles[0][1] +
                        mat1_tiles[1][0] + mat1_tiles[1][1]])

        # load from global to reg 
        with self.voiceover(text="""Then after performing a boundary check we load <bookmark mark='1'/>matrix A and matrix B from 
                            global memory to registers""") as trk:
            self.wait_until_bookmark("1")
            self.play(*[x.animate.set_opacity(1) for x in 
                                      mat2_tiles[0][0] + mat2_tiles[0][1] +
                                      mat2_tiles[1][0] + mat2_tiles[1][1] +

                                      mat3_tiles[0][0] + mat3_tiles[0][1] +
                                      mat3_tiles[1][0] + mat3_tiles[1][1]])

        # show tensor core matmul
        mat1_3d = mat1_tiles[0][0] + mat1_tiles[0][1] + mat1_tiles[1][0] + mat1_tiles[1][1]
        mat2_3d = mat2_tiles[0][0] + mat2_tiles[0][1] + mat2_tiles[1][0] + mat2_tiles[1][1]
        mat3_3d =  mat3_tiles[0][0] + mat3_tiles[0][1] + mat3_tiles[1][0] + mat3_tiles[1][1]
        dot_prod = []
        for j in range(16):
            for k in range(16):
                v1 = [mat2_3d[i*16 + k] for i in range(16)]
                v2 = [mat3_3d[j*16 + i] for i in range(16)]

                for i in range(16):
                    pos = mat1_3d[j*16 + k].get_center().copy()
                    pos[2] = v1[i].get_center()[2]# - (43) + i * 6 if tile_o == 0 and tile_i == 0 else v1[i].get_center()[2]
                    dot_prod.append(VCube(fill_color=YELLOW, side_length=1).move_to(pos))

        acc = VGroup(*dot_prod)

        anims = [Transform(VGroup(*mat2_3d).copy(), acc, remover=True),ReplacementTransform(VGroup(*mat3_3d).copy(), acc)]
        mat_group = VGroup(*mat1_3d)
        tmp = mat_group.copy().set_color(to_green(15, total_n)).set_opacity(1)
        anims2 = [Transform(acc, tmp, remover=True), Transform(mat_group, tmp)]


        run_time = 1
        with self.voiceover(text="""We then perform a tensor core tiled matrix multiplication and store <bookmark mark='1'/>the result in our accumulator""") as trk:
            self.play(*anims, run_time=run_time)
            self.wait_until_bookmark("1")
            self.play(*anims2, run_time=run_time)
        self.play(VGroup(*mat2_3d).animate.set_opacity(0.3), VGroup(*mat3_3d).animate.set_opacity(0.3))

        # show next timed matmul
        mat2_3d = mat2_tiles[2][0] + mat2_tiles[2][1] + mat2_tiles[3][0] + mat2_tiles[3][1]
        mat3_3d =  mat3_tiles[0][2] + mat3_tiles[0][3] + mat3_tiles[1][2] + mat3_tiles[1][3]

       
        with self.voiceover(text="""We then advance our pointer to the next tile""") as trk:
            self.play(VGroup(*mat2_3d).animate.set_opacity(0.5), VGroup(*mat3_3d).animate.set_opacity(0.5))

        with self.voiceover(text="""Load them from memory""") as trk:
            self.play(VGroup(*mat2_3d).animate.set_opacity(1), VGroup(*mat3_3d).animate.set_opacity(1))
        dot_prod = []
        for j in range(16):
            for k in range(16):
                v1 = [mat2_3d[i*16 + k] for i in range(16)]
                v2 = [mat3_3d[j*16 + i] for i in range(16)]

                for i in range(16):
                    pos = mat1_3d[j*16 + k].get_center().copy()
                    pos[2] = v1[i].get_center()[2]# - (43) + i * 6 if tile_o == 0 and tile_i == 0 else v1[i].get_center()[2]
                    dot_prod.append(VCube(fill_color=YELLOW, side_length=1).move_to(pos))

        acc = VGroup(*dot_prod)

        anims = [Transform(VGroup(*mat2_3d).copy(), acc, remover=True),ReplacementTransform(VGroup(*mat3_3d).copy(), acc)]
        mat_group = VGroup(*mat1_3d)
        tmp = mat_group.copy().set_color(GREEN).set_opacity(1)
        anims2 = [Transform(acc, tmp, remover=True), Transform(mat_group, tmp)]

        with self.voiceover(text="""And perform another tiled matrix multiplication for the final result""") as trk:
            self.play(*anims, run_time=run_time)
            self.wait(1)
            self.play(*anims2, run_time=run_time)
        self.wait(2)
        self.play(*[FadeOut(x) for x in self.mobjects])
