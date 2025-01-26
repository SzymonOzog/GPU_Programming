from manimlib import *
from math import radians

class TensorCores(Scene):
    def construct(self):
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

        self.play(*[ShowCreation(x) for x in mat2])
        self.play(*[ShowCreation(x) for x in mat3])

        #highlight vectors
        v1 = [mat2[i*8] for i in range(tile_n)]
        v2 = [mat3[i] for i in range(tile_n)]

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
                      vv2.animate.move_to(v3[i].get_center()))
            self.play(FadeOut(vv1), FadeOut(vv2), FadeIn(v3[i]))

        #accumulate
        def to_green(step, total = 6):
            percentage_red = 1 - step/total
            red = int(255*percentage_red + 131 * (1-percentage_red)) 
            return f"#{red:02x}C167"

            
        for i in range(tile_n-1):
            self.play(v3[i].animate.move_to(v3[i+1]),
                      v3[i+1].animate.set_color(to_green(i)))
            self.remove(v3[i])

        self.play(v3[-1].animate.move_to(mat1[0].get_center()))

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


        self.play(*[FadeOut(x)for x in mat1 + mat2 + mat3], 
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
            frame_end.saved_alpha = min(1, frame_end.saved_alpha+dt/25)
            m.interpolate(frame_start, frame_end, frame_end.saved_alpha)
            

        thread_numbers = [Text(str(i)).scale(3).move_to(x.get_center()) for i, x in enumerate(mat1_3d)]
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
                    self.play(Transform(dot_prod[-1], tmp), Transform(mat1_3d[j*8+k], tmp), run_time=run_time)
                    self.frame.add_updater(updater)
                else:
                    run_time = 0.2 if j < 2 else 0.05
                    dot_prod[-1].deactivate_depth_test()
                    tmp = dot_prod[-1].copy().set_color(to_green(7, total_n)).move_to(mat1_3d[j*8+k].get_center()).deactivate_depth_test()
                    self.play(*[Transform(x, tmp, run_time=run_time) for x in dot_prod])
                    self.play(*[Transform(x, tmp) for x in dot_prod], Transform(mat1_3d[j*8+k], tmp), run_time=run_time)

                self.play(Write(thread_numbers[j*tile_n + k]), run_time=0.05)
        self.frame.remove_updater(updater)

        self.play(*[m.animate.set_opacity(0.3) for m in mat3_3d+mat2_3d])

        #Show full matrix
        self.play(FadeIn(mat1_3d_f_g), FadeIn(mat2_3d_f_g), FadeIn(mat3_3d_f_g),
                  self.frame.animate.set_euler_angles(-2.24045432,  1.17009916,  1.86961547).set_shape(329, 187).move_to([-28.3, 8.62, -25.36]),
                  *[FadeOut(t) for t in thread_numbers]
                  )

        # Show tiling
        self.play(self.frame.animate.set_shape(329, 187).move_to([-28.3, 8.62, -25.36]))
        self.play(self.frame.animate.set_euler_angles(-2.24045432,  1.17009916,  1.86961547))

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

        self.play(*anims)

        #play next tile
        tile = 1
        mat2_3d = mat2_tiles[tile][0]
        mat3_3d = mat3_tiles[0][tile]
        self.play(VGroup(*mat2_3d).animate.set_opacity(0.5), VGroup(*mat3_3d).animate.set_opacity(0.5))
        anims = []
        for j in range(tile_n):
            for k in range(tile_n):
                run_time = 0.1

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

                self.play(*[Transform(x, tmp) for x in dot_prod], Transform(mat1_3d[j*8+k], tmp), run_time=run_time)
        self.play(*[m.animate.set_opacity(0.3) for m in mat3_3d+mat2_3d])
