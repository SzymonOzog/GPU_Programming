from manimlib import *

class TensorCores(Scene):
    def construct(self):
        # init scene
        mat1 = [Square(stroke_width=0, fill_color=GREEN, fill_opacity=0.5) for _ in range(64)]
        mat2 = [Square(stroke_width=0, fill_color=BLUE, fill_opacity=0.5) for _ in range(64)]
        mat3 = [Square(stroke_width=0, fill_color=ORANGE, fill_opacity=0.5) for _ in range(64)]

        self.frame.set_shape(110, 62)
        self.frame.move_to([4.5, 11, 0])

        g1 = Group(*mat1).arrange_in_grid(8, 8)
        # self.play(ShowCreation(g1))

        g2 = Group(*mat2).arrange_in_grid(8, 8).next_to(g1, UP, buff = 2)
        self.play(ShowCreation(g2))

        g3 = Group(*mat3).arrange_in_grid(8, 8).next_to(g1, LEFT, buff = 2)
        self.play(ShowCreation(g3))

        #highlight vectors
        v1 = [mat2[i*8] for i in range(8)]
        v2 = [mat3[i] for i in range(8)]

        self.play(*[v.animate.set_opacity(1) for v in v1])
        self.play(*[v.animate.set_opacity(1) for v in v2])

        #interm vector
        v3 = []
        for i in range(8):
            p1 = v2[i].get_center()
            p2 = v1[i].get_center()
            pos = [p1[0], p2[1], p2[2]] 
            v3.append(Square(stroke_width=0, fill_color=YELLOW, fill_opacity=1).move_to(pos))

        for i in range(8):
            vv1 = v1[i].copy()
            vv2 = v2[i].copy()
            self.play(vv1.animate.move_to(v3[i].get_center()),
                      vv2.animate.move_to(v3[i].get_center()))
            self.play(FadeOut(vv1), FadeOut(vv2), FadeIn(v3[i]))

        #accumulate
        def to_green(step):
            percentage_red = 1 - step/6
            red = int(255*percentage_red)
            return f"#{red:02x}ff00"

            
        for i in range(7):
            self.play(v3[i].animate.move_to(v3[i+1]),
                      v3[i+1].animate.set_color(to_green(i)))
            self.remove(v3[i])

        self.play(v3[-1].animate.move_to(mat1[0].get_center()))

