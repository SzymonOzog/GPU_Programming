from manim import *
from manim_voiceover import VoiceoverScene
from manim.mobject.text.text_mobject import remove_invisible_chars
from manim_voiceover.services.recorder import RecorderService
from manim_voiceover.services.gtts import GTTSService
import math

class TensorCoresGraph(VoiceoverScene):
    def construct(self):
        self.set_speech_service(
            # RecorderService(transcription_model="base")
            GTTSService(transcription_model="base")
            )
        normal_times = [0.0108544, 0.0195504, 0.0610426, 0.502562, 3.72811, ]
        tiled_times = [0.0186675, 0.0334739, 0.0679533, 0.153344, 0.822092, ]
        cublas_times = [0.00583872, 0.0082176, 0.027305, 0.0733114, 0.576111, ]

        
 
        def to_flops(times, ns):
            return [(2*n*n*n)/(t*1e6) for t, n in zip(times, ns)]

        ps = list(range(8, 9+len(tiled_times)))

        n = [2**p for p in ps] 

        normal_flops = to_flops(normal_times, n)
        tiled_flops = to_flops(tiled_times, n)
        cublas_flops = to_flops(cublas_times, n)

        ax = Axes(
            x_range=[ps[0] , ps[-1], 2],
            y_range=[0, 380000, 50000],
            x_axis_config={"scaling": LogBase(2)},
            axis_config={"include_numbers": True}).shift(0.4*RIGHT)

        labels = ax.get_axis_labels(x_label="n", y_label="Throughput[GFLOPS]")

        normal_graph = ax.plot_line_graph(
            x_values=n,
            y_values=normal_flops,
            line_color=RED,
            add_vertex_dots=False
        )

        tiled_graph = ax.plot_line_graph(
            x_values=n,
            y_values=tiled_flops,
            line_color=BLUE,
            add_vertex_dots=False
        )

        cublas_graph = ax.plot_line_graph(
            x_values=n,
            y_values=cublas_flops,
            line_color=GREEN,
            add_vertex_dots=False
        )
        normal_label = Text("Tensor Cores", font_size=32, color=RED).next_to(labels[1], DOWN, aligned_edge=LEFT)
        normal_label.shift(0.1*DOWN).align_to(labels[0], RIGHT)
        tiled_label = Text("Hierarchical\nTiling", font_size=32, color=BLUE).next_to(normal_label, DOWN, aligned_edge=LEFT) 
        cublas_label = Text("cuBLAS", font_size=32, color=GREEN).next_to(tiled_label, DOWN, aligned_edge=LEFT)
        with self.voiceover(text="""Last time we measured how much FLOPs we were gettin with our tensor cores kernel""") as trk:
            self.play(Create(ax), Write(labels))
            self.play(Create(normal_graph))
            self.play(Write(normal_label))


        theoretical_max_tc = ax.plot_line_graph(
            x_values=n,
            y_values=([330000] * len(cublas_flops)),
            line_color=GOLD,
            add_vertex_dots=False
        )

        theoretical_max_tc_t = Text("Tensor Core theoretical maximum (330 TFLOPS)", color=GOLD, font_size=24).next_to(theoretical_max_tc, DOWN)

        with self.voiceover(text="""and compared it to what is the theoretical maximum of our GPU""") as trk:
            self.play(Create(theoretical_max_tc), Write(theoretical_max_tc_t))

        with self.voiceover(text="""By utilizing hierarchical tiling, we can get much higher throughput, even up to 4.5x higher than 
                            we were getting with our original kernel""") as trk:
            self.play(Create(tiled_graph))
            self.play(Write(tiled_label))
        self.wait(1)

        with self.voiceover(text="""But that's still off from our theoretical maximum, and this is something that we will never achieve""") as trk:
            pass

        with self.voiceover(text="""To see what is the state of the art, we can run cublas kernels from nvidia, this will be our reference point from now on""") as trk:
            self.play(Create(cublas_graph))
            self.play(Write(cublas_label))
        self.wait(1)

        with self.voiceover(text="""And in the next episode, we'll once again move closer to this line""") as trk:
            pass



# template<int SM_TILES, int OUT_TILES>
# __global__ void tensor_core_matmul_reg_smem(int n_elem, half* a, half* b, half* c)
# {
#     const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
#     const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
#     const int32_t laneM = threadIdx.x/32;
#     const int32_t laneN = threadIdx.y;
#
#     extern __shared__ char smem[];
#
#     half (*a_smem)[WMMA_MKN*WMMA_MKN]
#         = reinterpret_cast<half(*)[WMMA_MKN*WMMA_MKN]>(smem);
#     half (*b_smem)[WMMA_MKN*WMMA_MKN]
#         = reinterpret_cast<half(*)[WMMA_MKN*WMMA_MKN]>(
#                 smem + SM_TILES*WMMA_MKN*WMMA_MKN*sizeof(half));
#
#     nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> a_frag[OUT_TILES];
#     nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_MKN, WMMA_MKN, WMMA_MKN, half, layout> b_frag;
#     nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc[OUT_TILES][OUT_TILES];
#
#     for(int32_t i = 0; i<OUT_TILES; i++)
#         for(int32_t j = 0; j<OUT_TILES; j++)
#             nvcuda::wmma::fill_fragment(acc[i][j], 0);
#
#     const int32_t matrix_a_row = warpM * WMMA_MKN * OUT_TILES;
#     const int32_t matrix_b_col = warpN * WMMA_MKN * OUT_TILES;
#
#     for (int32_t tile = 0; tile < n_elem; tile+=OUT_TILES*WMMA_MKN)
#     {
#         for (int k = 0; k < OUT_TILES; k++)
#         {
#             half* a_curr = a + blockIdx.x*SM_TILES*WMMA_MKN*n_elem + tile + k*WMMA_MKN;
#             half* b_curr = b + (k*WMMA_MKN+tile)*n_elem + blockIdx.y*SM_TILES*WMMA_MKN;
#             for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
#                     i < SM_TILES*WMMA_MKN*WMMA_MKN;
#                     i+=blockDim.x*blockDim.y*8)
#             {
#                 reinterpret_cast<float4*>(&a_smem[i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)])[0]
#                     = reinterpret_cast<float4*>(&a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN])[0];
#                 reinterpret_cast<float4*>(&b_smem[(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)])[0]
#                     = reinterpret_cast<float4*>(&b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)])[0];
#             }
#
#             for (int i = threadIdx.y * blockDim.x + threadIdx.x;
#                     i < SM_TILES*WMMA_MKN*WMMA_MKN;
#                     i+=blockDim.x*blockDim.y)
#             {
#                 a_smem[i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)] = a_curr[(i/WMMA_MKN)*n + i%WMMA_MKN];
#                 b_smem[(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)] = b_curr[(i/(SM_TILES*WMMA_MKN))*n + i%(SM_TILES*WMMA_MKN)];
#             }
#
#             __syncthreads();
#             for (int n = 0; n < OUT_TILES; n++)
#             {
#                 int32_t a_row = matrix_a_row + n*WMMA_MKN;
#                 int32_t a_col = tile + k*WMMA_MKN;
#                 if(a_row < n_elem && a_col < n_elem)
#                 {
#                     nvcuda::wmma::load_matrix_sync(a_frag[n], a_smem[laneM*OUT_TILES + n], WMMA_MKN);
#                 }
#             }
#             for (int n = 0; n < OUT_TILES; n++)
#             {
#                 int32_t b_col = matrix_b_col + (n)*WMMA_MKN;
#                 int32_t b_row = tile + k*WMMA_MKN;
#                 if (b_row < n_elem && b_col < n_elem)
#                 {
#                     nvcuda::wmma::load_matrix_sync(b_frag, b_smem[laneN*OUT_TILES + n], WMMA_MKN);
#                     for (int m = 0; m < OUT_TILES; m++)
#                     {
#                         nvcuda::wmma::mma_sync(acc[m][n], a_frag[m], b_frag, acc[m][n]);
#                     }
#                 }
#             }
#             __syncthreads();
#         }
#     }
#
#     for(int32_t i = 0; i<OUT_TILES; i++)
#     {
#         int32_t output_row = matrix_a_row + i*WMMA_MKN;
#         for(int32_t j = 0; j<OUT_TILES; j++)
#         {
#             int32_t output_col = matrix_b_col + j*WMMA_MKN;
#             if (output_row < n_elem && output_col < n_elem)
#             {
#                 nvcuda::wmma::store_matrix_sync(c + output_row * n_elem + output_col, acc[i][j], n_elem, nvcuda::wmma::mem_row_major);
#             }
#         }
#     }
# }

class TensorCoresCode2(Scene):
    def construct(self):
        timestamps = [i for i in range(1, 2048)]
        def wait_timestamp():
            self.wait(timestamps.pop(0) - self.last_t)

        code = """nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_MKN, WMMA_MKN, WMMA_MKN, half> acc[OUT_TILES][OUT_TILES];
for(int32_t i = 0; i<OUT_TILES; i++)
    for(int32_t j = 0; j<OUT_TILES; j++)
        nvcuda::wmma::fill_fragment(acc[i][j], 0);"""
        def create_code(code):
            code_obj = Code(code=code, tab_width=2, language="c++", style='monokai', margin=0.1, line_spacing=0.7, insert_line_no=False, font_size=12, corner_radius=0.1)
            code_obj.code = remove_invisible_chars(code_obj.code)
            return code_obj
        code_obj = create_code(code)

        self.play(Create(code_obj))
        wait_timestamp()

        code = """for (int i = threadIdx.y * blockDim.x + threadIdx.x;
     i < SM_TILES*WMMA_MKN*WMMA_MKN;
     i+=blockDim.x*blockDim.y)
{
    a_smem[i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)] =
        a_curr[(i/WMMA_MKN)*n + i%WMMA_MKN];
    b_smem[(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)] = 
        b_curr[(i/(SM_TILES*WMMA_MKN))*n + i%(SM_TILES*WMMA_MKN)];
}"""
        self.play(Transform(code_obj, create_code(code)))
        wait_timestamp()

        code = """for (int i = (threadIdx.y * blockDim.x + threadIdx.x)*8;
             i < SM_TILES*WMMA_MKN*WMMA_MKN;
             i+=blockDim.x*blockDim.y*8)
{
    reinterpret_cast<float4*>(&a_smem[i/(WMMA_MKN*WMMA_MKN)][i%(WMMA_MKN*WMMA_MKN)])[0]
        = reinterpret_cast<float4*>(&a_curr[(i/WMMA_MKN)*n_elem + i%WMMA_MKN])[0];
    reinterpret_cast<float4*>(&b_smem[(i/WMMA_MKN)%SM_TILES][(i/(SM_TILES*WMMA_MKN))*WMMA_MKN + i%(WMMA_MKN)])[0]
        = reinterpret_cast<float4*>(&b_curr[(i/(SM_TILES*WMMA_MKN))*n_elem + i%(SM_TILES*WMMA_MKN)])[0];
}"""
        self.play(Transform(code_obj, create_code(code)))
        wait_timestamp()
