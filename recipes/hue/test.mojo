from hue import Color
from hue.color import hsv, Luv, Luv_white_ref, D50, hcl, hcl_white_ref
import testing

# Note: the XYZ, L*a*b*, etc. are using D65 white and D50 white if postfixed by "50".
# See http:#www.brucelindbloom.com#index.html?ColorCalcHelp.html
# For d50 white, no "adaptation" and the sRGB model are used in colorful
# HCL values form http:#www.easyrgb.com#index.php?X=CALC and missing ones hand-computed from lab ones
@value
struct TestValues(CollectionElement):
    var c: Color
    var hsl: List[Float64]
    var hsv: List[Float64]
    var hex: String
    var xyz: List[Float64]
    var xyy: List[Float64]
    var lab: List[Float64]
    var lab50: List[Float64]
    var luv: List[Float64]
    var luv50: List[Float64]
    var hcl: List[Float64]
    var hcl50: List[Float64]
    var rgba: List[UInt32]
    var rgb255: List[UInt8]


# TODO: Need to wrap the test data in a struct because of a compiler bug leading to issues with mojo test undefined behavior.
@value
struct Data(CollectionElement):
    var vals: List[TestValues]

    fn __init__(inout self):
        self.vals = List[TestValues](
            TestValues(
                Color(1.0, 1.0, 1.0),
                List[Float64](0.0, 0.0, 1.00),
                List[Float64](0.0, 0.0, 1.0),
                String("#ffffff"),
                List[Float64](0.950470, 1.000000, 1.088830),
                List[Float64](0.312727, 0.329023, 1.000000),
                List[Float64](1.000000, 0.000000, 0.000000),
                List[Float64](1.000000, -0.023881, -0.193622),
                List[Float64](1.00000, 0.00000, 0.00000),
                List[Float64](1.00000, -0.14716, -0.25658),
                List[Float64](0.0000, 0.000000, 1.000000),
                List[Float64](262.9688, 0.195089, 1.000000),
                List[UInt32](65535, 65535, 65535, 65535),
                List[UInt8](255, 255, 255),
            ),
            TestValues(
                Color(0.5, 1.0, 1.0),
                List[Float64](180.0, 1.0, 0.75),
                List[Float64](180.0, 0.5, 1.0),
                String("#80ffff"),
                List[Float64](0.626296, 0.832848, 1.073634),
                List[Float64](0.247276, 0.328828, 0.832848),
                List[Float64](0.931390, -0.353319, -0.108946),
                List[Float64](0.931390, -0.374100, -0.301663),
                List[Float64](0.93139, -0.53909, -0.11630),
                List[Float64](0.93139, -0.67615, -0.35528),
                List[Float64](197.1371, 0.369735, 0.931390),
                List[Float64](218.8817, 0.480574, 0.931390),
                List[UInt32](32768, 65535, 65535, 65535),
                List[UInt8](128, 255, 255),
            ),
            TestValues(
                Color(1.0, 0.5, 1.0),
                List[Float64](300.0, 1.0, 0.75),
                List[Float64](300.0, 0.5, 1.0),
                String("#ff80ff"),
                List[Float64](0.669430, 0.437920, 0.995150),
                List[Float64](0.318397, 0.208285, 0.437920),
                List[Float64](0.720892, 0.651673, -0.422133),
                List[Float64](0.720892, 0.630425, -0.610035),
                List[Float64](0.72089, 0.60047, -0.77626),
                List[Float64](0.72089, 0.49438, -0.96123),
                List[Float64](327.0661, 0.776450, 0.720892),
                List[Float64](315.9417, 0.877257, 0.720892),
                List[UInt32](65535, 32768, 65535, 65535),
                List[UInt8](255, 128, 255),
            ),
            TestValues(
                Color(1.0, 1.0, 0.5),
                List[Float64](60.0, 1.0, 0.75),
                List[Float64](60.0, 0.5, 1.0),
                String("#ffff80"),
                List[Float64](0.808654, 0.943273, 0.341930),
                List[Float64](0.386203, 0.450496, 0.943273),
                List[Float64](0.977637, -0.165795, 0.602017),
                List[Float64](0.977637, -0.188424, 0.470410),
                List[Float64](0.97764, 0.05759, 0.79816),
                List[Float64](0.97764, -0.08628, 0.54731),
                List[Float64](105.3975, 0.624430, 0.977637),
                List[Float64](111.8287, 0.506743, 0.977637),
                List[UInt32](65535, 65535, 32768, 65535),
                List[UInt8](255, 255, 128),
            ),
            TestValues(
                Color(0.5, 0.5, 1.0),
                List[Float64](240.0, 1.0, 0.75),
                List[Float64](240.0, 0.5, 1.0),
                String("#8080ff"),
                List[Float64](0.345256, 0.270768, 0.979954),
                List[Float64](0.216329, 0.169656, 0.270768),
                List[Float64](0.590453, 0.332846, -0.637099),
                List[Float64](0.590453, 0.315806, -0.824040),
                List[Float64](0.59045, -0.07568, -1.04877),
                List[Float64](0.59045, -0.16257, -1.20027),
                List[Float64](297.5843, 0.718805, 0.590453),
                List[Float64](290.9689, 0.882482, 0.590453),
                List[UInt32](32768, 32768, 65535, 65535),
                List[UInt8](128, 128, 255),
            ),
            TestValues(
                Color(1.0, 0.5, 0.5),
                List[Float64](0.0, 1.0, 0.75),
                List[Float64](0.0, 0.5, 1.0),
                String("#ff8080"),
                List[Float64](0.527613, 0.381193, 0.248250),
                List[Float64](0.455996, 0.329451, 0.381193),
                List[Float64](0.681085, 0.483884, 0.228328),
                List[Float64](0.681085, 0.464258, 0.110043),
                List[Float64](0.68108, 0.92148, 0.19879),
                List[Float64](0.68106, 0.82106, 0.02393),
                List[Float64](25.2610, 0.535049, 0.681085),
                List[Float64](13.3347, 0.477121, 0.681085),
                List[UInt32](65535, 32768, 32768, 65535),
                List[UInt8](255, 128, 128),
            ),
            TestValues(
                Color(0.5, 1.0, 0.5),
                List[Float64](120.0, 1.0, 0.75),
                List[Float64](120.0, 0.5, 1.0),
                String("#80ff80"),
                List[Float64](0.484480, 0.776121, 0.326734),
                List[Float64](0.305216, 0.488946, 0.776121),
                List[Float64](0.906026, -0.600870, 0.498993),
                List[Float64](0.906026, -0.619946, 0.369365),
                List[Float64](0.90603, -0.58869, 0.76102),
                List[Float64](0.90603, -0.72202, 0.52855),
                List[Float64](140.2920, 0.781050, 0.906026),
                List[Float64](149.2134, 0.721640, 0.906026),
                List[UInt32](32768, 65535, 32768, 65535),
                List[UInt8](128, 255, 128),
            ),
            TestValues(
                Color(0.5, 0.5, 0.5),
                List[Float64](0.0, 0.0, 0.50),
                List[Float64](0.0, 0.0, 0.5),
                String("#808080"),
                List[Float64](0.203440, 0.214041, 0.233054),
                List[Float64](0.312727, 0.329023, 0.214041),
                List[Float64](0.533890, 0.000000, 0.000000),
                List[Float64](0.533890, -0.014285, -0.115821),
                List[Float64](0.53389, 0.00000, 0.00000),
                List[Float64](0.53389, -0.07857, -0.13699),
                List[Float64](0.0000, 0.000000, 0.533890),
                List[Float64](262.9688, 0.116699, 0.533890),
                List[UInt32](32768, 32768, 32768, 65535),
                List[UInt8](128, 128, 128),
            ),
            TestValues(
                Color(0.0, 1.0, 1.0),
                List[Float64](180.0, 1.0, 0.50),
                List[Float64](180.0, 1.0, 1.0),
                String("#00ffff"),
                List[Float64](0.538014, 0.787327, 1.069496),
                List[Float64](0.224656, 0.328760, 0.787327),
                List[Float64](0.911132, -0.480875, -0.141312),
                List[Float64](0.911132, -0.500630, -0.333781),
                List[Float64](0.91113, -0.70477, -0.15204),
                List[Float64](0.91113, -0.83886, -0.38582),
                List[Float64](196.3762, 0.501209, 0.911132),
                List[Float64](213.6923, 0.601698, 0.911132),
                List[UInt32](0, 65535, 65535, 65535),
                List[UInt8](0, 255, 255),
            ),
            TestValues(
                Color(1.0, 0.0, 1.0),
                List[Float64](300.0, 1.0, 0.50),
                List[Float64](300.0, 1.0, 1.0),
                String("#ff00ff"),
                List[Float64](0.592894, 0.284848, 0.969638),
                List[Float64](0.320938, 0.154190, 0.284848),
                List[Float64](0.603242, 0.982343, -0.608249),
                List[Float64](0.603242, 0.961939, -0.794531),
                List[Float64](0.60324, 0.84071, -1.08683),
                List[Float64](0.60324, 0.75194, -1.24161),
                List[Float64](328.2350, 1.155407, 0.603242),
                List[Float64](320.4444, 1.247640, 0.603242),
                List[UInt32](65535, 0, 65535, 65535),
                List[UInt8](255, 0, 255),
            ),
            TestValues(
                Color(1.0, 1.0, 0.0),
                List[Float64](60.0, 1.0, 0.50),
                List[Float64](60.0, 1.0, 1.0),
                String("#ffff00"),
                List[Float64](0.770033, 0.927825, 0.138526),
                List[Float64](0.419320, 0.505246, 0.927825),
                List[Float64](0.971393, -0.215537, 0.944780),
                List[Float64](0.971393, -0.237800, 0.847398),
                List[Float64](0.97139, 0.07706, 1.06787),
                List[Float64](0.97139, -0.06590, 0.81862),
                List[Float64](102.8512, 0.969054, 0.971393),
                List[Float64](105.6754, 0.880131, 0.971393),
                List[UInt32](65535, 65535, 0, 65535),
                List[UInt8](255, 255, 0),
            ),
            TestValues(
                Color(0.0, 0.0, 1.0),
                List[Float64](240.0, 1.0, 0.50),
                List[Float64](240.0, 1.0, 1.0),
                String("#0000ff"),
                List[Float64](0.180437, 0.072175, 0.950304),
                List[Float64](0.150000, 0.060000, 0.072175),
                List[Float64](0.322970, 0.791875, -1.078602),
                List[Float64](0.322970, 0.778150, -1.263638),
                List[Float64](0.32297, -0.09405, -1.30342),
                List[Float64](0.32297, -0.14158, -1.38629),
                List[Float64](306.2849, 1.338076, 0.322970),
                List[Float64](301.6248, 1.484014, 0.322970),
                List[UInt32](0, 0, 65535, 65535),
                List[UInt8](0, 0, 255),
            ),
            TestValues(
                Color(0.0, 1.0, 0.0),
                List[Float64](120.0, 1.0, 0.50),
                List[Float64](120.0, 1.0, 1.0),
                String("#00ff00"),
                List[Float64](0.357576, 0.715152, 0.119192),
                List[Float64](0.300000, 0.600000, 0.715152),
                List[Float64](0.877347, -0.861827, 0.831793),
                List[Float64](0.877347, -0.879067, 0.739170),
                List[Float64](0.87735, -0.83078, 1.07398),
                List[Float64](0.87735, -0.95989, 0.84887),
                List[Float64](136.0160, 1.197759, 0.877347),
                List[Float64](139.9409, 1.148534, 0.877347),
                List[UInt32](0, 65535, 0, 65535),
                List[UInt8](0, 255, 0),
            ),
            TestValues(
                Color(1.0, 0.0, 0.0),
                List[Float64](0.0, 1.0, 0.50),
                List[Float64](0.0, 1.0, 1.0),
                String("#ff0000"),
                List[Float64](0.412456, 0.212673, 0.019334),
                List[Float64](0.640000, 0.330000, 0.212673),
                List[Float64](0.532408, 0.800925, 0.672032),
                List[Float64](0.532408, 0.782845, 0.621518),
                List[Float64](0.53241, 1.75015, 0.37756),
                List[Float64](0.53241, 1.67180, 0.24096),
                List[Float64](39.9990, 1.045518, 0.532408),
                List[Float64](38.4469, 0.999566, 0.532408),
                List[UInt32](65535, 0, 0, 65535),
                List[UInt8](255, 0, 0),
            ),
            TestValues(
                Color(0.0, 0.0, 0.0),
                List[Float64](0.0, 0.0, 0.00),
                List[Float64](0.0, 0.0, 0.0),
                String("#000000"),
                List[Float64](0.000000, 0.000000, 0.000000),
                List[Float64](0.312727, 0.329023, 0.000000),
                List[Float64](0.000000, 0.000000, 0.000000),
                List[Float64](0.000000, 0.000000, 0.000000),
                List[Float64](0.00000, 0.00000, 0.00000),
                List[Float64](0.00000, 0.00000, 0.00000),
                List[Float64](0.0000, 0.000000, 0.000000),
                List[Float64](0.0000, 0.000000, 0.000000),
                List[UInt32](0, 0, 0, 65535),
                List[UInt8](0, 0, 0),
            ),
        )


def main():
    alias delta = 1.0 / 256.0

    ### RGBA ###
    ############

    def test_creation():
        var data = Data()
        # Test HSV Creation
        for i in range(len(data.vals)):
            var tt = data.vals[i]
            c = hsv(tt.hsv[0], tt.hsv[1], tt.hsv[2])
            testing.assert_true(
                c.almost_equal_rgb(tt.c),
                msg=String("{}. {}.hsv() => ({}), want {} (delta {})").format(
                    str(i), tt.hsv.__str__(), c, tt.c, str(delta)
                ),
            )

            # Test Luv Creation
            c = Luv(tt.luv[0], tt.luv[1], tt.luv[2])
            testing.assert_true(
                c.almost_equal_rgb(tt.c),
                msg=String("{}. Luv({}) => ({}), want {} (delta {})").format(str(i), tt.luv.__str__(), c, tt.c, str(delta)),
            )

            # Test Whiteref Creation
            c = Luv_white_ref(tt.luv50[0], tt.luv50[1], tt.luv50[2], D50)
            testing.assert_true(
                c.almost_equal_rgb(tt.c),
                msg=String("{}. Luv_white_ref({}, D50) => ({}), want {} (delta {})").format(
                    str(i), tt.luv50.__str__(), c, tt.c, str(delta)
                ),
            )

            # Test Hcl Creation
            c = hcl(tt.hcl[0], tt.hcl[1], tt.hcl[2])
            testing.assert_true(
                c.almost_equal_rgb(tt.c),
                msg=String("{}. hcl({}) => ({}), want {} (delta {})").format(str(i), tt.hcl.__str__(), c, tt.c, str(delta)),
            )

            # Test HclWhiteRef Creation
            c = hcl_white_ref(tt.hcl50[0], tt.hcl50[1], tt.hcl50[2], D50)
            testing.assert_true(
                c.almost_equal_rgb(tt.c),
                msg=String("{}. hcl_white_ref({}, D50) => ({}), want {} (delta {})").format(
                    str(i), tt.hcl50.__str__(), c, tt.c, str(delta)
                ),
            )