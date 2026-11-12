# Renders text headlessly through the installed package. Text needs the
# packaged Noto fonts and FreeType, so this fails if either can't be found
# after install. The windowed and GPU entry points are only type-checked:
# a CI runner has no display.

from create import *


@fieldwise_init
struct Smoke(Program):
    @staticmethod
    def create(mut context: Context, mut canvas: Canvas) raises -> Smoke:
        return Smoke()

    def update(mut self, mut context: Context, mut canvas: Canvas) raises:
        canvas.text("Smoke Test", (0, 0))


def _windowed_entry_point() raises:
    run[Smoke]("Smoke Test", width=320, height=240)
    run[Smoke]("Smoke Test", width=320, height=240, backend=RenderBackend.GPU)


def main() raises:
    var image = run_headless[Smoke](320, 240)
    var clear = Color(200, 200, 200)
    if image.pixel(5, 5) != clear:
        raise Error("the frame was not cleared")
    var inked = 0
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixel(x, y) != clear:
                inked += 1
    if inked == 0:
        raise Error("the text drew nothing")
    print("create: drew", inked, "pixels of text")
