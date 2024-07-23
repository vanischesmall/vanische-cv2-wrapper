import cv2 as cv
import numpy as np
from vanischeCV import *



BLACK_CR = ColorRange (
    (0, 215, 0),
    (85, 255, 50),
)

FIGURES_CR = ColorRange (
    (30, 50, 0),
    (180, 180, 190),
)

undertable_rect: Rect
table_roi_rect: Rect
figures_rects: list[tuple[Rect, Color]]
figures: Frame


def detect_table(hsv_frame: Frame) -> Frame:
    global undertable_rect, table_rect, table_roi_rect

    mask = hsv_frame.in_range(BLACK_CR).get_conts()

    for c in mask.conts:
        cont = Contour(c).get_area()
        if 500 < cont.area < 6000:
            cont.get_bounding_rect()

            if cont.w / cont.h > 4:
                undertable_rect = cont.rect

                table_roi_rect = Rect (
                    cont.x,
                    cont.y + cont.h - cont.w,
                    cont.w,
                    cont.w,
                ).with_offset(int(cont.w * 0.2))

    return hsv_frame.roi(table_roi_rect.to_roi())

def detect_figures(hsv_table_roi: Frame) -> Frame:
    global figures_rects, figures

    figures_mask = hsv_table_roi.in_range(FIGURES_CR).get_conts()
    bgr_figures_mask = figures_mask.cvt2bgr()

    figures_rects = [] 
    figures_list  = []

    for c in figures_mask.conts:
        cont = Contour(c).get_area()
        if 500 < cont.area < 4000:
            cont.get_bounding_rect()

            if abs(cont.w - cont.h) < cont.w * 0.75:
                figures_list.append(
                    bgr_figures_mask
                    .roi(cont.rect.to_roi())
                    .resize(150, 150)
                    .src
                )

                figures_rects.append((
                    cont.rect.with_roi_offset(table_roi_rect.to_roi()),
                    Colors.WHITE
                ))

                bgr_figures_mask.draw_cont_rect(cont, thickness = 2, color=Colors.RED)

    figures = Frame(cv.vconcat(figures_list), 'gray')

    return bgr_figures_mask


if __name__ == "__main__":
    cap = cv.VideoCapture('basicvideo1.mp4')


    while cv.waitKey(40) != ord('q'):
        ret, raw = cap.read()
        if not ret:
            continue

        frame = Frame(raw, 'bgr').blur(5)
        raw   = Frame(raw, 'bgr')
        try:
            hsv = frame.cvt2hsv()

            hsv_table_roi = detect_table(hsv)
            figures_mask = detect_figures(hsv_table_roi)

            figures.show('figures')
            

            for figure in figures_rects:
                frame.draw_rect(figure[0], figure[1])


            frame.draw_rect(undertable_rect, Colors.GREEN)
            frame.draw_rect(table_roi_rect, thickness=2)
        except Exception as err:
            print(err)

        frame.show('frame')
    
    cap.release()
    cv.destroyAllWindows()

