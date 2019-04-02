#!/usr/bin/env python3

import argparse
import io
import sys
import time

import numpy as np
import picamera
from PIL import Image, ImageDraw
from edgetpu.detection.engine import DetectionEngine


COLORS = [
    (255, 0, 0, 80),
    (0, 255, 0, 80),
    (0, 0, 255, 80),
    (127, 127, 0, 80),
    (127, 0, 127, 80),
    (0, 127, 127, 80),
    (127, 127, 127, 80),
    (127, 63, 63, 80),
    (63, 127, 63, 80),
    (63, 63, 127, 80)
]


def load_labels(path):
    labels = {}
    if path is not None:
        with open(path, 'r') as i_:
            for line in i_:
                pair = line.strip().split(maxsplit=1)
                labels[int(pair[0])] = pair[1].strip()
    return labels


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        help='Path of the detection model.',
        required=True
    )
    parser.add_argument(
        '--label',
        help='Path of the labels file.',
        default=None
    )
    parser.add_argument(
        '--time',
        help='Demonstration time length in sec.',
        type=float,
        default=10
    )
    return parser


def main(argv):
    argparser = build_argparser()
    args = argparser.parse_args(argv)

    labels = load_labels(args.label)
    engine = DetectionEngine(args.model)

    camera = picamera.PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    _, width, height, channels = engine.get_input_tensor_shape()

    overlay_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    overlay = camera.add_overlay(overlay_img.tobytes(), size=overlay_img.size)
    overlay.layer = 3

    try:
        start_time = time.time()
        camera.start_preview(fullscreen=True)
        buff = io.BytesIO()
        for _ in camera.capture_continuous(
            buff,
            format='rgb',
            use_video_port=True,
            resize=(width, height)
        ):
            buff.truncate()
            buff.seek(0)

            array = np.frombuffer(
                buff.getvalue(),
                dtype=np.uint8
            )

            # Do inference
            start_ms = time.time()
            detected = engine.DetectWithInputTensor(
                array,
                top_k=10
            )
            elapsed_ms = time.time() - start_ms

            if detected:
                camera.annotate_text = (
                    '%d objects detected.\n%.2fms'
                    % (len(detected), elapsed_ms*1000.0)
                )
                overlay_img = Image.new(
                    'RGBA',
                    (width, height),
                    (0, 0, 0, 0)
                )
                draw = ImageDraw.Draw(overlay_img)
                for obj in detected:
                    # relative coord to abs coord.
                    box = obj.bounding_box * [[width, height]]
                    draw.rectangle(
                        box.flatten().tolist(),
                        COLORS[obj.label_id % len(COLORS)]
                    )
                overlay.update(overlay_img.tobytes())
            if time.time() - start_time >= args.time:
                break
            time.sleep(0.1)
    finally:
        camera.stop_preview()
        camera.close()

    return 0


if __name__ == '__main__':
    exit(main(sys.argv[1:]))
