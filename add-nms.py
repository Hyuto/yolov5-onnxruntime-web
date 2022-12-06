"""Add NMS on YOLOv5 onnx models.

Usage:
    $ python add-nms.py --model <YOLOv5-MODEL>.onnx

Install Dependencies:
    $ pip install numpy onnx onnxruntime onnxsim      
    $ pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
"""
import argparse

from pathlib import Path

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import onnxruntime as ort

from onnxsim import simplify


@gs.Graph.register()
def slice(self, data, start, end, axis=2):
    return self.layer(op="Slice", inputs=[data, start, end, axis], outputs=["slice_out_gs"])[0]


@gs.Graph.register()
def cast(self, data, to):
    return self.layer(
        op="Cast",
        inputs=[data],
        outputs=["cast_out_gs"],
        attrs={"to": to},
    )[0]


@gs.Graph.register()
def mul(self, A, B):
    return self.layer(op="Mul", inputs=[A, B], outputs=["mul_out_gs"])[0]


@gs.Graph.register()
def squeeze(self, data, axis):
    return self.layer(
        op="Squeeze",
        inputs=[data],
        attrs={"axes": axis},
        outputs=["squeeze_out_gs"],
    )[0]


@gs.Graph.register()
def gather(self, data, indices, axis=2):
    return self.layer(
        op="Gather",
        inputs=[data, indices],
        outputs=["gather_out_gs"],
        attrs={"axis": axis},
    )[0]


@gs.Graph.register()
def transpose(self, data, axis):
    return self.layer(
        op="Transpose",
        inputs=[data],
        outputs=["transpose_out_gs"],
        attrs={"perm": axis},
    )[0]


@gs.Graph.register()
def concat(self, inputs, axis=0):
    return self.layer(
        op="Concat",
        inputs=inputs,
        outputs=["concat_out_gs"],
        attrs={"axis": axis},
    )[0]


@gs.Graph.register()
def argmax(self, data, axis=0, keepdims=1, select_last_index=0):
    return self.layer(
        op="ArgMax",
        inputs=[data],
        outputs=["argmax_out_gs"],
        attrs={"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index},
    )[0]


@gs.Graph.register()
def reduce_max(self, data, axis=[0], keepdims=1):
    return self.layer(
        op="ReduceMax",
        inputs=[data],
        outputs=["reduce_max_out_gs"],
        attrs={"axes": axis, "keepdims": keepdims},
    )[0]


@gs.Graph.register()
def non_max_suppression(
    self,
    boxes,
    scores,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    center_point_box=1,  # for yolo pytorch model
):
    # Docs : https://github.com/onnx/onnx/blob/main/docs/Operators.md#NonMaxSuppression
    return self.layer(
        op="NonMaxSuppression",
        inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold],
        outputs=["nms_out_gs"],
        attrs={"center_point_box": center_point_box},
    )[0]


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="YOLOv5 onnx model path")
    parser.add_argument(
        "--output-dir", type=str, default=".", help="Exported YOLOv5 onnx model with NMS directory"
    )
    parser.add_argument("--simplify", action="store_false", help="Simplify onnx model")
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
        help="Integer representing the maximum number of boxes to be selected per class",
    )
    parser.add_argument(
        "--iou-tresh",
        type=float,
        default=0.40,
        help="Float representing the threshold for deciding whether boxes overlap too much with respect to IOU",
    )
    parser.add_argument(
        "--conf-tresh",
        type=float,
        default=0.25,
        help="Float representing the threshold for deciding when to remove boxes based on confidence score",
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    model = Path(opt.model)
    assert model.exists(), "Model not exist!"

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{model.stem}-nms{model.suffix}"

    graph = gs.import_onnx(onnx.load(model))
    assert graph.opset == 12, "Only support opset 12"  # check opset
    assert len(graph.outputs) == 1, "Not single output model"  # check otput
    assert (
        len(graph.outputs[0].shape) == 3
    ), "Output doesn't follow [batch_size, num_detection, columns] format"  # check otput
    batch, num_detection, col = graph.outputs[0].shape
    assert batch == 1, "Currently only support batch_size == 1"  # check batch size

    boxes = graph.slice(
        graph.outputs[0],
        start=np.asarray([0], dtype=np.int32),
        end=np.asarray([4], dtype=np.int32),
        axis=np.asarray([2], dtype=np.int32),
    )  # slice boxes from outputs
    boxes.name = "raw-boxes"
    boxes.dtype = np.float32

    confidences = graph.slice(
        graph.outputs[0],
        start=np.asarray([4], dtype=np.int32),
        end=np.asarray([5], dtype=np.int32),
        axis=np.asarray([2], dtype=np.int32),
    )  # slice confidences from outputs
    confidences.name = "raw-confidences"
    confidences.dtype = np.float32

    scores = graph.mul(
        graph.slice(
            graph.outputs[0],
            start=np.asarray([5], dtype=np.int32),
            end=np.asarray([col - 5 + 6], dtype=np.int32),
            axis=np.asarray([2], dtype=np.int32),
        ),  # slice scores from outputs
        confidences,
    )  # multiplied scores by confidences
    scores.name = "raw-scores"
    scores.dtype = np.float32

    nms = graph.non_max_suppression(
        boxes,
        graph.transpose(
            confidences, axis=np.asarray((0, 2, 1), dtype=np.int32)
        ),  # transpose confidences [1, num_det, 1] to [1, 1, num_det]
        max_output_boxes_per_class=np.asarray([opt.topk], dtype=np.int64),
        iou_threshold=np.asarray([opt.iou_tresh], dtype=np.float32),
        score_threshold=np.asarray([opt.conf_tresh], dtype=np.float32),
    )  # perform NMS using boxes and confidences as input
    nms.name = "NMS"
    nms.dtype = np.int64

    idx = graph.transpose(
        graph.gather(
            nms, indices=np.asarray([2], dtype=np.int32), axis=1
        ),  # gether selected boxes index from NMS
        axis=np.asarray((1, 0), dtype=np.int32),
    )  # transpose index from [n, 1] to [1, n]
    idx.name = "selected-idx"
    idx.dtype = np.int64

    selected_boxes = graph.squeeze(
        graph.gather(boxes, indices=idx, axis=1),  # indexing boxes
        axis=[1],
    )  # squeeze tensor dimension [1, 1, n, 4] to [1, n, 4]
    selected_boxes.name = "boxes"
    selected_boxes.dtype = np.float32

    selected_scores_ = graph.squeeze(
        graph.gather(scores, indices=idx, axis=1),  # indexing scores
        axis=[1],
    )  # squeeze tensor dimension [1, 1, n, 1] to [1, n, 1]
    selected_scores_.dtype = np.float32

    labels = graph.cast(
        graph.argmax(selected_scores_, axis=2, keepdims=1),  # get labels id through argmax
        to=1,
    )  # casting tensor from int64 to float32 (onnxruntime-web friendly)
    labels.name = "labels"
    labels.dtype = np.float32

    selected_scores = graph.reduce_max(selected_scores_, axis=[2], keepdims=1)  # get max score
    selected_scores.name = "scores"
    selected_scores.dtype = np.float32

    output = graph.concat([selected_boxes, labels, selected_scores], axis=2)  # cancating output
    output.name = "output"
    output.dtype = np.float32
    output.shape = [1, None, 6]

    graph.outputs = [output]

    graph.cleanup().toposort()

    model = gs.export_onnx(graph)  # GS export to onnx

    if opt.simplify:
        model, check = simplify(gs.export_onnx(graph))  # simplify onnx model
        assert check, "Simplified ONNX model could not be validated"

    onnx.save(model, output_path)  # saving onnx model


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
