import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";
import { NMS } from "./nms";
import labels from "./labels.json";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv5 onnxruntime session
 * @param {Number} classThreshold class threshold
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 * @param {Boolean} withNMS model including NMS operator
 */
export const detectImage = async (image, canvas, session, classThreshold, inputShape, withNMS) => {
  const [modelWidth, modelHeight] = inputShape.slice(2);

  const mat = cv.imread(image); // read from img tag
  const matC3 = new cv.Mat(modelWidth, modelHeight, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR
  const input = cv.blobFromImage(
    matC3,
    1 / 255.0,
    new cv.Size(modelWidth, modelHeight),
    new cv.Scalar(0, 0, 0),
    true,
    false
  ); // preprocessing image matrix

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const result = await session.run({ images: tensor }); // run session and get output layer
  const output = withNMS ? result["output_nms"] : result["output0"];
  let boxes = [];

  // looping through output
  for (let r = 0; r < output.data.length; r += output.dims[2]) {
    let x,
      y,
      w,
      h,
      label,
      score,
      confidence = 0;

    if (withNMS) [x, y, w, h, label, score] = output.data.slice(r, r + output.dims[2]);
    else {
      const data = output.data.slice(r, r + output.dims[2]); // get rows
      [x, y, w, h] = data.slice(0, 4);
      confidence = data[4]; // detection confidence
      const scores = data.slice(5).map((e) => e * confidence); // classes probability scores
      score = Math.max(...scores); // maximum probability scores
      label = scores.indexOf(score); // class id of maximum probability scores
    }

    // filtering by thresholds
    if (score >= classThreshold) {
      boxes.push({
        label: labels[label],
        probability: score,
        confidence: confidence,
        bounding: [x - 0.5 * w, y - 0.5 * h, w, h],
      });
    }
  }

  if (!withNMS) boxes = NMS(boxes, 0.4); // perform NMS if not included in model
  renderBoxes(canvas, boxes); // Draw boxes

  // release
  mat.delete();
  matC3.delete();
  input.delete();
};
