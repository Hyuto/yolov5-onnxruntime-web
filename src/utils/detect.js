import { Tensor } from "onnxruntime-web";
import { NMS } from "./nms";
import { renderBoxes } from "./renderBox";
import labels from "./labels.json";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv5 onnxruntime session
 * @param {Number} confidenceThreshold confidence threshold
 * @param {Number} classThreshold class threshold
 * @param {Number} nmsThreshold NMS / IoU threshold
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  canvas,
  session,
  confidenceThreshold,
  classThreshold,
  nmsThreshold,
  inputShape
) => {
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
  const { output0 } = await session.run({ images: tensor }); // run session and get output layer

  const boxes = [];

  // looping through output
  for (let r = 0; r < output0.data.length; r += output0.dims[2]) {
    const data = output0.data.slice(r, r + output0.dims[2]); // get rows
    const scores = data.slice(5); // classes probability scores
    const confidence = data[4]; // detection confidence
    const classId = scores.indexOf(Math.max(...scores)); // class id of maximum probability scores
    const maxClassProb = scores[classId]; // maximum probability scores

    // filtering by thresholds
    if (confidence >= confidenceThreshold && maxClassProb >= classThreshold) {
      const [x, y, w, h] = data.slice(0, 4);
      boxes.push({
        classId: classId,
        probability: maxClassProb,
        confidence: confidence,
        bounding: [x - 0.5 * w, y - 0.5 * h, w, h],
      });
    }
  }

  // filtering boxes using Non Maximum Suppression algorithm
  const selectedBoxes = NMS(boxes, nmsThreshold);
  renderBoxes(canvas, selectedBoxes, labels); // Draw boxes

  // release
  mat.delete();
  matC3.delete();
  input.delete();
};
