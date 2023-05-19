import { Tensor } from "onnxruntime-web";
import { renderBoxes } from "./renderBox";

/**
 * Detect Image
 * @param {HTMLImageElement} image Image to detect
 * @param {HTMLCanvasElement} canvas canvas to draw boxes
 * @param {ort.InferenceSession} session YOLOv5 onnxruntime session
 * @param {Number} topk Integer representing the maximum number of boxes to be selected per class
 * @param {Number} iouThreshold Float representing the threshold for deciding whether boxes overlap too much with respect to IOU
 * @param {Number} confThreshold Float representing the threshold for deciding when to remove boxes based on confidence score
 * @param {Number} classThreshold class threshold
 * @param {Number[]} inputShape model input shape. Normally in YOLO model [batch, channels, width, height]
 */
export const detectImage = async (
  image,
  canvas,
  session,
  topk,
  iouThreshold,
  confThreshold,
  classThreshold,
  inputShape
) => {
  const [modelWidth, modelHeight] = inputShape.slice(2);

  const mat = cv.imread(image); // read from img tag
  const matC3 = new cv.Mat(mat.rows, mat.cols, cv.CV_8UC3); // new image matrix
  cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR

  // padding image to [n x n] dim
  const maxSize = Math.max(matC3.rows, matC3.cols); // get max size from width and height
  const xPad = maxSize - matC3.cols, // set xPadding
    xRatio = maxSize / matC3.cols; // set xRatio
  const yPad = maxSize - matC3.rows, // set yPadding
    yRatio = maxSize / matC3.rows; // set yRatio
  const matPad = new cv.Mat(); // new mat for padded image
  cv.copyMakeBorder(matC3, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT, [0, 0, 0, 255]); // padding black

  const input = cv.blobFromImage(
    matPad,
    1 / 255.0, // normalize
    new cv.Size(modelWidth, modelHeight), // resize to model input size
    new cv.Scalar(0, 0, 0),
    true, // swapRB
    false // crop
  ); // preprocessing image matrix

  const tensor = new Tensor("float32", input.data32F, inputShape); // to ort.Tensor
  const config = new Tensor("float32", new Float32Array([topk, iouThreshold, confThreshold])); // nms config tensor
  const start = Date.now();
  const { output0 } = await session.net.run({ images: tensor }); // run session and get output layer
  const { selected_idx } = await session.nms.run({ detection: output0, config: config }); // get selected idx from nms
  console.log(Date.now() - start);

  const boxes = [];

  // looping through output
  selected_idx.data.forEach((idx) => {
    const data = output0.data.slice(idx * output0.dims[2], (idx + 1) * output0.dims[2]); // get rows
    const [x, y, w, h] = data.slice(0, 4);
    const confidence = data[4]; // detection confidence
    const scores = data.slice(5); // classes probability scores
    let score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    score *= confidence; // multiply score by conf

    // filtering by score thresholds
    if (score >= classThreshold)
      boxes.push({
        label: label,
        probability: score,
        bounding: [
          Math.floor((x - 0.5 * w) * xRatio), // left
          Math.floor((y - 0.5 * h) * yRatio), //top
          Math.floor(w * xRatio), // width
          Math.floor(h * yRatio), // height
        ],
      });
  });

  renderBoxes(canvas, boxes); // Draw boxes

  // release mat opencv
  mat.delete();
  matC3.delete();
  matPad.delete();
  input.delete();
};
