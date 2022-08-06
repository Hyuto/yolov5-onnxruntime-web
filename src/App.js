import React, { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";
import Loader from "./components/loader";
import LocalImageButton from "./components/local-image";
import { NMS } from "./utils/nms";
import { renderBoxes } from "./utils/renderBox";
import labels from "./utils/labels.json";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // configs
  const modelName = "yolov5n";
  const confidenceThreshold = 0.25;
  const classThreshold = 0.6;
  const nmsThreshold = 0.5;

  /**
   * Callback function to detect image when loaded
   */
  const detectImage = async () => {
    const mat = cv.imread(imageRef.current); // read from img tag
    const matC3 = new cv.Mat(640, 640, cv.CV_8UC3); // new image matrix (640 x 640)
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR); // RGBA to BGR
    const input = cv.blobFromImage(
      matC3,
      1 / 255.0,
      new cv.Size(640, 640),
      new cv.Scalar(0, 0, 0),
      true,
      false
    ); // preprocessing image matrix
    // release
    mat.delete();
    matC3.delete();

    const tensor = new ort.Tensor("float32", input.data32F, [1, 3, 640, 640]); // to ort.Tensor
    const { output } = await session.run({ images: tensor }); // run session and get output layer

    const boxes = [];

    // looping through output
    for (let r = 0; r < output.data.length; r += output.dims[2]) {
      const data = output.data.slice(r, r + output.dims[2]); // get rows
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
    renderBoxes(canvasRef, selectedBoxes, labels); // Draw boxes
  };

  useEffect(() => {
    cv["onRuntimeInitialized"] = () => {
      ort.InferenceSession.create(`${process.env.PUBLIC_URL}/model/${modelName}.onnx`).then(
        (yolov5) => {
          setSession(yolov5);
          setLoading(false);
        }
      );
    };
  }, []);

  return (
    <div className="App">
      <h2>
        Object Detection Using YOLOv5 & <code>onnxruntime-web</code>
      </h2>
      {loading ? (
        <Loader>Getting things ready...</Loader>
      ) : (
        <p>
          <code>onnxruntime-web</code> serving {modelName}
        </p>
      )}

      <div className="content">
        <img ref={imageRef} src="#" alt="" />
        <canvas id="canvas" width={640} height={640} ref={canvasRef} />
      </div>

      <LocalImageButton imageRef={imageRef} callback={detectImage} />
    </div>
  );
};

export default App;
