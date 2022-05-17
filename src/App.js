import React, { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";
import Loader from "./components/loader";
import LocalImageButton from "./components/local-image";
import { IOU, NMS } from "./utils/nms";
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
  const confidenceThreshold = 0.3;
  const classThreshold = 0.5;
  const nmsThreshold = 0.6;

  const detectImage = async () => {
    const mat = cv.imread(imageRef.current);
    const matC3 = new cv.Mat(640, 640, cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);
    const input = cv.blobFromImage(
      matC3,
      1 / 255.0,
      new cv.Size(640, 640),
      new cv.Scalar(0, 0, 0),
      true,
      false
    );
    mat.delete();
    matC3.delete();

    const tensor = new ort.Tensor("float32", input.data32F, [1, 3, 640, 640]);
    const { output } = await session.run({ images: tensor });

    const boxes = [];

    for (let r = 0; r < output.data.length; r += output.dims[2]) {
      const data = output.data.slice(r, r + output.dims[2]);
      const scores = data.slice(5);
      const confidence = data[4];
      const classId = scores.indexOf(Math.max(...scores));
      const maxClassProb = scores[classId];

      if (confidence >= confidenceThreshold && maxClassProb >= classThreshold) {
        const [x, y, w, h] = data.slice(0, 4);
        boxes.push([x - 0.5 * w, y - 0.5 * h, w, h]);
      }
    }

    // Draw boxes
    // renderBoxes(canvasRef, boxes, probabilities, classIds);
    const test = NMS(boxes, nmsThreshold);
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 2;

    test.forEach(({ x1, y1, width, height }) => {
      ctx.strokeRect(x1, y1, width, height);
    });
  };

  useEffect(() => {
    cv["onRuntimeInitialized"] = () => {
      ort.InferenceSession.create(`${window.location.origin}/model/${modelName}.onnx`).then(
        (yolov5) => {
          setSession(yolov5);
          setLoading(false);
        }
      );
    };
  }, []);

  return (
    <div className="App">
      <h2>Object Detection Using YOLOv5 & Tensorflow.js</h2>
      {loading ? <Loader>Getting things ready...</Loader> : null}

      <div className="content">
        <img ref={imageRef} src="#" alt="" />
        <canvas id="canvas" width={640} height={640} ref={canvasRef} />
      </div>

      <LocalImageButton imageRef={imageRef} callback={detectImage} />
    </div>
  );
};

export default App;
