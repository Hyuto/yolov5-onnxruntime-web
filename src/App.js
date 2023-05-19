import React, { useState, useRef } from "react";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import "./style/App.css";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  // configs
  const modelName = "yolov5n.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 100;
  const iouThreshold = 0.45;
  const confThreshold = 0.2;
  const classThreshold = 0.2;

  // wait until opencv.js initialized
  cv["onRuntimeInitialized"] = async () => {
    const baseModelURL = `${process.env.PUBLIC_URL}/model`;

    // create session
    const arrBufNet = await download(
      `${baseModelURL}/${modelName}`, // url
      ["Loading YOLOv5 model", setLoading] // logger
    ); // get model arraybuffer
    const yolov5 = await InferenceSession.create(arrBufNet);
    const arrBufNMS = await download(
      `${baseModelURL}/nms-yolov5.onnx`, // url
      ["Loading NMS model", setLoading] // logger
    ); // get nms model arraybuffer
    const nms = await InferenceSession.create(arrBufNMS);

    // warmup model
    setLoading({ text: "Warming up model...", progress: null });
    const tensor = new Tensor(
      "float32",
      new Float32Array(modelInputShape.reduce((a, b) => a * b)),
      modelInputShape
    );
    const config = new Tensor("float32", new Float32Array([topk, iouThreshold, confThreshold]));
    const { output0 } = await yolov5.run({ images: tensor });
    await nms.run({ detection: output0, config: config });

    setSession({ net: yolov5, nms: nms });
    setLoading(null);
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>YOLOv5 Object Detection App</h1>
        <p>
          YOLOv5 object detection application live on browser powered by{" "}
          <code>onnxruntime-web</code>
        </p>
        <p>
          Serving : <code className="code">{modelName}</code>
        </p>
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={() => {
            detectImage(
              imageRef.current,
              canvasRef.current,
              session,
              topk,
              iouThreshold,
              confThreshold,
              classThreshold,
              modelInputShape
            );
          }}
        />
        <canvas
          id="canvas"
          width={modelInputShape[2]}
          height={modelInputShape[3]}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(e.target.files[0]); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
