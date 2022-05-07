import React, { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";
import Loader from "./components/loader";
import { Webcam } from "./utils/webcam";
/* import { renderBoxes } from "./utils/renderBox"; */
import "./style/App.css";

const App = () => {
  const [loading, setLoading] = useState(true);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [canvasSize, setCanvasSize] = useState([0, 0]);

  // configs
  const modelName = "yolov5n";
  // const threshold = 0.25;

  /**
   * Function to detect every frame loaded from webcam in video tag.
   * @param {tf.GraphModel} model loaded YOLOv5 tensorflow.js model
   */
  /* const detectFrame = async (model) => {
    tf.engine().startScope();
    let [modelWidth, modelHeight] = model.inputs[0].shape.slice(1, 3);
    const input = tf.tidy(() => {
      return tf.image
        .resizeBilinear(tf.browser.fromPixels(videoRef.current), [modelWidth, modelHeight])
        .div(255.0)
        .expandDims(0);
    });

    await model.executeAsync(input).then((res) => {
      const [boxes, scores, classes] = res.slice(0, 3);
      const boxes_data = boxes.dataSync();
      const scores_data = scores.dataSync();
      const classes_data = classes.dataSync();
      renderBoxes(canvasRef, threshold, boxes_data, scores_data, classes_data);
      tf.dispose(res);
    });

    requestAnimationFrame(() => detectFrame(model)); // get another frame
    tf.engine().endScope();
  }; */

  useEffect(() => {
    ort.InferenceSession.create(`${window.location.origin}/model/${modelName}.onnx`).then(
      (yolov5) => {
        // TODO : Warmup the model before using real data.
        /* const dummyInput = tf.ones(yolov5.inputs[0].shape);
      await yolov5.executeAsync(dummyInput).then((warmupResult) => {
        tf.dispose(warmupResult);
        tf.dispose(dummyInput);

        setLoading({ loading: false, progress: 1 });
        webcam.open(videoRef, () => detectFrame(yolov5));
      }); */
        const webcam = new Webcam();
        const drawFrames = () => {
          const ctx = canvasRef.current.getContext("2d");
          ctx.drawImage(videoRef.current, 0, 0);
          const frame = ctx.getImageData(
            0,
            0,
            videoRef.current.offsetWidth,
            videoRef.current.offsetHeight
          );

          const rgbFrame = [];
          for (let i = 0; i < frame.data.length / 4; i++) {
            rgbFrame.push(frame.data[i * 4 + 0]);
            rgbFrame.push(frame.data[i * 4 + 1]);
            rgbFrame.push(frame.data[i * 4 + 2]);
          }

          // TODO : do preprocessing
          // migrate to numjs https://github.com/nicolaspanel/numjs
          const tensor = new ort.Tensor("float32", rgbFrame, [frame.width, frame.height, 3]);
          console.log(tensor);

          requestAnimationFrame(() => drawFrames()); // get another frame
        };

        webcam.open(videoRef, () => {
          setLoading(false);
          setCanvasSize([videoRef.current.offsetWidth, videoRef.current.offsetHeight]);
          drawFrames();
        });
      }
    );
  }, []);

  return (
    <div className="App">
      <h2>Object Detection Using YOLOv5 & Tensorflow.js</h2>
      {loading ? <Loader>Getting things ready...</Loader> : null}

      <div className="content">
        <video autoPlay playsInline muted ref={videoRef} />
        <canvas width={canvasSize[0]} height={canvasSize[1]} ref={canvasRef} />
      </div>
    </div>
  );
};

export default App;
