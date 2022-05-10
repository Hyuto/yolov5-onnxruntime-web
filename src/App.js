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

        const drawFrames = async (model) => {
          const ctx = canvasRef.current.getContext("2d");
          ctx.drawImage(videoRef.current, 0, 0, 640, 640);
          const frame = ctx.getImageData(0, 0, 640, 640);

          const rgbFrame = [];
          for (let i = 0; i < frame.data.length / 4; i++) {
            rgbFrame.push(frame.data[i * 4 + 0] / 255.0);
            rgbFrame.push(frame.data[i * 4 + 1] / 255.0);
            rgbFrame.push(frame.data[i * 4 + 2] / 255.0);
          }

          // TODO : Fixing input dim problem
          const tensor = new ort.Tensor("float32", rgbFrame, [1, frame.width, frame.height, 3]);
          const results = await model.run({ images: tensor });
          console.log(results);

          requestAnimationFrame(() => drawFrames()); // get another frame
        };

        webcam.open(videoRef, () => {
          setLoading(false);
          drawFrames(yolov5);
          console.log(yolov5);
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
        <canvas width={640} height={640} ref={canvasRef} />
      </div>
    </div>
  );
};

export default App;
