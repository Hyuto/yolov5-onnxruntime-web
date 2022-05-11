import React, { useState, useEffect, useRef } from "react";
import * as ort from "onnxruntime-web";
import Loader from "./components/loader";
/* import { renderBoxes } from "./utils/renderBox"; */
import "./style/App.css";
import LocalImageButton from "./components/local-image";

const App = () => {
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(true);
  const imageRef = useRef(null);
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

  const detectImage = async () => {
    let mat = cv.imread(imageRef.current);
    let matC3 = new cv.Mat(640, 640, cv.CV_8UC3);
    cv.cvtColor(mat, matC3, cv.COLOR_RGBA2BGR);
    let input = cv.blobFromImage(
      matC3,
      1,
      new cv.Size(640, 640),
      new cv.Scalar(127.5, 127.5, 127.5),
      true
    );
    mat.delete();
    matC3.delete();

    const tensor = new ort.Tensor("float32", input.data32F, [1, 3, 640, 640]);
    const results = await session.run({ images: tensor });
    console.log(results);
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
