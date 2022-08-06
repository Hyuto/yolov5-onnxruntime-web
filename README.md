# YOLOv5 on onnxruntime-web

<p align="center">
  <img src="./sample.png" />
</p>

![love](https://img.shields.io/badge/Made%20with-🖤-white)
![react](https://img.shields.io/badge/React-blue?logo=react)
![onnxruntime-web](https://img.shields.io/badge/onnxruntime--web-white?logo=onnx&logoColor=black)
![opencv.js-4.5.5](https://img.shields.io/badge/opencv.js-4.5.5-green?logo=opencv)

---

Object Detection application right in your browser.
Serving YOLOv5 in browser using onnxruntime-web with `wasm` backend.

## Setup

```bash
git clone https://github.com/Hyuto/yolov5-onnxruntime-web.git
cd yolov5-onnxruntime-web
yarn install # Install dependencies
```

## Scripts

```bash
yarn start # Start dev server
yarn build # Build for productions
```

## Model

YOLOv5n model converted to onnx model.

```
used model : yolov5n
size       : 7.5 Mb
```

### Use another model

> :warning: **Size Overload** : used YOLOv5 model in this repo is the smallest with size of 7.5 MB, so other models is definitely bigger than this which can cause memory problems on browser.

Use another YOLOv5 model.

1. Clone [yolov5](https://github.com/ultralytics/yolov5) repository

   ```bash
   git clone https://github.com/ultralytics/yolov5.git && cd yolov5
   ```

   Install `requirements.txt` first

   ```bash
   pip install -r requirements.txt
   ```

2. Export model to onnx format
   ```bash
   export.py --weights yolov5*.pt --include onnx
   ```
3. Copy `yolov5*.onnx` to `./public/model`
4. Update `modelName` in `App.jsx` to new model name
   ```jsx
   ...
   // configs
   const modelName = "yolov5*"; // change to new model name
   ...
   ```
5. Done! 😊

## Reference

https://github.com/ultralytics/yolov5
