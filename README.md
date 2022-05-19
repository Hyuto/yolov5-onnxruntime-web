# YOLOv5 on onnxruntime-web

Object Detection application right in your browser.
Serving YOLOv5 in browser using onnxruntime-web with `wasm` backend.

**Setup**

```bash
git clone https://github.com/Hyuto/yolov5-onnxruntime-web.git
cd yolov5-onnxruntime-web
yarn install # Install dependencies
```

**Scripts**

```bash
yarn start # Start dev server
yarn build # Build for productions
```

## Used Technologies

1. `onnxruntime-web`
2. `opencv.js`
3. `react` (frontend)

## Model

YOLOv5n model converted to onnx model.

```
used model : yolov5n
size       : 7.5 Mb
```

**Use another model**

Use another YOLOv5 model.

1. Clone [yolov5](https://github.com/ultralytics/yolov5) repository

   ```bash
   git clone https://github.com/ultralytics/yolov5.git && cd yolov5
   ```

   Install `requirements.txt` first

   ```bash
   pip install -r requirements.txt
   ```

2. Export model to tensorflow.js format
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
