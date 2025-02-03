from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
from typing import List
import uuid
from model.AlignModule.generator import FaceGenerator
from model.BlendModule.generator import Generator as Decoder
from model.AlignModule.config import Params as AlignParams
from model.BlendModule.config import Params as BlendParams 
from model.third.faceParsing.model import BiSeNet
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch
from process.process_func import Process
from process.process_utils import *
import onnxruntime as ort
from inference import Infer

app = FastAPI(title="HeadSwap API")

# Initialize model globally
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = Infer(
        # These are the exact same paths used in the notebook
        'pretrained_models/epoch_00190_iteration_000400000_checkpoint.pt',  # align_path (PIRender model)
        'pretrained_models/Blender-401-00012900.pth',  # blend_path
        'pretrained_models/parsing.pth',  # parsing_path
        'pretrained_models/epoch_20.pth',  # params_path
        'pretrained_models/BFM'  # bfm_folder
    )

@app.post("/swap-face/")
async def swap_face(
    source_image: UploadFile = File(...),
    target_image: UploadFile = File(...),
):
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("assets", exist_ok=True)
        
        # Save uploaded files with proper extensions
        source_path = f"assets/{uuid.uuid4()}.jpg"
        target_path = f"assets/{uuid.uuid4()}.jpg"
        
        # Save uploaded files
        source_content = await source_image.read()
        with open(source_path, "wb") as f:
            f.write(source_content)
            
        target_content = await target_image.read()
        with open(target_path, "wb") as f:
            f.write(target_content)

        # Run face swap with exact same parameters as notebook
        result = model.run_single(
            src_img_path=source_path,
            tgt_img_path=target_path,
            crop_align=True,  # Important: notebook uses this
            cat=True  # Important: notebook uses this 
        )

        if result is None:
            raise HTTPException(status_code=400, detail="Face detection failed")

        # Save result
        output_path = f"assets/result_{uuid.uuid4()}.png"
        cv2.imwrite(output_path, result)

        # Cleanup temp files
        os.remove(source_path)
        os.remove(target_path)

        # Return the result image
        return FileResponse(
            output_path, 
            media_type="image/png",
            filename="swapped_face.png"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0"}