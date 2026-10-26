from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import Response, JSONResponse
from src.services.model_service import model_service
from src.services.image_processing_service import image_processing_service
import time
from typing import Optional
import base64

router = APIRouter()

@router.get("/")
def read_root():
    return {
        "status": "online",
        "models": {
            "vae": model_service.vae_encoder is not None,
            "gan": model_service.gan_generator is not None
        },
        "services": {
            "image_processing": True,
            "augmentation": True,
            "degradation": True
        }
    }

@router.post("/predict/vae")
async def predict_vae(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        result_image = await model_service.predict_vae(contents)
        process_time = time.time() - start_time
        return Response(content=result_image, media_type="image/png", headers={"X-Process-Time": str(process_time)})
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/gan")
async def predict_gan(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        result_image = await model_service.predict_gan(contents)
        process_time = time.time() - start_time
        return Response(content=result_image, media_type="image/png", headers={"X-Process-Time": str(process_time)})
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/augment")
async def augment_image(
    file: UploadFile = File(...),
    stretch: bool = Form(True),
    rotation: bool = Form(True),
    zoom: bool = Form(True),
    color: bool = Form(True)
):
    """Apply geometric and colorimetric augmentations to an image"""
    start_time = time.time()
    try:
        contents = await file.read()
        result_image = image_processing_service.augment_image(
            contents,
            stretch=stretch,
            rotation=rotation,
            zoom=zoom,
            color=color
        )
        process_time = time.time() - start_time
        return Response(
            content=result_image,
            media_type="image/png",
            headers={"X-Process-Time": str(process_time)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Augmentation failed: {str(e)}")


@router.post("/degrade")
async def degrade_image(
    file: UploadFile = File(...),
    gaussian_noise: bool = Form(True),
    salt_pepper_noise: bool = Form(True),
    blur_gaussian: bool = Form(True),
    blur_motion: bool = Form(True),
    occlusion: bool = Form(True),
    random_masks: bool = Form(True),
    scratches: bool = Form(True),
    rain_drops: bool = Form(True)
):
    """Apply realistic degradations to an image"""
    start_time = time.time()
    try:
        contents = await file.read()
        degradation_config = {
            'gaussian_noise': gaussian_noise,
            'salt_pepper_noise': salt_pepper_noise,
            'blur_gaussian': blur_gaussian,
            'blur_motion': blur_motion,
            'occlusion': occlusion,
            'random_masks': random_masks,
            'scratches': scratches,
            'rain_drops': rain_drops,
        }
        result_image = image_processing_service.degrade_image(contents, degradation_config)
        process_time = time.time() - start_time
        return Response(
            content=result_image,
            media_type="image/png",
            headers={"X-Process-Time": str(process_time)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Degradation failed: {str(e)}")


@router.post("/enhance")
async def enhance_image(
    file: UploadFile = File(...),
    model: str = Form("gan"),
    preprocess: bool = Form(True),
    apply_augment: bool = Form(False),
    apply_degrade: bool = Form(True)
):
    """
    Complete enhancement pipeline: 
    1. Optional augmentation
    2. Optional degradation
    3. Model inference (VAE or GAN)
    """
    start_time = time.time()
    try:
        contents = await file.read()
        
        # Preprocessing
        if preprocess:
            if apply_augment:
                contents = image_processing_service.augment_image(contents)
            if apply_degrade:
                contents = image_processing_service.degrade_image(contents)
        
        # Model inference
        if model.lower() == "vae":
            result_image = await model_service.predict_vae(contents)
        elif model.lower() == "gan":
            result_image = await model_service.predict_gan(contents)
        else:
            raise ValueError(f"Unknown model: {model}. Use 'vae' or 'gan'")
        
        process_time = time.time() - start_time
        return Response(
            content=result_image,
            media_type="image/png",
            headers={
                "X-Process-Time": str(process_time),
                "X-Model-Used": model,
                "X-Preprocessing": str(preprocess)
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@router.post("/preview")
async def preview_variants(
    file: UploadFile = File(...),
    num_variants: int = Form(4)
):
    """
    Generate multiple variants of an image for preview:
    - Original
    - Augmented
    - Degraded
    - Augmented + Degraded
    """
    start_time = time.time()
    try:
        contents = await file.read()
        variants_bytes = image_processing_service.create_preview_variants(
            contents,
            num_variants=min(num_variants, 6)  # Limit to 6 variants max
        )
        
        # Convert to base64 for JSON response
        variants_b64 = [base64.b64encode(img).decode('utf-8') for img in variants_bytes]
        
        process_time = time.time() - start_time
        return JSONResponse(
            content={
                "variants": variants_b64,
                "count": len(variants_b64),
                "labels": ["Original", "Augmented", "Degraded", "Aug+Deg"][:len(variants_b64)]
            },
            headers={"X-Process-Time": str(process_time)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")

