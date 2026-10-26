import requests
import io
from PIL import Image

class APIClient:
    """API Client for backend communication - minimal interface"""
    
    def __init__(self, base_url: str = "http://backend:8000"):
        self.base_url = base_url

    def check_status(self):
        """Check backend status and available services"""
        try:
            return requests.get(f"{self.base_url}/", timeout=2).json()
        except:
            return None

    def predict(self, image_bytes: bytes, model_type: str):
        """Basic model prediction without preprocessing"""
        endpoint = "/predict/vae" if "VAE" in model_type else "/predict/gan"
        files = {"file": image_bytes}
        
        try:
            response = requests.post(f"{self.base_url}{endpoint}", files=files)
            if response.status_code == 200:
                return {
                    "image": Image.open(io.BytesIO(response.content)),
                    "process_time": float(response.headers.get("X-Process-Time", 0))
                }
            else:
                return {"error": response.json().get('detail')}
        except Exception as e:
            return {"error": str(e)}

    def enhance(self, image_bytes: bytes, model_type: str, preprocess=True, 
                apply_augment=False, apply_degrade=True):
        """Complete enhancement pipeline with optional preprocessing"""
        files = {"file": image_bytes}
        model = "vae" if "VAE" in model_type else "gan"
        data = {
            "model": model,
            "preprocess": str(preprocess).lower(),
            "apply_augment": str(apply_augment).lower(),
            "apply_degrade": str(apply_degrade).lower()
        }
        
        try:
            response = requests.post(f"{self.base_url}/enhance", files=files, data=data)
            if response.status_code == 200:
                return {
                    "image": Image.open(io.BytesIO(response.content)),
                    "process_time": float(response.headers.get("X-Process-Time", 0)),
                    "model_used": response.headers.get("X-Model-Used", model)
                }
            else:
                return {"error": response.json().get('detail')}
        except Exception as e:
            return {"error": str(e)}

    def preview_variants(self, image_bytes: bytes, num_variants: int = 4):
        """Generate multiple preview variants of an image (original, augmented, degraded, etc.)"""
        files = {"file": image_bytes}
        data = {"num_variants": num_variants}
        
        try:
            response = requests.post(f"{self.base_url}/preview", files=files, data=data)
            if response.status_code == 200:
                data = response.json()
                return {
                    "variants": data["variants"],
                    "labels": data["labels"],
                    "count": data["count"],
                    "process_time": float(response.headers.get("X-Process-Time", 0))
                }
            else:
                return {"error": response.json().get('detail')}
        except Exception as e:
            return {"error": str(e)}


api_client = APIClient()


