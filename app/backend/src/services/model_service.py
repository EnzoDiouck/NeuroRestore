import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging
from src.core.config import settings
from src.services.custom_layers import ResNetBlock, SubPixelConv2D, UpsampleBlock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.vae_encoder = None
        self.vae_decoder = None
        self.gan_generator = None
        self.load_models()

    def load_models(self):
        logger.info("Loading models...")
        
        # Load VAE
        try:
            vae_encoder_exists = os.path.exists(settings.VAE_ENCODER_PATH)
            vae_decoder_exists = os.path.exists(settings.VAE_DECODER_PATH)
            vae_weights_exists = os.path.exists(settings.VAE_WEIGHTS_PATH)
            
            if vae_encoder_exists and vae_decoder_exists:
                logger.info(f"Loading VAE encoder from {settings.VAE_ENCODER_PATH}...")
                
                # Custom objects dictionary with all custom layers
                custom_objects = {
                    'ResNetBlock': ResNetBlock,
                    'SubPixelConv2D': SubPixelConv2D,
                    'UpsampleBlock': UpsampleBlock
                }
                
                self.vae_encoder = tf.keras.models.load_model(
                    settings.VAE_ENCODER_PATH,
                    custom_objects=custom_objects,
                    compile=False
                )
                logger.info(f"Loading VAE decoder from {settings.VAE_DECODER_PATH}...")
                self.vae_decoder = tf.keras.models.load_model(
                    settings.VAE_DECODER_PATH,
                    custom_objects=custom_objects,
                    compile=False
                )
                logger.info(f"✓ VAE models loaded successfully from {settings.MODELS_DIR}")
                
            else:
                missing_files = []
                if not vae_encoder_exists:
                    missing_files.append(settings.VAE_ENCODER_PATH)
                if not vae_decoder_exists:
                    missing_files.append(settings.VAE_DECODER_PATH)
                logger.warning(f"⚠ VAE models not found. Missing files: {', '.join(missing_files)}")
        except Exception as e:
            logger.error(f"❌ Error loading VAE models: {e}")
            self.vae_encoder = None
            self.vae_decoder = None

        # Load GAN
        try:
            gan_path = settings.GAN_GENERATOR_PATH
            if os.path.exists(gan_path):
                logger.info(f"Loading GAN generator from {gan_path}...")
                self.gan_generator = tf.keras.models.load_model(gan_path, compile=False)
                logger.info(f"✓ GAN generator loaded successfully from {gan_path}")
            else:
                logger.warning(f"⚠ GAN generator not found at {gan_path}")
        except Exception as e:
            logger.error(f"❌ Error loading GAN generator: {e}")
            self.gan_generator = None

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Convert bytes to tensor [-1, 1]"""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image = image.resize((settings.IMG_WIDTH, settings.IMG_HEIGHT))
            img_array = np.array(image)
            img_array = (img_array.astype(np.float32) / 127.5) - 1.0
            return np.expand_dims(img_array, axis=0)
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError("Invalid image data")

    def postprocess_image(self, tensor: np.ndarray) -> bytes:
        """Convert tensor [-1, 1] to bytes"""
        try:
            img_array = tensor[0]
            img_array = (img_array + 1.0) * 127.5
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            image = Image.fromarray(img_array)
            
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Error postprocessing image: {e}")
            raise ValueError("Error generating output image")

    def warmup(self):
        """Run a dummy inference to initialize TensorFlow graphs"""
        logger.info("🔥 Warming up models...")
        dummy_input = np.zeros((1, settings.IMG_HEIGHT, settings.IMG_WIDTH, 3), dtype=np.float32)
        
        try:
            if self.vae_encoder and self.vae_decoder:
                # VAE Warmup
                z_mean, z_log_var, s1, s2, s3, s4 = self.vae_encoder.predict(dummy_input, verbose=0)
                z = z_mean # Simplified for warmup
                self.vae_decoder.predict([z, s1, s2, s3, s4], verbose=0)
                logger.info("  ✓ VAE warmed up")
                
            if self.gan_generator:
                # GAN Warmup
                self.gan_generator(dummy_input, training=False)
                logger.info("  ✓ GAN warmed up")
                
        except Exception as e:
            logger.warning(f"⚠️ Warmup failed (non-critical): {e}")

    def _predict_vae_sync(self, image_bytes: bytes) -> bytes:
        """Synchronous implementation of VAE inference"""
        if not self.vae_encoder or not self.vae_decoder:
            raise ValueError("VAE models are not loaded")
            
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image_bytes)
            
            # Encoder prediction
            encoder_outputs = self.vae_encoder.predict(input_tensor, verbose=0)
            
            # Unpack outputs
            z_mean = encoder_outputs[0]
            z_log_var = encoder_outputs[1]
            skip1 = encoder_outputs[2]
            skip2 = encoder_outputs[3]
            skip3 = encoder_outputs[4]
            skip4 = encoder_outputs[5]
            
            # Reparameterization trick (sampling)
            epsilon = tf.random.normal(shape=tf.shape(z_mean))
            z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            # Decoder prediction
            decoder_inputs = [z, skip1, skip2, skip3, skip4]
            reconstructed = self.vae_decoder.predict(decoder_inputs, verbose=0)
            
            # Postprocess
            return self.postprocess_image(reconstructed)
            
        except Exception as e:
            logger.error(f"VAE Inference failed: {str(e)}")
            raise e

    async def predict_vae(self, image_bytes: bytes) -> bytes:
        """Async wrapper for VAE inference"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_vae_sync, image_bytes)

    def _predict_gan_sync(self, image_bytes: bytes) -> bytes:
        """Synchronous implementation of GAN inference"""
        if not self.gan_generator:
            raise ValueError("GAN model not loaded")
            
        try:
            input_tensor = self.preprocess_image(image_bytes)
            
            # GAN Inference
            reconstruction = self.gan_generator(input_tensor, training=False)
            
            return self.postprocess_image(reconstruction.numpy())
        except Exception as e:
            logger.error(f"GAN Inference failed: {e}")
            raise e

    async def predict_gan(self, image_bytes: bytes) -> bytes:
        """Async wrapper for GAN inference"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._predict_gan_sync, image_bytes)

# Global instance
model_service = ModelService()
model_service.warmup()
