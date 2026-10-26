import os

class Settings:
    PROJECT_NAME: str = "Image Reconstructor API"
    VERSION: str = "1.0.0"
    
    # Model Configuration
    IMG_HEIGHT: int = 256
    IMG_WIDTH: int = 256
    MODELS_DIR: str = os.getenv("MODELS_DIR", "/models")
    
    # Paths
    VAE_ENCODER_PATH: str = os.path.join(MODELS_DIR, "vae_encoder.keras")
    VAE_DECODER_PATH: str = os.path.join(MODELS_DIR, "vae_decoder.keras")
    VAE_WEIGHTS_PATH: str = os.path.join(MODELS_DIR, "vae_weights.weights.h5")
    GAN_GENERATOR_PATH: str = os.path.join(MODELS_DIR, "generator.keras")

settings = Settings()
