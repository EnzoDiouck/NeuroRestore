import tensorflow as tf

def load_and_preprocess_image(degraded_path: str, clean_path: str, img_h: int, img_w: int):
    """Load and preprocess image pair"""
    dtype = tf.float16

    # Load degraded image
    degraded_img = tf.io.read_file(degraded_path)
    degraded_img = tf.image.decode_jpeg(degraded_img, channels=3)
    degraded_img = tf.image.resize(degraded_img, [img_h, img_w])
    degraded_img = tf.cast(degraded_img, dtype) / 127.5 - 1  # Normalize to [-1, 1]

    # Load clean image
    clean_img = tf.io.read_file(clean_path)
    clean_img = tf.image.decode_jpeg(clean_img, channels=3)
    clean_img = tf.image.resize(clean_img, [img_h, img_w])
    clean_img = tf.cast(clean_img, dtype) / 127.5 - 1  # Normalize to [-1, 1]

    return degraded_img, clean_img

def create_dataset(
    degraded_dir: str,
    clean_dir: str,
    img_h: int,
    img_w: int,
    batch_size: int,
    buffer_size: int = 1000,
    shuffle=True
):
    """Create TensorFlow dataset from image directories"""
    # Get sorted file lists
    degraded_files = sorted(tf.io.gfile.glob(f"{degraded_dir}/*.jpg"))
    clean_files = sorted(tf.io.gfile.glob(f"{clean_dir}/*.jpg"))

    assert len(degraded_files) == len(clean_files), "Mismatch in number of degraded and clean images"

    print(f"Found {len(degraded_files)} image pairs")

    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((degraded_files, clean_files))

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    # Load and preprocess images
    dataset = dataset.map(
        lambda x, y: load_and_preprocess_image(x, y, img_h, img_w),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset