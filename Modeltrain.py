"""
STEP 3: Train U-Net Model (look more into lit but i think good starting CNN for Colony Segmentation
==================================================

Steps in my big ol brain
1. U-Net architecture for semantic segmentation
2. Load training and validation data
3. Train the model with callbacks
4. Save the best model

"""

import numpy as np
import cv2
import os
from pathlib import Path
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import CONFIG


# ============================================================================
# DATA LOADING AND AUG
# ============================================================================

def load_data_generator(data_dir, batch_size, augment=False):
    """
    PSEUDOCODE: Data generator for training

    INPUT:
        data_dir = directory containing images/ and masks/ folders
        batch_size = number of samples per batch
        augment = whether to apply data augmentation

    OUTPUT:
        Generator that yields (images_batch, masks_batch)
    """

    images_dir = Path(data_dir) / 'images'
    masks_dir = Path(data_dir) / 'masks'

    # Get all dem image files
    image_files = sorted(list(images_dir.glob('*.png')))

    while True:  # Infinite generator
        # but shuffle at start of each epoch
        np.random.shuffle(image_files)

        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i + batch_size]

            images_batch = []
            masks_batch = []

            for image_path in batch_files:
                # Load image
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (CONFIG['IMAGE_WIDTH'], CONFIG['IMAGE_HEIGHT']))

                # Load corresponding mask
                mask_path = masks_dir / image_path.name
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (CONFIG['IMAGE_WIDTH'], CONFIG['IMAGE_HEIGHT']))

                # Normalize
                image = image.astype(np.float32) / 255.0
                mask = mask.astype(np.float32) / 255.0

                # Data aug (if training)
                if augment and np.random.random() > 0.5:
                    # Random horizontal flip
                    if np.random.random() > 0.5:
                        image = np.fliplr(image)
                        mask = np.fliplr(mask)

                    # Random vertical flip
                    if np.random.random() > 0.5:
                        image = np.flipud(image)
                        mask = np.flipud(mask)

                    # Random rotation 
                    k = np.random.randint(0, 4)
                    image = np.rot90(image, k)
                    mask = np.rot90(mask, k)

                    # Random brightness 
                    brightness_factor = np.random.uniform(0.8, 1.2)
                    image = np.clip(image * brightness_factor, 0, 1)

                # Add channel dimension
                image = np.expand_dims(image, axis=-1)
                mask = np.expand_dims(mask, axis=-1)

                images_batch.append(image)
                masks_batch.append(mask)

            yield np.array(images_batch), np.array(masks_batch)


def count_samples(data_dir):
    """Count number of samples in a directory"""
    images_dir = Path(data_dir) / 'images'
    return len(list(images_dir.glob('*.png')))


# ============================================================================
# U-NET MODEL ARCHITECTURE
# ============================================================================

def build_unet_model(input_shape=(512, 512, 1)):
    """
    Build dat U-Net architecture here sum basics

    U-Net consists of:
    1. Encoder (contracting path) - downsamples and extracts features
    2. Bottleneck - lowest resolution, most features
    3. Decoder (expanding path) - upsamples and localizes
    4. Skip connections - preserves spatial information

    INPUT: input_shape = (height, width, channels)
    OUTPUT: Keras Model
    """

    inputs = layers.Input(input_shape)

    # ========== ENCODER (Contracting Path) ==========

    # Block 1
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    # Block 2
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Block 3
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    # Block 4
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    # ========== BOTTLENECK ==========

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = layers.Dropout(0.5)(conv5)

    # ========== DECODER (Expanding Path) ==========

    # Block 6 - Upsample and concatenate with conv4
    up6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)
    merge6 = layers.concatenate([drop4, up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    # Block 7 - Upsample and concatenate with conv3
    up7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
    merge7 = layers.concatenate([conv3, up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    # Block 8 - Upsample and concatenate with conv2
    up8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
    merge8 = layers.concatenate([conv2, up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # Block 9 - Upsample and concatenate with conv1
    up9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
    merge9 = layers.concatenate([conv1, up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    # ========== OUTPUT LAYER ==========

    # 1x1 convolution to get final prediction
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


# ============================================================================
# CUSTOM LOSS FUNC FO YO BUNC
# ============================================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    PSEUDOCODE: Dice coefficient (F1 score for segmentation)

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Higher is better (1.0 = perfect overlap)
    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def dice_loss(y_true, y_pred):
    """Dice loss = 1 - Dice coefficient"""
    return 1 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    """
    PSEUDOCODE: Combined Binary Cross-Entropy + Dice Loss

    This combination:
    - BCE: Good for pixel-wise classification
    - Dice: Good for handling class imbalance
    """
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)

    return bce + dice


def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    PSEUDOCODE: Intersection over Union (IoU) metric

    IoU = |A ∩ B| / |A ∪ B|
    """
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Threshold predictions
    y_pred_binary = tf.cast(y_pred_flat > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true_flat * y_pred_binary)
    union = tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_binary) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model():
    """
     Main training pipeline

    1. Build model
    2. Compile with optimizer and loss
    3. Setup callbacks
    4. Train with data generators
    5. Save best model
    """


    # Build model
    input_shape = (CONFIG['IMAGE_HEIGHT'], CONFIG['IMAGE_WIDTH'], CONFIG['IMAGE_CHANNELS'])
    model = build_unet_model(input_shape)

    # Display model architecture
    model.summary()

    # Compile model
    print("\nCompiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
        loss=combined_loss,
        metrics=[dice_coefficient, iou_metric, 'accuracy']
    )

    # Count samples
    train_samples = count_samples(CONFIG['TRAIN_DIR'])
    val_samples = count_samples(CONFIG['VALIDATION_DIR'])

    print(f"\nTraining samples: {train_samples}")
    print(f"Validation samples: {val_samples}")

    if train_samples == 0:
        print("\nERROR: No training data found!")
        print("Please run scripts/02_annotation_helper.py first")
        return

    # Calculate steps per epoch
    steps_per_epoch = max(1, train_samples // CONFIG['BATCH_SIZE'])
    validation_steps = max(1, val_samples // CONFIG['BATCH_SIZE'])

    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")

    # Setup data generators
    print("\nSetting up data generators...")
    train_generator = load_data_generator(
        CONFIG['TRAIN_DIR'],
        CONFIG['BATCH_SIZE'],
        augment=True
    )

    val_generator = load_data_generator(
        CONFIG['VALIDATION_DIR'],
        CONFIG['BATCH_SIZE'],
        augment=False
    )

    # Setup callbacks
    print("\nSetting up training callbacks...")

    # 1. Model checkpoint - save best model
    checkpoint_path = os.path.join(CONFIG['MODEL_DIR'], 'best_unet_model.h5')
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )

    # 2. Early stopping - stop if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=CONFIG['EARLY_STOPPING_PATIENCE'],
        mode='min',
        verbose=1,
        restore_best_weights=True
    )

    # 3. Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=CONFIG['REDUCE_LR_FACTOR'],
        patience=CONFIG['REDUCE_LR_PATIENCE'],
        mode='min',
        verbose=1,
        min_lr=1e-7
    )

    # 4. TensorBoard logging
    log_dir = os.path.join(CONFIG['MODEL_DIR'], 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks = [checkpoint, early_stop, reduce_lr, tensorboard]

    # Train model
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: {CONFIG['EPOCHS']}")
    print(f"Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"Learning rate: {CONFIG['LEARNING_RATE']}")
    print("=" * 60)
    print()

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=CONFIG['EPOCHS'],
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    final_model_path = os.path.join(CONFIG['MODEL_DIR'], 'final_unet_model.h5')
    model.save(final_model_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE BITCH!")
    print("=" * 60)
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Final model saved to: {final_model_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {log_dir}")


    return history, model


