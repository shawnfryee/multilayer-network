import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.preprocessing.image import save_img


def augment_images(image_dir, save_dir, augmentations, save_format='jpg'):


    datagen = ImageDataGenerator(**augmentations)


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
            # Load the image
            img_path = os.path.join(image_dir, filename)
            img = load_img(img_path)
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to (1, height, width, channels)

            # Generate one augmented version of the image

            for batch in datagen.flow(img_array, batch_size=1, save_to_dir=save_dir, save_prefix='aug',
                                      save_format=save_format):
                break  # Stop after generating one image

            # Rename the augmented image to match the original name
            original_name, _ = os.path.splitext(filename)
            for aug_filename in os.listdir(save_dir):
                if aug_filename.startswith('aug'):
                    new_filename = original_name + '.' + save_format
                    os.rename(os.path.join(save_dir, aug_filename), os.path.join(save_dir, new_filename))
                    print(f"Augmented and saved: {new_filename}")


# Configuration for data augmentation
augmentations = {
    'rotation_range': 35,
    'zoom_range': 0.2,
    'brightness_range': (0.8, 1.2),
    'shear_range': 0.2,
    'channel_shift_range': 50.0,
    'fill_mode': 'nearest'
}

# Example usage
image_dir = 'images'  # Update with the path to your images
save_dir = 'images'  # Update with where you want to save augmented images

augment_images(image_dir, save_dir, augmentations)