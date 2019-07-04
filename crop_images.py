import os
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split
# TODO: use os.path to join directory

def crop(input_dir, image, height, width, output_dir):
    im = Image.open(input_dir + '/' + image)
    imgwidth, imgheight = im.size
  
    if height > imgheight or width > imgwidth:
        raise Exception('Crop image that larger than orginal size')
  
    replaced_suffix = ('.png', '.jpg', '.tif')
    for suffix in replaced_suffix: 
        image_name = image.replace(suffix, '')
  
    k = 0
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            top_left_x = j
            top_left_y = i
            if top_left_x + width > imgwidth:
                top_left_x = imgwidth - width
            if top_left_y + height > imgheight:
                top_left_y = imgheight - height
        
            box = (top_left_x, top_left_y, top_left_x+width, top_left_y+height)
            absolute_output_image = '{0}/{1}-{2}.png'.format(output_dir, image_name, k)
            try:
                im.crop(box).save(absolute_output_image)
            except:
                pass
            k +=1
    print("Cropped:" + image + "to {}".format(k) + "images")  

if __name__ == '__main__':
  CROP_SIZE = 256, 256

  cur_dir = os.getcwd()
  original_image_dir = '/var/scratch/tnguyenh/datasets/inria_aerial/AerialImageDataset/train/images'
  original_mask_dir = '/var/scratch/tnguyenh/datasets/inria_aerial/AerialImageDataset/train/gt'
  
  processed_train_image_dir = cur_dir + '/train_data_256/images'
  processed_train_mask_dir = cur_dir + '/train_data_256/masks'
  
  processed_validation_image_dir = cur_dir + '/validation_data_256/images'
  processed_validation_mask_dir = cur_dir + '/validation_data_256/masks'

  processed_test_image_dir = cur_dir + '/test_data_256/images'
  processed_test_mask_dir = cur_dir + '/test_data_256/masks'
  
  
  if not os.path.exists(processed_train_image_dir) or \
     not os.path.exists(processed_train_mask_dir) or \
     not os.path.exists(processed_test_image_dir) or \
     not os.path.exists(processed_test_mask_dir):
                os.makedirs(processed_train_image_dir)
                os.makedirs(processed_train_mask_dir)
                os.makedirs(processed_validation_image_dir)
                os.makedirs(processed_validation_mask_dir)
               	os.makedirs(processed_test_image_dir)
                os.makedirs(processed_test_mask_dir)
  else:
    shutil.rmtree(processed_train_image_dir)
    shutil.rmtree(processed_train_mask_dir)
    shutil.rmtree(processed_validation_image_dir)
    shutil.rmtree(processed_validation_mask_dir)
    shutil.rmtree(processed_test_image_dir)
    shutil.rmtree(processed_test_mask_dir)
    
  images = os.listdir(original_image_dir)
  masks = os.listdir(original_mask_dir)
  
  if images != masks:
    raise Exception("Images is not matched to masks")
  
  # Divide train and test
  train_image_names, validation_and_test_image_names = train_test_split(images, test_size=0.2, random_state=42)
  validation_image_names, test_image_names = train_test_split(validation_and_test_image_names, test_size=0.5, random_state=42)

  
  print("Train images len {}, Test images len {}".format(len(train_image_names), len(test_image_names)))
  for image in images:
    if image in train_image_names:
      output_dir= processed_train_image_dir
    elif image in validation_image_names:
      output_dir = processed_validation_image_dir
    else:
      output_dir = processed_test_image_dir

    crop(input_dir=original_image_dir, image=image,
      height = CROP_SIZE[0], width = CROP_SIZE[1],
      output_dir=output_dir)

  for mask in masks:
    if mask in train_image_names:
      output_dir= processed_train_mask_dir
    elif mask in validation_image_names:
      output_dir = processed_validation_mask_dir
    else:
      output_dir = processed_test_mask_dir
 
    crop(input_dir=original_mask_dir, image = mask,
       height = CROP_SIZE[0], width = CROP_SIZE[1],
       output_dir=output_dir)

      
